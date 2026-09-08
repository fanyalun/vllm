# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real Qwen DSpark branch KV on one GPU with synthetic Target features."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config import replace, set_current_vllm_config
from vllm.distributed import destroy_model_parallel
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.async_draft.adapters import get_async_draft_adapter
from vllm.v1.worker.gpu.spec_decode.async_draft.runtime import (
    DraftBlockPool,
    _dspark_private_query,
    _initialize_draft_kv_cache,
)
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager


@torch.inference_mode()
def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["ASYNC_DRAFT_DSPARK_FAN_OUT"] = "3"
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    config = EngineArgs(
        model="/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.15,
        enable_prefix_caching=False,
        async_scheduling=False,
        language_model_only=True,
        speculative_config={
            "method": "dspark",
            "model": "/data1/fanya/models/Qwen3.6-35B-A3B-speculator.dspark",
            "num_speculative_tokens": 3,
            "draft_sample_method": "greedy",
        },
    ).create_engine_config()
    config = replace(
        config,
        attention_config=replace(
            config.attention_config,
            use_non_causal=True,
            backend=config.speculative_config.attention_backend,
        ),
    )
    adapter = get_async_draft_adapter(config)
    device = torch.device("cuda:0")
    with set_current_vllm_config(config):
        init_worker_distributed_environment(
            config,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="nccl",
        )
        init_workspace_manager(device)
        runner = GPUModelRunner(config, device)
        runner.async_branch_backbone_width = 3
        runner.async_draft_fan_out = 3
        adapter.load_standalone_draft(runner)
        _initialize_draft_kv_cache(runner, 4099)
        spec = runner.speculator
        pool = DraftBlockPool(runner)
        anchor = 2 * max(pool.block_tables.block_sizes) - 1
        slots = pool.ensure(["r"], [0], np.array([anchor + 8]))
        positions = torch.arange(anchor, device=device)
        mappings = runner.block_tables.compute_slot_mappings(
            slots,
            torch.tensor([0, anchor], dtype=torch.int32, device=device),
            positions,
            anchor,
        )
        features = torch.randn(
            anchor,
            adapter.target_state_layout().width,
            dtype=torch.bfloat16,
            device=device,
        )
        hidden = spec.model.combine_hidden_states(features)
        spec.model.precompute_and_store_context_kv(hidden, positions, mappings[0])
        torch.cuda.synchronize()

        def canonical_bytes():
            result = []
            for group, caches in enumerate(pool._kv_caches_by_group()):
                indices = torch.tensor(pool.allocations["r"][group], device=device)
                for cache in caches:
                    dim = 1 if cache.shape[0] == 2 else 0
                    result.append(
                        cache.index_select(dim, indices)
                        .contiguous()
                        .view(torch.uint8)
                        .clone()
                    )
            return result

        before = canonical_bytes()
        free = [len(x) for x in pool.free_blocks]
        batch = SimpleNamespace(req_ids=["r"], generation=1)
        records = []
        for depth in range(4):
            prefix = torch.tensor(
                [[10, *[11, 12, 13][:depth], r] for r in [20, 21, 22]], device=device
            )
            logits, query, start = _dspark_private_query(
                runner,
                pool,
                batch,
                [0, 0, 0],
                prefix,
                torch.full((3,), anchor, device=device),
                3,
            )
            assert logits.shape[:2] == (3, 3)
            assert torch.isfinite(logits).all()
            assert not torch.equal(logits[0], logits[1])
            assert [len(x) for x in pool.free_blocks] == free
            assert list(pool.allocations) == ["r"]
            assert all(torch.equal(a, b) for a, b in zip(before, canonical_bytes()))
            records.append(
                {
                    "depth": depth,
                    "query": query.tolist(),
                    "sample_start": start,
                    "logits_shape": list(logits.shape),
                    "canonical_kv_unchanged": True,
                    "temporary_pages_reclaimed": True,
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "scope": (
                        "real Qwen Draft backbone with synthetic Target features; "
                        "not end-to-end parity"
                    ),
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
    destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run()

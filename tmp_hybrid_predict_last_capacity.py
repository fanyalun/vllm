import gc
import json
import sys

import torch

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "/home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"

COMMON = dict(
    model=MODEL,
    tokenizer=MODEL,
    tensor_parallel_size=2,
    trust_remote_code=True,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=2,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    mamba_cache_mode="align",
    enforce_eager=True,
    seed=123,
)


def cleanup() -> None:
    cleanup_dist_env_and_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collect_worker_capacity(worker) -> dict[str, object]:
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    kv_cache_config = worker.model_runner.kv_cache_config
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        worker.vllm_config, kv_cache_config
    )
    max_model_len = worker.vllm_config.model_config.max_model_len

    kv_cache_groups: list[dict[str, object]] = []
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        kv_cache_groups.append(
            {
                "group_idx": group_idx,
                "kind": get_kv_cache_spec_kind(spec).value,
                "block_size": int(spec.block_size),
                "page_size_bytes": int(spec.page_size_bytes),
                "sliding_window": getattr(spec, "sliding_window", None),
                "num_layers": len(group.layer_names),
            }
        )

    return {
        "rank": getattr(worker, "rank", None),
        "available_kv_cache_memory_bytes": int(
            worker.available_kv_cache_memory_bytes
        ),
        "num_blocks": int(kv_cache_config.num_blocks),
        "num_groups": len(kv_cache_config.kv_cache_groups),
        "max_model_len": int(max_model_len),
        "max_concurrency": float(max_concurrency),
        "gpu_kv_cache_tokens": int(max_concurrency * max_model_len),
        "kv_cache_groups": kv_cache_groups,
    }


def snapshot_capacity(mode: str, num_speculative_tokens: int) -> dict[str, object]:
    llm = LLM(
        **COMMON,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
            "hybrid_spec_state_offload_mode": mode,
            "hybrid_spec_state_ewma_alpha": 0.5,
        },
    )

    engine = llm.llm_engine
    engine_core = engine.engine_core
    result: dict[str, object] = {
        "mode": mode,
        "num_speculative_tokens": num_speculative_tokens,
        "engine_type": type(engine).__name__,
        "engine_core_type": type(engine_core).__name__,
    }

    worker_snapshots = engine.collective_rpc(_collect_worker_capacity)
    result["worker_snapshots"] = worker_snapshots

    if worker_snapshots:
        result["num_workers"] = len(worker_snapshots)
        result["num_blocks"] = worker_snapshots[0]["num_blocks"]
        result["num_groups"] = worker_snapshots[0]["num_groups"]
        result["max_model_len"] = worker_snapshots[0]["max_model_len"]
        result["max_concurrency"] = worker_snapshots[0]["max_concurrency"]
        result["gpu_kv_cache_tokens"] = worker_snapshots[0]["gpu_kv_cache_tokens"]
        result["kv_cache_groups"] = worker_snapshots[0]["kv_cache_groups"]
        result["min_available_kv_cache_memory_bytes"] = min(
            snapshot["available_kv_cache_memory_bytes"]
            for snapshot in worker_snapshots
        )
        result["max_available_kv_cache_memory_bytes"] = max(
            snapshot["available_kv_cache_memory_bytes"]
            for snapshot in worker_snapshots
        )

    del llm
    cleanup()
    return result


def main() -> None:
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in (
        "disabled",
        "predict_last",
    ):
        raise SystemExit(
            "Usage: tmp_hybrid_predict_last_capacity.py "
            "[disabled|predict_last] [num_speculative_tokens=2]"
        )
    num_speculative_tokens = int(sys.argv[2]) if len(sys.argv) == 3 else 2
    print(
        json.dumps(
            snapshot_capacity(sys.argv[1], num_speculative_tokens),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

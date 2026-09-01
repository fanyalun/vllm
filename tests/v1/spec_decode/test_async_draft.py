# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.async_draft import runtime
from vllm.v1.worker.gpu.spec_decode.async_draft.cache import (
    BranchCache,
    CachedBranch,
    select_branches_within_budget,
    select_recovery_candidates,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.speculator import (
    AsyncDraftSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.weights import (
    load_safetensors_key,
    materialize_standalone_eagle_weights,
)


def test_async_draft_device_none_keeps_local_speculator(monkeypatch) -> None:
    from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle_module

    class FakeLocalSpeculator:
        def __init__(self, vllm_config, device) -> None:
            self.vllm_config = vllm_config
            self.device = device

    monkeypatch.setattr(eagle_module, "EagleSpeculator", FakeLocalSpeculator)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            async_draft_device=None,
            method="eagle3",
            use_eagle=lambda: True,
            use_gemma4_mtp=lambda: False,
        )
    )

    speculator = init_speculator(config, torch.device("cpu"))

    assert isinstance(speculator, FakeLocalSpeculator)


def test_async_draft_device_selects_standalone_speculator(monkeypatch) -> None:
    from vllm.v1.worker.gpu.spec_decode import async_draft as async_module

    class FakeAsyncSpeculator:
        def __init__(self, vllm_config, device) -> None:
            self.vllm_config = vllm_config
            self.device = device

    monkeypatch.setattr(async_module, "AsyncDraftSpeculator", FakeAsyncSpeculator)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            async_draft_device=1,
            method="eagle3",
        )
    )

    speculator = init_speculator(config, torch.device("cpu"))

    assert isinstance(speculator, FakeAsyncSpeculator)


def test_select_recovery_candidates_excludes_returned_tokens() -> None:
    logits = torch.tensor(
        [
            [
                [9.0, 8.0, 7.0, 6.0, 5.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [1.0, 3.0, 5.0, 4.0, 2.0],
            ]
        ]
    )
    returned_tokens = torch.tensor([[0, 4]])

    candidates = select_recovery_candidates(logits, returned_tokens)

    assert candidates.tolist() == [[[1, 2, 3], [3, 2, 1], [2, 3, 1]]]


def test_branch_cache_discards_request_siblings_after_pop() -> None:
    cache = BranchCache()
    tokens = torch.tensor([1, 2, 3])
    hidden_states = torch.zeros(3, 4)
    selected = CachedBranch("branch-a", tokens, hidden_states)
    sibling = CachedBranch("branch-b", tokens, hidden_states)
    other_request = CachedBranch("branch-c", tokens, hidden_states)
    cache.add(("engine", "request", 2, 1, 10), selected)
    cache.add(("engine", "request", 2, 1, 11), sibling)
    cache.add(("engine", "other", 0, 0, 12), other_request)

    assert cache.pop(("engine", "request", 2, 1, 10)) is selected
    discarded = cache.discard_request("request")

    assert discarded == ["branch-b"]
    assert list(cache.entries) == [("engine", "other", 0, 0, 12)]


def test_branch_cache_discards_all_entries() -> None:
    cache = BranchCache()
    tokens = torch.tensor([1, 2, 3])
    hidden_states = torch.zeros(3, 4)
    cache.add(
        ("engine", "request-a", 0, 0, 10),
        CachedBranch("branch-a", tokens, hidden_states),
    )
    cache.add(
        ("engine", "request-b", 0, 0, 11),
        CachedBranch("branch-b", tokens, hidden_states),
    )

    assert cache.discard_all() == ["branch-a", "branch-b"]
    assert not cache.entries


def test_branch_budget_prioritizes_top_candidates_across_requests() -> None:
    selected = select_branches_within_budget(
        required_seq_lens=[16] * 8,
        request_indices=[0, 0, 0, 0, 1, 1, 1, 1],
        accepted_counts=[0, 0, 1, 1, 0, 0, 1, 1],
        candidate_indices=[0, 1, 0, 1, 0, 1, 0, 1],
        block_size=16,
        available_slots=4,
        available_blocks=4,
    )

    assert selected == [0, 2, 4, 6]


def test_branch_budget_skips_branches_that_do_not_fit() -> None:
    selected = select_branches_within_budget(
        required_seq_lens=[33, 16, 16],
        request_indices=[0, 0, 1],
        accepted_counts=[0, 1, 0],
        candidate_indices=[0, 0, 0],
        block_size=16,
        available_slots=2,
        available_blocks=2,
    )

    assert selected == [1, 2]


def test_branch_budget_charges_only_copy_on_write_blocks() -> None:
    selected = select_branches_within_budget(
        required_seq_lens=[160, 160],
        request_indices=[0, 1],
        accepted_counts=[0, 0],
        candidate_indices=[0, 0],
        block_size=16,
        available_slots=2,
        available_blocks=4,
        shared_prefix_blocks=[8, 8],
    )

    assert selected == [0, 1]


def test_draft_block_pool_clone_copies_only_mutated_tail() -> None:
    class FakeBlockTables:
        block_sizes = [4]

        def __init__(self) -> None:
            self.num_blocks = SimpleNamespace(
                np=torch.zeros(1, 4, dtype=torch.int32).numpy(),
                copy_to_uva=lambda: None,
            )

        def append_block_ids(self, req_slot, block_ids, *, overwrite) -> None:
            del req_slot, block_ids, overwrite

        def apply_staged_writes(self) -> None:
            pass

    runner = SimpleNamespace(
        block_tables=FakeBlockTables(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()], num_blocks=8),
        max_num_reqs=4,
        num_speculative_steps=3,
        device=torch.device("cpu"),
        kv_caches=[torch.arange(32).view(8, 4, 1).clone()],
    )
    pool = runtime.DraftBlockPool(runner)
    pool.ensure(["request"], [0], torch.tensor([8]).numpy())
    source_blocks = pool.allocations["request"][0].copy()

    pool.clone(
        "request",
        ["branch"],
        torch.tensor([12]).numpy(),
        torch.tensor([5]).numpy(),
    )

    branch_blocks = pool.allocations["branch"][0]
    assert branch_blocks[0] == source_blocks[0]
    assert branch_blocks[1] != source_blocks[1]
    assert torch.equal(
        runner.kv_caches[0][branch_blocks[1]],
        runner.kv_caches[0][source_blocks[1]],
    )
    assert pool.block_refcounts[0][source_blocks[0]] == 2
    assert pool.block_refcounts[0][source_blocks[1]] == 1

    pool.release(["branch", "request"])

    assert len(pool.free_blocks[0]) == 7
    assert not any(pool.block_refcounts[0])


def test_draft_block_pool_rejects_mismatched_batch_metadata() -> None:
    pool = runtime.DraftBlockPool.__new__(runtime.DraftBlockPool)

    with pytest.raises(ValueError, match="IDs, epochs, and sequence lengths"):
        pool.ensure(["request"], [], torch.tensor([1]).numpy())


def test_branch_cudagraph_sizes_cover_b1_b4_b16() -> None:
    assert runtime._branch_cudagraph_capture_sizes(7, 16) == {
        24,
        48,
        96,
        192,
        384,
    }


def test_cache_hit_does_not_promote_approximate_branch(
    monkeypatch,
) -> None:
    class FakeDraftSpeculator:
        hidden_size = 4

    class FakeBlockPool:
        def __init__(self) -> None:
            self.released: list[str] = []

        def promote(self, *args) -> None:
            raise AssertionError("Approximate branch must not replace canonical KV")

        def release(self, request_ids: list[str]) -> None:
            self.released.extend(request_ids)

    monkeypatch.setattr(runtime, "DraftModelSpeculator", FakeDraftSpeculator)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.delenv("ASYNC_DRAFT_FORCE_JIT", raising=False)
    monkeypatch.delenv("ASYNC_DRAFT_VALIDATE_HITS", raising=False)

    cache = BranchCache()
    cached_tokens = torch.tensor([3, 4, 5])
    cached_hidden_states = torch.ones(3, 4)
    cache.add(
        ("engine", "request", 0, 0, 10),
        CachedBranch(
            "approximate-branch",
            cached_tokens,
            cached_hidden_states,
        ),
    )
    ring_slot = SimpleNamespace(
        draft_tokens=torch.zeros(1, 3, dtype=torch.int64),
        num_sampled=torch.tensor([1]),
        last_sampled=torch.tensor([10]),
    )
    runner = SimpleNamespace(
        speculator=FakeDraftSpeculator(),
        num_speculative_steps=3,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32)),
        device=torch.device("cpu"),
    )
    block_pool = FakeBlockPool()
    batch = SimpleNamespace(
        slot=0,
        engine_instance_id="engine",
        num_reqs=1,
        transient=False,
        req_ids=["request"],
        request_epochs=[0],
    )

    (
        metrics,
        feedback_hidden_states,
        hit_indices,
        trace_top2,
    ) = runtime._run_proposal(runner, block_pool, cache, [ring_slot], batch, (4,))

    assert hit_indices == [0]
    assert trace_top2 is None
    assert metrics["cache_hits"] == 1
    assert metrics["cache_misses"] == 0
    assert block_pool.released == ["approximate-branch"]
    assert torch.equal(ring_slot.draft_tokens[0], cached_tokens)
    assert torch.equal(feedback_hidden_states[0], cached_hidden_states)


def test_materialize_missing_embedding_reads_only_indexed_shard(
    tmp_path,
    monkeypatch,
) -> None:
    from vllm.v1.worker.gpu.spec_decode.async_draft import weights

    embedding = torch.arange(12, dtype=torch.float32).view(4, 3)
    embedding_shard = tmp_path / "model-00001-of-00002.safetensors"
    unused_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"model.embed_tokens.weight": embedding}, embedding_shard)
    save_file({"model.layers.0.weight": torch.ones(2, 2)}, unused_shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": embedding_shard.name,
                    "model.layers.0.weight": unused_shard.name,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeEagleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(4, 3)
            self.lm_head = nn.Linear(3, 4, bias=False)
            self.has_own_embed_tokens = False
            self.has_own_lm_head = True

    opened: list[str] = []
    original_safe_open = weights.safe_open

    def tracked_safe_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(weights, "safe_open", tracked_safe_open)
    model = FakeEagleModel()
    materialized = materialize_standalone_eagle_weights(model, str(tmp_path))

    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert opened == [str(embedding_shard)]
    assert [entry.source for entry in materialized] == ["target", "draft"]
    assert materialized[0].checkpoint_file == embedding_shard.name


def test_monolithic_bin_shared_weight_fails_closed(tmp_path) -> None:
    (tmp_path / "pytorch_model.bin").touch()

    with pytest.raises(ValueError, match="monolithic .bin"):
        load_safetensors_key(str(tmp_path), "model.embed_tokens.weight")


def test_control_response_metrics_are_recorded() -> None:
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._metrics = {
        "cache_hits": 0,
        "cache_misses": 0,
        "jit_fallbacks": 0,
        "cache_evictions": 0,
        "ipc_bytes": 0,
        "wait_seconds": 0.0,
        "branch_build_seconds": 0.0,
    }
    speculator._step_metrics = speculator._metrics.copy()

    speculator._record_metrics({"branch_build_seconds": 0.25, "cache_evictions": 2})

    assert speculator._metrics["branch_build_seconds"] == 0.25
    assert speculator._metrics["cache_evictions"] == 2
    assert speculator._step_metrics["branch_build_seconds"] == 0.25
    assert speculator._step_metrics["cache_evictions"] == 2


class _FakeAsyncDraftProcess:
    pid = 1234

    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode


class _FakeAsyncDraftConnection:
    def __init__(self, response=None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error

    def poll(self, timeout: float) -> bool:
        del timeout
        return self.response is not None or self.error is not None

    def recv(self):
        if self.error is not None:
            raise self.error
        return self.response


def test_recv_fails_fast_when_child_exits() -> None:
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._connection = _FakeAsyncDraftConnection()
    speculator._process = _FakeAsyncDraftProcess(exitcode=9)

    with pytest.raises(RuntimeError, match="exit_code=9"):
        speculator._recv(10.0, "proposal")


def test_recv_reports_child_ipc_eof() -> None:
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._connection = _FakeAsyncDraftConnection(error=EOFError())
    speculator._process = _FakeAsyncDraftProcess(exitcode=1)

    with pytest.raises(RuntimeError, match="closed its IPC channel"):
        speculator._recv(10.0, "proposal")


def test_recv_rejects_invalid_response() -> None:
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._connection = _FakeAsyncDraftConnection(response="invalid")
    speculator._process = _FakeAsyncDraftProcess(exitcode=None)

    with pytest.raises(RuntimeError, match="Invalid async draft response"):
        speculator._recv(10.0, "proposal")


def test_recv_times_out_while_child_is_alive() -> None:
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._connection = _FakeAsyncDraftConnection()
    speculator._process = _FakeAsyncDraftProcess(exitcode=None)

    with pytest.raises(TimeoutError, match="Timed out after 0.0s"):
        speculator._recv(0.0, "proposal")


@pytest.mark.parametrize(
    "response",
    [
        {"generation": 6, "slot": 1},
        {"generation": 7, "slot": 0},
        {"generation": 7},
    ],
)
def test_response_identity_rejects_stale_or_wrong_slot(response) -> None:
    with pytest.raises(RuntimeError, match="generation mismatch"):
        AsyncDraftSpeculator._validate_response_identity(response, 7, 1)


def test_response_identity_accepts_current_generation_and_slot() -> None:
    AsyncDraftSpeculator._validate_response_identity({"generation": 7, "slot": 1}, 7, 1)


def test_request_epoch_lifecycle_handles_preemption_and_id_reuse() -> None:
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._request_epochs = {}
    speculator._active_requests = set()
    speculator._preempted_requests = set()
    controls: list[tuple[str, list[str]]] = []

    def record_control(command, request_ids) -> None:
        controls.append((command, list(request_ids)))

    speculator._control = record_control

    speculator.on_requests_added(["request"])
    assert speculator._request_epochs == {"request": 0}
    assert speculator._active_requests == {"request"}

    speculator.on_requests_preempted(["request"])
    assert speculator._request_epochs == {"request": 1}
    assert speculator._preempted_requests == {"request"}

    speculator.on_requests_added(["request"])
    assert speculator._request_epochs == {"request": 1}
    assert speculator._active_requests == {"request"}
    assert not speculator._preempted_requests

    speculator.on_requests_finished(["request"])
    speculator.on_requests_added(["request"])
    assert speculator._request_epochs == {"request": 2}
    assert controls == [
        ("reset", []),
        ("reset", ["request"]),
        ("reset", []),
        ("release", ["request"]),
        ("reset", ["request"]),
    ]


def test_internal_proposal_trace_records_outcome_and_cache_hit(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("REPLAYSSM_SPEC_DECODE_TRACE_PATH", str(trace_path))
    speculator = AsyncDraftSpeculator.__new__(AsyncDraftSpeculator)
    speculator._request_epochs = {"request": 3}
    speculator._active_trace_req_ids = ["request"]
    speculator._last_cache_hit_indices = {0}

    speculator.record_proposal_trace(
        SimpleNamespace(num_reqs=1, req_ids=["request"]),
        torch.tensor([[11, 12, 13]]),
        torch.tensor([[21, 22, 23]]),
        torch.tensor([3]),
        torch.tensor([1]),
        torch.tensor([99]),
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record == {
        "trace_step": 0,
        "request_id": "request",
        "accepted_draft_count": 2,
        "num_rejected": 1,
        "recovery_token": 99,
        "accepted_draft_tokens": [11, 12],
        "draft_tokens": [21, 22, 23],
        "request_epoch": 3,
        "cache_hit": True,
    }


def _make_async_draft_validation_config() -> SimpleNamespace:
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        nnodes=1,
        distributed_executor_backend="uni",
        enable_dbo=False,
        assigned_physical_gpu_ids=[0],
    )
    model_config = SimpleNamespace(
        architecture="LlamaForCausalLM",
        is_multimodal_model=False,
        enable_prompt_embeds=False,
    )
    draft_model_config = SimpleNamespace(architecture="LlamaForCausalLMEagle3")
    speculative_config = SimpleNamespace(
        async_draft_device=1,
        draft_tensor_parallel_size=1,
        method="eagle3",
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        draft_model_config=draft_model_config,
    )
    return SimpleNamespace(
        speculative_config=speculative_config,
        use_v2_model_runner=True,
        parallel_config=parallel_config,
        model_config=model_config,
        lora_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )


class _FakeCudaPlatform:
    device_name = "cuda"

    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def device_id_to_physical_device_id(device_id: int) -> int:
        return device_id


def test_async_draft_supported_config_validates() -> None:
    config = _make_async_draft_validation_config()

    VllmConfig._validate_async_draft_config(config, _FakeCudaPlatform())


def test_async_draft_validation_reports_all_incompatible_fields() -> None:
    config = _make_async_draft_validation_config()
    config.parallel_config.tensor_parallel_size = 2
    config.cache_config.enable_prefix_caching = True
    config.speculative_config.async_draft_device = 0

    try:
        VllmConfig._validate_async_draft_config(config, _FakeCudaPlatform())
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected invalid asynchronous draft config to fail")

    assert "tensor_parallel_size=2" in message
    assert "enable_prefix_caching=True" in message
    assert "overlaps target device 0" in message


@pytest.mark.parametrize(
    ("object_path", "field", "value", "expected"),
    [
        ("config", "use_v2_model_runner", False, "model_runner=v1"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel_size=2"),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_size=2"),
        (
            "parallel_config",
            "decode_context_parallel_size",
            2,
            "decode_context_parallel_size=2",
        ),
        ("parallel_config", "nnodes", 2, "nnodes=2"),
        (
            "parallel_config",
            "distributed_executor_backend",
            "ray",
            "distributed_executor_backend=ray",
        ),
        (
            "parallel_config",
            "distributed_executor_backend",
            "external_launcher",
            "distributed_executor_backend=external_launcher",
        ),
        ("parallel_config", "enable_dbo", True, "enable_dbo=True"),
        (
            "speculative_config",
            "draft_tensor_parallel_size",
            2,
            "draft_tensor_parallel_size=2",
        ),
        ("speculative_config", "method", "mtp", "method='mtp'"),
        (
            "speculative_config",
            "draft_sample_method",
            "random",
            "draft_sample_method='random'",
        ),
        (
            "speculative_config",
            "rejection_sample_method",
            "synthetic",
            "rejection_sample_method='synthetic'",
        ),
        (
            "model_config",
            "architecture",
            "Qwen3ForCausalLM",
            "target_architecture='Qwen3ForCausalLM'",
        ),
        (
            "model_config",
            "is_multimodal_model",
            True,
            "multimodal_model=True",
        ),
        (
            "model_config",
            "enable_prompt_embeds",
            True,
            "enable_prompt_embeds=True",
        ),
        ("config", "lora_config", object(), "lora_config"),
        (
            "cache_config",
            "enable_prefix_caching",
            True,
            "enable_prefix_caching=True",
        ),
        (
            "draft_model_config",
            "architecture",
            "Qwen3MTP",
            "draft_architecture='Qwen3MTP'",
        ),
        (
            "speculative_config",
            "async_draft_device",
            -1,
            "async_draft_device=-1",
        ),
    ],
)
def test_async_draft_validation_fails_closed_for_unsupported_combinations(
    object_path, field, value, expected
) -> None:
    config = _make_async_draft_validation_config()
    objects = {
        "config": config,
        "parallel_config": config.parallel_config,
        "speculative_config": config.speculative_config,
        "model_config": config.model_config,
        "cache_config": config.cache_config,
        "draft_model_config": config.speculative_config.draft_model_config,
    }
    setattr(objects[object_path], field, value)

    with pytest.raises(ValueError, match=expected):
        VllmConfig._validate_async_draft_config(config, _FakeCudaPlatform())


def test_async_draft_validation_rejects_non_cuda_platform() -> None:
    class FakeCpuPlatform(_FakeCudaPlatform):
        device_name = "cpu"

        @staticmethod
        def is_cuda() -> bool:
            return False

    config = _make_async_draft_validation_config()

    with pytest.raises(ValueError, match="device=cpu"):
        VllmConfig._validate_async_draft_config(config, FakeCpuPlatform())

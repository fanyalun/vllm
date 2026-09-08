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
from vllm.v1.worker.gpu.spec_decode.async_draft.adapters import (
    DSparkAsyncDraftAdapter,
    Eagle3AsyncDraftAdapter,
    QwenMTPAsyncDraftAdapter,
    get_async_draft_adapter,
)
from vllm.v1.worker.gpu.spec_decode.async_draft.cache import (
    BranchCache,
    CachedBranch,
    select_branches_with_group_budget,
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


@pytest.mark.parametrize(
    ("method", "adapter_type", "layer_ids", "expected_layout"),
    [
        ("eagle3", Eagle3AsyncDraftAdapter, [1, 4, 7], (8, 8, 8)),
        ("mtp", QwenMTPAsyncDraftAdapter, None, (8,)),
        ("dspark", DSparkAsyncDraftAdapter, [2, 10, 20, 30, 37], (8,) * 5),
    ],
)
def test_async_draft_adapter_selects_conditioning_layout(
    method, adapter_type, layer_ids, expected_layout
) -> None:
    hf_config = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=layer_ids,
        aux_hidden_state_layer_ids=layer_ids,
    )
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(hf_config=hf_config),
        num_speculative_tokens=4,
        use_gemma4_mtp=lambda: False,
    )
    config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(get_hidden_size=lambda: 8),
    )

    adapter = get_async_draft_adapter(config)

    assert isinstance(adapter, adapter_type)
    assert adapter.target_state_layout().splits == expected_layout


def test_gemma4_mtp_adapter_uses_target_kv_snapshots() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp", use_gemma4_mtp=lambda: True)
    )

    adapter = get_async_draft_adapter(config)
    assert type(adapter).__name__ == "Gemma4MTPAsyncDraftAdapter"
    assert not adapter.uses_kv_branches
    assert not adapter.provisional_state_is_canonical


def test_dspark_adapter_accepts_normalized_speculators_aux_layout() -> None:
    hf_config = SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 10, 20, 30, 37])
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=4,
            use_gemma4_mtp=lambda: False,
        ),
        model_config=SimpleNamespace(get_hidden_size=lambda: 4096),
    )

    layout = get_async_draft_adapter(config).target_state_layout()

    assert layout.name == "dspark_aux_hidden_states"
    assert layout.splits == (4096,) * 5


@pytest.mark.parametrize(
    ("sample_from_anchor", "block_size", "expected"),
    [(True, 8, 8), (False, 7, 6)],
)
def test_dspark_adapter_derives_native_proposal_bank_width(
    sample_from_anchor, block_size, expected
) -> None:
    hf_config = SimpleNamespace(
        sample_from_anchor=sample_from_anchor,
        block_size=block_size,
        speculators_config={},
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=4,
        )
    )

    assert DSparkAsyncDraftAdapter(config).proposal_bank_width() == expected


def test_dspark_adapter_prefers_checkpoint_proposal_width() -> None:
    hf_config = SimpleNamespace(
        block_size=99,
        speculators_config={"proposal_methods": [{"speculative_tokens": 8}]},
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=4,
        )
    )

    assert DSparkAsyncDraftAdapter(config).proposal_bank_width() == 8


def test_dspark_adapter_rejects_bank_narrower_than_target_width() -> None:
    hf_config = SimpleNamespace(
        speculators_config={"proposal_methods": [{"speculative_tokens": 3}]}
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=4,
        )
    )

    with pytest.raises(ValueError, match="must cover"):
        DSparkAsyncDraftAdapter(config).proposal_bank_width()


@pytest.mark.parametrize(
    ("verify_width", "checkpoint_width", "expected"),
    [(3, 8, 3), (2, 6, 2)],
)
def test_dspark_adapter_derives_round_branch_backbone_width(
    verify_width, checkpoint_width, expected
) -> None:
    hf_config = SimpleNamespace(
        speculators_config={
            "proposal_methods": [{"speculative_tokens": checkpoint_width}]
        }
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=verify_width,
        )
    )

    assert DSparkAsyncDraftAdapter(config).branch_backbone_width() == expected


def test_dspark_adapter_prefix_length_is_not_output_width() -> None:
    hf_config = SimpleNamespace(
        speculators_config={"proposal_methods": [{"speculative_tokens": 8}]}
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(hf_config=hf_config),
            num_speculative_tokens=4,
        )
    )

    assert DSparkAsyncDraftAdapter(config).branch_backbone_width() == 4


def test_dspark_adapter_accepts_internal_fan_out_override(monkeypatch) -> None:
    monkeypatch.setenv("ASYNC_DRAFT_DSPARK_FAN_OUT", "12")
    adapter = DSparkAsyncDraftAdapter(SimpleNamespace())

    assert adapter.fan_out() == 12


def test_mtp_adapter_accepts_internal_fan_out_override(monkeypatch) -> None:
    monkeypatch.setenv("ASYNC_DRAFT_MTP_FAN_OUT", "12")
    adapter = QwenMTPAsyncDraftAdapter(SimpleNamespace())

    assert adapter.fan_out() == 12


def test_mtp_adapter_selects_tuned_default_fan_out(monkeypatch) -> None:
    monkeypatch.delenv("ASYNC_DRAFT_MTP_FAN_OUT", raising=False)

    assert QwenMTPAsyncDraftAdapter(SimpleNamespace()).fan_out() == 96


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("Qwen3_5MoeForConditionalGeneration", 24),
        ("Gemma4ForConditionalGeneration", 48),
        ("OtherForCausalLM", 3),
    ],
)
def test_dspark_adapter_selects_tuned_default_fan_out(
    monkeypatch, architecture, expected
) -> None:
    monkeypatch.delenv("ASYNC_DRAFT_DSPARK_FAN_OUT", raising=False)
    config = SimpleNamespace(model_config=SimpleNamespace(architecture=architecture))

    assert DSparkAsyncDraftAdapter(config).fan_out() == expected


@pytest.mark.parametrize("value", ("0", "513", "invalid"))
def test_dspark_adapter_rejects_invalid_internal_fan_out(monkeypatch, value) -> None:
    monkeypatch.setenv("ASYNC_DRAFT_DSPARK_FAN_OUT", value)
    adapter = DSparkAsyncDraftAdapter(SimpleNamespace())

    with pytest.raises(ValueError, match="ASYNC_DRAFT_DSPARK_FAN_OUT"):
        adapter.fan_out()


@pytest.mark.parametrize("value", ("0", "513", "invalid"))
def test_mtp_adapter_rejects_invalid_internal_fan_out(monkeypatch, value) -> None:
    monkeypatch.setenv("ASYNC_DRAFT_MTP_FAN_OUT", value)
    adapter = QwenMTPAsyncDraftAdapter(SimpleNamespace())

    with pytest.raises(ValueError, match="ASYNC_DRAFT_MTP_FAN_OUT"):
        adapter.fan_out()


@pytest.mark.parametrize(
    ("sample_from_anchor", "width", "expected_query_width", "expected_anchors"),
    [
        (True, 3, 3, [0, 3, 6]),
        (False, 2, 3, [0, 3, 6]),
    ],
)
def test_dspark_execution_width_updates_query_layout(
    sample_from_anchor, width, expected_query_width, expected_anchors
) -> None:
    speculator = runtime.DSparkSpeculator.__new__(runtime.DSparkSpeculator)
    speculator.sample_from_anchor = sample_from_anchor
    speculator._max_execution_width = 7
    speculator.max_num_reqs = 4
    speculator.num_speculative_steps = 7
    speculator.num_query_per_req = 7 if sample_from_anchor else 8
    speculator._request_indices = torch.arange(4, dtype=torch.int64)
    speculator._anchor_indices_by_query_width = {}

    speculator.set_execution_width(width)

    assert speculator.num_speculative_steps == width
    assert speculator.num_query_per_req == expected_query_width
    assert speculator.anchor_indices(3).tolist() == expected_anchors
    with pytest.raises(ValueError, match="allocated maximum"):
        speculator.set_execution_width(8)


def test_dspark_reserves_private_branch_width_without_changing_d() -> None:
    speculator = runtime.DSparkSpeculator.__new__(runtime.DSparkSpeculator)
    speculator._max_execution_width = 3
    speculator.proposal_bank_width = 8
    speculator.num_speculative_steps = 3
    speculator.num_query_per_req = 3
    speculator.max_num_reqs = 2
    speculator.device = torch.device("cpu")
    speculator.draft_logits = None
    speculator._trace_top2_values = torch.empty(2, 3, 2)
    speculator._trace_top2_ids = torch.empty(2, 3, 2, dtype=torch.int64)

    speculator.reserve_execution_width(7)

    assert speculator._max_execution_width == 7
    assert speculator.num_speculative_steps == 3
    assert speculator.num_query_per_req == 3
    assert speculator.draft_tokens.shape == (2, 7)
    assert speculator.sample_indices.shape == (14,)
    assert speculator.sample_pos.shape == (14,)
    assert speculator.sample_idx_mapping.shape == (14,)
    assert speculator._trace_top2_values.shape == (2, 7, 2)
    assert speculator._trace_top2_ids.shape == (2, 7, 2)


def test_dspark_base_logits_only_skips_markov_sampling() -> None:
    class FakeModel:
        @staticmethod
        def compute_draft_logits(hidden_states):
            return hidden_states

        @staticmethod
        def markov_embed(tokens):
            del tokens
            raise AssertionError("base-logit-only execution must skip Markov")

    speculator = runtime.DSparkSpeculator.__new__(runtime.DSparkSpeculator)
    speculator.num_speculative_steps = 2
    speculator.sample_indices = torch.tensor([0, 1], dtype=torch.int64)
    speculator.model = FakeModel()
    speculator._async_base_logits = torch.zeros(1, 5, 4)
    speculator._async_base_logits_only = True
    speculator._trace_base_top2_values = None
    speculator.draft_tokens = torch.full((1, 5), -1, dtype=torch.int64)

    runtime.DSparkSpeculator._sample_sequential(
        speculator,
        1,
        torch.arange(8, dtype=torch.float32).view(2, 4),
    )

    assert speculator._async_base_logits[0, :2].tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]
    assert speculator.draft_tokens.tolist() == [[-1, -1, -1, -1, -1]]


def test_dspark_recovery_candidates_exclude_returned_target_token() -> None:
    class FakeModel:
        @staticmethod
        def map_draft_to_target(token_ids: torch.Tensor) -> torch.Tensor:
            return token_ids + 100

    speculator = SimpleNamespace(model=FakeModel())
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])

    candidates = runtime._dspark_top_recovery_candidates(
        speculator,
        logits,
        returned_token=torch.tensor([100]),
        fan_out=3,
    )

    assert candidates.shape == (1, 3)
    assert candidates.tolist() == [[101, 102, 103]]


def test_dspark_cache_miss_executes_normal_d_backbone(monkeypatch) -> None:
    class FakeDSparkSpeculator:
        hidden_size = 4
        sample_from_anchor = True

        def __init__(self) -> None:
            self._async_base_logits = torch.zeros(1, 7, 8)
            self.input_buffers = SimpleNamespace(
                input_ids=torch.tensor([9, 0, 0, 0, 0, 0, 0])
            )

        @staticmethod
        def anchor_indices(num_reqs, *, execution_width=None):
            assert num_reqs == 1
            assert execution_width == 3
            return torch.tensor([0])

    speculator = FakeDSparkSpeculator()
    monkeypatch.setattr(runtime, "DraftModelSpeculator", FakeDSparkSpeculator)
    monkeypatch.setattr(runtime, "DSparkSpeculator", FakeDSparkSpeculator)
    observed = {}

    def fake_execute(
        runner,
        block_pool,
        branch_cache,
        batch,
        ring_slot,
        conditioning_splits,
        num_speculative_steps=None,
        dspark_base_logits_only=False,
    ):
        del runner, block_pool, branch_cache, ring_slot, conditioning_splits
        observed["num_reqs"] = batch.num_reqs
        observed["width"] = num_speculative_steps
        observed["base_logits_only"] = dspark_base_logits_only
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), None, 0

    monkeypatch.setattr(runtime, "_execute_jit_proposal", fake_execute)
    ring_slot = SimpleNamespace(
        draft_tokens=torch.zeros(1, 3, dtype=torch.int64),
        num_sampled=torch.tensor([1]),
        last_sampled=torch.tensor([9]),
    )
    runner = SimpleNamespace(
        speculator=speculator,
        num_speculative_steps=3,
        async_target_verify_width=3,
        async_branch_backbone_width=3,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32)),
        device=torch.device("cpu"),
    )
    batch = SimpleNamespace(
        slot=0,
        engine_instance_id="engine",
        num_reqs=1,
        transient=True,
        req_ids=["request"],
        request_epochs=[0],
    )

    metrics, _, _, _, round_state = runtime._run_proposal(
        runner,
        SimpleNamespace(release=lambda request_ids: None),
        BranchCache(),
        [ring_slot],
        batch,
        (4,),
    )

    assert observed == {
        "num_reqs": 1,
        "width": 3,
        "base_logits_only": False,
    }
    assert ring_slot.draft_tokens.tolist() == [[1, 2, 3]]
    assert round_state is not None
    assert torch.equal(round_state.base_logits[0], torch.zeros(3, 8))
    assert round_state.anchor_tokens[0].item() == 9
    assert metrics["dspark_current_backbone_runs"] == 1
    assert metrics["dspark_current_backbone_seconds"] >= 0
    assert metrics["dspark_backbone_refreshes"] == 0


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


def test_select_recovery_candidates_accepts_dynamic_fan_out() -> None:
    logits = torch.arange(12, dtype=torch.float32).view(1, 2, 6)
    returned_tokens = torch.tensor([[5]])

    candidates = select_recovery_candidates(logits, returned_tokens, fan_out=4)

    assert candidates.shape == (1, 2, 4)
    assert 5 not in candidates[0, 0].tolist()


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


@pytest.mark.parametrize("mutation_start", [4, 5, 7])
def test_draft_block_pool_clone_copies_only_mutated_tail(mutation_start) -> None:
    class FakeBlockTables:
        block_sizes = [4]
        blocks_per_kv_block = [1]

        def __init__(self) -> None:
            self.block_tables = [
                SimpleNamespace(gpu=torch.zeros(4, 8, dtype=torch.int32))
            ]
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
        torch.tensor([mutation_start]).numpy(),
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

    canonical = runner.kv_caches[0][source_blocks].clone()
    for position in range(mutation_start, 12):
        block = branch_blocks[position // 4]
        runner.kv_caches[0][block, position % 4] = -1
    assert torch.equal(runner.kv_caches[0][source_blocks], canonical)

    pool.release(["branch", "request"])

    assert len(pool.free_blocks[0]) == 7
    assert not any(pool.block_refcounts[0])


def test_draft_block_pool_clone_handles_each_kv_group() -> None:
    class FakeBlockTables:
        block_sizes = [4, 2]
        blocks_per_kv_block = [1, 1]

        def __init__(self) -> None:
            self.block_tables = [
                SimpleNamespace(gpu=torch.zeros(4, 8, dtype=torch.int32)),
                SimpleNamespace(gpu=torch.zeros(4, 8, dtype=torch.int32)),
            ]
            self.num_blocks = SimpleNamespace(
                np=torch.zeros(2, 4, dtype=torch.int32).numpy(),
                copy_to_uva=lambda: None,
            )

        def append_block_ids(self, req_slot, block_ids, *, overwrite) -> None:
            del req_slot, block_ids, overwrite

        def apply_staged_writes(self) -> None:
            pass

    caches = [
        torch.arange(32).view(8, 4, 1).clone(),
        torch.arange(16).view(8, 2, 1).clone(),
    ]
    runner = SimpleNamespace(
        block_tables=FakeBlockTables(),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[object(), object()], num_blocks=8
        ),
        max_num_reqs=4,
        num_speculative_steps=3,
        device=torch.device("cpu"),
        kv_caches=caches,
        kv_caches_by_group=[[caches[0]], [caches[1]]],
    )
    pool = runtime.DraftBlockPool(runner)
    pool.ensure(["request"], [0], torch.tensor([8]).numpy())
    source = [blocks.copy() for blocks in pool.allocations["request"]]

    pool.clone(
        "request",
        ["branch"],
        torch.tensor([10]).numpy(),
        torch.tensor([5]).numpy(),
    )

    branch = pool.allocations["branch"]
    assert branch[0][0] == source[0][0]
    assert branch[1][:2] == source[1][:2]
    assert branch[0][1] != source[0][1]
    assert branch[1][2] != source[1][2]
    assert torch.equal(caches[0][branch[0][1]], caches[0][source[0][1]])
    assert torch.equal(caches[1][branch[1][2]], caches[1][source[1][2]])

    pool.release(["branch", "request"])
    assert all(not any(group) for group in pool.block_refcounts)


def test_group_branch_budget_fails_independently_per_group() -> None:
    selected = select_branches_with_group_budget(
        block_costs=[[1, 3], [2, 1], [1, 1]],
        request_indices=[0, 0, 1],
        accepted_counts=[0, 1, 0],
        candidate_indices=[0, 0, 0],
        available_slots=2,
        available_blocks=[3, 2],
    )

    assert selected == [1, 2]


def test_draft_block_pool_rejects_mismatched_batch_metadata() -> None:
    pool = runtime.DraftBlockPool.__new__(runtime.DraftBlockPool)

    with pytest.raises(ValueError, match="IDs, epochs, and sequence lengths"):
        pool.ensure(["request"], [], torch.tensor([1]).numpy())


def test_draft_block_pool_rejects_sequence_beyond_table_row() -> None:
    block_tables = SimpleNamespace(
        block_sizes=[16],
        blocks_per_kv_block=[1],
        block_tables=[SimpleNamespace(gpu=torch.zeros(1, 32, dtype=torch.int32))],
        num_blocks=SimpleNamespace(
            np=torch.zeros(1, 1, dtype=torch.int32).numpy(),
            copy_to_uva=lambda: None,
        ),
    )
    runner = SimpleNamespace(
        block_tables=block_tables,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()], num_blocks=1024),
        max_num_reqs=1,
    )
    pool = runtime.DraftBlockPool(runner)

    with pytest.raises(runtime.DraftCapacityError, match="exceeds one block-table row"):
        pool.ensure(["profile"], [0], torch.tensor([4098]).numpy())


def test_draft_kv_cache_reserves_transient_profile_row(monkeypatch) -> None:
    observed_max_model_lens = []
    runner = SimpleNamespace(
        device=torch.device("cuda:0"),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.8),
        max_model_len=512,
        vllm_config=object(),
        get_kv_cache_spec=lambda: object(),
    )

    def initialize_kv_cache(config) -> None:
        observed_max_model_lens.append(runner.max_model_len)
        runner.kv_cache_config = config

    runner.initialize_kv_cache = initialize_kv_cache
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (900, 1000))
    monkeypatch.setattr(
        runtime,
        "get_kv_cache_configs",
        lambda *args: [SimpleNamespace(num_blocks=64)],
    )

    runtime._initialize_draft_kv_cache(runner, min_block_table_seq_len=4101)

    assert observed_max_model_lens == [4101]
    assert runner.max_model_len == 512
    assert runner.cache_config.num_gpu_blocks == 64


def test_branch_cudagraph_sizes_cover_b1_b4_b16() -> None:
    assert runtime._branch_cudagraph_capture_sizes(7, 16) == {
        24,
        48,
        96,
        192,
        384,
    }
    assert runtime._branch_cudagraph_capture_sizes(3, 4, fan_out=12) == {
        48,
        96,
        192,
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
        dspark_round_state,
    ) = runtime._run_proposal(runner, block_pool, cache, [ring_slot], batch, (4,))

    assert hit_indices == [0]
    assert trace_top2 is None
    assert dspark_round_state is None
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


@pytest.mark.parametrize(
    ("method", "target_architecture", "draft_architecture"),
    [
        (
            "mtp",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeMTP",
        ),
        (
            "dspark",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3DSparkModel",
        ),
        (
            "dspark",
            "Gemma4ForConditionalGeneration",
            "Qwen3DSparkModel",
        ),
    ],
)
def test_phase_b_async_draft_configs_validate(
    method, target_architecture, draft_architecture
) -> None:
    config = _make_async_draft_validation_config()
    config.speculative_config.method = method
    config.speculative_config.use_gemma4_mtp = lambda: False
    config.model_config.architecture = target_architecture
    config.speculative_config.draft_model_config.architecture = draft_architecture

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


@pytest.mark.parametrize("invalid", [None, "batch", "graph", "kv", "length"])
def test_gemma4_async_config_bounds(invalid) -> None:
    config = _make_async_draft_validation_config()
    config.speculative_config.method = "mtp"
    config.speculative_config.draft_model_config.architecture = "Gemma4MTPModel"
    config.model_config.architecture = "Gemma4ForConditionalGeneration"
    config.model_config.enforce_eager = invalid != "graph"
    config.model_config.max_model_len = 2048 if invalid == "length" else 512
    config.model_config.hf_config = SimpleNamespace(
        get_text_config=lambda: SimpleNamespace(sliding_window=1024)
    )
    config.scheduler_config = SimpleNamespace(
        max_num_seqs=2 if invalid == "batch" else 1
    )
    config.cache_config.cache_dtype = "fp8" if invalid == "kv" else "auto"
    if invalid is None:
        VllmConfig._validate_async_draft_config(config, _FakeCudaPlatform())
    else:
        with pytest.raises(ValueError, match="Gemma4 MTP Async"):
            VllmConfig._validate_async_draft_config(config, _FakeCudaPlatform())


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
        ("speculative_config", "method", "unknown", "method='unknown'"),
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.async_draft import runtime
from vllm.v1.worker.gpu.spec_decode.async_draft.cache import BranchCache, CachedBranch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def test_markov_diagnostics_do_not_change_greedy_tie_breaking():
    spec = SimpleNamespace(
        draft_logits=None,
        model=SimpleNamespace(
            markov_embed=lambda tokens: tokens,
            markov_bias=lambda tokens: torch.zeros(tokens.shape[0], 4),
            map_draft_to_target=lambda tokens: tokens,
        ),
    )
    logits = torch.ones(2, 3, 4)
    plain, _ = runtime._dspark_markov_propose(
        spec, logits, torch.tensor([0, 1]), record_top2=False
    )
    traced, _ = runtime._dspark_markov_propose(
        spec, logits, torch.tensor([0, 1]), record_top2=True
    )
    assert torch.equal(plain, traced)
    assert traced.tolist() == [[0, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_explicit_branch_query(sample_from_anchor, depth):
    spec = SimpleNamespace(
        sample_from_anchor=sample_from_anchor,
        num_speculative_steps=3,
        parallel_drafting_token_id=99,
    )
    prefix = torch.tensor([[10, *[11, 12, 13][:depth], 20]])
    query, offset = DSparkSpeculator.branch_query(spec, prefix)
    expected = prefix[0].tolist() + [99] * (2 if sample_from_anchor else 3)
    assert query.tolist() == [expected]
    assert offset == depth + (1 if sample_from_anchor else 2)
    assert query.shape[1] - offset == 3
    assert spec.num_speculative_steps == 3


@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("sample_count", [1, 3])
def test_eager_query_samples_only_selected_positions(
    monkeypatch, sample_from_anchor, sample_count
):
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as module

    spec = SimpleNamespace(
        max_num_reqs=2,
        max_num_tokens=32,
        max_model_len=512,
        sample_from_anchor=sample_from_anchor,
        num_speculative_steps=3,
        parallel_drafting_token_id=99,
        input_buffers=SimpleNamespace(
            input_ids=torch.zeros(32, dtype=torch.long),
            positions=torch.zeros(32, dtype=torch.long),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            seq_lens=torch.zeros(2, dtype=torch.int32),
        ),
        kv_cache_config=None,
        model=SimpleNamespace(compute_draft_logits=lambda hidden: hidden),
    )
    calls = []
    slots = torch.tensor([7, 3])

    def gather(actual, count):
        assert torch.equal(actual, slots) and count == 2
        calls.append("gather")

    def mapping(actual, starts, positions, count):
        assert torch.equal(actual, slots)
        assert starts.tolist() == [0, count // 2, count]
        calls.append("slots")
        return positions

    def metadata(self, num_reqs, padded, count, query_width, causal):
        assert num_reqs == padded == 2 and count == 2 * query_width
        assert causal is False
        calls.append("metadata")
        return {}

    spec.block_tables = SimpleNamespace(
        gather_block_tables=gather, compute_slot_mappings=mapping
    )
    spec._run_model = lambda count, *args: torch.stack(
        (spec.input_buffers.input_ids[:count], spec.input_buffers.positions[:count]),
        dim=-1,
    )
    monkeypatch.setattr(
        module.DraftModelSpeculator, "_build_draft_attn_metadata", metadata
    )
    monkeypatch.setattr(
        module, "build_slot_mappings_by_layer", lambda mappings, config: {}
    )
    query, start = DSparkSpeculator.branch_query(
        spec, torch.tensor([[10, 20], [11, 21]])
    )
    result = DSparkSpeculator.branch_query_logits(
        spec, slots, query, torch.tensor([15, 31]), start, sample_count
    )
    assert result[:, :, 0].tolist() == query[:, start : start + sample_count].tolist()
    assert result[:, :, 1].tolist() == [
        list(range(p + start, p + start + sample_count)) for p in [15, 31]
    ]
    assert calls == ["gather", "slots", "metadata"]
    assert spec.num_speculative_steps == 3


@pytest.mark.parametrize("fail", [False, True])
def test_private_query_reclaims_pages_after_gpu_completion(monkeypatch, fail):
    events = []

    def clone(sources, ids, lengths, mutation_start_positions):
        events.append("clone")
        assert sources == ["r", "r"]
        assert lengths.tolist() == [20, 20]
        assert mutation_start_positions.tolist() == [15, 15]
        return torch.tensor([1, 2])

    def forward(*args):
        events.append("forward")
        if fail:
            raise RuntimeError("forward failed")
        return torch.zeros(2, 3, 8)

    spec = SimpleNamespace(
        max_model_len=512,
        branch_query=lambda prefix: (torch.zeros(2, 5), 2),
        branch_query_logits=forward,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("done"))
    pool = SimpleNamespace(
        clone_many=clone, release=lambda ids: events.append("release")
    )
    args = (
        SimpleNamespace(speculator=spec, device="cpu"),
        pool,
        SimpleNamespace(req_ids=["r"], generation=3),
        [0, 0],
        torch.tensor([[10, 11, 20], [10, 11, 21]]),
        torch.tensor([15, 15]),
        3,
    )
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            runtime._dspark_private_query(*args)
    else:
        runtime._dspark_private_query(*args)
    assert events == ["clone", "forward", "done", "release"]


def test_fanout_reexecutes_each_depth_without_future_prefix(monkeypatch):
    class Spec:
        model = SimpleNamespace(
            markov_embed=lambda x: x,
            markov_bias=lambda x: torch.zeros(x.shape[0], 8),
        )

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    queries = []
    exclusions = []

    def query(runner, pool, batch, indices, prefix, positions, count):
        queries.append((prefix.tolist(), count))
        assert positions.tolist() == [15] * len(indices)
        return torch.zeros(len(indices), count, 8), prefix, prefix.shape[1] - 1

    def candidates(spec, logits, returned, fan_out):
        exclusions.append(None if returned is None else returned.tolist())
        return torch.tensor([[20, 21, 22]])

    monkeypatch.setattr(runtime, "DSparkSpeculator", Spec)
    monkeypatch.setattr(runtime, "_dspark_private_query", query)
    monkeypatch.setattr(runtime, "_dspark_top_recovery_candidates", candidates)
    monkeypatch.setattr(
        runtime,
        "_dspark_markov_propose",
        lambda s, x, p, **kw: (p[:, None].expand(-1, 3).clone(), None),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    cache = BranchCache()
    _, evictions, metrics = runtime._build_dspark_round_fanout_branches(
        SimpleNamespace(
            speculator=Spec(),
            async_target_verify_width=3,
            async_draft_fan_out=3,
            device="cpu",
        ),
        None,
        cache,
        SimpleNamespace(
            draft_tokens=torch.tensor([[11, 12, 13]]),
            query_start_loc=torch.tensor([0, 1]),
            num_rejected=torch.tensor([0]),
            positions=torch.tensor([14]),
        ),
        SimpleNamespace(
            num_reqs=1,
            req_ids=["r"],
            request_epochs=[2],
            engine_instance_id="e",
            generation=7,
        ),
        (8,),
        [],
        runtime.DSparkRoundState(
            base_logits=[torch.zeros(3, 8)], anchor_tokens=[torch.tensor(10)]
        ),
    )
    assert evictions == 0
    assert exclusions == [[[11]], [[12]], [[13]], None]
    assert metrics["dspark_candidate_backbone_forwards"] == 3
    assert metrics["dspark_branch_backbone_forwards"] == 4
    assert metrics["fanout_branches"] == len(cache.entries) == 12
    branch_queries = [prefix for prefix, count in queries if count == 3]
    for depth, prefixes in enumerate(branch_queries):
        assert prefixes == [[10, *[11, 12, 13][:depth], r] for r in [20, 21, 22]]
    candidate_queries = [prefix for prefix, count in queries if count == 1]
    assert candidate_queries == [[[10, 11]], [[10, 11, 12]], [[10, 11, 12, 13]]]


def test_mixed_dspark_hits_and_miss_preserve_request_order(monkeypatch):
    class Spec:
        hidden_size = 4
        sample_from_anchor = True
        _async_base_logits = torch.zeros(1, 3, 8)
        input_buffers = SimpleNamespace(input_ids=torch.tensor([20, 99, 99]))

        def anchor_indices(self, count, **kwargs):
            return torch.tensor([0])

    monkeypatch.setattr(runtime, "DSparkSpeculator", Spec)
    monkeypatch.setattr(runtime, "DraftModelSpeculator", Spec)
    monkeypatch.delenv("ASYNC_DRAFT_FORCE_JIT", raising=False)
    monkeypatch.delenv("REPLAYSSM_SPEC_DECODE_TRACE_LOGITS", raising=False)
    batch = SimpleNamespace(
        num_reqs=3,
        transient=False,
        slot=0,
        engine_instance_id="e",
        req_ids=["a", "b", "c"],
        request_epochs=[1, 2, 3],
    )
    slot = SimpleNamespace(
        num_sampled=torch.tensor([1, 2, 4]),
        last_sampled=torch.tensor([10, 20, 30]),
        draft_tokens=torch.zeros(3, 3, dtype=torch.long),
    )
    cache = BranchCache()
    cache.add(
        ("e", "a", 1, 0, 10),
        CachedBranch(branch_id="cache_a", tokens=torch.tensor([11, 12, 13])),
    )
    cache.add(
        ("e", "c", 3, 3, 30),
        CachedBranch(branch_id="cache_c", tokens=torch.tensor([31, 32, 33])),
    )

    def slice_batch(batch, slot, indices):
        assert indices == [1]
        return SimpleNamespace(num_reqs=1), slot

    def jit(*args, **kwargs):
        assert kwargs["num_speculative_steps"] == 3
        return torch.tensor([[21, 22, 23]]), None, 0

    monkeypatch.setattr(runtime, "_slice_proposal_batch", slice_batch)
    monkeypatch.setattr(runtime, "_execute_jit_proposal", jit)
    metrics, _, hits, _, state = runtime._run_proposal(
        SimpleNamespace(
            speculator=Spec(),
            num_speculative_steps=3,
            async_target_verify_width=3,
            async_branch_backbone_width=3,
        ),
        SimpleNamespace(release=lambda ids: None),
        cache,
        [slot],
        batch,
        (4,),
    )
    assert slot.draft_tokens.tolist() == [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
    assert hits == [0, 2]
    assert metrics["cache_hits"] == 2
    assert state.base_logits[0] is None and state.base_logits[2] is None
    assert state.base_logits[1].shape == (3, 8)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.moe_skip import speculator as module
from vllm.v1.worker.gpu.spec_decode.moe_skip.speculator import (
    _external_request_id,
    _is_internal_request,
)


def test_moe_skip_trace_filters_internal_requests():
    assert _is_internal_request("_warmup_0_")
    assert _is_internal_request("_profile_3_")
    assert not _is_internal_request("0-bd638b82")


def test_moe_skip_trace_restores_external_request_id():
    assert _external_request_id("0-bd638b82") == "0"
    assert _external_request_id("user-request-0123abcd") == "user-request"
    assert _external_request_id("user-request-nothexid") == "user-request-nothexid"


def test_text_draft_preserves_shared_target_compiled_input_signature(monkeypatch):
    """The shared AOT graph requires token IDs, not an embeddings-only call."""
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(module, "set_forward_context", context)
    spec = module.MoeSkipSpeculator.__new__(module.MoeSkipSpeculator)
    ids = torch.tensor([7, 8, 0])
    spec.input_buffers = SimpleNamespace(
        input_ids=ids, positions=torch.arange(3), is_padding=torch.tensor([0, 0, 1])
    )
    spec.position_dims = 1
    spec.vllm_config = None
    spec.top_h = 8
    spec.min_weight = None
    spec.batch_policy = "batch_top_half"
    spec.preserve_weights = True
    spec.model = Mock(return_value=torch.ones(2, 4))
    output = spec._run_model(2, {}, {}, None, CUDAGraphMode.NONE)
    assert output.shape == (2, 4)
    kwargs = spec.model.call_args.kwargs
    assert torch.equal(kwargs["input_ids"], ids[:2])
    assert kwargs.get("inputs_embeds") is None
    spec.model.embed_input_ids.assert_not_called()
    assert (
        context.call_args.kwargs["additional_forward_kwargs"]["routing_batch_policy"]
        == "batch_top_half"
    )


def test_draft_ignores_padded_lengths_when_requests_finish(monkeypatch):
    """Three remaining requests can still arrive in a four-request graph bucket."""
    prepare = Mock()
    monkeypatch.setattr(module, "prepare_decode_inputs", prepare)
    desc = SimpleNamespace(num_reqs=4, num_tokens=4, cg_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(module, "dispatch_cg_and_sync_dp", lambda *a, **k: (desc, None))
    spec = module.MoeSkipSpeculator.__new__(module.MoeSkipSpeculator)
    spec.input_buffers = SimpleNamespace(
        positions=torch.zeros(4, dtype=torch.int64),
        is_padding=torch.zeros(4, dtype=torch.bool),
    )
    spec.idx_mapping = torch.zeros(4, dtype=torch.int64)
    spec.initial_tokens = torch.zeros(4, dtype=torch.int64)
    spec.sample_src_positions = torch.zeros(4, dtype=torch.int64)
    spec.draft_tokens = torch.zeros((4, 4), dtype=torch.int64)
    spec.max_model_len, spec.max_num_reqs = 128, 4
    spec.num_speculative_steps = 0
    spec.decode_cudagraph_manager = Mock()
    spec.block_tables, spec.model_state = Mock(), Mock()
    spec.trace_dir = None
    batch = SimpleNamespace(
        num_reqs=3,
        idx_mapping=torch.tensor([0, 2, 3]),
        seq_lens=torch.tensor([10, 20, 30, 999]),
    )
    result = spec.propose(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=None,
        aux_hidden_states=None,
        num_sampled=None,
        num_rejected=torch.tensor([0, 2, 3, 99]),
        last_sampled=torch.arange(4).reshape(4, 1),
        next_prefill_tokens=None,
        temperature=None,
        seeds=None,
    )
    assert result.shape == (3, 4)
    assert spec.input_buffers.positions[:3].tolist() == [9, 17, 26]
    assert spec.input_buffers.is_padding.tolist() == [False, False, False, True]
    assert prepare.call_args.args[1].tolist() == [10, 20, 30]
    assert prepare.call_args.args[2].tolist() == [0, 2, 3]

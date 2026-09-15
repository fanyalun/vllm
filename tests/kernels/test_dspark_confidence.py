# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    Qwen3DSparkForCausalLM,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


@pytest.mark.parametrize("rank", [0, 3])
def test_confidence_head_reference(rank):
    torch.manual_seed(0)
    head = DSparkConfidenceHead(8, rank)
    hidden = torch.randn(2, 4, 8)
    embedding = torch.randn(2, 4, 3)
    features = torch.cat((hidden, embedding), -1) if rank else hidden
    expected = torch.sigmoid(
        (
            features.double() @ head.proj.weight.double().T + head.proj.bias.double()
        ).squeeze(-1)
    )
    torch.testing.assert_close(head(hidden, embedding).double(), expected)


@pytest.mark.parametrize(
    "enabled,missing", [(True, False), (True, True), (False, False)]
)
def test_confidence_weight_loading(enabled, missing):
    model = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.confidence_head = DSparkConfidenceHead(8, 3) if enabled else None
    model.model._build_fused_kv_buffers = lambda: None
    weights = {"confidence_head.proj.weight": torch.full((1, 11), 0.25)}
    if not missing:
        weights["confidence_head.proj.bias"] = torch.tensor([0.5])
    if missing:
        with pytest.raises(ValueError, match="Missing DSpark confidence weights"):
            model.load_weights(weights.items())
    else:
        model.load_weights(weights.items())
        if enabled:
            torch.testing.assert_close(
                model.model.confidence_head.proj.weight,
                weights["confidence_head.proj.weight"],
            )
            torch.testing.assert_close(
                model.model.confidence_head.proj.bias, torch.tensor([0.5])
            )


@pytest.mark.parametrize("draft", [4, 8])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_confidence_sampling_alignment(draft, graph, enabled):
    device = "cuda"
    torch.manual_seed(17)
    head = DSparkConfidenceHead(8, 3).to(device)
    embeddings = torch.randn(20, 3, device=device)
    projection = torch.randn(3, 5, device=device)
    hidden = torch.randn(2 * draft, 8, device=device)
    model = SimpleNamespace(
        compute_draft_logits=lambda h: h[:, :5],
        markov_embed=lambda t: embeddings[t.long()],
        markov_bias=lambda e: e @ projection,
        map_draft_to_target=lambda t: t + 10,
        compute_draft_confidence=head,
    )
    obj = SimpleNamespace(
        model=model,
        max_num_reqs=3,
        num_speculative_steps=draft,
        sample_indices=torch.arange(2 * draft, device=device).flip(0),
        sample_idx_mapping=torch.zeros(2 * draft, device=device, dtype=torch.int32),
        sample_pos=torch.arange(2 * draft, device=device),
        input_buffers=SimpleNamespace(input_ids=torch.tensor([2, 7], device=device)),
        _anchor_idx=torch.arange(2, device=device),
        draft_logits=None,
        draft_lengths=None,
        draft_tokens=torch.zeros(3, draft, dtype=torch.int64, device=device),
        draft_confidence=torch.full((3, draft), float("nan"), device=device)
        if enabled
        else None,
    )

    def run():
        DSparkSpeculator._sample_sequential(obj, 2, hidden)

    run()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            run()
    for step in range(3):
        hidden.add_(0.1)
        obj.input_buffers.input_ids.copy_(torch.tensor([step, 7 - step], device=device))
        if graph:
            captured.replay()
        else:
            run()
        sampled_hidden = hidden[obj.sample_indices].view(2, draft, 8)
        prev = obj.input_buffers.input_ids.clone()
        for i in range(draft):
            previous_embedding = embeddings[prev]
            expected_token = (
                sampled_hidden[:, i, :5] + previous_embedding @ projection
            ).argmax(-1) + 10
            torch.testing.assert_close(obj.draft_tokens[:2, i], expected_token)
            if enabled:
                features = torch.cat((sampled_hidden[:, i], previous_embedding), -1)
                expected = (
                    (features @ head.proj.weight.T + head.proj.bias)
                    .squeeze(-1)
                    .sigmoid()
                )
                torch.testing.assert_close(obj.draft_confidence[:2, i], expected)
            prev = expected_token
        confidence = DSparkSpeculator.get_draft_confidence(obj, 2)
        if enabled:
            assert confidence.shape == (2, draft)
            assert torch.isnan(obj.draft_confidence[2]).all()
        else:
            assert confidence is None

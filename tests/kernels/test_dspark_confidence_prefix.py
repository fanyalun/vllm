# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import confidence_prefix_lengths
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


@pytest.mark.parametrize("graph", [False, True])
def test_prefix_lengths_and_scheduler_transport(graph):
    confidence = torch.tensor(
        [[0.9, 0.8, 0.7, 0.99], [0.6, 0.99, 0.99, 0.99], [0.9] * 4], device="cuda"
    )
    lengths = torch.empty(3, dtype=torch.int32, device="cuda")

    def run():
        lengths.copy_(confidence_prefix_lengths(confidence, 0.8))

    run()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            run()
        captured.replay()
    assert lengths.tolist() == [2, 0, 4]
    handler = DraftTokensHandler(torch.device("cuda"))
    batch = SimpleNamespace(
        req_ids=["a", "b", "c"], num_reqs=3, has_structured_output_reqs=False
    )
    tokens = torch.arange(12, device="cuda").view(3, 4)
    handler.set_draft_tokens(batch, tokens, lengths)
    out = handler.get_draft_tokens()
    assert [len(x) for x in out.draft_token_ids] == [2, 0, 4]
    batch.has_structured_output_reqs = True
    handler.set_draft_tokens(batch, tokens, lengths)
    assert handler.get_draft_tokens().draft_token_ids == [[0, 1], [], [8, 9, 10, 11]]
    confidence.fill_(float("nan"))
    if graph:
        captured.replay()
    else:
        run()
    assert lengths.tolist() == [0, 0, 0]
    handler.set_draft_tokens(batch, tokens)
    assert handler.get_draft_tokens().draft_token_ids == tokens.tolist()

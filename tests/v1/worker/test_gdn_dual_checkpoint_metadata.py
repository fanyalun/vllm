# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states import mamba_hybrid


def test_v2_prefill_lengths_follow_request_order(monkeypatch):
    monkeypatch.setattr(
        mamba_hybrid.DefaultModelState, "add_request", lambda *args: None
    )
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", lambda **kwargs: kwargs)
    state = object.__new__(mamba_hybrid.MambaHybridModelState)
    state._dual_checkpoint = True
    state._align_mode = False
    state._replayssm_prefill_lens = torch.zeros(3, dtype=torch.int32)
    state.num_accepted_tokens_gpu = torch.ones(3, dtype=torch.int32)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=4)
    for index, length in [(0, 20), (2, 10), (0, 24)]:
        state.add_request(index, SimpleNamespace(prefill_token_ids=[1] * length))
    batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        num_tokens=10,
        num_tokens_after_padding=10,
        query_start_loc_np=np.array([0, 5, 10, 10], dtype=np.int32),
        query_start_loc=torch.tensor([0, 5, 10, 10], dtype=torch.int32),
        num_scheduled_tokens=np.array([5, 5], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([15, 29, 0], dtype=torch.int32),
        seq_lens=torch.tensor([15, 29, 0], dtype=torch.int32),
        is_prefilling_np=np.array([False, False]),
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        num_draft_tokens_per_req=np.array([4, 4], dtype=np.int32),
        prompt_lens=None,
        dcp_local_seq_lens=None,
    )
    kwargs = state.prepare_attn(batch, CUDAGraphMode.FULL, (), (), [], None)
    metadata = kwargs["model_specific_attn_metadata"]
    common = metadata.get_extra_common_attn_kwargs(0, 3)
    assert common["num_prompt_tokens_cpu"].tolist() == [10, 24, 0]
    assert common["is_prefilling"].tolist() == [False, False, False]

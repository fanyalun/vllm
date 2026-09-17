# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct real candidate mismatches and audit four rounds without early stop."""

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.worker.gpu.spec_decode.hierarchical.speculator import accepted_prefix


@torch.inference_mode()
def audit(spec, template, position, anchor):
    state = spec.state
    initial = state.snapshot()
    trajectories = []
    axis = 2 if is_conv_state_dim_first() else 1
    try:
        for desired in ([4, 4, 4, 4], [2, 2, 2, 2], [0, 4, 4, 4], [0, 0, 0, 0]):
            state.restore(initial)
            offset, correction = 0, anchor.clone()
            trace = []
            for requested in desired:
                before = state.snapshot()
                inputs = torch.zeros(5, device=spec.device, dtype=torch.int64)
                inputs[0] = correction

                def forward(offset=offset, inputs=inputs):
                    batch, metadata, slots = spec._batch(
                        template, position + offset, inputs
                    )
                    return spec._verify_eager(batch, metadata, slots)[0].clone()

                # Causality lets each probe fix one more matching candidate.
                for index in range(requested + (requested < 4)):
                    state.restore(before)
                    predicted = forward()
                    inputs[index + 1] = predicted[index]
                    if index == requested:
                        inputs[index + 1] = (
                            predicted[index] + 1
                        ) % spec.last_logits.shape[-1]
                state.restore(before)
                predicted = forward()
                accepted = accepted_prefix(inputs[1:], predicted)
                assert accepted == requested
                tail = state.snapshot()
                state.advance(accepted)
                for name, (conv, temporal) in state.caches.items():
                    assert torch.equal(temporal, tail[name][1])
                    indices = (
                        torch.arange(conv.shape[axis], device=conv.device) + accepted
                    ).clamp(max=conv.shape[axis] - 1)
                    assert torch.equal(conv, tail[name][0].index_select(axis, indices))
                correction = predicted[accepted].clone()
                trace.append(
                    dict(
                        offset=offset,
                        input_ids=inputs.tolist(),
                        predictions=predicted.tolist(),
                        accepted=accepted,
                        correction=int(correction),
                        ssm_tail_bitwise_preserved=True,
                        conv_accepted_position_equal=True,
                    )
                )
                offset += accepted + 1
            for previous, current in zip(trace, trace[1:]):
                assert current["input_ids"][0] == previous["correction"]
                assert (
                    current["offset"] == previous["offset"] + previous["accepted"] + 1
                )
            trajectories.append(dict(requested=desired, rounds=trace))
        return trajectories
    finally:
        state.restore(initial)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-window forward and consumed-prefix cost, outside generation timing."""

import copy
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.gdn.replay_tail_update import (
    advance_replay_tail_conv,
    advance_replay_tail_convs,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


@torch.inference_mode()
def measure(spec, template, path):
    original_state, original_config = spec.state, spec.config
    tokens = template.input_ids.clone()
    position = int(template.positions[0])
    source = original_state.snapshot()
    source_slot = 1 if original_state.mode == "none" else 0
    cases = ("exact", "three_level_base", "three_level", "repair_on_reject")
    rows, parity = [], {}
    flush = torch.empty(128 * 1024 * 1024, device=spec.device, dtype=torch.uint8)
    references = {}
    captures = {}
    try:
        for case in cases:
            state = PreverifyState(
                spec.model,
                5,
                spec.device,
                "replay_tail",
                "exact" if case == "exact" else "three_level_p50",
                "repair_on_reject" if case == "repair_on_reject" else "carry",
            )
            if case == "three_level_base":
                state.execution_optimized = state.kernel_tuned = False
            for name, (conv, temporal) in state.caches.items():
                state._copy_conv(
                    conv, source[name][0][source_slot : source_slot + 1], 0
                )
                temporal.copy_(source[name][1][source_slot : source_slot + 1])
            initial = state.snapshot()
            spec.state = state
            spec.config = copy.copy(original_config)
            spec.config.preverify_gdn_mode = "replay_tail"
            batch, metadata, slots = spec._batch(template, position, tokens)
            for boundary in ("forward", "combined"):

                def execute(
                    boundary=boundary,
                    case=case,
                    state=state,
                    batch=batch,
                    metadata=metadata,
                    slots=slots,
                ):
                    output = spec._verify_eager(batch, metadata, slots)
                    if boundary == "combined":
                        if case == "repair_on_reject":
                            state._repair_prefix(3)
                        if (
                            state.execution_optimized
                            and state.conv_pointers is not None
                        ):
                            advance_replay_tail_convs(
                                state.conv_pointers,
                                next(iter(state.caches.values()))[0],
                                2,
                                is_conv_state_dim_first(),
                            )
                        else:
                            for conv, _ in state.caches.values():
                                advance_replay_tail_conv(
                                    conv, 2, is_conv_state_dim_first()
                                )
                    return output

                for _ in range(3):
                    state.restore(initial)
                    execute()
                state.restore(initial)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = execute()
                prediction = output[0].clone()
                if boundary == "forward":
                    if case == "three_level_base":
                        references["predictions"] = prediction
                        references["logits"] = spec.last_logits.clone()
                        references["tails"] = {
                            n: x.clone() for n, x in state.tails.items()
                        }
                    elif case in ("three_level", "repair_on_reject"):
                        torch.testing.assert_close(
                            prediction, references["predictions"], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            spec.last_logits, references["logits"], rtol=1e-3, atol=1e-3
                        )
                        for name, tail in state.tails.items():
                            torch.testing.assert_close(
                                tail, references["tails"][name], rtol=1e-3, atol=1e-3
                            )
                        parity[case] = dict(
                            tokens_equal=True,
                            max_logit_error=float(
                                (spec.last_logits - references["logits"]).abs().max()
                            ),
                        )
                captures[case, boundary] = (
                    state,
                    initial,
                    graph,
                    output,
                    metadata,
                    slots,
                )
        timings = {key: [] for key in captures}
        keys = list(captures)
        for repeat in range(250):
            for key in keys if repeat % 2 == 0 else keys[::-1]:
                state, initial, graph, *_ = captures[key]
                state.restore(initial)
                flush.zero_()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                if repeat >= 50:
                    timings[key].append(start.elapsed_time(end))
        for (case, boundary), times in timings.items():
            rows.append(
                dict(
                    case=case,
                    boundary=boundary,
                    median_ms=statistics.median(times),
                    samples_ms=times,
                )
            )
    finally:
        spec.state, spec.config = original_state, original_config
        original_state.restore(source)
    Path(path).write_text(
        json.dumps(
            dict(
                rows=rows,
                parity=parity,
                consumed=3,
                position=position,
                input_ids=tokens.tolist(),
                warmup=50,
                repeats=200,
                complete=True,
            ),
            indent=2,
        )
    )

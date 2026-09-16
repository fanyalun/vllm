# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-window shared-input ablations, with state restoration outside timing."""

import copy

import torch
from replay_tail_cost_worker import ReplayTailCostWorker

from vllm.model_executor.layers.mamba.gdn import grouped_input
from vllm.v1.worker.gpu.spec_decode.hierarchical import speculator as impl
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState

CASES = [
    f"{s}:{g}"
    for s in ("none", "replay_tail")
    for g in ("none", "serial", "projection", "full")
]
BOUNDARIES = ("projection", "gdn", "forward", "advance", "combined")


class GroupedGDNCostWorker(ReplayTailCostWorker):
    @torch.inference_mode()
    def _measure(self, spec, original_batch):
        tokens = original_batch.input_ids.clone()
        assert tokens.numel() == 5
        position = int(original_batch.positions[0])
        self._reuse_ids, self._reuse_position = tokens.tolist(), position
        original_state, original_config = spec.state, spec.config
        initial = original_state.snapshot()
        states, snapshots, graphs, configs = {}, {}, {}, {}
        state_pools = {}
        plan = spec.grouped_gdn
        anchors = {}
        original_projection = plan.projections
        flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        try:
            spec.config = copy.copy(original_config)
            spec.config.preverify_gdn_group_mode = "serial"
            batch, metadata, slots = spec._batch(original_batch, position, tokens)

            def capture(group, anchor, serial=False):
                anchors[group] = anchor.clone()
                return original_projection(group, anchor, serial)

            plan.projections = capture
            spec._verify_eager(batch, metadata, slots)
            plan.projections = original_projection
            original_state.restore(initial)
            assert len(anchors) == 7
            for case in CASES:
                state_mode, group_mode = case.split(":")
                if state_mode not in state_pools:
                    state = PreverifyState(spec.model, 5, spec.device, state_mode)
                    slot = 1 if state_mode == "none" else 0
                    for name, (conv, ssm) in state.caches.items():
                        state._copy_conv(conv[slot : slot + 1], initial[name][0], 0)
                        ssm[slot : slot + 1].copy_(initial[name][1])
                    state_pools[state_mode] = state, state.snapshot()
                state, snapshot = state_pools[state_mode]
                spec.state = states[case] = state
                snapshots[case] = snapshot
                state.restore(snapshot)
                spec.config = configs[case] = copy.copy(original_config)
                spec.config.preverify_gdn_mode = state_mode
                spec.config.preverify_gdn_group_mode = group_mode
                batch, metadata, slots = spec._batch(original_batch, position, tokens)

                def isolated(
                    boundary,
                    state=state,
                    metadata=metadata,
                    slots=slots,
                    state_mode=state_mode,
                    group_mode=group_mode,
                ):
                    with (
                        state.activate(),
                        impl.use_workspace_lane(0),
                        impl.set_forward_context(
                            metadata,
                            spec.vllm_config,
                            num_tokens=5,
                            cudagraph_runtime_mode=impl.CUDAGraphMode.NONE,
                            slot_mapping=slots,
                            additional_forward_kwargs={
                                "preverify_gdn_mode": state_mode
                            },
                        ),
                    ):
                        result = []
                        for group, anchor in anchors.items():
                            qkvz, ba = plan.projections(
                                group, anchor, serial=group_mode in ("none", "serial")
                            )
                            if boundary == "projection":
                                result.append((qkvz, ba))
                            elif group_mode == "full":
                                result.append(
                                    plan.branches(group, qkvz, ba, state_mode)
                                )
                            else:
                                result.append(
                                    torch.stack(
                                        [
                                            plan.branch(
                                                plan.model.layers[index].linear_attn,
                                                qkvz[j],
                                                ba[j],
                                                state_mode,
                                            )
                                            for j, index in enumerate(group)
                                        ]
                                    )
                                )
                        return result

                for boundary in BOUNDARIES:

                    def execute(
                        boundary=boundary,
                        state=state,
                        batch=batch,
                        metadata=metadata,
                        slots=slots,
                        isolated=isolated,
                    ):
                        if boundary in ("projection", "gdn"):
                            return isolated(boundary)
                        output = None
                        if boundary in ("forward", "combined"):
                            output = spec._verify_eager(batch, metadata, slots)
                        if boundary in ("advance", "combined"):
                            state.advance(2)
                        return output

                    for _ in range(3):
                        state.restore(snapshots[case])
                        execute()
                    state.restore(snapshots[case])
                    reference = execute()
                    expected_state = state.snapshot()
                    expected_logits = (
                        spec.last_logits.clone()
                        if boundary in ("forward", "combined")
                        else None
                    )
                    expected_tokens = (
                        reference[0].clone() if expected_logits is not None else None
                    )
                    state.restore(snapshots[case])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = execute()
                    graphs[case, boundary] = graph, output
                    state.restore(snapshots[case])
                    graph.replay()
                    if expected_logits is not None:
                        torch.testing.assert_close(
                            output[0], expected_tokens, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            spec.last_logits, expected_logits, atol=1e-3, rtol=1e-3
                        )
                    for name, cache in state.caches.items():
                        for actual, expected in zip(
                            cache, expected_state[name], strict=True
                        ):
                            torch.testing.assert_close(
                                actual, expected, atol=1e-3, rtol=1e-3
                            )
            reference_tokens = {}
            for repeat in range(35):
                for case in CASES if repeat % 2 == 0 else CASES[::-1]:
                    spec.state, spec.config = states[case], configs[case]
                    for boundary in BOUNDARIES:
                        spec.state.restore(snapshots[case])
                        if boundary == "advance":
                            graphs[case, "forward"][0].replay()
                        flush.zero_()
                        graph, output = graphs[case, boundary]
                        start, end = [
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        ]
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        if boundary in ("forward", "combined"):
                            predictions = output[0].tolist()
                            reference_tokens.setdefault((case, boundary), predictions)
                            assert predictions == reference_tokens[case, boundary]
                        if repeat >= 5:
                            self._reuse_rows.append(
                                dict(
                                    case=case,
                                    boundary=boundary,
                                    repeat=repeat - 5,
                                    ms=start.elapsed_time(end),
                                )
                            )
            self._reuse_checks = dict(
                gates="per_token",
                repeatable_predictions=True,
                graph_eager_states_equal=True,
                selected_groups=7,
                isolated_inputs=(
                    "identical shared anchors; none uses serial shared-input reference"
                ),
                accepted_drafts_for_advance=2,
            )
            self._reuse_checks["compiled_kernels"] = [
                dict(
                    name=name,
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                )
                for name in (
                    "_norm",
                    "_add_norm",
                    "_linear",
                    "_conv",
                    "_recurrent",
                    "_gated_norm",
                )
                for cache in getattr(grouped_input, name).device_caches.values()
                for kernel in cache[0].values()
            ]
        finally:
            plan.projections = original_projection
            spec.config, spec.state = original_config, original_state
            original_state.restore(initial)
            spec._batch(original_batch, position, tokens)

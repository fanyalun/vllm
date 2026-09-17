# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-input window quality and committed-prefix isolation outside timing."""

import copy
import json
from pathlib import Path

import torch
from three_level_worker import ThreeLevelWorker

from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class WindowedQualityWorker(ThreeLevelWorker):
    def prepare_window_quality(self, inputs, targets, path, rejection_audit=False):
        self._quality_pending = (inputs, targets, path)
        self._quality_rejections = rejection_audit
        spec = self.model_runner.speculator
        if getattr(self, "_quality_installed", False):
            return
        self._quality_installed = True
        original = spec._verify

        def verify(batch, metadata, slots):
            if self._quality_pending is not None:
                pending, self._quality_pending = self._quality_pending, None
                self._measure_window_quality(batch, *pending)
                # The audit uses shared input buffers; reconstruct the real window.
                batch, metadata, slots = spec._batch(
                    batch, self._quality_position, self._quality_tokens
                )
            return original(batch, metadata, slots)

        spec._verify = verify

    @torch.inference_mode()
    def _measure_window_quality(self, template, inputs, targets, path):
        spec = self.model_runner.speculator
        original_state, original_config = spec.state, spec.config
        assert original_state.mode == "none"
        source = original_state.snapshot()
        self._quality_tokens = template.input_ids.clone()
        position = int(template.positions[0])
        self._quality_position = position
        tokens = torch.tensor(inputs, device=spec.device, dtype=torch.int64)
        observed_anchor = int(self._quality_tokens[0])
        tables = spec.block_tables.gather_block_tables(template.idx_mapping, 1)
        attention = {
            m.layer_name: m for m in spec.model.modules() if isinstance(m, Attention)
        }
        request = int(template.idx_mapping[0])
        model_state = spec.model_state
        bias = int(model_state.num_accepted_tokens_gpu[request]) - 1
        source_index = (
            int(model_state._mamba_state_idx_gpu[request])
            if model_state._align_mode
            else 0
        )
        gdn_slots = {}
        for gid, group in enumerate(spec.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if name in original_state.layers:
                    gdn_slots[name] = (
                        int(tables[gid][0, source_index]),
                        int(tables[gid][0, source_index + bias]),
                    )

        def committed():
            values = {}
            for name, layer in original_state.layers.items():
                values[name] = tuple(
                    x[index : index + 1].cpu().clone()
                    for x, index in zip(layer.kv_cache, gdn_slots[name])
                )
            for gid, group in enumerate(spec.kv_cache_config.kv_cache_groups):
                for name in group.layer_names:
                    if name not in attention:
                        continue
                    cache = attention[name].kv_cache
                    assert cache.ndim == 4, f"Expected BHNC cache, got {cache.shape}"
                    block_size = cache.shape[2]
                    blocks = tables[gid][0].cpu().tolist()
                    pieces = []
                    for start in range(0, position, block_size):
                        pieces.append(
                            cache[
                                blocks[start // block_size],
                                :,
                                : min(block_size, position - start),
                            ]
                            .cpu()
                            .clone()
                        )
                    values[name] = tuple(pieces)
            return values

        canonical = committed()
        cases = [
            ("v0", "none", "exact", 5, "none", 0.36328125),
            ("v1", "replay_tail", "windowed_three_level", 5, "none", 0.0),
            ("v2", "replay_tail", "windowed_three_level", 1, "none", 0.36328125),
        ]
        cases += [
            (name, "replay_tail", "windowed_three_level", 5, opt, 0.36328125)
            for name, opt in (
                ("v3", "none"),
                ("v4_d", "cumulative_decay"),
                ("v4_q", "multi_query"),
                ("v4_dq", "combined"),
            )
        ]
        rows, reference_states, reference_predictions = [], {}, None
        target_case = ("target", "none", "exact", 5, "none", 0.36328125)
        target_logits = None
        costs = {}
        try:
            for name, mode, policy, window, optimization, beta in (
                [target_case] + cases + [cases[0], target_case]
            ):
                state = PreverifyState(
                    spec.model,
                    5,
                    spec.device,
                    mode,
                    policy,
                    window_size=window,
                    optimization=optimization,
                    tau_beta=beta,
                )
                for layer, (conv, temporal) in state.caches.items():
                    slot = 1 if mode == "none" else 0
                    state._copy_conv(conv[slot : slot + 1], source[layer][0][1:2], 0)
                    temporal[slot : slot + 1].copy_(source[layer][1][1:2])
                state.initialized, state.request_id = True, template.req_ids[0]
                state.valid.fill_(1)
                captures = []
                if (
                    name == "v1"
                    and Path(path).stem == "sample_0_boundary_96"
                    and not self._quality_rejections
                ):
                    original_update = state.update

                    def capture(
                        layer,
                        q,
                        k,
                        v,
                        a,
                        b,
                        _captures=captures,
                        _state=state,
                        _original=original_update,
                        **kwargs,
                    ):
                        if len(_captures) < 30:
                            values = [
                                q,
                                k,
                                v,
                                a,
                                b,
                                layer.A_log,
                                layer.dt_bias,
                                _state.caches[layer.prefix][1],
                            ]
                            _captures.append(
                                dict(
                                    layer=layer.prefix,
                                    strides=[list(x.stride()) for x in values],
                                    values=[x.detach().cpu().clone() for x in values],
                                )
                            )
                        return _original(layer, q, k, v, a, b, **kwargs)

                    state.update = capture
                spec.state, spec.config = state, copy.copy(original_config)
                spec.config.preverify_gdn_mode = mode
                spec.config.preverify_gdn_update_policy = policy
                if name == "target":
                    spec.config.moe_skip_top_h = 8
                    spec.config.moe_skip_min_weight = None
                rejections = None
                if self._quality_rejections and state.windowed:
                    from windowed_rejections import audit

                    rejections = audit(spec, template, position, tokens[0])
                predictions, states = [], {}
                logits = []
                for offset in range(0, len(inputs), 5):
                    width = min(5, len(inputs) - offset)
                    batch, metadata, slots = spec._batch(
                        template, position + offset, tokens[offset : offset + width]
                    )
                    if (
                        offset == 0
                        and name not in costs
                        and Path(path).stem == "sample_0_boundary_96"
                        and not self._quality_rejections
                    ):
                        from windowed_cost import measure

                        costs[name] = measure(
                            spec,
                            batch,
                            metadata,
                            slots,
                            Path(path).with_name(f"profile_{name}.json"),
                        )
                    result = spec._verify_eager(batch, metadata, slots)
                    predictions.extend(result[0].tolist())
                    if name == "target":
                        logits.append(spec.last_logits.clone())
                    endpoint = offset + width
                    if name == "v0":
                        snapshot = {
                            n: c[1][width : width + 1].clone()
                            for n, c in state.caches.items()
                        }
                        if endpoint not in reference_states:
                            reference_states[endpoint] = snapshot
                    elif name != "target":
                        numerator = denominator = 0.0
                        for layer, cache in state.caches.items():
                            expected = reference_states[endpoint][layer]
                            numerator += (cache[1] - expected).square().sum().item()
                            denominator += expected.square().sum().item()
                        states[endpoint] = (numerator / max(denominator, 1e-20)) ** 0.5
                    state.advance(width - 1)
                if name == "target":
                    current_logits = torch.cat(logits)
                    if target_logits is None:
                        target_logits = current_logits
                    else:
                        assert torch.equal(target_logits, current_logits), (
                            "Same-input Target logits changed"
                        )
                    continue
                if name == "v0":
                    if reference_predictions is None:
                        reference_predictions = predictions
                    else:
                        assert predictions == reference_predictions, (
                            "Same-input native regression after approximation"
                        )
                        continue
                accepted = next(
                    (i for i, (a, b) in enumerate(zip(predictions, targets)) if a != b),
                    len(targets),
                )
                rows.append(
                    dict(
                        case=name,
                        predictions=predictions,
                        targets=targets,
                        L5=min(accepted, 5),
                        L16=min(accepted, 16),
                        accepted_prefix=accepted,
                        state_relative_l2=states,
                        rejection_trajectories=rejections,
                    )
                )
                if captures:
                    assert len(captures) == 30
                    torch.save(captures, Path(path).with_name("raw_inputs.pt"))
                current = committed()
                for layer, tensors in canonical.items():
                    for kind, (before, after) in enumerate(
                        zip(tensors, current[layer])
                    ):
                        assert torch.equal(
                            before.contiguous().view(torch.uint8),
                            after.contiguous().view(torch.uint8),
                        ), (
                            f"Committed cache changed: case={name} "
                            f"layer={layer} kind={kind}"
                        )
        finally:
            spec.state, spec.config = original_state, original_config
            original_state.restore(source)
        Path(path).write_text(
            json.dumps(
                dict(
                    rows=rows,
                    costs=costs,
                    position=position,
                    inputs=inputs,
                    committed_gdn_and_attention_unchanged=True,
                    same_input_native_before_after_equal=True,
                    same_input_target_logits_bitwise_equal=True,
                    observed_anchor=observed_anchor,
                    ar_anchor_equal=observed_anchor == inputs[0],
                    semantics="teacher-forced inputs; windows <=5; carry between calls",
                ),
                indent=2,
            )
        )

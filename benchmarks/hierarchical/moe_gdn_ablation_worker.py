# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired five-token verification on one canonical prefix and MTP window."""

import copy
import json
from pathlib import Path

import torch
from batch_worker import BatchWorker
from windowed_cost import measure

from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class MoeGdnAblationWorker(BatchWorker):
    def initialize_ablation(self):
        spec = self.model_runner.speculator
        assert all(layer.enable_mean_preverify for layer in spec.state.layers.values())
        spec.state = PreverifyState(spec.model, 5, spec.device, "none")
        spec.config.preverify_gdn_mode = "none"
        spec.config.preverify_gdn_update_policy = "exact"
        spec.preverify_graphs = {}

    def prepare_ablation(self, path, reverse=False):
        self.ablation_pending = (path, reverse)
        if getattr(self, "ablation_installed", False):
            return
        self.ablation_installed = True
        spec = self.model_runner.speculator
        original = spec._verify

        def verify(batch, metadata, slots):
            if self.ablation_pending is not None:
                pending, self.ablation_pending = self.ablation_pending, None
                tokens, position = batch.input_ids.clone(), int(batch.positions[0])
                self._measure_ablation(batch, *pending)
                batch, metadata, slots = spec._batch(batch, position, tokens)
            return original(batch, metadata, slots)

        spec._verify = verify

    @torch.inference_mode()
    def _measure_ablation(self, template, path, reverse):
        spec = self.model_runner.speculator
        original_state, original_config = spec.state, spec.config
        assert original_state.mode == "none" and template.num_tokens == 5
        source = original_state.snapshot()
        canonical = self._committed_batch(template)
        tokens, position = template.input_ids.clone(), int(template.positions[0])
        cells = [(moe, v) for moe in ("h8", "h4", "p0125") for v in ("v0", "v2", "v3")]
        if reverse:
            cells.reverse()
        rows = []
        references = []
        path = Path(path)
        try:
            for moe, variant in [("h8", "v0"), *cells, ("h8", "v0")]:
                h = 4 if moe == "h4" else 8
                reference_only = len(references) == 0 or len(rows) == len(cells)
                mode = "none" if variant == "v0" else "replay_tail"
                policy = "exact" if variant == "v0" else "windowed_three_level"
                state = PreverifyState(
                    spec.model,
                    5,
                    spec.device,
                    mode,
                    policy,
                    window_size=1 if variant == "v2" else 5,
                )
                destination = 1 if mode == "none" else 0
                for name, (conv, ssm) in state.caches.items():
                    state._copy_conv(
                        conv[destination : destination + 1], source[name][0][1:2], 0
                    )
                    ssm[destination : destination + 1].copy_(source[name][1][1:2])
                state.initialized, state.request_id = True, template.req_ids[0]
                state.valid.fill_(1)
                spec.state, spec.config = state, copy.copy(original_config)
                spec.config.preverify_gdn_mode = mode
                spec.config.preverify_gdn_update_policy = policy
                spec.config.moe_skip_top_h = h
                spec.config.moe_skip_min_weight = 0.125 if moe == "p0125" else None
                batch, metadata, slots = spec._batch(template, position, tokens)
                initial = state.snapshot()
                predictions = spec._verify_eager(batch, metadata, slots)[0].clone()
                if reference_only:
                    references.append(spec.last_logits.cpu().clone())
                else:
                    state.restore(initial)
                    timing = measure(
                        spec,
                        batch,
                        metadata,
                        slots,
                        path.with_name(f"{path.stem}_{moe}_{variant}_profile.json"),
                    )
                    events = json.loads(Path(timing["profile"]).read_text())[
                        "traceEvents"
                    ]
                    kernel_names = [
                        e["name"] for e in events if e.get("cat") == "kernel"
                    ]
                    windowed_launches = sum(
                        "_windowed_update" in n for n in kernel_names
                    )
                    assert windowed_launches == (30 if state.windowed else 0)
                    actions = None
                    if state.windowed:
                        state.action_counts = torch.zeros(
                            8, dtype=torch.int64, device=spec.device
                        )
                    expert_counts = []
                    routers = [
                        m.router
                        for m in spec.model.modules()
                        if hasattr(getattr(m, "router", None), "set_capture_fn")
                    ]
                    callbacks = [r.capture_fn for r in routers]

                    def capture(ids, counts=expert_counts):
                        counts.extend((ids >= 0).sum(-1).cpu().tolist())

                    try:
                        for router in routers:
                            router.set_capture_fn(capture)
                        state.restore(initial)
                        audited = spec._verify_eager(batch, metadata, slots)[0]
                        assert torch.equal(audited, predictions)
                    finally:
                        for router, callback in zip(routers, callbacks, strict=True):
                            router.set_capture_fn(callback)
                    assert len(expert_counts) == 40 * 5, len(expert_counts)
                    if state.windowed:
                        actions = state.action_counts.tolist()
                        assert sum(actions[:3]) > 0
                    rows.append(
                        dict(
                            top_h=h,
                            moe=moe,
                            min_weight=spec.config.moe_skip_min_weight,
                            gdn=variant,
                            action_counts=actions,
                            windowed_kernel_launches=windowed_launches,
                            selected_experts_per_token_layer=expert_counts,
                            predictions=predictions.tolist(),
                            accepted=int(
                                tokens[1:].eq(predictions[:-1]).int().cumprod(0).sum()
                            ),
                            **timing,
                        )
                    )
                current = self._committed_batch(template)
                for name in canonical:
                    for before, after in zip(
                        canonical[name], current[name], strict=True
                    ):
                        assert torch.equal(before, after), name
            assert torch.equal(references[0], references[1])
            target = references[0].argmax(-1).tolist()
            candidates = tokens[1:].tolist()
            target_accepted = next(
                (i for i, (a, b) in enumerate(zip(candidates, target[:4])) if a != b),
                4,
            )
            for row in rows:
                matches = [
                    a == b for a, b in zip(row["predictions"], target, strict=True)
                ]
                row["target_agreement"] = sum(matches)
                row["target_prefix_agreement"] = next(
                    (i for i, equal in enumerate(matches[:4]) if not equal), 4
                )
                row["accepted_beyond_target"] = max(
                    0, row["accepted"] - target_accepted
                )
                row["rejected_target_accepted"] = max(
                    0, target_accepted - row["accepted"]
                )
            path.write_text(
                json.dumps(
                    dict(
                        position=position,
                        input_ids=tokens.tolist(),
                        target_predictions=target,
                        target_accepted=target_accepted,
                        rows=rows,
                        committed_prefix_bitwise=True,
                        target_repeat_bitwise=True,
                        timing_boundary=(
                            "five-token full model forward plus logits and selection"
                        ),
                    ),
                    indent=2,
                )
            )
        finally:
            spec.state, spec.config = original_state, original_config
            original_state.restore(source)
            spec._batch(template, position, tokens)

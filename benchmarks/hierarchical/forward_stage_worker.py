# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired fixed-input graph replay diagnostics on private preverify state."""

import copy
import os
from pathlib import Path

import torch


class ForwardStageWorker:
    def begin_forward_stages(self):
        self._stage_rows = []
        spec = self.model_runner.speculator
        if not hasattr(self, "_stage_original"):
            self._stage_original = spec._verify

            def verify(batch, metadata, slots):
                if self._stage_enabled and not self._stage_rows:
                    self._measure_pair(spec, batch)
                return self._stage_original(batch, metadata, slots)

            spec._verify = verify
        self._stage_enabled = True

    def collect_forward_stages(self):
        self._stage_enabled = False
        return self._stage_rows

    def _instrument(self, spec, spans, annotations=False):
        undo = []
        seen = set()

        def wrap(obj, method, phase, name):
            key = (id(obj), method)
            if key in seen:
                return
            seen.add(key)
            original = getattr(obj, method)

            def forward(*args, **kwargs):
                if annotations:
                    with torch.profiler.record_function("stage/" + phase + "/" + name):
                        return original(*args, **kwargs)
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                spans.append((phase, name, start, end))
                start.record()
                result = original(*args, **kwargs)
                end.record()
                return result

            setattr(obj, method, forward)
            undo.append((obj, method, original))

        from vllm.model_executor.layers.fused_moe import MoERunner

        for name, module in spec.model.named_modules():
            cls = type(module).__name__
            leaf = name.rsplit(".", 1)[-1]
            if leaf == "self_attn":
                wrap(module, "forward", "attention", name)
            elif leaf == "linear_attn":
                wrap(module, "forward", "gdn", name)
            elif "Norm" in cls:
                wrap(module, "forward", "norm", name)
            elif cls == "Gemma4MLP":
                wrap(module, "forward", "dense_mlp", name)
            elif leaf in ("embed_tokens", "embed_tokens_per_layer"):
                wrap(module, "forward", "embedding", name)
            elif cls == "Gemma4Router":
                wrap(module, "forward", "routing", name)
            if isinstance(module, MoERunner):
                wrap(module, "forward", "moe_envelope", name)
                wrap(module.router, "select_experts", "routing", name + ".topk")
                wrap(module.routed_experts, "forward_modular", "routed", name)
                if module.gate is not None:
                    wrap(module.gate, "forward", "routing", name + ".gate")
                if module._shared_experts is not None:
                    wrap(
                        module._shared_experts._layer,
                        "forward",
                        "shared",
                        name + ".shared",
                    )
        wrap(spec.logits_model, "compute_logits", "lm_head", "compute_logits")
        return undo

    def _kernel_trace(self, spec, batch, metadata, slots, state, graphs, top_h):
        root = Path(os.environ["FORWARD_STAGE_OUTPUT"])
        prefix = f"w{batch.num_tokens}_h{top_h}"
        spec.state.restore(state)
        undo = self._instrument(spec, [], annotations=True)
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                spec._verify_eager(batch, metadata, slots)
                torch.accelerator.synchronize()
        finally:
            for obj, method, original in reversed(undo):
                setattr(obj, method, original)
        eager_path = root / f"{prefix}_eager_trace.json.gz"
        profile.export_chrome_trace(str(eager_path))
        spec.state.restore(state)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            graphs[top_h, False][0].replay()
            torch.accelerator.synchronize()
        graph_path = root / f"{prefix}_graph_trace.json.gz"
        profile.export_chrome_trace(str(graph_path))

    @torch.inference_mode()
    def _measure_pair(self, spec, original_batch):
        tokens = original_batch.input_ids.clone()
        position = int(original_batch.positions[0].item())
        state = spec.state.snapshot()
        config = spec.config
        modes = (
            (False,)
            if os.environ.get("FORWARD_STAGE_COMPILED") == "1"
            else (False, True)
        )
        try:
            for width in (4, 5):
                assert tokens.numel() >= width
                batch, metadata, slots = spec._batch(
                    original_batch, position, tokens[:width]
                )
                graphs = {}
                for top_h in (8, 4):
                    spec.config = copy.copy(config)
                    spec.config.moe_skip_top_h = top_h
                    for _ in range(3):
                        spec.state.restore(state)
                        spec._verify_eager(batch, metadata, slots)
                    torch.accelerator.synchronize()
                    for detailed in modes:
                        spec.state.restore(state)
                        spans = []
                        undo = self._instrument(spec, spans) if detailed else []
                        graph = torch.cuda.CUDAGraph()
                        try:
                            with torch.cuda.graph(graph):
                                output = spec._verify_eager(batch, metadata, slots)
                        finally:
                            for obj, method, original in reversed(undo):
                                setattr(obj, method, original)
                        graphs[top_h, detailed] = (graph, output, spans)
                reference = {}
                for repeat in range(23):
                    order = (8, 4) if repeat % 2 == 0 else (4, 8)
                    for top_h in order:
                        for detailed in modes:
                            graph, output, spans = graphs[top_h, detailed]
                            spec.state.restore(state)
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            graph.replay()
                            end.record()
                            end.synchronize()
                            predictions = output[0].tolist()
                            key = top_h
                            if key not in reference:
                                reference[key] = predictions
                            assert predictions == reference[key], (
                                "Repeated or instrumented graph changed predictions"
                            )
                            if repeat < 3:
                                continue
                            self._stage_rows.append(
                                {
                                    "width": width,
                                    "position": position,
                                    "input_ids": tokens[:width].tolist(),
                                    "top_h": top_h,
                                    "detailed": detailed,
                                    "repeat": repeat - 3,
                                    "total_ms": start.elapsed_time(end),
                                    "predictions": predictions,
                                    "spans": [
                                        {
                                            "phase": phase,
                                            "name": name,
                                            "start_ms": start.elapsed_time(a),
                                            "end_ms": start.elapsed_time(b),
                                        }
                                        for phase, name, a, b in spans
                                    ],
                                }
                            )
                if len(modes) == 2 and not getattr(self, "_kernel_traced", False):
                    for top_h in (8, 4):
                        spec.config = copy.copy(config)
                        spec.config.moe_skip_top_h = top_h
                        self._kernel_trace(
                            spec, batch, metadata, slots, state, graphs, top_h
                        )
                if len(modes) == 1 and not getattr(self, "_kernel_traced", False):
                    for top_h in (8, 4):
                        spec.state.restore(state)
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            graphs[top_h, False][0].replay()
                            torch.accelerator.synchronize()
                        path = Path(os.environ["FORWARD_STAGE_OUTPUT"])
                        profile.export_chrome_trace(
                            str(path / f"w{width}_h{top_h}_graph_trace.json.gz")
                        )
                del graphs
            self._kernel_traced = True
        finally:
            spec.config = config
            spec.state.restore(state)
            spec._batch(original_batch, position, tokens)

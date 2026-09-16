# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-prefix verification-width probe; diagnostics stay outside timing."""

import copy

import torch


class VerifyWidthWorker:
    def begin_width_probe(self, candidates):
        self._width_candidates = candidates
        self._width_rows = []
        spec = self.model_runner.speculator
        if not hasattr(self, "_width_original"):
            self._width_original = spec._verify

            def verify(batch, metadata, slots):
                if self._width_candidates is not None:
                    candidates = self._width_candidates
                    self._width_candidates = None
                    batch, metadata, slots = self._measure_widths(
                        spec, batch, candidates
                    )
                return self._width_original(batch, metadata, slots)

            spec._verify = verify

    def collect_width_probe(self):
        return self._width_rows

    @torch.inference_mode()
    def _measure_widths(self, spec, original, candidates):
        from flashinfer.testing import bench_gpu_time_with_cupti

        from vllm.model_executor.layers.fused_moe import MoERunner
        from vllm.v1.worker.gpu.spec_decode.hierarchical.batched import make_batch

        n = original.num_reqs
        assert n == len(candidates), (n, len(candidates))
        assert not spec.state.layers
        saved = original.input_ids.view(n, -1).clone()
        positions = original.positions.view(n, -1)[:, 0].tolist()
        tokens = torch.tensor(candidates, device=saved.device, dtype=saved.dtype)
        tokens[:, 0] = saved[:, 0]
        config = spec.config
        try:
            for width, top_h in ((5, 8), (31, 8), (11, 8), (21, 8), (5, 4)):
                batch, metadata, slots = make_batch(
                    spec, original, list(range(n)), positions, tokens[:, :width]
                )
                spec.config = copy.copy(config)
                spec.config.moe_skip_top_h = top_h
                routes = []
                undo = []
                for name, module in spec.model.named_modules():
                    if not isinstance(module, MoERunner):
                        continue
                    router = module.router
                    old = router.select_experts

                    def select(*args, _old=old, _name=name, _routes=routes, **kwargs):
                        weights, ids = _old(*args, **kwargs)
                        _routes.append((_name, ids.clone()))
                        return weights, ids

                    router.select_experts = select
                    undo.append((router, old))
                try:
                    eager = spec._verify_eager(batch, metadata, slots)[0].clone()
                finally:
                    for router, old in undo:
                        router.select_experts = old
                assert len(routes) == 30, len(routes)
                route_stats = [
                    {
                        "layer": name,
                        "unique_experts": ids.unique().numel(),
                        "assignments": ids.numel(),
                    }
                    for name, ids in routes
                ]
                for _ in range(3):
                    spec._verify_eager(batch, metadata, slots)
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = spec._verify_eager(batch, metadata, slots)
                graph.replay()
                torch.accelerator.synchronize()
                assert torch.equal(eager, output[0]), "eager/graph mismatch"
                times = bench_gpu_time_with_cupti(
                    graph.replay,
                    dry_run_iters=3,
                    repeat_iters=20,
                    cold_l2_cache=True,
                )
                assert torch.equal(eager, output[0]), "replay changed output"
                self._width_rows.append(
                    {
                        "batch": n,
                        "width": width,
                        "top_h": top_h,
                        "positions": positions,
                        "input_ids": tokens[:, :width].tolist(),
                        "milliseconds": times,
                        "routes": route_stats,
                        "exact_eager_graph": True,
                    }
                )
                print(f"WIDTH_COMPLETE B={n} W={width} h={top_h}", flush=True)
                del graph, output
        finally:
            spec.config = config
            restored = make_batch(spec, original, list(range(n)), positions, saved)
        return restored

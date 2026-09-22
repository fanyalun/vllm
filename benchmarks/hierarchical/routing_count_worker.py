# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only route counts, accumulated once per actual Pre-Verify call."""

from collections import Counter

import torch


def route_counts(native_ids, retained_ids, padding, num_experts):
    counts = []
    for ids in (native_ids, retained_ids):
        valid = (ids >= 0) & (ids < num_experts)
        if padding is not None:
            valid = valid & ~padding[:, None]
        hits = torch.zeros(num_experts, dtype=torch.int64, device=ids.device)
        hits.scatter_add_(
            0, ids.long().clamp(0, num_experts - 1).flatten(), valid.long().flatten()
        )
        counts.extend([(hits > 0).sum(), valid.sum()])
    return torch.stack(counts)


class RoutingCountWorker:
    def setup_routing_counts(self):
        from vllm.model_executor.layers.fused_moe.router import batch_expert_selection

        spec = self.model_runner.speculator
        if spec.config.preverify_gdn_mode != "none":
            raise ValueError("Routing probe currently requires exact GDN")
        if spec.config.moe_skip_batch_policy is None:
            raise ValueError("Routing probe requires a batch policy")
        if getattr(self, "_routing_installed", False):
            raise RuntimeError("Routing probe already installed")
        self._routing_installed = True
        self._routing_enabled = False
        self._routing_current = None
        self._routing_buffers = {}
        self._routing_calls = Counter()
        layers = spec.vllm_config.model_config.hf_text_config.num_hidden_layers
        self._routing_totals = torch.zeros(
            (layers, 4), dtype=torch.int64, device=spec.device
        )
        self._routing_by_requests = {}
        select = batch_expert_selection.select_batch_experts
        eager = spec._verify_eager
        verify = spec._verify

        def selected(weights, ids, logits, policy, is_padding=None):
            result = select(weights, ids, logits, policy, is_padding)
            if self._routing_current is not None:
                self._routing_current.append(
                    route_counts(ids, result[1], is_padding, logits.shape[1])
                )
            return result

        def measured_eager(batch, metadata, slots):
            self._routing_current = []
            try:
                output = eager(batch, metadata, slots)
                assert len(self._routing_current) == layers
                self._routing_buffers[(batch.num_reqs, batch.num_tokens)] = (
                    self._routing_current
                )
                return output
            finally:
                self._routing_current = None

        def measured_verify(batch, metadata, slots):
            output = verify(batch, metadata, slots)
            if self._routing_enabled:
                key = (batch.num_reqs, batch.num_tokens)
                counts = torch.stack(self._routing_buffers[key])
                self._routing_totals.add_(counts)
                if batch.num_reqs not in self._routing_by_requests:
                    self._routing_by_requests[batch.num_reqs] = torch.zeros_like(
                        self._routing_totals
                    )
                self._routing_by_requests[batch.num_reqs].add_(counts)
                self._routing_calls[key] += 1
            return output

        # Existing graphs predate instrumentation; rebuild only private verify graphs.
        spec.preverify_graphs.clear()
        batch_expert_selection.select_batch_experts = selected
        spec._verify_eager = measured_eager
        spec._verify = measured_verify

    def begin_routing_counts(self):
        self._routing_totals.zero_()
        self._routing_calls.clear()
        self._routing_by_requests.clear()
        self._routing_enabled = True

    def collect_routing_counts(self):
        self._routing_enabled = False
        layers = self._routing_totals.cpu().tolist()
        native, connections, retained, kept_connections = map(sum, zip(*layers))
        assert 0 < retained <= native
        assert 0 < kept_connections <= connections
        return dict(
            native_expert_invocations=native,
            retained_expert_invocations=retained,
            native_connections=connections,
            retained_connections=kept_connections,
            expert_skip_fraction=1 - retained / native,
            connection_skip_fraction=1 - kept_connections / connections,
            layer_columns=[
                "native_experts",
                "native_connections",
                "retained_experts",
                "retained_connections",
            ],
            per_layer=layers,
            by_active_requests={
                str(n): value.cpu().tolist()
                for n, value in self._routing_by_requests.items()
            },
            preverify_calls=[
                dict(active_requests=n, token_rows=m, calls=count)
                for (n, m), count in sorted(self._routing_calls.items())
            ],
        )

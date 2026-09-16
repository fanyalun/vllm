# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only preservation of native top-k weights during MoE-Skip."""

import os

import torch


def install():
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
        fused_topk,
    )

    def patch(cls):
        original = cls._compute_routing

        def routing(
            self, hidden_states, router_logits, indices_type, *, input_ids=None
        ):
            h = self.get_routing_top_k()
            if h == self.top_k:
                return original(
                    self,
                    hidden_states,
                    router_logits,
                    indices_type,
                    input_ids=input_ids,
                )
            if isinstance(self, CustomRoutingRouter):
                weights, ids = self.custom_routing_function(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=self.top_k,
                    renormalize=self.renormalize,
                )
            else:
                weights, ids, _ = fused_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=self.top_k,
                    renormalize=self.renormalize,
                    indices_type=indices_type,
                    scoring_func=self.scoring_func,
                )
            scores = router_logits.gather(1, ids.long())
            selected = scores.topk(h, dim=-1).indices
            return (
                weights.gather(1, selected).float().contiguous(),
                ids.gather(1, selected)
                .to(torch.int32 if indices_type is None else indices_type)
                .contiguous(),
            )

        cls._compute_routing = routing

    for cls in (FusedTopKRouter, CustomRoutingRouter):
        patch(cls)


class WeightAblationWorker:
    pass


if os.environ.get("MOE_SKIP_WEIGHT_MODE") == "preserve":
    install()

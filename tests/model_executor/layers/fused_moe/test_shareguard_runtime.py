# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.shareguard_runtime import (
    apply_shareguard_to_topk,
    expert_to_rank,
    get_shareguard_stats,
    init_shareguard_from_env,
    maybe_apply_compensation,
    reset_shareguard_stats,
    shareguard_select_and_drop,
)


def _configure_shareguard(monkeypatch, tmp_path, *, capacity: float = 0.5):
    table_path = tmp_path / "rho_eps.pt"
    torch.save(
        {
            "rho_by_expert": torch.full((1, 5), 0.5),
            "eps_by_expert": torch.tensor([[1.0, 4.0, 1.0, 4.0, 1.0]]),
        },
        table_path,
    )
    monkeypatch.setenv("SHAREGUARD_ENABLE", "1")
    monkeypatch.setenv("SHAREGUARD_MODE", "shareguard")
    monkeypatch.setenv("SHAREGUARD_CAPACITY", str(capacity))
    monkeypatch.setenv("SHAREGUARD_RHO_PATH", str(table_path))
    monkeypatch.setenv("SHAREGUARD_COMPENSATE", "1")
    return init_shareguard_from_env()


def test_expert_to_rank_handles_uneven_linear_placement():
    expert_ids = torch.arange(5)
    ranks = expert_to_rank(expert_ids, ep_size=2, num_experts=5)
    assert ranks.tolist() == [0, 0, 0, 1, 1]


def test_select_and_drop_enforces_capacity_with_compensation():
    ids = torch.tensor([[0, 1], [2, 3]])
    weights = torch.tensor([[0.1, 0.8], [0.2, 0.7]])
    new_ids, new_weights, compensation, info = shareguard_select_and_drop(
        ids,
        weights,
        ep_size=2,
        num_experts=4,
        mode="shareguard",
        capacity=0.5,
        eps_row=torch.tensor([1.0, 4.0, 1.0, 4.0]),
        rho_row=torch.full((4,), 0.5),
    )

    assert info == {
        "dropped": 2.0,
        "cap": 1.0,
        "max_before": 2.0,
        "max_after": 1.0,
        "total": 4.0,
    }
    assert new_ids.tolist() == [[-1, 1], [-1, 3]]
    torch.testing.assert_close(
        new_weights, torch.tensor([[0.0, 0.8], [0.0, 0.7]])
    )
    assert compensation.tolist() == pytest.approx([0.05, 0.1])


def test_apply_rejects_non_linear_expert_placement(monkeypatch, tmp_path):
    _configure_shareguard(monkeypatch, tmp_path)
    layer = SimpleNamespace(
        ep_size=2,
        global_num_experts=5,
        expert_placement_strategy="round_robin",
    )
    with pytest.raises(RuntimeError, match="linear expert placement"):
        apply_shareguard_to_topk(
            layer, torch.tensor([[0, 3]]), torch.tensor([[0.2, 0.8]])
        )


def test_moe_runner_is_unchanged_when_shareguard_is_disabled(monkeypatch):
    monkeypatch.setenv("SHAREGUARD_ENABLE", "0")
    monkeypatch.setenv("SHAREGUARD_MODE", "off")
    init_shareguard_from_env()
    topk_ids = torch.tensor([[0, 1]])
    topk_weights = torch.tensor([[0.25, 0.75]])

    class Router:
        def select_experts(self, **_kwargs):
            return topk_weights, topk_ids

    class QuantMethod:
        is_monolithic = False

        def apply(self, **kwargs):
            self.topk_ids = kwargs["topk_ids"]
            self.topk_weights = kwargs["topk_weights"]
            return kwargs["x"]

    quant_method = QuantMethod()
    shared_experts = SimpleNamespace(output=torch.ones(1, 3))
    runner = SimpleNamespace(
        _quant_method=quant_method,
        _shared_experts=shared_experts,
        router=Router(),
        _maybe_apply_shared_experts=lambda *_args: None,
    )
    shared_output, _ = MoERunner._apply_quant_method(
        runner,
        SimpleNamespace(),
        hidden_states=torch.ones(1, 3),
        router_logits=torch.ones(1, 2),
        shared_experts_input=torch.ones(1, 3),
    )

    assert quant_method.topk_ids is topk_ids
    assert quant_method.topk_weights is topk_weights
    assert shared_output is shared_experts.output


def test_moe_runner_applies_drop_and_consumes_compensation(monkeypatch, tmp_path):
    _configure_shareguard(monkeypatch, tmp_path)
    reset_shareguard_stats()

    topk_ids = torch.tensor([[0, 1], [3, 4]])
    topk_weights = torch.tensor([[0.1, 0.8], [0.7, 0.2]])

    class Router:
        def select_experts(self, **_kwargs):
            return topk_weights, topk_ids

    class QuantMethod:
        is_monolithic = False

        def apply(self, **kwargs):
            self.topk_ids = kwargs["topk_ids"]
            self.topk_weights = kwargs["topk_weights"]
            return kwargs["x"]

    quant_method = QuantMethod()
    shared_experts = SimpleNamespace(output=torch.ones(2, 3))
    runner = SimpleNamespace(
        _quant_method=quant_method,
        _shared_experts=shared_experts,
        router=Router(),
        _maybe_apply_shared_experts=lambda *_args: None,
    )
    layer = SimpleNamespace(
        ep_size=2,
        global_num_experts=5,
        expert_placement_strategy="linear",
    )

    shared_output, fused_output = MoERunner._apply_quant_method(
        runner,
        layer,
        hidden_states=torch.ones(2, 3),
        router_logits=torch.ones(2, 5),
        shared_experts_input=torch.ones(2, 3),
    )

    assert int((quant_method.topk_ids == -1).sum()) == 2
    assert int((quant_method.topk_weights == 0).sum()) == 2
    assert not torch.equal(shared_output, shared_experts.output)
    assert torch.equal(fused_output, torch.ones(2, 3))
    assert get_shareguard_stats()["branches_dropped"] == 2
    assert maybe_apply_compensation(shared_output) is shared_output

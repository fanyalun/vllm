#!/usr/bin/env python3
"""
ShareGuard ablations on Qwen3.6-35B-A3B (HF transformers, 4x GPU via device_map).

Ablation 1: q = p^2 * eps factor contributions (M2 strategies).
Ablation 2: eps vs Top-k rank, grouped by domain (general/math/code).
Ablation 3: scalar rho vs low-rank projection residual (r=1,2,4,8).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Reuse data / helpers from run_m1_m2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m1_m2 import (  # noqa: E402
    GENERAL_PROMPTS,
    MomentAcc,
    M2Acc,
    _expert_unweighted,
    load_prompts,
)

DOMAINS = ("general", "math", "code")
DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAINS)}

# Ablation-1 strategies (plus baselines for plots)
ABLATION1_STRATEGIES = [
    "min_weight",          # select min p, no compensate
    "min_eps",             # select min eps, Shared Expert compensate
    "min_weight_comp",     # select min p, Shared Expert compensate
    "shareguard",          # select min p^2*eps, Shared Expert compensate
    "random",              # reference
]


# ---------------------------------------------------------------------------
# Ablation 2: domain-aware rank moments
# ---------------------------------------------------------------------------

@dataclass
class DomainRankAcc:
    """MomentAcc per domain × layer × topk-rank."""

    by_domain: list[MomentAcc]  # len=3, each shape [L, K]

    @staticmethod
    def create(num_layers: int, top_k: int) -> "DomainRankAcc":
        return DomainRankAcc(
            by_domain=[MomentAcc.create(num_layers, top_k) for _ in DOMAINS]
        )

    def add(self, domain: str, layer: int, rank: int, E: torch.Tensor, S: torch.Tensor) -> None:
        di = DOMAIN_TO_IDX[domain]
        self.by_domain[di].add((layer, rank), E, S)


# ---------------------------------------------------------------------------
# Ablation 3: reservoir of (E,S) by top-k rank for low-rank fit
# ---------------------------------------------------------------------------

@dataclass
class RankReservoir:
    """Keep up to max_n (E,S) pairs per top-k rank (CPU float16 to save RAM)."""

    top_k: int
    max_n: int
    hidden_dim: int
    E: list[torch.Tensor] = field(default_factory=list)
    S: list[torch.Tensor] = field(default_factory=list)
    filled: list[int] = field(default_factory=list)
    seen: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.E:
            self.E = [
                torch.zeros(self.max_n, self.hidden_dim, dtype=torch.float16)
                for _ in range(self.top_k)
            ]
            self.S = [
                torch.zeros(self.max_n, self.hidden_dim, dtype=torch.float16)
                for _ in range(self.top_k)
            ]
            self.filled = [0] * self.top_k
            self.seen = [0] * self.top_k

    def add(self, rank: int, E: torch.Tensor, S: torch.Tensor) -> None:
        if E.numel() == 0:
            return
        Ef = E.detach().float().reshape(-1, self.hidden_dim).cpu()
        Sf = S.detach().float().reshape(-1, self.hidden_dim).cpu()
        n = Ef.shape[0]
        for i in range(n):
            self.seen[rank] += 1
            if self.filled[rank] < self.max_n:
                idx = self.filled[rank]
                self.E[rank][idx] = Ef[i].half()
                self.S[rank][idx] = Sf[i].half()
                self.filled[rank] += 1
            else:
                # reservoir sampling
                j = torch.randint(0, self.seen[rank], (1,)).item()
                if j < self.max_n:
                    self.E[rank][j] = Ef[i].half()
                    self.S[rank][j] = Sf[i].half()


def fit_scalar_and_lowrank(
    E: torch.Tensor, S: torch.Tensor, ranks: list[int]
) -> dict[str, float]:
    """E,S: [N,D] float. Returns mean residual for scalar and each rank r."""
    Ef = E.float()
    Sf = S.float()
    n, d = Ef.shape
    out: dict[str, float] = {"n": float(n), "d": float(d)}
    ss = (Sf * Sf).sum().clamp(min=1e-12)
    rho = (Ef * Sf).sum() / ss
    out["scalar"] = float(((Ef - rho * Sf) ** 2).mean().item())
    out["rho"] = float(rho.item())

    # Unconstrained linear map: S @ M ≈ E, then SVD-truncate M to rank r
    # Use float32 for stability; subsample if N is huge
    max_fit = min(n, 4096)
    if n > max_fit:
        sel = torch.randperm(n)[:max_fit]
        Ef, Sf = Ef[sel], Sf[sel]
    try:
        # M: [D, D] minimizing ||Sf @ M - Ef||_F
        M = torch.linalg.lstsq(Sf, Ef).solution  # [D, D]
        U, sig, Vh = torch.linalg.svd(M, full_matrices=False)
        for r in ranks:
            r_use = min(r, int(sig.numel()))
            Mr = (U[:, :r_use] * sig[:r_use]) @ Vh[:r_use, :]
            out[f"rank_{r}"] = float(((Ef - Sf @ Mr) ** 2).mean().item())
    except Exception as exc:  # noqa: BLE001
        for r in ranks:
            out[f"rank_{r}"] = float("nan")
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

@dataclass
class AblationCollector:
    num_layers: int
    num_experts: int
    top_k: int
    hidden_dim: int
    drop_ratios: list[float]
    reservoir_n: int = 2048
    lowrank_ranks: list[int] = field(default_factory=lambda: [1, 2, 4, 8])

    domain: str = "general"
    active: bool = False
    enable_m1: bool = True
    enable_m2: bool = False
    m2_seed: int = 0

    by_rank: MomentAcc = field(init=False)
    by_expert: MomentAcc = field(init=False)
    by_domain_rank: DomainRankAcc = field(init=False)
    reservoir: RankReservoir = field(init=False)
    m2: M2Acc = field(init=False)
    epsilon_table: torch.Tensor | None = None
    rho_table: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.by_rank = MomentAcc.create(self.num_layers, self.top_k)
        self.by_expert = MomentAcc.create(self.num_layers, self.num_experts)
        self.by_domain_rank = DomainRankAcc.create(self.num_layers, self.top_k)
        self.reservoir = RankReservoir(self.top_k, self.reservoir_n, self.hidden_dim)
        self.m2 = M2Acc(drop_ratios=self.drop_ratios, strategies=ABLATION1_STRATEGIES)


def select_drop_indices(
    strategy: str,
    flat_p: torch.Tensor,
    flat_eps: torch.Tensor,
    n_drop: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    n_branches = flat_p.numel()
    if strategy == "random":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randperm(n_branches, generator=g, device=device)[:n_drop]
    if strategy in ("min_weight", "min_weight_comp"):
        return torch.topk(flat_p, n_drop, largest=False).indices
    if strategy == "min_eps":
        score = torch.where(torch.isfinite(flat_eps), flat_eps, torch.full_like(flat_eps, 1e9))
        return torch.topk(score, n_drop, largest=False).indices
    if strategy == "shareguard":
        q = (flat_p ** 2) * flat_eps
        score = torch.where(torch.isfinite(q), q, torch.full_like(q, 1e9))
        return torch.topk(score, n_drop, largest=False).indices
    raise ValueError(strategy)


def strategy_uses_compensation(strategy: str) -> bool:
    return strategy in ("min_eps", "min_weight_comp", "shareguard")


def make_patched_forward(layer_idx: int, collector: AblationCollector):
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        h = hidden_states.view(-1, hidden_dim)
        S_raw = self.shared_expert(h)
        _, routing_weights, selected_experts = self.gate(h)
        expert_output = self.experts(h, selected_experts, routing_weights)
        g_s = torch.sigmoid(self.shared_expert_gate(h))
        S_scaled = g_s * S_raw
        y = expert_output + S_scaled

        if collector.active and (collector.enable_m1 or collector.enable_m2):
            with torch.no_grad():
                T, K = selected_experts.shape
                expert_mask = F.one_hot(selected_experts, num_classes=collector.num_experts).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                E_ur = torch.zeros(T, K, hidden_dim, device=h.device, dtype=h.dtype)

                for expert_idx in expert_hit:
                    e = int(expert_idx[0].item())
                    if e >= collector.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[e])
                    E = _expert_unweighted(self.experts, h, e, token_idx)
                    E_ur[token_idx, top_k_pos] = E.to(E_ur.dtype)

                    if collector.enable_m1:
                        S_tok = S_raw[token_idx]
                        collector.by_expert.add((layer_idx, e), E, S_tok)
                        for r in top_k_pos.unique().tolist():
                            m = top_k_pos == r
                            Er, Sr = E[m], S_tok[m]
                            collector.by_rank.add((layer_idx, int(r)), Er, Sr)
                            collector.by_domain_rank.add(collector.domain, layer_idx, int(r), Er, Sr)
                            collector.reservoir.add(int(r), Er, Sr)

                if collector.enable_m2 and collector.epsilon_table is not None:
                    p = routing_weights.float()
                    eps_tbl = collector.epsilon_table[layer_idx].to(device=h.device, dtype=torch.float32)
                    rho_tbl = collector.rho_table[layer_idx].to(device=h.device, dtype=torch.float32)
                    eps_b = eps_tbl[selected_experts]
                    rho_b = rho_tbl[selected_experts]
                    n_branches = T * K
                    flat_p = p.reshape(-1)
                    flat_eps = eps_b.reshape(-1)

                    for ri, ratio in enumerate(collector.drop_ratios):
                        n_drop = int(math.floor(n_branches * ratio))
                        if n_drop <= 0:
                            continue
                        for strategy in ABLATION1_STRATEGIES:
                            drop_flat = select_drop_indices(
                                strategy,
                                flat_p,
                                flat_eps,
                                n_drop,
                                collector.m2_seed + layer_idx * 10007 + ri,
                                h.device,
                            )
                            keep = torch.ones(n_branches, dtype=torch.bool, device=h.device)
                            keep[drop_flat] = False
                            keep = keep.view(T, K)
                            drop_mask = ~keep
                            p_use = torch.where(keep, p, torch.zeros_like(p))
                            routed = (p_use.unsqueeze(-1) * E_ur.float()).sum(dim=1)
                            if strategy_uses_compensation(strategy):
                                comp = (p * drop_mask.float() * rho_b).sum(dim=-1, keepdim=True) * S_raw.float()
                                y_mod = routed + S_scaled.float() + comp
                            else:
                                y_mod = routed + S_scaled.float()
                            y_orig = (p.unsqueeze(-1) * E_ur.float()).sum(dim=1) + S_scaled.float()
                            collector.m2.add(strategy, ri, y_orig, y_mod)

        return y.reshape(batch_size, sequence_length, hidden_dim)

    return forward


def install_patches(model, collector: AblationCollector) -> list:
    language = model.model.language_model
    restored = []
    for i, layer in enumerate(language.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "shared_expert"):
            continue
        orig = mlp.forward
        mlp.forward = make_patched_forward(i, collector).__get__(mlp, type(mlp))
        restored.append((mlp, orig))
    return restored


def restore_patches(restored: list) -> None:
    for mlp, orig in restored:
        mlp.forward = orig


# ---------------------------------------------------------------------------
# Logit-level ablation-1
# ---------------------------------------------------------------------------

def make_apply_drop_forward(
    layer_idx: int,
    strategy: str,
    drop_ratio: float,
    rho_table: torch.Tensor,
    eps_table: torch.Tensor,
    seed: int,
):
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        h = hidden_states.view(-1, hidden_dim)
        S_raw = self.shared_expert(h)
        _, routing_weights, selected_experts = self.gate(h)
        g_s = torch.sigmoid(self.shared_expert_gate(h))
        S_scaled = g_s * S_raw

        T, K = selected_experts.shape
        n_branches = T * K
        n_drop = int(math.floor(n_branches * drop_ratio))

        expert_mask = F.one_hot(selected_experts, num_classes=self.experts.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        E_ur = torch.zeros(T, K, hidden_dim, device=h.device, dtype=h.dtype)
        for expert_idx in expert_hit:
            e = int(expert_idx[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[e])
            E_ur[token_idx, top_k_pos] = _expert_unweighted(self.experts, h, e, token_idx).to(E_ur.dtype)

        p = routing_weights.float()
        if n_drop <= 0:
            routed = (p.unsqueeze(-1) * E_ur.float()).sum(dim=1)
            return (routed + S_scaled.float()).reshape(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)

        eps_b = eps_table[layer_idx].to(h.device)[selected_experts]
        rho_b = rho_table[layer_idx].to(h.device)[selected_experts]
        drop_flat = select_drop_indices(
            strategy, p.reshape(-1), eps_b.reshape(-1), n_drop, seed + layer_idx, h.device
        )
        keep = torch.ones(n_branches, dtype=torch.bool, device=h.device)
        keep[drop_flat] = False
        keep = keep.view(T, K)
        drop_mask = ~keep
        p_use = torch.where(keep, p, torch.zeros_like(p))
        routed = (p_use.unsqueeze(-1) * E_ur.float()).sum(dim=1)
        if strategy_uses_compensation(strategy):
            comp = (p * drop_mask.float() * rho_b.float()).sum(dim=-1, keepdim=True) * S_raw.float()
            y = routed + S_scaled.float() + comp
        else:
            y = routed + S_scaled.float()
        return y.reshape(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)

    return forward


@torch.no_grad()
def run_logit_ablation1(
    model,
    tokenizer,
    prompts: list[str],
    rho_table: torch.Tensor,
    eps_table: torch.Tensor,
    drop_ratios: list[float],
    strategies: list[str],
    max_length: int,
) -> list[dict]:
    language = model.model.language_model
    moes = []
    for i, layer in enumerate(language.layers):
        if hasattr(layer.mlp, "shared_expert"):
            moes.append((i, layer.mlp, layer.mlp.forward))

    rows = []
    device = next(model.parameters()).device
    base_logits = []
    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        base_logits.append(out.logits[0, -1].float().cpu())

    for strategy in strategies:
        for ratio in drop_ratios:
            for i, mlp, _ in moes:
                mlp.forward = make_apply_drop_forward(
                    i, strategy, ratio, rho_table, eps_table, seed=123
                ).__get__(mlp, type(mlp))
            kls = []
            for text, base in zip(prompts, base_logits):
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits[0, -1].float().cpu()
                p = F.log_softmax(base, dim=-1)
                q = F.log_softmax(logits, dim=-1)
                kls.append(F.kl_div(q, p.exp(), reduction="sum").item())
            rows.append(
                {
                    "strategy": strategy,
                    "drop_ratio": ratio,
                    "logit_kl": sum(kls) / len(kls),
                    "n": len(kls),
                }
            )
            print(f"[A1-logit] {strategy} drop={ratio:.2f} KL={rows[-1]['logit_kl']:.6f}", flush=True)

    for _, mlp, orig in moes:
        mlp.forward = orig
    return rows


# ---------------------------------------------------------------------------
# Analysis / plots
# ---------------------------------------------------------------------------

def analyze_ablation2(collector: AblationCollector, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    summary: dict[str, Any] = {"domains": {}}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"general": "#2c7bb6", "math": "#d7191c", "code": "#1a9641"}
    xs = list(range(1, collector.top_k + 1))
    active_domains: list[str] = []

    for domain, acc in zip(DOMAINS, collector.by_domain_rank.by_domain):
        tok = float(acc.count.sum().item())
        if tok <= 0:
            summary["domains"][domain] = {
                "rank_eps_mean": [],
                "rank_eps_std": [],
                "monotonic_decreasing": None,
                "tokens_by_rank": acc.count.sum(dim=0).tolist(),
                "skipped": True,
            }
            continue
        active_domains.append(domain)
        _, eps_r = acc.rho_eps(collector.hidden_dim)
        means, stds = [], []
        for r in range(collector.top_k):
            vals = eps_r[:, r]
            vals = vals[torch.isfinite(vals)]
            means.append(float(vals.mean()) if len(vals) else float("nan"))
            stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
        mono = all(
            means[i] >= means[i + 1] - 1e-12
            for i in range(len(means) - 1)
            if math.isfinite(means[i]) and math.isfinite(means[i + 1])
        )
        summary["domains"][domain] = {
            "rank_eps_mean": means,
            "rank_eps_std": stds,
            "monotonic_decreasing": mono,
            "tokens_by_rank": acc.count.sum(dim=0).tolist(),
            "skipped": False,
        }
        ax.plot(xs, means, marker="o", color=colors[domain], label=f"{domain} (mono={mono})")
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=colors[domain],
            alpha=0.12,
        )

    ax.set_xlabel("Top-k rank (1 = highest router weight)")
    ax.set_ylabel("Residual ε")
    ax.set_title("Ablation 2: ε vs Top-k rank by domain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation2_eps_by_domain.png", dpi=150)
    plt.close(fig)

    checked = [
        summary["domains"][d]["monotonic_decreasing"]
        for d in active_domains
        if summary["domains"][d]["monotonic_decreasing"] is not None
    ]
    all_mono = bool(checked) and all(checked)
    summary["active_domains"] = active_domains
    summary["all_domains_monotonic"] = all_mono
    summary["go_nogo"] = "GO" if all_mono else "WEAK / CHECK"
    (out_dir / "ablation2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def analyze_ablation3(collector: AblationCollector, out_dir: Path) -> dict:
    import matplotlib.pyplot as plt

    ranks = collector.lowrank_ranks
    per_topk = []
    # Aggregate across top-k ranks: also compute global pool
    global_E, global_S = [], []
    for r in range(collector.top_k):
        n = collector.reservoir.filled[r]
        if n < 32:
            continue
        E = collector.reservoir.E[r][:n].float()
        S = collector.reservoir.S[r][:n].float()
        res = fit_scalar_and_lowrank(E, S, ranks)
        res["topk_rank"] = r + 1
        per_topk.append(res)
        global_E.append(E)
        global_S.append(S)
        print(f"[A3] topk_rank={r+1} n={n} scalar={res['scalar']:.6g} "
              + " ".join(f"r{rr}={res.get(f'rank_{rr}', float('nan')):.6g}" for rr in ranks), flush=True)

    pooled = {}
    if global_E:
        Ep = torch.cat(global_E, dim=0)
        Sp = torch.cat(global_S, dim=0)
        # subsample for fit cost
        if Ep.shape[0] > 8192:
            sel = torch.randperm(Ep.shape[0])[:8192]
            Ep, Sp = Ep[sel], Sp[sel]
        pooled = fit_scalar_and_lowrank(Ep, Sp, ranks)
        print(f"[A3] pooled n={int(pooled['n'])} scalar={pooled['scalar']:.6g} "
              + " ".join(f"r{rr}={pooled.get(f'rank_{rr}', float('nan')):.6g}" for rr in ranks), flush=True)

    summary = {"per_topk_rank": per_topk, "pooled": pooled, "ranks": ranks}
    # Relative reduction vs scalar
    if pooled and "scalar" in pooled and pooled["scalar"] > 0:
        summary["reduction_vs_scalar"] = {
            f"rank_{r}": (pooled["scalar"] - pooled.get(f"rank_{r}", float("nan"))) / pooled["scalar"]
            for r in ranks
        }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [0] + ranks  # 0 = scalar
    if pooled:
        ys = [pooled["scalar"]] + [pooled.get(f"rank_{r}", float("nan")) for r in ranks]
        ax.plot(xs, ys, "o-", color="#3b6ea5", linewidth=2, label="pooled tokens")
    ax.set_xticks(xs)
    ax.set_xticklabels(["scalar"] + [f"r={r}" for r in ranks])
    ax.set_xlabel("Projection type / rank")
    ax.set_ylabel("Residual ε (mean squared)")
    ax.set_title("Ablation 3: scalar vs low-rank projection residual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation3_rank_vs_eps.png", dpi=150)
    plt.close(fig)

    (out_dir / "ablation3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "reservoir_E": [collector.reservoir.E[r][: collector.reservoir.filled[r]] for r in range(collector.top_k)],
            "reservoir_S": [collector.reservoir.S[r][: collector.reservoir.filled[r]] for r in range(collector.top_k)],
            "filled": collector.reservoir.filled,
            "summary": summary,
        },
        out_dir / "ablation3_fit.pt",
    )
    return summary


def analyze_ablation1(collector: AblationCollector, out_dir: Path, logit_rows: list[dict] | None) -> dict:
    import matplotlib.pyplot as plt

    rows = collector.m2.summary()
    (out_dir / "ablation1_moe_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if logit_rows:
        (out_dir / "ablation1_logit_metrics.json").write_text(json.dumps(logit_rows, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    style = {
        "min_weight": ("#888888", "o"),
        "min_eps": ("#d7191c", "s"),
        "min_weight_comp": ("#fdae61", "^"),
        "shareguard": ("#2c7bb6", "D"),
        "random": ("#cccccc", "x"),
    }
    for s in ABLATION1_STRATEGIES:
        xs, ys = [], []
        for row in rows:
            if row["strategy"] == s:
                xs.append(row["drop_ratio"])
                ys.append(row["moe_output_mse"])
        c, m = style.get(s, ("#333", "o"))
        ax.plot(xs, ys, marker=m, color=c, label=s)
    ax.set_xlabel("Branch drop ratio")
    ax.set_ylabel("MoE output MSE")
    ax.set_title("Ablation 1: factor contributions (p vs ε vs joint)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation1_moe_mse.png", dpi=150)
    plt.close(fig)

    if logit_rows:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        by: dict[str, list] = {}
        for row in logit_rows:
            by.setdefault(row["strategy"], []).append(row)
        for s, rs in by.items():
            rs = sorted(rs, key=lambda x: x["drop_ratio"])
            c, m = style.get(s, ("#333", "o"))
            ax.plot([x["drop_ratio"] for x in rs], [x["logit_kl"] for x in rs], marker=m, color=c, label=s)
        ax.set_xlabel("Branch drop ratio")
        ax.set_ylabel("Logit KL")
        ax.set_title("Ablation 1: Logit KL — factor contributions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation1_logit_kl.png", dpi=150)
        plt.close(fig)

    target = 0.15

    def mse_at(strategy: str) -> float:
        cands = [r for r in rows if r["strategy"] == strategy and abs(r["drop_ratio"] - target) < 1e-9]
        return cands[0]["moe_output_mse"] if cands else float("nan")

    at = {s: mse_at(s) for s in ABLATION1_STRATEGIES}
    summary = {"at_drop_15pct_mse": at}
    # Expected ordering: shareguard < min_weight_comp < min_weight; shareguard < min_eps
    sg, mwc, mw, me = at["shareguard"], at["min_weight_comp"], at["min_weight"], at["min_eps"]
    ok = (
        math.isfinite(sg)
        and math.isfinite(mwc)
        and math.isfinite(mw)
        and math.isfinite(me)
        and sg <= mwc <= mw * 1.05  # allow tiny noise
        and sg <= me
        and mwc < mw
    )
    summary["expected_ordering_holds"] = bool(ok)
    summary["go_nogo"] = "GO" if ok else "CHECK"
    summary["note"] = (
        "Expect: shareguard < min_weight_comp < min_weight; "
        "shareguard < min_eps. Both factors needed."
    )
    if logit_rows:
        def kl_at(strategy: str) -> float:
            cands = [r for r in logit_rows if r["strategy"] == strategy and abs(r["drop_ratio"] - target) < 1e-9]
            return cands[0]["logit_kl"] if cands else float("nan")

        summary["at_drop_15pct_logit_kl"] = {s: kl_at(s) for s in ABLATION1_STRATEGIES}

    (out_dir / "ablation1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_m1_tables(collector: AblationCollector, out_dir: Path) -> dict:
    rho_r, eps_r = collector.by_rank.rho_eps(collector.hidden_dim)
    rho_e, eps_e = collector.by_expert.rho_eps(collector.hidden_dim)
    collector.rho_table = rho_e.nan_to_num(0.0)
    collector.epsilon_table = eps_e.nan_to_num(
        eps_e.nanmean().item() if torch.isfinite(eps_e.nanmean()) else 1.0
    )
    torch.save(
        {
            "rho_by_expert": rho_e,
            "eps_by_expert": eps_e,
            "rho_by_rank": rho_r,
            "eps_by_rank": eps_r,
            "count_by_expert": collector.by_expert.count,
            "count_by_rank": collector.by_rank.count,
            "domain_count_by_rank": {
                d: collector.by_domain_rank.by_domain[i].count for i, d in enumerate(DOMAINS)
            },
            "domain_eps_by_rank": {
                d: collector.by_domain_rank.by_domain[i].rho_eps(collector.hidden_dim)[1]
                for i, d in enumerate(DOMAINS)
            },
        },
        out_dir / "m1_rho_epsilon_with_domain.pt",
    )
    rank_eps_mean = []
    for r in range(collector.top_k):
        vals = eps_r[:, r]
        vals = vals[torch.isfinite(vals)]
        rank_eps_mean.append(float(vals.mean()) if len(vals) else float("nan"))
    return {"rank_eps_mean": rank_eps_mean, "eps_global_mean": float(eps_e[torch.isfinite(eps_e)].mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/model/Qwen3.6-35B-A3B")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/results/shareguard_ablations")
    ap.add_argument("--n-per-domain", type=int, default=64)
    ap.add_argument("--n-general", type=int, default=None, help="Override general-domain count (stabilize A2)")
    ap.add_argument("--n-math", type=int, default=None)
    ap.add_argument("--n-code", type=int, default=None)
    ap.add_argument(
        "--code-source",
        default="humaneval_mbpp",
        choices=["humaneval", "humaneval_mbpp", "mbpp"],
        help="Code-domain data source",
    )
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--drop-ratios", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--logit-eval-n", type=int, default=16)
    ap.add_argument("--reservoir-n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-logit", action="store_true")
    ap.add_argument("--rho-path", default="", help="Reuse existing m1_rho_epsilon.pt for A1 only")
    ap.add_argument("--only", default="all", choices=["all", "a1", "a2", "a3", "a2a3"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_ratios = [float(x) for x in args.drop_ratios.split(",")]
    t0 = time.time()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}", flush=True)
    prompts = load_prompts(
        args.n_per_domain,
        seed=args.seed,
        n_general=args.n_general,
        n_math=args.n_math,
        n_code=args.n_code,
        prefer_long_general=True,
        code_source=args.code_source,
    )
    (out_dir / "prompts_meta.json").write_text(
        json.dumps(
            {
                "n": len(prompts),
                "n_per_domain": args.n_per_domain,
                "n_general": args.n_general,
                "n_math": args.n_math,
                "n_code": args.n_code,
                "code_source": args.code_source,
                "domains": {d: sum(1 for x, _ in prompts if x == d) for d in DOMAINS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

    print(f"[model] loading {args.model} ...", flush=True)
    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[model] loaded in {time.time()-t1:.1f}s", flush=True)

    text_cfg = model.config.text_config
    collector = AblationCollector(
        num_layers=text_cfg.num_hidden_layers,
        num_experts=text_cfg.num_experts,
        top_k=text_cfg.num_experts_per_tok,
        hidden_dim=text_cfg.hidden_size,
        drop_ratios=drop_ratios,
        reservoir_n=args.reservoir_n,
        m2_seed=args.seed,
    )

    need_m1 = args.only in ("all", "a2", "a3", "a2a3") or not args.rho_path
    need_a1 = args.only in ("all", "a1")
    results: dict[str, Any] = {}

    restored = install_patches(model, collector)
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def embed_device():
        try:
            return model.model.language_model.embed_tokens.weight.device
        except Exception:
            return device0

    if need_m1:
        print("[M1/A2/A3] profiling with domain tags + reservoir ...", flush=True)
        collector.active = True
        collector.enable_m1 = True
        collector.enable_m2 = False
        with torch.no_grad():
            for i, (domain, text) in enumerate(prompts):
                collector.domain = domain
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                enc = {k: v.to(embed_device()) for k, v in enc.items()}
                _ = model(**enc)
                if (i + 1) % 16 == 0 or i == 0:
                    print(f"  [{i+1}/{len(prompts)}] domain={domain}", flush=True)

        m1_sum = save_m1_tables(collector, out_dir)
        results["m1"] = m1_sum
        if args.only in ("all", "a2", "a2a3"):
            results["ablation2"] = analyze_ablation2(collector, out_dir)
            print("[A2]", json.dumps(results["ablation2"], indent=2)[:800], flush=True)
        if args.only in ("all", "a3", "a2a3"):
            results["ablation3"] = analyze_ablation3(collector, out_dir)
            print("[A3]", json.dumps({k: results["ablation3"].get(k) for k in ("pooled", "reduction_vs_scalar", "ranks")}, indent=2), flush=True)
    else:
        # load rho/eps
        path = Path(args.rho_path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        collector.rho_table = data["rho_by_expert"].nan_to_num(0.0)
        eps = data["eps_by_expert"]
        collector.epsilon_table = eps.nan_to_num(eps.nanmean().item())
        print(f"[A1] loaded rho/eps from {path}", flush=True)

    if need_a1:
        if collector.rho_table is None:
            # ensure tables exist
            save_m1_tables(collector, out_dir)
        print("[A1] oracle MoE MSE with factor strategies ...", flush=True)
        collector.enable_m1 = False
        collector.enable_m2 = True
        collector.active = True
        collector.m2 = M2Acc(drop_ratios=drop_ratios, strategies=ABLATION1_STRATEGIES)
        m2_prompts = prompts[:: max(1, len(prompts) // min(len(prompts), args.n_per_domain))]
        m2_prompts = m2_prompts[: max(48, args.n_per_domain)]
        with torch.no_grad():
            for i, (domain, text) in enumerate(m2_prompts):
                collector.domain = domain
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                enc = {k: v.to(embed_device()) for k, v in enc.items()}
                _ = model(**enc)
                if (i + 1) % 16 == 0 or i == 0:
                    print(f"  [A1-mse {i+1}/{len(m2_prompts)}] {domain}", flush=True)

        logit_rows = None
        restore_patches(restored)
        if not args.skip_logit:
            print("[A1] logit KL ...", flush=True)
            logit_texts = [t for _, t in prompts[: args.logit_eval_n]]
            logit_rows = run_logit_ablation1(
                model,
                tokenizer,
                logit_texts,
                collector.rho_table,
                collector.epsilon_table,
                drop_ratios,
                ["min_weight", "min_eps", "min_weight_comp", "shareguard"],
                args.max_length,
            )
        results["ablation1"] = analyze_ablation1(collector, out_dir, logit_rows)
        print("[A1]", json.dumps(results["ablation1"], indent=2), flush=True)
    else:
        restore_patches(restored)

    overall = {
        "model": args.model,
        "n_per_domain": args.n_per_domain,
        "max_length": args.max_length,
        "drop_ratios": drop_ratios,
        "only": args.only,
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"[done] results in {out_dir}  elapsed={overall['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

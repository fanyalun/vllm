#!/usr/bin/env python3
"""
ShareGuard Motivation Experiments M1 + M2 on Qwen3.6-35B-A3B.

M1: Shared Expert vs routed-expert approximability (rho / epsilon by Top-k rank).
M2: Oracle quality comparison under branch dropping (ShareGuard vs baselines).

Uses HuggingFace transformers (no vLLM dispatch changes). Results written to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

GENERAL_PROMPTS_SHORT = [
    "Explain what cloud computing is in simple terms.",
    "What are three healthy breakfast ideas?",
    "Summarize the plot of Romeo and Juliet in two paragraphs.",
    "How does a refrigerator keep food cold?",
    "Write a polite email declining a meeting invitation.",
    "What is the difference between weather and climate?",
    "Give tips for improving focus while studying.",
    "Describe the water cycle for a middle-school student.",
    "What causes earthquakes?",
    "Compare trains and airplanes for long-distance travel.",
    "How should a beginner start learning a new language?",
    "Explain compound interest with a short example.",
    "What are the benefits of regular exercise?",
    "Write a short product description for wireless headphones.",
    "How do vaccines work at a high level?",
    "What is photosynthesis?",
    "Suggest weekend activities in a rainy city.",
    "Explain the difference between HTTP and HTTPS.",
    "How can teams run effective remote meetings?",
    "What is inflation and why does it matter?",
    "Describe how GPS navigation works.",
    "Give advice for first-time home cooks.",
    "What is machine learning used for in everyday apps?",
    "Explain supply and demand with groceries as an example.",
    "How do libraries help communities?",
    "Write a checklist for planning a short trip.",
    "What is recycling and why is it important?",
    "Explain the role of the United Nations briefly.",
    "How does sleep affect memory?",
    "Compare e-books and paper books.",
    "What makes a good password?",
    "Describe the stages of a software development lifecycle.",
    "How do birds migrate?",
    "Explain what a budget is and how to make one.",
    "What is the greenhouse effect?",
    "Give etiquette tips for dining at a restaurant.",
    "How do search engines rank web pages (high level)?",
    "What is the difference between a virus and bacteria?",
    "Explain democracy in one paragraph.",
    "How can someone reduce plastic waste at home?",
    "What is an API in software engineering?",
    "Describe how coffee is grown and processed.",
    "What are renewable energy sources?",
    "Explain why the sky appears blue.",
    "How do credit cards work?",
    "Give tips for writing a clear resume.",
    "What is biodiversity?",
    "Explain the difference between RAM and storage.",
    "How should you prepare for a job interview?",
    "What is the purpose of a thesis statement in an essay?",
]

# Longer general prompts: more tokens per sequence → stabler ε estimates.
GENERAL_PROMPTS_LONG = [
    "Write a detailed 3-paragraph explanation of how the internet works, covering packets, routers, and DNS, for a curious high-school student.",
    "Compare remote work and office work across productivity, collaboration, mental health, and career growth. Give concrete examples and a balanced conclusion.",
    "Explain the history and modern uses of vaccination in about 400 words, including how mRNA vaccines differ from traditional ones.",
    "Describe a week-long beginner fitness plan for someone who sits all day, including warm-ups, strength, cardio, recovery, and common mistakes to avoid.",
    "Write a thoughtful guide on personal budgeting for a recent graduate: income tracking, needs vs wants, emergency funds, and a sample monthly budget.",
    "Explain climate change causes and local mitigation actions a city can take, with sections on energy, transport, buildings, and community programs.",
    "Summarize how large language models are trained and used in everyday products, including benefits, risks, and practical tips for non-experts.",
    "Write an essay-style answer: what makes a good manager? Cover communication, feedback, hiring, conflict, and measuring team success with examples.",
    "Explain how elections work in a representative democracy, including campaigns, voting systems, media roles, and why civic participation matters.",
    "Describe the end-to-end process of planning an international trip: passports, budgeting, packing, jet lag, cultural etiquette, and emergency contingency.",
    "Write a long product review style article comparing electric cars and gasoline cars on cost, maintenance, range, environment, and charging infrastructure.",
    "Explain nutrition basics for busy adults: macronutrients, meal prep strategies, hydration, reading labels, and a 5-day sample meal outline.",
    "Describe how public libraries can support lifelong learning in the digital age, with programs for kids, seniors, job seekers, and small businesses.",
    "Write a detailed explanation of sleep science for lay readers: stages, circadian rhythm, caffeine, screens, and a practical wind-down routine.",
    "Compare three major renewable energy sources (solar, wind, hydro) on cost, reliability, land use, and suitability for different regions.",
    "Explain how credit scores are calculated and how a young adult can build credit responsibly over two years, with pitfalls to avoid.",
    "Write a comprehensive overview of cybersecurity hygiene for non-technical users: passwords, 2FA, phishing, updates, backups, and travel tips.",
    "Describe the water cycle and local watershed protection in depth, connecting weather, agriculture, pollution, and household conservation habits.",
    "Explain the basics of supply chains using groceries as a running example, from farm to warehouse to store shelf, including disruptions and resilience.",
    "Write a mentoring letter to a first-year college student on studying, time management, social life, mental health, and exploring career interests.",
    "Discuss the ethics of AI assistants in education and workplaces: cheating risks, accessibility benefits, disclosure norms, and policy recommendations.",
    "Explain how cities design public transit systems, covering buses, rail, last-mile access, fares, equity, and measuring ridership success.",
    "Write a long-form explainer on journalism literacy: source types, bias, verification, headlines vs evidence, and how to read scientific claims.",
    "Describe gardening for beginners in an apartment: soil, light, watering, pests, seasonal planning, and a starter plant list with care notes.",
    "Explain negotiation skills for salary and freelance rates with scripts, BATNA, common mistakes, and a post-offer checklist.",
    "Write about cultural exchange through food: pick three cuisines, describe signature dishes, techniques, and what they reveal about history and geography.",
    "Explain disaster preparedness for households: kits, communication plans, insurance, evacuation, and community mutual-aid considerations.",
    "Describe how universities support research commercialization, from labs to patents to startups, including funding stages and common failure modes.",
    "Write a detailed comparison of paper books, e-books, and audiobooks for learning and leisure, including memory, accessibility, and cost.",
    "Explain the greenhouse effect and urban heat islands, then propose a neighborhood-scale cooling plan with trees, materials, and water features.",
    "Write a practical guide to hosting inclusive meetings: agendas, facilitation, remote equity, decision logs, and follow-ups that actually happen.",
    "Explain personal data privacy for everyday apps: permissions, trackers, account deletion, encryption basics, and a monthly privacy checklist.",
    "Describe the craft of clear nonfiction writing: structure, evidence, voice, revision, and a step-by-step process for a 1000-word essay.",
    "Explain how vaccination campaigns succeed or fail socially, covering trust, logistics, misinformation, and communication strategies.",
    "Write an overview of sustainable fashion for consumers: materials, labor, repair, thrifting, and evaluating brand claims critically.",
    "Describe mentorship vs sponsorship in careers, with scenarios, how to ask for each, and how organizations can institutionalize both.",
    "Explain coastal erosion and sea-level rise impacts on a mid-sized city, including adaptation options and trade-offs for residents.",
    "Write a long FAQ-style article answering common misconceptions about nutrition, exercise, and weight, grounded in general scientific consensus.",
    "Explain how podcasts are produced end-to-end: ideation, scripting, recording, editing, hosting, marketing, and measuring audience engagement.",
    "Describe conflict resolution between roommates with mediation steps, agreements, and examples of fair chore and finance arrangements.",
]

GENERAL_PROMPTS = GENERAL_PROMPTS_SHORT + GENERAL_PROMPTS_LONG


def load_prompts(
    n_per_domain: int,
    seed: int = 42,
    n_general: int | None = None,
    n_math: int | None = None,
    n_code: int | None = None,
    prefer_long_general: bool = True,
    code_source: str = "humaneval_mbpp",
) -> list[tuple[str, str]]:
    """Return list of (domain, text). Offline-friendly.

    n_general/n_math/n_code override n_per_domain per domain when set.
    code_source: "humaneval" | "humaneval_mbpp" | "mbpp"
    """
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    ng = n_general if n_general is not None else n_per_domain
    nm = n_math if n_math is not None else n_per_domain
    nc = n_code if n_code is not None else n_per_domain

    # general — prefer long prompts for stabler ε; fall back to short pool
    if ng > 0:
        if prefer_long_general:
            general = list(GENERAL_PROMPTS_LONG) + list(GENERAL_PROMPTS_SHORT)
        else:
            general = list(GENERAL_PROMPTS)
        extras = []
        suffixes = [
            " Please answer in at least three paragraphs with concrete examples.",
            " Expand with background, a worked example, and practical takeaways.",
            " Cover motivations, mechanisms, trade-offs, and a short checklist.",
            " Include historical context, current practice, and common pitfalls.",
        ]
        i = 0
        while len(general) + len(extras) < ng:
            base = GENERAL_PROMPTS[i % len(GENERAL_PROMPTS)]
            extras.append(base + suffixes[(i // len(GENERAL_PROMPTS)) % len(suffixes)])
            i += 1
        general = general + extras
        rng.shuffle(general)
        for t in general[:ng]:
            out.append(("general", t))

    # math: gsm8k
    if nm > 0:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        idxs = list(range(len(gsm)))
        rng.shuffle(idxs)
        for i in idxs[:nm]:
            out.append(("math", gsm[i]["question"]))

    # code
    if nc > 0:
        code_texts: list[str] = []
        if code_source in ("humaneval", "humaneval_mbpp"):
            he = load_dataset("openai/openai_humaneval", split="test")
            code_texts.extend(he[i]["prompt"] for i in range(len(he)))
        if code_source in ("mbpp", "humaneval_mbpp"):
            try:
                mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
                for i in range(len(mbpp)):
                    code_texts.append(mbpp[i]["text"] + "\n" + mbpp[i].get("code", ""))
            except Exception:
                if code_source == "mbpp":
                    raise
        if not code_texts:
            raise ValueError(f"No code prompts for code_source={code_source}")
        # If requesting more than available, cycle (with optional completion hint)
        if len(code_texts) < nc:
            base = list(code_texts)
            i = 0
            while len(code_texts) < nc:
                code_texts.append(
                    base[i % len(base)]
                    + "\n# Write a complete, correct implementation with comments.\n"
                )
                i += 1
        rng.shuffle(code_texts)
        for t in code_texts[:nc]:
            out.append(("code", t))

    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

@dataclass
class MomentAcc:
    """Accumulate sum <E,S>, ||S||^2, ||E||^2, counts for closed-form rho/eps."""

    sum_dot: torch.Tensor  # float64 CPU
    sum_ss: torch.Tensor
    sum_ee: torch.Tensor
    count: torch.Tensor  # number of (token) samples

    @staticmethod
    def create(*shape: int) -> "MomentAcc":
        return MomentAcc(
            sum_dot=torch.zeros(*shape, dtype=torch.float64),
            sum_ss=torch.zeros(*shape, dtype=torch.float64),
            sum_ee=torch.zeros(*shape, dtype=torch.float64),
            count=torch.zeros(*shape, dtype=torch.float64),
        )

    def add(self, idx, E: torch.Tensor, S: torch.Tensor) -> None:
        """E,S: [N, D] float."""
        if E.numel() == 0:
            return
        Ef = E.detach().float().reshape(-1, E.shape[-1])
        Sf = S.detach().float().reshape(-1, S.shape[-1])
        n = Ef.shape[0]
        dot = (Ef * Sf).sum(dim=-1).sum().double().cpu()
        ss = (Sf * Sf).sum(dim=-1).sum().double().cpu()
        ee = (Ef * Ef).sum(dim=-1).sum().double().cpu()
        self.sum_dot[idx] += dot
        self.sum_ss[idx] += ss
        self.sum_ee[idx] += ee
        self.count[idx] += float(n)

    def rho_eps(self, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        ss = self.sum_ss.clamp(min=1e-12)
        rho = self.sum_dot / ss
        # mean over all elements (tokens * dims), matching doc ((E-rhoS)**2).mean()
        denom = (self.count * hidden_dim).clamp(min=1.0)
        eps = (self.sum_ee - 2.0 * rho * self.sum_dot + (rho ** 2) * self.sum_ss) / denom
        # where count==0
        eps = torch.where(self.count > 0, eps, torch.full_like(eps, float("nan")))
        rho = torch.where(self.count > 0, rho, torch.full_like(rho, float("nan")))
        return rho, eps


@dataclass
class M2Acc:
    """Accumulate MoE-output MSE / cosine under drop strategies."""

    drop_ratios: list[float]
    strategies: list[str]
    # key (strategy, ratio_idx) -> sum_mse, sum_cos, n_layers_tokens
    sum_mse: dict[tuple[str, int], float] = field(default_factory=dict)
    sum_cos: dict[tuple[str, int], float] = field(default_factory=dict)
    n: dict[tuple[str, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for s in self.strategies:
            for i, _ in enumerate(self.drop_ratios):
                self.sum_mse[(s, i)] = 0.0
                self.sum_cos[(s, i)] = 0.0
                self.n[(s, i)] = 0.0

    def add(self, strategy: str, ratio_idx: int, y: torch.Tensor, y_mod: torch.Tensor) -> None:
        yf = y.detach().float().reshape(-1, y.shape[-1])
        mf = y_mod.detach().float().reshape(-1, y_mod.shape[-1])
        mse = ((yf - mf) ** 2).mean().item()
        cos = F.cosine_similarity(yf, mf, dim=-1).mean().item()
        key = (strategy, ratio_idx)
        self.sum_mse[key] += mse
        self.sum_cos[key] += cos
        self.n[key] += 1.0

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for s in self.strategies:
            for i, r in enumerate(self.drop_ratios):
                key = (s, i)
                nn = max(self.n[key], 1.0)
                rows.append(
                    {
                        "strategy": s,
                        "drop_ratio": r,
                        "moe_output_mse": self.sum_mse[key] / nn,
                        "moe_output_cosine": self.sum_cos[key] / nn,
                        "n_updates": int(self.n[key]),
                    }
                )
        return rows


# ---------------------------------------------------------------------------
# MoE patch
# ---------------------------------------------------------------------------

STRATEGIES = [
    "random",
    "min_weight",
    "capacity_aware",  # same selection as min_weight; name kept for paper table
    "renorm",
    "shareguard",
]


def _expert_unweighted(experts, hidden: torch.Tensor, expert_idx: int, token_idx: torch.Tensor) -> torch.Tensor:
    current_state = hidden[token_idx]
    gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    h = experts.act_fn(gate) * up
    return F.linear(h, experts.down_proj[expert_idx])


@dataclass
class Collector:
    num_layers: int
    num_experts: int
    top_k: int
    hidden_dim: int
    drop_ratios: list[float]
    enable_m1: bool = True
    enable_m2: bool = True
    m2_seed: int = 0

    by_rank: MomentAcc = field(init=False)
    by_expert: MomentAcc = field(init=False)
    m2: M2Acc = field(init=False)
    epsilon_table: torch.Tensor | None = None  # [L, E] filled after M1
    rho_table: torch.Tensor | None = None
    layer_counter: int = 0
    active: bool = False

    def __post_init__(self) -> None:
        self.by_rank = MomentAcc.create(self.num_layers, self.top_k)
        self.by_expert = MomentAcc.create(self.num_layers, self.num_experts)
        self.m2 = M2Acc(drop_ratios=self.drop_ratios, strategies=STRATEGIES)

    def reset_layer_counter(self) -> None:
        self.layer_counter = 0


def make_patched_forward(orig_forward, layer_idx: int, collector: Collector):
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        h = hidden_states.view(-1, hidden_dim)
        S_raw = self.shared_expert(h)
        _, routing_weights, selected_experts = self.gate(h)  # [T,K], [T,K]
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
                            collector.by_rank.add((layer_idx, int(r)), E[m], S_tok[m])

                if collector.enable_m2 and collector.epsilon_table is not None:
                    # Branch list: all T*K branches
                    p = routing_weights.float()  # [T,K]
                    eps_tbl = collector.epsilon_table[layer_idx].to(device=h.device, dtype=torch.float32)
                    rho_tbl = collector.rho_table[layer_idx].to(device=h.device, dtype=torch.float32)
                    eps_b = eps_tbl[selected_experts]  # [T,K]
                    rho_b = rho_tbl[selected_experts]

                    n_branches = T * K
                    for ri, ratio in enumerate(collector.drop_ratios):
                        n_drop = int(math.floor(n_branches * ratio))
                        if n_drop <= 0:
                            continue
                        # scores flat
                        flat_p = p.reshape(-1)
                        flat_eps = eps_b.reshape(-1)
                        flat_q = (flat_p ** 2) * flat_eps

                        for strategy in STRATEGIES:
                            if strategy == "random":
                                g = torch.Generator(device=h.device)
                                g.manual_seed(collector.m2_seed + layer_idx * 10007 + ri)
                                perm = torch.randperm(n_branches, generator=g, device=h.device)
                                drop_flat = perm[:n_drop]
                            elif strategy in ("min_weight", "capacity_aware", "renorm"):
                                drop_flat = torch.topk(flat_p, n_drop, largest=False).indices
                            elif strategy == "shareguard":
                                # prefer finite eps; nan -> large
                                score = flat_q.clone()
                                score = torch.where(torch.isfinite(score), score, torch.full_like(score, 1e9))
                                drop_flat = torch.topk(score, n_drop, largest=False).indices
                            else:
                                continue

                            keep = torch.ones(n_branches, dtype=torch.bool, device=h.device)
                            keep[drop_flat] = False
                            keep = keep.view(T, K)
                            drop_mask = ~keep

                            # routed part: sum keep p * E
                            p_use = p.clone()
                            p_use = torch.where(keep, p_use, torch.zeros_like(p_use))
                            if strategy == "renorm":
                                denom = p_use.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                                p_use = p_use / denom

                            routed = (p_use.unsqueeze(-1) * E_ur.float()).sum(dim=1)  # [T,D]

                            if strategy == "shareguard":
                                # compensate: add sum_{dropped} p * rho * S_raw
                                comp = (p * drop_mask.float() * rho_b).sum(dim=-1, keepdim=True) * S_raw.float()
                                y_mod = routed + S_scaled.float() + comp
                            else:
                                y_mod = routed + S_scaled.float()

                            y_orig = (p.unsqueeze(-1) * E_ur.float()).sum(dim=1) + S_scaled.float()
                            collector.m2.add(strategy, ri, y_orig, y_mod)

        return y.reshape(batch_size, sequence_length, hidden_dim)

    return forward


def install_patches(model, collector: Collector) -> list:
    """Patch each MoE block; return list of (module, orig_forward) for restore."""
    language = model.model.language_model
    restored = []
    for i, layer in enumerate(language.layers):
        mlp = layer.mlp
        # Only SparseMoeBlock has shared_expert
        if not hasattr(mlp, "shared_expert"):
            continue
        orig = mlp.forward
        mlp.forward = make_patched_forward(orig, i, collector).__get__(mlp, type(mlp))
        restored.append((mlp, orig))
    return restored


def restore_patches(restored: list) -> None:
    for mlp, orig in restored:
        mlp.forward = orig


# ---------------------------------------------------------------------------
# Analysis / plots
# ---------------------------------------------------------------------------

def analyze_m1(collector: Collector, out_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rho_r, eps_r = collector.by_rank.rho_eps(collector.hidden_dim)
    rho_e, eps_e = collector.by_expert.rho_eps(collector.hidden_dim)
    collector.rho_table = rho_e.nan_to_num(0.0)
    collector.epsilon_table = eps_e.nan_to_num(eps_e.nanmean().item() if torch.isfinite(eps_e.nanmean()) else 1.0)

    # mean eps by rank across layers
    rank_eps_mean = []
    rank_eps_std = []
    for r in range(collector.top_k):
        vals = eps_r[:, r]
        vals = vals[torch.isfinite(vals)]
        rank_eps_mean.append(float(vals.mean()) if len(vals) else float("nan"))
        rank_eps_std.append(float(vals.std()) if len(vals) > 1 else 0.0)

    global_eps = eps_e[torch.isfinite(eps_e)]
    eps_bar = float(global_eps.mean()) if len(global_eps) else float("nan")
    # low-rank experts (ranks 5,6,7): fraction of expert-rank samples with eps < bar/2
    # Use by_rank counts: for ranks 5-7, fraction of (layer,rank) with eps < bar/2
    low_ranks = list(range(max(0, collector.top_k - 3), collector.top_k))
    low_vals = eps_r[:, low_ranks]
    low_vals = low_vals[torch.isfinite(low_vals)]
    frac_low = float((low_vals < (eps_bar / 2)).float().mean()) if len(low_vals) else float("nan")
    high_ranks = list(range(min(2, collector.top_k)))
    high_vals = eps_r[:, high_ranks]
    high_vals = high_vals[torch.isfinite(high_vals)]
    mean_low = float(low_vals.mean()) if len(low_vals) else float("nan")
    mean_high = float(high_vals.mean()) if len(high_vals) else float("nan")

    go = bool(
        math.isfinite(mean_low)
        and math.isfinite(mean_high)
        and mean_low < mean_high
        and math.isfinite(frac_low)
        and frac_low >= 0.20
    )

    summary = {
        "rank_eps_mean": rank_eps_mean,
        "rank_eps_std": rank_eps_std,
        "eps_global_mean": eps_bar,
        "eps_mean_top1_2": mean_high,
        "eps_mean_bottom3": mean_low,
        "frac_bottom3_eps_lt_half_mean": frac_low,
        "go_nogo": "GO" if go else "NO-GO / WEAK",
        "go_reason": (
            f"bottom3 eps {mean_low:.4g} < top1-2 eps {mean_high:.4g}; "
            f"frac(bottom3 < mean/2)={frac_low:.3f} (need >=0.20)"
        ),
        "num_experts_observed": int((collector.by_expert.count > 0).sum().item()),
        "tokens_by_rank": collector.by_rank.count.sum(dim=0).tolist(),
    }

    # save tables
    torch.save(
        {
            "rho_by_expert": rho_e,
            "eps_by_expert": eps_e,
            "rho_by_rank": rho_r,
            "eps_by_rank": eps_r,
            "count_by_expert": collector.by_expert.count,
            "count_by_rank": collector.by_rank.count,
        },
        out_dir / "m1_rho_epsilon.pt",
    )
    (out_dir / "m1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(1, collector.top_k + 1))
    ax.bar(xs, rank_eps_mean, yerr=rank_eps_std, capsize=3, color="#3b6ea5")
    ax.axhline(eps_bar, color="#c44", linestyle="--", label=f"global mean ε={eps_bar:.4g}")
    ax.axhline(eps_bar / 2, color="#888", linestyle=":", label="mean/2")
    ax.set_xlabel("Top-k rank (1 = highest router weight)")
    ax.set_ylabel("Residual ε (mean over layers)")
    ax.set_title("M1: Shared-Expert approximability by Top-k rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m1_eps_by_rank.png", dpi=150)
    plt.close(fig)

    # also box-ish line: per-layer scatter mean
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in range(collector.top_k):
        vals = eps_r[:, r]
        vals = vals[torch.isfinite(vals)].numpy()
        ax.scatter([r + 1] * len(vals), vals, s=8, alpha=0.35)
    ax.plot(xs, rank_eps_mean, "k-o", label="mean")
    ax.set_xlabel("Top-k rank")
    ax.set_ylabel("ε")
    ax.set_title("M1: per-layer ε by rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m1_eps_by_rank_scatter.png", dpi=150)
    plt.close(fig)

    return summary


def analyze_m2(collector: Collector, out_dir: Path, logit_rows: list[dict] | None = None) -> dict:
    import matplotlib.pyplot as plt

    rows = collector.m2.summary()
    (out_dir / "m2_moe_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if logit_rows:
        (out_dir / "m2_logit_metrics.json").write_text(json.dumps(logit_rows, indent=2), encoding="utf-8")

    # plot MSE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in STRATEGIES:
        xs, ys = [], []
        for row in rows:
            if row["strategy"] == s:
                xs.append(row["drop_ratio"])
                ys.append(row["moe_output_mse"])
        ax.plot(xs, ys, marker="o", label=s)
    ax.set_xlabel("Branch drop ratio")
    ax.set_ylabel("MoE output MSE (vs original)")
    ax.set_title("M2: Oracle drop strategies — MoE output MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "m2_moe_mse.png", dpi=150)
    plt.close(fig)

    # KL plot if available
    if logit_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        by = {}
        for row in logit_rows:
            by.setdefault(row["strategy"], []).append(row)
        for s, rs in by.items():
            rs = sorted(rs, key=lambda x: x["drop_ratio"])
            ax.plot([x["drop_ratio"] for x in rs], [x["logit_kl"] for x in rs], marker="o", label=s)
        ax.set_xlabel("Branch drop ratio")
        ax.set_ylabel("Logit KL (mean)")
        ax.set_title("M2: Logit KL under drop strategies")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "m2_logit_kl.png", dpi=150)
        plt.close(fig)

    # go/no-go at 15%
    target = 0.15
    def mse_at(strategy: str) -> float:
        cands = [r for r in rows if r["strategy"] == strategy and abs(r["drop_ratio"] - target) < 1e-9]
        return cands[0]["moe_output_mse"] if cands else float("nan")

    sg = mse_at("shareguard")
    base = mse_at("min_weight")
    reduction = (base - sg) / base if base and math.isfinite(base) and base > 0 else float("nan")
    summary = {
        "at_drop_15pct": {
            "shareguard_mse": sg,
            "min_weight_mse": base,
            "mse_reduction_vs_min_weight": reduction,
        },
        "go_nogo": "GO" if (math.isfinite(reduction) and reduction >= 0.30) else "NO-GO / WEAK",
        "note": "Primary M2 criterion in doc is Logit KL -30%; MoE MSE used as layer-oracle proxy; logit metrics in m2_logit_metrics.json if present.",
    }
    if logit_rows:
        def kl_at(strategy: str) -> float:
            cands = [r for r in logit_rows if r["strategy"] == strategy and abs(r["drop_ratio"] - target) < 1e-9]
            return cands[0]["logit_kl"] if cands else float("nan")

        sg_kl = kl_at("shareguard")
        base_kl = kl_at("min_weight")
        red_kl = (base_kl - sg_kl) / base_kl if base_kl and base_kl > 0 else float("nan")
        summary["at_drop_15pct"]["shareguard_logit_kl"] = sg_kl
        summary["at_drop_15pct"]["min_weight_logit_kl"] = base_kl
        summary["at_drop_15pct"]["kl_reduction_vs_min_weight"] = red_kl
        summary["go_nogo"] = "GO" if (math.isfinite(red_kl) and red_kl >= 0.30) else summary["go_nogo"]

    (out_dir / "m2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Logit-level M2 (second pass with actual modified MoE output)
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
            y = routed + S_scaled.float()
            return y.reshape(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)

        eps_b = eps_table[layer_idx].to(h.device)[selected_experts]
        rho_b = rho_table[layer_idx].to(h.device)[selected_experts]
        flat_p = p.reshape(-1)
        flat_q = (flat_p ** 2) * eps_b.reshape(-1)

        if strategy == "random":
            g = torch.Generator(device=h.device)
            g.manual_seed(seed + layer_idx)
            drop_flat = torch.randperm(n_branches, generator=g, device=h.device)[:n_drop]
        elif strategy in ("min_weight", "capacity_aware", "renorm"):
            drop_flat = torch.topk(flat_p, n_drop, largest=False).indices
        else:  # shareguard
            score = torch.where(torch.isfinite(flat_q), flat_q, torch.full_like(flat_q, 1e9))
            drop_flat = torch.topk(score, n_drop, largest=False).indices

        keep = torch.ones(n_branches, dtype=torch.bool, device=h.device)
        keep[drop_flat] = False
        keep = keep.view(T, K)
        drop_mask = ~keep
        p_use = torch.where(keep, p, torch.zeros_like(p))
        if strategy == "renorm":
            p_use = p_use / p_use.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        routed = (p_use.unsqueeze(-1) * E_ur.float()).sum(dim=1)
        if strategy == "shareguard":
            comp = (p * drop_mask.float() * rho_b.float()).sum(dim=-1, keepdim=True) * S_raw.float()
            y = routed + S_scaled.float() + comp
        else:
            y = routed + S_scaled.float()
        return y.reshape(batch_size, sequence_length, hidden_dim).to(hidden_states.dtype)

    return forward


@torch.no_grad()
def run_logit_m2(
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

    # baseline logits
    base_logits = []
    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        # ConditionalGeneration returns logits
        logits = out.logits[0, -1].float().cpu()
        base_logits.append(logits)

    for strategy in strategies:
        for ratio in drop_ratios:
            # install
            for i, mlp, _ in moes:
                mlp.forward = make_apply_drop_forward(
                    i, strategy, ratio, rho_table, eps_table, seed=123
                ).__get__(mlp, type(mlp))
            kls = []
            cos_h = []
            for text, base in zip(prompts, base_logits):
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits[0, -1].float().cpu()
                # KL(base || mod) on last-token softmax
                p = F.log_softmax(base, dim=-1)
                q = F.log_softmax(logits, dim=-1)
                kl = F.kl_div(q, p.exp(), reduction="sum").item()
                kls.append(kl)
                cos_h.append(F.cosine_similarity(base.unsqueeze(0), logits.unsqueeze(0)).item())
            rows.append(
                {
                    "strategy": strategy,
                    "drop_ratio": ratio,
                    "logit_kl": sum(kls) / len(kls),
                    "logit_cosine": sum(cos_h) / len(cos_h),
                    "n": len(kls),
                }
            )
            print(f"[M2-logit] {strategy} drop={ratio:.2f} KL={rows[-1]['logit_kl']:.4f}", flush=True)

    # restore
    for _, mlp, orig in moes:
        mlp.forward = orig
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/model/Qwen3.6-35B-A3B")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/results/shareguard_motivation")
    ap.add_argument("--n-per-domain", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--drop-ratios", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--logit-eval-n", type=int, default=24, help="prompts for full forward logit KL")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-m2-logit", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_ratios = [float(x) for x in args.drop_ratios.split(",")]

    print("[env] activating context under vllm-main/.venv", flush=True)
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}", flush=True)

    t0 = time.time()
    prompts = load_prompts(args.n_per_domain, seed=args.seed)
    (out_dir / "prompts_meta.json").write_text(
        json.dumps(
            {
                "n": len(prompts),
                "n_per_domain": args.n_per_domain,
                "domains": {d: sum(1 for x, _ in prompts if x == d) for d in ("general", "math", "code")},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[data] {len(prompts)} prompts loaded in {time.time()-t0:.1f}s", flush=True)

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
    collector = Collector(
        num_layers=text_cfg.num_hidden_layers,
        num_experts=text_cfg.num_experts,
        top_k=text_cfg.num_experts_per_tok,
        hidden_dim=text_cfg.hidden_size,
        drop_ratios=drop_ratios,
        enable_m1=True,
        enable_m2=False,  # first pass M1 only
        m2_seed=args.seed,
    )

    # Fix by_expert: MomentAcc.create(L,E) — add() must use (layer, e).
    # In make_patched_forward there was a buggy first add; we only use the corrected loop.

    restored = install_patches(model, collector)

    # ---- M1 pass ----
    print("[M1] profiling shared vs routed experts ...", flush=True)
    collector.active = True
    collector.enable_m1 = True
    collector.enable_m2 = False
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (domain, text) in enumerate(prompts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            # device_map=auto: inputs on first device of embed
            try:
                embed_dev = model.model.language_model.embed_tokens.weight.device
            except Exception:
                embed_dev = device0
            enc = {k: v.to(embed_dev) for k, v in enc.items()}
            _ = model(**enc)
            if (i + 1) % 16 == 0 or i == 0:
                print(f"  [{i+1}/{len(prompts)}] domain={domain} len={enc['input_ids'].shape[1]}", flush=True)

    m1_summary = analyze_m1(collector, out_dir)
    print("[M1] summary:", json.dumps(m1_summary, indent=2), flush=True)

    # ---- M2 pass (layer oracle MSE during forward; still return original y) ----
    print("[M2] oracle MoE-output metrics ...", flush=True)
    collector.enable_m1 = False
    collector.enable_m2 = True
    collector.m2 = M2Acc(drop_ratios=drop_ratios, strategies=STRATEGIES)
    # subsample for M2 speed
    m2_prompts = prompts[:: max(1, len(prompts) // min(len(prompts), args.n_per_domain))]
    m2_prompts = m2_prompts[: max(48, args.n_per_domain)]

    with torch.no_grad():
        for i, (domain, text) in enumerate(m2_prompts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            try:
                embed_dev = model.model.language_model.embed_tokens.weight.device
            except Exception:
                embed_dev = device0
            enc = {k: v.to(embed_dev) for k, v in enc.items()}
            _ = model(**enc)
            if (i + 1) % 16 == 0 or i == 0:
                print(f"  [M2 {i+1}/{len(m2_prompts)}] {domain}", flush=True)

    logit_rows = None
    if not args.skip_m2_logit:
        print("[M2] logit-level KL (applied drop) ...", flush=True)
        # temporarily restore original then run apply-drop forwards
        restore_patches(restored)
        logit_texts = [t for _, t in prompts[: args.logit_eval_n]]
        logit_rows = run_logit_m2(
            model,
            tokenizer,
            logit_texts,
            collector.rho_table,
            collector.epsilon_table,
            drop_ratios,
            ["random", "min_weight", "renorm", "shareguard"],
            args.max_length,
        )
        # re-install not needed
    else:
        restore_patches(restored)

    m2_summary = analyze_m2(collector, out_dir, logit_rows)
    print("[M2] summary:", json.dumps(m2_summary, indent=2), flush=True)

    overall = {
        "model": args.model,
        "n_per_domain": args.n_per_domain,
        "max_length": args.max_length,
        "drop_ratios": drop_ratios,
        "m1": m1_summary,
        "m2": m2_summary,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"[done] results in {out_dir}  elapsed={overall['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

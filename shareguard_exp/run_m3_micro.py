#!/usr/bin/env python3
"""
ShareGuard M3 EP-straggler microbenchmark (PyTorch / HF).

vLLM full EP on RTX 5090 (SM12) is blocked by FlashInfer arch detection.
This measures the same claim at MoE-layer granularity under simulated EP=4:
  dropping overloaded-rank branches reduces max-rank (straggler) expert compute time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m1_m2 import _expert_unweighted, load_prompts  # noqa: E402


def expert_to_rank(expert_ids: torch.Tensor, ep_size: int, num_experts: int) -> torch.Tensor:
    base = num_experts // ep_size
    rem = num_experts % ep_size
    if rem == 0:
        return (expert_ids // max(base, 1)).clamp(max=ep_size - 1)
    bounds = []
    start = 0
    for r in range(ep_size):
        n = base + (1 if r < rem else 0)
        bounds.append(start + n)
        start += n
    b = torch.tensor(bounds, device=expert_ids.device, dtype=expert_ids.dtype)
    return torch.searchsorted(b, expert_ids, right=False).clamp(max=ep_size - 1)


def select_drops(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    eps_row: torch.Tensor,
    ep_size: int,
    num_experts: int,
    capacity: float,
    mode: str,
) -> torch.Tensor:
    T, K = topk_ids.shape
    device = topk_ids.device
    valid = topk_ids >= 0
    dest = expert_to_rank(topk_ids.clamp(min=0), ep_size, num_experts)
    dest = torch.where(valid, dest, torch.full_like(dest, -1))
    loads = torch.zeros(ep_size, device=device, dtype=torch.float32)
    flat_dest, flat_valid = dest.reshape(-1), valid.reshape(-1)
    for r in range(ep_size):
        loads[r] = ((flat_dest == r) & flat_valid).sum()
    total = float(valid.sum().item())
    cap = max(1.0, capacity * (total / max(ep_size, 1)))

    flat_w = topk_weights.float().reshape(-1)
    if mode == "min_weight":
        scores = flat_w.clone()
    else:
        eids = topk_ids.clamp(min=0).reshape(-1)
        eps_b = eps_row.to(device=device, dtype=torch.float32)[eids]
        scores = (flat_w ** 2) * eps_b
        scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, 1e9))
    scores = torch.where(flat_valid, scores, torch.full_like(scores, 1e30))

    drop_mask = torch.zeros(T * K, dtype=torch.bool, device=device)
    for r in range(ep_size):
        overload = int(loads[r].item() - cap)
        if overload <= 0:
            continue
        on_rank = (flat_dest == r) & flat_valid & (~drop_mask)
        cand = on_rank.nonzero(as_tuple=False).reshape(-1)
        if cand.numel() == 0:
            continue
        n_drop = min(overload, int(cand.numel()))
        _, local = torch.topk(scores[cand], n_drop, largest=False)
        drop_mask[cand[local]] = True
    return drop_mask.view(T, K)


@torch.no_grad()
def time_straggler_ms(
    mlp,
    h: torch.Tensor,
    selected_experts: torch.Tensor,
    drop_mask: torch.Tensor | None,
    ep_size: int,
    num_experts: int,
) -> tuple[float, float, float]:
    """Return (straggler_ms, max_rank_load, n_branches_kept)."""
    ids = selected_experts.clone()
    if drop_mask is not None:
        ids = ids.masked_fill(drop_mask, -1)
    keep = ids >= 0
    n_kept = float(keep.sum().item())
    dest = expert_to_rank(ids.clamp(min=0), ep_size, num_experts)
    dest = torch.where(keep, dest, torch.full_like(dest, -1))
    loads = [float((dest == r).sum().item()) for r in range(ep_size)]
    max_load = max(loads) if loads else 0.0

    base = num_experts // ep_size
    rem = num_experts % ep_size
    rank_ms: list[float] = []
    for r in range(ep_size):
        start = r * base + min(r, rem)
        nloc = base + (1 if r < rem else 0)
        if h.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for e in range(start, start + nloc):
            token_idx, _ = torch.where(ids == e)
            if token_idx.numel() == 0:
                continue
            _ = _expert_unweighted(mlp.experts, h, e, token_idx)
        if h.is_cuda:
            torch.cuda.synchronize()
        rank_ms.append((time.perf_counter() - t0) * 1000.0)
    return max(rank_ms) if rank_ms else 0.0, max_load, n_kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/model/Qwen3.6-35B-A3B")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/results/shareguard_m3")
    ap.add_argument(
        "--rho-path",
        default="/root/autodl-tmp/results/shareguard_ablations/m1_rho_epsilon_with_domain.pt",
    )
    ap.add_argument("--n-per-domain", type=int, default=12)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--capacity", type=float, default=0.85)
    ap.add_argument("--ep-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-layers-per-prompt", type=int, default=8, help="Time subset of MoE layers for speed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    data = torch.load(args.rho_path, map_location="cpu", weights_only=False)
    eps = data["eps_by_expert"].float()
    eps = eps.nan_to_num(float(eps[torch.isfinite(eps)].mean()))

    prompts = load_prompts(args.n_per_domain, seed=args.seed)
    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

    print(f"[M3-micro] loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    n_exp = int(model.config.text_config.num_experts)
    language = model.model.language_model
    moe_layers = [(i, layer.mlp) for i, layer in enumerate(language.layers) if hasattr(layer.mlp, "shared_expert")]
    # subsample layers evenly
    if args.max_layers_per_prompt < len(moe_layers):
        idxs = torch.linspace(0, len(moe_layers) - 1, args.max_layers_per_prompt).long().tolist()
        timed_set = {moe_layers[i][0] for i in idxs}
    else:
        timed_set = {i for i, _ in moe_layers}
    print(f"[M3-micro] MoE layers={len(moe_layers)} timed={len(timed_set)} prompts={len(prompts)}", flush=True)

    try:
        embed_dev = language.embed_tokens.weight.device
    except Exception:
        embed_dev = torch.device("cuda:0")

    modes = ["baseline", "min_weight", "shareguard"]
    acc = {m: {"straggler_ms": 0.0, "max_load": 0.0, "n_branches": 0.0, "n": 0} for m in modes}

    with torch.no_grad():
        for pi, (domain, text) in enumerate(prompts):
            captured: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            restores = []
            for li, mlp in moe_layers:
                orig = mlp.forward

                def make_fwd(layer_idx: int, mlp_ref):
                    def fwd(hidden_states):
                        b, s, d = hidden_states.shape
                        h = hidden_states.view(-1, d)
                        S_raw = mlp_ref.shared_expert(h)
                        _, rw, se = mlp_ref.gate(h)
                        g = torch.sigmoid(mlp_ref.shared_expert_gate(h))
                        y = mlp_ref.experts(h, se, rw) + g * S_raw
                        if layer_idx in timed_set:
                            captured.append((layer_idx, h.detach(), se.detach(), rw.detach()))
                        return y.view(b, s, d)

                    return fwd

                mlp.forward = make_fwd(li, mlp)
                restores.append((mlp, orig))

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(embed_dev) for k, v in enc.items()}
            _ = model(**enc)
            for mlp, orig in restores:
                mlp.forward = orig

            for layer_idx, h, se, rw in captured:
                eps_row = eps[layer_idx].to(h.device)
                mlp = next(m for i, m in moe_layers if i == layer_idx)
                drops = {
                    "baseline": None,
                    "min_weight": select_drops(
                        se, rw, eps_row, args.ep_size, n_exp, args.capacity, "min_weight"
                    ),
                    "shareguard": select_drops(
                        se, rw, eps_row, args.ep_size, n_exp, args.capacity, "shareguard"
                    ),
                }
                for mode in modes:
                    ms, mx, nk = time_straggler_ms(
                        mlp, h, se, drops[mode], args.ep_size, n_exp
                    )
                    acc[mode]["straggler_ms"] += ms
                    acc[mode]["max_load"] += mx
                    acc[mode]["n_branches"] += nk
                    acc[mode]["n"] += 1

            if (pi + 1) % 4 == 0 or pi == 0:
                print(f"  [{pi+1}/{len(prompts)}] domain={domain} timed_layers={len(captured)}", flush=True)

    summary: dict = {"capacity": args.capacity, "ep_size": args.ep_size, "modes": {}}
    for mode in modes:
        n = max(acc[mode]["n"], 1)
        summary["modes"][mode] = {
            "mean_straggler_ms": acc[mode]["straggler_ms"] / n,
            "mean_max_rank_load": acc[mode]["max_load"] / n,
            "mean_branches_kept": acc[mode]["n_branches"] / n,
            "n_layer_calls": acc[mode]["n"],
        }
    base = summary["modes"]["baseline"]["mean_straggler_ms"]
    summary["straggler_reduction_vs_baseline"] = {
        m: (base - summary["modes"][m]["mean_straggler_ms"]) / base if base > 0 else float("nan")
        for m in ("min_weight", "shareguard")
    }
    bload = summary["modes"]["baseline"]["mean_max_rank_load"]
    summary["max_load_reduction_vs_baseline"] = {
        m: (bload - summary["modes"][m]["mean_max_rank_load"]) / bload if bload > 0 else float("nan")
        for m in ("min_weight", "shareguard")
    }
    red = summary["straggler_reduction_vs_baseline"]["shareguard"]
    summary["go_nogo"] = "GO" if (isinstance(red, float) and math.isfinite(red) and red >= 0.08) else "WEAK / CHECK"
    summary["note"] = (
        "PyTorch simulated EP=4 straggler microbenchmark. "
        "vLLM full-server EP blocked on RTX 5090 (SM12) by FlashInfer arch detection; "
        "ShareGuard hooks are already patched into site-packages moe_runner for when runtime is fixed."
    )
    (out_dir / "m3_micro_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "m3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[M3-micro] done → {out_dir}", flush=True)


if __name__ == "__main__":
    main()

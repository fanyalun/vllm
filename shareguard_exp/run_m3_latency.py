#!/usr/bin/env python3
"""
ShareGuard M3: real EP latency benchmark on vLLM.

Compares:
  - baseline (ShareGuard off)
  - min_weight drop under capacity
  - shareguard drop under capacity

Metrics: wall-clock throughput, TPOT proxy, ShareGuard overhead, load imbalance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure shareguard_exp importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def ensure_python_tools_on_path() -> None:
    """Expose venv console tools such as ninja to spawned vLLM workers."""
    python_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join((python_bin, *path_entries))


PROMPTS_MIXED = [
    # general
    "Explain what cloud computing is in simple terms.",
    "What are three healthy breakfast ideas?",
    "Summarize the plot of Romeo and Juliet in two paragraphs.",
    "How does a refrigerator keep food cold?",
    "Write a polite email declining a meeting invitation.",
    "What is the difference between weather and climate?",
    "Give tips for improving focus while studying.",
    "Describe the water cycle for a middle-school student.",
    # math
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Mark has a garden with flowers. He planted 25 tulips, and half as many roses as tulips. He planted 10 more daisies than roses. How many flowers does he have in total?",
    "Albert is wondering how much pizza he can eat. He buys 3 large pizzas that are each cut into 8 slices. If he eats 2/3 of the pizza, how many slices does he eat?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    # code
    "Write a Python function to check if a number is prime.",
    "Implement binary search in Python on a sorted list.",
    "Write a function that merges two sorted lists into one sorted list.",
    "Write a Python generator that yields Fibonacci numbers.",
    "Implement a function to reverse a linked list.",
    "Write a Python function to compute the edit distance between two strings.",
    "Implement BFS traversal of a graph represented as an adjacency list.",
    "Write a function that finds the longest common subsequence of two strings.",
]


def build_prompts(n: int, seed: int = 0) -> list[str]:
    import random

    rng = random.Random(seed)
    out = list(PROMPTS_MIXED)
    while len(out) < n:
        out.extend(PROMPTS_MIXED)
    rng.shuffle(out)
    return out[:n]


def aggregate_worker_stats(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("ShareGuard worker stats are empty")
    calls = max(int(row["calls"]) for row in rows)
    branches_seen = max(int(row["branches_seen"]) for row in rows)
    branches_dropped = max(int(row["branches_dropped"]) for row in rows)

    return {
        "workers": len(rows),
        "aggregation": "max_across_replicated_ep_workers",
        "calls": calls,
        "overload_calls": max(int(row["overload_calls"]) for row in rows),
        "branches_seen": branches_seen,
        "branches_dropped": branches_dropped,
        "drop_rate": branches_dropped / branches_seen if branches_seen else 0.0,
        "select_ms_total": max(float(row["select_ms_total"]) for row in rows),
        "select_ms_avg": max(float(row["select_ms_avg"]) for row in rows),
        "max_load_before_mean": max(
            float(row["max_load_before_mean"]) for row in rows
        ),
        "max_load_after_mean": max(
            float(row["max_load_after_mean"]) for row in rows
        ),
    }


def run_one_config(
    *,
    model: str,
    tp: int,
    mode: str,
    capacity: float,
    rho_path: str,
    prompts: list[str],
    max_tokens: int,
    max_num_seqs: int,
    warmup: int,
    enable_ep: bool,
) -> dict:
    ensure_python_tools_on_path()
    # Set ShareGuard env before importing/patching vLLM internals
    if mode == "baseline":
        os.environ["SHAREGUARD_ENABLE"] = "0"
        os.environ["SHAREGUARD_MODE"] = "off"
    else:
        os.environ["SHAREGUARD_ENABLE"] = "1"
        os.environ["SHAREGUARD_MODE"] = mode
        os.environ["SHAREGUARD_CAPACITY"] = str(capacity)
        os.environ["SHAREGUARD_RHO_PATH"] = rho_path
        os.environ["SHAREGUARD_COMPENSATE"] = "1"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    # Env vars are enough for EP workers (site-packages moe_runner hooks).
    # Avoid importing torch/CUDA before LLM construction.
    print(
        f"[M3] loading LLM mode={mode} tp={tp} ep={enable_ep} "
        "moe_backend=triton ...",
        flush=True,
    )
    t_load0 = time.time()
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.shareguard_runtime import (
        get_shareguard_stats_for_model,
        register_moe_layers,
        reset_shareguard_stats_for_model,
    )

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        enable_expert_parallel=enable_ep,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_log_stats=False,
        disable_custom_all_reduce=True,  # PCIe multi-GPU
        moe_backend="triton",  # Required by the ShareGuard top-k hook.
    )
    print(f"[M3] loaded in {time.time()-t_load0:.1f}s", flush=True)

    registered_layers = None
    if mode != "baseline":
        registered_layers = llm.apply_model(register_moe_layers)
        if not registered_layers or min(registered_layers) <= 0:
            raise RuntimeError(
                f"ShareGuard did not register MoE layers: {registered_layers}"
            )
        if len(set(registered_layers)) != 1:
            raise RuntimeError(
                "ShareGuard workers registered different MoE layer counts: "
                f"{registered_layers}"
            )

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    # warmup
    warm = prompts[: max(1, min(warmup, len(prompts)))]
    _ = llm.generate(warm, sp)

    if mode != "baseline":
        llm.apply_model(reset_shareguard_stats_for_model)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    n_out_tokens = 0
    for o in outputs:
        for out in o.outputs:
            n_out_tokens += len(out.token_ids)

    worker_stats = None
    shareguard_stats = None
    if mode != "baseline":
        worker_stats = llm.apply_model(get_shareguard_stats_for_model)
        shareguard_stats = aggregate_worker_stats(worker_stats)
        if shareguard_stats["branches_seen"] <= 0:
            raise RuntimeError("ShareGuard hook ran without observing any branches")
        if capacity < 1.0 and shareguard_stats["branches_dropped"] <= 0:
            raise RuntimeError(
                "ShareGuard hook observed branches but did not drop any at "
                f"capacity={capacity}"
            )

    # Prefer vLLM metrics if present
    try:
        metrics = llm.get_metrics() if hasattr(llm, "get_metrics") else None
    except Exception:
        metrics = None

    result = {
        "mode": mode,
        "capacity": capacity if mode != "baseline" else None,
        "tp": tp,
        "enable_expert_parallel": enable_ep,
        "n_prompts": len(prompts),
        "max_tokens": max_tokens,
        "max_num_seqs": max_num_seqs,
        "elapsed_s": elapsed,
        "output_tokens": n_out_tokens,
        "throughput_tok_s": n_out_tokens / elapsed if elapsed > 0 else 0.0,
        "tpot_ms": (elapsed / n_out_tokens * 1000.0) if n_out_tokens else None,
        "registered_moe_layers": registered_layers,
        "shareguard_stats": shareguard_stats,
        "shareguard_stats_workers": worker_stats,
        "metrics_available": metrics is not None,
    }
    dropped = shareguard_stats["branches_dropped"] if shareguard_stats else 0
    print(
        f"[M3] mode={mode} tok/s={result['throughput_tok_s']:.2f} "
        f"tpot_ms={result['tpot_ms']:.3f} dropped={dropped}",
        flush=True,
    )

    # destroy to free GPUs for next config when run in-process (best-effort)
    try:
        del llm
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    ap.add_argument(
        "--out-dir", default="/home/fanya/sharedguard/results/shareguard_m3"
    )
    ap.add_argument("--rho-path", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--capacity", type=float, default=0.85)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--modes", default="baseline,min_weight,shareguard")
    ap.add_argument(
        "--no-ep", action="store_true", help="Disable expert parallel (debug)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(args.n_prompts, seed=args.seed)
    (out_dir / "prompts.json").write_text(
        json.dumps(prompts, indent=2), encoding="utf-8"
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    # IMPORTANT: each mode in a subprocess to avoid engine reuse issues
    import subprocess

    results = []
    for mode in modes:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-run-one",
            "--model", args.model,
            "--out-dir", str(out_dir),
            "--rho-path", args.rho_path,
            "--tp", str(args.tp),
            "--capacity", str(args.capacity),
            "--n-prompts", str(args.n_prompts),
            "--max-tokens", str(args.max_tokens),
            "--max-num-seqs", str(args.max_num_seqs),
            "--warmup", str(args.warmup),
            "--modes", mode,
            "--seed", str(args.seed),
        ]
        if args.no_ep:
            cmd.append("--no-ep")
        print(f"[M3] launching subprocess for mode={mode}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get(
            "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.tp))
        )
        # Avoid nested re-dispatch: child uses --internal-run-one
        rc = subprocess.run(cmd, env=env)
        if rc.returncode != 0:
            print(f"[M3] mode={mode} FAILED rc={rc.returncode}", flush=True)
            results.append({"mode": mode, "error": f"rc={rc.returncode}"})
            continue
        result_path = out_dir / f"m3_{mode}.json"
        one = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(one)

    # summarize
    failed_modes = [row["mode"] for row in results if "error" in row]
    summary = {
        "runs": results,
        "tp": args.tp,
        "capacity": args.capacity,
        "model": args.model,
        "complete": not failed_modes,
        "failed_modes": failed_modes,
    }
    base = next(
        (
            row
            for row in results
            if row.get("mode") == "baseline" and "throughput_tok_s" in row
        ),
        None,
    )
    sg = next(
        (
            row
            for row in results
            if row.get("mode") == "shareguard" and "throughput_tok_s" in row
        ),
        None,
    )
    if base and sg and base["throughput_tok_s"] > 0:
        summary["throughput_gain_vs_baseline"] = (
            sg["throughput_tok_s"] - base["throughput_tok_s"]
        ) / base["throughput_tok_s"]
        if base.get("tpot_ms") and sg.get("tpot_ms"):
            summary["tpot_reduction_vs_baseline"] = (
                base["tpot_ms"] - sg["tpot_ms"]
            ) / base["tpot_ms"]
        summary["go_nogo"] = (
            "GO"
            if summary.get("tpot_reduction_vs_baseline", 0) >= 0.08
            or summary.get("throughput_gain_vs_baseline", 0) >= 0.08
            else "WEAK / CHECK"
        )
    (out_dir / "m3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[M3] summary:", json.dumps(summary, indent=2), flush=True)
    if failed_modes:
        raise SystemExit(f"M3 failed modes: {','.join(failed_modes)}")


def _internal_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal-run-one", action="store_true")
    ap.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    ap.add_argument(
        "--out-dir", default="/home/fanya/sharedguard/results/shareguard_m3"
    )
    ap.add_argument("--rho-path", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--capacity", type=float, default=0.85)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--modes", default="baseline")
    ap.add_argument("--no-ep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mode = args.modes.split(",")[0].strip()
    prompts = build_prompts(args.n_prompts, seed=args.seed)
    result = run_one_config(
        model=args.model,
        tp=args.tp,
        mode=mode,
        capacity=args.capacity,
        rho_path=args.rho_path,
        prompts=prompts,
        max_tokens=args.max_tokens,
        max_num_seqs=args.max_num_seqs,
        warmup=args.warmup,
        enable_ep=not args.no_ep,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"m3_{mode}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if "--internal-run-one" in sys.argv:
        _internal_main()
    else:
        main()

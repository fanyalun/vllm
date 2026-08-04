#!/usr/bin/env python3
"""对比策略扫描：只评 ACC / pass@1

策略: baseline / min_weight / renorm / shareguard
削减比例: 5% / 10% / 15% / 20%

ACC:
  - GSM8K: 最终答案对错
  - HumanEval+MBPP: pass@1
  - MMLU: 选项字母对错
  - LiveBench: 与 answer / <solution> 匹配

用法:
  python run_strategy_scan.py
  LIMIT=16 bash run_strategy_scan.sh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(
    os.environ.get("SHAREDGUARD_ROOT", "/home/fanya/sharedguard")
).expanduser()
EXP_DIR = Path(
    os.environ.get("SHAREGUARD_EXP", str(Path(__file__).resolve().parent))
).expanduser()
if not (EXP_DIR / "run_m1_m2.py").is_file():
    raise FileNotFoundError(
        f"missing ShareGuard experiment helpers: {EXP_DIR / 'run_m1_m2.py'}; "
        "set SHAREGUARD_EXP"
    )
sys.path.insert(0, str(EXP_DIR))
from run_m1_m2 import make_apply_drop_forward  # noqa: E402

DEFAULT_MODEL = "/data1/fanya/Qwen/Qwen3.6-35B-A3B"
DEFAULT_TEST_DIR = str(REPO_ROOT / "dataset/slide/test")
DEFAULT_FULL_DATA_DIR = str(REPO_ROOT / "dataset/full")
DEFAULT_RHO = str(
    REPO_ROOT / "results/shareguard_ablations/m1_rho_epsilon_with_domain.pt"
)
PREFERRED_RHO = str(REPO_ROOT / "code/pt/mixed_bf16.pt")
DEFAULT_OUT = str(REPO_ROOT / "results/strategy_scan")

STRATEGIES = ["baseline", "min_weight", "renorm", "shareguard"]
DOMAIN_FILES = {
    "math": "GSM8K.jsonl",
    "code": "humaneval_mbpp.jsonl",
    "mmlu": "mmlu.jsonl",
    "livebench": "livebench.jsonl",
}
DOMAINS = list(DOMAIN_FILES.keys())


def load_rho_eps(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    rho = data["rho_by_expert"].float().nan_to_num(0.0)
    eps = data["eps_by_expert"].float()
    mean_eps = (
        float(eps[torch.isfinite(eps)].mean())
        if torch.isfinite(eps).any()
        else 1.0
    )
    eps = eps.nan_to_num(mean_eps)
    return rho, eps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{lineno}: {exc}") from exc
    return rows


def load_gsm8k_answers(path: Path) -> dict[str, str]:
    q2a: dict[str, str] = {}
    for row in read_jsonl(path):
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        if prompt and row.get("answer") is not None:
            q2a[prompt] = str(row["answer"])
    return q2a


def load_code_gold(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def load_test_set(test_dir: Path, full_data_dir: Path) -> list[dict]:
    gsm_ans = load_gsm8k_answers(full_data_dir / "GSM8K.jsonl")
    code_gold = load_code_gold(full_data_dir / "humaneval_mbpp.jsonl")
    rows: list[dict] = []
    for domain, fname in DOMAIN_FILES.items():
        path = test_dir / fname
        for obj in read_jsonl(path):
            prompt = (obj.get("prompt") or "").strip()
            if not prompt:
                continue
            item = {
                "id": obj.get("id"),
                "domain": obj.get("domain") or domain,
                "source": obj.get("source"),
                "prompt": prompt,
            }
            if item["domain"] == "math":
                item["gold_answer"] = obj.get("answer") or gsm_ans.get(prompt)
            elif item["domain"] == "code":
                g = code_gold.get(item["id"], {})
                item["source"] = item.get("source") or g.get("source")
                item["entry_point"] = g.get("entry_point")
                item["test"] = g.get("test")
                item["test_list"] = g.get("test_list")
            elif item["domain"] == "mmlu":
                item["gold_answer"] = obj.get("answer")
            elif item["domain"] == "livebench":
                item["gold_answer"] = obj.get("answer")
            rows.append(item)

    missing_required = [
        item["id"]
        for item in rows
        if (
            item["domain"] in ("math", "mmlu")
            and not item.get("gold_answer")
        )
        or (
            item["domain"] == "code"
            and not item.get("test")
            and not item.get("test_list")
        )
    ]
    if missing_required:
        shown = ", ".join(str(x) for x in missing_required[:5])
        raise ValueError(
            f"missing required gold data for {len(missing_required)} samples: {shown}"
        )
    return rows


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def list_moe_layers(model):
    language = model.model.language_model
    moes = []
    for i, layer in enumerate(language.layers):
        if hasattr(layer.mlp, "shared_expert"):
            moes.append((i, layer.mlp, layer.mlp.forward))
    return moes


def install_strategy(moes, strategy: str, drop_ratio: float, rho, eps, seed: int = 123):
    for i, mlp, orig in moes:
        if strategy == "baseline" or drop_ratio <= 0:
            mlp.forward = orig
        else:
            fwd = make_apply_drop_forward(i, strategy, drop_ratio, rho, eps, seed=seed)
            mlp.forward = fwd.__get__(mlp, type(mlp))


def restore_moe(moes):
    for _, mlp, orig in moes:
        mlp.forward = orig


def embed_device(model):
    try:
        return model.model.language_model.embed_tokens.weight.device
    except Exception:
        return next(model.parameters()).device


# ---- ACC helpers ----

_ANS_RE = re.compile(r"####\s*([^\n]+)")
_FLEX_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_LETTER_RE = re.compile(r"\b([A-F])\b")
_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.IGNORECASE | re.DOTALL)


def extract_gsm8k_answer(text: str) -> str | None:
    if not text:
        return None
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = _FLEX_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def gsm8k_correct(pred_text: str, gold_answer: str | None) -> bool:
    if not gold_answer:
        return False
    gold = extract_gsm8k_answer(gold_answer) or gold_answer.strip()
    pred = extract_gsm8k_answer(pred_text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return pred.strip() == gold.strip()


def extract_code_completion(
    prompt: str,
    generated: str,
    source: str | None,
    entry_point: str | None,
) -> str:
    gen = generated
    if "```" in gen:
        parts = gen.split("```")
        for i, p in enumerate(parts):
            if i % 2 == 1:
                lines = p.splitlines()
                if lines and lines[0].strip().lower() in ("python", "py"):
                    lines = lines[1:]
                gen = "\n".join(lines).strip("\n")
                break
    if source != "humaneval":
        return gen
    prompt_prefix = prompt.strip()[: min(40, len(prompt.strip()))]
    if prompt_prefix and gen.lstrip().startswith(prompt_prefix):
        return gen
    if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", gen):
        return gen
    return prompt.rstrip() + "\n" + gen


_CODE_EVAL_WORKER = r"""
import json
import os
import resource
import socket
import sys

payload = json.load(sys.stdin)
memory = 2 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))

def blocked_socket(*args, **kwargs):
    raise PermissionError("network disabled during code evaluation")

socket.socket = blocked_socket
namespace = {"__name__": "__main__"}
with open(os.devnull, "w") as devnull:
    sys.stdout = devnull
    sys.stderr = devnull
    exec(compile(payload["program"], "<candidate>", "exec"), namespace, namespace)
    if payload["kind"] == "humaneval":
        exec(compile(payload["test"], "<humaneval_test>", "exec"), namespace, namespace)
        namespace["check"](namespace[payload["entry_point"]])
    else:
        for statement in payload["test_list"]:
            exec(compile(statement, "<mbpp_test>", "exec"), namespace, namespace)
"""


def run_code_test(payload: dict[str, Any], timeout_s: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="shareguard-code-") as workdir:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _CODE_EVAL_WORKER],
            cwd=workdir,
            env={"PATH": os.environ.get("PATH", "")},
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            proc.communicate(json.dumps(payload), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return False, "timeout"
    return (True, "pass") if proc.returncode == 0 else (False, "fail")


def run_humaneval(
    program: str,
    test_code: str,
    entry_point: str,
    timeout_s: float,
) -> tuple[bool, str]:
    return run_code_test(
        {
            "kind": "humaneval",
            "program": program,
            "test": test_code or "",
            "entry_point": entry_point,
        },
        timeout_s,
    )


def run_mbpp(
    program: str,
    test_list: list | None,
    timeout_s: float,
) -> tuple[bool, str]:
    if not test_list:
        return False, "missing_tests"
    return run_code_test(
        {"kind": "mbpp", "program": program, "test_list": test_list},
        timeout_s,
    )


def mmlu_correct(pred_text: str, gold: str | None) -> bool:
    if not gold:
        return False
    gold = str(gold).strip().upper()[:1]
    m = re.search(r"(?:answer|选项)\s*[:：]\s*([A-F])", pred_text, re.IGNORECASE)
    if m:
        return m.group(1).upper() == gold
    letters = _LETTER_RE.findall((pred_text or "").upper())
    return bool(letters) and letters[-1] == gold


def livebench_correct(pred_text: str, gold: str | None) -> bool:
    if gold is None:
        return False
    gold_s = str(gold).strip()
    m = _SOLUTION_RE.search(pred_text or "")
    pred = (m.group(1).strip() if m else (pred_text or "").strip())
    if not pred:
        return False

    def norm(x: str) -> str:
        return re.sub(r"\s+", " ", x.replace("，", ",")).strip().lower()

    if norm(pred) == norm(gold_s) or norm(gold_s) in norm(pred):
        return True
    last = pred.strip().splitlines()[-1].strip()
    return norm(last) == norm(gold_s)


@torch.no_grad()
def greedy_generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_prompt_len: int,
    max_new_tokens: int,
    device,
) -> list[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_width = enc["input_ids"].shape[1]
    return [
        tokenizer.decode(tokens[input_width:], skip_special_tokens=True)
        for tokens in out
    ]


def score_prediction(
    sample: dict[str, Any],
    text: str,
    code_timeout_s: float,
) -> tuple[bool | None, str]:
    domain = sample["domain"]
    if domain == "math":
        return gsm8k_correct(text, sample.get("gold_answer")), "scored"
    if domain == "code":
        program = extract_code_completion(
            sample["prompt"],
            text,
            sample.get("source"),
            sample.get("entry_point"),
        )
        if sample.get("source") == "humaneval":
            return run_humaneval(
                program,
                sample.get("test") or "",
                sample.get("entry_point") or "",
                code_timeout_s,
            )
        return run_mbpp(program, sample.get("test_list"), code_timeout_s)
    if domain == "mmlu":
        return mmlu_correct(text, sample.get("gold_answer")), "scored"
    if domain == "livebench":
        if not sample.get("gold_answer"):
            return None, "unscorable_missing_gold"
        return livebench_correct(text, sample.get("gold_answer")), "scored"
    return None, "unscorable_unknown_domain"


def record_key(record: dict[str, Any]) -> str:
    return f"{record['domain']}:{record['id']}"


def load_generation_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignoring incomplete {path.name}:{lineno}", flush=True)
                continue
            records[record_key(record)] = record
    return records


def compact_generation_records(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as output:
        for record in records.values():
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(path)


@torch.no_grad()
def eval_acc(
    model,
    tokenizer,
    samples: list[dict],
    strategy: str,
    drop_ratio: float,
    rho,
    eps,
    moes,
    max_prompt_len: int,
    max_new_tokens: int,
    batch_size: int,
    code_timeout_s: float,
    records_path: Path,
    seed: int,
) -> dict:
    install_strategy(moes, strategy, drop_ratio, rho, eps, seed=seed)
    device = embed_device(model)
    records = load_generation_records(records_path)
    expected_keys = {f"{sample['domain']}:{sample['id']}" for sample in samples}
    records = {key: value for key, value in records.items() if key in expected_keys}
    if records_path.exists():
        compact_generation_records(records_path, records)
    pending = [
        sample
        for sample in samples
        if f"{sample['domain']}:{sample['id']}" not in records
    ]
    print(
        f"  [resume] completed={len(records)} pending={len(pending)} "
        f"records={records_path.name}",
        flush=True,
    )

    records_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with records_path.open("a", encoding="utf-8") as output:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                scorable = [
                    sample
                    for sample in batch
                    if not (
                        sample["domain"] == "livebench"
                        and not sample.get("gold_answer")
                    )
                ]
                predictions: dict[str, str] = {}
                if scorable:
                    texts = greedy_generate_batch(
                        model,
                        tokenizer,
                        [sample["prompt"] for sample in scorable],
                        max_prompt_len,
                        max_new_tokens,
                        device,
                    )
                    predictions = {
                        f"{sample['domain']}:{sample['id']}": text
                        for sample, text in zip(scorable, texts)
                    }

                for sample in batch:
                    key = f"{sample['domain']}:{sample['id']}"
                    text = predictions.get(key, "")
                    correct, status = score_prediction(
                        sample, text, code_timeout_s
                    )
                    record = {
                        "id": sample["id"],
                        "domain": sample["domain"],
                        "source": sample.get("source"),
                        "strategy": strategy,
                        "drop_ratio": 0.0 if strategy == "baseline" else drop_ratio,
                        "prediction": text,
                        "gold_answer": sample.get("gold_answer"),
                        "correct": correct,
                        "status": status,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    records[key] = record

                completed = len(records)
                if completed == len(batch) or completed % 20 < len(batch):
                    print(
                        f"  [{strategy}@{drop_ratio:.2f}] "
                        f"{completed}/{len(samples)}",
                        flush=True,
                    )
    finally:
        restore_moe(moes)

    missing = expected_keys - records.keys()
    if missing:
        raise RuntimeError(f"generation records incomplete: {len(missing)} missing")

    acc = {domain: [] for domain in DOMAINS}
    status_counts: dict[str, int] = {}
    for record in records.values():
        status = str(record.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if record.get("correct") is not None:
            acc[record["domain"]].append(1.0 if record["correct"] else 0.0)

    return {
        "strategy": strategy,
        "drop_ratio": 0.0 if strategy == "baseline" else drop_ratio,
        "acc_gsm8k": mean(acc["math"]),
        "acc_code_pass1": mean(acc["code"]),
        "acc_mmlu": mean(acc["mmlu"]),
        "acc_livebench": mean(acc["livebench"]),
        "acc_n_math": len(acc["math"]),
        "acc_n_code": len(acc["code"]),
        "acc_n_mmlu": len(acc["mmlu"]),
        "acc_n_livebench": len(acc["livebench"]),
        "n": len(samples),
        "n_scorable": sum(len(values) for values in acc.values()),
        "n_unscorable": sum(
            count
            for status, count in status_counts.items()
            if status.startswith("unscorable")
        ),
        "status_counts": status_counts,
        "records_path": str(records_path),
    }


def plot_acc(rows: list[dict], out_dir: Path, strategies: list[str]):
    import matplotlib.pyplot as plt

    ratios = sorted({r["drop_ratio"] for r in rows if r["strategy"] != "baseline"})
    ratios_pct = [r * 100 for r in ratios]

    def series(metric: str):
        out = {}
        for strategy in strategies:
            if strategy == "baseline":
                base = next(
                    (r for r in rows if r["strategy"] == "baseline"), None
                )
                if base is None:
                    continue
                out[strategy] = [base.get(metric)] * len(ratios)
                continue
            ys = []
            for ratio in ratios:
                hit = next(
                    (
                        r
                        for r in rows
                        if r["strategy"] == strategy
                        and abs(r["drop_ratio"] - ratio) < 1e-9
                    ),
                    None,
                )
                ys.append(hit.get(metric) if hit else None)
            out[strategy] = ys
        return out

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, metric, title in (
        (axes[0, 0], "acc_gsm8k", "GSM8K Acc"),
        (axes[0, 1], "acc_code_pass1", "HumanEval+MBPP pass@1"),
        (axes[1, 0], "acc_mmlu", "MMLU Acc"),
        (axes[1, 1], "acc_livebench", "LiveBench Acc"),
    ):
        ser = series(metric)
        for strategy, ys in ser.items():
            if any(v is None for v in ys):
                continue
            style = "--" if strategy == "baseline" else "-o"
            ax.plot(ratios_pct, ys, style, label=strategy, linewidth=2)
        ax.set_xlabel("Drop ratio (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scan_acc.png", dpi=160)
    plt.close(fig)


def config_slug(strategy: str, drop_ratio: float) -> str:
    ratio = 0.0 if strategy == "baseline" else drop_ratio
    return f"{strategy}_drop_{ratio:.2f}".replace(".", "p")


def write_json_atomic(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def sample_signature(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(str(sample.get("domain")).encode())
        digest.update(b"\0")
        digest.update(str(sample.get("id")).encode())
        digest.update(b"\0")
        digest.update(sample.get("prompt", "").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_run_manifest(
    out_dir: Path,
    manifest: dict[str, Any],
    overwrite: bool,
) -> None:
    path = out_dir / "scan_manifest.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            if not overwrite:
                raise RuntimeError(
                    f"output directory contains a different scan: {out_dir}; "
                    "use another OUT_DIR or pass --overwrite"
                )
            for pattern in ("*_generations.jsonl", "*_summary.json"):
                for artifact in out_dir.glob(pattern):
                    artifact.unlink()
            for name in (
                "strategy_scan_summary.json",
                "strategy_scan_table.json",
                "scan_acc.png",
            ):
                (out_dir / name).unlink(missing_ok=True)
    elif any(out_dir.glob("*_generations.jsonl")) or any(
        out_dir.glob("*_summary.json")
    ):
        if not overwrite:
            raise RuntimeError(
                f"output directory has legacy checkpoints without a manifest: "
                f"{out_dir}; use another OUT_DIR or pass --overwrite"
            )
        for pattern in ("*_generations.jsonl", "*_summary.json"):
            for artifact in out_dir.glob(pattern):
                artifact.unlink()
    write_json_atomic(path, manifest)


def write_scan_outputs(
    out_dir: Path,
    rows: list[dict],
    *,
    model: str,
    rho_path: str,
    test_dir: str,
    full_data_dir: str,
    counts: dict[str, int],
    drop_ratios: list[float],
    strategies: list[str],
    n_samples: int,
    batch_size: int,
    max_prompt_len: int,
    max_new_tokens: int,
) -> None:
    summary = {
        "model": model,
        "rho_path": rho_path,
        "test_dir": test_dir,
        "full_data_dir": full_data_dir,
        "n_samples": n_samples,
        "domain_counts": counts,
        "drop_ratios": drop_ratios,
        "strategies": strategies,
        "batch_size": batch_size,
        "max_prompt_len": max_prompt_len,
        "max_new_tokens": max_new_tokens,
        "metric": "acc",
        "rows": rows,
    }
    write_json_atomic(out_dir / "strategy_scan_summary.json", summary)
    table = [
        {
            "strategy": row["strategy"],
            "drop_ratio": row["drop_ratio"],
            "acc_gsm8k": row["acc_gsm8k"],
            "acc_code_pass1": row["acc_code_pass1"],
            "acc_mmlu": row["acc_mmlu"],
            "acc_livebench": row["acc_livebench"],
            "n_scorable": row["n_scorable"],
            "n_unscorable": row["n_unscorable"],
        }
        for row in rows
    ]
    write_json_atomic(out_dir / "strategy_scan_table.json", table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    ap.add_argument("--full-data-dir", default=DEFAULT_FULL_DATA_DIR)
    ap.add_argument("--rho-path", default="")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--drop-ratios", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--strategies", default="baseline,min_weight,renorm,shareguard")
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--code-timeout-s", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing per-configuration checkpoints",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and calibration table without loading the model",
    )
    args = ap.parse_args()

    drop_ratios = [float(x) for x in args.drop_ratios.split(",") if x.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"unsupported strategies: {', '.join(unknown)}")
    if not strategies:
        raise ValueError("at least one strategy is required")
    if not drop_ratios or any(ratio <= 0 or ratio >= 1 for ratio in drop_ratios):
        raise ValueError("drop ratios must be between 0 and 1")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.batch_size > 1 and any(strategy != "baseline" for strategy in strategies):
        print(
            "[warning] batch_size > 1 pads unequal-length prompts; padding branches "
            "participate in fixed-ratio drop selection. Use batch_size=1 for "
            "reportable algorithm-accuracy results.",
            flush=True,
        )
    if args.code_timeout_s <= 0:
        raise ValueError("code timeout must be positive")

    rho_path = args.rho_path
    if not rho_path:
        rho_path = PREFERRED_RHO if Path(PREFERRED_RHO).exists() else DEFAULT_RHO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_test_set(Path(args.test_dir), Path(args.full_data_dir))
    if args.limit:
        per = max(1, args.limit // len(DOMAINS))
        picked = []
        for d in DOMAINS:
            picked.extend([s for s in samples if s["domain"] == d][:per])
        samples = picked[: args.limit]

    counts = {d: sum(1 for s in samples if s["domain"] == d) for d in DOMAINS}
    print(f"[data] n={len(samples)} domain_counts={counts}", flush=True)
    print(f"[rho] {rho_path}", flush=True)
    rho, eps = load_rho_eps(rho_path)
    if rho.shape != eps.shape or rho.ndim != 2:
        raise ValueError(
            f"rho/epsilon must have matching [layers, experts] shapes: "
            f"rho={tuple(rho.shape)} eps={tuple(eps.shape)}"
        )
    print(
        f"[rho] shape={tuple(rho.shape)} "
        f"finite={int(torch.isfinite(eps).sum())}/{eps.numel()}",
        flush=True,
    )
    if args.validate_only:
        scorable = sum(
            not (sample["domain"] == "livebench" and not sample.get("gold_answer"))
            for sample in samples
        )
        print(
            f"[validate] OK samples={len(samples)} scorable={scorable} "
            f"unscorable={len(samples) - scorable}",
            flush=True,
        )
        return

    configs = [
        (strategy, ratio)
        for strategy in strategies
        for ratio in ([0.0] if strategy == "baseline" else drop_ratios)
    ]
    manifest = {
        "schema_version": 1,
        "model": str(Path(args.model).resolve()),
        "rho_path": str(Path(rho_path).resolve()),
        "rho_size": Path(rho_path).stat().st_size,
        "rho_mtime_ns": Path(rho_path).stat().st_mtime_ns,
        "test_dir": str(Path(args.test_dir).resolve()),
        "full_data_dir": str(Path(args.full_data_dir).resolve()),
        "sample_count": len(samples),
        "sample_signature": sample_signature(samples),
        "max_prompt_len": args.max_prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "code_timeout_s": args.code_timeout_s,
        "seed": args.seed,
    }
    prepare_run_manifest(out_dir, manifest, args.overwrite)
    if args.overwrite:
        for strategy, ratio in configs:
            slug = config_slug(strategy, ratio)
            for suffix in ("_generations.jsonl", "_summary.json"):
                (out_dir / f"{slug}{suffix}").unlink(missing_ok=True)

    rows: list[dict] = []
    pending_configs: list[tuple[str, float]] = []
    for strategy, ratio in configs:
        path = out_dir / f"{config_slug(strategy, ratio)}_summary.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            print(
                f"[resume] completed config {config_slug(strategy, ratio)}",
                flush=True,
            )
        else:
            pending_configs.append((strategy, ratio))

    if not pending_configs:
        write_scan_outputs(
            out_dir,
            rows,
            model=args.model,
            rho_path=rho_path,
            test_dir=args.test_dir,
            full_data_dir=args.full_data_dir,
            counts=counts,
            drop_ratios=drop_ratios,
            strategies=strategies,
            n_samples=len(samples),
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
        )
        plot_acc(rows, out_dir, strategies)
        print(f"[done] all configurations already complete -> {out_dir}", flush=True)
        return

    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

    torch.manual_seed(args.seed)
    print(f"[model] loading {args.model} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)

    moes = list_moe_layers(model)
    print(f"[patch] moe_layers={len(moes)}", flush=True)
    text_config = model.config.text_config
    expected_table_shape = (len(moes), int(text_config.num_experts))
    if not moes or tuple(rho.shape) != expected_table_shape:
        raise ValueError(
            f"calibration table does not match model MoE topology: "
            f"table={tuple(rho.shape)} model={expected_table_shape}"
        )

    for strategy, ratio in pending_configs:
        slug = config_slug(strategy, ratio)
        print(f"[eval] {strategy} drop={ratio:.2f} ...", flush=True)
        row = eval_acc(
            model,
            tokenizer,
            samples,
            strategy,
            ratio,
            rho,
            eps,
            moes,
            args.max_prompt_len,
            args.max_new_tokens,
            args.batch_size,
            args.code_timeout_s,
            out_dir / f"{slug}_generations.jsonl",
            args.seed,
        )
        write_json_atomic(out_dir / f"{slug}_summary.json", row)
        rows.append(row)
        write_scan_outputs(
            out_dir,
            rows,
            model=args.model,
            rho_path=rho_path,
            test_dir=args.test_dir,
            full_data_dir=args.full_data_dir,
            counts=counts,
            drop_ratios=drop_ratios,
            strategies=strategies,
            n_samples=len(samples),
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"  -> gsm={row['acc_gsm8k']} code={row['acc_code_pass1']} "
            f"mmlu={row['acc_mmlu']} lb={row['acc_livebench']} "
            f"scorable={row['n_scorable']}/{row['n']}",
            flush=True,
        )

    plot_acc(rows, out_dir, strategies)
    table = json.loads(
        (out_dir / "strategy_scan_table.json").read_text(encoding="utf-8")
    )
    print("[done] table:", json.dumps(table, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] outputs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

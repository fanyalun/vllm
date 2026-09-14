# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path


def preserve_contract(run_dir: Path, contract: dict) -> None:
    path = run_dir / "EXPERIMENT_CONTRACT.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            changed = sorted(
                key
                for key in previous.keys() | contract.keys()
                if previous.get(key) != contract.get(key)
            )
            raise RuntimeError(f"Resume contract mismatch: {changed}")
        return
    if any(run_dir.iterdir()):
        raise RuntimeError(f"Missing original experiment contract: {path}")
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")


def validate_target(cell: dict, model: str) -> None:
    actual = cell.get("model")
    if not actual or Path(actual).resolve() != Path(model).resolve():
        raise RuntimeError(f"Target model mismatch: expected {model!r}, got {actual!r}")


def refuse_incomplete_trace(cell_dir: Path) -> None:
    trace = cell_dir / "trace"
    if trace.exists() and any(
        p.is_file() and p.stat().st_size for p in trace.rglob("*")
    ):
        raise RuntimeError(
            f"Refusing to append to incomplete trace: {trace}. "
            "Preserve this attempt and use a fresh --run-dir."
        )


def prepare_performance_retry(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    previous = [p for p in directory.iterdir() if p.is_file()]
    if previous:
        attempts = directory / "attempts"
        attempts.mkdir(exist_ok=True)
        index = 1
        while (attempts / str(index)).exists():
            index += 1
        archive = attempts / str(index)
        archive.mkdir()
        for path in previous:
            path.rename(archive / path.name)

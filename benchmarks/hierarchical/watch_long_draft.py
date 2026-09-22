# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wait for an idle GPU and run both D32 smokes before acceptance measurements."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks/hierarchical/previous_config_20260909/samples_16.jsonl"


def command_output(command):
    return subprocess.check_output(command, text=True, cwd=ROOT, timeout=30)


def fingerprint():
    digest = hashlib.sha256()
    for command in (
        ["git", "rev-parse", "HEAD:vllm"],
        ["git", "diff", "HEAD", "--", "vllm"],
    ):
        digest.update(command_output(command).encode())
    digest.update(DATASET.read_bytes())
    for name in (
        "watch_long_draft.py",
        "run_long_draft.py",
        "long_draft_worker.py",
        "batch_worker.py",
    ):
        digest.update((ROOT / "benchmarks/hierarchical" / name).read_bytes())
    return digest.hexdigest()


def refresh_source(output, request_path, source, current):
    if current == source:
        return source
    if any(output.glob("*_command.json")) or any(output.glob("*_success.json")):
        raise RuntimeError("Source changed after first launch; queue stopped")
    request = json.loads(request_path.read_text())
    request.setdefault("source_history", []).append(
        dict(
            previous=source,
            current=current,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason="Rebased before any experiment launched",
        )
    )
    request["fingerprint"] = current
    write_json(request_path, request)
    return current


def gpu_snapshot():
    rows = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    applications = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = {}
    for uuid, pid in csv.reader(applications.splitlines(), skipinitialspace=True):
        processes.setdefault(uuid, []).append(int(pid))
    return [
        dict(
            index=int(index),
            uuid=uuid,
            total_mib=int(total),
            used_mib=int(used),
            utilization=int(utilization),
            pids=processes.get(uuid, []),
        )
        for index, uuid, total, used, utilization in csv.reader(
            rows.splitlines(), skipinitialspace=True
        )
    ]


def is_idle(gpu):
    return (
        not gpu["pids"]
        and gpu["used_mib"] < 1024
        and gpu["utilization"] <= 2
        and gpu["total_mib"] >= 80000
    )


def descendants(root_pid):
    pairs = [
        tuple(map(int, line.split()))
        for line in command_output(["ps", "-eo", "pid=,ppid="]).splitlines()
    ]
    owned = {root_pid}
    while True:
        expanded = owned | {pid for pid, parent in pairs if parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_queue(output, interval):
    with ExitStack() as handles:
        _run_queue(output, interval, handles)


def _run_queue(output, interval, handles):
    output.mkdir(parents=True, exist_ok=True)
    lock = handles.enter_context((output / "watcher.lock").open("a"))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = fingerprint()
    request_path = output / "request.json"
    if request_path.exists():
        previous = json.loads(request_path.read_text())["fingerprint"]
        source = refresh_source(output, request_path, previous, source)
    else:
        write_json(
            request_path,
            dict(
                fingerprint=source,
                scope="Experiments 1 and 2 only; forward matrix not implemented",
                dataset=str(DATASET),
                polling_seconds=interval,
                required_idle_samples=3,
                gpu_rule="No compute PID, <1024 MiB used, <=2% utilization",
                smokes=dict(samples=1, tokens=64, repeats=1),
                formal=dict(samples=16, tokens=256, repeats=2),
            ),
        )

    def status(state, **details):
        value = dict(
            state=state,
            timestamp=datetime.now(timezone.utc).isoformat(),
            watcher_pid=os.getpid(),
            **details,
        )
        write_json(output / "status.json", value)

    gpu_file = output / "selected_gpu.json"
    selected = json.loads(gpu_file.read_text()) if gpu_file.exists() else None
    for phase in ("smoke", "formal"):
        for mode in ("two_level_fixed", "three_level_balanced"):
            job = f"{phase}_{mode}"
            destination = output / job
            receipt = output / f"{job}_success.json"
            if receipt.exists():
                continue
            log_path = output / f"{job}.log"
            if log_path.exists() or destination.exists():
                raise RuntimeError(f"Incomplete job {job}; inspect logs before retry")
            stable_uuid, count = None, 0
            gpu_lock = None
            while True:
                if (output / "STOP").exists():
                    raise InterruptedError("Stopped by STOP file")
                current = fingerprint()
                if current != source:
                    source = refresh_source(output, request_path, source, current)
                    stable_uuid, count = None, 0
                try:
                    snapshot = gpu_snapshot()
                    idle = [
                        gpu
                        for gpu in snapshot
                        if is_idle(gpu)
                        and (selected is None or gpu["uuid"] == selected)
                    ]
                    candidate = idle[0]["uuid"] if idle else None
                    count = count + 1 if candidate and candidate == stable_uuid else 1
                    stable_uuid = candidate
                    status(
                        "WAITING_FOR_IDLE_GPU",
                        job=job,
                        gpus=snapshot,
                        stable_samples=count if candidate else 0,
                    )
                    if candidate and count >= 3:
                        gpu_lock = handles.enter_context(
                            Path(f"/tmp/vllm_long_draft_{candidate}.lock").open("a")  # noqa: SIM115
                        )
                        try:
                            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            gpu_lock.close()
                            gpu_lock = None
                        else:
                            selected = candidate
                            write_json(gpu_file, selected)
                            break
                except (subprocess.SubprocessError, ValueError) as error:
                    count, stable_uuid = 0, None
                    status("GPU_QUERY_ERROR", job=job, error=str(error))
                time.sleep(interval)
            environment = os.environ.copy()
            environment.update(
                CUDA_VISIBLE_DEVICES=selected,
                HF_HUB_OFFLINE="1",
                HF_DATASETS_OFFLINE="1",
                VLLM_USE_V2_MODEL_RUNNER="1",
                PYTHONPATH=f"{ROOT / 'benchmarks/hierarchical'}:{ROOT}",
                PATH=f"{ROOT / '.venv/bin'}:{environment.get('PATH', '')}",
            )
            command = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "benchmarks/hierarchical/run_long_draft.py"),
                "--dataset",
                str(DATASET),
                "--output",
                str(destination),
                "--mode",
                mode,
                "--samples",
                "1" if phase == "smoke" else "16",
                "--tokens",
                "64" if phase == "smoke" else "256",
                "--repeats",
                "1" if phase == "smoke" else "2",
            ]
            write_json(output / f"{job}_command.json", command)
            if fingerprint() != source:
                raise RuntimeError("Source changed immediately before launch")
            write_json(
                output / f"{job}_source.json",
                dict(
                    fingerprint=source,
                    commit=command_output(["git", "rev-parse", "HEAD"]).strip(),
                    runtime_diff=command_output(["git", "diff", "HEAD", "--", "vllm"]),
                ),
            )
            with log_path.open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    while child.poll() is None:
                        status("RUNNING", job=job, gpu=selected, child_pid=child.pid)
                        time.sleep(interval)
                        if child.poll() is not None:
                            break
                        if (output / "STOP").exists():
                            raise InterruptedError("Stopped by STOP file")
                        if fingerprint() != source:
                            raise RuntimeError("Source changed during measurement")
                        gpu = next(g for g in gpu_snapshot() if g["uuid"] == selected)
                        owned = descendants(child.pid)
                        external = set(gpu["pids"]) - owned
                        if external:
                            raise RuntimeError(f"GPU interference from PIDs {external}")
                finally:
                    stop_child(child)
                    gpu_lock.close()
            complete = destination / "complete.json"
            if child.returncode or not complete.exists():
                raise RuntimeError(
                    f"{job} failed, exit={child.returncode}; see {log_path}"
                )
            if fingerprint() != source:
                raise RuntimeError("Source changed before result validation")
            if not json.loads(complete.read_text()).get("completed"):
                raise RuntimeError(f"Invalid completion marker for {job}")
            write_json(receipt, dict(completed=True, gpu=selected, exit_code=0))
    status("ACCEPTANCE_QUEUE_COMPLETE", forward_matrix="NOT_IMPLEMENTED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("interval must be 1..60 seconds")
    output = args.output.resolve()
    try:
        run_queue(output, args.interval)
    except BlockingIOError:
        raise SystemExit("A watcher already owns this queue") from None
    except InterruptedError as error:
        write_json(
            output / "status.json",
            dict(state="STOPPED", reason=str(error), watcher_pid=os.getpid()),
        )
    except Exception as error:
        if output.exists():
            write_json(
                output / "status.json",
                dict(state="FAILED", error=str(error), watcher_pid=os.getpid()),
            )
        raise


if __name__ == "__main__":
    main()

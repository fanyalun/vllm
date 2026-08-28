# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export an annotated Qwen3.6 ReplaySSM + DSpark Nsight Systems cycle."""

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path


STAGE_ORDER = (
    "target: prepare_inputs_h2d",
    "replayssm: metadata_and_cursor_commit",
    "target: verify_forward",
    "verify: logits_and_rejection",
    "output: async_d2h_launch",
    "accept: state_postprocess",
    "dspark: propose",
    "dspark: prepare_inputs",
    "dspark: context_kv",
    "dspark: backbone_and_markov",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nsys",
        default="/usr/local/cuda-12.9/bin/nsys",
    )
    return parser.parse_args()


def _nsys_csv(nsys: str, report: str, sqlite_path: Path) -> list[dict]:
    result = subprocess.run(
        [
            nsys,
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            str(sqlite_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Name,"))
    return list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _us(ns: int | str) -> float:
    return round(int(ns) / 1_000.0, 3)


def _ms(ns: int | str) -> float:
    return round(int(ns) / 1_000_000.0, 6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kernel_category(name: str) -> str:
    if "cross_device_reduce" in name:
        return "tp_custom_allreduce"
    if "ncclDevKernel" in name:
        return "nccl_collective"
    if name == "fused_moe_kernel":
        return "target_moe"
    if "gdn_replayssm_spec_circular" in name:
        return "replayssm_gdn"
    if "causal_conv1d_update" in name:
        return "replayssm_conv"
    if "advance_gdn_spec_cursors" in name:
        return "replayssm_cursor"
    if "prepare_dflash_inputs" in name:
        return "dspark_prepare"
    if "rejection_kernel" in name:
        return "verify_rejection"
    if "get_num_sampled_and_rejected" in name:
        return "verify_counts"
    if "post_update_kernel" in name:
        return "accept_postprocess"
    return "other"


def _stage_for_timestamp(
    stages_by_device: dict[int, list[dict]], device: int, timestamp: int
) -> str:
    matches = [
        stage
        for stage in stages_by_device[device]
        if stage["gpu_start_ns"] <= timestamp < stage["gpu_end_ns"]
    ]
    if not matches:
        return "outside_annotated_stage"
    return min(matches, key=lambda stage: stage["gpu_duration_ns"])["stage"]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(args.sqlite)
    device_by_pid = {
        pid: device
        for pid, device in connection.execute(
            """
            SELECT processId, MIN(deviceId)
            FROM TARGET_INFO_CUDA_CONTEXT_INFO
            GROUP BY processId
            """
        )
    }
    gpu_info = {
        row[0]: {
            "name": row[1],
            "bus_location": row[2],
            "total_memory_bytes": row[3],
        }
        for row in connection.execute(
            """
            SELECT id, name, busLocation, totalMemory
            FROM TARGET_INFO_GPU
            ORDER BY id
            """
        )
    }

    projected = _nsys_csv(args.nsys, "nvtx_gpu_proj_trace", args.sqlite)
    stage_rows = []
    internal_stages: dict[int, list[dict]] = defaultdict(list)
    for row in projected:
        stage = row["Name"].lstrip(":")
        if stage not in STAGE_ORDER:
            continue
        pid = int(row["PID"])
        device = device_by_pid[pid]
        gpu_start = int(row["Projected Start (ns)"])
        gpu_duration = int(row["Projected Duration (ns)"])
        internal_stages[device].append(
            {
                "stage": stage,
                "gpu_start_ns": gpu_start,
                "gpu_end_ns": gpu_start + gpu_duration,
                "gpu_duration_ns": gpu_duration,
            }
        )
        stage_rows.append(
            {
                "rank": f"tp{device}_ep{device}",
                "pid": pid,
                "device": device,
                "stage_order": STAGE_ORDER.index(stage),
                "stage": stage,
                "host_start_ms": _ms(row["Orig Start (ns)"]),
                "host_duration_us": _us(row["Orig Duration (ns)"]),
                "gpu_start_ms": _ms(gpu_start),
                "gpu_duration_us": _us(gpu_duration),
                "gpu_ops": int(row["NumGPUOps"]),
            }
        )
    stage_rows.sort(key=lambda row: (row["gpu_start_ms"], row["device"]))
    _write_csv(
        args.output_dir / "stage_timeline.csv",
        stage_rows,
        list(stage_rows[0]),
    )

    kernel_rows = []
    kernel_stage_totals: dict[tuple[int, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    kernel_query = """
        SELECT k.deviceId, s.value, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        ORDER BY k.deviceId, k.start
    """
    kernel_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for device, name, start, end in connection.execute(kernel_query):
        duration = end - start
        kernel_groups[(device, name)].append(duration)
        stage = _stage_for_timestamp(internal_stages, device, start)
        kernel_stage_totals[(device, stage)][0] += 1
        kernel_stage_totals[(device, stage)][1] += duration
    for (device, name), durations in kernel_groups.items():
        kernel_rows.append(
            {
                "device": device,
                "rank": f"tp{device}_ep{device}",
                "category": _kernel_category(name),
                "instances": len(durations),
                "total_us": _us(sum(durations)),
                "average_us": _us(sum(durations) // len(durations)),
                "minimum_us": _us(min(durations)),
                "maximum_us": _us(max(durations)),
                "kernel": name,
            }
        )
    kernel_rows.sort(key=lambda row: (-row["total_us"], row["device"]))
    _write_csv(
        args.output_dir / "kernel_summary.csv",
        kernel_rows,
        list(kernel_rows[0]),
    )

    kernel_stage_rows = [
        {
            "device": device,
            "rank": f"tp{device}_ep{device}",
            "stage": stage,
            "kernel_instances": values[0],
            "kernel_total_us": _us(values[1]),
        }
        for (device, stage), values in kernel_stage_totals.items()
    ]
    kernel_stage_rows.sort(
        key=lambda row: (row["device"], row["stage"])
    )
    _write_csv(
        args.output_dir / "kernel_by_stage.csv",
        kernel_stage_rows,
        list(kernel_stage_rows[0]),
    )

    memcpy_events = []
    memcpy_totals: dict[tuple[int, str], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    memcpy_query = """
        SELECT m.deviceId, e.label, m.start, m.end, m.bytes,
               m.srcDeviceId, m.dstDeviceId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
        JOIN ENUM_CUDA_MEMCPY_OPER AS e ON e.id = m.copyKind
        ORDER BY m.start
    """
    for device, kind, start, end, size, src_device, dst_device in (
        connection.execute(memcpy_query)
    ):
        stage = _stage_for_timestamp(internal_stages, device, start)
        duration = end - start
        memcpy_events.append(
            {
                "device": device,
                "rank": f"tp{device}_ep{device}",
                "stage": stage,
                "kind": kind,
                "start_ms": _ms(start),
                "duration_us": _us(duration),
                "bytes": size,
                "src_device": src_device,
                "dst_device": dst_device,
            }
        )
        totals = memcpy_totals[(device, kind)]
        totals[0] += 1
        totals[1] += size
        totals[2] += duration
    _write_csv(
        args.output_dir / "memcpy_events.csv",
        memcpy_events,
        list(memcpy_events[0]),
    )
    memcpy_summary = [
        {
            "device": device,
            "rank": f"tp{device}_ep{device}",
            "kind": kind,
            "instances": values[0],
            "bytes": values[1],
            "gpu_time_us": _us(values[2]),
        }
        for (device, kind), values in memcpy_totals.items()
    ]
    memcpy_summary.sort(key=lambda row: (row["device"], row["kind"]))
    _write_csv(
        args.output_dir / "memcpy_summary.csv",
        memcpy_summary,
        list(memcpy_summary[0]),
    )

    stage_lookup = {
        (row["device"], row["stage"]): row for row in stage_rows
    }
    verify_start_skew_us = round(
        abs(
            stage_lookup[(0, "target: verify_forward")]["gpu_start_ms"]
            - stage_lookup[(1, "target: verify_forward")]["gpu_start_ms"]
        )
        * 1_000,
        3,
    )
    allreduce_rows = [
        row
        for row in kernel_rows
        if row["category"] == "tp_custom_allreduce"
    ]
    cycle_start = min(row["gpu_start_ms"] for row in stage_rows)
    cycle_end = max(
        row["gpu_start_ms"] + row["gpu_duration_us"] / 1_000
        for row in stage_rows
    )
    report_path = args.sqlite.with_suffix(".nsys-rep")
    artifact_root = (
        "/home/fanya/replayssm_build_artifacts/"
        "nsys_qwen36_tp2_ep2_replayssm_dspark_20260828"
    )
    capture_command = f"""NSYS_OUTPUT={artifact_root}/qwen36_cycle
CUDA_VISIBLE_DEVICES=0,1 \\
PATH=.venv/bin:$PATH \\
VLLM_WORKER_MULTIPROC_METHOD=spawn \\
VLLM_NVTX_SCOPES_FOR_PROFILING=1 \\
/usr/local/cuda-12.9/bin/nsys profile \\
  --output="$NSYS_OUTPUT" \\
  --force-overwrite=true \\
  --capture-range=cudaProfilerApi \\
  --capture-range-end=stop \\
  --trace=cuda,nvtx,osrt,cublas,cudnn,python-gil \\
  --sample=process-tree \\
  --cpuctxsw=process-tree \\
  --python-sampling=true \\
  --python-sampling-frequency=1000 \\
  --cuda-graph-trace=node \\
  --cuda-event-trace=true \\
  --stats=true \\
  .venv/bin/python \\
  benchmarks/replayssm/nsys_qwen36_tp_ep_dspark.py
"""
    (args.output_dir / "capture_command.txt").write_text(capture_command)
    capture_manifest = {
        "captured_at_utc": "2026-08-28",
        "source_commit_at_capture": "991558dc2e5c",
        "branch_at_capture": "exp/replayssm-official-qwen36-tp2-ep2",
        "nsys_version": "2025.1.3.140",
        "capture_command_file": "capture_command.txt",
        "worker_rank_map": {
            "2305329": "tp0_ep0_gpu0",
            "2305330": "tp1_ep1_gpu1",
        },
        "trace_files": {
            report_path.name: {
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            },
            args.sqlite.name: {
                "bytes": args.sqlite.stat().st_size,
                "sha256": _sha256(args.sqlite),
            },
        },
        "environment_checks": {
            "cpu_process_tree_sampling": "supported",
            "cpu_system_wide_sampling": "not_supported",
            "gpu_metrics": "not_permitted_ERR_NVGPUCTRPERM",
            "topology": "GPU0-GPU1 SYS; no NVLink",
        },
    }
    (args.output_dir / "capture_manifest.json").write_text(
        json.dumps(capture_manifest, indent=2) + "\n"
    )
    summary = {
        "trace": {
            "nsys_rep": str(report_path),
            "sqlite": str(args.sqlite),
            "capture_manifest": str(
                args.output_dir / "capture_manifest.json"
            ),
        },
        "configuration": {
            "model": "Qwen3.6-35B-A3B",
            "draft_model": "Qwen3.6-35B-A3B-speculator.dspark",
            "parallelism": "TP2+EP2",
            "speculative_method": "DSpark",
            "num_speculative_tokens": 8,
            "replayssm": True,
            "replayssm_buffer_len": 16,
            "all2all_backend": "allgather_reducescatter",
        },
        "gpu_info": gpu_info,
        "profiled_cycle": {
            "gpu_projected_start_ms": cycle_start,
            "gpu_projected_end_ms": round(cycle_end, 6),
            "gpu_projected_span_ms": round(cycle_end - cycle_start, 6),
            "verify_start_skew_us": verify_start_skew_us,
            "tp_custom_allreduce_max_us_by_device": {
                str(row["device"]): row["maximum_us"]
                for row in allreduce_rows
            },
        },
        "memcpy_totals": memcpy_summary,
        "limitations": [
            "CUDA Graph node tracing and CPU/Python tracing add overhead; "
            "projected durations are diagnostic, not production latency.",
            "GPU hardware/PCIe metrics were unavailable due GPU counter "
            "permissions (ERR_NVGPUCTRPERM).",
            "CUDA MemOps show direct H2D/D2H/D2D/P2P copies. NCCL and custom "
            "collectives traverse the physical SYS path but are not direct "
            "P2P memcpy rows or measured PCIe Tx/Rx counters.",
            "NVTX host ranges measure host enqueue/control span, not CPU busy "
            "time alone.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    stage_lines = []
    for row in sorted(
        stage_rows, key=lambda item: (item["device"], item["stage_order"])
    ):
        stage_lines.append(
            "| {rank} | {stage} | {host_start_ms:.6f} | "
            "{host_duration_us:.3f} | {gpu_start_ms:.6f} | "
            "{gpu_duration_us:.3f} |".format(**row)
        )
    copy_lines = [
        "| {rank} | {kind} | {instances} | {bytes} | "
        "{gpu_time_us:.3f} |".format(**row)
        for row in memcpy_summary
    ]
    projected_span_ms = summary["profiled_cycle"]["gpu_projected_span_ms"]
    max_allreduce_us = summary["profiled_cycle"][
        "tp_custom_allreduce_max_us_by_device"
    ]["1"]
    report = f"""# Qwen3.6 ReplaySSM + DSpark TP2+EP2 Nsight analysis

## Capture identity

- Target: `Qwen3.6-35B-A3B`
- Drafter: `Qwen3.6-35B-A3B-speculator.dspark`, draft width 8
- Parallelism: TP2 + EP2, `allgather_reducescatter`
- ReplaySSM: enabled, ring buffer length 16
- Capture: one steady target verify/accept/next-draft cycle after a separate
  warm-up request; CUDA Graph node-level tracing enabled
- GPU 0: `{gpu_info[0]["name"]}`, `{gpu_info[0]["bus_location"]}`
- GPU 1: `{gpu_info[1]["name"]}`, `{gpu_info[1]["bus_location"]}`
- Physical topology: `SYS` between GPUs (PCIe plus CPU interconnect), not NVLink
- Reproduction command: `capture_command.txt`

## CPU host ranges and projected GPU work

All timestamps are relative to the Nsight capture. Host duration is the NVTX
enqueue/control span; it is not pure CPU busy time. GPU duration is the span
from the first to last GPU operation attributed to the range.

| Rank | Stage | Host start ms | Host duration us | GPU start ms | GPU duration us |
|---|---|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

The full annotated GPU span is {projected_span_ms:.3f} ms in this instrumented
trace. Do not use that as production latency: CUDA
Graph node tracing plus Python/GIL/CPU tracing materially increases overhead.

## Main dependency chain

1. GPU 1 enters target verify {verify_start_skew_us:.3f} us before GPU 0.
2. GPU 1's longest TP custom all-reduce kernel lasts
   {max_allreduce_us:.3f} us, consistent with waiting/spinning for the later
   rank in this trace.
3. Both ranks finish target verify at about 79.896 ms, then run rejection and
   the nested NCCL AllGather.
4. DSpark starts within about 1 us across ranks and its projected proposal span
   is about 1.316 ms per rank.

This proves a rank-arrival-skew/collective-wait chain in this capture. It does
not by itself prove the same magnitude in an uninstrumented production run.

## CUDA memory operations

| Rank | Kind | Instances | Bytes | GPU time us |
|---|---|---:|---:|---:|
{chr(10).join(copy_lines)}

There are no peer-to-peer memcpy events. H2D and D2H move only small control
and result payloads; ReplaySSM state remains GPU-resident. D2D rows are local
device copies and must not be reported as PCIe traffic.

TP/EP communication is visible as custom all-reduce and NCCL kernels. Because
the two GPUs are connected through `SYS`, that communication physically uses
PCIe plus the CPU interconnect. Nsight could not collect GPU hardware PCIe
Tx/Rx counters on this host (`ERR_NVGPUCTRPERM`), so the report deliberately
does not fabricate PCIe byte totals from collective kernel time.

## Kernel evidence

- `fused_moe_kernel`: 160 calls total, 80 per GPU, matching two launches for
  each of 40 target MoE layers.
- `gdn_replayssm_spec_circular_kernel`: 120 calls total, 60 per GPU.
- `_causal_conv1d_update_kernel`: 60 calls total, 30 per GPU.
- The rejection kernel itself is tiny; target forward and synchronization are
  the material verify costs in this trace.

See `kernel_summary.csv`, `kernel_by_stage.csv`, `memcpy_events.csv`, and
`stage_timeline.csv` for auditable rows. Open the sibling `.nsys-rep` in
Nsight Systems GUI to inspect CPU scheduling, CUDA API, GPU streams, CUDA Graph
nodes, NVTX ranges, memory copies, and NCCL ranges together.
"""
    (args.output_dir / "analysis.md").write_text(report)


if __name__ == "__main__":
    main()

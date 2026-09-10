# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import sqlite3

import pytest

from benchmarks.replayssm.analyze_async_draft_profile import (
    analyze,
    critical_path,
    interval_coverage,
)


def test_launch_correlation_is_process_scoped_and_keeps_queued_gpu_work(tmp_path):
    path = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE StringIds(id, value);
        CREATE TABLE PROCESSES(globalPid, pid);
        CREATE TABLE NVTX_EVENTS(start, end, globalTid, text, textId);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start, end, globalTid, correlationId, nameId);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start, end, globalPid, correlationId, deviceId);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start, end, globalPid, correlationId, deviceId);
    """)
    pid = (1 << 48) | (123 << 24)
    connection.execute("INSERT INTO PROCESSES VALUES (?,123)", (pid,))
    connection.execute("INSERT INTO StringIds VALUES (1,'cudaLaunchKernel')")
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES (0,100,?,'async_draft: cache_hit',NULL)",
        (pid | 123,),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10,20,?,7,1)",
        (pid | 123,),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,0)",
        [(200, 250, pid, 7), (400, 900, pid + (1 << 24), 7)],
    )
    connection.commit()
    connection.close()
    analyze(path)
    with (tmp_path / "stage_timeline.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["pid"] == "123"
    assert row["gpu_operations"] == "1"
    assert row["gpu_start_ns"] == "200"
    assert row["gpu_end_ns"] == "250"


def test_partial_capture_is_not_a_valid_critical_path():
    with pytest.raises(ValueError, match="counts differ"):
        critical_path([{"stage": "eagle3: propose"}])


def test_gpu_idle_excludes_overlapping_activity_only_once():
    assert interval_coverage([(0, 20), (15, 40), (60, 90)], 10, 80) == 50

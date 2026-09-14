# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

import pytest

from benchmarks.replayssm import async_ssd_phase_b_throughput as throughput
from benchmarks.replayssm.async_ssd_eagle3_matrix import (
    Cell,
    server_command,
    write_json,
)


def make_experiments(tmp_path: Path) -> tuple[throughput.ModelExperiment, ...]:
    qwen = tmp_path / "qwen"
    qwen_dspark = tmp_path / "qwen_dspark"
    gemma = tmp_path / "gemma"
    gemma_dspark = tmp_path / "gemma_dspark"
    for path in (qwen, qwen_dspark, gemma, gemma_dspark):
        path.mkdir()
    (qwen / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
    (gemma / "config.json").write_text('{"model_type":"gemma4"}', encoding="utf-8")
    dspark_config = {
        "speculators_config": {"proposal_methods": [{"speculative_tokens": 6}]}
    }
    for path in (qwen_dspark, gemma_dspark):
        (path / "config.json").write_text(json.dumps(dspark_config), encoding="utf-8")
    return (
        throughput.ModelExperiment(
            "qwen36_mtp", "Qwen3.6 + MTP", "mtp", str(qwen), None, "qwen36"
        ),
        throughput.ModelExperiment(
            "qwen36_dspark",
            "Qwen3.6 + DSpark",
            "dspark",
            str(qwen),
            str(qwen_dspark),
            "qwen36",
        ),
        throughput.ModelExperiment(
            "gemma4_dspark",
            "Gemma4 + DSpark",
            "dspark",
            str(gemma),
            str(gemma_dspark),
            "gemma4",
        ),
    )


def test_fixed_matrix_has_three_modes_for_three_model_configs(tmp_path) -> None:
    cells = throughput.matrix_cells(make_experiments(tmp_path), 46000)

    assert len(cells) == 9
    assert [cell.mode for cell in cells[:3]] == ["ar", "sync", "async_cache"]
    assert len({cell.key for cell in cells}) == 9
    assert [cell.port for cell in cells] == list(range(46000, 46009))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--num-speculative-tokens", "4"], "requires D=3"),
        (["--output-length", "512"], "requires output_length=128"),
        (["--warmup-seconds", "0"], "positive warmup_seconds"),
    ],
)
def test_cli_rejects_changes_to_the_formal_contract(argv, message) -> None:
    with pytest.raises(SystemExit):
        throughput.parse_args(argv)


def test_run_args_route_replayssm_and_real_async_cache(tmp_path) -> None:
    args = throughput.parse_args([])
    experiments = make_experiments(tmp_path)
    for experiment in experiments:
        run_args = throughput.make_run_args(args, experiment, tmp_path)
        for mode in throughput.DECODE_MODES:
            command = server_command(
                run_args, Cell("performance", mode, "eager", 1), 46000
            )
            assert command[command.index("--dtype") + 1] == "bfloat16"
            if mode != "ar":
                config = json.loads(command[command.index("--speculative-config") + 1])
                assert config["num_speculative_tokens"] == 3
                assert config["method"] == experiment.method
            if experiment.target_family == "qwen36":
                expected = "--use-replayssm" if mode == "ar" else "--use-replayssm-spec"
                assert expected in command
            else:
                assert "--use-replayssm" not in command
                assert "--use-replayssm-spec" not in command
            if mode == "async_cache":
                config = json.loads(command[command.index("--speculative-config") + 1])
                assert config["async_draft_device"] == 1


def write_complete_cell(
    args,
    output_root: Path,
    experiment: throughput.ModelExperiment,
    mode: str,
    port: int,
    value: float,
) -> None:
    cell = Cell("performance", mode, "eager", 1)
    experiment_root = throughput.model_output_root(output_root, experiment)
    cell_dir = experiment_root / "cells" / cell.name
    cell_dir.mkdir(parents=True)
    run_args = throughput.make_run_args(args, experiment, experiment_root)
    metrics = {}
    if mode == "async_cache":
        metrics = {
            'vllm:async_draft_cache_hits_total{engine="0"}': 3,
            'vllm:async_draft_cache_misses_total{engine="0"}': 2,
            'vllm:async_draft_branch_build_seconds_total{engine="0"}': 1,
        }
    result = {
        "status": "complete",
        "warmup": {"seconds": 30.1, "requests": 2, "completion_tokens": 128},
        "summary": {
            "completion_tokens": 2048,
            "completed_request_count": 16,
            "completion_throughput_tok_s": value,
            "tokens_per_gpu_second": value / (2 if mode == "async_cache" else 1),
            "ttft_p50_seconds": 0.1,
            "ttft_p95_seconds": 0.2,
            "tpot_p50_seconds": 0.01,
            "tpot_p95_seconds": 0.02,
        },
        "metrics_delta": metrics,
    }
    write_json(cell_dir / "result.json", result)
    write_json(
        cell_dir / "cell_complete.json",
        {"status": "complete", "cell": cell.name},
    )
    write_json(
        cell_dir / "shutdown.json",
        {"exit_code": 0, "forced_kill": False},
    )
    write_json(cell_dir / "command.json", server_command(run_args, cell, port))


def test_audit_and_plot_require_and_render_all_nine_cells(tmp_path) -> None:
    args = throughput.parse_args([])
    experiments = make_experiments(tmp_path)
    cells = throughput.matrix_cells(experiments, 46000)
    experiment_by_key = {experiment.key: experiment for experiment in experiments}
    for index, matrix_cell in enumerate(cells):
        write_complete_cell(
            args,
            tmp_path / "artifact",
            experiment_by_key[matrix_cell.model_key],
            matrix_cell.mode,
            matrix_cell.port,
            10.0 + index,
        )

    output_root = tmp_path / "artifact"
    assert throughput.audit_matrix(args, output_root, experiments, cells)
    throughput.render_outputs(output_root, experiments)

    assert (output_root / "matrix_complete.json").is_file()
    assert (output_root / "performance_measurement_complete.json").is_file()
    gate = json.loads((output_root / "performance_gate_passed.json").read_text())
    assert gate["status"] == "passed"
    assert len(gate["records"]) == 3
    assert (output_root / "throughput.png").stat().st_size > 0
    assert (output_root / "throughput.svg").stat().st_size > 0
    assert "Qwen3.6 + MTP" in (output_root / "README.md").read_text()
    with (output_root / "throughput.csv").open(newline="", encoding="utf-8") as source:
        assert len(list(csv.DictReader(source))) == 9

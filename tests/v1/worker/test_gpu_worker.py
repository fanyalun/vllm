# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.v1.worker.gpu_worker import Worker


def _make_worker(enforce_eager: bool) -> Worker:
    worker = Worker.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
    )
    worker.compilation_config = SimpleNamespace(
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker.model_config = SimpleNamespace(enforce_eager=enforce_eager, seed=0)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1)
    worker.scheduler_config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=1,
    )
    worker.use_v2_model_runner = False
    worker.model_runner = SimpleNamespace(
        _dummy_run=Mock(return_value=(None, None)),
        maybe_remove_all_loras=Mock(),
        lora_config=None,
        initialize_hybrid_temporal_runtime=Mock(),
        capture_model=Mock(return_value=0),
        model_memory_usage=0,
        is_pooling_model=False,
    )
    return worker


def test_compile_or_warm_up_model_initializes_hybrid_temporal_runtime_for_eager(
    monkeypatch,
):
    worker = _make_worker(enforce_eager=True)

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", Mock())
    monkeypatch.setattr(gpu_worker_module, "set_random_seed", Mock())
    monkeypatch.setattr(
        gpu_worker_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    worker.compile_or_warm_up_model()

    worker.model_runner.initialize_hybrid_temporal_runtime.assert_called_once_with()
    worker.model_runner.capture_model.assert_not_called()


def test_compile_or_warm_up_model_defers_runtime_init_to_capture_when_not_eager(
    monkeypatch,
):
    worker = _make_worker(enforce_eager=False)

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", Mock())
    monkeypatch.setattr(gpu_worker_module, "set_random_seed", Mock())
    monkeypatch.setattr(
        gpu_worker_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=False),
    )

    worker.compile_or_warm_up_model()

    worker.model_runner.initialize_hybrid_temporal_runtime.assert_not_called()
    worker.model_runner.capture_model.assert_called_once_with()

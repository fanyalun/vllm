import json
import os
import time

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager


MODEL = "/home/fanya/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MAMBA_BLOCK_SIZE = 16
LAYER_IDX = 0
PREFIX = f"model.layers.{LAYER_IDX}"
BATCH_SIZES = (32, 64, 128)
DRAFT_LENGTHS = (4,)
WARMUP_ITERS = 10
MEASURE_ITERS = 30


def _ensure_single_process_model_parallel() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=1)


class DummySpecConfig:
    def __init__(self, num_speculative_tokens: int,
                 offload_mode: str = "predict_last") -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self.hybrid_spec_state_offload_mode = offload_mode

    def hybrid_spec_state_offload_enabled(self) -> bool:
        return self.hybrid_spec_state_offload_mode != "disabled"

    def resident_speculative_mamba_blocks(self) -> int:
        if self.hybrid_spec_state_offload_enabled():
            return 0
        return self.num_speculative_tokens


def _make_vllm_config(num_spec: int, max_num_seqs: int) -> VllmConfig:
    model_config = ModelConfig(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    cache_config = CacheConfig(
        block_size=MAMBA_BLOCK_SIZE,
        gpu_memory_utilization=0.5,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_block_size=MAMBA_BLOCK_SIZE,
        mamba_cache_mode="align",
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_seqs * (num_spec + 1),
        max_model_len=4096,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    vllm_config.speculative_config = DummySpecConfig(num_spec)
    return vllm_config


def _move_module_to_device(layer: Qwen3_5DecoderLayer) -> None:
    layer.to(device=DEVICE, dtype=DTYPE)
    layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.to(
        device=DEVICE, dtype=torch.float32
    )
    layer.linear_attn.dt_bias.data = layer.linear_attn.dt_bias.data.to(
        device=DEVICE, dtype=torch.float32
    )
    layer.mlp.experts.quant_method.process_weights_after_loading(layer.mlp.experts)


def _build_layer_and_buffers(num_spec: int, batch_size: int) -> dict:
    vllm_config = _make_vllm_config(num_spec, batch_size)
    with set_current_vllm_config(vllm_config):
        hf_config = vllm_config.model_config.hf_text_config
        layer_type = hf_config.layer_types[LAYER_IDX]
        if layer_type != "linear_attention":
            raise ValueError(
                f"Layer {LAYER_IDX} is {layer_type}, expected linear_attention"
            )
        layer = Qwen3_5DecoderLayer(
            vllm_config=vllm_config,
            layer_type=layer_type,
            prefix=PREFIX,
        )
    _move_module_to_device(layer)
    layer.eval()

    conv_shape, temporal_shape = layer.linear_attn.get_state_shape()
    conv_dtype, temporal_dtype = layer.linear_attn.get_state_dtype()
    conv_state = torch.zeros(
        (batch_size, *conv_shape), dtype=conv_dtype, device=DEVICE
    )
    ssm_state = torch.zeros(
        (batch_size, *temporal_shape), dtype=temporal_dtype, device=DEVICE
    )
    layer.linear_attn.kv_cache = (conv_state, ssm_state)

    q_len = num_spec + 1
    num_tokens = batch_size * q_len
    hidden_states = torch.randn(
        num_tokens, hf_config.hidden_size, dtype=DTYPE, device=DEVICE
    )
    linear_attn_output = torch.empty_like(hidden_states)

    spec_query_start_loc_cpu = torch.arange(
        0, num_tokens + 1, q_len, dtype=torch.int32
    )
    spec_query_start_loc = spec_query_start_loc_cpu.to(device=DEVICE)
    spec_state_indices = torch.arange(batch_size, dtype=torch.int32).view(-1, 1)
    spec_state_indices_cuda = spec_state_indices.to(device=DEVICE)
    spec_token_indx = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    non_spec_token_indx = torch.empty(0, dtype=torch.int32, device=DEVICE)
    spec_sequence_masks = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    non_spec_query_start_loc = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    non_spec_state_indices = torch.empty(0, dtype=torch.int32, device=DEVICE)
    accepted = torch.full((batch_size,), q_len, dtype=torch.int32, device=DEVICE)
    reload_mode_cpu = torch.zeros(batch_size, dtype=torch.int32)
    reload_slot_cpu = torch.zeros(batch_size, dtype=torch.int32)
    reload_generation_cpu = torch.zeros(batch_size, dtype=torch.int32)
    req_indices_cpu = torch.arange(batch_size, dtype=torch.int32)
    pred_accept_cpu = torch.full((batch_size,), q_len, dtype=torch.int32)

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=batch_size,
        num_spec_decode_tokens=num_tokens,
        num_actual_tokens=num_tokens,
        has_initial_state=None,
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=non_spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_cuda,
        non_spec_state_indices_tensor=non_spec_state_indices,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=accepted,
        spec_max_query_len=q_len,
        spec_query_start_loc_cpu=spec_query_start_loc_cpu,
        spec_req_indices_cpu=req_indices_cpu,
        non_spec_req_indices_cpu=None,
        predicted_accept_len_cpu=pred_accept_cpu,
        temporal_reload_mode_cpu=reload_mode_cpu,
        reload_slot_cpu=reload_slot_cpu,
        reload_generation_cpu=reload_generation_cpu,
        chunk_indices=None,
        chunk_offsets=None,
        nums_dict=None,
        batch_ptr=None,
        token_chunk_offset_ptr=None,
    )

    forward_context = create_forward_context(
        attn_metadata={f"{PREFIX}.linear_attn": metadata},
        vllm_config=vllm_config,
        slot_mapping={},
        skip_compiled=True,
    )

    return {
        "layer": layer,
        "hidden_states": hidden_states,
        "linear_attn_output": linear_attn_output,
        "forward_context": forward_context,
        "temporal_shape": temporal_shape,
        "temporal_dtype": temporal_dtype,
    }


def _measure_copy_ms(batch_size: int, temporal_shape: tuple[int, ...],
                     temporal_dtype: torch.dtype) -> dict:
    cpu = torch.empty(
        (batch_size, *temporal_shape),
        dtype=temporal_dtype,
        device="cpu",
        pin_memory=True,
    )
    gpu = torch.empty(
        (batch_size, *temporal_shape),
        dtype=temporal_dtype,
        device=DEVICE,
    )

    for _ in range(WARMUP_ITERS):
        gpu.copy_(cpu, non_blocking=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(MEASURE_ITERS):
        start.record()
        gpu.copy_(cpu, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    avg_ms = total_ms / MEASURE_ITERS
    total_bytes = gpu.numel() * gpu.element_size()
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return {
        "t_load_ms": avg_ms,
        "copy_total_bytes": total_bytes,
        "copy_gbps": gbps,
    }


def _measure_linear_attn_ms(
    layer: Qwen3_5DecoderLayer,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    forward_context,
) -> float:
    with override_forward_context(forward_context):
        for _ in range(WARMUP_ITERS):
            layer.linear_attn(hidden_states, output)
        torch.cuda.synchronize()

        total_s = 0.0
        for _ in range(MEASURE_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            layer.linear_attn(hidden_states, output)
            torch.cuda.synchronize()
            total_s += time.perf_counter() - start
    return total_s * 1000.0 / MEASURE_ITERS


def _measure_decoder_block_ms(
    layer: Qwen3_5DecoderLayer,
    hidden_states: torch.Tensor,
    forward_context,
) -> float:
    with override_forward_context(forward_context):
        for _ in range(WARMUP_ITERS):
            layer(hidden_states=hidden_states, residual=None, positions=None)
        torch.cuda.synchronize()

        total_s = 0.0
        for _ in range(MEASURE_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            layer(hidden_states=hidden_states, residual=None, positions=None)
            torch.cuda.synchronize()
            total_s += time.perf_counter() - start
    return total_s * 1000.0 / MEASURE_ITERS


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    _ensure_single_process_model_parallel()
    init_workspace_manager(torch.device(DEVICE))
    results = []

    for draft_len in DRAFT_LENGTHS:
        for batch_size in BATCH_SIZES:
            built = _build_layer_and_buffers(draft_len, batch_size)
            copy_stats = _measure_copy_ms(
                batch_size,
                built["temporal_shape"],
                built["temporal_dtype"],
            )
            gdn_ms = _measure_linear_attn_ms(
                built["layer"],
                built["hidden_states"],
                built["linear_attn_output"],
                built["forward_context"],
            )
            block_ms = _measure_decoder_block_ms(
                built["layer"],
                built["hidden_states"],
                built["forward_context"],
            )
            results.append({
                "batch_size": batch_size,
                "draft_length": draft_len,
                "layer_idx": LAYER_IDX,
                "temporal_shape": list(built["temporal_shape"]),
                "temporal_dtype": str(built["temporal_dtype"]),
                "t_gdn_ms": gdn_ms,
                "t_comp_ms": block_ms,
                "gdn_over_load_ratio": gdn_ms / copy_stats["t_load_ms"],
                "comp_over_load_ratio": block_ms / copy_stats["t_load_ms"],
                **copy_stats,
            })
            del built
            torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

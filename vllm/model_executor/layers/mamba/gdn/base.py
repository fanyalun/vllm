# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PretrainedConfig

from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.worker.workspace import current_workspace_manager


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """Base class for GatedDeltaNet attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.hybrid_spec_state_offload_enabled = bool(
            self.speculative_config is not None
            and self.speculative_config.hybrid_spec_state_offload_enabled()
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_hybrid_temporal_scratch_spec(
        self, max_rows: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        state_shapes = tuple(self.get_state_shape())
        assert len(state_shapes) >= 2
        _, temporal_state_dtype = self.get_state_dtype()
        return (max_rows, *state_shapes[1]), temporal_state_dtype

    def reserve_hybrid_temporal_scratch(
        self, max_num_reqs: int
    ) -> tuple[tuple[int, ...], torch.dtype]:
        spec = self.get_hybrid_temporal_scratch_spec(
            max_num_reqs * (self.num_spec + 1)
        )
        current_workspace_manager().reserve_simultaneous_for_all_ubatches(spec)
        return spec

    def acquire_hybrid_temporal_scratch(self, num_rows: int) -> torch.Tensor:
        spec = self.get_hybrid_temporal_scratch_spec(num_rows)
        (scratch,) = current_workspace_manager().get_simultaneous(spec)
        return scratch

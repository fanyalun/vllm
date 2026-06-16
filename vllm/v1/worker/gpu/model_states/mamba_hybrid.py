# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.hybrid_spec_offload import HybridSpecReloadMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


def init_hybrid_predicted_accept_len(num_spec_tokens: int) -> int:
    return 1 + num_spec_tokens


def update_hybrid_accepted_len_ewma(
    prev_ewma: float,
    accepted_len: int,
    alpha: float,
) -> float:
    return alpha * accepted_len + (1.0 - alpha) * prev_ewma


def predict_hybrid_accept_len(ewma: float, num_spec_tokens: int) -> int:
    return int(
        max(1, min(1 + num_spec_tokens, round(float(ewma))))
    )


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None
    spec_req_indices_cpu: torch.Tensor | None = None
    predicted_accept_len_cpu: torch.Tensor | None = None
    temporal_reload_mode_cpu: torch.Tensor | None = None
    reload_slot_cpu: torch.Tensor | None = None
    reload_generation_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
            "spec_req_indices_cpu": None
            if self.spec_req_indices_cpu is None
            else self.spec_req_indices_cpu[:num_reqs],
            "predicted_accept_len_cpu": None
            if self.predicted_accept_len_cpu is None
            else self.predicted_accept_len_cpu[:num_reqs],
            "temporal_reload_mode_cpu": None
            if self.temporal_reload_mode_cpu is None
            else self.temporal_reload_mode_cpu[:num_reqs],
            "reload_slot_cpu": None
            if self.reload_slot_cpu is None
            else self.reload_slot_cpu[:num_reqs],
            "reload_generation_cpu": None
            if self.reload_generation_cpu is None
            else self.reload_generation_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.speculative_config = vllm_config.speculative_config
        self.num_spec_tokens = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config is not None
            else 0
        )
        self.hybrid_spec_state_offload_enabled = bool(
            self.speculative_config is not None
            and self.speculative_config.hybrid_spec_state_offload_enabled()
        )
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        if self.hybrid_spec_state_offload_enabled:
            assert self.speculative_config is not None
            init_pred = init_hybrid_predicted_accept_len(self.num_spec_tokens)
            self.hybrid_spec_state_ewma = torch.full(
                (self.max_num_reqs,), float(init_pred), dtype=torch.float32
            )
            self.predicted_accept_len_cpu = torch.full(
                (self.max_num_reqs,), init_pred, dtype=torch.int32
            )
            self.reload_required_cpu = torch.zeros(
                self.max_num_reqs, dtype=torch.bool
            )
            self.reload_mode_cpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32
            )
            self.reload_slot_cpu = torch.zeros(self.max_num_reqs, dtype=torch.int32)
            self.reload_generation_cpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32
            )
            self.hybrid_spec_state_ewma_alpha = (
                self.speculative_config.hybrid_spec_state_ewma_alpha
            )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        if self.hybrid_spec_state_offload_enabled:
            init_pred = init_hybrid_predicted_accept_len(self.num_spec_tokens)
            self.hybrid_spec_state_ewma[req_index] = float(init_pred)
            self.predicted_accept_len_cpu[req_index] = init_pred
            self.reload_required_cpu[req_index] = False
            self.reload_mode_cpu[req_index] = int(HybridSpecReloadMode.NONE)
            self.reload_slot_cpu[req_index] = 0
            self.reload_generation_cpu[req_index] = 0

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        spec_req_indices_cpu = None
        predicted_accept_len_cpu = None
        temporal_reload_mode_cpu = None
        reload_slot_cpu = None
        reload_generation_cpu = None
        if not for_capture:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            if input_batch.num_draft_tokens_per_req is not None:
                spec_decode_mask = (
                    input_batch.num_draft_tokens_per_req > 0
                ) & ~input_batch.is_prefilling_np
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    input_batch.num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)
            if self.hybrid_spec_state_offload_enabled:
                req_indices_np = np.full(num_reqs, -1, dtype=np.int32)
                req_indices_np[: input_batch.num_reqs] = input_batch.idx_mapping_np
                spec_req_indices_cpu = torch.from_numpy(req_indices_np)

                predicted_accept_len_np = np.ones(num_reqs, dtype=np.int32)
                temporal_reload_mode_np = np.zeros(num_reqs, dtype=np.int32)
                reload_slot_np = np.zeros(num_reqs, dtype=np.int32)
                reload_generation_np = np.zeros(num_reqs, dtype=np.int32)
                for batch_idx, req_idx in enumerate(input_batch.idx_mapping_np):
                    predicted_len = int(self.predicted_accept_len_cpu[req_idx].item())
                    if input_batch.num_draft_tokens_per_req is not None:
                        max_accept_len = (
                            1 + int(input_batch.num_draft_tokens_per_req[batch_idx])
                        )
                    else:
                        max_accept_len = 1
                    predicted_accept_len_np[batch_idx] = min(
                        predicted_len, max_accept_len
                    )
                    temporal_reload_mode_np[batch_idx] = int(
                        self.reload_mode_cpu[req_idx].item()
                    )
                    reload_slot_np[batch_idx] = int(
                        self.reload_slot_cpu[req_idx].item()
                    )
                    reload_generation_np[batch_idx] = int(
                        self.reload_generation_cpu[req_idx].item()
                    )
                predicted_accept_len_cpu = torch.from_numpy(predicted_accept_len_np)
                temporal_reload_mode_cpu = torch.from_numpy(
                    temporal_reload_mode_np
                )
                reload_slot_cpu = torch.from_numpy(reload_slot_np)
                reload_generation_cpu = torch.from_numpy(reload_generation_np)

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            spec_req_indices_cpu=spec_req_indices_cpu,
            predicted_accept_len_cpu=predicted_accept_len_cpu,
            temporal_reload_mode_cpu=temporal_reload_mode_cpu,
            reload_slot_cpu=reload_slot_cpu,
            reload_generation_cpu=reload_generation_cpu,
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
        )

    def postprocess_state(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        accepted_lens = torch.clamp(num_sampled, min=1)
        self.num_accepted_tokens_gpu[input_batch.idx_mapping] = accepted_lens
        if not self.hybrid_spec_state_offload_enabled:
            return

        spec_decode_mask = np.zeros(input_batch.num_reqs, dtype=bool)
        if input_batch.num_draft_tokens_per_req is not None:
            spec_decode_mask = (
                input_batch.num_draft_tokens_per_req > 0
            ) & ~input_batch.is_prefilling_np

        for batch_idx, req_idx in enumerate(input_batch.idx_mapping_np):
            if not spec_decode_mask[batch_idx]:
                self.reload_required_cpu[req_idx] = False
                self.reload_mode_cpu[req_idx] = int(HybridSpecReloadMode.NONE)
                continue
            accepted_len = int(accepted_lens[batch_idx].item())
            predicted_len = int(self.predicted_accept_len_cpu[req_idx].item())
            self.reload_required_cpu[req_idx] = predicted_len != accepted_len
            self.reload_slot_cpu[req_idx] = accepted_len - 1
            if self.reload_required_cpu[req_idx]:
                self.reload_generation_cpu[req_idx] += 1
                self.reload_mode_cpu[req_idx] = int(
                    HybridSpecReloadMode.CPU_SHADOW
                )
            else:
                self.reload_mode_cpu[req_idx] = int(HybridSpecReloadMode.NONE)
            ewma = update_hybrid_accepted_len_ewma(
                float(self.hybrid_spec_state_ewma[req_idx].item()),
                accepted_len,
                self.hybrid_spec_state_ewma_alpha,
            )
            self.hybrid_spec_state_ewma[req_idx] = ewma
            self.predicted_accept_len_cpu[req_idx] = predict_hybrid_accept_len(
                ewma, self.num_spec_tokens
            )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    get_dspark_proposal_bank_width,
    load_dspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Output width is D, including in the asynchronous child. Explicit
        # branch prefixes extend only the eager query, never the output width.
        bank_width = get_dspark_proposal_bank_width(self.draft_model_config.hf_config)
        if bank_width < self.num_speculative_steps:
            raise ValueError(
                "DSpark proposal bank width must cover the configured execution "
                f"width: bank={bank_width}, configured={self.num_speculative_steps}"
            )
        self.proposal_bank_width = bank_width
        logger.info(
            "DSpark execution width=%d, checkpoint proposal bank width=%d",
            self.num_speculative_steps,
            self.proposal_bank_width,
        )

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps
        self._max_execution_width = self.num_speculative_steps
        self._request_indices = torch.arange(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self._anchor_indices_by_query_width = {
            self.num_query_per_req: self._request_indices * self.num_query_per_req
        }

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self.dflash_causal = False

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = self._anchor_indices_by_query_width[self.num_query_per_req]
        self._async_base_logits_only = False

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        self._trace_top2_values: torch.Tensor | None = None
        self._trace_top2_ids: torch.Tensor | None = None
        self._trace_base_top2_values: torch.Tensor | None = None
        self._trace_base_top2_ids: torch.Tensor | None = None
        if os.environ.get("REPLAYSSM_SPEC_DECODE_TRACE_LOGITS") == "1":
            trace_shape = (self.max_num_reqs, self.num_speculative_steps, 2)
            self._trace_top2_values = torch.empty(
                trace_shape, dtype=torch.float32, device=device
            )
            self._trace_top2_ids = torch.empty(
                trace_shape, dtype=torch.int64, device=device
            )
            self._trace_base_top2_values = torch.empty(
                trace_shape, dtype=torch.float32, device=device
            )
            self._trace_base_top2_ids = torch.empty(
                trace_shape, dtype=torch.int64, device=device
            )

    def branch_query(self, prefix: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Build a query whose last prefix token seeds D predictions."""
        if prefix.ndim != 2 or prefix.shape[1] < 1:
            raise ValueError("DSpark branch prefix must be a nonempty token matrix")
        sample_start = prefix.shape[1] - int(self.sample_from_anchor)
        query = prefix.new_full(
            (prefix.shape[0], sample_start + self.num_speculative_steps),
            self.parallel_drafting_token_id,
        )
        query[:, : prefix.shape[1]] = prefix
        return query, sample_start

    @torch.inference_mode()
    def branch_query_logits(
        self,
        slots: torch.Tensor,
        query: torch.Tensor,
        anchor_positions: torch.Tensor,
        sample_start: int,
        sample_count: int,
    ) -> torch.Tensor:
        """Run an eager provisional query on caller-owned private KV pages."""
        num_reqs, query_width = query.shape
        num_tokens = query.numel()
        if num_reqs > self.max_num_reqs or num_tokens > self.max_num_tokens:
            raise ValueError("DSpark branch query exceeds temporary buffer capacity")
        if not 0 <= sample_start < sample_start + sample_count <= query_width:
            raise ValueError("DSpark branch sample positions are outside the query")
        seq_lens = anchor_positions + query_width
        self.draft_max_seq_len = int(seq_lens.max().item())
        if self.draft_max_seq_len > self.max_model_len:
            raise ValueError("DSpark branch query exceeds maximum model length")
        positions = anchor_positions[:, None] + torch.arange(
            query_width, device=query.device
        )
        self.input_buffers.input_ids[:num_tokens].copy_(query.flatten())
        self.input_buffers.positions[:num_tokens].copy_(positions.flatten())
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            torch.arange(num_reqs + 1, device=query.device) * query_width
        )
        self.input_buffers.seq_lens[:num_reqs].copy_(seq_lens)
        self.block_tables.gather_block_tables(slots, num_reqs)
        slot_mappings = self.block_tables.compute_slot_mappings(
            slots,
            self.input_buffers.query_start_loc[: num_reqs + 1],
            self.input_buffers.positions[:num_tokens],
            num_tokens,
        )
        metadata = DraftModelSpeculator._build_draft_attn_metadata(
            self, num_reqs, num_reqs, num_tokens, query_width, causal=False
        )
        hidden = self._run_model(
            num_tokens,
            metadata,
            build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config),
            None,
        )
        selected = hidden.view(num_reqs, query_width, -1)[
            :, sample_start : sample_start + sample_count
        ]
        return self.model.compute_draft_logits(
            selected.reshape(num_reqs * sample_count, -1)
        ).view(num_reqs, sample_count, -1)

    def set_execution_width(self, width: int) -> None:
        """Select a DSpark execution width within the allocated maximum."""
        if width <= 0 or width > self._max_execution_width:
            raise ValueError(
                "DSpark execution width must be within the allocated maximum: "
                f"width={width}, maximum={self._max_execution_width}"
            )
        self.num_speculative_steps = width
        self.num_query_per_req = width if self.sample_from_anchor else width + 1
        self._anchor_idx = self.anchor_indices(self.max_num_reqs)

    def reserve_execution_width(self, width: int) -> None:
        """Grow private eager buffers without changing the configured width."""
        if width <= self._max_execution_width:
            return
        if width > self.proposal_bank_width:
            raise ValueError(
                "DSpark reserved width exceeds the checkpoint proposal width: "
                f"width={width}, checkpoint={self.proposal_bank_width}"
            )
        if self.draft_logits is not None:
            raise ValueError(
                "Dynamic-width DSpark buffers currently require greedy drafting"
            )

        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            width,
            dtype=torch.int64,
            device=self.device,
        )
        max_num_sampled_tokens = self.max_num_reqs * width
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=self.device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=self.device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=self.device
        )
        self.sample_col = torch.arange(
            width, dtype=torch.int32, device=self.device
        ).repeat(self.max_num_reqs)
        self._step_cols = torch.arange(width, dtype=torch.int32, device=self.device)
        if self._trace_top2_values is not None:
            self._trace_top2_values = torch.empty(
                self.max_num_reqs,
                width,
                2,
                dtype=torch.float32,
                device=self.device,
            )
            self._trace_top2_ids = torch.empty(
                self.max_num_reqs,
                width,
                2,
                dtype=torch.int64,
                device=self.device,
            )
            self._trace_base_top2_values = torch.empty(
                self.max_num_reqs,
                width,
                2,
                dtype=torch.float32,
                device=self.device,
            )
            self._trace_base_top2_ids = torch.empty(
                self.max_num_reqs,
                width,
                2,
                dtype=torch.int64,
                device=self.device,
            )
        self._max_execution_width = width

    def anchor_indices(
        self,
        num_reqs: int,
        *,
        execution_width: int | None = None,
    ) -> torch.Tensor:
        """Return anchor positions for the requested DSpark execution width."""
        width = (
            self.num_speculative_steps if execution_width is None else execution_width
        )
        if width <= 0 or width > self._max_execution_width:
            raise ValueError(
                "DSpark anchor width must be within the allocated maximum: "
                f"width={width}, maximum={self._max_execution_width}"
            )
        query_width = width if self.sample_from_anchor else width + 1
        indices = self._anchor_indices_by_query_width.get(query_width)
        if indices is None:
            indices = self._request_indices * query_width
            self._anchor_indices_by_query_width[query_width] = indices
        return indices[:num_reqs]

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        if self._trace_base_top2_values is not None:
            assert self._trace_base_top2_ids is not None
            base_values, base_ids = base_logits.float().topk(2, dim=-1)
            self._trace_base_top2_values[:num_reqs, :n_spec].copy_(base_values)
            self._trace_base_top2_ids[:num_reqs, :n_spec].copy_(
                self.model.map_draft_to_target(base_ids)
            )
        async_base_logits = getattr(self, "_async_base_logits", None)
        if async_base_logits is not None:
            async_base_logits[:num_reqs, :n_spec].copy_(base_logits)
        if self._async_base_logits_only:
            return

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            if self._trace_top2_values is not None:
                assert self._trace_top2_ids is not None
                top_values, top_ids = logits_i.float().topk(2, dim=-1)
                self._trace_top2_values[:num_reqs, i].copy_(top_values)
                self._trace_top2_ids[:num_reqs, i].copy_(
                    self.model.map_draft_to_target(top_ids)
                )
            async_candidates = getattr(self, "_async_candidate_ids", None)
            if async_candidates is not None:
                top_ids = logits_i.topk(async_candidates.shape[-1] + 1, dim=-1).indices
                async_candidates[:num_reqs, i].copy_(
                    self.model.map_draft_to_target(top_ids[:, 1:])
                )
            if self.draft_logits is not None:
                # Probabilistic: sample in target vocab (a reduced draft vocab is
                # scattered into its target columns; full vocab is already there).
                if self._d2t_scatter_index is not None:
                    assert self._draft_scatter_buf is not None
                    buf = self._draft_scatter_buf[:num_reqs]
                    buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                    logits_i = buf
                # sample_pos is the predicted token's position Q; the target
                # verifies it with the predecessor's Gumbel key (Q-1). Pass Q-1.
                draft_sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    self.temperature,
                    self.seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self._step_cols[i],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                draft_sampled_i = self.model.map_draft_to_target(
                    logits_i.argmax(dim=-1)
                )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

    def proposal_trace_metadata(self, num_reqs: int) -> list[dict[str, Any]]:
        if self._trace_top2_values is None:
            return super().proposal_trace_metadata(num_reqs)
        assert self._trace_top2_ids is not None
        width = getattr(self, "num_speculative_steps", self._trace_top2_values.shape[1])
        values = self._trace_top2_values[:num_reqs, :width].cpu().tolist()
        ids = self._trace_top2_ids[:num_reqs, :width].cpu().tolist()
        execution_metadata = self.proposal_execution_trace_metadata(num_reqs)
        return [
            {
                "draft_top2": [
                    {
                        "token_ids": step_ids,
                        "logits": step_values,
                        "gap": step_values[0] - step_values[1],
                    }
                    for step_ids, step_values in zip(request_ids, request_values)
                ],
                **execution_metadata[index],
            }
            for index, (request_ids, request_values) in enumerate(zip(ids, values))
        ]

    def proposal_execution_trace_metadata(self, num_reqs: int) -> list[dict[str, Any]]:
        trace_base_values = getattr(self, "_trace_base_top2_values", None)
        if trace_base_values is None:
            return [{} for _ in range(num_reqs)]
        assert self._trace_base_top2_ids is not None
        width = self.num_speculative_steps
        values = trace_base_values[:num_reqs, :width].cpu().tolist()
        ids = self._trace_base_top2_ids[:num_reqs, :width].cpu().tolist()
        anchors = (
            self.input_buffers.input_ids[self.anchor_indices(num_reqs)].cpu().tolist()
        )
        query_width = self.num_query_per_req
        query_ids = (
            self.input_buffers.input_ids[: num_reqs * query_width]
            .view(num_reqs, query_width)
            .cpu()
            .tolist()
        )
        query_positions = (
            self.input_buffers.positions[: num_reqs * query_width]
            .view(num_reqs, query_width)
            .cpu()
            .tolist()
        )
        return [
            {
                "draft_base_top2": [
                    {
                        "token_ids": step_ids,
                        "logits": step_values,
                    }
                    for step_ids, step_values in zip(request_ids, request_values)
                ],
                "dspark_anchor_token": anchors[index],
                "dspark_query_input_ids": query_ids[index],
                "dspark_query_positions": query_positions[index],
            }
            for index, (request_ids, request_values) in enumerate(zip(ids, values))
        ]

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)

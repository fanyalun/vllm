# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1 quality probes using private V2 state and unchanged Target verification."""

import copy
import inspect

import torch
from batch_worker import BatchWorker

from vllm.v1.worker.gpu.spec_decode.hierarchical.state import PreverifyState


class _DraftLimitReached(Exception):
    pass


class LongDraftWorker(BatchWorker):
    def setup_long_draft(self, mode):
        if mode not in ("two_level_fixed", "three_level_balanced"):
            raise ValueError(mode)
        spec = self.model_runner.speculator
        assert spec.max_num_reqs == 1 and spec.depth == 4
        assert spec.capacity >= 36
        assert all(layer.enable_mean_preverify for layer in spec.state.layers.values())
        spec.config = copy.copy(spec.config)
        spec.config.preverify_gdn_update_policy = "windowed_three_level"
        spec.config.preverify_gdn_mode_window_size = 1
        spec.config.hierarchical_stop_policy = "balanced"
        spec.state = PreverifyState(
            spec.model,
            5,
            spec.device,
            "replay_tail",
            "windowed_three_level",
            window_size=1,
        )
        spec.preverify_graphs = {}
        spec.rounds = 32
        original = spec.propose
        signature = inspect.signature(original)
        small_propose = spec.small.propose

        def limited_small(*args, **kwargs):
            if sum(row["emitted"] for row in spec.last_trace) >= 32:
                raise _DraftLimitReached
            return small_propose(*args, **kwargs)

        if mode == "three_level_balanced":
            spec.small.propose = limited_small

        def propose(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            batch = values["input_batch"]
            if (
                any(
                    values[name]
                    for name in ("dummy_run", "is_profile", "skip_attn_for_dummy_run")
                )
                or not batch.num_reqs
                or all(req.startswith("_warmup_") for req in batch.req_ids)
            ):
                return original(*args, **kwargs)
            if mode == "three_level_balanced":
                try:
                    result = original(*args, **kwargs)
                except _DraftLimitReached:
                    count = sum(row["emitted"] for row in spec.last_trace)
                    spec.draft_lengths.fill_(count)
                    spec.pending_req_id = batch.req_ids[0]
                    spec.state.invalidate()
                    result = spec.draft_tokens
                length = min(32, int(spec.draft_lengths[0]))
                spec.draft_lengths.fill_(length)
                spec.draft_tokens[:, length:] = -1
                if spec.last_trace:
                    spec.last_trace[-1]["delivery_limit"] = length == 32
                return result
            return self._two_level_propose(spec, values)

        spec.propose = propose
        self.setup_batch_measurement()

    @torch.inference_mode()
    def _two_level_propose(self, spec, values):
        batch = values["input_batch"]
        assert batch.num_reqs == 1
        assert not batch.has_structured_output_reqs and values["mm_inputs"] is None
        spec.draft_tokens.fill_(-1)
        spec.draft_lengths.zero_()
        spec.last_trace = []
        if int(values["num_sampled"][0]) == 0:
            return spec.draft_tokens
        tables = spec.block_tables.gather_block_tables(batch.idx_mapping, 1)
        spec.state.begin(spec.model_state, batch, tables, spec.kv_cache_config)
        position = int((batch.seq_lens - values["num_rejected"])[0])
        anchor = values["last_sampled"][batch.idx_mapping, 0].reshape(-1).clone()
        length = min(32, spec.vllm_config.model_config.max_model_len - position)
        for offset in range(length):
            current, metadata, slots = spec._batch(batch, position + offset, anchor)
            predictions, _, _ = spec._verify(current, metadata, slots)
            anchor = predictions[:1].clone()
            spec.draft_tokens[0, offset] = anchor[0]
            spec.last_trace.append(
                dict(
                    inner_round=offset,
                    proposed=0,
                    accepted=0,
                    emitted=1,
                    offset=offset,
                    stop_reason=2 if offset == length - 1 else 0,
                    probe_mode="two_level_fixed",
                )
            )
            if offset + 1 < length:
                spec.state.advance(0)
        spec.draft_lengths.fill_(length)
        spec.pending_req_id = batch.req_ids[0]
        spec.state.invalidate()
        return spec.draft_tokens

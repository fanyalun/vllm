# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact active Gemma MTP requests between hierarchical inner rounds."""

from dataclasses import replace

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.spec_decode.hierarchical.speculator import should_stop_inner


def make_batch(spec, template, rows, positions, tokens):
    n, width = tokens.shape
    total = n * width
    device = tokens.device
    row_ids = torch.tensor(rows, dtype=torch.int64, device=device)
    mapping = template.idx_mapping[row_ids]
    buffers = spec.buffers
    starts = np.arange(n + 1, dtype=np.int32) * width
    buffers.input_ids[:total].copy_(tokens.flatten())
    buffers.positions[:total].copy_(
        (
            torch.tensor(positions, device=device)[:, None]
            + torch.arange(width, device=device)[None, :]
        ).flatten()
    )
    buffers.is_padding[:total].zero_()
    buffers.query_start_loc[: n + 1].copy_(torch.from_numpy(starts))
    buffers.seq_lens[:n].copy_(torch.tensor(positions, device=device) + width)
    batch = replace(
        template,
        req_ids=[template.req_ids[i] for i in rows],
        num_reqs=n,
        num_reqs_after_padding=n,
        idx_mapping=mapping,
        idx_mapping_np=template.idx_mapping_np[rows],
        expanded_idx_mapping=mapping.repeat_interleave(width),
        expanded_local_pos=torch.arange(width, device=device, dtype=torch.int32).repeat(
            n
        ),
        num_scheduled_tokens=np.full(n, width, dtype=np.int32),
        num_tokens=total,
        num_tokens_after_padding=total,
        num_draft_tokens=n * (width - 1),
        num_draft_tokens_per_req=np.full(n, width - 1, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: n + 1],
        query_start_loc_np=starts,
        seq_lens=buffers.seq_lens[:n],
        seq_lens_cpu_upper_bound=torch.full(
            (n,), spec.vllm_config.model_config.max_model_len, dtype=torch.int32
        ),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array(positions, dtype=np.int32),
        prefill_len_np=np.zeros(n, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(n, dtype=np.int32),
        is_prefilling_np=np.zeros(n, dtype=np.bool_),
        has_prefill=False,
        input_ids=buffers.input_ids[:total],
        positions=buffers.positions[:total],
        is_padding=buffers.is_padding[:total],
        logits_indices=torch.arange(total, device=device, dtype=torch.int64),
        cu_num_logits=buffers.query_start_loc[: n + 1],
        cu_num_logits_np=starts,
        has_structured_output_reqs=False,
        prompt_lens=None,
        max_query_len=width,
    )
    tables = spec.block_tables.gather_block_tables(mapping, n)
    slots = spec.block_tables.compute_slot_mappings(
        mapping, batch.query_start_loc, batch.positions, total
    )
    metadata = spec.model_state.prepare_attn(
        batch, CUDAGraphMode.NONE, tables, slots, spec.attn_groups, spec.kv_cache_config
    )
    return batch, metadata, build_slot_mappings_by_layer(slots, spec.kv_cache_config)


def propose_gemma(
    spec,
    original,
    metadata,
    slots,
    hidden,
    sampled,
    rejected,
    last_sampled,
    next_prefill_tokens,
    temperature,
    seeds,
):
    if original.has_structured_output_reqs:
        raise ValueError("hierarchical supports unstructured text requests only")
    positions = (original.seq_lens - rejected).cpu().tolist()
    sampled_cpu = sampled.cpu().tolist()
    limit = spec.vllm_config.model_config.max_model_len
    rows = [i for i, count in enumerate(sampled_cpu) if count and positions[i] < limit]
    if not rows:
        return spec.draft_tokens[: original.num_reqs]
    if spec.state.layers:
        raise ValueError("Batched hierarchical decoding requires attention-only state")
    small = spec.small.propose(
        original,
        metadata,
        slots,
        hidden,
        None,
        sampled,
        rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
    )[rows].clone()
    anchors = last_sampled[original.idx_mapping[rows], 0].clone()
    positions = [positions[i] for i in rows]
    counts = [0] * original.num_reqs
    spec.last_trace = []
    spec.policy_metrics["proposals"] += len(rows)
    for round_idx in range(spec.rounds):
        width = min(spec.depth + 1, limit - max(positions))
        tokens = torch.cat((anchors[:, None], small[:, : width - 1]), dim=1)
        batch, metadata, slots = make_batch(spec, original, rows, positions, tokens)
        predictions, hidden, _ = spec._verify(batch, metadata, slots)
        predictions = predictions.view(len(rows), width)
        matches = tokens[:, 1:].eq(predictions[:, :-1])
        accepted_gpu = matches.int().cumprod(1).sum(1)
        accepted = accepted_gpu.cpu().tolist()
        margins = (
            spec.last_margins.view(len(rows), width)
            .gather(1, accepted_gpu[:, None])[:, 0]
            .cpu()
            .tolist()
        )
        keep = []
        spec.policy_metrics["batch_round_calls"] += 1
        for local, (row, count, margin) in enumerate(
            zip(rows, accepted, margins, strict=True)
        ):
            offset = counts[row]
            spec.draft_tokens[row, offset : offset + count] = tokens[
                local, 1 : count + 1
            ]
            spec.draft_tokens[row, offset + count] = predictions[local, count]
            counts[row] += count + 1
            spec.policy_metrics["inner_rounds"] += 1
            spec.policy_metrics["inner_proposed"] += width - 1
            spec.policy_metrics["inner_accepted"] += count
            stop = round_idx < spec.rounds - 1 and should_stop_inner(
                spec.config.hierarchical_stop_policy, count, width - 1, margin
            )
            if stop:
                spec.policy_metrics["early_stops"] += 1
                spec.policy_metrics["skipped_rounds"] += spec.rounds - round_idx - 1
            if (
                not stop
                and width == spec.depth + 1
                and round_idx < spec.rounds - 1
                and positions[local] + count + 1 < limit
            ):
                keep.append(local)
        if not keep:
            break
        anchors = predictions[keep, accepted_gpu[keep]].clone()
        hidden = hidden.view(len(rows), width, -1)[keep].flatten(0, 1).contiguous()
        tokens = tokens[keep].clone()
        rows = [rows[i] for i in keep]
        positions = [positions[i] for i in keep]
        emitted = torch.tensor(
            [accepted[i] + 1 for i in keep], device=spec.device, dtype=torch.int32
        )
        batch, metadata, slots = make_batch(spec, original, rows, positions, tokens)
        spec.last_sampled[batch.idx_mapping, 0] = anchors
        # Compacted requests have fresh metadata, not the Target capture layout.
        small = spec.small.propose(
            batch,
            metadata,
            slots,
            hidden,
            None,
            emitted,
            width - emitted,
            spec.last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            is_profile=True,
        ).clone()
        positions = [p + accepted[i] + 1 for p, i in zip(positions, keep, strict=True)]
    spec.draft_lengths.copy_(
        torch.tensor(counts, device=spec.device, dtype=torch.int32)
    )
    return spec.draft_tokens[: original.num_reqs]

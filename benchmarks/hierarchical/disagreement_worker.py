# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only full-vocabulary diagnostics for greedy Gemma hierarchy."""

import json
import math
import os
from pathlib import Path

import torch


def distribution_pair(left, right):
    """Compare logits on one common token prefix; probabilities use T=1."""
    left, right = left.double(), right.double()
    assert left.ndim == right.ndim == 1 and left.shape == right.shape
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    lp, lq = left.log_softmax(-1), right.log_softmax(-1)
    p, q = lp.exp(), lq.exp()
    lm = torch.logaddexp(lp, lq) - math.log(2.0)
    a, b = int(left.argmax()), int(right.argmax())

    def describe(logits, prob, own, other):
        values, ids = logits.topk(min(8, logits.numel()))
        return {
            "top_tokens": ids.tolist(),
            "top_logits": values.tolist(),
            "top_probs": prob[ids].tolist(),
            "margin": float(values[0] - values[1]),
            "other_top1_rank": int((logits > logits[other]).sum()) + 1,
            "other_top1_prob": float(prob[other]),
            "other_top1_logit_gap": float(logits[own] - logits[other]),
            "entropy": float(-(prob * prob.log()).nan_to_num().sum()),
        }

    return {
        "left": describe(left, p, a, b),
        "right": describe(right, q, b, a),
        "same_top1": a == b,
        "js_nats": float((p * (lp - lm)).sum() / 2 + (q * (lq - lm)).sum() / 2),
        "tv": float((p - q).abs().sum() / 2),
        "kl_left_right": float((p * (lp - lq)).sum()),
        "kl_right_left": float((q * (lq - lp)).sum()),
    }


def rejection_outcome(offset, local_accepted, outer_accepted, scheduled):
    position = offset + local_accepted
    reached = position < scheduled and outer_accepted >= position
    return {
        "position": position,
        "target_reached": reached,
        "correction_accepted": outer_accepted > position if reached else None,
        "suffix_scheduled": max(0, scheduled - position - 1),
        "suffix_accepted": max(0, outer_accepted - position - 1) if reached else None,
    }


class DisagreementWorker:
    def begin_disagreement(self, directory):
        spec = self.model_runner.speculator
        spec.disagreement_directory = Path(directory)
        spec.disagreement_directory.mkdir(exist_ok=True)
        spec.disagreement_active = True


def install():
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.hierarchical.speculator import (
        HierarchicalSpeculator,
        accepted_prefix,
    )

    original_sample = AutoRegressiveSpeculator._sample_draft
    original_verify_eager = HierarchicalSpeculator._verify_eager
    original_verify = HierarchicalSpeculator._verify
    original_propose = HierarchicalSpeculator.propose
    original_record = HierarchicalSpeculator.record_verification

    def sample(self, hidden, positions, mapping, step):
        result = original_sample(self, hidden, positions, mapping, step)
        logits = self.model.compute_logits(hidden)
        if not hasattr(self, "disagreement_logits"):
            self.disagreement_logits = torch.empty(
                (self.max_num_reqs, self.num_speculative_steps, logits.shape[-1]),
                dtype=logits.dtype,
                device=logits.device,
            )
        self.disagreement_logits[: hidden.shape[0]].index_copy_(
            1, step.view(1), logits.unsqueeze(1)
        )
        return result

    def verify_eager(self, batch, metadata, slots):
        output = original_verify_eager(self, batch, metadata, slots)
        if not hasattr(self, "disagreement_preverify"):
            self.disagreement_preverify = {}
        width = batch.num_tokens
        if width not in self.disagreement_preverify:
            self.disagreement_preverify[width] = torch.empty_like(self.last_logits)
        self.disagreement_preverify[width].copy_(self.last_logits)
        return output

    def verify(self, batch, metadata, slots):
        output = original_verify(self, batch, metadata, slots)
        if getattr(self, "disagreement_active", False):
            width = batch.num_tokens
            pv = self.disagreement_preverify[width].float().cpu()
            draft = self.small.disagreement_logits[0, : width - 1].float().cpu()
            tokens = self.small.draft_tokens[0, : width - 1].cpu()
            assert torch.equal(draft.argmax(-1), tokens)
            assert torch.equal(pv.argmax(-1), output[0].cpu())
            accepted = accepted_prefix(tokens, pv.argmax(-1))
            self.disagreement_rounds.append(
                {
                    "draft_tokens": tokens.tolist(),
                    "accepted": accepted,
                    "pair": distribution_pair(draft[accepted], pv[accepted])
                    if accepted < width - 1
                    else None,
                    "pv": pv[: accepted + 1].clone(),
                }
            )
        return output

    def propose(self, *args, **kwargs):
        self.disagreement_rounds = []
        return original_propose(self, *args, **kwargs)

    def record(self, logits, batch, num_sampled):
        if (
            getattr(self, "disagreement_active", False)
            and self.pending_req_id in batch.req_ids
        ):
            scheduled = int(batch.num_draft_tokens_per_req[0])
            accepted = int(num_sampled[0]) - 1
            target = logits[:scheduled].float().cpu()
            candidates = self.draft_tokens[0, :scheduled].cpu()
            assert accepted_prefix(candidates, target.argmax(-1)) == accepted
            pv = torch.cat([r["pv"] for r in self.disagreement_rounds])[:scheduled]
            assert torch.equal(pv.argmax(-1), candidates)
            rounds = []
            for trace, data in zip(
                self.last_trace, self.disagreement_rounds, strict=True
            ):
                assert trace["accepted"] == data["accepted"]
                rounds.append(
                    {
                        **trace,
                        "draft_tokens": data["draft_tokens"],
                        "draft_preverify": data["pair"],
                        **rejection_outcome(
                            trace["offset"], trace["accepted"], accepted, scheduled
                        ),
                    }
                )
            row = {
                "request_id": self.pending_req_id,
                "outer_scheduled": scheduled,
                "outer_accepted": accepted,
                "candidate_tokens": candidates.tolist(),
                "inner_rounds": rounds,
                "preverify_target": [
                    distribution_pair(p, t) for p, t in zip(pv, target, strict=True)
                ],
            }
            with (self.disagreement_directory / "distributions.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        return original_record(self, logits, batch, num_sampled)

    AutoRegressiveSpeculator._sample_draft = sample
    HierarchicalSpeculator._verify_eager = verify_eager
    HierarchicalSpeculator._verify = verify
    HierarchicalSpeculator.propose = propose
    HierarchicalSpeculator.record_verification = record


if os.environ.get("HIERARCHICAL_DISAGREEMENT") == "1":
    install()

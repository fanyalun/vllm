# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read existing h4 logits after verification, without changing captured graphs."""

import json

import torch
from disagreement_worker import distribution_pair


def install():
    from vllm.v1.worker.gpu.spec_decode.hierarchical.speculator import (
        HierarchicalSpeculator,
        accepted_prefix,
    )

    original_verify = HierarchicalSpeculator._verify
    original_propose = HierarchicalSpeculator.propose
    original_record = HierarchicalSpeculator.record_verification

    def verify(self, batch, metadata, slots):
        output = original_verify(self, batch, metadata, slots)
        if getattr(self, "disagreement_active", False):
            assert batch.num_tokens == self.depth + 1
            pv = self.last_logits.cpu().float()
            predictions = output[0].cpu()
            assert torch.equal(pv.argmax(-1), predictions)
            tokens = self.small.draft_tokens[0, : self.depth].cpu()
            accepted = accepted_prefix(tokens, predictions)
            self.confidence_rounds.append(pv[: accepted + 1].clone())
        return output

    def propose(self, *args, **kwargs):
        self.confidence_rounds = []
        return original_propose(self, *args, **kwargs)

    def record(self, logits, batch, num_sampled):
        if (
            getattr(self, "disagreement_active", False)
            and self.pending_req_id in batch.req_ids
        ):
            scheduled = int(batch.num_draft_tokens_per_req[0])
            accepted = int(num_sampled[0]) - 1
            target = logits[:scheduled].cpu().float()
            candidates = self.draft_tokens[0, :scheduled].cpu()
            pv = torch.cat(self.confidence_rounds)[:scheduled]
            assert torch.equal(pv.argmax(-1), candidates)
            assert accepted_prefix(candidates, target.argmax(-1)) == accepted
            rounds = []
            for trace, values in zip(
                self.last_trace, self.confidence_rounds, strict=True
            ):
                assert trace["emitted"] == values.shape[0]
                rounds.append(
                    {
                        **trace,
                        "correction_margin": float(
                            values[-1].topk(2).values.diff().neg()[0]
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

    HierarchicalSpeculator._verify = verify
    HierarchicalSpeculator.propose = propose
    HierarchicalSpeculator.record_verification = record

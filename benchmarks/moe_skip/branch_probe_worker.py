# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare counterfactual draft branches without advancing the target state."""

import torch


class BranchProbeWorker:
    def begin_branch_probe(self, prompt_length, output_length, threshold):
        self._probe_rows = []
        self._probe_proposals = 0
        self._probe_prompt_length = prompt_length
        self._probe_output_length = output_length
        self._probe_threshold = threshold
        if hasattr(self, "_probe_installed"):
            return
        spec = self.model_runner.speculator
        original = spec.propose
        compute_logits = spec.logits_model.compute_logits
        self._probe_force = None
        self._probe_step = 0

        def logits_hook(*args, **kwargs):
            logits = compute_logits(*args, **kwargs)
            step = self._probe_step
            if step == 0:
                self._probe_anchor = int(spec.input_buffers.positions[0].item())
            self._probe_step += 1
            if self._probe_force is not None and step == self._probe_force[0]:
                logits = logits.clone()
                chosen = self._probe_force[1]
                logits[0, chosen] = logits[0].max() + 100
            return logits

        spec.logits_model.compute_logits = logits_hook

        @torch.inference_mode()
        def probe(input_batch, *args, **kwargs):
            assert input_batch.num_reqs == 1

            def run(force=None):
                self._probe_step = 0
                self._probe_force = force
                result = original(input_batch, *args, **kwargs)
                return result, result[0].cpu().tolist()

            result, baseline = run()
            anchor = self._probe_anchor
            emitted = anchor - self._probe_prompt_length + 1
            limit = min(len(baseline), self._probe_output_length - emitted)
            top2 = spec.draft_top8_tokens[0, :, :2].cpu().tolist()
            logits = spec.draft_top2_logits[0].cpu().tolist()
            proposal = self._probe_proposals
            self._probe_proposals += 1
            events = []
            for j in range(max(0, limit - 1)):
                margin = logits[j][0] - logits[j][1]
                if margin >= self._probe_threshold:
                    continue
                if top2[j][0] != baseline[j]:
                    assert margin == 0, "Non-tied topk disagrees with argmax"
                    top2[j] = [baseline[j], top2[j][0]]
                _, branch = run((j, top2[j][1]))
                assert branch[:j] == baseline[:j], "Shared prefix changed"
                assert baseline[j] == top2[j][0]
                assert branch[j] == top2[j][1] != baseline[j]
                equal = [
                    a == b
                    for a, b in zip(
                        baseline[j + 1 : limit], branch[j + 1 : limit], strict=True
                    )
                ]
                consecutive = next(
                    (i for i, same in enumerate(equal) if not same), len(equal)
                )
                events.append(
                    {
                        "proposal_index": proposal,
                        "emitted_tokens": emitted,
                        "has_prefill": bool(input_batch.has_prefill),
                        "draft_position": j + 1,
                        "output_position": emitted + j + 1,
                        "margin": margin,
                        "top2_token_ids": top2[j],
                        "baseline": baseline[:limit],
                        "branch": branch[:limit],
                        "suffix_length": len(equal),
                        "suffix_matches": equal,
                        "consecutive_matches": consecutive,
                        "all_equal": all(equal),
                        "all_different": not any(equal),
                    }
                )
            if events:
                result, restored = run()
                assert restored == baseline, "Baseline replay did not restore state"
                self._probe_rows.extend(events)
            self._probe_force = None
            return result

        spec.propose = probe
        self._probe_installed = True

    def collect_branch_probe(self):
        return {"events": self._probe_rows, "proposals": self._probe_proposals}

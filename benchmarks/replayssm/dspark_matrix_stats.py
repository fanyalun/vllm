# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side verification counters using actual scheduled draft lengths."""

import json
import os
from collections import Counter

from vllm.v1.core.sched.scheduler import Scheduler


class MatrixScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matrix_stats = {}
        self._matrix_admission_width = {}
        self._matrix_path = os.environ["DSPARK_MATRIX_STATS_PATH"]

    def schedule(self, throttle_prefills=False):
        output = super().schedule(throttle_prefills)
        width = len(output.scheduled_new_reqs)
        for request in output.scheduled_new_reqs:
            self._matrix_admission_width[request.req_id] = width
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        for req_id in scheduler_output.num_scheduled_tokens:
            idx = model_runner_output.req_id_to_index.get(req_id)
            request = self.requests.get(req_id)
            if idx is None or request is None or request.is_finished():
                continue
            generated = model_runner_output.sampled_token_ids[idx]
            if not generated:
                continue
            n = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            accepted = max(len(generated) - 1, 0)
            assert 0 <= accepted <= n, (req_id, n, accepted)
            row = self._matrix_stats.setdefault(req_id, Counter())
            row[f"window_{n}_accepted_{accepted}"] += 1
        output = super().update_from_output(scheduler_output, model_runner_output)
        finished = [
            req_id for req_id in self._matrix_stats if req_id not in self.requests
        ]
        if finished:
            with open(self._matrix_path, "a") as f:
                for req_id in finished:
                    f.write(
                        json.dumps(
                            dict(
                                req_id=req_id,
                                initial_admission_width=self._matrix_admission_width.pop(
                                    req_id
                                ),
                                histogram=dict(self._matrix_stats.pop(req_id)),
                            )
                        )
                        + "\n"
                    )
        return output

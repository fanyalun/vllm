# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnose request admission shapes outside the performance matrix."""

import json
import os
import sys
from pathlib import Path

from benchmarks.replayssm.dspark_matrix_stats import MatrixScheduler


class CohortAuditScheduler(MatrixScheduler):
    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        if output.scheduled_new_reqs:
            path = Path(os.environ["DSPARK_MATRIX_STATS_PATH"]).parent
            with (path / "admissions.jsonl").open("a") as file:
                file.write(
                    json.dumps(
                        dict(
                            new_ids=[r.req_id for r in output.scheduled_new_reqs],
                            scheduled=output.num_scheduled_tokens,
                        )
                    )
                    + "\n"
                )
        return output


if __name__ == "__main__":
    from benchmarks.replayssm import dspark_matrix_cell as cell
    from vllm import LLM

    atomic = "--atomic-cohort" in sys.argv
    if atomic:
        sys.argv.remove("--atomic-cohort")
    original_init = LLM.__init__

    def init(self, *args, **kwargs):
        kwargs["scheduler_cls"] = (
            "benchmarks.replayssm.dspark_cohort_audit.CohortAuditScheduler"
        )
        original_init(self, *args, **kwargs)

    def generate(self, prompts, sampling_params, **kwargs):
        core = self.llm_engine.engine_core
        core.call_utility("pause_scheduler", "keep", False)
        try:
            self.enqueue(prompts, sampling_params, use_tqdm=False)
        finally:
            core.call_utility("resume_scheduler")
        return self.wait_for_completion(use_tqdm=False)

    LLM.__init__ = init
    if atomic:
        LLM.generate = generate

    def ordinary_generate(llm, prompts, params):
        return llm.generate(prompts, params, use_tqdm=False)

    cell.generate_cohort = ordinary_generate
    cell.main()

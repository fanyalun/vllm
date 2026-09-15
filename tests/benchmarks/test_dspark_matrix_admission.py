# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from benchmarks.replayssm.dspark_matrix_cell import generate_cohort


@pytest.mark.parametrize("enqueue_error", [False, True])
def test_cohort_stays_paused_until_all_requests_are_enqueued(enqueue_error):
    events = []

    def utility(*args):
        events.append(args)

    def enqueue(prompts, params, use_tqdm):
        assert events == [("pause_scheduler", "keep", False)]
        events.append(("enqueue", tuple(prompts)))
        if enqueue_error:
            raise ValueError("invalid request")

    def complete(use_tqdm):
        assert events[-1] == ("resume_scheduler",)
        events.append(("complete",))
        return ["a", "b"]

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(call_utility=utility)),
        enqueue=enqueue,
        wait_for_completion=complete,
    )
    if enqueue_error:
        with pytest.raises(ValueError, match="invalid request"):
            generate_cohort(llm, ["a", "b"], None)
        assert events[-1] == ("resume_scheduler",)
    else:
        assert generate_cohort(llm, ["a", "b"], None) == ["a", "b"]
        assert events == [
            ("pause_scheduler", "keep", False),
            ("enqueue", ("a", "b")),
            ("resume_scheduler",),
            ("complete",),
        ]

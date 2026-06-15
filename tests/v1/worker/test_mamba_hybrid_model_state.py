# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    init_hybrid_predicted_accept_len,
    predict_hybrid_accept_len,
    update_hybrid_accepted_len_ewma,
)

pytestmark = pytest.mark.cpu_test


def test_hybrid_acceptance_predictor_initializes_to_max_len():
    assert init_hybrid_predicted_accept_len(6) == 7


def test_hybrid_acceptance_predictor_updates_with_ewma():
    ewma = float(init_hybrid_predicted_accept_len(4))
    ewma = update_hybrid_accepted_len_ewma(ewma, accepted_len=2, alpha=0.5)
    assert ewma == pytest.approx(3.5)
    assert predict_hybrid_accept_len(ewma, 4) == 4


def test_hybrid_acceptance_predictor_clamps_to_valid_range():
    assert predict_hybrid_accept_len(0.2, 3) == 1
    assert predict_hybrid_accept_len(9.8, 3) == 4

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.gpu.spec_decode.moe_skip.speculator import (
    _external_request_id,
    _is_internal_request,
)


def test_moe_skip_trace_filters_internal_requests():
    assert _is_internal_request("_warmup_0_")
    assert _is_internal_request("_profile_3_")
    assert not _is_internal_request("0-bd638b82")


def test_moe_skip_trace_restores_external_request_id():
    assert _external_request_id("0-bd638b82") == "0"
    assert _external_request_id("user-request-0123abcd") == "user-request"
    assert _external_request_id("user-request-nothexid") == "user-request-nothexid"

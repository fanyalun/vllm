# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded DSpark debug tracing; not suitable for throughput measurements."""

import hashlib
import inspect

import torch


class DSparkConfidenceTraceWorkerExtension:
    def install_dspark_confidence_trace(self):
        install_trace(self)

    def read_dspark_confidence_trace(self):
        return read_trace(self)


def install_trace(worker):
    speculator = worker.model_runner.speculator
    if speculator.draft_confidence is None:
        raise ValueError("The loaded DSpark model has no confidence head")
    if hasattr(speculator, "_confidence_trace"):
        raise ValueError("Confidence tracing is already installed")
    original = speculator.propose
    signature = inspect.signature(original)
    speculator._confidence_trace = []

    def propose(input_batch, *args, **kwargs):
        bound = signature.bind(input_batch, *args, **kwargs)
        tokens = original(input_batch, *args, **kwargs)
        if len(speculator._confidence_trace) >= 256:
            raise RuntimeError("Confidence debug trace exceeded 256 rounds")
        n = input_batch.num_reqs
        d = speculator.num_speculative_steps
        speculator._confidence_trace.append(
            dict(
                req_ids=list(input_batch.req_ids[:n]),
                draft_tokens=tokens.clone(),
                confidence=speculator.get_draft_confidence(n).clone(),
                positions=speculator.sample_pos[: n * d].view(n, d).clone(),
                previous_num_rejected=bound.arguments["num_rejected"][:n].clone(),
                previous_num_sampled=bound.arguments["num_sampled"][:n].clone(),
            )
        )
        return tokens

    speculator.propose = propose
    speculator._confidence_original_propose = original


def read_trace(worker):
    speculator = worker.model_runner.speculator
    rows = [
        {k: v.tolist() if hasattr(v, "tolist") else v for k, v in row.items()}
        for row in speculator._confidence_trace
    ]
    head = speculator.model.model.confidence_head
    weights = {
        name: hashlib.sha256(
            value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        for name, value in head.named_parameters()
    }
    speculator.propose = speculator._confidence_original_propose
    del speculator._confidence_original_propose
    del speculator._confidence_trace
    return dict(weight_sha256=weights, rounds=rows)

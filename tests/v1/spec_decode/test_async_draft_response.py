# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
from types import SimpleNamespace

import pytest
import torch


def _response_child(connection):
    torch.accelerator.set_device_index(1)
    buffers = [torch.zeros(1, 4, dtype=torch.int64, device="cuda:1") for _ in range(2)]
    # torch.Event.ipc_handle() is not implemented in the supported runtime.
    events = [torch.cuda.Event(interprocess=True) for _ in buffers]
    connection.send((buffers, [event.ipc_handle() for event in events]))
    for generation in range(8):
        assert connection.recv() == generation
        slot = generation % 2
        torch.cuda._sleep(20_000_000)
        buffers[slot].fill_(generation + 1)
        events[slot].record()
        connection.send(generation)
    assert connection.recv() == "shutdown"
    torch.accelerator.synchronize()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="Requires two CUDA devices"
)
def test_response_stream_dependency_survives_ipc_event_and_ring_reuse():
    from vllm.v1.worker.gpu.spec_decode.async_draft.speculator import (
        AsyncDraftSpeculator,
    )

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_response_child, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(60), "Draft child did not publish its IPC buffers"
        buffers, handles = parent.recv()
        speculator = object.__new__(AsyncDraftSpeculator)
        speculator.device = torch.device("cuda:0")
        speculator._draft_tokens = torch.empty(1, 4, dtype=torch.int64, device="cuda:0")
        speculator._response_events = [
            torch.cuda.Event.from_ipc_handle("cuda:1", handle) for handle in handles
        ]
        stream = torch.cuda.Stream(device="cuda:0")
        for generation in range(8):
            parent.send(generation)
            assert parent.poll(60), "Draft child did not record the response event"
            assert parent.recv() == generation
            slot = generation % 2
            with torch.cuda.stream(stream):
                speculator._copy_response(
                    SimpleNamespace(draft_tokens=buffers[slot]), slot, 1
                )
                consumed = speculator._draft_tokens + 100
            stream.synchronize()
            assert consumed.cpu().tolist() == [[generation + 101] * 4]
        parent.send("shutdown")
        del buffers, speculator
        process.join(60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()

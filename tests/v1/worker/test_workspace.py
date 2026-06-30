# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.v1.worker.workspace as workspace_module
from vllm.v1.worker.workspace import WorkspaceManager


def test_get_simultaneous_waits_for_all_pending_reuse_events(monkeypatch) -> None:
    waited_events: list[object] = []

    class FakeStream:

        def wait_event(self, event) -> None:
            waited_events.append(event)

    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1)
    first_event = object()
    second_event = object()
    manager.mark_in_use_until(first_event)
    manager.mark_in_use_until(second_event)

    monkeypatch.setattr(
        workspace_module.torch.cuda,
        "current_stream",
        lambda: FakeStream(),
    )

    [scratch] = manager.get_simultaneous(((4,), torch.uint8))

    assert waited_events == [first_event, second_event]
    assert tuple(scratch.shape) == (4,)
    assert scratch.dtype == torch.uint8

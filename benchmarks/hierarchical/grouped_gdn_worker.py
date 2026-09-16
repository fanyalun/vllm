# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired ablations for state policy and shared-input GDN execution."""

from replay_tail_worker import ReplayTailWorker

from vllm.v1.worker.gpu.spec_decode.hierarchical.grouped_gdn import GroupedGDNPreverify


class GroupedGDNWorker(ReplayTailWorker):
    def set_replay_case(self, case):
        state_mode, group_mode = case.split(":")
        if state_mode not in ("none", "replay_tail") or group_mode not in (
            "none",
            "serial",
            "projection",
            "full",
        ):
            raise ValueError(case)
        result = super().set_replay_case(state_mode)
        spec = self.model_runner.speculator
        if spec.grouped_gdn is None:
            spec.grouped_gdn = GroupedGDNPreverify(spec.model, spec.vllm_config)
        if not hasattr(self, "_group_graphs"):
            self._group_graphs = {}
        spec.preverify_graphs, self._replay_graph_checks = (
            self._group_graphs.setdefault(case, ({}, set()))
        )
        spec.config.preverify_gdn_group_mode = group_mode
        result.update(
            case=case,
            graphs=len(spec.preverify_graphs),
            graph_eager_checked_widths=sorted(self._replay_graph_checks),
        )
        return result

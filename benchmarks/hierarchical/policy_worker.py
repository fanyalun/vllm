# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read hierarchical counters without instrumenting timed model calls."""


class PolicyWorker:
    def policy_counters(self, reset=False):
        if not hasattr(self, "_policy_engine_steps"):
            self._policy_engine_steps = 0
            execute = self.model_runner.execute_model

            def counted_execute(*args, **kwargs):
                self._policy_engine_steps += 1
                return execute(*args, **kwargs)

            self.model_runner.execute_model = counted_execute
        spec = self.model_runner.speculator
        result = {"engine_steps": self._policy_engine_steps}
        if spec is not None and hasattr(spec, "policy_metrics"):
            result.update(spec.policy_metrics)
            result["preverify_graphs"] = len(spec.preverify_graphs)
            if reset:
                spec.reset_policy_metrics()
        if reset:
            self._policy_engine_steps = 0
        return result

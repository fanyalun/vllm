# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.async_draft.cache import CachedBranch
from vllm.v1.worker.gpu.spec_decode.async_draft.ipc import ASYNC_DRAFT_FAN_OUT

_DSPARK_DEFAULT_FAN_OUTS = {
    "Qwen3_5MoeForConditionalGeneration": 24,
    "Gemma4ForConditionalGeneration": 48,
}
_QWEN_MTP_DEFAULT_FAN_OUT = 96


def _internal_fan_out(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        fan_out = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not 1 <= fan_out <= 512:
        raise ValueError(f"{name} must be between 1 and 512, got {fan_out}")
    return fan_out


@dataclass(frozen=True)
class TargetStateLayout:
    name: str
    splits: tuple[int, ...]

    @property
    def width(self) -> int:
        return sum(self.splits)


class AsyncDraftMethodAdapter(ABC):
    """Internal drafter-specific hooks for the asynchronous SSD runtime."""

    method: str
    provisional_state_is_canonical = False
    uses_kv_branches = True

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    def target_state_layout(self) -> TargetStateLayout:
        raise NotImplementedError

    def load_standalone_draft(self, runner: Any) -> list[dict[str, object]]:
        from vllm.v1.worker.gpu.spec_decode.async_draft.runtime import (
            _standalone_load_draft_model,
        )

        return _standalone_load_draft_model(runner, self)

    def materialize_shared_weights(
        self, draft_model: torch.nn.Module
    ) -> list[dict[str, object]]:
        return []

    def jit_propose(self, operation: Callable[..., Any], *args, **kwargs):
        return operation(*args, **kwargs)

    def canonical_commit(self, operation: Callable[..., Any], *args, **kwargs):
        return operation(*args, **kwargs)

    def select_outcome_candidates(
        self,
        logits: torch.Tensor,
        returned_tokens: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.v1.worker.gpu.spec_decode.async_draft.cache import (
            select_recovery_candidates,
        )

        return select_recovery_candidates(logits, returned_tokens, self.fan_out())

    def build_fanout_branches(self, operation: Callable[..., Any], *args, **kwargs):
        return operation(*args, **kwargs)

    def capture_sizes(self, max_num_reqs: int) -> set[int]:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        branches_per_request = (
            speculative_config.num_speculative_tokens + 1
        ) * self.fan_out()
        return {
            branches_per_request * (1 << power)
            for power in range(max_num_reqs.bit_length())
            if 1 << power <= max_num_reqs
        }

    def release_branch_state(self, branch: CachedBranch) -> list[str]:
        return branch.resource_ids()

    def cached_provisional_state_size(self, runner: Any) -> int:
        return runner.speculator.hidden_size

    def fan_out(self) -> int:
        return ASYNC_DRAFT_FAN_OUT

    def proposal_bank_width(self) -> int:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        return speculative_config.num_speculative_tokens

    def branch_backbone_width(self) -> int:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        return speculative_config.num_speculative_tokens


class Eagle3AsyncDraftAdapter(AsyncDraftMethodAdapter):
    method = "eagle3"

    def target_state_layout(self) -> TargetStateLayout:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        draft_hf_config = speculative_config.draft_model_config.hf_config
        layer_ids = getattr(draft_hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if layer_ids is None:
            eagle_config = getattr(draft_hf_config, "eagle_config", {}) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        count = len(layer_ids) if layer_ids else 3
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return TargetStateLayout("eagle3_aux_hidden_states", (hidden_size,) * count)

    def materialize_shared_weights(
        self, draft_model: torch.nn.Module
    ) -> list[dict[str, object]]:
        return _materialize_shared_weights(self.vllm_config, draft_model)


class QwenMTPAsyncDraftAdapter(AsyncDraftMethodAdapter):
    method = "mtp"

    def target_state_layout(self) -> TargetStateLayout:
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return TargetStateLayout("target_last_hidden_state", (hidden_size,))

    def materialize_shared_weights(
        self, draft_model: torch.nn.Module
    ) -> list[dict[str, object]]:
        from vllm.v1.worker.gpu.spec_decode.async_draft.weights import (
            audit_safetensors_prefixes,
        )

        shared = _materialize_shared_weights(self.vllm_config, draft_model)
        mtp = audit_safetensors_prefixes(self.vllm_config.model_config.model, ("mtp.",))
        return [*shared, *(item.to_dict() for item in mtp)]

    def fan_out(self) -> int:
        return _internal_fan_out("ASYNC_DRAFT_MTP_FAN_OUT", _QWEN_MTP_DEFAULT_FAN_OUT)


class DSparkAsyncDraftAdapter(AsyncDraftMethodAdapter):
    method = "dspark"

    def target_state_layout(self) -> TargetStateLayout:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        draft_hf_config = speculative_config.draft_model_config.hf_config
        layer_ids = getattr(draft_hf_config, "aux_hidden_state_layer_ids", None)
        if layer_ids is None:
            layer_ids = getattr(draft_hf_config, "dspark_target_layer_ids", None)
        if layer_ids is None:
            layer_ids = getattr(
                draft_hf_config, "eagle_aux_hidden_state_layer_ids", None
            )
        if not layer_ids:
            raise ValueError("DSpark checkpoint does not declare target aux layers")
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return TargetStateLayout(
            "dspark_aux_hidden_states", (hidden_size,) * len(layer_ids)
        )

    def cached_provisional_state_size(self, runner: Any) -> int:
        return 0

    def fan_out(self) -> int:
        name = "ASYNC_DRAFT_DSPARK_FAN_OUT"
        if name in os.environ:
            return _internal_fan_out(name, ASYNC_DRAFT_FAN_OUT)
        architecture = self.vllm_config.model_config.architecture
        default = _DSPARK_DEFAULT_FAN_OUTS.get(architecture, ASYNC_DRAFT_FAN_OUT)
        return _internal_fan_out(name, default)

    def capture_sizes(self, max_num_reqs: int) -> set[int]:
        del max_num_reqs
        return set()

    def proposal_bank_width(self) -> int:
        from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
            get_dspark_proposal_bank_width,
        )

        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        draft_config = speculative_config.draft_model_config.hf_config
        bank_width = get_dspark_proposal_bank_width(draft_config)
        verify_width = speculative_config.num_speculative_tokens
        if bank_width < verify_width:
            raise ValueError(
                "DSpark proposal bank must cover the Target verification width: "
                f"bank_width={bank_width}, num_speculative_tokens={verify_width}"
            )
        return bank_width

    def branch_backbone_width(self) -> int:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        verify_width = speculative_config.num_speculative_tokens
        self.proposal_bank_width()
        return verify_width


def _materialize_shared_weights(
    vllm_config: VllmConfig, draft_model: torch.nn.Module
) -> list[dict[str, object]]:
    from vllm.v1.worker.gpu.spec_decode.async_draft.weights import (
        materialize_standalone_draft_weights,
    )

    return [
        item.to_dict()
        for item in materialize_standalone_draft_weights(
            draft_model, vllm_config.model_config.model
        )
    ]


def get_async_draft_adapter(vllm_config: VllmConfig) -> AsyncDraftMethodAdapter:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    method = speculative_config.method
    if method == "eagle3":
        return Eagle3AsyncDraftAdapter(vllm_config)
    if method == "mtp":
        if speculative_config.use_gemma4_mtp():
            raise ValueError(
                "Gemma4 MTP asynchronous drafting is blocked until a compatible "
                "26B assistant checkpoint is supplied and validated"
            )
        return QwenMTPAsyncDraftAdapter(vllm_config)
    if method == "dspark":
        return DSparkAsyncDraftAdapter(vllm_config)
    raise ValueError(f"Unsupported asynchronous draft method: {method!r}")

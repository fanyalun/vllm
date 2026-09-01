# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open


@dataclass(frozen=True)
class MaterializedWeight:
    name: str
    source: str
    checkpoint_key: str
    checkpoint_file: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().to(device="cpu").contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _safetensors_files(model_path: Path) -> list[Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return sorted({model_path / filename for filename in weight_map.values()})

    single_file = model_path / "model.safetensors"
    if single_file.is_file():
        return [single_file]
    return sorted(model_path.glob("*.safetensors"))


def _locate_safetensors_key(model_path: Path, key: str) -> Path:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)
        filename = index.get("weight_map", {}).get(key)
        if filename is None:
            raise KeyError(f"Checkpoint key {key!r} is absent from {index_path}")
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    for path in _safetensors_files(model_path):
        with safe_open(path, framework="pt", device="cpu") as file:
            if key in file:
                return path

    if (model_path / "pytorch_model.bin").exists() or list(
        model_path.glob("pytorch_model-*.bin")
    ):
        raise ValueError(
            f"Checkpoint key {key!r} cannot be located without loading a "
            f"monolithic .bin checkpoint at {model_path}. Convert the target "
            "checkpoint to indexed safetensors first."
        )
    raise KeyError(f"Checkpoint key {key!r} is absent from {model_path}")


def load_safetensors_key(model_path: str, key: str) -> tuple[torch.Tensor, Path]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(
            "Standalone asynchronous draft shared-weight materialization "
            f"requires a local target checkpoint directory, got {model_path!r}."
        )
    path = _locate_safetensors_key(root, key)
    with safe_open(path, framework="pt", device="cpu") as file:
        return file.get_tensor(key), path


def _copy_weight(
    module: nn.Module,
    checkpoint_tensor: torch.Tensor,
    name: str,
) -> torch.Tensor:
    parameter = getattr(module, "weight", None)
    if parameter is None:
        raise ValueError(f"{name} has no weight parameter")
    if tuple(parameter.shape) != tuple(checkpoint_tensor.shape):
        raise ValueError(
            f"{name} shape mismatch: model={tuple(parameter.shape)}, "
            f"checkpoint={tuple(checkpoint_tensor.shape)}"
        )
    parameter.data.copy_(
        checkpoint_tensor.to(device=parameter.device, dtype=parameter.dtype)
    )
    return parameter


def materialize_standalone_eagle_weights(
    eagle_model: nn.Module,
    target_model_path: str,
) -> list[MaterializedWeight]:
    """Fill EAGLE weights that the in-process path normally shares.

    Only individual safetensors keys are read from the target checkpoint. The
    full target model is never instantiated in the standalone draft process.
    """

    inner = getattr(eagle_model, "model", None)
    if inner is None:
        raise ValueError("EAGLE model has no inner model")

    specs = (
        (
            "model.embed_tokens.weight",
            bool(getattr(eagle_model, "has_own_embed_tokens", False)),
            getattr(inner, "embed_tokens", None),
            "model.embed_tokens.weight",
        ),
        (
            "lm_head.weight",
            bool(getattr(eagle_model, "has_own_lm_head", False)),
            getattr(eagle_model, "lm_head", None),
            "lm_head.weight",
        ),
    )

    results: list[MaterializedWeight] = []
    for name, has_own_weight, module, checkpoint_key in specs:
        if module is None:
            raise ValueError(f"EAGLE model is missing {name}")

        if has_own_weight:
            parameter = module.weight
            results.append(
                MaterializedWeight(
                    name=name,
                    source="draft",
                    checkpoint_key=name,
                    checkpoint_file="",
                    shape=tuple(parameter.shape),
                    dtype=str(parameter.dtype),
                    sha256=_tensor_sha256(parameter),
                )
            )
            continue

        checkpoint_tensor, checkpoint_file = load_safetensors_key(
            target_model_path, checkpoint_key
        )
        parameter = _copy_weight(module, checkpoint_tensor, name)
        if _tensor_sha256(parameter) != _tensor_sha256(
            checkpoint_tensor.to(dtype=parameter.dtype)
        ):
            raise ValueError(f"Checksum mismatch after materializing {name}")
        results.append(
            MaterializedWeight(
                name=name,
                source="target",
                checkpoint_key=checkpoint_key,
                checkpoint_file=checkpoint_file.name,
                shape=tuple(parameter.shape),
                dtype=str(parameter.dtype),
                sha256=_tensor_sha256(parameter),
            )
        )

    return results

# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model/config/checkpoint inventory and classified structure differences."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class DifferenceCategory(str, Enum):
    """Action category for one authoritative-versus-candidate difference."""

    GENERIC_HF = "generic_hf_supported"
    MODEL_ADAPTER = "model_adapter"
    COMPONENT_REPLACEMENT = "component_replacement"
    CUSTOM_MODEL = "hf_component_custom_model"
    FRAMEWORK_GAP = "framework_contract_gap"
    OUT_OF_SCOPE = "decode_cache_or_quantization_out_of_scope"


@dataclass(frozen=True)
class InventoryDifference:
    """One classified structural difference."""

    path: str
    reference: Any
    candidate: Any
    category: DifferenceCategory
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["category"] = self.category.value
        return payload


@dataclass
class ModelInventory:
    """Serializable inventory of a model's final module and state tree."""

    source: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    modules: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    buffers: list[dict[str, Any]] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _shape(value: Any) -> list[int]:
    return [int(size) for size in getattr(value, "shape", ())]


def _layout(value: Any) -> dict[str, Any] | None:
    layout = getattr(value, "layout", None)
    if layout is None:
        placements = getattr(value, "placements", None)
        mesh = getattr(value, "device_mesh", None)
    else:
        placements = getattr(layout, "placements", None)
        mesh = getattr(layout, "mesh", None)
    if placements is None and mesh is None:
        return None
    return {
        "mesh_dim_names": list(getattr(mesh, "mesh_dim_names", ()) or ()),
        "mesh_shape": list(getattr(getattr(mesh, "mesh", None), "shape", ()) or ()),
        "placements": [repr(placement) for placement in (placements or ())],
    }


def _forward_signature(module: Any) -> str | None:
    forward = getattr(module, "forward", None)
    if not callable(forward):
        return None
    try:
        return str(inspect.signature(forward))
    except (TypeError, ValueError):
        return None


def _config_dict(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(config, "__dict__"):
        return {
            name: value
            for name, value in vars(config).items()
            if isinstance(value, (str, int, float, bool, type(None), list, dict))
        }
    return {"repr": repr(config)}


def build_model_inventory(
    model: Any,
    *,
    source: Optional[dict[str, Any]] = None,
    checkpoint: Optional[dict[str, Any]] = None,
) -> ModelInventory:
    """Inventory the instantiated final model tree, including aliases.

    Args:
        model: Torch-compatible module exposing named module/state iterators.
        source: Recorded source/revision/import metadata.
        checkpoint: Optional checkpoint index inventory.

    Returns:
        Serializable model inventory.
    """
    modules = []
    for module_fqn, module in model.named_modules(remove_duplicate=False):
        modules.append(
            {
                "fqn": module_fqn or "<root>",
                "type": _qualified_type(module),
                "forward_signature": _forward_signature(module),
                "training": bool(getattr(module, "training", False)),
            }
        )

    parameter_aliases: dict[int, list[str]] = {}
    parameter_by_id = {}
    for parameter_fqn, parameter in model.named_parameters(remove_duplicate=False):
        parameter_aliases.setdefault(id(parameter), []).append(parameter_fqn)
        parameter_by_id[id(parameter)] = parameter
    parameters = []
    for parameter_id, fqns in sorted(parameter_aliases.items(), key=lambda item: min(item[1])):
        parameter = parameter_by_id[parameter_id]
        parameters.append(
            {
                "fqn": min(fqns),
                "aliases": sorted(fqns),
                "shape": _shape(parameter),
                "dtype": str(getattr(parameter, "dtype", None)),
                "device": str(getattr(parameter, "device", None)),
                "requires_grad": bool(getattr(parameter, "requires_grad", False)),
                "is_meta": bool(getattr(parameter, "is_meta", False)),
                "layout": _layout(parameter),
            }
        )

    non_persistent_ids = set()
    for module in model.modules():
        for name in getattr(module, "_non_persistent_buffers_set", set()):
            buffer_value = getattr(module, name, None)
            if buffer_value is not None:
                non_persistent_ids.add(id(buffer_value))
    buffer_aliases: dict[int, list[str]] = {}
    buffer_by_id = {}
    for buffer_fqn, buffer in model.named_buffers(remove_duplicate=False):
        buffer_aliases.setdefault(id(buffer), []).append(buffer_fqn)
        buffer_by_id[id(buffer)] = buffer
    buffers = []
    for buffer_id, fqns in sorted(buffer_aliases.items(), key=lambda item: min(item[1])):
        buffer = buffer_by_id[buffer_id]
        buffers.append(
            {
                "fqn": min(fqns),
                "aliases": sorted(fqns),
                "shape": _shape(buffer),
                "dtype": str(getattr(buffer, "dtype", None)),
                "device": str(getattr(buffer, "device", None)),
                "is_meta": bool(getattr(buffer, "is_meta", False)),
                "persistent": buffer_id not in non_persistent_ids,
                "layout": _layout(buffer),
            }
        )
    aliases = {
        "parameters": [fqns for fqns in parameter_aliases.values() if len(fqns) > 1],
        "buffers": [fqns for fqns in buffer_aliases.values() if len(fqns) > 1],
    }
    return ModelInventory(
        source=dict(source or {}),
        config=_config_dict(model),
        modules=modules,
        parameters=parameters,
        buffers=buffers,
        aliases=aliases,
        checkpoint=dict(checkpoint or {}),
    )


def inspect_local_model_assets(model_path: str | Path) -> dict[str, Any]:
    """Inspect local config and checkpoint indices without loading tensors."""
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"model asset path is not a directory: {root}")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    checkpoint = {
        "keys": [],
        "weight_map": {},
        "metadata": {},
        "lfs_pointers": {},
    }

    def lfs_pointer(path: Path) -> dict[str, Any] | None:
        """Parse one small Git-LFS pointer without resolving its payload."""
        if path.stat().st_size > 1024:
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
            return None
        fields = {}
        for line in lines[1:]:
            name, _, value = line.partition(" ")
            fields[name] = value
        oid = fields.get("oid", "")
        size = fields.get("size", "")
        if not oid.startswith("sha256:") or not size.isdigit():
            raise ValueError(f"invalid Git-LFS pointer: {path}")
        return {
            "oid": oid,
            "payload_size": int(size),
            "payload_available": False,
        }

    index_paths = sorted(root.glob("*.index.json"))
    for index_path in index_paths:
        pointer = lfs_pointer(index_path)
        if pointer is not None:
            checkpoint["lfs_pointers"][index_path.name] = pointer
            checkpoint["metadata"][index_path.name] = {"git_lfs_pointer": True}
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint index is neither JSON nor a Git-LFS pointer: {index_path}") from exc
        weight_map = index.get("weight_map", {})
        checkpoint["weight_map"].update(weight_map)
        checkpoint["metadata"][index_path.name] = index.get("metadata", {})
    for weight_path in sorted(root.glob("*.safetensors")):
        pointer = lfs_pointer(weight_path)
        if pointer is not None:
            checkpoint["lfs_pointers"][weight_path.name] = pointer
    checkpoint["keys"] = sorted(checkpoint["weight_map"])
    checkpoint["weights_available"] = not checkpoint["lfs_pointers"]
    return {
        "path": str(root),
        "config": config,
        "checkpoint": checkpoint,
        "files": sorted(path.name for path in root.iterdir() if path.is_file()),
    }


def diff_inventories(
    reference: ModelInventory,
    candidate: ModelInventory,
    classifier: Optional[Callable[[str, Any, Any], tuple[DifferenceCategory, str]]] = None,
) -> list[InventoryDifference]:
    """Compare module/state records and classify every difference."""
    differences = []

    def classify(path: str, expected: Any, actual: Any) -> tuple[DifferenceCategory, str]:
        """Classify one inventory difference into its owning integration layer."""
        if classifier is not None:
            return classifier(path, expected, actual)
        if path.startswith("parameters.") or path.startswith("buffers."):
            return DifferenceCategory.MODEL_ADAPTER, "state layout/name requires explicit adapter mapping"
        if path.startswith("modules."):
            return DifferenceCategory.COMPONENT_REPLACEMENT, "module structure requires replacement review"
        return DifferenceCategory.CUSTOM_MODEL, "configuration differs from the authoritative architecture"

    for collection_name in ("modules", "parameters", "buffers"):
        reference_entries = {
            entry["fqn"]: entry for entry in getattr(reference, collection_name)
        }
        candidate_entries = {
            entry["fqn"]: entry for entry in getattr(candidate, collection_name)
        }
        for fqn in sorted(set(reference_entries) | set(candidate_entries)):
            expected = reference_entries.get(fqn)
            actual = candidate_entries.get(fqn)
            if expected == actual:
                continue
            path = f"{collection_name}.{fqn}"
            category, reason = classify(path, expected, actual)
            differences.append(
                InventoryDifference(path, expected, actual, category, reason)
            )
    if reference.config != candidate.config:
        category, reason = classify("config", reference.config, candidate.config)
        differences.append(
            InventoryDifference("config", reference.config, candidate.config, category, reason)
        )
    return differences


__all__ = [
    "DifferenceCategory",
    "InventoryDifference",
    "ModelInventory",
    "build_model_inventory",
    "diff_inventories",
    "inspect_local_model_assets",
]

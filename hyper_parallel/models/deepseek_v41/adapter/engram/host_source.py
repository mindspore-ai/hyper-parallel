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
"""Resolve and materialize only the Host rows owned by this EP rank."""
# pylint: disable=forbidden-backend-import

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    rename_source_key,
)
from hyper_parallel.components.checkpoint.load_groups import (
    base_weights_mapping,
    build_load_targets,
    resolve_checkpoint_index,
)
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.components.checkpoint.quantized_hf_storage import build_quantized_hf_reader
from hyper_parallel.models.external_state import ExternalLoadContext


def _row_manifest(checkpoint_index: Any) -> tuple[Path, dict[str, Any]]:
    """Read optional logical-table to physical-safetensors row mapping."""
    root = next(iter(checkpoint_index.files_by_key.values())).parent
    path = root / "engram_host_rows.json"
    if not path.is_file():
        return root, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not isinstance(payload.get("tables"), dict):
        raise ValueError("Invalid Engram Host row manifest")
    return root, payload["tables"]


def _read_manifest_rows(root: Path, source_key: str, spec: dict[str, Any],
                        table: HostEngramTable) -> torch.Tensor:
    """Validate complete physical row coverage and read only this owner."""
    if spec.get("shape") != [table.physical_rows, table.width]:
        raise ValueError(f"Host Engram manifest shape mismatch: {source_key}")
    fragments = spec.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError(f"Host Engram manifest has no row fragments: {source_key}")
    rows = torch.empty((table.rows_per_owner, table.width), dtype=torch.float32)
    owner_start = table.global_row_start
    owner_end = owner_start + table.rows_per_owner
    cursor = 0
    for fragment in sorted(fragments, key=lambda item: item["start"]):
        start, length = int(fragment["start"]), int(fragment["rows"])
        end = start + length
        if start != cursor or length <= 0:
            raise ValueError(f"Host Engram manifest row gap or overlap: {source_key}")
        cursor = end
        left, right = max(start, owner_start), min(end, owner_end)
        if left >= right:
            continue
        path = (root / fragment["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.suffix != ".safetensors":
            raise ValueError(f"Host Engram manifest has invalid fragment path: {source_key}")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            tensor_slice = source.get_slice(fragment["key"])
            if tuple(tensor_slice.get_shape()) != (length, table.width):
                raise ValueError(f"Host Engram manifest fragment shape mismatch: {source_key}")
            part = tensor_slice[left - start:right - start]
        if part.dtype != torch.float32:
            raise ValueError(f"Host Engram manifest fragment dtype mismatch: {source_key}")
        rows[left - owner_start:right - owner_start] = part
    if cursor != table.physical_rows:
        raise ValueError(f"Host Engram manifest row coverage incomplete: {source_key}")
    return rows


def _host_source_keys(model: nn.Module, checkpoint_index: Any,
                      manifest: dict[str, Any], weights_mapping: Any,
                      targets: frozenset[str]) -> dict[str, str]:
    """Resolve one sliceable source tensor for each Host target FQN."""
    replacement_mapping = getattr(model, "_hp_replacement_weight_conversions", None)
    mapping = base_weights_mapping(weights_mapping or [], replacement_mapping or [])
    renamings = [item for item in mapping if isinstance(item, WeightRenaming)]
    converters = [item for item in mapping if isinstance(item, WeightConverter)]
    all_targets = build_load_targets(model)
    found: dict[str, str] = {}
    for source_key in sorted(set(checkpoint_index.keys()) | manifest.keys()):
        target, source_pattern = rename_source_key(
            source_key, renamings, converters, getattr(model, "base_model_prefix", None),
            meta_state_dict=all_targets,
        )
        if target not in targets and source_key in targets:
            target, source_pattern = rename_source_key(
                source_key, [], [], getattr(model, "base_model_prefix", None),
                meta_state_dict=all_targets,
            )
        if target not in targets:
            continue
        if source_pattern is not None:
            raise ValueError(f"Host Engram source conversion cannot be sliced: {source_key}")
        if target in found:
            raise ValueError(f"Multiple Host Engram source keys map to {target}")
        found[target] = source_key
    missing = targets - found.keys()
    if missing:
        raise KeyError(f"Missing Host Engram source key for {sorted(missing)}")
    return found


def _seed_owner_rows(fqn: str, table: HostEngramTable) -> torch.Tensor:
    """Initialize each owner row independently of EP replica placement."""
    rows = torch.empty((table.rows_per_owner, table.width), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    layer_seed = int.from_bytes(hashlib.sha256(fqn.encode()).digest()[:8], "little")
    for local_row in range(table.rows_per_owner):
        generator.manual_seed((1729 + layer_seed + table.global_row_start + local_row) % (2**63 - 1))
        rows[local_row] = torch.randn(table.width, generator=generator)
    return rows


def materialize_owner_rows(model: nn.Module, tables: Mapping[str, HostEngramTable],
                           context: ExternalLoadContext) -> frozenset[str]:
    """Install each owner's CPU rows into its existing parameter object.

    Args:
        model: Model whose source key mappings are resolved.
        tables: Exact named Host tables whose weights are replaced in place.
        context: Source path and optional mapping for pretrained loading.

    Returns:
        Source keys consumed by the Host tables, or an empty set for initialization.
    """
    source_keys = set()
    index = None
    source_by_target = {}
    manifest_root = None
    manifest = {}
    quantized_reader = None
    if context.load_base_model and tables:
        index = resolve_checkpoint_index(context.pretrained_path)
        manifest_root, manifest = _row_manifest(index)
        quantized_reader = build_quantized_hf_reader(manifest_root)
        targets = frozenset(f"{fqn}.weight" for fqn in tables)
        source_by_target = _host_source_keys(
            model, index, manifest, context.weights_mapping, targets,
        )
    for fqn, table in tables.items():
        target = f"{fqn}.weight"
        if index is None:
            rows = _seed_owner_rows(fqn, table)
        else:
            source_key = source_by_target[target]
            if source_key in manifest:
                rows = _read_manifest_rows(manifest_root, source_key, manifest[source_key], table)
            elif quantized_reader is not None:
                rows = quantized_reader.read_region(
                    source_key, (table.global_row_start, 0), (table.rows_per_owner, table.width),
                ).float()
            else:
                with safe_open(str(index.files_by_key[source_key]), framework="pt", device="cpu") as source:
                    tensor_slice = source.get_slice(source_key)
                    if tuple(tensor_slice.get_shape()) != (table.physical_rows, table.width):
                        raise ValueError(f"Host Engram source shape mismatch: {target}")
                    rows = tensor_slice[table.global_row_start:table.global_row_start + table.rows_per_owner]
            source_keys.add(source_key)
        if rows.dtype != torch.float32 or tuple(rows.shape) != tuple(table.weight.shape):
            raise ValueError(f"Host Engram source dtype or local shape mismatch: {target}")
        torch.utils.swap_tensors(table.weight, nn.Parameter(rows.contiguous()))
        table.weight._is_hf_initialized = True  # pylint: disable=W0212
    return frozenset(source_keys)

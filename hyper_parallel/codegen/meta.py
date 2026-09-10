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
"""``codegen_meta.json`` read/write and schema validation.

The meta file is the single source of truth for an artifact bundle's
identity: it records the signature, YAML path + hash, source file the
modeling output was generated from, parallel dims, and the resolved
replacements/injections/param plan so a later preflight pass can detect
drift without re-running the whole generation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.hash import sha256_file


@dataclass
class CodegenMeta:
    """Serializable artifact metadata for one generated modeling bundle."""

    codegen_version: str
    signature: str
    yaml_path: str
    yaml_sha256: str
    source: dict[str, Any]
    parallel_dims: dict[str, Any]
    model_name: Optional[str] = None
    outputs: dict[str, Any] = field(default_factory=dict)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    injections: list[dict[str, Any]] = field(default_factory=list)
    skipped_overrides: list[dict[str, Any]] = field(default_factory=list)
    param_plan: dict[str, Any] = field(default_factory=dict)
    frozen_sharded_params: list[str] = field(default_factory=list)
    tied_pairs: list[list[str]] = field(default_factory=list)
    special_handlers: dict[str, str] = field(default_factory=dict)
    #: FQN -> owning class name for every boundary. The emitter
    #: groups boundary FQNs by this to rewrite one ``class.forward`` per class.
    boundary_classes: dict[str, str] = field(default_factory=dict)
    entrypoints: dict[str, str] = field(default_factory=dict)
    #: Sunk ``replace_module`` overrides: one JSON-safe record per
    #: matched target, ``{match, fqn, fqns, module_type, factory, exact_type}``.
    #: The record's FQN set must equal the generated
    #: ``_HYPER_MODULE_OVERRIDES`` literal's FQN set.
    module_overrides: list[dict[str, Any]] = field(default_factory=list)
    #: Entry model class name (the architecture name, e.g. ``AnthropicV3ForCausalLM``)
    #: whose ``__init__`` tail receives the ``hyper_apply_replacements(self)`` call.
    model_class: Optional[str] = None
    #: ``covered`` must always carry a bool ``sharding_plan``,
    #: so the default mirrors ``COVERED_DEFAULTS`` rather than ``{}`` — a bare
    #: ``CodegenMeta`` round-trips through ``load_codegen_meta`` instead of
    #: failing schema validation.  The manager overwrites this explicitly.
    covered: dict[str, Any] = field(default_factory=lambda: dict(COVERED_DEFAULTS))
    not_covered: list[str] = field(default_factory=list)
    #: Plan-coordinate active axes (``FrozenPlan.mesh_dim_names``), JSON form.
    #: The runtime slices the live mesh down to these axes so placements never
    #: land on dp axes the plan never shards on. ``None`` means the runtime
    #: uses the full mesh for compatibility with older artifacts.
    mesh_dim_names: Optional[list[str]] = None
    #: ``<remote>`` source sibling modules copied into the bundle:
    #: ``["configuration_deepseek.py", ...]``
    #: listing every package sibling the generated file's relative imports
    #: resolve against.  Empty for installed transformers sources (they use
    #: absolute imports) and for artifacts generated before this field existed.
    remote_siblings: list[str] = field(default_factory=list)


#: Stages the generated artifact declares it covers.  ``sharding_plan`` starts
#: as the default here (False, kept for metas that predate the field), but
#: ``_fill_plan_fields`` (manager.py) sets it True once the plan is frozen.
#: The generated module carries the parallel logic and injects
#: ``hyper_parallelize``. When True,
#: the trainer short-circuits the planner+applier and runs the generated
#: ``hyper_parallelize``.  ``module_overrides`` turns True only once the
#: generated artifact sinks ``replace_module`` as ``_HYPER_MODULE_OVERRIDES``
#: so the trainer's HF replacement step is skipped, because
#: the generated ``__init__`` already applied it.
COVERED_DEFAULTS = {
    "sharding_plan": False,
    "module_overrides": False,
    "axes": [],
}

#: Stages that remain trainer-owned.
NOT_COVERED_DEFAULT = [
    "mesh",
    "fsdp2",
    "pp",
    "activation_checkpoint",
    "compile",
]


def meta_to_dict(meta: CodegenMeta) -> dict[str, Any]:
    """Serialize a CodegenMeta to a JSON-friendly dict."""
    return asdict(meta)


def meta_from_dict(data: dict[str, Any]) -> CodegenMeta:
    """Rebuild a CodegenMeta from a JSON dict."""
    validate_meta_schema(data)
    return CodegenMeta(
        **{k: v for k, v in data.items() if k in CodegenMeta.__dataclass_fields__}
    )


def load_codegen_meta(meta_path: str) -> Optional[CodegenMeta]:
    """Read the meta file; ``None`` if absent or malformed."""
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return meta_from_dict(json.load(handle))
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def write_codegen_meta(meta: CodegenMeta, path: str) -> None:
    """Write the meta as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta_to_dict(meta), handle, indent=2, sort_keys=True)


def validate_meta_schema(data: dict[str, Any]) -> None:
    """Fail fast if required meta fields are missing or mistyped."""
    required = ("codegen_version", "signature", "yaml_path", "yaml_sha256")
    for key in required:
        if key not in data:
            raise ValueError(f"codegen_meta missing required field {key!r}")
        if not isinstance(data[key], str):
            raise TypeError(f"codegen_meta field {key!r} must be a string")
    for key in ("source", "parallel_dims", "outputs"):
        if key in data and not isinstance(data[key], dict):
            raise TypeError(f"codegen_meta field {key!r} must be a dict")
    if "entrypoints" in data:
        entrypoints = data["entrypoints"]
        if not isinstance(entrypoints, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in entrypoints.items()
        ):
            raise TypeError("codegen_meta field 'entrypoints' must be a dict[str, str]")
    if "covered" in data:
        covered = data["covered"]
        if not isinstance(covered, dict):
            raise TypeError("codegen_meta field 'covered' must be a dict")
        if not isinstance(covered.get("sharding_plan"), bool):
            raise TypeError("codegen_meta field 'covered.sharding_plan' must be a bool")
    if "not_covered" in data:
        not_covered = data["not_covered"]
        if not isinstance(not_covered, list) or not all(
            isinstance(item, str) for item in not_covered
        ):
            raise TypeError("codegen_meta field 'not_covered' must be a list[str]")
    if "module_overrides" in data:
        overrides = data["module_overrides"]
        if not isinstance(overrides, list) or not all(
            isinstance(item, dict) for item in overrides
        ):
            raise TypeError(
                "codegen_meta field 'module_overrides' must be a list[dict]"
            )
    if "remote_siblings" in data:
        siblings = data["remote_siblings"]
        if not isinstance(siblings, list) or not all(
            isinstance(item, str) for item in siblings
        ):
            raise TypeError("codegen_meta field 'remote_siblings' must be a list[str]")


def record_output_hashes(
    meta: CodegenMeta,
    layout: ArtifactLayout,
    *,
    base_dir: str = "",
) -> CodegenMeta:
    """Populate ``meta.outputs`` with per-file hashes and line counts.

    When ``base_dir`` is given every regular file is hashed from *there* (the
    staging directory); otherwise the artifact directory is used. Hashing must
    happen where the bytes actually live at write time: the atomic bundle
    writes ``codegen_meta.json`` into the staging dir, so ``base_dir`` is the
    staging dir and the layout paths still point at the old bundle.

    The meta file itself is deliberately NOT recorded: its sha256 cannot be
    stored in its own bytes (the value changes the file, so the hash could
    never be a fixed point), and the previous record-then-write order hashed
    the *old* meta and made preflight report drift on every second
    generation.  The meta file's integrity is covered by the signature check,
    the schema validation, and the hashes of every other bundle file — it does
    not need a self-hash.
    """
    output_dir = base_dir or layout.artifact_dir
    role_by_name = {
        os.path.basename(layout.modeling_path): "modeling",
        os.path.basename(layout.diff_path): "diff",
        os.path.basename(layout.init_path): "init",
    }
    meta.outputs = {}
    for name in sorted(os.listdir(output_dir)):
        target = os.path.join(output_dir, name)
        if name == os.path.basename(layout.meta_path) or not os.path.isfile(target):
            continue
        key = role_by_name.get(name, name)
        meta.outputs[key] = {
            "sha256": sha256_file(target),
            "lines": _line_count(target),
        }
    return meta


def _line_count(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


__all__ = [
    "CodegenMeta",
    "load_codegen_meta",
    "meta_from_dict",
    "meta_to_dict",
    "record_output_hashes",
    "validate_meta_schema",
    "write_codegen_meta",
]

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
"""Preserve the native model's observed parameter freezing in ordinary source."""

from __future__ import annotations

from typing import Any

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits
from hyper_parallel.codegen.astkit.index import build_source_index


def observe_frozen_modules(model: Any) -> list[dict]:
    """Compact already frozen parameter subtrees without interpreting configuration."""
    modules = dict(model.named_modules())
    selected = []
    result = []
    for path, module in modules.items():
        parameters = list(module.parameters())
        if not parameters or any(parameter.requires_grad for parameter in parameters):
            continue
        if any(parent == "" or path.startswith(parent + ".") for parent in selected):
            continue
        selected.append(path)
        result.append({"fqn": path, "candidates": _frozen_candidates(path, modules)})
    for path, parameter in model.named_parameters():
        if parameter.requires_grad or any(parent == "" or path.startswith(parent + ".") for parent in selected):
            continue
        result.append({"fqn": path, "candidates": _frozen_candidates(path, modules, parameter=True)})
    return result


def _frozen_candidates(path: str, modules: dict, *, parameter: bool = False) -> list[dict]:
    parts = path.split(".") if path else []
    result = []
    for length in range(len(parts) - int(parameter), -1, -1):
        owner = modules[".".join(parts[:length])]
        relative = ".".join(parts[length:])
        if not all(part.isidentifier() for part in parts[length:]):
            continue
        compatible = True
        for peer in modules.values():
            if type(peer) is not type(owner):
                continue
            try:
                target = peer.get_parameter(relative) if parameter else peer.get_submodule(relative)
            except (AttributeError, KeyError):
                compatible = False
                break
            parameters = [target] if parameter else list(target.parameters())
            if not parameters or any(item.requires_grad for item in parameters):
                compatible = False
                break
        if compatible:
            result.append({"owner": type(owner).__name__, "relative": relative})
    return result


def materialize_frozen_modules(source: str, records: list[dict]) -> str:
    """Emit requires_grad_ calls after the source owner has built its children."""
    index = build_source_index(source)
    edits = []
    emitted = set()
    for record in records:
        for candidate in record["candidates"]:
            host = index.classes.get(candidate["owner"])
            initializer = host.methods.get("__init__") if host else None
            if initializer is None:
                continue
            target = "self" + ("." + candidate["relative"] if candidate["relative"] else "")
            key = (host.name, target)
            if key not in emitted:
                edits.append(TextEdit(initializer.body_end, initializer.body_end,
                                      f"        {target}.requires_grad_(False)\n"))
                emitted.add(key)
            break
        else:
            raise ValueError(f"Codegen cannot express the observed frozen subtree {record['fqn']!r}")
    return apply_edits(source, edits)

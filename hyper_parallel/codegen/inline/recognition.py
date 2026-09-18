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
"""Framework structural recognition for inline component selection.

Which generic component replaces a matched source module is decided by the
*structure of the family's own replacement factory* -- the generic component it
constructs and the kernel entry it hands over -- never by the model's name.
The factory is not codegen-specific: the YAML already names it for the native
path, so reading it adds no family codegen declaration.

This is what makes "native HF model + one YAML" sufficient: a family whose
attention is fused-QKV, whose experts are batched, or whose norm is an RMSNorm
is covered the moment its runtime factory exists.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

#: Generic component kinds the framework can wire.
COMPONENT_RMS_NORM = "rms_norm"
COMPONENT_GROUPED_EXPERTS = "grouped_experts"
COMPONENT_GQA_ATTENTION = "gqa_attention"

#: Generic component class name -> component kind.
_COMPONENT_BY_CLASS = {
    "RMSNorm": COMPONENT_RMS_NORM,
    "GroupedExperts": COMPONENT_GROUPED_EXPERTS,
    "GQAAttention": COMPONENT_GQA_ATTENTION,
}


def class_name_of(qualified_type: str | None) -> str | None:
    """Return the bare class name of a dotted ``module_type`` path."""
    if not qualified_type:
        return None
    return qualified_type.rsplit(".", 1)[-1] or None


def family_from_target(target: str | None) -> str | None:
    """Infer the adapter family directory from a framework adapter path.

    ``hyper_parallel.models.<family>.adapter...`` -> ``<family>``. Returns
    ``None`` for targets that are not model-adapter paths (for example a
    framework recipe), which callers treat as "no family provider".
    """
    parts = (target or "").split(".")
    if len(parts) >= 3 and parts[:2] == ["hyper_parallel", "models"]:
        return parts[2]
    return None


def adapter_path_parts(target: str | None) -> bool:
    """Whether a target lives under the ``models/<family>/adapter`` convention."""
    parts = (target or "").split(".")
    return len(parts) >= 4 and parts[:2] == ["hyper_parallel", "models"] and parts[3] == "adapter"


def factory_component(factory: Any) -> tuple[str | None, str | None]:
    """Read which generic component a replacement factory constructs.

    Returns ``(component_kind, attention_interface_name)``. The interface name
    is the symbol the factory hands the fused attention component, read from
    its own ``attention_interface=<Name>`` keyword — so the framework never
    hardcodes a family kernel entry.
    """
    try:
        source = textwrap.dedent(inspect.getsource(factory))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None, None
    interface: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        class_name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if class_name not in _COMPONENT_BY_CLASS:
            continue
        for keyword in node.keywords:
            if keyword.arg == "attention_interface" and isinstance(keyword.value, ast.Name):
                interface = keyword.value.id
        return _COMPONENT_BY_CLASS[class_name], interface
    return None, None


def archetype_for_factory(factory_path: str | None) -> Any | None:
    """Reverse-lookup a MoE archetype by its compute-factory ``_target_``.

    A YAML ``local_compute_fn._target_`` that names a known archetype factory
    resolves to that archetype: the strategy kind / target class / boundary
    sub-patterns / EP body keys are then structural, not declared per family.
    """
    if not factory_path:
        return None
    from hyper_parallel.distributed.expert_parallel.archetypes import (  # pylint: disable=C0415
        moe_archetypes,
    )

    for archetype in moe_archetypes().values():
        if archetype.compute_factory == factory_path:
            return archetype
    return None


__all__ = [
    "COMPONENT_GQA_ATTENTION",
    "COMPONENT_GROUPED_EXPERTS",
    "COMPONENT_RMS_NORM",
    "adapter_path_parts",
    "archetype_for_factory",
    "class_name_of",
    "factory_component",
    "family_from_target",
]

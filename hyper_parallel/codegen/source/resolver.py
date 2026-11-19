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
"""Find package siblings for the modeling source observed by the native build."""

import os
import re
from collections.abc import Mapping
from typing import Any


def _relative_imports_of(path: str) -> list[str]:
    """Names a file relatively imports from its own package (one level).

    Mirrors ``transformers.dynamic_module_utils.get_relative_imports`` without
    importing transformers (source resolution must stay dependency-free).  Only
    single-dot relatives are considered: a sibling ``from .name import x`` /
    ``import .name`` resolves within the same package directory, which is what
    a ``<remote>`` modeling source ships.  Multi-dot relatives (``..x``) do not
    resolve against a flat remote code dir and are ignored.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return []
    names = re.findall(r"(?m)^\s*import\s+\.([A-Za-z_][A-Za-z0-9_]*)\s*$", content)
    names += re.findall(r"(?m)^\s*from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import", content)
    return sorted(set(names))


def resolve_remote_siblings(source: Mapping[str, Any]) -> dict[str, str]:
    """Return package siblings reachable through the observed source's relative imports.

    Remote checkpoint code and Hugging Face dynamic modules need their package
    siblings copied into the bundle. Installed modeling modules use absolute
    imports and need no copies.
    """
    if isinstance(source, Mapping):
        module_name = source.get("module_name")
        file_path = source.get("file_path")
    else:
        module_name = getattr(source, "module_name", None)
        file_path = getattr(source, "file_path", None)
    is_remote = module_name == "<remote>" or (
        isinstance(module_name, str)
        and module_name.startswith("transformers_modules.")
    )
    if not is_remote or not file_path:
        return {}
    base_dir = os.path.dirname(os.path.abspath(file_path))
    seen: dict[str, str] = {}
    queue: list[str] = [os.path.abspath(file_path)]
    while queue:
        module_file = queue.pop(0)
        for name in _relative_imports_of(module_file):
            sibling = os.path.join(base_dir, f"{name}.py")
            if not os.path.isfile(sibling):
                continue
            if sibling in seen:
                continue
            seen[sibling] = sibling
            queue.append(sibling)
    return seen


__all__ = ["resolve_remote_siblings"]

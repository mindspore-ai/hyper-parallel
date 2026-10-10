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
"""Data contract for the inline pipeline's resolved source-level specs."""

from __future__ import annotations

from dataclasses import dataclass

from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch


@dataclass(frozen=True)
class ReplacementSpec:
    """Source-level form of one ``replace_module`` target."""

    old_ctor: str
    new_ctor: str
    imports: tuple[ImportPatch, ...]
    mode: str = "name"
    keyword_args: tuple[str, ...] = ()
    snippets: tuple[ModuleSnippetPatch, ...] = ()
    remove_class: bool = True
    replacement_note: str | None = None


@dataclass(frozen=True)
class StrategySpec:
    """Source-level form of one parallel strategy target."""

    kind: str
    imports: tuple[ImportPatch, ...]
    target_class: str | None = None
    method_name: str = "forward"
    body_template: str | None = None
    snippets: tuple[ModuleSnippetPatch, ...] = ()
    strip_boundary_subpatterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require a complete method patch or an imports-only declaration."""
        if (self.target_class is None) != (self.body_template is None):
            raise ValueError("Inline strategy target_class and body_template must be provided together")
        if self.target_class is not None and (not self.target_class or not self.body_template.strip()):
            raise ValueError("Inline strategy class and body must be nonempty")
        if not self.method_name.isidentifier():
            raise ValueError("Inline strategy method_name must be a Python identifier")


__all__ = [
    "ReplacementSpec",
    "StrategySpec",
]

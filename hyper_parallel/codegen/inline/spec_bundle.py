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
    """Source-level form of one parallel strategy target.

    A ``body_template`` replaces its host class's whole ``forward`` with an
    inline orchestration body (the EP shell).  A ``subclass_shell`` instead
    renders an in-artifact *subclass* of the host whose ``forward`` is the
    dispatch body (inline CP): the host keeps its compiled boundary, so the
    shell never carries a ``target_class`` — the materialization pass derives
    the host from the frozen boundary map.  The two forms are exclusive.
    """

    kind: str
    imports: tuple[ImportPatch, ...]
    target_class: str | None = None
    method_name: str = "forward"
    body_template: str | None = None
    subclass_shell: str | None = None
    snippets: tuple[ModuleSnippetPatch, ...] = ()
    strip_boundary_subpatterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require a complete method patch or an imports-only declaration."""
        if self.target_class is not None and self.body_template is None:
            raise ValueError("Inline strategy target_class requires a body_template")
        if self.target_class is not None and (not self.target_class or not self.body_template.strip()):
            raise ValueError("Inline strategy class and body must be nonempty")
        if not self.method_name.isidentifier():
            raise ValueError("Inline strategy method_name must be a Python identifier")
        if self.subclass_shell is not None:
            if self.target_class is not None or self.body_template is not None:
                raise ValueError(
                    "Inline strategy subclass_shell is exclusive with "
                    "target_class and body_template"
                )
            if not self.subclass_shell.strip():
                raise ValueError("Inline strategy subclass shell must be nonempty")


__all__ = [
    "ReplacementSpec",
    "StrategySpec",
]

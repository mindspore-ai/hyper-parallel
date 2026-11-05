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
"""Intermediate representation for inline generated-model patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class SourceObservation:
    """One normalized replacement or injection observed during native build."""

    match: tuple[str, ...]
    module_type: Optional[str] = None
    replace_target: Optional[str] = None
    inner_wrapper_target: Optional[str] = None
    local_compute_target: Optional[str] = None


@dataclass(frozen=True)
class ImportPatch:
    """A ``from module import names`` insertion."""

    module: str
    names: tuple[str, ...]
    raw: str | None = None


@dataclass(frozen=True)
class ModuleSnippetPatch:
    """A top-level source snippet inserted after imports."""

    text: str


@dataclass(frozen=True)
class ConstructorReplacePatch:
    """Replace constructor calls in source text.

    ``scope_attrs`` carries the attribute names of the rule's *matched* FQNs
    (``{"input_layernorm", "post_attention_layernorm", "norm"}`` for the
    RMSNorm rule).  Native replaces module instances by FQN, so a whole-file
    class-name rewrite would silently replace call sites the rule never
    matched -- for example the Q/K norm inside the attention class that the
    recipe deliberately leaves as the original implementation.  An empty scope
    means the caller imposes no restriction; the YAML-driven pass never emits
    an unscoped patch.
    """

    old_ctor: str
    new_ctor: str
    mode: Literal["name", "wrap_source"] = "name"
    keyword_args: tuple[str, ...] = ()
    scope_attrs: tuple[str, ...] = ()
    scope_fqns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForwardExtractPatch:
    """Keep the original method as ``_forward_impl`` and replace its body."""

    class_name: str
    method_name: str
    body: str
    merge_original_body: bool = False


@dataclass(frozen=True)
class ClassRemovalPatch:
    """Remove an unused original HF class definition and rewrite its references.

    ``old_name`` is the class being removed; ``new_name`` is the fused
    replacement that any surviving bare reference (e.g. inside
    ``_can_record_outputs`` or ``isinstance`` checks) should be rewritten to.
    Constructor calls are handled separately by ``ConstructorReplacePatch``.
    """

    old_name: str
    new_name: str


@dataclass
class InlinePatchSet:
    """All edits emitted by one inline pass."""

    imports: list[ImportPatch] = field(default_factory=list)
    module_snippets: list[ModuleSnippetPatch] = field(default_factory=list)
    constructor_replaces: list[ConstructorReplacePatch] = field(default_factory=list)
    forward_extracts: list[ForwardExtractPatch] = field(default_factory=list)
    class_removals: list[ClassRemovalPatch] = field(default_factory=list)


@dataclass(frozen=True)
class ReplacementSpec:
    """Source-level form of one module replacement implementation."""

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
    subclass_shell: str | None = None
    subclass_state_attr: str | None = None
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

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
"""Apply source-visible forward extraction patches."""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterable

from hyper_parallel.codegen.astkit.edits import TextEdit, replace_function_body
from hyper_parallel.codegen.astkit.index import ClassInfo, FunctionInfo, build_source_index
from hyper_parallel.codegen.emit.parallel import _build_forward_impl_edit
from hyper_parallel.codegen.inline.ir import ForwardExtractPatch

_HIDDEN_ARG_TOKEN = "__hp_hidden_arg__"


def forward_extract_edits(
    source_text: str,
    patches: Iterable[ForwardExtractPatch],
) -> list[TextEdit]:
    """Build edits that preserve original forwards and install dispatch bodies."""
    patch_list = tuple(patches)
    if not patch_list:
        return []
    index = build_source_index(source_text)
    edits: list[TextEdit] = []
    for patch in patch_list:
        cls = index.find_class(patch.class_name)
        if cls is None:
            continue
        if patch.definition is not None:
            edits.append(_definition_edit(source_text, cls, patch))
            continue
        func = cls.methods.get(patch.method_name)
        if func is None:
            continue
        body = _bind_forward_args(_bind_hidden_arg(patch.body, func), func)
        prefix = _parallel_state_prefix(body)
        impl = _build_forward_impl_edit(source_text, func, prefix=prefix)
        if impl is not None:
            edits.append(impl)
        edits.extend(replace_function_body(source_text, func, body, indent="        "))
    return edits


def _definition_edit(source_text: str, cls: ClassInfo, patch: ForwardExtractPatch) -> TextEdit:
    """Install actual native methods, preserving the old method only when called."""
    names = {node.name for node in ast.parse(patch.definition).body if isinstance(node, ast.FunctionDef)}
    collisions = (names - {patch.method_name}) & cls.methods.keys()
    if collisions:
        raise RuntimeError(f"codegen observed methods collide with source methods {sorted(collisions)}")
    definition = textwrap.indent(patch.definition.strip(), "    ") + "\n"
    func = cls.methods.get(patch.method_name)
    if func is None:
        return TextEdit(cls.end_offset, cls.end_offset, "\n" + definition)
    if "self._forward_impl(" in patch.definition:
        impl = _build_forward_impl_edit(source_text, func)
        if impl is None or "_forward_impl" in cls.methods:
            raise RuntimeError("codegen cannot preserve the original method for the observed wrapper")
        original = ast.parse(textwrap.dedent(source_text[func.def_offset:func.body_end])).body[0]
        decorators = "".join(f"    @{ast.unparse(item)}\n" for item in original.decorator_list)
        definition = decorators + impl.replacement + "\n" + definition
    return TextEdit(func.def_offset, func.body_end, definition)


def _parallel_state_prefix(body: str) -> str:
    """Render static enable flags read by a generated dispatch body."""
    lines = [
        f"    {axis}_enable = False"
        for axis in ("tp", "cp", "ep")
        if f"self.{axis}_enable" in body
    ]
    return "\n".join(lines) + "\n\n" if lines else ""


def _bind_hidden_arg(body: str, func: FunctionInfo) -> str:
    """Bind a strategy shell's hidden-input placeholder to its host signature."""
    if _HIDDEN_ARG_TOKEN not in body:
        return body
    if not func.param_names:
        raise RuntimeError(
            f"codegen: inline strategy body references {_HIDDEN_ARG_TOKEN!r} "
            "but the target method has no positional parameter to bind it to"
        )
    return body.replace(_HIDDEN_ARG_TOKEN, func.param_names[0])


def _bind_forward_args(body: str, func: FunctionInfo) -> str:
    """Bind a dispatch shell's argument token to the original signature."""
    token = "__hp_forward_args__"
    args = list(func.param_names)
    if func.vararg_name:
        args.append(f"*{func.vararg_name}")
    args.extend(f"{name}={name}" for name in func.kwonly_names)
    if func.kwarg_name:
        args.append(f"**{func.kwarg_name}")
    positional = list(func.param_names)
    if func.vararg_name:
        positional.append(f"*{func.vararg_name}")
    keywords = [f"{name!r}: {name}" for name in func.kwonly_names]
    if func.kwarg_name:
        keywords.append(f"**{func.kwarg_name}")
    positional_text = ", ".join(positional) + ("," if positional else "")
    bindings = []
    if "__hp_forward_positional__" in body:
        bindings.append(f"forward_args = ({positional_text})")
        body = body.replace("__hp_forward_positional__", "forward_args")
    if "__hp_forward_keywords__" in body:
        bindings.append("forward_kwargs = {" + ", ".join(keywords) + "}")
        body = body.replace("__hp_forward_keywords__", "forward_kwargs")
    return "\n".join([*bindings, body.replace(token, ", ".join(args))])


__all__ = ["forward_extract_edits"]

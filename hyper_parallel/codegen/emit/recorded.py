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
"""Materialize builder-recorded forward patches into generated model source."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import defaultdict
from typing import Any

from hyper_parallel.codegen.astkit.edits import (
    TextEdit,
    apply_edits,
    replace_function_body,
)
from hyper_parallel.codegen.astkit.index import FunctionInfo, build_source_index
from hyper_parallel.codegen.emit.parallel import (
    _add_runtime_helper_import,
    _build_forward_impl_edit,
    _build_forward_body,
)
from hyper_parallel.codegen.plan.boundary_forms import (
    FORM_GENERIC,
    FORM_REGION,
    classify_boundary_form,
)


def materialize_recorded_forwards(source_text: str, meta: Any) -> str:
    """Lower recorded local-compute decisions without model-specific tables."""
    patches = getattr(meta, "_build_patches", None)
    records = getattr(patches, "local_computes", ()) if patches is not None else ()
    if not records:
        return source_text

    groups: dict[tuple[type, str], list[Any]] = defaultdict(list)
    for record in records:
        original = _unbound(record.original_forward)
        groups[(record.module_type, _source_function_name(original))].append(record)

    rendered = source_text
    for (host_type, source_function), group in sorted(
        groups.items(), key=lambda item: (item[0][0].__name__, item[0][1])
    ):
        _require_one_contract(group, meta)
        if source_function and _has_module_function(rendered, source_function):
            rendered = _materialize_bound_module_function(
                rendered, source_function, group[0], meta
            )
            continue
        if _has_source_class(rendered, host_type.__name__):
            rendered = _materialize_source_class(
                rendered, host_type.__name__, group[0], meta
            )
            for record in group:
                _entry_for_record(meta, record)["skip_source_lowering"] = True
            continue
        rendered = _materialize_imported_host(rendered, host_type, group[0], meta)
    return _add_runtime_helper_import(rendered)


def _unbound(value: Any) -> Any:
    return getattr(value, "__func__", value)


def _source_function_name(value: Any) -> str:
    """Return a top-level source function name, excluding class methods."""
    name = getattr(value, "__qualname__", "")
    return name if name and "." not in name else ""


def _has_module_function(source_text: str, name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.parse(source_text).body
    )


def _has_source_class(source_text: str, name: str) -> bool:
    return any(
        isinstance(node, ast.ClassDef) and node.name == name
        for node in ast.parse(source_text).body
    )


def _require_one_contract(records: list[Any], meta: Any) -> None:
    """A shared source forward can represent only one boundary contract."""
    contracts = {
        repr({
            key: value
            for key, value in _entry_for_record(meta, record).items()
            if key != "params"
        })
        for record in records
    }
    factories = {record.factory for record in records}
    if len(contracts) != 1 or len(factories) != 1:
        fqns = ", ".join(sorted(record.module_fqn for record in records))
        raise RuntimeError(
            "codegen: recorded modules sharing one forward have incompatible "
            f"patch contracts: {fqns}"
        )


def _entry_for_record(meta: Any, record: Any) -> dict[str, Any]:
    entry = (getattr(meta, "param_plan", None) or {}).get(record.module_fqn)
    if entry is None:
        raise RuntimeError(
            f"codegen: recorded patch {record.module_fqn!r} has no frozen plan entry"
        )
    return entry


def _injection_for_record(meta: Any, record: Any) -> dict[str, Any] | None:
    for injection in getattr(meta, "injections", None) or ():
        if injection.get("match") == record.module_fqn:
            return injection
    return None


def _function_info(node: ast.FunctionDef | ast.AsyncFunctionDef, source_text: str) -> FunctionInfo:
    lines = source_text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def start(line_number: int) -> int:
        return offsets[line_number - 1]

    positional = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
    if positional and positional[0] in ("self", "module"):
        positional.pop(0)
    first_line = min(
        [node.lineno, *(item.lineno for item in node.decorator_list)]
    )
    return FunctionInfo(
        name=node.name,
        body_start=start(node.body[0].lineno),
        body_end=offsets[node.end_lineno],
        def_offset=start(first_line),
        param_names=positional,
        kwonly_names=[arg.arg for arg in node.args.kwonlyargs],
        vararg_name=node.args.vararg.arg if node.args.vararg else None,
        kwarg_name=node.args.kwarg.arg if node.args.kwarg else None,
    )


def _recorded_body(source_text: str, func: FunctionInfo, record: Any, meta: Any) -> str:
    entry = _entry_for_record(meta, record)
    injection = _injection_for_record(meta, record)
    form = classify_boundary_form(
        entry,
        tuple(getattr(meta, "mesh_dim_names", None) or ()),
        injection,
    )
    emitted = FORM_REGION if record.factory is not None else FORM_GENERIC
    return _build_forward_body(source_text, injection, func, form, emitted)


def _materialize_bound_module_function(
    source_text: str,
    function_name: str,
    record: Any,
    meta: Any,
) -> str:
    """Turn an instance-bound module function into impl + explicit wrapper."""
    tree = ast.parse(source_text)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    info = _function_info(node, source_text)
    impl_name = f"{function_name}_hp_impl"
    original = source_text[info.def_offset:info.body_end]
    original = original.replace(f"def {function_name}(", f"def {impl_name}(", 1)
    if not original.endswith("\n"):
        original += "\n"
    body = textwrap.dedent(_recorded_body(source_text, info, record, meta))
    body_edit = TextEdit(info.body_start, info.body_end, textwrap.indent(body, "    "))
    rendered = apply_edits(
        source_text,
        [TextEdit(info.def_offset, info.def_offset, original + "\n"), body_edit],
    )
    return _bind_module_impl(rendered, function_name, impl_name)


def _materialize_source_class(
    source_text: str,
    class_name: str,
    record: Any,
    meta: Any,
) -> str:
    """Rewrite a source-defined host even when old meta normalization skips it."""
    func = build_source_index(source_text).find_forward(class_name)
    if func is None:
        raise RuntimeError(
            f"codegen: recorded source host {class_name!r} defines no forward"
        )
    body = _recorded_body(source_text, func, record, meta)
    impl = _build_forward_impl_edit(source_text, func)
    edits = list(replace_function_body(source_text, func, body, indent=""))
    if impl is not None:
        edits.append(impl)
    return apply_edits(source_text, edits)


def _bind_module_impl(source_text: str, wrapper_name: str, impl_name: str) -> str:
    """Add the instance ``_forward_impl`` binding beside MethodType(forward)."""
    tree = ast.parse(source_text)
    lines = source_text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits: list[TextEdit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == "forward"
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "MethodType"
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == wrapper_name
        ):
            continue
        owner = ast.unparse(target.value)
        indent = " " * node.col_offset
        binding = f"{indent}{owner}._forward_impl = MethodType({impl_name}, {owner})\n"
        edits.append(TextEdit(offsets[node.lineno - 1], offsets[node.lineno - 1], binding))
    if not edits:
        raise RuntimeError(
            f"codegen: source function {wrapper_name!r} is not installed with MethodType"
        )
    return apply_edits(source_text, edits)


def _materialize_imported_host(
    source_text: str,
    host_type: type,
    record: Any,
    meta: Any,
) -> str:
    """Generate a source-owned subclass and replace its constructor calls."""
    host_name = host_type.__name__
    generated_name = f"HPCodegen{host_name}"
    if f"class {generated_name}(" in source_text:
        return source_text
    signature = inspect.signature(_unbound(record.original_forward))
    info = _signature_info(signature)
    call = _render_super_call(signature)
    body = _recorded_body(source_text, info, record, meta)
    snippet = "\n".join([
        f"class {generated_name}({host_name}):",
        "    \"\"\"Source-owned form of the forward patch installed by the trainer.\"\"\"",
        "",
        f"    def _forward_impl{signature}:",
        f"        return super().forward({call})",
        "",
        f"    def forward{signature}:",
        body,
        "",
    ])
    rendered = _insert_before_first_class(source_text, snippet)
    return _replace_constructor_calls(rendered, host_name, generated_name)


def _signature_info(signature: inspect.Signature) -> FunctionInfo:
    positional = []
    keyword_only = []
    vararg = None
    kwarg = None
    for parameter in signature.parameters.values():
        if parameter.name in ("self", "module"):
            continue
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional.append(parameter.name)
        elif parameter.kind == parameter.KEYWORD_ONLY:
            keyword_only.append(parameter.name)
        elif parameter.kind == parameter.VAR_POSITIONAL:
            vararg = parameter.name
        elif parameter.kind == parameter.VAR_KEYWORD:
            kwarg = parameter.name
    return FunctionInfo("forward", 0, 0, 0, positional, keyword_only, vararg, kwarg)


def _render_super_call(signature: inspect.Signature) -> str:
    args = []
    for parameter in signature.parameters.values():
        if parameter.name in ("self", "module"):
            continue
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            args.append(parameter.name)
        elif parameter.kind == parameter.VAR_POSITIONAL:
            args.append(f"*{parameter.name}")
        elif parameter.kind == parameter.KEYWORD_ONLY:
            args.append(f"{parameter.name}={parameter.name}")
        elif parameter.kind == parameter.VAR_KEYWORD:
            args.append(f"**{parameter.name}")
    return ", ".join(args)


def _insert_before_first_class(source_text: str, snippet: str) -> str:
    tree = ast.parse(source_text)
    first = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
    if first is None:
        raise RuntimeError("codegen: generated model source defines no class")
    lines = source_text.splitlines(keepends=True)
    offset = sum(len(line) for line in lines[:first.lineno - 1])
    return apply_edits(source_text, [TextEdit(offset, offset, snippet + "\n\n")])


def _replace_constructor_calls(source_text: str, old_name: str, new_name: str) -> str:
    tree = ast.parse(source_text)
    lines = source_text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == old_name:
            start = offsets[node.func.lineno - 1] + node.func.col_offset
            edits.append(TextEdit(start, start + len(old_name), new_name))
    if not edits:
        raise RuntimeError(
            f"codegen: imported recorded host {old_name!r} has no constructor call to replace"
        )
    return apply_edits(source_text, edits)


__all__ = ["materialize_recorded_forwards"]

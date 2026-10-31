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
"""Materialize builder-observed forward patches into generated model source."""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from collections import defaultdict
from types import SimpleNamespace
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
    _render_forward_call,
    _render_input_redistribute,
    _render_local_compute,
    _render_output_redistribute,
)
from hyper_parallel.codegen.plan.boundary_forms import (
    FORM_GENERIC,
    FORM_REGION,
    classify_boundary_form,
)


def materialize_recorded_forwards(
    source_text: str,
    meta: Any,
    *,
    patch_plan: Any = None,
    source_path: str | None = None,
) -> str:
    """Lower observed local-compute decisions into their source-owned classes."""
    patches = patch_plan
    records = getattr(patches, "local_computes", ()) if patches is not None else ()
    if not records:
        return _materialize_declared_local_computes(source_text, meta)

    _restore_recorded_boundary_ownership(meta, records)

    groups: dict[tuple[type, str], list[Any]] = defaultdict(list)
    for record in records:
        if source_path is not None and not _record_belongs_to_source(
            record, source_path
        ):
            continue
        original = _unbound(record.original_forward)
        groups[(record.module_type, _source_function_name(original))].append(record)

    rendered = source_text
    materialized = False
    for (host_type, source_function), group in sorted(
        groups.items(), key=lambda item: (item[0][0].__name__, item[0][1])
    ):
        _require_one_contract(group, meta)
        if source_function and _has_module_function(rendered, source_function):
            rendered = _materialize_bound_module_function(
                rendered, source_function, group[0], meta
            )
            materialized = True
            continue
        if not _has_source_class(rendered, host_type.__name__):
            continue
        rendered = _materialize_source_class(
            rendered, host_type.__name__, group[0], meta
        )
        materialized = True
        _mark_class_source_lowering_skipped(meta, host_type.__name__)
    return _add_runtime_helper_import(rendered) if materialized else rendered


def _materialize_declared_local_computes(source_text: str, meta: Any) -> str:
    """Lower explicit frozen local-compute injections when no live record exists."""
    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    injections = getattr(meta, "injections", None) or ()
    source_classes = {
        node.name
        for node in ast.parse(source_text).body
        if isinstance(node, ast.ClassDef)
    }
    grouped: dict[str, list[Any]] = defaultdict(list)
    for injection in injections:
        local = injection.get("local_compute_fn")
        module_fqn = injection.get("match")
        class_name = boundary_classes.get(module_fqn)
        if local is None or not module_fqn or class_name not in source_classes:
            continue
        grouped[class_name].append(SimpleNamespace(
            module_fqn=module_fqn,
            factory=_callable_identity(local),
        ))
    rendered = source_text
    materialized = getattr(meta, "materialized_patches", None)
    for class_name, records in sorted(grouped.items()):
        _require_one_contract(records, meta)
        rendered = _materialize_source_class(
            rendered, class_name, records[0], meta
        )
        _mark_class_source_lowering_skipped(meta, class_name)
        for record in records:
            if isinstance(materialized, list):
                materialized.append({
                    "fqn": record.module_fqn,
                    "host_class": class_name,
                    "kind": "local_compute",
                    "source_form": "same_class_toggle_forward",
                })
    return _add_runtime_helper_import(rendered) if grouped else rendered


def _mark_class_source_lowering_skipped(meta: Any, class_name: str) -> None:
    """Keep generic boundary lowering off a source class with an EP toggle.

    A source-visible local-compute patch rewrites the class method, not just
    one module instance. Every boundary instance of that class must therefore
    use the runtime per-instance boundary wrapper. Lowering another instance's
    boundary into the shared method would extract a second ``_forward_impl``
    and wrap the EP toggle a second time.
    """
    param_plan = getattr(meta, "param_plan", None) or {}
    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    for module_fqn, owner_class in boundary_classes.items():
        if owner_class != class_name:
            continue
        entry = param_plan.get(module_fqn)
        if isinstance(entry, dict) and entry.get("is_boundary"):
            entry["skip_source_lowering"] = True


def _callable_identity(value: Any) -> str:
    """Return a stable identity for a frozen callable declaration."""
    if isinstance(value, dict):
        return str(value.get("_target_") or value.get("target") or value.get("name"))
    return str(value)


def _record_belongs_to_source(record: Any, source_path: str) -> bool:
    """Return whether a recorded host is defined by this source unit."""
    candidates = (
        inspect.getsourcefile(record.module_type),
        inspect.getsourcefile(_unbound(record.original_forward)),
    )
    expected = os.path.normcase(os.path.abspath(source_path))
    return any(
        candidate is not None
        and os.path.normcase(os.path.abspath(candidate)) == expected
        for candidate in candidates
    )


def _restore_recorded_boundary_ownership(meta: Any, records: Any) -> None:
    """Make recorded local-region forwards runtime-owned boundaries again.

    The generic inline pass initially classifies known EP strategy targets as
    external-state classes. When builder recording supplies the actual forward,
    this emitter materializes a compiled-boundary local region instead of that
    generic inline body. Restore the matching frozen entries so runtime binding
    installs ``_hp_boundary`` and the recorded compute method they reference.

    Args:
        meta: Mutable generation metadata.
        records: Builder-recorded local-compute patches.
    """
    param_plan = getattr(meta, "param_plan", None)
    boundary_classes = getattr(meta, "boundary_classes", None)
    external_classes = set(getattr(meta, "external_state_classes", None) or ())
    recorded_classes: set[str] = set()
    for record in records:
        class_name = record.module_type.__name__
        recorded_classes.add(class_name)
        if isinstance(param_plan, dict):
            entry = param_plan.get(record.module_fqn)
            if isinstance(entry, dict):
                entry["is_boundary"] = True
        if isinstance(boundary_classes, dict):
            boundary_classes[record.module_fqn] = class_name
    if hasattr(meta, "external_state_classes"):
        meta.external_state_classes = sorted(external_classes - recorded_classes)


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
    local = injection.get("local_compute_fn") if injection else None
    if local is not None:
        body = _render_local_compute(
            local,
            _render_input_redistribute(func),
            _render_output_redistribute(),
            form,
        )
    else:
        body = _build_forward_body(source_text, injection, func, form, emitted)
    fallback = _render_forward_call(func).replace("outputs =", "return", 1)
    return "\n".join([
        "        if self.ep_enable:",
        textwrap.indent(body, "    "),
        "",
        fallback,
    ])


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
    impl = _build_forward_impl_edit(
        source_text,
        func,
        prefix="    ep_enable = False\n\n",
    )
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


def _fqn_sort_key(fqn: str) -> tuple[tuple[int, int, str], ...]:
    """Natural sort key for plan FQNs — layer indices order by value.

    A plain lexical sort renders ``layers.10`` before ``layers.2``; the binder
    stays reviewable only when recorded layer paths appear in numeric order.
    """
    return tuple(
        (0, int(segment), "") if segment.isdigit() else (1, 0, segment)
        for segment in fqn.split(".")
    )


__all__ = ["materialize_recorded_forwards"]

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
import textwrap
from collections import defaultdict
from typing import Any

from hyper_parallel.codegen.astkit.edits import (
    TextEdit,
    apply_edits,
    insert_after_imports,
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


def materialize_recorded_forwards(
    source_text: str,
    meta: Any,
    *,
    patch_plan: Any = None,
) -> str:
    """Lower observed local-compute decisions without model-specific tables."""
    patches = patch_plan if patch_plan is not None else getattr(meta, "_build_patches", None)
    records = getattr(patches, "local_computes", ()) if patches is not None else ()
    if not records:
        return source_text

    _restore_recorded_boundary_ownership(meta, records)

    groups: dict[tuple[type, str], list[Any]] = defaultdict(list)
    for record in records:
        original = _unbound(record.original_forward)
        groups[(record.module_type, _source_function_name(original))].append(record)

    rendered = source_text
    bindings: dict[str, list[str]] = {}
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
        rendered, constructor_replaced = _materialize_imported_host(
            rendered, host_type, group[0], meta
        )
        if not constructor_replaced:
            # The host is instantiated by an imported subtree (no source-level
            # constructor call to retarget): every recorded module reaches the
            # generated subclass only through the explicit FQN binder below.
            bindings.setdefault(host_type.__name__, []).extend(
                record.module_fqn for record in group
            )
    if bindings:
        rendered = _emit_recorded_forward_bindings(rendered, bindings)
    return _add_runtime_helper_import(rendered)


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
) -> tuple[str, bool]:
    """Generate a source-owned subclass and retarget its construction sites.

    Returns ``(rendered, constructor_replaced)``.  The subclass and its host
    import are always emitted (the host class lives in an imported module —
    the recorded ``__module__`` is the only source of truth for the import).
    ``constructor_replaced`` is ``False`` when the host is instantiated by an
    imported subtree, so no source-level constructor call exists to retarget;
    the caller then binds the recorded FQNs through the explicit binder.
    """
    host_name = host_type.__name__
    generated_name = f"HPCodegen{host_name}"
    if f"class {generated_name}(" in source_text:
        return source_text, f"{generated_name}(" in source_text
    host_module = getattr(host_type, "__module__", None)
    if not host_module or host_module == "__main__":
        raise RuntimeError(
            f"codegen: recorded host {host_name!r} has no importable module"
        )
    rendered = source_text
    for import_line in _host_import_lines(host_module, host_name):
        if import_line in rendered:
            continue
        rendered = _ensure_import(rendered, import_line, host_name)
    signature = inspect.signature(_unbound(record.original_forward))
    info = _signature_info(signature)
    call = _render_super_call(signature)
    body = _recorded_body(rendered, info, record, meta)
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
    rendered = _insert_before_first_class(rendered, snippet)
    return _replace_constructor_calls(rendered, host_name, generated_name)


def _emit_recorded_forward_bindings(
    source_text: str,
    bindings: dict[str, list[str]],
) -> str:
    """Emit one reviewable binder that binds each recorded FQN explicitly.

    ``bindings`` maps an imported host class name to the module FQNs the
    builder actually patched.  The host instances are created by an imported
    subtree (``AutoModel.from_config``-style construction), so the generated
    module reaches them only through the recorded paths — one explicit
    ``MethodType`` binding per FQN, no type registry and no scanning.
    """
    lines = [
        "def hp_bind_recorded_forwards(model):",
        "    \"\"\"Bind builder-recorded forward patches onto their recorded modules.\"\"\"",
    ]
    for host_name in sorted(bindings):
        generated_name = f"HPCodegen{host_name}"
        for fqn in sorted(bindings[host_name], key=_fqn_sort_key):
            accessor = _fqn_accessor(fqn)
            lines.extend([
                f"    module = {accessor}",
                f"    module._forward_impl = MethodType({generated_name}._forward_impl, module)",
                f"    module.forward = MethodType({generated_name}.forward, module)",
            ])
    rendered = _ensure_import(
        source_text, "from types import MethodType", "MethodType"
    )
    return rendered.rstrip() + "\n\n\n" + "\n".join(lines) + "\n"


def _fqn_accessor(fqn: str) -> str:
    """Render a plan FQN as a Python accessor expression rooted at ``model``.

    Mirrors the runtime ``_resolve_module`` semantics segment by segment:
    numeric segments index a module list, every other segment is a plain
    attribute (the first FQN segment names the root's submodule).
    """
    accessor = "model"
    for segment in fqn.split("."):
        if segment.isdigit():
            accessor += f"[{segment}]"
        else:
            accessor += f".{segment}"
    return accessor


def _fqn_sort_key(fqn: str) -> tuple[tuple[int, int, str], ...]:
    """Natural sort key for plan FQNs — layer indices order by value.

    A plain lexical sort renders ``layers.10`` before ``layers.2``; the binder
    stays reviewable only when recorded layer paths appear in numeric order.
    """
    return tuple(
        (0, int(segment), "") if segment.isdigit() else (1, 0, segment)
        for segment in fqn.split(".")
    )


def _host_import_lines(host_module: str, host_name: str) -> tuple[str, ...]:
    """Import lines binding ``host_name`` from the recorded ``host_module``.

    HF remote-code cache paths can contain hash components that start with a
    digit (``transformers_modules.kimi_k3.7dc8....modeling_kimi_linear``); a
    ``from`` statement over such a path is a syntax error, so identifier-unsafe
    modules import through ``importlib.import_module`` with the path as a
    string literal instead.
    """
    if all(part.isidentifier() for part in host_module.split(".")):
        return (f"from {host_module} import {host_name}",)
    return (
        "import importlib as _hp_importlib",
        f'{host_name} = getattr(_hp_importlib.import_module("{host_module}"), "{host_name}")',
    )


def _ensure_import(source_text: str, import_line: str, symbol: str) -> str:
    """Insert ``import_line`` after the import block unless ``symbol`` is bound."""
    if _imports_symbol(source_text, symbol):
        return source_text
    index = build_source_index(source_text)
    edit = insert_after_imports(
        source_text,
        index.import_end,
        import_line,
        default_offset=index.docstring_end or 0,
    )
    return apply_edits(source_text, [edit]) if edit is not None else source_text


def _imports_symbol(source_text: str, symbol: str) -> bool:
    """Return whether a top-level import already binds ``symbol``."""
    for node in ast.parse(source_text).body:
        if isinstance(node, ast.Import) and any(
            alias.asname == symbol
            or (alias.asname is None and alias.name == symbol)
            for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and any(
            alias.asname == symbol
            or (alias.asname is None and alias.name == symbol)
            for alias in node.names
        ):
            return True
    return False


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
    # A decorated class's ``lineno`` is the ``class`` line itself; anchor on
    # the first decorator instead so the insertion cannot steal the target's
    # decorator block.
    first_line = min([first.lineno, *(item.lineno for item in first.decorator_list)])
    offset = sum(len(line) for line in lines[:first_line - 1])
    return apply_edits(source_text, [TextEdit(offset, offset, snippet + "\n\n")])


def _replace_constructor_calls(
    source_text: str, old_name: str, new_name: str
) -> tuple[str, bool]:
    """Retarget every source-level ``old_name(...)`` construction to ``new_name``.

    Returns ``(rendered, replaced)``; ``replaced`` is ``False`` when the host
    has no source-level construction site (imported-subtree instantiation).
    """
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
        return source_text, False
    return apply_edits(source_text, edits), True


__all__ = ["materialize_recorded_forwards"]

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
"""Generate a self-contained attention class from real component source.

The generated artifact preserves the genuine construction and fused QKV weight
layout (``linear_qkv`` + ``InterleaveQKV`` checkpoint conversion) by copying the
real ``GQAAttention`` methods verbatim; ``forward`` is emitted exactly as the
native class writes it, keeping its ``self.attention_interface(...)`` call.  The
kernel entry function that the real class normally receives as
``attention_interface`` is inlined from its real source (plan §6) so the
artifact stays readable down to the single hardware kernel
``torch_npu.npu_fusion_attention``.

Neither CP nor TP orchestration is inlined into this rendered class, and the
class is not an external-state class: it keeps the component's own forward, so
the emitted boundary form wraps ``_forward_impl`` around it (``emit.parallel``
lowering + ``hp_install_boundaries``) and the declared CP inner wrapper is
installed by ``hp_apply_inner_wrapper`` — the same shared mechanisms native
uses at apply time.

This generator is platform-agnostic ``core`` code: the module type and kernel
entry are handed in, never imported here.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import symtable
import textwrap
from types import ModuleType
from typing import Any, Callable, Iterable

from hyper_parallel.codegen.inline.expansion import ExpandedSource, expand_function


#: The generated forward is the native GQAAttention.forward verbatim — its
#: ``self.attention_interface(...)`` call site is kept unmodified, and neither CP
#: nor TP orchestration is inlined into it.  The rendered class keeps the
#: component's own forward, so the emitted boundary form owns the TP
#: redistribution (``emit.parallel`` lowering, installed by
#: ``hp_install_boundaries``) and the CP inner wrapper is installed by
#: ``hp_apply_inner_wrapper``.


_FUSION_REPLACEMENT_NOTE = (
    "# [HYPER INLINE] fused replacement for the original model's per-head\n"
    "# projections: q_proj/k_proj/v_proj are packed into a single linear_qkv\n"
    "# weight, whose checkpoint layout is converted via InterleaveQKV.\n"
)


def _fusion_replacement_note(class_name: str) -> str:
    """Annotate that the emitted fused module replaces the original projections.

    The artifact keeps the real fused class (``linear_qkv`` takes over the
    weight the original model spread across ``q_proj``/``k_proj``/``v_proj``),
    so a reader of the generated model file sees a genuine class rather than a
    black-box substitution, with a comment naming what the fusion swallowed.
    """
    return f"# [HYPER INLINE] {class_name}: {_FUSION_REPLACEMENT_NOTE}"


def _build_forward(module_type: type) -> ast.FunctionDef:
    """Derive forward from real source verbatim.

    The body is the real ``forward`` exactly as written (its native
    ``self.attention_interface(...)`` call site is left completely unmodified).
    Neither CP nor TP orchestration is inlined here: the emitted boundary form
    redistributes around this forward, and the CP wrapper swaps
    ``attention_interface`` at install time — exactly as native's applier does
    at apply time.
    """
    source = textwrap.dedent(inspect.getsource(module_type.forward))
    tree = ast.parse(source)
    forward = tree.body[0]
    if not isinstance(forward, ast.FunctionDef) or forward.name != "forward":
        raise ValueError(f"codegen attention: {module_type.__name__}.forward is not a plain def")

    _strip_annotations(forward)
    ast.fix_missing_locations(forward)
    return forward


def _strip_annotations(statement: ast.FunctionDef) -> None:
    """Drop type hints so the artifact has no import-time resolution burden.

    Only annotations are cleared; ``*args`` and ``**kwargs`` parameters are kept
    because real forward bodies may reference them.
    """
    statement.returns = None
    for arg in (*statement.args.posonlyargs, *statement.args.args, *statement.args.kwonlyargs):
        arg.annotation = None
    if statement.args.vararg is not None:
        statement.args.vararg.annotation = None
    if statement.args.kwarg is not None:
        statement.args.kwarg.annotation = None


def _drop_docstring(statement: ast.FunctionDef) -> None:
    """Drop a leading string-expression docstring from an emitted definition."""
    if not statement.body:
        return
    first = statement.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        del statement.body[0]


def _referenced_global_names(source: str) -> set[str]:
    """Module-scope names ``source`` reads from every scope it defines.

    ``symtable`` resolves the scopes, so parameters, locals, loop variables,
    comprehension targets and attribute accesses (``x.y`` contributes only
    ``x``) never show up as global reads.
    """
    referenced: set[str] = set()
    pending = [symtable.symtable(source, "<codegen attention>", "exec")]
    while pending:
        table = pending.pop()
        referenced.update(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_global() and symbol.is_referenced()
        )
        pending.extend(table.get_children())
    return referenced


def _module_bound_names(source: str) -> set[str]:
    """Names the module-level statements of ``source`` bind."""
    table = symtable.symtable(source, "<codegen attention>", "exec")
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }


def _import_bound_names(lines: Iterable[str]) -> set[str]:
    """Names a sequence of ``import`` lines binds at module scope."""
    names: set[str] = set()
    for line in lines:
        for alias in ast.parse(line).body[0].names:
            names.add(alias.asname or alias.name.split(".")[0])
    return names


def _is_module_global(name: str) -> bool:
    """Whether ``name`` resolves in any module without an import of ours.

    Builtins do, and so do dunders such as ``__name__``.  A component definition
    of the same name wins over this: it is looked up first.
    """
    return hasattr(builtins, name) or name.startswith("__")


def _owner_definitions(owner: ModuleType) -> dict[str, ast.stmt]:
    """Module-level definitions of the component's own module, by bound name."""
    definitions: dict[str, ast.stmt] = {}
    for node in ast.parse(inspect.getsource(owner)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
            continue
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                definitions[target.id] = node
    return definitions


def _owner_import_lines(owner: ModuleType) -> dict[str, str]:
    """Import statement of the component's own module, per name it binds.

    The artifact must import what the component's *module* imports (a re-export
    such as ``WeightConverter`` only exists under the path that module uses), so
    each line is copied from that module's own source rather than reconstructed
    from the value's ``__module__``.
    """
    lines: dict[str, str] = {}
    for node in ast.parse(inspect.getsource(owner)).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                line = f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                lines.setdefault(bound, line)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                bound = alias.asname or alias.name
                line = f"from {node.module} import {alias.name}" + (
                    f" as {alias.asname}" if alias.asname else ""
                )
                lines.setdefault(bound, line)
    return lines


def _definition_source(node: ast.stmt) -> str:
    """Copy one module-level definition, dropping hints and docstrings."""
    if isinstance(node, ast.FunctionDef):
        _strip_annotations(node)
        _drop_docstring(node)
    return ast.unparse(node)


def _component_dependencies(
    module_type: type, code: str, *, provided: Iterable[str]
) -> tuple[str, tuple[str, ...]]:
    """Inline and import every name ``code`` reads from the component's module.

    Copying a class body copies its references too, so a module-level helper
    added to the component after this renderer was written would be emitted as an
    undefined name.  Each global name the copied code reads is therefore resolved
    against the component's own source module (``inspect.getmodule``, never an
    import of the component here): a module-level definition is inlined -- its own
    references first, so a helper referencing another helper completes the
    closure -- and a name that module imports is imported the same way.  A name
    that resolves to neither fails generation rather than emitting an artifact
    with a latent ``NameError``.

    Args:
        module_type: The real component whose source module is the access path.
        code: Emitted artifact text (the inlined class) to close over.
        provided: Names other emitted parts already bind in the artifact.

    Returns:
        The inlined definitions and the import lines the copied code needs.
    """
    owner = inspect.getmodule(module_type)
    if owner is None:
        raise ValueError(
            f"codegen attention: cannot resolve the source module of {module_type.__name__!r}"
        )
    definitions = _owner_definitions(owner)
    owner_imports = _owner_import_lines(owner)
    inlined: list[str] = []
    imports: list[str] = []
    resolved = set(provided)
    unresolved: list[str] = []

    def resolve(name: str) -> None:
        """Inline or import ``name``, or record it as unresolvable."""
        if name in resolved:
            return
        node = definitions.get(name)
        if node is not None:
            resolved.add(name)  # before recursing: a helper may reference itself
            for dependency in sorted(_referenced_global_names(ast.unparse(node))):
                resolve(dependency)
            inlined.append(_definition_source(node))
            return
        line = owner_imports.get(name)
        if line is not None:
            resolved.add(name)
            if line not in imports:
                imports.append(line)
            return
        if _is_module_global(name):
            resolved.add(name)
            return
        unresolved.append(name)

    for name in sorted(_referenced_global_names(code)):
        resolve(name)
    if unresolved:
        raise ValueError(
            f"codegen attention: {module_type.__name__} references "
            f"{sorted(set(unresolved))} which neither {owner.__name__!r} defines nor "
            "imports; the generated class would carry an undefined name"
        )
    return "\n\n\n".join(inlined), tuple(imports)


def _class_methods(module_type: type) -> list[ast.stmt]:
    """Copy every real method except ``forward``, dropping the module decorator."""
    source = textwrap.dedent(inspect.getsource(module_type))
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == module_type.__name__:
            methods = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.FunctionDef) and statement.name == "forward"
                )
            ]
            for statement in methods:
                statement.decorator_list = []
                if isinstance(statement, ast.FunctionDef):
                    _strip_annotations(statement)
            return methods
    raise ValueError(f"codegen attention: no class {module_type.__name__!r} in its source module")


def render_attention_class(module_type: type, *, interface: Callable[..., Any]) -> ExpandedSource:
    """Render a self-contained attention class with a visible TP boundary.

    The copied class methods and the module-level helpers they call are closed
    over the component's own source module, so the artifact carries everything it
    references: helpers are inlined and imports derived there, never listed here.

    Args:
        module_type: The real grouped-query attention component (e.g. ``modules.GQAAttention``).
        interface: The attention kernel entry function to inline (e.g.
            ``run_qwen3_moe_flash_attention``); its closure is expanded into the
            artifact and it must stop at the hardware kernel.
    """
    expanded_interface = expand_function(interface)
    forward = _build_forward(module_type)
    class_node = ast.ClassDef(
        name=module_type.__name__,
        bases=[ast.Name(id="nn.Module", ctx=ast.Load())],
        keywords=[],
        body=_class_methods(module_type) + [forward],
        decorator_list=[],
    )
    built_class = _fusion_replacement_note(module_type.__name__) + ast.unparse(class_node)
    shared_defs = expanded_interface.source
    helpers, imports = _component_dependencies(
        module_type,
        built_class,
        provided=(_module_bound_names(shared_defs) | _import_bound_names(expanded_interface.imports)),
    )
    return ExpandedSource(
        (*expanded_interface.imports, *imports),
        "\n\n\n".join(part for part in (helpers, shared_defs, built_class) if part),
    )

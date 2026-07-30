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
"""Case collection: ``register`` / ``register_op_family`` / loader."""
import ast
import importlib
import inspect
import sys
from typing import Dict, List, Optional

from tests.shard_ops.framework.case_spec import (
    CaseSpec,
    InputSpec,
    OpShardCase,
    OpSpec,
)

# Module-level collector. cases/*.py call register/register_op_family at
# import time; suite.py drives the collector via load_cases_from_package().
_COLLECTOR: List[OpShardCase] = []

# Per-package cache so the second call to load_cases_from_package() does not
# silently return an empty list just because Python's module cache short-
# circuits the second importlib.import_module.
_PKG_CACHE: Dict[str, List[OpShardCase]] = {}


def _reset() -> None:
    _COLLECTOR.clear()


def _validate(case: OpShardCase) -> None:
    """Validate case name uniqueness and input/placement count match."""
    if not case.name:
        raise ValueError("OpShardCase.name must be non-empty")
    if len(case.inputs) != len(case.placements):
        raise ValueError(
            f"case {case.name!r}: inputs ({len(case.inputs)}) and "
            f"placements ({len(case.placements)}) length mismatch"
        )
    seen = {c.name for c in _COLLECTOR}
    if case.name in seen:
        raise ValueError(f"duplicate case name: {case.name!r}")


def register(case: OpShardCase) -> None:
    """Register a single ``OpShardCase``. Used directly for ad-hoc ops."""
    if not case.source_module:
        # Record the module that defined this case so CLI can later filter
        # by file path (e.g. ``case_sort.py::sort[2d_dp]``).
        frame = inspect.currentframe()
        while frame:
            mod = frame.f_globals.get("__name__", "")
            if mod != __name__:
                case.source_module = mod
                break
            frame = frame.f_back
    _validate(case)
    _COLLECTOR.append(case)


def register_op_family(ops: List[OpSpec], cases: List[CaseSpec]) -> None:
    """Cross-multiply ``ops × cases`` and register each combination.

    A case can opt-out via ``only_for`` / ``skip_for``. The resulting
    ``OpShardCase.name`` is ``"{op.name}[{case.name}]"`` so single-case
    selection via ``-k`` or ``--case`` keeps working.
    """
    for op in ops:
        for case in cases:
            if case.only_for and op.name not in case.only_for:
                continue
            if op.name in case.skip_for:
                continue
            register(_expand(op, case))


def _expand(op: OpSpec, case: CaseSpec) -> OpShardCase:
    merged = dict(op.default_input)
    if case.init_override is not None:
        merged["init"] = case.init_override
    inputs = [
        InputSpec(shape=tuple(s), seed=case.seed, **merged)
        for s in case.shapes
    ]
    return OpShardCase(
        name=f"{op.name}[{case.name}]",
        fn=op.fn,
        inputs=inputs,
        placements=list(case.placements),
        kwargs=dict(case.kwargs),
        extra_inputs=list(case.extra_inputs),
        compare=case.compare_override or op.default_compare,
        tags=op.tags,
    )


def load_cases_from_package(pkg_path: str,
                            force_reload: bool = False) -> List[OpShardCase]:
    """Import the package, triggering register/register_op_family in each
    ``case_*.py``. Returns a list of registered cases.

    Repeat calls hit a per-package cache so they stay cheap and consistent
    even though Python's module cache short-circuits the second import
    (which would otherwise yield an empty collector).
    """
    if not force_reload and pkg_path in _PKG_CACHE:
        return list(_PKG_CACHE[pkg_path])
    _reset()
    if force_reload:
        _drop_cached_submodules(pkg_path)
    importlib.import_module(pkg_path)
    collected = list(_COLLECTOR)
    _PKG_CACHE[pkg_path] = collected
    return list(collected)


def _drop_cached_submodules(pkg_path: str) -> None:
    """Remove ``pkg_path`` and all its descendants from ``sys.modules`` so
    a subsequent ``import_module`` re-executes ``case_*.py``.
    """
    prefix = pkg_path + "."
    stale = [m for m in sys.modules if m == pkg_path or m.startswith(prefix)]
    for m in stale:
        del sys.modules[m]


class _NonLiteral(ValueError):
    """Raised when an AST node is not a pure literal we can evaluate."""


def _ast_literal(node):
    """Evaluate a constant AST expression (numbers, strings, tuples, lists)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_literal(node.operand)
    if isinstance(node, ast.Tuple):
        return tuple(_ast_literal(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_ast_literal(elt) for elt in node.elts]
    raise _NonLiteral(f"unsupported AST node: {type(node).__name__}")


def _opshard_kwargs_from_call(call: ast.Call) -> Optional[dict]:
    """Extract planning kwargs from an ``OpShardCase(...)`` call node."""
    if not isinstance(call.func, ast.Name) or call.func.id != "OpShardCase":
        return None
    kwargs = {}
    for kw in call.keywords:
        if kw.arg in (
            "name", "tags", "mesh_shape", "mesh_dim_names",
            "num_proc", "needs_mesh", "solo_launcher",
        ):
            try:
                kwargs[kw.arg] = _ast_literal(kw.value)
            except _NonLiteral:
                return None
    if "name" not in kwargs or not isinstance(kwargs["name"], str):
        return None
    return kwargs


def _append_plan_stub(
        stubs: List[OpShardCase],
        seen: set,
        *,
        name: str,
        tags,
        mesh_shape,
        mesh_dim_names,
        num_proc=None,
        needs_mesh: bool = False,
        solo_launcher: bool = False,
        source_module: str = "",
) -> None:
    """Append one planning stub if ``name`` has not been seen yet."""
    if name in seen:
        return
    seen.add(name)

    def _stub_fn(*_a, **_k):
        raise RuntimeError(
            "planning stub OpShardCase.fn must not run in the parent process"
        )

    if isinstance(tags, list):
        tags = tuple(tags)
    if isinstance(mesh_shape, list):
        mesh_shape = tuple(mesh_shape)
    if isinstance(mesh_dim_names, list):
        mesh_dim_names = tuple(mesh_dim_names)
    stubs.append(OpShardCase(
        name=name,
        fn=_stub_fn,
        inputs=(),
        placements=(),
        tags=tuple(tags) if tags else (),
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        num_proc=num_proc,
        needs_mesh=needs_mesh,
        solo_launcher=solo_launcher,
        source_module=source_module,
    ))


def _register_opshard_call(node: ast.AST) -> Optional[ast.Call]:
    """Return the ``OpShardCase(...)`` call inside ``register(...)``, if any."""
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Name) and node.func.id == "register"):
        return None
    if not (node.args and isinstance(node.args[0], ast.Call)):
        return None
    call = node.args[0]
    if not (isinstance(call.func, ast.Name) and call.func.id == "OpShardCase"):
        return None
    return call


def _helper_spec_from_function(node: ast.FunctionDef) -> Optional[dict]:
    """If ``node`` is a thin ``register(OpShardCase(...))`` wrapper, describe it."""
    param_index = {arg.arg: i for i, arg in enumerate(node.args.args)}
    for sub in ast.walk(node):
        call = _register_opshard_call(sub)
        if call is None:
            continue
        mapping = {}
        tags_lit = ()
        for kw in call.keywords:
            if kw.arg == "tags":
                try:
                    tags_lit = _ast_literal(kw.value)
                    if isinstance(tags_lit, list):
                        tags_lit = tuple(tags_lit)
                except _NonLiteral:
                    tags_lit = ()
            elif (
                kw.arg in ("name", "mesh_shape", "mesh_dim_names")
                and isinstance(kw.value, ast.Name)
                and kw.value.id in param_index
            ):
                mapping[kw.arg] = param_index[kw.value.id]
        if "name" not in mapping:
            continue
        return {
            "name_i": mapping["name"],
            "mesh_i": mapping.get("mesh_shape"),
            "names_i": mapping.get("mesh_dim_names"),
            "tags": tags_lit if isinstance(tags_lit, tuple) else (),
        }
    return None


def _record_from_helper_call(node: ast.Call, meta: dict) -> Optional[dict]:
    """Extract one planning record from a helper call site."""
    try:
        name = _ast_literal(node.args[meta["name_i"]])
        mesh = (
            _ast_literal(node.args[meta["mesh_i"]])
            if meta["mesh_i"] is not None else None
        )
        names = (
            _ast_literal(node.args[meta["names_i"]])
            if meta["names_i"] is not None else None
        )
    except (IndexError, _NonLiteral):
        return None
    if not isinstance(name, str):
        return None
    return {
        "name": name,
        "tags": meta["tags"],
        "mesh_shape": mesh,
        "mesh_dim_names": names,
    }


def _helper_calls_from_tree(tree: ast.AST) -> List[dict]:
    """Collect planning metadata from thin register helpers like ``_bsh`` / ``_reg``.

    Pattern::

        def _bsh(name, fn, pl, mesh, names):
            register(OpShardCase(name=name, ..., mesh_shape=mesh,
                                 mesh_dim_names=names, tags=("npu_level1",)))

        _bsh("case_name", ..., (2,), ("dp",))
    """
    helpers = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.FunctionDef):
            continue
        spec = _helper_spec_from_function(node)
        if spec is not None:
            helpers[node.name] = spec

    records = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in helpers:
            continue
        rec = _record_from_helper_call(node, helpers[node.func.id])
        if rec is not None:
            records.append(rec)
    return records


def load_case_plan_from_package(pkg_path: str) -> List[OpShardCase]:
    """Build lightweight planning stubs by AST-parsing ``case_*.py``.

    Parent-side suite launchers only need name / tags / mesh metadata to
    bucket cases. Importing the real case modules would pull ``mindspore`` /
    ``torch`` into the pytest parent (and into forked ``msrun``/``torchrun``
    wrappers). Workers still call :func:`load_cases_from_package` and execute
    the real ``fn``.

    Resolves the package directory on disk without importing
    ``tests.mindspore`` / ``tests.torch`` (their ``__init__`` side effects are
    heavy).
    """
    from pathlib import Path  # pylint: disable=C0415

    parts = pkg_path.split(".")
    if parts[0] != "tests":
        raise ValueError(
            f"load_case_plan_from_package expects a tests.* package, got {pkg_path!r}"
        )
    # tests/shard_ops/framework/registry.py -> parents[2] == tests/
    tests_root = Path(__file__).resolve().parents[2]
    pkg_dir = tests_root.joinpath(*parts[1:])
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"case package directory not found: {pkg_dir}")

    stubs: List[OpShardCase] = []
    seen = set()

    for path in sorted(pkg_dir.glob("case_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        mod_name = f"{pkg_path}.{path.stem}"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            candidates = []
            if isinstance(node.func, ast.Name) and node.func.id == "register":
                if node.args and isinstance(node.args[0], ast.Call):
                    candidates.append(node.args[0])
            candidates.append(node)
            for call in candidates:
                kwargs = _opshard_kwargs_from_call(call)
                if kwargs is None:
                    continue
                _append_plan_stub(
                    stubs, seen,
                    name=kwargs["name"],
                    tags=kwargs.get("tags", ()),
                    mesh_shape=kwargs.get("mesh_shape"),
                    mesh_dim_names=kwargs.get("mesh_dim_names"),
                    num_proc=kwargs.get("num_proc"),
                    needs_mesh=bool(kwargs.get("needs_mesh", False)),
                    solo_launcher=bool(kwargs.get("solo_launcher", False)),
                    source_module=mod_name,
                )
        for rec in _helper_calls_from_tree(tree):
            _append_plan_stub(
                stubs, seen,
                name=rec["name"],
                tags=rec["tags"],
                mesh_shape=rec["mesh_shape"],
                mesh_dim_names=rec["mesh_dim_names"],
                source_module=mod_name,
            )
    return stubs

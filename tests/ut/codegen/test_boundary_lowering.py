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
"""Contract tests for deterministic ``lower_forward_boundaries`` rewrites.

The model file produced during generation once differed from the file
re-emitted from its own persisted
``meta.param_plan``, because ``meta.param_plan`` is written with
``json.dump(..., sort_keys=True)`` (which reorders the keys alphabetically) and
``lower_forward_boundaries`` iterated the boundary classes in ``param_plan``
insertion order.  Reloading the meta therefore reversed the class order, which
reordered the appended ``_HYPER_BOUNDARY_<CLASS>`` constants and changed the
emitted bytes, so offline drift validation failed consistently.

The lowerer now iterates boundary classes in ``sorted(groups)`` order, making
the byte stream a pure function of *which* classes are boundaries, independent
of how the ``param_plan`` reached us.  This test locks that contract: two plans
carrying the same entries but with opposite key insertion orders must lower to
the identical byte string.
"""

from __future__ import annotations

import ast
from functools import wraps
import importlib
from pathlib import Path
from importlib import util
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import hyper_parallel
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.components.modules import GatedDeltaNet, NativeRMSNormModule, RMSNormModule

from hyper_parallel.codegen.emit.modeling import _verify_recorded_patch_materialization
from hyper_parallel.codegen.emit.patches import emission_patches_from_meta
from hyper_parallel.codegen.emit.snapshot_patches import emission_patches_from_snapshot_payload
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
from hyper_parallel.codegen.inline.framework_spec import (
    _cp_strategy_spec,
    _ignored_constructor_parameters,
    observed_constructor_spec,
    strategy_spec_for,
)
from hyper_parallel.codegen.inline.ir import (
    ConstructorReplacePatch,
    ForwardExtractPatch,
    ImportPatch,
    InlinePatchSet,
    ModuleSnippetPatch,
    ReplacementSpec,
    SourceObservation,
)
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline import replacement_pass
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import (
    _add_recorded_forward_host_imports,
    _materialize_imported_forward_hosts,
    _observed_boundary_classes,
    render_inline_modeling_with_plan,
    render_inline_strategies_for_source_unit,
)
from hyper_parallel.distributed.context_parallel import wrappers as cp_wrappers
from hyper_parallel.codegen.inline.recognition import inner_wrapper_target
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.codegen.inline.framework_spec import cp_dispatch_shell_body
from hyper_parallel.codegen.inline.observations import collect_source_observations
from hyper_parallel.codegen.manager import ensure_codegen_artifact_from_snapshot
from hyper_parallel.codegen.session import CodegenBuildSession
from tests.common.mark_utils import arg_mark

SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward gets rewritten."""

    def forward(self, x):
        return x * 2


class Beta(nn.Module):
    """A second boundary class whose forward gets rewritten."""

    def forward(self, x):
        return x + 1
'''


def _load_closure_fixture(path: Path, source: str, namespace: dict) -> Any:
    """Load a real source unit with explicit CPU-only primitive fixtures."""
    path.write_text(source, encoding="utf-8")
    spec = util.spec_from_file_location(path.stem, path)
    module = util.module_from_spec(spec)
    module.__dict__.update(namespace)
    spec.loader.exec_module(module)
    return module


def test_native_source_snippets_reject_conflicting_class_definitions():
    """Do not hide different family implementations behind the same artifact name."""
    patches = InlinePatchSet(module_snippets=[ModuleSnippetPatch("class Alpha:\n    pass\n")])
    with pytest.raises(ValueError, match="conflicting source definition 'Alpha'"):
        apply_patch_set(SOURCE_TEXT, patches)


def test_native_capture_bindings_reject_conflicting_module_globals():
    """Two closures cannot silently share a name bound to different primitives."""
    patches = InlinePatchSet(module_snippets=[ModuleSnippetPatch("original_sdpa = F.scaled_dot_product_attention")])
    with pytest.raises(ValueError, match="conflicting source definition 'original_sdpa'"):
        apply_patch_set("original_sdpa = another_primitive\n", patches)


def test_identical_native_capture_bindings_are_emitted_once():
    """Repeated rules keep a single early-bound primitive alias."""
    binding = "original_sdpa = F.scaled_dot_product_attention"
    patches = InlinePatchSet(module_snippets=[ModuleSnippetPatch(binding), ModuleSnippetPatch(binding)])
    assert apply_patch_set(binding + "\n", patches).count(binding) == 1


@inner_wrapper
def _closure_cp_wrapper(target_module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                        ep_mesh: Any) -> _ForwardRewriteRequest:
    """Provide a model-independent closure fixture for source lowering."""
    del mesh, tp_mesh, ep_mesh
    context = {"enabled": cp_mesh is not None}
    original_forward = target_module.forward

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Forward the captured state through an explicit keyword."""
        call_kwargs = dict(kwargs)
        call_kwargs["cp_context"] = context
        return original_forward(*args, **call_kwargs)

    return _ForwardRewriteRequest(
        target_module, cp_forward, companion_attrs={"_test_cp_context": context},
    )


def test_inline_meta_normalization_does_not_mutate_native_observation():
    """Source normalization returns an emission copy and preserves native metadata."""
    meta = SimpleNamespace(
        param_plan={
            "model.block": {"is_boundary": True, "in_src": {"x": ["R"]}},
            "model.shared": {"is_boundary": False},
            "model.unchanged": {"is_boundary": True},
        },
        boundary_classes={"model.block": "Block", "model.shared": "Block"},
    )

    normalized = normalize_inline_meta(meta)

    assert normalized is not meta
    assert normalized.param_plan["model.block"]["is_boundary"] is True
    assert normalized.param_plan["model.block"]["skip_source_lowering"] is True
    assert normalized.boundary_classes == meta.boundary_classes
    assert normalized.boundary_classes is not meta.boundary_classes
    assert meta.param_plan["model.block"]["is_boundary"] is True
    assert normalized.param_plan["model.block"] is not meta.param_plan["model.block"]
    assert normalized.param_plan["model.block"]["in_src"] is meta.param_plan["model.block"]["in_src"]
    assert normalized.param_plan["model.unchanged"] is meta.param_plan["model.unchanged"]
    assert "skip_source_lowering" not in meta.param_plan["model.block"]
    assert meta.boundary_classes == {"model.block": "Block", "model.shared": "Block"}


def test_unbound_executor_closure_is_rejected_before_source_emission(monkeypatch):
    """An unpublished executor remains an error regardless of its model family."""
    source = '''
def wrapper(target_module, executor):
    original_forward = target_module.forward
    def cp_forward(*args, **kwargs):
        return executor(*args, **kwargs)
    return _ForwardRewriteRequest(target_module, cp_forward)
'''
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree", lambda _: ast.parse(source))
    with pytest.raises(RuntimeError, match="unresolved captured state.*executor"):
        inner_wrapper_target("demo.wrapper")


@pytest.mark.parametrize("wrapper_name", ["sdpa_qkv_hybrid_cp_wrapper", "flex_qkv_hybrid_cp_wrapper"])
def test_hybrid_wrapper_cannot_be_proven_by_an_unrelated_cp_toggle(wrapper_name):
    """An existing CP branch is not evidence that the selected Hybrid patch was emitted."""
    source = """class Attention:
    cp_enable = False
    def forward(self, q, k, v, **kwargs):
        if self.cp_enable:
            return q + k
        return q
"""
    record = SimpleNamespace(
        module_fqn="attn", target_fqn="self", target_type=type("Attention", (), {}),
        wrapper=getattr(cp_wrappers, wrapper_name), factory_kwargs={"ulysses_degree": 2},
    )
    patches = SimpleNamespace(local_computes=(), module_replacements=(), inner_wrappers=(record,))
    with pytest.raises(RuntimeError, match="patches were not materialized"):
        _verify_recorded_patch_materialization(source, SimpleNamespace(materialized_patches=[]), patch_plan=patches)


@pytest.mark.parametrize("receiver", ["self", "another_module"])
def test_selected_sdpa_wrapper_requires_its_native_helper_call(receiver):
    """The normal emitted SDPA dispatch remains a valid materialization proof."""
    record = SimpleNamespace(
        module_fqn="attn", target_fqn="self", target_type=type("Attention", (), {}),
        wrapper=cp_wrappers.sdpa_hf_cp_wrapper, factory_kwargs={},
    )
    patches = SimpleNamespace(local_computes=(), module_replacements=(), inner_wrappers=(record,))
    source = """class Attention:
    cp_enable = False
    def forward(self, x):
        if self.cp_enable:
            return sdpa_hf_cp_forward(self, x)
        return self._forward_impl(x)
"""
    if receiver != "self":
        source = source.replace("sdpa_hf_cp_forward(self, x)", "sdpa_hf_cp_forward(another_module, x)")
        with pytest.raises(RuntimeError, match="patches were not materialized"):
            _verify_recorded_patch_materialization(source, SimpleNamespace(materialized_patches=[]), patch_plan=patches)
    else:
        _verify_recorded_patch_materialization(source, SimpleNamespace(materialized_patches=[]), patch_plan=patches)


def test_hybrid_configuration_survives_snapshot_patch_projection():
    """Verification receives the same explicit wrapper arguments as source generation."""
    path = sys.modules[nn.Module.__module__].__file__
    payload = {
        "replacements": {"applications": []},
        "source": {"module_name": nn.Module.__module__, "file_path": path},
        "plan": {"boundary_classes": {"attn": "Module"}, "injections": [{
            "match": "attn", "inner_target": "self", "inner_wrapper": {
                "_target_": f"{cp_wrappers.__name__}.sdpa_qkv_hybrid_cp_wrapper", "ulysses_degree": 3,
            },
        }]},
    }
    serialized = emission_patches_from_snapshot_payload(payload)
    patches = emission_patches_from_meta(SimpleNamespace(emission_patches=serialized))
    assert patches.inner_wrappers[0].factory_kwargs == {"ulysses_degree": 3}


@pytest.mark.parametrize("mismatch", ["mask", "degree"])
def test_hybrid_verification_rejects_another_mode_with_the_same_helper(mismatch):
    """Helper names alone cannot prove the selected native branch and configuration."""
    wrapper = "flex_qkv_hybrid_cp_wrapper" if mismatch == "mask" else "sdpa_qkv_hybrid_cp_wrapper"
    degree = 3 if mismatch == "degree" else 2
    patches = build_strategy_patches(
        (SourceObservation(match=("attn",), inner_wrapper_target=f"{cp_wrappers.__name__}.{wrapper}",
                           construction_kwargs=(f"ulysses_degree={degree}",)),),
        boundary_classes={"attn": "Attention"},
    )
    source = "class Attention:\n    def forward(self, q, k, v, **kwargs):\n        return q\n"
    rendered = apply_patch_set(source, patches)
    record = SimpleNamespace(
        module_fqn="attn", target_fqn="self", target_type=type("Attention", (), {}),
        wrapper=cp_wrappers.sdpa_qkv_hybrid_cp_wrapper, factory_kwargs={"ulysses_degree": 2},
    )
    plan = SimpleNamespace(local_computes=(), module_replacements=(), inner_wrappers=(record,))
    with pytest.raises(RuntimeError, match="patches were not materialized"):
        _verify_recorded_patch_materialization(rendered, SimpleNamespace(materialized_patches=[]), patch_plan=plan)


@pytest.mark.parametrize("preexisting_cp_dispatch", [False, True])
@pytest.mark.parametrize("wrapper_name", [
    "flex_hf_cp_wrapper", "sdpa_hf_load_balance_cp_wrapper", "sdpa_hf_hybrid_cp_wrapper",
    "flex_hf_hybrid_cp_wrapper", "sdpa_qkv_hybrid_cp_wrapper", "flex_qkv_hybrid_cp_wrapper",
])
def test_native_wrapper_snapshot_publishes_complete_source(
    monkeypatch, tmp_path, wrapper_name, preexisting_cp_dispatch,
):
    """Exercise snapshot, static arguments, emission checks and atomic publication together."""
    path = tmp_path / "native_attention.py"
    primitive = "flex_ops.flex_attention" if wrapper_name.startswith("flex") else "F.scaled_dot_product_attention"
    source = ("from torch import nn\nfrom torch.nn import functional as F\n"
              "from torch.nn.attention import flex_attention as flex_ops\n\n"
              "class Attention(nn.Module):\n"
              "    cp_enable = False\n"
              "    def forward(self, q, k, v, **kwargs):\n"
              "        if self.cp_enable:\n            q = q + 1\n"
              f"        return {primitive}(q, k, v, **kwargs)\n")
    if not preexisting_cp_dispatch:
        source = source.replace("    cp_enable = False\n", "").replace(
            "        if self.cp_enable:\n            q = q + 1\n", "")
    path.write_text(source, encoding="utf-8")
    module_name = "native_codegen_wrapper_source"
    native_spec = util.spec_from_file_location(module_name, path)
    module = util.module_from_spec(native_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    native_spec.loader.exec_module(module)
    arguments = {"ulysses_degree": 2} if "hybrid" in wrapper_name else {}
    session = CodegenBuildSession()
    session.record_source({"file_path": str(path), "module_name": module_name, "architecture": "Attention"})
    session.record_replacements()
    session.record_plan({
        "param_plan": {}, "boundary_classes": {"attn": "Attention"}, "parallel_dims": {"cp_size": 6},
        "injections": [{"match": "attn", "inner_target": "self", "inner_wrapper": {
            "_target_": f"{cp_wrappers.__name__}.{wrapper_name}", **arguments,
        }}],
    })

    def _reject_runtime(*_args, **_kwargs):
        pytest.fail("source publication queried distributed runtime")

    for name in ("get_rank", "get_world_size", "new_group", "init_process_group"):
        monkeypatch.setattr(torch.distributed, name, _reject_runtime)
    layout = ensure_codegen_artifact_from_snapshot(session.finish(), str(tmp_path / "bundle"))
    files = sorted(item.name for item in Path(layout.artifact_dir).iterdir())
    assert len(files) == 3 and "__init__.py" in files
    rendered = Path(layout.modeling_path).read_text(encoding="utf-8")
    generated_spec = util.spec_from_file_location("generated_wrapper_publication", layout.modeling_path)
    generated = util.module_from_spec(generated_spec)
    generated_spec.loader.exec_module(generated)
    assert callable(generated.Attention.forward)
    assert "def _forward_impl" in rendered
    expected = "hybrid_cp_attention" if "hybrid" in wrapper_name else "original_"
    assert expected in rendered


@pytest.mark.parametrize("wrapper_name", [
    "flex_hf_cp_wrapper", "sdpa_hf_load_balance_cp_wrapper",
    "sdpa_hf_hybrid_cp_wrapper", "flex_hf_hybrid_cp_wrapper",
])
def test_native_hf_primitive_captures_have_source_bindings(wrapper_name):
    """Primitive aliases are captured once, with the native forward signature and finally block."""
    arguments = ("ulysses_degree=2",) if "hybrid" in wrapper_name else ()
    spec = _cp_strategy_spec(f"{cp_wrappers.__name__}.{wrapper_name}", construction_kwargs=arguments)
    assert spec.definition is not None
    definition = ast.parse(spec.definition).body[0]
    assert definition.args.args[0].arg == "self"
    assert definition.args.vararg.arg == "args"
    assert definition.args.kwarg.arg == "kwargs"
    assert any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(definition))
    bindings = "\n".join(item.text for item in spec.snippets)
    alias = "original_flex_attention" if wrapper_name.startswith("flex") else "original_sdpa"
    assert alias in bindings
    if "hybrid" in wrapper_name:
        calls = [node for node in ast.walk(definition) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "hybrid_cp_attention"]
        assert len(calls) == 1 and ast.literal_eval(calls[0].args[-1]) == 2


@pytest.mark.parametrize("wrapper_name,prepare_mask", [
    ("sdpa_qkv_hybrid_cp_wrapper", True), ("flex_qkv_hybrid_cp_wrapper", False),
])
@pytest.mark.parametrize("degree", [2, 3])
def test_native_qkv_hybrid_delegate_preserves_bound_configuration(wrapper_name, prepare_mask, degree):
    """The shared factory delegate supplies its actual mask mode and configured Ulysses degree."""
    spec = _cp_strategy_spec(f"{cp_wrappers.__name__}.{wrapper_name}",
                             construction_kwargs=(f"ulysses_degree={degree}",))
    assert spec.definition is not None
    definition = ast.parse(spec.definition).body[0]
    assert [arg.arg for arg in definition.args.args] == ["self", "q", "k", "v"]
    assert isinstance(definition.body[1], ast.If)
    assert ast.literal_eval(definition.body[1].test) is prepare_mask
    call = definition.body[-1].value
    assert call.func.id == "hybrid_cp_attention"
    assert ast.literal_eval(call.args[-1]) == degree


@pytest.mark.parametrize("wrapper_name,degree", [
    ("flex_hf_cp_wrapper", None), ("sdpa_hf_load_balance_cp_wrapper", None),
    ("sdpa_hf_hybrid_cp_wrapper", 2), ("sdpa_hf_hybrid_cp_wrapper", 3),
    ("flex_hf_hybrid_cp_wrapper", 2), ("flex_hf_hybrid_cp_wrapper", 3),
    ("sdpa_qkv_hybrid_cp_wrapper", 2), ("sdpa_qkv_hybrid_cp_wrapper", 3),
    ("flex_qkv_hybrid_cp_wrapper", 2), ("flex_qkv_hybrid_cp_wrapper", 3),
])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("source_unit", [False, True])
def test_native_and_generated_closures_preserve_calls_and_cleanup(
    monkeypatch, tmp_path, wrapper_name, degree, failure, source_unit,
):
    """Compare real native wrappers with emitted forwards, mocking only primitives and communication."""
    events = []

    def _decorate(original):
        @wraps(original)
        def _decorated(*args, **kwargs):
            events.append(("decorator", getattr(api, attribute) is _primitive))
            return original(*args, **kwargs)
        return _decorated

    def _primitive(q, k, v, **kwargs):
        events.append(("primitive", dict(kwargs)))
        return q + 2 * k + 3 * v

    def _prepare(kwargs, *_unused):
        events.append(("mask",))
        return kwargs

    def _hybrid(callback, q, k, v, kwargs, _mesh, ulysses_degree):
        events.append(("hybrid", ulysses_degree))
        return callback(q + ulysses_degree, k, v, kwargs)

    def _head_tail(callback, q, k, v, kwargs, _mesh, **_options):
        events.append(("head_tail",))
        return callback(q + 5, k, v, kwargs)

    def _gather(k, v, axis, _mesh):
        events.append(("gather", axis))
        return k + 4, v + 6

    flex = importlib.import_module("torch.nn.attention.flex_attention")
    use_flex = wrapper_name.startswith("flex")
    api, attribute = (flex, "flex_attention") if use_flex else (cp_wrappers.F, "scaled_dot_product_attention")
    monkeypatch.setattr(api, attribute, _primitive)
    monkeypatch.setattr(cp_wrappers, "_prepare_hybrid_sdpa_kwargs", _prepare)
    monkeypatch.setattr(cp_wrappers, "hybrid_cp_attention", _hybrid)
    monkeypatch.setattr(cp_wrappers, "head_tail_load_balance_attention", _head_tail)
    monkeypatch.setattr(cp_wrappers, "_prepare_head_tail_sdpa_kwargs", lambda kwargs, *unused: (kwargs, kwargs))
    monkeypatch.setattr(cp_wrappers, "flex_cp_allgather", _gather)
    qkv = "qkv" in wrapper_name
    parameters = "q, k, v" if qkv else "hidden_states"
    arguments = "q, k, v" if qkv else "hidden_states, hidden_states + 1, hidden_states + 2"
    source = ("class Attention:\n"
              "    @decorate\n"
              f"    def forward(self, {parameters}, *, extra=1, **kwargs):\n"
              "        if kwargs.pop('failure', False):\n"
              "            raise ValueError('native forward failure')\n"
              f"        return primitive_api.{attribute}({arguments}, **kwargs) + extra\n")
    native_module = _load_closure_fixture(tmp_path / "native_closure.py", source,
                                           {"primitive_api": api, "decorate": _decorate})
    native = native_module.Attention()
    cp_mesh = SimpleNamespace(size=lambda: 6)
    kwargs = {"ulysses_degree": degree} if degree is not None else {}
    native.forward = getattr(cp_wrappers, wrapper_name)(native, None, None, cp_mesh, None, **kwargs)
    target = f"{cp_wrappers.__name__}.{wrapper_name}"
    if source_unit:
        meta = SimpleNamespace(
            injections=[{"match": "attn", "inner_wrapper": {"_target_": target, **kwargs}}],
            boundary_classes={"attn": "Attention"}, module_overrides=[],
        )
        rendered = render_inline_strategies_for_source_unit(source, meta)
        unrelated = "class Unrelated:\n    pass\n"
        assert render_inline_strategies_for_source_unit(unrelated, meta) == unrelated
    else:
        patches = build_strategy_patches(
            (SourceObservation(match=("attn",), inner_wrapper_target=target,
                               construction_kwargs=tuple(f"{key}={value}" for key, value in kwargs.items())),),
            boundary_classes={"attn": "Attention"},
        )
        rendered = apply_patch_set(source, patches)
    generated_module = _load_closure_fixture(tmp_path / "generated_closure.py", rendered,
                                              {"primitive_api": api, "decorate": _decorate})
    generated = generated_module.Attention()
    generated._hp_boundary = SimpleNamespace(compiled=SimpleNamespace(mesh={"cp": cp_mesh}, mesh_dim_names=("cp",)))
    values = (torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)) if qkv else (torch.tensor(2.0),)
    results, observations = [], []
    for target in (native, generated):
        events.clear()
        if not qkv:
            monkeypatch.setattr(api, attribute, lambda *args, **kw: pytest.fail("lost the captured primitive"))
        if failure:
            with pytest.raises(ValueError, match="native forward failure"):
                target.forward(*values, extra=7, failure=True, marker="preserved")
        else:
            results.append(target.forward(*values, extra=7, marker="preserved"))
        assert getattr(api, attribute) is _primitive
        observations.append(list(events))
    assert observations[0] == observations[1]
    if not failure:
        torch.testing.assert_close(results[0], results[1])


@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("imported", [False, True])
@pytest.mark.parametrize("boundary_enabled", [False, True])
def test_helper_delegate_preserves_bound_keyword_and_original_fallback(
    monkeypatch, tmp_path, override, imported, boundary_enabled,
):
    """A single helper call retains the native target, override and user kwargs."""
    wrapper_source = '''
def kernel(module, value, *, enabled=None, **kwargs):
    return value

def wrapper(target_module, enabled=None):
    original_forward = target_module.forward
    def cp_forward(value, **kwargs):
        return kernel(target_module, value, enabled=enabled, **kwargs)
    return _ForwardRewriteRequest(target_module, cp_forward,
                                  companion_attrs={"parallel_enabled": enabled})
'''
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree",
                        lambda _: ast.parse(wrapper_source))
    spec = _cp_strategy_spec("demo.wrapper", construction_kwargs=(f"enabled={override!r}",))
    assert spec.imports[0] == ImportPatch("demo", ("kernel",))
    source = '''
class Attention:
    cp_enable = True
    def forward(self, value, **kwargs):
        return value + 1
'''
    patches = InlinePatchSet(forward_extracts=[
        ForwardExtractPatch("Attention", "forward", spec.subclass_shell),
    ])
    helper_source = '''
calls = []
def kernel(module, value, *, enabled=None, **kwargs):
    calls.append((module, enabled, kwargs))
    return value * 2 if enabled else value
'''
    def _load(name, text):
        path = tmp_path / f"{name}.py"
        path.write_text(text, encoding="utf-8")
        module_spec = util.spec_from_file_location(name, path)
        module = util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    calls = []
    if imported:
        base = _load("helper_base", helper_source + source)
        monkeypatch.setitem(sys.modules, "demo", base)
        patches.imports.append(ImportPatch("demo", ("Attention", "kernel")))
        _materialize_imported_forward_hosts("", patches)
        rendered = apply_patch_set("", patches)
        calls = base.calls
    else:
        rendered = helper_source + apply_patch_set(source, patches)
    if boundary_enabled:
        _, entries, classes = _boundary("attn", "Attention")
        rendered = lower_forward_boundaries(rendered, {
            "param_plan": entries, "injections": [], "mesh_dim_names": ("cp", "tp"),
        }, boundary_classes=classes)
    module = _load("helper_delegate", rendered)
    if not imported:
        calls = module.calls
    attention = module.Attention()
    attention._hp_boundary = SimpleNamespace(redistribute_inputs=lambda arguments: arguments,
                                            redistribute_outputs=lambda output: output)
    attention.cp_enable = True
    assert attention.forward(3, token=7) == (6 if override else 3)
    assert calls == [(attention, override, {"token": 7})]
    attention.cp_enable = False
    calls.clear()
    assert attention.forward(3, token=7) == 4
    assert not calls


@pytest.mark.parametrize("override", [None, False, True])
def test_dsa_constructor_keeps_dense_stage_override(override):
    """The current native family constructor retains an explicit stage override."""
    owner = "hyper_parallel.components.modules.attention.deepseek_sparse_attention"
    rule = SourceObservation(
        match=("model.attn",), module_type="demo.Attention",
        construction_kwargs=(f"dense_warmup={override!r}",),
        construction={"implementation": owner + ":CPDeepseekSparseAttentionModule"},
    )
    spec = observed_constructor_spec(rule)
    assert spec.new_ctor == "CPDeepseekSparseAttentionModule"
    expected = () if override is None else (f"dense_warmup={override!r}",)
    assert spec.keyword_args == expected


def test_wrapper_override_survives_observation_and_strategy_cache(monkeypatch):
    """Different instances must not reuse another wrapper configuration's source."""
    target = "hyper_parallel.models.demo.adapter.distributed.context_parallel.stage_wrapper"
    source = """
def stage_kernel(module, value, *, use_dense=None, **kwargs):
    return value

def stage_wrapper(target_module, dense_warmup=None):
    original_forward = target_module.forward
    def cp_forward(value, **kwargs):
        return stage_kernel(target_module, value, use_dense=dense_warmup, **kwargs)
    return _ForwardRewriteRequest(target_module, cp_forward)
"""
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree",
                        lambda _: ast.parse(source))
    bodies = []
    rules = []
    for override in (True, False, None):
        meta = SimpleNamespace(injections=[{
            "match": "model.attn", "inner_wrapper": {"_target_": target, "dense_warmup": override},
        }])
        observations = collect_source_observations(meta)
        assert len(observations) == 1
        rule = observations[0]
        assert rule.construction_kwargs == (f"dense_warmup={override!r}",)
        patches = build_strategy_patches((rule,), boundary_classes={"model.attn": "Attention"})
        body = patches.forward_extracts[0].body
        assert f"use_dense={override!r}" in body
        bodies.append(body)
        rules.append(rule)
    assert len(set(bodies)) == 3
    with pytest.raises(RuntimeError, match="conflicting CP bindings"):
        build_strategy_patches(tuple(rules), boundary_classes={"model.attn": "Attention"})


@pytest.mark.parametrize(("helper_args", "forward_args", "call_args"), [
    ("module, value=1", "value=2", "target_module, value"),
    ("module, first, second", "first, second", "target_module, second, first"),
    ("module, value", "value", "target_module, value + 1"),
])
def test_helper_token_rejects_argument_adaptation(monkeypatch, helper_args, forward_args, call_args):
    """Default changes, reordering and transformed values cannot use passthrough lowering."""
    source = f'''
def kernel({helper_args}, *, enabled=None, **kwargs):
    return None

def wrapper(target_module, enabled=None):
    original_forward = target_module.forward
    def cp_forward({forward_args}, **kwargs):
        return kernel({call_args}, enabled=enabled, **kwargs)
    return _ForwardRewriteRequest(target_module, cp_forward, companion_attrs={{"enabled": enabled}})
'''
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree", lambda _: ast.parse(source))
    with pytest.raises(RuntimeError, match="not an argument passthrough"):
        _cp_strategy_spec("demo.wrapper")


@pytest.mark.parametrize("parameters", ["target_module", "target_module, cp_mesh"])
def test_local_or_reassigned_mesh_cannot_borrow_boundary_mesh(monkeypatch, parameters):
    """Only an unchanged native argument may use the compiled CP submesh."""
    source = f"""
def wrapper({parameters}):
    cp_mesh = object()
    original_forward = target_module.forward
    def patched(value):
        return original_forward(value, mesh=cp_mesh)
    return _ForwardRewriteRequest(target_module, patched)
"""
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree",
                        lambda _: ast.parse(source))
    with pytest.raises(RuntimeError, match="unresolved captured state.*cp_mesh"):
        inner_wrapper_target("demo.wrapper")


def test_helper_mesh_keyword_uses_live_native_boundary(monkeypatch, tmp_path):
    """The actual async wrapper forwards the current mesh, not generation rank zero."""
    owner = "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel"
    spec = _cp_strategy_spec(owner + ".qwen3_vl_moe_async_ulysses_cp_wrapper")
    calls = []

    def kernel(module, *args, cp_mesh, **kwargs):
        calls.append((module, args, cp_mesh, kwargs))
        return cp_mesh.get_local_rank()

    monkeypatch.setitem(sys.modules, owner, SimpleNamespace(_async_ulysses_forward=kernel))
    source = """
class Host:
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        return -1
"""
    patches = InlinePatchSet(imports=list(spec.imports), forward_extracts=[
        ForwardExtractPatch("Host", "forward", spec.subclass_shell),
    ])
    path = tmp_path / "mesh_helper.py"
    path.write_text(apply_patch_set(source, patches), encoding="utf-8")
    module_spec = util.spec_from_file_location("mesh_helper", path)
    module = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    host = module.Host()
    host.cp_enable = True
    compiled = SimpleNamespace(mesh={}, mesh_dim_names=("cp",))
    host._hp_boundary = SimpleNamespace(compiled=compiled)
    for rank in (0, 1):
        mesh = SimpleNamespace(get_local_rank=lambda value=rank: value)
        compiled.mesh["cp"] = mesh
        assert host.forward(3, (4, 5), None, token=7) == rank
        assert calls[-1] == (host, (3, (4, 5), None, None), mesh, {"token": 7})


def test_native_text_input_closure_shards_each_rank(monkeypatch, tmp_path):
    """The real VL closure preserves slicing and empty input independently of the working directory."""
    monkeypatch.chdir(tmp_path)
    owner = "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel"
    package_root = Path(hyper_parallel.__file__).resolve().parent
    wrapper_file = package_root / "models/qwen3_vl_moe/adapter/distributed/context_parallel.py"
    helper = next(node for node in ast.parse(wrapper_file.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.FunctionDef) and node.name == "_slice_dense_visual_features")
    helper_path = tmp_path / "visual_helper.py"
    helper_path.write_text("import torch\n" + ast.unparse(helper), encoding="utf-8")
    helper_spec = util.spec_from_file_location("visual_helper", helper_path)
    helper_module = util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    monkeypatch.setitem(sys.modules, owner, helper_module)
    spec = _cp_strategy_spec(owner + ".qwen3_vl_moe_text_input_cp_wrapper")
    source = """
import torch
class Host:
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, visual_pos_masks=None, deepstack_visual_embeds=None, **kwargs):
        return inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds, kwargs
"""
    patches = InlinePatchSet(imports=list(spec.imports), forward_extracts=[
        ForwardExtractPatch("Host", "forward", spec.subclass_shell),
    ])
    path = tmp_path / "text_input.py"
    path.write_text(apply_patch_set(source, patches), encoding="utf-8")
    module_spec = util.spec_from_file_location("text_input", path)
    module = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    host = module.Host()
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 0)
    host._hp_boundary = SimpleNamespace(compiled=SimpleNamespace(mesh={"cp": mesh}, mesh_dim_names=("cp",)))
    values = torch.arange(8).reshape(1, 4, 2)
    positions = torch.arange(4).reshape(1, 4)
    mask = torch.tensor([[True, False, True, False]])
    features = torch.tensor([[10], [20]])
    for rank in (0, 1):
        mesh.get_local_rank = lambda value=rank: value
        output = host.forward(inputs_embeds=values, position_ids=positions, visual_pos_masks=mask,
                              deepstack_visual_embeds=[features], token=7)
        assert torch.equal(output[0], values[:, rank * 2:(rank + 1) * 2])
        assert torch.equal(output[1], positions[:, rank * 2:(rank + 1) * 2])
        assert torch.equal(output[2], mask[:, rank * 2:(rank + 1) * 2])
        assert torch.equal(output[3][0], features[rank:rank + 1])
        assert output[4] == {"token": 7}
    assert host.forward()[0] is None
    with pytest.raises(ValueError, match="divisible"):
        host.forward(inputs_embeds=values[:, :3])



def test_helper_keyword_does_not_leak_factory_module_globals(monkeypatch):
    """An imported helper does not import the globals used to build its arguments."""
    source = '''
ENABLED = True
def kernel(module, value, *, enabled=None, **kwargs):
    return value
def wrapper(target_module):
    original_forward = target_module.forward
    def cp_forward(value, **kwargs):
        return kernel(target_module, value, enabled=ENABLED, **kwargs)
    return _ForwardRewriteRequest(target_module, cp_forward)
'''
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree", lambda _: ast.parse(source))
    with pytest.raises(RuntimeError, match="unresolved keyword names.*ENABLED"):
        _cp_strategy_spec("demo.wrapper")


def test_framework_inner_wrapper_resolves_without_model_adapter_path():
    """A declared framework wrapper participates in generic CP lowering."""
    target = (
        "hyper_parallel.distributed.context_parallel.wrappers."
        "sdpa_hf_cp_wrapper"
    )

    spec = strategy_spec_for(target, "Qwen3_5ForCausalLM", inner_wrapper=True)

    assert spec is not None
    assert spec.subclass_shell is not None
    assert spec.imports[0].names == ("sdpa_hf_cp_forward",)



def test_identity_wrapper_keeps_existing_no_rewrite_contract(monkeypatch):
    """A wrapper with no replacement closure does not acquire a new failure."""
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree",
                        lambda _: ast.parse("def identity(target_module):\n    return target_module.forward\n"))
    assert inner_wrapper_target("demo.identity") is None


@pytest.mark.parametrize("imported", [False, True])
def test_closure_preserves_original_override_and_finally(monkeypatch, tmp_path, imported):
    """A saved forward refers to this class's override and keeps cleanup on failure."""
    wrapper_source = """
def wrapper(target_module):
    original_forward = target_module.forward
    def patched(*args, **kwargs):
        saved = target_module.marker
        target_module.marker = 9
        kwargs["delta"] = kwargs.get("delta", 0) + 1
        def invoke():
            return original_forward(*args, **kwargs)
        try:
            return invoke()
        finally:
            target_module.marker = saved
    return _ForwardRewriteRequest(target_module, patched)
"""
    monkeypatch.setattr("hyper_parallel.codegen.inline.recognition._module_source_tree",
                        lambda _: ast.parse(wrapper_source))
    spec = _cp_strategy_spec("demo.wrapper")
    assert "cp_enable" not in spec.subclass_shell
    assert "target_module" not in spec.subclass_shell
    assert "self._forward_impl(*__hp_forward_positional__, **__hp_forward_keywords__)" in spec.subclass_shell
    source = """
class Base:
    def forward(self, value, **kwargs):
        raise AssertionError("the saved method is not this inherited forward")

class Host(Base):
    marker = 2
    cp_enable = False
    def forward(self, value, *, extra=4, **kwargs):
        if kwargs.get("fail"):
            raise ValueError("original failure")
        return value + self.marker + extra + kwargs["delta"]
"""
    def load(name, text):
        path = tmp_path / f"{name}.py"
        path.write_text(text, encoding="utf-8")
        module_spec = util.spec_from_file_location(name, path)
        module = util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    patches = build_strategy_patches(
        (SourceObservation(match=("host",), inner_wrapper_target="demo.wrapper"),),
        boundary_classes={"host": "Host"},
    )
    if imported:
        base = load("native_host", source)
        monkeypatch.setitem(sys.modules, "demo", base)
        patches.imports.append(ImportPatch("demo", ("Host",)))
        _materialize_imported_forward_hosts("", patches)
        source = ""
    module = load("generated_host", apply_patch_set(source, patches))
    host = module.Host()
    assert host.forward(3) == 17
    assert host.forward(value=3, extra=8) == 21
    assert host.marker == 2
    with pytest.raises(ValueError, match="original failure"):
        host.forward(3, fail=True)
    assert host.marker == 2


def test_native_packed_decoder_lowering_runs_at_cp1(tmp_path):
    """The actual Qwen packed branch must work without CP or an inherited forward."""
    target = "hyper_parallel.models.qwen3_5.adapter.linear_attention_decoder.qwen3_5_linear_attention_decoder_wrapper"
    patches = build_strategy_patches(
        (SourceObservation(match=("layer",), inner_wrapper_target=target),),
        boundary_classes={"layer": "Layer"},
    )
    source = """
class Layer:
    cp_enable = False
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        return hidden_states + 100
"""
    rendered = apply_patch_set(source, patches)
    assert "target_module" not in rendered
    assert "if self.cp_enable" not in rendered
    path = tmp_path / "packed_decoder.py"
    path.write_text(rendered, encoding="utf-8")
    module_spec = util.spec_from_file_location("packed_decoder", path)
    module = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    layer = module.Layer()
    calls = []

    def linear_attn(hidden_states, **kwargs):
        calls.append(kwargs)
        return hidden_states * 3

    layer.linear_attn = GatedDeltaNet(config=SimpleNamespace(
        hidden_size=8, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4), chunk_size=4)
    layer.linear_attn.forward = linear_attn
    layer.input_layernorm = lambda value: value + 1
    layer.post_attention_layernorm = lambda value: value + 2
    layer.mlp = lambda value: value * 2
    assert layer.forward(3) == 103
    assert not calls
    packed = object()
    assert layer.forward(3, packed_sequence_info=packed) == 49
    assert calls[0]["packed_sequence_info"] is packed
    with pytest.raises(NotImplementedError, match="recurrent cache"):
        layer.forward(3, packed_sequence_info=packed, use_cache=True)


def test_closure_cp_wrapper_exposes_source_visible_forward():
    """Closure state published by a rewrite request lowers into its host."""
    factory = f"{__name__}._closure_cp_wrapper"

    target = inner_wrapper_target(factory)

    assert target is not None
    assert target.helper_name is None
    assert target.inline_body is not None
    assert "self.cp_enable" not in target.inline_body
    assert "self._test_cp_context" in target.inline_body
    assert "self._forward_impl(*__hp_forward_positional__, **call_kwargs)" in target.inline_body


def test_closure_cp_wrapper_materializes_imported_replacement_host(monkeypatch):
    """A closure wrapper and an imported replacement compose end to end."""
    wrapper = f"{__name__}._closure_cp_wrapper"
    monkeypatch.setattr(
        "hyper_parallel.codegen.inline.strategy_pass.strategy_spec",
        lambda *_args, **_kwargs: _cp_strategy_spec(wrapper),
    )
    patches = build_strategy_patches(
        (SourceObservation(match=("model.layers.0.self_attn",), inner_wrapper_target=wrapper),),
        boundary_classes={"model.layers.0.self_attn": "DemoAttention"},
    )
    patches.imports.append(
        ImportPatch("demo.components", ("DemoAttention",))
    )

    _materialize_imported_forward_hosts(SOURCE_TEXT, patches)
    rendered = apply_patch_set(SOURCE_TEXT, patches)

    ast.parse(rendered)
    assert "class DemoAttention(demo.components.DemoAttention):" in rendered
    assert "if self.cp_enable:" not in rendered
    assert "self._test_cp_context" in rendered
    assert "return original_forward(*args, **call_kwargs)" in rendered
    assert "original_forward = super().forward" in rendered

VARIADIC_SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward takes variadic arguments."""

    def forward(self, x, *args, **kwargs):
        return x * 2
'''

KWARGS_TUPLE_SOURCE_TEXT = '''\
import torch
from torch import nn


class Alpha(nn.Module):
    """A boundary class whose forward keeps HF-style **kwargs and tuple output."""

    def forward(self, x, y=None, **kwargs):
        hidden = x if y is None else x + y
        return hidden, kwargs.get("aux")
'''


def _boundary(fqn: str, class_name: str) -> tuple[str, dict]:
    """A minimal but real-shaped frozen boundary entry for one class.

    The input side carries a cp transition (``S(1) -> R``) that is not
    TP-lowerable, so with the plan's active axes (``("cp", "tp")``) the
    entry classifies as the generic redistribute form.
    """
    entry = {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)", "ep": "R", "cp": "S(1)"}},
        "in_dst": {"x": {"tp": "R", "ep": "R", "cp": "R"}},
        "out_src": {"output": {"tp": "S(-1)", "ep": "R", "cp": "S(1)"}},
        "out_dst": {"output": {"tp": "R", "ep": "R", "cp": "S(1)"}},
        "params": {"weight": {"tp": "S(0)", "ep": "R", "cp": "R"}},
    }
    return [fqn], {fqn: entry}, {fqn: class_name}


def _recorded_alpha_forward(self, x):
    del self
    return x * 2


def _plan(*, alpha_first: bool) -> dict:
    """A frozen-plan dict whose two boundary classes appear in one of two orders."""
    _, alpha_entry, alpha_map = _boundary("blocks.alpha", "Alpha")
    _, beta_entry, beta_map = _boundary("blocks.beta", "Beta")
    if alpha_first:
        plan = {**alpha_entry, **beta_entry}
        boundary_classes = {**alpha_map, **beta_map}
    else:
        plan = {**beta_entry, **alpha_entry}
        boundary_classes = {**beta_map, **alpha_map}
    return (
        {"param_plan": plan, "injections": [], "mesh_dim_names": ("cp", "tp")},
        boundary_classes,
    )


def test_lowering_bytes_are_order_independent():
    """Reversing the ``param_plan`` key order must not change the emitted bytes.

    This is the property the sort fix restores: the two plans carry identical
    entry content, so they must lower to the identical byte string.  Before the
    fix the ``_HYPER_BOUNDARY_Alpha``/``_HYPER_BOUNDARY_Beta`` constants were
    appended in insertion order, so ``alpha_first`` and the reversed plan
    produced different output bytes.

    Feature: codegen-lowering
    Description: Two plans with identical entries but reversed key insertion
        order are lowered through ``lower_forward_boundaries``.
    Expectation: The emitted byte strings are identical (order-independent).
    """
    plan_a, classes_a = _plan(alpha_first=True)
    plan_b, classes_b = _plan(alpha_first=False)

    text_a = lower_forward_boundaries(SOURCE_TEXT, plan_a, boundary_classes=classes_a)
    text_b = lower_forward_boundaries(SOURCE_TEXT, plan_b, boundary_classes=classes_b)

    assert text_a == text_b


def test_rewritten_forwards_bind_instance_boundary_plans():
    """No per-boundary global constants are emitted; forwards use the bound plan.

    Explicitly checks that a real rewrite happened, so the byte-identity test
    above cannot pass vacuously (e.g. on a plan that never rewrote a forward).

    Feature: codegen-lowering
    Description: A plan with two boundary classes is lowered and the emitted
        text is inspected for the absence of per-boundary global constants
        and forward-time globals, and the presence of instance-bound calls.
    Expectation: No ``_HYPER_BOUNDARY_`` constants, no ``mesh_context``;
        ``_forward_impl`` and ``self._hp_boundary.redistribute_*`` present.
    """
    plan, classes = _plan(alpha_first=True)
    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    # The rewritten forwards reference the instance-bound compiled plan that
    # runtime binding supplies — no per-boundary global constants
    # and no forward-time globals.  The artifact carries no plan literals at
    # all: runtime state is bound later from the live Trainer plan.
    assert "_HYPER_BOUNDARY_" not in text
    assert "mesh_context" not in text
    assert "_forward_impl" in text
    assert text.count("self._hp_boundary.redistribute_inputs") >= 2
    assert text.count("self._hp_boundary.redistribute_outputs") >= 2
    assert "rewrap_outputs" not in text
    assert "hp_to_local_if_dtensor" not in text
    assert "generated_support" not in text


def _tp_collective_entry() -> dict:
    """A TP-lowerable boundary: in S(1)->R, out P(sum)->S(1) on the tp axis."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "S(1)"}},
        "in_dst": {"x": {"tp": "R"}},
        "out_src": {"output": {"tp": "P(sum)"}},
        "out_dst": {"output": {"tp": "S(1)"}},
    }


def _identity_entry() -> dict:
    """An all-identity boundary: no transition on any axis, nothing to lower."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "R"}},
        "in_dst": {"x": {"tp": "R"}},
        "out_src": {"output": {"tp": "R"}},
        "out_dst": {"output": {"tp": "R"}},
    }


def _single_plan(entry: dict) -> tuple[dict, dict]:
    """A frozen plan whose only boundary is ``blocks.alpha`` (class Alpha)."""
    plan = {"param_plan": {"blocks.alpha": entry}, "injections": []}
    return plan, {"blocks.alpha": "Alpha"}


def test_local_region_lowering_requires_builder_recording():
    """A local-compute boundary cannot fall back to an equivalent source rewrite.

    Feature: codegen-lowering
    Description: A boundary whose frozen injection declares ``local_compute_fn``
        reaches ``lower_forward_boundaries`` without the recorder's
        ``skip_source_lowering`` marker.
    Expectation: Generation fails instead of cloning native Phase C and changing
        the numerical execution order.
    """
    plan, classes = _single_plan(_identity_entry())
    plan["injections"] = [
        {"match": "blocks.alpha", "local_compute_fn": {"_target_": "some.factory"}},
    ]

    with pytest.raises(RuntimeError, match="builder recording"):
        lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)


def test_runtime_only_local_compute_does_not_poison_shared_source_class():
    """A skipped local-compute injection must not affect another class instance."""
    entry = _identity_entry()
    skipped = {**_identity_entry(), "skip_source_lowering": True}
    plan = {
        "param_plan": {"blocks.alpha": entry, "blocks.runtime": skipped},
        "injections": [{
            "match": "blocks.runtime",
            "local_compute_fn": {"_target_": "some.factory"},
        }],
        "mesh_dim_names": ("tp",),
    }
    classes = {"blocks.alpha": "Alpha", "blocks.runtime": "Alpha"}

    assert lower_forward_boundaries(
        SOURCE_TEXT, plan, boundary_classes=classes
    ) == SOURCE_TEXT


def test_recorded_local_compute_rewrites_same_class_with_ep_toggle():
    """A recorded EP patch keeps the original class and forward implementation."""
    alpha_type = type("Alpha", (), {"forward": _recorded_alpha_forward})
    record = SimpleNamespace(
        module_fqn="blocks.alpha",
        module_type=alpha_type,
        original_forward=alpha_type.forward,
        factory="demo.ep.factory",
    )
    meta = SimpleNamespace(
        param_plan={"blocks.alpha": _identity_entry()},
        boundary_classes={"blocks.alpha": "Alpha"},
        mesh_dim_names=("ep",),
        injections=[{
            "match": "blocks.alpha",
            "local_compute_fn": {"_target_": "demo.ep.compute"},
        }],
    )
    patches = SimpleNamespace(local_computes=(record,))

    rendered = materialize_recorded_forwards(
        SOURCE_TEXT, meta, patch_plan=patches
    )

    expected_tokens = (
        "class Alpha(nn.Module):",
        "ep_enable = False",
        "def _forward_impl(self, x):",
        "if self.ep_enable:",
    )
    for token in expected_tokens:
        assert token in rendered, (
            f"Expected generated EP source to contain {token!r}, "
            f"got source={rendered!r}"
        )
    for token in ("HPCodegen", "hp_recorded_local_compute_patches"):
        assert token not in rendered, (
            f"Expected generated EP source to omit {token!r}, "
            f"got source={rendered!r}"
        )


def test_recorded_ep_toggle_prevents_second_same_class_boundary_lowering():
    """One class-level EP toggle excludes all its FQNs from later lowering."""
    alpha_type = type("Alpha", (), {"forward": _recorded_alpha_forward})
    record = SimpleNamespace(
        module_fqn="blocks.alpha",
        module_type=alpha_type,
        original_forward=alpha_type.forward,
        factory="demo.ep.factory",
    )
    meta = SimpleNamespace(
        param_plan={
            "blocks.alpha": _identity_entry(),
            "blocks.other_alpha": _boundary("unused", "Alpha")[1]["unused"],
        },
        boundary_classes={
            "blocks.alpha": "Alpha",
            "blocks.other_alpha": "Alpha",
        },
        mesh_dim_names=("cp", "ep"),
        injections=[{
            "match": "blocks.alpha",
            "local_compute_fn": {"_target_": "demo.ep.compute"},
        }],
    )

    rendered = materialize_recorded_forwards(
        SOURCE_TEXT,
        meta,
        patch_plan=SimpleNamespace(local_computes=(record,)),
    )
    lowered = lower_forward_boundaries(
        rendered, meta, boundary_classes=meta.boundary_classes
    )
    tree = ast.parse(lowered)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Alpha"
    )
    methods = [
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert all(
        entry["skip_source_lowering"]
        for entry in meta.param_plan.values()
    )
    assert methods.count("_forward_impl") == 1
    assert methods.count("forward") == 1
    assert lowered == rendered


def test_frozen_local_compute_rewrites_class_without_live_record():
    """Explicit frozen injections remain visible in offline-only snapshots."""
    meta = SimpleNamespace(
        param_plan={"blocks.alpha": _identity_entry()},
        boundary_classes={"blocks.alpha": "Alpha"},
        mesh_dim_names=("ep",),
        materialized_patches=[],
        injections=[{
            "match": "blocks.alpha",
            "local_compute_fn": {"_target_": "demo.ep.compute"},
        }],
    )

    rendered = materialize_recorded_forwards(
        SOURCE_TEXT,
        meta,
        patch_plan=SimpleNamespace(local_computes=()),
    )

    assert "ep_enable = False" in rendered
    assert "def _forward_impl(self, x):" in rendered
    assert "if self.ep_enable:" in rendered
    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha",
        "host_class": "Alpha",
        "kind": "local_compute",
        "source_form": "same_class_toggle_forward",
    }]


def test_cp_dispatch_rewrites_same_class_without_constructor_swap():
    """CP dispatch is an in-place forward rewrite with the original signature."""
    patches = InlinePatchSet(
        forward_extracts=[
            ForwardExtractPatch(
                class_name="Alpha",
                method_name="forward",
                body=cp_dispatch_shell_body("kimi_cp_forward"),
            )
        ]
    )

    rendered = apply_patch_set(SOURCE_TEXT, patches)

    expected_tokens = (
        "class Alpha(nn.Module):",
        "cp_enable = False",
        "def _forward_impl(self, x):",
        "return self._forward_impl(x)",
        "return kimi_cp_forward(self, x)",
    )
    for token in expected_tokens:
        assert token in rendered, (
            f"Expected generated CP source to contain {token!r}, "
            f"got source={rendered!r}"
        )
    for token in ("class AlphaCP", "HPCodegen"):
        assert token not in rendered, (
            f"Expected generated CP source to omit {token!r}, "
            f"got source={rendered!r}"
        )


def test_same_class_cp_and_boundary_keep_pure_forward_impl():
    """A source-owned CP host keeps one method after boundary lowering."""
    patches = InlinePatchSet(
        forward_extracts=[
            ForwardExtractPatch(
                class_name="Alpha",
                method_name="forward",
                body=cp_dispatch_shell_body("kimi_cp_forward"),
            )
        ]
    )
    rendered = apply_patch_set(SOURCE_TEXT, patches)
    _, entries, classes = _boundary("blocks.alpha", "Alpha")
    plan = {
        "param_plan": entries,
        "injections": [],
        "mesh_dim_names": ("cp", "tp"),
    }

    rendered = lower_forward_boundaries(
        rendered, plan, boundary_classes=classes
    )
    tree = ast.parse(rendered)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Alpha"
    )
    methods = [
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert methods.count("_forward_impl") == 1
    assert methods.count("forward") == 1
    local = next(node for node in host.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_impl")
    assert ast.unparse(local.body[0]) == "return x * 2"
    assert "outputs = kimi_cp_forward(self, *args, **kwargs)" in rendered
    assert "return self._forward_impl(x)" not in rendered
    assert "return x * 2" in rendered


@pytest.mark.parametrize("cp_enabled", [False, True])
@pytest.mark.parametrize("boundary_enabled", [False, True])
@pytest.mark.parametrize("imported", [False, True])
def test_sdpa_helper_preserves_local_forward_and_boundary_order(
    monkeypatch, tmp_path, cp_enabled, boundary_enabled, imported,
):
    """A real callback helper reaches local attention once, using redistributed inputs."""
    source = '''from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.events = []

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, *extra, required=0, **kwargs):
        self.events.append(("local", position_embeddings, attention_mask, extra, required, kwargs))
        if kwargs.get("fail"):
            raise ValueError("local failure")
        return F.scaled_dot_product_attention(hidden_states, hidden_states, hidden_states)
'''

    def load(name, text):
        path = tmp_path / f"{name}.py"
        path.write_text(text, encoding="utf-8")
        spec = util.spec_from_file_location(name, path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    wrapper = "hyper_parallel.distributed.context_parallel.wrappers.sdpa_hf_cp_wrapper"
    meta = SimpleNamespace(injections=[{"match": "attn", "inner_wrapper": wrapper, "inner_target": "self"}],
                           boundary_classes={"attn": "Attention"}, module_overrides=[])
    if imported:
        base = load("sdpa_base", source)
        monkeypatch.setitem(sys.modules, "sdpa_base", base)
        patches = build_strategy_patches(collect_source_observations(meta), boundary_classes=meta.boundary_classes)
        patches.imports.append(ImportPatch("sdpa_base", ("Attention",)))
        _materialize_imported_forward_hosts("", patches)
        rendered = apply_patch_set("", patches)
    else:
        rendered, _ = render_inline_modeling_with_plan(source, meta)
    if boundary_enabled:
        _, entries, classes = _boundary("attn", "Attention")
        entries["attn"]["in_src"] = {"hidden_states": {"cp": "S(1)"}}
        entries["attn"]["in_dst"] = {"hidden_states": {"cp": "R"}}
        rendered = lower_forward_boundaries(rendered, {"param_plan": entries, "injections": meta.injections,
            "mesh_dim_names": ("cp",)}, boundary_classes=classes)
    model = load("sdpa_generated", rendered).Attention()
    model.cp_enable = cp_enabled
    model._hp_sdpa_hf_cp_mesh = object()  # Resource binding belongs to the test, not generation.
    native_sdpa = cp_wrappers.F.scaled_dot_product_attention

    def communicate(original, mesh, query, key, value, kwargs):
        assert mesh is model._hp_sdpa_hf_cp_mesh
        model.events.append("cp")
        return original(query, key, value, **kwargs)

    def enter(arguments):
        model.events.append("in")
        args, kwargs = arguments
        return (args[0] * 2, *args[1:]), {**kwargs, "required": kwargs["required"] + 1}

    def leave(output):
        model.events.append("out")
        return output + 3

    model._hp_boundary = SimpleNamespace(redistribute_inputs=enter, redistribute_outputs=leave)
    monkeypatch.setattr(cp_wrappers, "_cp_sdpa_call", communicate)
    inputs = torch.randn(1, 1, 2, 4)
    output = model(inputs, "positions", None, "extra", required=7, token=9)
    local_inputs = inputs * 2 if boundary_enabled else inputs
    expected = native_sdpa(local_inputs, local_inputs, local_inputs) + (3 if boundary_enabled else 0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    local_event = ("local", "positions", None, ("extra",), 8 if boundary_enabled else 7, {"token": 9})
    assert model.events == (["in"] if boundary_enabled else []) + [local_event] + (
        ["cp"] if cp_enabled else []) + (["out"] if boundary_enabled else [])
    assert cp_wrappers.F.scaled_dot_product_attention is native_sdpa
    assert "cp_enable" not in model._forward_impl.__code__.co_names
    model.events.clear()
    with pytest.raises(ValueError, match="local failure"):
        model(inputs, required=7, fail=True)
    assert cp_wrappers.F.scaled_dot_product_attention is native_sdpa
    assert "out" not in model.events


def test_same_class_cp_toggle_satisfies_inner_wrapper_verification():
    """The verifier accepts the same-class CP form used by generated models."""
    alpha_type = type("Alpha", (), {})
    wrapper = SimpleNamespace(
        module_fqn="blocks.alpha",
        target_fqn="self",
        target_type=alpha_type,
        request=SimpleNamespace(target=object()),
    )
    patches = SimpleNamespace(
        local_computes=(),
        module_replacements=(),
        inner_wrappers=(wrapper,),
    )
    meta = SimpleNamespace(materialized_patches=[])
    rendered = apply_patch_set(
        SOURCE_TEXT,
        InlinePatchSet(
            forward_extracts=[
                ForwardExtractPatch(
                    class_name="Alpha",
                    method_name="forward",
                    body=cp_dispatch_shell_body("kimi_cp_forward"),
                )
            ]
        ),
    )
    _, entries, classes = _boundary("blocks.alpha", "Alpha")
    rendered = lower_forward_boundaries(
        rendered,
        {
            "param_plan": entries,
            "injections": [],
            "mesh_dim_names": ("cp", "tp"),
        },
        boundary_classes=classes,
    )

    _verify_recorded_patch_materialization(rendered, meta, patch_plan=patches)

    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha",
        "host_class": "Alpha",
        "kind": "inner_wrapper",
        "source_form": "same_class_toggle_forward",
    }]


def test_replacement_verifier_requires_constructor_call():
    """An imported replacement symbol alone is not a materialized replacement."""
    source_type = type("SourceBlock", (), {})
    replacement_type = type("ReplacementBlock", (), {})
    record = SimpleNamespace(
        module_fqn="blocks.alpha",
        source_type=source_type,
        replacement_type=replacement_type,
    )
    patches = SimpleNamespace(
        local_computes=(),
        module_replacements=(record,),
        inner_wrappers=(),
    )
    meta = SimpleNamespace(materialized_patches=[])
    imported_only = "from demo import ReplacementBlock\n"

    with pytest.raises(RuntimeError, match="were not materialized"):
        _verify_recorded_patch_materialization(
            imported_only, meta, patch_plan=patches
        )

    rendered = imported_only + "\nblock = ReplacementBlock(source)\n"
    _verify_recorded_patch_materialization(rendered, meta, patch_plan=patches)

    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha",
        "host_class": "ReplacementBlock",
        "kind": "module_replacement",
        "source_form": "constructor",
    }]


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_bound_method_local_compute_satisfies_verification():
    """Feature: Bound-method source export.

    Description: Render a local compute method within the observed replacement.
    Expectation: Exported construction and forward ownership satisfy replacement verification.
    """
    module_type = type("ImportedMoe", (), {})

    def source_forward(self: Any, hidden_states: Any) -> Any:
        """Return the generic fixture input unchanged."""
        del self
        return hidden_states

    record = SimpleNamespace(
        module_fqn="blocks.alpha.mlp",
        module_type=module_type,
        original_forward=source_forward,
    )
    patches = SimpleNamespace(
        local_computes=(record,),
        module_replacements=(),
        inner_wrappers=(),
    )
    meta = SimpleNamespace(materialized_patches=[])
    rendered = """
def source_forward_hp_impl(self, hidden_states):
    return hidden_states

def source_forward(self, hidden_states):
    if self.ep_enable:
        return self.__hp_compute__(hidden_states)
    return self._forward_impl(hidden_states)

owner._forward_impl = MethodType(source_forward_hp_impl, owner)
owner.forward = MethodType(source_forward, owner)
"""

    _verify_recorded_patch_materialization(rendered, meta, patch_plan=patches)

    assert meta.materialized_patches == [{
        "fqn": "blocks.alpha.mlp",
        "host_class": "ImportedMoe",
        "kind": "local_compute",
        "source_form": "bound_method_toggle_forward",
    }]


def test_imported_cp_host_is_rendered_as_same_name_class():
    """An imported replacement gets a visible same-name CP class."""
    patches = InlinePatchSet(
        imports=[ImportPatch("demo.components", ("MLAAttention",))],
        forward_extracts=[
            ForwardExtractPatch(
                class_name="MLAAttention",
                method_name="forward",
                body=cp_dispatch_shell_body("mla_cp_forward"),
            )
        ],
    )

    _materialize_imported_forward_hosts(SOURCE_TEXT, patches)
    rendered = apply_patch_set(SOURCE_TEXT, patches)

    assert "import demo.components" in rendered
    assert "class MLAAttention(demo.components.MLAAttention):" in rendered
    assert "class HPCodegen" not in rendered
    assert "if self.cp_enable:" in rendered
    assert "return super().forward(*args, **kwargs)" in rendered
    assert rendered.count("def _forward_impl(") == 1
    assert "return self._forward_impl(*args, **kwargs)" in rendered


def test_observed_replacement_supplies_missing_cp_host_import():
    """A native replacement record supplies an imported CP host generically."""
    replacement_type = type("ObservedAttention", (), {})
    replacement_type.__module__ = "demo.components"
    patches = InlinePatchSet(
        forward_extracts=[
            ForwardExtractPatch(
                class_name="ObservedAttention",
                method_name="forward",
                body=cp_dispatch_shell_body("observed_cp_forward"),
            )
        ]
    )
    patch_plan = SimpleNamespace(
        module_replacements=(
            SimpleNamespace(replacement_type=replacement_type),
        )
    )

    _add_recorded_forward_host_imports(SOURCE_TEXT, patches, patch_plan)
    _materialize_imported_forward_hosts(SOURCE_TEXT, patches)
    rendered = apply_patch_set(SOURCE_TEXT, patches)

    assert "import demo.components" in rendered
    assert "class ObservedAttention(demo.components.ObservedAttention):" in rendered
    assert "return observed_cp_forward(self, *args, **kwargs)" in rendered


def test_observed_inner_wrapper_rebinds_boundary_to_replacement_type():
    """Strategy ownership follows the target selected by the native wrapper."""
    replacement_type = type("ObservedAttention", (), {})
    patch_plan = SimpleNamespace(
        inner_wrappers=(
            SimpleNamespace(
                module_fqn="model.layers.0.self_attn",
                target_fqn="self",
                target_type=replacement_type,
            ),
        )
    )
    meta = SimpleNamespace(
        boundary_classes={
            "model.layers.0.self_attn": "OriginalAttention",
        }
    )

    classes = _observed_boundary_classes(meta, patch_plan)

    assert classes == {
        "model.layers.0.self_attn": "ObservedAttention",
    }


def test_model_adapter_replacement_keeps_declared_factory_contract():
    """Observed types do not erase a model-owned factory's interface facts."""
    meta = SimpleNamespace(
        module_overrides=[{
            "factory": "hyper_parallel.components.modules.GroupedQueryAttention",
            "replacement_target": (
                "hyper_parallel.models.qwen3_moe.adapter.conversion."
                "module_replacement.replace_qwen3_moe_flash_attention"
            ),
            "module_type": "demo.OriginalAttention",
            "fqns": ["model.layers.0.self_attn"],
        }],
        injections=[],
    )

    rules = collect_source_observations(meta)

    assert len(rules) == 1
    assert rules[0].replace_target == meta.module_overrides[0]["replacement_target"]


def test_imported_cp_host_and_boundary_share_one_forward_impl():
    """CP dispatch and boundary lowering compose into one implementation."""
    patches = InlinePatchSet(
        imports=[ImportPatch("demo.components", ("MLAAttention",))],
        forward_extracts=[
            ForwardExtractPatch(
                class_name="MLAAttention",
                method_name="forward",
                body=cp_dispatch_shell_body("mla_cp_forward"),
            )
        ],
    )
    _materialize_imported_forward_hosts(SOURCE_TEXT, patches)
    rendered = apply_patch_set(SOURCE_TEXT, patches)
    _, entries, classes = _boundary(
        "model.layers.0.self_attn", "MLAAttention"
    )
    plan = {
        "param_plan": entries,
        "injections": [],
        "mesh_dim_names": ("cp", "tp"),
    }

    rendered = lower_forward_boundaries(
        rendered, plan, boundary_classes=classes
    )
    tree = ast.parse(rendered)
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLAAttention"
    )
    methods = [
        node.name
        for node in host.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert methods.count("_forward_impl") == 1
    assert methods.count("forward") == 1
    assert "outputs = mla_cp_forward(self, *args, **kwargs)" in rendered
    assert "return super().forward(*args, **kwargs)" in rendered
    assert "outputs = self._forward_impl(*args, **kwargs)" in rendered
    assert "return self._forward_impl(*args, **kwargs)" not in rendered


@pytest.mark.parametrize("composed_source", [False, True])
def test_imported_replacement_owner_is_lowered_after_root_construction(composed_source):
    """An observed wrapper needs no constructor call in the copied source."""
    source = """\
class Root(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ImportedDecoder(config)
"""
    if composed_source:
        source += """\
class OtherCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ImportedDecoder(config)
"""
    patches = InlinePatchSet(
        imports=[ImportPatch("demo.replacements", ("WrappedExperts",))],
        constructor_replaces=[
            ConstructorReplacePatch(
                old_ctor="ImportedExperts",
                new_ctor="WrappedExperts",
                mode="wrap_source",
                keyword_args=("module_fqn=''", "context={}"),
                scope_attrs=("experts",),
                scope_fqns=(
                    "model.layers.0.mlp.experts",
                    "model.layers.1.mlp.experts",
                ),
            )
        ],
    )

    if composed_source:
        for root_class_name in (None, "MissingRoot"):
            with pytest.raises(RuntimeError, match="ImportedExperts"):
                apply_patch_set(source, patches, root_class_name=root_class_name)
    rendered = apply_patch_set(source, patches, root_class_name="Root")

    ast.parse(rendered)
    assert "self.model = ImportedDecoder(config)" in rendered
    assert "_hp_replacement_cache" not in rendered
    assert "for _hp_fqn" not in rendered
    assert "getattr(self.model.layers, '0').mlp.experts = WrappedExperts(" in rendered
    assert "getattr(self.model.layers, '1').mlp.experts = WrappedExperts(" in rendered
    assert "WrappedExperts(" in rendered
    assert "module=getattr(self.model.layers, '0').mlp.experts" in rendered
    assert "module_fqn='model.layers.1.mlp.experts', context={}" in rendered
    if composed_source:
        other_class = next(node for node in ast.parse(rendered).body
                           if isinstance(node, ast.ClassDef) and node.name == "OtherCausalLM")
        assert ast.dump(other_class) == ast.dump(ast.parse(source).body[-1])


def test_tp_collective_boundary_renders_static_operators():
    """A TP-lowerable boundary with a static signature renders bare operators.

    Feature: codegen-lowering
    Description: A boundary whose declared transitions are all TP-lowerable
        and whose forward passes the structural gates is rewritten to the
        static template: the class marker, ``_forward_impl`` extraction, and
        ``self._hp_tp`` operator calls instead of the generic redistribute.
    Expectation: The emitted text carries the ``tp_collective`` marker, the
        input all_gather call, the extracted impl call, and the output
        reduce_scatter call; no ``self._hp_boundary`` reference remains.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert '_hp_boundary_form = "tp_collective"' in text
    assert "x = self._hp_tp.all_gather(x, dim=1)" in text
    assert "outputs = self._forward_impl(x)" in text
    assert "outputs = self._hp_tp.reduce_scatter(outputs, dim=1)" in text
    assert "self._hp_boundary" not in text


def test_identity_boundary_forward_is_pruned():
    """An all-identity boundary keeps its original forward untouched.

    Feature: codegen-lowering
    Description: A boundary whose declared transitions are all identity is
        pruned: the runtime install path covers the exact no-op / to-local
        semantics with its generic wrapper, so rewriting the forward would
        add indirection without changing behavior.
    Expectation: The emitted text equals the source text (no
        ``_forward_impl``, no marker, no redistribute calls).
    """
    plan, classes = _single_plan(_identity_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert text == SOURCE_TEXT


def _degenerate_entry() -> dict:
    """A degenerate-topology boundary: full per-axis dicts, every axis identity.

    This is the frozen shape a tp=cp=ep=1 regeneration produces: the freeze
    declares every axis (``cp``/``ep``/``tp``) with canonical strings, so an
    identity ``ep: R -> R`` key is present even though the entry is not
    EP-dependent.
    """
    return {
        "is_boundary": True,
        "in_src": {"x": {"cp": "R", "ep": "R", "tp": "R"}},
        "in_dst": {"x": {"cp": "R", "ep": "R", "tp": "R"}},
        "out_src": {"output": {"cp": "R", "ep": "R", "tp": "R"}},
        "out_dst": {"output": {"cp": "R", "ep": "R", "tp": "R"}},
    }


def test_degenerate_topology_identity_ep_keys_are_pruned():
    """Empty active axes + identity ``ep`` keys prune the rewrite entirely.

    Regression lock for the tp=cp=ep=1 failure: reading ``ep`` key
    *existence* as EP dependence forced the generic form on boundaries whose
    semantics are an exact no-op, while the install path is skipped entirely
    for such meshes — the rewritten forward's ``_hp_boundary`` reference
    was never bound and training crashed with ``AttributeError``.  Only an
    ``ep`` placement that actually *changes* is an expert-mesh dependency.

    Feature: codegen-lowering
    Description: A plan with no active axes (``mesh_dim_names`` empty/None)
        and a boundary declaring full per-axis dicts with identity ``ep``
        keys is lowered through ``lower_forward_boundaries``.
    Expectation: The emitted text equals the source text — no
        ``_forward_impl``, no marker, no ``_hp_boundary`` reference.
    """
    plan, classes = _single_plan(_degenerate_entry())

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert text == SOURCE_TEXT


def test_degenerate_topology_changing_ep_renders_generic():
    """Empty active axes + a changing ``ep`` placement keep the generic rewrite.

    The mirror of the pruning test: an entry whose ``ep`` placement actually
    changes (``R -> S(0)``) is genuinely expert-mesh dependent — not
    statically decidable — so it must keep the rewritten redistribute forward
    required by the emitted redistribution interface.

    Feature: codegen-lowering
    Description: A plan with no active axes and a boundary whose ``ep``
        placement changes is lowered through ``lower_forward_boundaries``.
    Expectation: The forward is rewritten to the generic form:
        ``_forward_impl`` extraction and ``self._hp_boundary`` calls.
    """
    entry = _degenerate_entry()
    entry["in_dst"]["x"]["ep"] = "S(0)"
    entry["out_dst"]["output"]["ep"] = "S(0)"
    plan, classes = _single_plan(entry)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    assert "_forward_impl" in text
    assert "self._hp_boundary.redistribute_inputs" in text
    assert "self._hp_boundary.redistribute_outputs" in text


def test_tp_collective_falls_back_to_generic_on_variadic_signature():
    """A ``*args``/``**kwargs`` forward cannot carry the static template.

    Feature: codegen-lowering
    Description: The static template re-passes every declared parameter by
        name, so a variadic signature fails the structural gates and the
        boundary renders as the generic redistribute form instead — even
        though the plan's transitions classify as tp_collective.
    Expectation: No marker and no ``_hp_tp`` calls; the forward goes
        through ``self._hp_boundary.redistribute_inputs/outputs``.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(
        VARIADIC_SOURCE_TEXT, plan, boundary_classes=classes
    )

    assert "_hp_boundary_form" not in text
    assert "_hp_tp" not in text
    assert "_forward_impl" in text
    assert "self._hp_boundary.redistribute_inputs" in text
    assert "self._hp_boundary.redistribute_outputs" in text


def test_generic_boundary_keeps_adjacent_composed_class_separate():
    """A consumed trailing newline cannot join a return to the following class."""
    source = ("class Alpha:\n    def forward(self, x, **kwargs):\n        return x\n"
              "class Beta:\n    def forward(self, x):\n        return x + 1\n")
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)
    text = lower_forward_boundaries(source, plan, boundary_classes=classes)
    tree = ast.parse(text)
    assert [node.name for node in tree.body if isinstance(node, ast.ClassDef)] == ["Alpha", "Beta"]
    assert "return outputs\nclass Beta" in text


def test_tp_collective_allows_kwargs_and_tuple_output():
    """HF-style ``**kwargs`` and tuple output can still expose bare TP ops.

    Feature: codegen-lowering
    Description: A TP-lowerable boundary whose forward has ``**kwargs`` and
        returns a tuple is lowered to the static template. The generated body
        re-passes ``**kwargs`` and applies output redistribution to index 0,
        matching the frozen output-plan convention.
    Expectation: The emitted text carries the TP marker, keeps ``**kwargs`` in
        the ``_forward_impl`` call, rebuilds the tuple, and has no generic
        boundary calls.
    """
    plan, classes = _single_plan(_tp_collective_entry())
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(
        KWARGS_TUPLE_SOURCE_TEXT, plan, boundary_classes=classes
    )

    assert '_hp_boundary_form = "tp_collective"' in text
    assert "outputs = self._forward_impl(x, y, **kwargs)" in text
    assert "_hp_output_0 = self._hp_tp.reduce_scatter(outputs[0], dim=1)" in text
    assert "outputs = (_hp_output_0, *outputs[1:])" in text
    assert "self._hp_boundary" not in text


# ---------------------------------------------------------------------------
# D-22: deferred bias exit hook rendering
# ---------------------------------------------------------------------------

_HOOK_LINE = "outputs = self._hp_deferred_bias(outputs)"


def _generic_entry() -> dict:
    """A generic-form boundary (cp transition is not TP-lowerable)."""
    return {
        "is_boundary": True,
        "in_src": {"x": {"tp": "R", "cp": "S(1)"}},
        "in_dst": {"x": {"tp": "R", "cp": "R"}},
        "out_src": {"output": {"tp": "P(sum)", "cp": "S(1)"}},
        "out_dst": {"output": {"tp": "R", "cp": "S(1)"}},
    }


def test_generic_deferred_bias_hook_renders_last():
    """D-22: the exit hook is the LAST statement of the generic exit block.

    Feature: codegen-lowering
    Description: A generic-form boundary whose frozen entry carries
        ``deferred_bias_params`` is lowered; the rowwise bias is suppressed
        inside the region and must be re-added once, after the output
        redistribution.
    Expectation: The hook line is emitted after ``redistribute_outputs``.
    """
    entry = _generic_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["mesh_dim_names"] = ("cp", "tp")

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    hook_at = text.index(_HOOK_LINE)
    assert hook_at > text.index("self._hp_boundary.redistribute_outputs(outputs)")


def test_tp_collective_deferred_bias_hook_renders_after_exit_ops():
    """D-22: the static template re-adds the bias after the exit reduction.

    Feature: codegen-lowering
    Description: A tp_collective boundary whose frozen entry carries deferred
        bias params renders the hook after the output-side bare operators (the
        Partial -> R all_reduce / reduce_scatter IS the exit reduction the
        deferral waits for) and before the return.
    Expectation: The hook line sits after the reduce_scatter call and before
        ``return outputs``.
    """
    entry = _tp_collective_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["mesh_dim_names"] = ("tp",)

    text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)

    hook_at = text.index(_HOOK_LINE)
    assert hook_at > text.index(
        "outputs = self._hp_tp.reduce_scatter(outputs, dim=1)"
    )
    assert hook_at < text.index("return outputs")


def test_local_region_deferred_bias_still_requires_native_phase_c():
    """D-22 does not permit a local-region source rewrite.

    Feature: codegen-lowering
    Description: A region boundary with deferred bias reaches source lowering.
    Expectation: Generation fails so native Phase C owns both the region and
        its deferred-bias ordering.
    """
    entry = _identity_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["injections"] = [
        {"match": "blocks.alpha", "local_compute_fn": {"_target_": "some.factory"}},
    ]

    with pytest.raises(RuntimeError, match="native Phase C wrapper"):
        lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)


def test_no_deferred_bias_no_hook_line():
    """Entries without deferred params emit no hook (byte-stable artifacts).

    Feature: codegen-lowering
    Description: Generic and static boundaries whose entries carry no
        ``deferred_bias_params`` key are lowered.
    Expectation: No ``_hp_deferred_bias`` reference appears — the emitted
        bytes for non-deferring plans are unchanged by the D-22 feature.
    """
    for entry in (_generic_entry(), _tp_collective_entry()):
        plan, classes = _single_plan(entry)
        plan["mesh_dim_names"] = ("cp", "tp")

        text = lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)
        assert "_hp_deferred_bias" not in text


def test_identity_boundary_with_deferred_bias_fails_fast():
    """D-22: an identity-pruned boundary cannot carry deferred bias params.

    Feature: codegen-lowering
    Description: An all-identity boundary whose entry declares deferred bias
        params is lowered. The identity form leaves the original forward
        untouched, so the suppressed bias would never be re-added.
    Expectation: ``NotImplementedError`` is raised at generation time instead
        of silently dropping the bias at train time.
    """
    entry = _identity_entry()
    entry["deferred_bias_params"] = ["o_proj.bias"]
    plan, classes = _single_plan(entry)
    plan["mesh_dim_names"] = ("tp",)

    try:
        lower_forward_boundaries(SOURCE_TEXT, plan, boundary_classes=classes)
    except NotImplementedError as exc:
        assert "identity" in str(exc)
        assert "deferred bias" in str(exc)
    else:
        raise AssertionError("identity + deferred bias must fail fast")


@pytest.mark.parametrize("container", ["ModuleList", "ModuleDict"])
def test_explicit_replacements_preserve_partial_layers_aliases_and_initialization(container, tmp_path):
    """Explicit paths preserve native late wrapping, distinct instances and shared parameters."""
    source = """class Root(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = build_layers()
        self.model.alias = self.model.layers[0] if isinstance(self.model.layers, nn.ModuleList) else self.model.layers['0']
        initialize_layers(self.model.layers)
        self.initialized = True
"""
    calls = []
    originals = [nn.Linear(1, 1) for _ in range(3)]

    def build_layers():
        return nn.ModuleList(originals) if container == "ModuleList" else nn.ModuleDict(
            {str(index): module for index, module in enumerate(originals)})

    def initialize_layers(layers):
        for module in layers.children():
            module.weight.detach().fill_(7)

    class Wrapped(nn.Module):
        def __init__(self, *, module, module_fqn, mode):
            super().__init__()
            self.weight = module.weight
            self.mode = mode
            self.register_buffer("initial_weight", module.weight.detach().clone())
            calls.append(module_fqn)

    patches = InlinePatchSet(constructor_replaces=[ConstructorReplacePatch(
        old_ctor="Linear", new_ctor="Wrapped", mode="wrap_source", defer_to_fqns=True,
        keyword_args=("mode='first'",), scope_attrs=("0", "alias"),
        scope_fqns=("model.layers.0", "model.alias"),
        shared_fqns=(("model.layers.0", "model.alias"),),
    ), ConstructorReplacePatch(
        old_ctor="Linear", new_ctor="Wrapped", mode="wrap_source", defer_to_fqns=True,
        keyword_args=("mode='last'",), scope_attrs=("2",), scope_fqns=("model.layers.2",),
    )])
    rendered = apply_patch_set(source, patches, root_class_name="Root")
    path = tmp_path / "explicit_model.py"
    path.write_text(rendered, encoding="utf-8")
    spec = util.spec_from_file_location("explicit_model", path)
    generated = util.module_from_spec(spec)
    generated.nn, generated.build_layers, generated.Wrapped = nn, build_layers, Wrapped
    generated.initialize_layers = initialize_layers
    spec.loader.exec_module(generated)
    model = generated.Root(None)
    first = model.get_submodule("model.layers.0")
    assert model.initialized
    assert model.model.alias is first
    assert first.weight is originals[0].weight
    assert torch.equal(first.initial_weight, torch.full_like(first.initial_weight, 7))
    assert first.mode == "first"
    assert model.get_submodule("model.layers.2").mode == "last"
    assert model.get_submodule("model.layers.1") is originals[1]
    assert model.get_submodule("model.layers.2").weight is originals[2].weight
    assert calls == ["model.layers.0", "model.layers.2"]
    assert rendered.index("self.initialized = True") < rendered.index("Wrapped(")
    assert "model.layers.0: Linear -> Wrapped" in rendered
    assert "_hp_replacement_cache" not in rendered


def test_shared_replacement_groups_survive_existing_snapshot_projection(monkeypatch):
    """Retain native alias groups when equal replacement rules are coalesced."""
    monkeypatch.setattr(replacement_pass, "replacement_spec_for_observation", lambda *_args, **_kwargs: ReplacementSpec(
        old_ctor="Original", new_ctor="Wrapped", imports=(), mode="wrap_source", defer_to_fqns=True))
    rules = collect_source_observations(SimpleNamespace(module_overrides=[
        {"fqns": ["model.layers.0", "model.alias"], "factory": "demo.Wrapped"},
        {"fqns": ["model.layers.2"], "factory": "demo.Wrapped"},
    ]))
    patches = replacement_pass.build_replacement_patches(rules)
    assert patches.constructor_replaces[0].shared_fqns == (("model.layers.0", "model.alias"),)
    assert set(patches.constructor_replaces[0].scope_fqns) == {
        "model.layers.0", "model.alias", "model.layers.2"}


def test_explicit_replacement_supports_non_identifier_module_names():
    """Numeric and keyword child names remain native attribute names."""
    source = """class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({'class': nn.Linear(1, 1)})
"""
    patches = InlinePatchSet(constructor_replaces=[ConstructorReplacePatch(
        old_ctor="Linear", new_ctor="Wrapped", mode="wrap_source", defer_to_fqns=True,
        scope_attrs=("class",), scope_fqns=("model.class",),
    )])
    rendered = apply_patch_set(source, patches, root_class_name="Root")
    ast.parse(rendered)
    assert "setattr(self.model, 'class', Wrapped(" in rendered
    assert "module=getattr(self.model, 'class')" in rendered


@pytest.mark.parametrize("missing_fqn", ["model.layers.0.attn", "model.layers.2.attn"])
def test_explicit_replacement_verification_requires_each_observed_layer(missing_fqn):
    """A visible constructor at one layer cannot stand in for another observed replacement."""
    replacement_type = type("Wrapped", (), {})
    records = [SimpleNamespace(module_fqn=fqn, replacement_type=replacement_type)
               for fqn in ("model.layers.0.attn", "model.layers.2.attn")]
    plan = SimpleNamespace(module_replacements=records, local_computes=(), inner_wrappers=())
    calls = [f"Wrapped(module=source, module_fqn={record.module_fqn!r})"
             for record in records if record.module_fqn != missing_fqn]
    with pytest.raises(RuntimeError, match=missing_fqn):
        _verify_recorded_patch_materialization("\n".join(calls), SimpleNamespace(), patch_plan=plan)


@pytest.mark.parametrize("partial", [False, True])
def test_observed_norm_constructor_preserves_policy_and_instance_scope(partial):
    """Concrete norm policy is retained and partial selection never rewrites the shared class."""
    source = """class OriginalNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = OriginalNorm(config.hidden_size, eps=config.rms_norm_eps)
class Root(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.ModuleList([Layer(config), Layer(config)])
"""
    rule = SourceObservation(
        match=("model.0.norm",) if partial else ("model.0.norm", "model.1.norm"),
        module_type="demo.OriginalNorm", replace_target="unused.factory",
        construction={"implementation": f"{RMSNormModule.__module__}:{RMSNormModule.__qualname__}",
                      "source_fqns": ["model.0.norm", "model.1.norm"],
                      "arguments": {"cast_order": "before_weight", "fp32_weight": False}},
    )
    patches = replacement_pass.build_replacement_patches((rule,))
    rendered = apply_patch_set(source, patches, root_class_name="Root")
    ast.parse(rendered)
    assert "cast_order='before_weight'" in rendered
    assert "fp32_weight=False" in rendered
    if partial:
        assert "self.norm = OriginalNorm(" in rendered
        assert "getattr(self.model, '0').norm = RMSNormModule(" in rendered
        assert "getattr(self.model, '1').norm =" not in rendered
    else:
        assert "self.norm = RMSNormModule(config.hidden_size, eps=config.rms_norm_eps," in rendered
        assert "getattr(self.model" not in rendered


class _RequiredSourceWrapper(nn.Module):
    """Norm-like option names do not make a source wrapper directly constructible."""

    def __init__(self, module, *, hidden_size=4, eps=1e-6):
        super().__init__()
        self.weight = module.weight
        self.original_forward = module.forward
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, value):
        """Delegate to the source module without changing parameter ownership."""
        return self.original_forward(value)


class _KeywordSourceWrapper(_RequiredSourceWrapper):
    """A keyword-only source argument is equally mandatory."""

    def __init__(self, *, module, hidden_size=4, eps=1e-6):
        super().__init__(module, hidden_size=hidden_size, eps=eps)


@pytest.mark.parametrize("component", [_RequiredSourceWrapper, _KeywordSourceWrapper])
def test_observed_constructor_requires_source_despite_norm_like_options(component):
    """A native-valid wrapper remains constructible and retains the source Parameter."""
    source = """class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(4, 4, bias=False)
    def forward(self, value):
        return self.layer(value)
"""
    rule = SourceObservation(
        match=("layer",), module_type="torch.nn.modules.linear.Linear", replace_target="unused.factory",
        construction={"implementation": f"{__name__}:{component.__name__}", "source_fqns": ["layer"]},
    )
    patches = replacement_pass.build_replacement_patches((rule,))
    assert patches.constructor_replaces[0].mode == "wrap_source"
    rendered = apply_patch_set(source, patches)
    assert f"self.layer = {component.__name__}(module=Linear(4, 4, bias=False))" in rendered
    scope = {"nn": nn, "Linear": nn.Linear}
    exec(compile(rendered, "generated_wrapper.py", "exec"), scope)  # pylint: disable=exec-used
    generated = scope["Root"]()
    original = generated.layer.original_forward.__self__
    assert generated.layer.weight is original.weight
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(generated(value), original(value), rtol=0, atol=0)


@pytest.mark.parametrize("component", [RMSNormModule, NativeRMSNormModule])
@pytest.mark.parametrize("parameters,invocation,expected_eps", [
    ("hidden_size, eps=1e-6", "4", 1e-6),
    ("hidden_size, eps=1e-6", "hidden_size=4", 1e-6),
    ("hidden_size, eps=1e-6", "4, 2e-4", 2e-4),
    ("hidden_size, eps=1e-6", "4, eps=2e-4", 2e-4),
    ("hidden_size, *, eps=1e-6", "4", 1e-6),
    ("hidden_size, *, eps=1e-6", "4, eps=2e-4", 2e-4),
])
def test_direct_norm_constructor_preserves_source_defaults(component, parameters, invocation, expected_eps):
    """Omitted source defaults and explicit overrides both survive direct construction."""
    source = f"""class OriginalNorm(nn.Module):
    def __init__(self, {parameters}):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = OriginalNorm({invocation})
"""
    rule = SourceObservation(
        match=("norm",), module_type="demo.OriginalNorm", replace_target="unused.factory",
        construction={"implementation": f"{component.__module__}:{component.__qualname__}",
                      "source_fqns": ["norm"],
                      "arguments": {"cast_order": "before_weight", "fp32_weight": False}},
    )
    scope = {"nn": nn, "torch": torch}
    exec(compile(source, "original_norm.py", "exec"), scope)  # pylint: disable=exec-used
    original = scope["Root"]().norm
    native = component(module=original, cast_order="before_weight", fp32_weight=False)
    patches = replacement_pass.build_replacement_patches((rule,))
    assert patches.constructor_replaces[0].mode == "construct"
    rendered = apply_patch_set(source, patches)
    exec(compile(rendered, "generated_norm.py", "exec"), scope)  # pylint: disable=exec-used
    generated = scope["Root"]().norm
    assert generated.eps == native.eps == expected_eps
    assert generated.rms_norm_cast_order == native.rms_norm_cast_order == "before_weight"
    assert generated.fp32_weight is False
    assert native.fp32_weight is False
    assert native.weight is original.weight
    torch.testing.assert_close(generated.weight, native.weight, rtol=0, atol=0)
    if component is NativeRMSNormModule:
        value = torch.linspace(-1e-3, 1e-3, steps=8).reshape(2, 4)
        torch.testing.assert_close(generated(value), native(value), rtol=0, atol=0)


class _UnusedContextConstructor:
    """Source conversion deliberately ignores framework context."""

    def __init__(self, *, module=None, context=None):
        del context
        self.module = module


class _ForwardedContextConstructor(_UnusedContextConstructor):
    """The child forwards a context that its parent does not inspect."""

    def __init__(self, *, module=None, context=None):
        super().__init__(module=module, context=context)
        self.prepared = True


class _ConsumedContextConstructor(_ForwardedContextConstructor):
    """Logical CP selection reads context before forwarding it."""

    def __init__(self, *, module=None, context=None):
        self.cp_size = (context or {}).get("cp_size", 1)
        super().__init__(module=module, context=context)


@pytest.mark.parametrize("constructor,ignored", [
    (_UnusedContextConstructor, True), (_ForwardedContextConstructor, True), (_ConsumedContextConstructor, False),
])
def test_constructor_context_is_removed_only_when_native_code_does_not_read_it(constructor, ignored):
    """Dropping ignored metadata must not remove a logical topology dependency."""
    assert ("context" in _ignored_constructor_parameters(constructor)) is ignored


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("same_iteration", [False, True])
def test_imported_experts_reuse_native_layer_preparation_loop(complete, same_iteration):
    """Imported constructors can use a local owner loop without patching other owners."""
    source = """class Body(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([ImportedLayer(config, i) for i in range(config.num_hidden_layers)])
        for i in range(config.num_hidden_layers):
            layer = self.layers[i]
            prepare(layer)
class Root(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Body(config)
"""
    if not same_iteration:
        source = source.replace("for i in range(config.num_hidden_layers):", "for i in range(1):")
    patch = ConstructorReplacePatch(
        old_ctor="ImportedExperts", new_ctor="Wrapped", mode="wrap_source", scope_attrs=("experts",),
        scope_fqns=("model.layers.0.mlp.experts", "model.layers.2.mlp.experts"), scope_complete=complete,
    )
    rendered = apply_patch_set(source, InlinePatchSet(constructor_replaces=[patch]), root_class_name="Root")
    ast.parse(rendered)
    if not same_iteration:
        assert "layer.mlp.experts =" not in rendered
        assert "getattr(self.model.layers, '0').mlp.experts = Wrapped(" in rendered
        return
    assert "layer.mlp.experts = Wrapped(module=layer.mlp.experts)" in rendered
    assert "getattr(self.model" not in rendered
    assert ("if i in (0, 2):" in rendered) is not complete
    assert rendered.index("layer.mlp.experts =") > rendered.index("prepare(layer)")

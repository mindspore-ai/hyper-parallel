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
"""Family preparation and codegen observation share the native installation contract."""

import json
import gc
from functools import wraps
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from torch import nn

from hyper_parallel.codegen.model_build_overlay import (
    _observe_plan, _observe_replacement_installations, _observe_replacements, use_codegen_model_build_overlay,
)
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.build import _ObservationMesh
from hyper_parallel.codegen.inline.observations import collect_source_observations
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.ir import ImportPatch
from hyper_parallel.codegen.inline.pipeline import _materialize_imported_forward_hosts
from hyper_parallel.distributed.recipe_spec import inner_wrapper, ModuleShardingSpec
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest, _commit_forward_rewrite
from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.components.modules import family as registry
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.models import HyperAutoModelForCausalLM, module_build
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.module_build import attach_build_context, build_module_context
from hyper_parallel.models.replacement import (
    MethodReplacementSpec, ModuleReplacementRequest, ModuleReplacementSpec, apply_module_replacements,
    compile_module_replacements, method_replacement, module_replacement,
)
from tests.common.module_family_fixtures import (
    FamilyModelConfig, LinearFamily, LocalLinear, ParallelLinear, create_family_model, register_linear_family,
)


class _MethodModel(nn.Module):
    """A real model with an explicit non-forward stage for shared policy acceptance."""

    def __init__(self, config, *, context):
        super().__init__()
        del context
        self.config = config
        self.weight = nn.Parameter(torch.ones(()))

    def stage(self, inputs):
        return inputs * self.weight

    def forward(self, inputs):
        return self.stage(inputs)


class _TiedModel(nn.Module):
    """Expose the native tied-weight contract without a Transformers dependency."""

    def __init__(self, config, *, context):
        super().__init__()
        del context
        self.config = config
        self.embedding = nn.Embedding(4, 3)
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.all_tied_weights_keys = {"lm_head.weight": "embedding.weight"}

    def tie_weights(self, *, recompute_mapping: bool) -> None:
        """Establish the declared parameter alias using the existing mapping."""
        if recompute_mapping:
            raise ValueError("The native helper must reuse the declared mapping")
        self.lm_head.weight = self.embedding.weight

    def get_parameter_or_buffer(self, name: str) -> nn.Parameter:
        """Resolve a declared alias endpoint for the native identity check."""
        return self.get_parameter(name)


_OBSERVED_CALLS = []


class _ObservedLayer(nn.Module):
    """Host whose native factory chooses one of two different closures."""

    def __init__(self):
        """Keep one scalar configuration on the source host."""
        super().__init__()
        self.bias = 7

    def forward(self, value):
        """Provide the original forward that both closures capture."""
        return value * 2


@inner_wrapper
def _selected_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Choose the forward in the native factory, not in the source recognizer."""
    del mesh, tp_mesh, cp_mesh, ep_mesh
    _OBSERVED_CALLS.append(id(target_module))
    original_forward = target_module.forward
    bias = target_module.bias
    if bias == 0:
        raise ValueError("native wrapper failure")
    if bias == 99:
        return _ForwardRewriteRequest(target_module, original_forward)

    @wraps(original_forward)
    def positive(value):
        return original_forward(value) + bias

    @wraps(original_forward)
    def negative(value):
        return original_forward(value) - bias

    if getattr(target_module, "inplace", False):
        target_module.forward = negative
        return None
    return _ForwardRewriteRequest(target_module, positive if bias > 0 else negative)



@inner_wrapper
def _mesh_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Capture the exact CP mesh provided by native resolution."""
    del mesh, tp_mesh, ep_mesh
    _OBSERVED_CALLS.append(id(target_module))
    original = target_module.forward

    @wraps(original)
    def selected(value):
        return original(value) + cp_mesh.get_local_rank()

    return _ForwardRewriteRequest(target_module, selected)



def _source_helper(module, value, *, bias=None, **kwargs):
    """Make a delegated helper's keyword and target observable in execution."""
    return value * 2 + bias + kwargs.get("extra", 0) + module.bias


@inner_wrapper
def _helper_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Delegate actual arguments while capturing a native configuration value."""
    del mesh, tp_mesh, cp_mesh, ep_mesh
    _OBSERVED_CALLS.append(id(target_module))
    original = target_module.forward
    bias = 3

    @wraps(original)
    def selected(value, **kwargs):
        return _source_helper(target_module, value, bias=bias, **kwargs)

    return _ForwardRewriteRequest(target_module, selected)



@inner_wrapper
def _state_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh, selection=None):
    """Publish optional native state separately from the explicit configuration."""
    del mesh, cp_mesh, ep_mesh
    original = target_module.forward
    context = None

    @wraps(original)
    def selected(value: Any) -> Any:
        return original(value), context, selection, tp_mesh

    return _ForwardRewriteRequest(
        target_module, selected,
        companion_attrs={"published_context": context, "published_selection": selection, "published_tp": tp_mesh},
    )



class _WeightedLayer(_ObservedLayer):
    """Keep parameter identity observable through executor materialization."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, value):
        return value * self.weight * 2


class _FlatExecutorBase(nn.Module):
    """Flat native executor with inherited computation and runtime mesh metadata."""

    def __init__(self, module, cp_mesh):
        super().__init__()
        self.module = module
        self.cp_mesh = cp_mesh
        self.cp_rank = cp_mesh.get_local_rank()
        self.scale = 3

    def _project(self, value):
        return value * self.module.weight * self.scale + self.module.bias

    def forward(self, value):
        return self._project(value) + self.cp_rank


class _FlatExecutor(_FlatExecutorBase):
    """Exercise inherited forward and helper source discovery."""


@inner_wrapper
def _executor_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Return the executor selected by native construction exactly once."""
    del mesh, tp_mesh, ep_mesh
    _OBSERVED_CALLS.append(id(target_module))
    executor = _FlatExecutor(target_module, cp_mesh)
    original = target_module.forward

    @wraps(original)
    def selected(*args, **kwargs):
        return executor(*args, **kwargs)

    return _ForwardRewriteRequest(target_module, selected)


class TestObservedWrapperConnection(unittest.TestCase):
    """Native factory decisions must survive the frozen plan and existing inline passes."""

    def test_executor_is_visible_on_copied_and_imported_hosts_without_new_state(self):
        """Use the production observation/inline path and preserve parameter ownership."""
        for imported in (False, True):
            with self.subTest(imported=imported):
                _OBSERVED_CALLS.clear()
                model = nn.Sequential(_WeightedLayer())
                plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
                    is_boundary=True, inner_wrapper=_executor_wrapper, inner_target="self", region_dispatch=False,
                )}, mesh_dim_names=("cp", "tp"), parallel_modules={})
                context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                          tp_size=1, cp_size=2, ep_size=1,
                                          sequence_parallel=False, loss_parallel=False)
                session = CodegenBuildSession()
                session.record_source({"model_type": "test"})
                session.observe_plan(plan, model, context)
                payload = session.finish()._emission_view()["plan"]
                json.dumps(payload)
                rules = collect_source_observations(SimpleNamespace(injections=payload["injections"]))
                patches = build_strategy_patches(rules, boundary_classes={"0": "_WeightedLayer"})
                source = ("class _WeightedLayer:\n    def __init__(self):\n        self.bias = 7\n"
                          "    def forward(self, value):\n        return value * 2\n")
                if imported:
                    patches.imports.append(ImportPatch(__name__, ("_WeightedLayer",)))
                    _materialize_imported_forward_hosts("", patches)
                    source = ""
                rendered = apply_patch_set(source, patches)
                self.assertNotIn("executor(", rendered)
                self.assertNotIn("self.module", rendered)
                self.assertIn("def _project", rendered)
                self.assertNotIn("_forward_impl", rendered)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "visible_executor.py"
                    path.write_text(rendered, encoding="utf-8")
                    spec = util.spec_from_file_location("visible_executor", path)
                    module = util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    layer = module._WeightedLayer()
                    parameter = model[0].weight
                    layer.weight = parameter
                    for rank in (0, 1):
                        mesh = SimpleNamespace(get_local_rank=lambda value=rank: value)
                        layer._hp_boundary = SimpleNamespace(compiled=SimpleNamespace(mesh={"cp": mesh}))
                        self.assertEqual(layer.forward(5).item(), 22 + rank)
                    self.assertIs(layer.weight, parameter)
                self.assertEqual(_OBSERVED_CALLS, [id(model[0])])

    def test_optional_native_state_is_preserved_without_annotation_imports(self):
        """None state keeps its native field while None configuration remains literal."""
        model = nn.Sequential(_ObservedLayer())
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_state_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp",), parallel_modules={})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                  tp_size=1, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.observe_plan(plan, model, context)
        result = session.finish()._emission_view()["plan"]["injections"][0]["observed_wrapper"]
        self.assertIn("self.published_context", result["body"])
        self.assertIn("self.published_tp", result["body"])
        self.assertNotIn("published_selection", result["body"])
        self.assertIn("None", result["body"])
        self.assertFalse(any(name == "Any" for _, name in result["imports"]))

    def test_native_class_owned_boundary_does_not_run_external_wrapper(self):
        """Use native ownership decisions instead of injecting CP a second time."""
        _OBSERVED_CALLS.clear()
        model = nn.Sequential(_ObservedLayer())
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_selected_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp", "tp"), parallel_modules={"0": object()})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                  tp_size=1, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.observe_plan(plan, model, context)
        result = session.finish()._emission_view()["plan"]["injections"][0]["observed_wrapper"]
        self.assertEqual(_OBSERVED_CALLS, [])
        self.assertIsNone(result["body"])


    def test_already_installed_native_forward_is_observed_without_factory_replay(self):
        """Native preparation may already own the method before Planner observation."""
        model = nn.Sequential(_ObservedLayer())
        _commit_forward_rewrite(_selected_wrapper(model[0], None, None, None, None))
        _OBSERVED_CALLS.clear()
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_selected_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp", "tp"), parallel_modules={"0": object()})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                  tp_size=1, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.observe_plan(plan, model, context)
        result = session.finish()._emission_view()["plan"]["injections"][0]["observed_wrapper"]
        self.assertEqual(_OBSERVED_CALLS, [])
        self.assertIn("+ 7", result["body"])

    def test_native_child_target_does_not_rewrite_the_parent_boundary(self):
        """Source placement follows the resolved target rather than the boundary class."""
        _OBSERVED_CALLS.clear()
        parent = nn.Module()
        parent.layer = _ObservedLayer()
        model = nn.Sequential(parent)
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_selected_wrapper, inner_target="layer", region_dispatch=False,
        )}, mesh_dim_names=("cp", "tp"), parallel_modules={})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                  tp_size=1, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.observe_plan(plan, model, context)
        payload = session.finish()._emission_view()["plan"]
        rules = collect_source_observations(SimpleNamespace(injections=payload["injections"]))
        patches = build_strategy_patches(rules, boundary_classes={"0": "Parent"})
        self.assertEqual([item.class_name for item in patches.forward_extracts], ["_ObservedLayer"])
        self.assertEqual(_OBSERVED_CALLS, [id(parent.layer)])

    def test_observed_helper_binds_arguments_on_imported_host(self):
        """A real native helper must survive the imported host's variadic signature."""
        _OBSERVED_CALLS.clear()
        model = nn.Sequential(_ObservedLayer())
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_helper_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp", "tp"), parallel_modules={})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                  tp_size=1, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        with patch.object(_ObservedLayer.forward, "__module__", "unrelated.original_model"):
            session.observe_plan(plan, model, context)
        payload = session.finish()._emission_view()["plan"]
        rules = collect_source_observations(SimpleNamespace(injections=payload["injections"]))
        patches = build_strategy_patches(rules, boundary_classes={"0": "_ObservedLayer"})
        patches.imports.append(ImportPatch(__name__, ("_ObservedLayer",)))
        _materialize_imported_forward_hosts("", patches)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "imported.py"
            path.write_text(apply_patch_set("", patches), encoding="utf-8")
            module_spec = util.spec_from_file_location("imported", path)
            module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            host = module._ObservedLayer()
            host.cp_enable = True
            self.assertEqual(host.forward(5, extra=2), 22)
        self.assertEqual(_OBSERVED_CALLS, [id(model[0])])

    def test_observed_mesh_stays_a_runtime_expression(self):
        """Observation rank zero must not be frozen into the generated body."""
        _OBSERVED_CALLS.clear()
        model = nn.Sequential(_ObservedLayer())
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_mesh_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp", "tp"), parallel_modules={})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                  tp_size=1, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.observe_plan(plan, model, context)
        source = session.finish()._emission_view()["plan"]["injections"][0]["observed_wrapper"]
        self.assertIn("get_local_rank()", source["body"])
        self.assertIn("self._hp_boundary.compiled.mesh", source["body"])
        self.assertEqual(_OBSERVED_CALLS, [id(model[0])])
        self.assertEqual(source["imports"], [("hyper_parallel.distributed._builder.applier", "_get_cp_submesh")])

    def test_native_factory_failure_does_not_seal_or_retain_model(self):
        """A native constructor error is not turned into successful source facts."""
        _OBSERVED_CALLS.clear()
        model = nn.Sequential(_ObservedLayer())
        model[0].bias = 0
        reference = weakref.ref(model[0])
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_selected_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp", "tp"), parallel_modules={})
        context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)))
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        with self.assertRaisesRegex(ValueError, "native wrapper failure"):
            session.observe_plan(plan, model, context)
        self.assertEqual(len(_OBSERVED_CALLS), 1)
        with self.assertRaisesRegex(RuntimeError, "still collecting"):
            session.finish()
        del model
        self.assertIsNone(reference())

    def test_selected_forward_reaches_existing_emitter_once_without_live_references(self):
        """Emit the selected native branch and release all observation model references."""
        for bias, inplace in ((7, False), (-3, False), (-3, True), (99, False)):
            with self.subTest(bias=bias, inplace=inplace):
                _OBSERVED_CALLS.clear()
                model = nn.Sequential(_ObservedLayer())
                target = model[0]
                target.bias = bias
                target.inplace = inplace
                original = target.forward.__func__
                reference = weakref.ref(target)
                plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
                    is_boundary=True, inner_wrapper=_selected_wrapper, inner_target="self", region_dispatch=False,
                )}, mesh_dim_names=("cp", "tp"), parallel_modules={})
                context = SimpleNamespace(device_mesh=_ObservationMesh(("cp", "tp"), (2, 1)),
                                          tp_size=1, cp_size=2, ep_size=1,
                                          sequence_parallel=False, loss_parallel=False)
                session = CodegenBuildSession()
                session.record_source({"model_type": "test"})
                session.observe_plan(plan, model, context)
                snapshot = session.finish()
                payload = snapshot._emission_view()
                json.dumps(payload)
                self.assertEqual(_OBSERVED_CALLS, [id(target)])
                if not inplace:
                    self.assertIs(target.forward.__func__, original)
                injections = payload["plan"]["injections"]
                meta = SimpleNamespace(injections=injections)
                rules = collect_source_observations(meta)
                patches = build_strategy_patches(rules, boundary_classes={"0": "_ObservedLayer"})
                source = "class _ObservedLayer:\n    def forward(self, value):\n        return value * 2\n"
                generated = apply_patch_set(source, patches)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "observed.py"
                    path.write_text(generated, encoding="utf-8")
                    module_spec = util.spec_from_file_location("observed", path)
                    module = util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    result = module._ObservedLayer().forward(5)
                expected = 10 if bias == 99 else (10 + bias if bias > 0 else 10 - bias)
                self.assertEqual(result, expected)
                del target, model
                gc.collect()
                self.assertIsNone(reference())


class TestModelBuildOverlay(unittest.TestCase):
    """Exercise real module identities, meta construction and snapshot contents."""

    def setUp(self):
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()

    def test_defaults_and_legacy_rules_share_one_plan_and_snapshot_order(self):
        """Default and legacy requests share compilation and preserve observed installation order."""
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        request = ModuleReplacementRequest(("0",), LinearFamily)
        legacy = ModuleReplacementSpec(("1",), LinearFamily)
        spec = ModelAdapterSpec("Sequential", "toy", replacements=lambda: (request,))
        setup = SimpleNamespace(module_replacements=(legacy,), mesh_context=None)
        attach_build_context(model, build_module_context(setup, model_spec=spec))
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})

        with patch.object(module_build, "compile_module_replacements",
                          wraps=compile_module_replacements) as compile_plan:
            result = _observe_replacements(model, setup, None, False, session)

        self.assertIs(result, model)
        self.assertEqual(compile_plan.call_count, 1)
        self.assertIs(type(model[0]), LocalLinear)
        self.assertIs(type(model[1]), LocalLinear)
        session.mark_not_required("snapshot_test")
        replacements = session.finish().to_dict()["replacements"]
        declarations, applications = replacements["declarations"], replacements["applications"]
        self.assertEqual([item["rule_id"] for item in declarations],
                         [item["rule_id"] for item in applications])
        self.assertEqual(declarations[0]["replacement_target"],
                         f"{LinearFamily.__module__}:{LinearFamily.__qualname__}")
        self.assertEqual([item["primary_fqn"] for item in applications], ["0", "1"])

    def test_factory_build_uses_shared_meta_preparation_without_replacement(self):
        """Direct factory builds finalize meta members once without synthetic replacement records."""
        setup = DistributedSetup(mesh_context=MeshContext(dp_size=2, dp_shard_size=2, ep_size=2))
        session = CodegenBuildSession()
        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch.object(module_build, "_finalize_family_modules",
                             wraps=module_build._finalize_family_modules) as finalize, \
                use_codegen_model_build_overlay(session) as result:
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=create_family_model, distributed_setup=setup,
                torch_dtype=torch.float32,
            )
        self.assertEqual(finalize.call_count, 1)
        self.assertTrue(model._hp_components_prepared)
        self.assertIs(type(model.projection), ParallelLinear)
        self.assertEqual(model.projection.weight.device.type, "meta")
        self.assertEqual(result.snapshot.to_dict()["replacements"]["applications"], [])

    def test_failed_atomic_batch_records_no_installations(self):
        """A failed candidate batch preserves the original model and publishes no installations."""
        @module_replacement
        def fail(*, module, module_fqn, context):
            raise ValueError("candidate failed")

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        original = tuple(model)
        plan = compile_module_replacements(model, defaults=(
            ModuleReplacementRequest(("0",), LinearFamily), ModuleReplacementRequest(("1",), fail),
        ))
        with _observe_replacement_installations(plan.specs) as applications:
            with self.assertRaisesRegex(ValueError, "candidate failed"):
                apply_module_replacements(model, plan)
        self.assertEqual(applications, [])
        self.assertEqual(tuple(model), original)

    def test_observation_ties_weights_before_freezing_and_planning(self):
        """The observation planner sees native parameter aliases and their frozen state."""
        def _check_plan(model, *args):
            self.assertIs(model.lm_head.weight, model.embedding.weight)
            self.assertFalse(model.lm_head.weight.requires_grad)
            _observe_plan(model, *args)

        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch("hyper_parallel.codegen.model_build_overlay._observe_plan", side_effect=_check_plan) as plan, \
                use_codegen_model_build_overlay(CodegenBuildSession()):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=_TiedModel, distributed_setup=DistributedSetup(),
                torch_dtype=torch.float32, freeze_config=("lm_head",),
            )

        plan.assert_called_once()
        self.assertIs(model.lm_head.weight, model.embedding.weight)
        self.assertEqual(model.lm_head.weight.device.type, "meta")

    def test_observation_does_not_keep_replaced_sources_alive(self):
        """Codegen observation releases source modules after the shared installer replaces them."""
        references = []

        @module_replacement
        def replace(*, module, module_fqn, context):
            self.assertTrue(all(reference() is None for reference in references))
            references.append(weakref.ref(module))
            return LocalLinear(module=module, module_fqn=module_fqn, context=context)

        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        plan = compile_module_replacements(model, (ModuleReplacementSpec(("*",), replace),))
        with _observe_replacement_installations(plan.specs) as applications:
            apply_module_replacements(model, plan)
        self.assertEqual(len(applications), 2)
        self.assertTrue(all(reference() is None for reference in references))

    def test_wildcard_user_request_has_distinct_effective_declarations(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(4, 3))
        request = ModuleReplacementRequest(("*",), LinearFamily, declaration_id="user-rule")
        setup = SimpleNamespace(module_replacement_requests=(request,), mesh_context=None)
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        _observe_replacements(model, setup, None, False, session)
        session.mark_not_required("snapshot_test")
        records = replacement_records_from_snapshot(session.finish())
        self.assertEqual([record["fqn"] for record in records], ["0", "1"])

    def test_size_one_owned_layout_is_planned_during_observation(self):
        """A module-owned layout is planned even when every parallel axis has size one."""
        model = nn.Module()
        model._hp_model_adapter_spec = ModelAdapterSpec("Module", "toy", sharding=lambda **_: {})
        mesh = MeshContext()
        mesh.device_mesh = LogicalDeviceMesh(mesh_dim_names=("dp",), mesh_shape=(1,))
        planner = SimpleNamespace(plan=lambda *_args, **_kwargs: ShardingPlan())
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        session.record_replacements()
        with patch.object(planner, "plan", wraps=planner.plan) as plan:
            _observe_plan(model, mesh, planner, True, session)
        self.assertEqual(plan.call_count, 1)
        self.assertEqual(session.finish().to_dict()["status"], "derived")

    def test_native_and_observation_share_method_policy_and_prepared_context(self):
        """Native and offline builds reuse the prepared context for identical method replacement policy."""
        events = []

        @method_replacement
        def replace_stage(*, module, method_fqn, original_method, context):
            self.assertIsInstance(module, _MethodModel)
            self.assertEqual(method_fqn, "0.stage")
            self.assertEqual(context["dtype"], "float32")
            events.append(context)

            def scaled(inputs):
                return original_method(inputs) * 2

            return scaled

        setup = DistributedSetup(method_replacements=(MethodReplacementSpec("0.stage", replace_stage),))
        def factory(config, *, context):
            parent = nn.Sequential(_MethodModel(config, context=context))
            parent.config = config
            return parent

        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                patch.object(model_builder, "build_module_context",
                             side_effect=AssertionError("prepared context must be reused")):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=factory, distributed_setup=setup, torch_dtype=torch.float32,
            )
            torch.testing.assert_close(model(torch.tensor([3.])), torch.tensor([6.]))
            with use_codegen_model_build_overlay(CodegenBuildSession()):
                observed = HyperAutoModelForCausalLM.from_config(
                    FamilyModelConfig(), model_factory=factory, distributed_setup=setup, torch_dtype=torch.float32,
                )
        self.assertEqual(len(events), 2)
        self.assertIs(events[0]["model_spec"], model._hp_model_adapter_spec)
        self.assertIs(events[1]["model_spec"], observed._hp_model_adapter_spec)
        self.assertEqual(observed[0].weight.device.type, "meta")

    def test_replacement_only_observation_uses_empty_native_plan_projection(self):
        """Replacement-only builds publish an empty layout plan together with installation metadata."""
        model = nn.Sequential(nn.Linear(4, 3))
        setup = SimpleNamespace(module_replacement_requests=(ModuleReplacementRequest(("0",), LinearFamily),),
                                mesh_context=None)
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        model = _observe_replacements(model, setup, None, False, session)
        mesh = MeshContext()
        mesh.device_mesh = LogicalDeviceMesh(mesh_dim_names=("dp",), mesh_shape=(1,))
        planner = SimpleNamespace(plan=Mock(side_effect=AssertionError("pure DP needs no layout plan")))
        _observe_plan(model, mesh, planner, True, session)
        result = session.finish().to_dict()
        self.assertEqual(result["status"], "derived")
        self.assertEqual(result["plan"]["param_plan"], {})
        self.assertEqual(result["plan"]["injections"], [])
        self.assertEqual(result["plan"]["parallel_dims"]["dp_size"], 1)
        self.assertEqual(len(result["replacements"]["applications"]), 1)

    def test_source_compat_uses_the_effective_per_build_spec(self):
        """An explicit model spec supplies its import policy without a second registry lookup."""
        prepare = Mock()
        provider = Mock(return_value=SimpleNamespace(prepare_model_import=prepare))
        spec = ModelAdapterSpec("FamilyModel", "test_family_model", codegen=provider)
        with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                use_codegen_model_build_overlay(CodegenBuildSession()):
            model = HyperAutoModelForCausalLM.from_config(
                FamilyModelConfig(), model_factory=create_family_model, model_spec=spec, torch_dtype=torch.float32,
            )
        provider.assert_called_once_with()
        prepare.assert_called_once_with()
        self.assertIs(model._hp_model_adapter_spec, spec)

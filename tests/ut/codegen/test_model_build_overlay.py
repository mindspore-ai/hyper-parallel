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

import ast
import builtins
import unittest
import weakref
from functools import wraps
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import DeepseekV3Config

from hyper_parallel.codegen.emit.modeling import verify_recorded_patch_bundle
from hyper_parallel.codegen.emit.patches import emission_patches_from_meta
from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.codegen.emit.snapshot_patches import emission_patches_from_snapshot_payload
from hyper_parallel.codegen.inline.component_source import render_component_source
from hyper_parallel.codegen.inline.ir import ImportPatch
from hyper_parallel.codegen.inline.observations import collect_source_observations
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import _materialize_imported_forward_hosts
from hyper_parallel.codegen.inline.recognition import _function_source_qualname, _source_qualname
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.codegen.model_build_overlay import (
    _observe_plan,
    _observe_replacement_installations,
    _observe_replacements,
    use_codegen_model_build_overlay,
)
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.components.modules import family as registry
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, inner_wrapper
from hyper_parallel.models import HyperAutoModelForCausalLM, module_build
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.kimi_k26.modeling_kimi_k26 import KimiLanguageModel, bind_context_parallel
from hyper_parallel.models.module_build import attach_build_context, build_module_context
from hyper_parallel.models.replacement import (
    MethodReplacementSpec,
    ModuleReplacementRequest,
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
    method_replacement,
    module_replacement,
)
from tests.common.module_family_fixtures import (
    FamilyModelConfig,
    LinearFamily,
    LocalLinear,
    ParallelLinear,
    create_family_model,
    register_linear_family,
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


class _ObservedLayer(nn.Module):
    """Small source host whose final patch does not require a distributed runtime."""

    def forward(self, value):
        """Provide an ordinary source method for the patch to preserve."""
        return value * 2


class _ObservedTextTower(KimiLanguageModel):
    """Exercise native input preparation while observing the decoder call."""

    def __init__(self) -> None:
        """Initialize a tiny real decoder while observing only its input preparation."""
        config = DeepseekV3Config.from_dict({
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "moe_intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
            "q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
            "v_head_dim": 4, "n_routed_experts": 4, "num_experts_per_tok": 2, "first_k_dense_replace": 0,
            "n_group": 1, "topk_group": 1,
        })
        super().__init__(config)

    def _run_language(self, **kwargs: Any) -> dict[str, Any]:
        """Return the native class's prepared decoder inputs."""
        return kwargs


def _bind_constant_cp_rank(model):
    """An unrelated scalar with a CP-like name must not acquire a mesh binding."""
    cp_rank = 7
    original_forward = model.forward

    def forward(value: Any) -> Any:
        """Expose an unrelated scalar to the closure observer."""
        return original_forward(value) + cp_rank

    model.forward = forward


@inner_wrapper
def _runtime_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Native runtime preparation must never execute during offline plan projection."""
    raise AssertionError("offline codegen executed the runtime wrapper")


def _bind_factory_tail(model):
    """Represent an already-installed native patch without reading mesh state."""
    original_forward = model.forward

    @wraps(original_forward)
    def tail(*args, **kwargs):
        if args:
            raise TypeError("factory tail accepts only keyword arguments")
        return original_forward(**kwargs) + 1

    model.forward = tail


def _bind_published_mesh(model, cp_mesh):
    """Represent a native patch whose resource binding belongs to Trainer."""
    original_forward = model.forward

    @wraps(original_forward)
    def forward(value):
        return original_forward(value) + cp_mesh.get_local_rank()

    model.native_cp_mesh = cp_mesh
    model.forward = forward


def _legacy_code_getattr(value, name, *default):
    """Exercise the pre-3.11 code-object interface without patching global builtins."""
    if name == "co_qualname":
        if default:
            return default[0]
        raise AttributeError(name)
    return builtins.getattr(value, name, *default)


class TestOfflinePlanObservation(unittest.TestCase):
    """Offline review consumes declarations rather than constructing runtime executors."""

    def test_decorated_forward_uses_lexical_owner_without_code_qualname(self):
        """Source position distinguishes decorated closures sharing the same public name."""
        for binder, nested in ((_bind_factory_tail, "tail"), (_bind_published_mesh, "forward")):
            with self.subTest(binder=binder.__name__):
                model = _ObservedLayer()
                if binder is _bind_published_mesh:
                    binder(model, Mock())
                else:
                    binder(model)
                self.assertEqual(model.forward.__qualname__, _ObservedLayer.forward.__qualname__)
                with patch("hyper_parallel.codegen.inline.recognition.getattr", _legacy_code_getattr, create=True):
                    self.assertEqual(_function_source_qualname(model.forward),
                                     f"{binder.__qualname__}.<locals>.{nested}")

    def test_source_qualname_keeps_nested_class_and_method_scopes(self):
        """Restored names match Python's lexical class/function scope convention."""
        source = ("class Outer:\n    def build(self):\n        class Inner:\n"
                  "            async def forward(self):\n                pass\n")
        code = SimpleNamespace(co_name="forward", co_firstlineno=4)
        self.assertEqual(_source_qualname(ast.parse(source), code), "Outer.build.<locals>.Inner.forward")
        code.co_firstlineno = 20
        self.assertIsNone(_source_qualname(ast.parse(source), code))

    def test_kimi_owned_cp_source_without_code_qualname(self):
        """The legacy interpreter path keeps rank independence and the complete native input contract."""
        with patch("hyper_parallel.codegen.inline.recognition.getattr", _legacy_code_getattr, create=True):
            self.test_kimi_owned_cp_source_preserves_inputs()

    def test_plan_projection_never_executes_wrapper_or_queries_distributed_state(self):
        """An active CP declaration remains visible with no runtime mesh API available."""
        model = nn.Sequential(_ObservedLayer())
        plan = SimpleNamespace(modules={"0": ModuleShardingSpec(
            is_boundary=True, inner_wrapper=_runtime_wrapper, inner_target="self", region_dispatch=False,
        )}, mesh_dim_names=("cp",), parallel_modules={})
        logical = LogicalDeviceMesh(("dp", "cp", "tp"), (2, 2, 2))
        self.assertFalse(hasattr(logical, "get_group"))
        self.assertFalse(hasattr(logical, "get_local_rank"))
        context = SimpleNamespace(device_mesh=logical, dp_size=2, tp_size=2, cp_size=2, ep_size=1,
                                  sequence_parallel=False, loss_parallel=False)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        with patch.object(torch.distributed, "get_rank", side_effect=AssertionError("runtime rank lookup")), \
                patch.object(torch.distributed, "get_world_size", side_effect=AssertionError("runtime WORLD lookup")), \
                patch.object(torch.distributed, "new_group", side_effect=AssertionError("runtime group creation")):
            session.observe_plan(plan, model, context)
        payload = session.finish()._emission_view()["plan"]
        self.assertEqual(len(payload["injections"]), 1)
        self.assertNotIn("observed_wrapper", payload["injections"][0])
        self.assertIn("_runtime_wrapper", str(payload["injections"][0]["inner_wrapper"]))
        self.assertEqual(payload["parallel_dims"]["cp_size"], 2)

    def test_published_mesh_is_kept_as_source_without_reading_its_runtime_state(self):
        """The review source retains the native field; Codegen never resolves its rank."""
        model = _ObservedLayer()
        originals = {"": model.forward}
        runtime_mesh = Mock()
        runtime_mesh.get_local_rank.side_effect = AssertionError("Codegen queried a runtime rank")
        runtime_mesh.get_group.side_effect = AssertionError("Codegen queried a communication group")
        _bind_published_mesh(model, runtime_mesh)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.mark_not_required("no_layout_patch")
        session.observe_final_forwards(model, originals)
        observed = session.finish()._emission_view()["plan"]["injections"][0]["observed_wrapper"]
        self.assertIn("self.native_cp_mesh.get_local_rank()", observed["definition"])
        runtime_mesh.get_local_rank.assert_not_called()
        runtime_mesh.get_group.assert_not_called()

    def test_kimi_owned_cp_source_preserves_inputs(self):
        """The owned Kimi input protocol emits no forward patch and keeps runtime rank binding."""
        rendered_sources = []
        values = torch.arange(32).reshape(1, 8, 4)
        padding = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        for rank in (0, 1):
            with self.subTest(cp_rank=rank):
                model = nn.Module()
                model.config = SimpleNamespace(model_type="kimi_k25")
                model.model = nn.Module()
                model.model.language_model = _ObservedTextTower()
                target = model.model.language_model
                original = target.forward
                bind_context_parallel(model, SimpleNamespace(cp_size=2, cp_rank=rank))
                self.assertIs(target.forward.__func__, original.__func__)
                self.assertEqual(_function_source_qualname(target.forward), "KimiLanguageModel.forward")
                session = CodegenBuildSession()
                session.record_source({"model_type": "kimi_k25"})
                session.mark_not_required("no_parameter_layout")
                with patch.object(torch.distributed, "get_rank", side_effect=AssertionError("runtime rank query")), \
                        patch.object(torch.distributed, "new_group", side_effect=AssertionError("runtime group")):
                    session.observe_final_forwards(model, {"model.language_model": original})
                    expanded = render_component_source(type(target))
                payload = session.finish()._emission_view()["plan"]
                self.assertIsNone(payload)
                rendered = "\n".join(expanded.imports) + "\n" + expanded.source
                rendered_sources.append(rendered)
                self.assertNotIn("cp_rank = 0", rendered)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "kimi_cp.py"
                    path.write_text(rendered, encoding="utf-8")
                    spec = util.spec_from_file_location("kimi_cp_review", path)
                    module = util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    generated = module._ObservedTextTower()
                    generated.bind_context_parallel(2, rank)
                    inputs = {"inputs_embeds": values, "attention_mask": padding, "use_cache": False,
                              "extra_option": "preserved"}
                    expected, actual = target(**inputs), generated.forward(**inputs)
                    self.assertEqual(set(actual), set(expected))
                    for name in expected:
                        if isinstance(expected[name], torch.Tensor):
                            torch.testing.assert_close(actual[name], expected[name])
                        else:
                            self.assertEqual(actual[name], expected[name])
                    self.assertEqual(tuple(actual["inputs_embeds"].shape), (1, 4, 4))
                    torch.testing.assert_close(actual["position_ids"], torch.arange(rank * 4, (rank + 1) * 4)[None])
                    self.assertFalse(inputs["use_cache"])
                    with self.assertRaisesRegex(TypeError, "by keyword"):
                        generated.forward(values)
                    with self.assertRaisesRegex(ValueError, "KV cache"):
                        generated.forward(**{**inputs, "use_cache": True})
        self.assertEqual(rendered_sources[0], rendered_sources[1])

    def test_cp_like_scalar_without_native_context_provenance_remains_an_error(self):
        """A scalar name alone cannot authorize replacing a captured value with a runtime rank."""
        model = _ObservedLayer()
        originals = {"": model.forward}
        _bind_constant_cp_rank(model)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.mark_not_required("no_layout_patch")
        with self.assertRaisesRegex(RuntimeError, "unresolved captured state.*cp_rank"):
            session.observe_final_forwards(model, originals)


class TestModelBuildOverlay(unittest.TestCase):
    """Exercise real module identities, meta construction and snapshot contents."""

    def setUp(self):
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()

    def test_build_entries_preserve_backend_without_reading_device_ordinal(self):
        """Both native entry points must reach meta observation without initializing an accelerator."""
        for backend in ("cpu", "cuda"):
            for entry in ("from_config", "from_pretrained"):
                with self.subTest(backend=backend, entry=entry):
                    config = FamilyModelConfig()
                    setup = DistributedSetup(mesh_context=MeshContext())
                    publisher = "hyper_parallel.codegen.model_build_overlay.ensure_codegen_artifact_from_snapshot"
                    with patch.object(auto_model, "get_device_type", return_value=backend), \
                            patch.object(auto_model, "get_device_id", side_effect=AssertionError("device ordinal")), \
                            patch.object(auto_model, "_load_pretrained_config", return_value=config), \
                            patch(publisher), use_codegen_model_build_overlay(CodegenBuildSession()):
                        build = getattr(HyperAutoModelForCausalLM, entry)
                        model = build(
                            config if entry == "from_config" else "config-only-model",
                            model_factory=_MethodModel, torch_dtype=torch.float32, distributed_setup=setup,
                        )
                        self.assertEqual(model._hp_module_build_context["device_type"], backend)
                        self.assertTrue(all(value.device.type == "meta" for value in model.parameters()))

    def test_device_query_is_restored_after_observation_and_failures(self):
        """Offline construction must not replace native device resolution after its scope ends."""
        original_build = auto_model._BaseHyperAutoModelClass.__dict__["_build_model"]
        for failure in (None, "construction", "publication"):
            with self.subTest(failure=failure):
                native_query = Mock(return_value=torch.device("cuda", 3))
                session = CodegenBuildSession()
                publisher = "hyper_parallel.codegen.model_build_overlay.ensure_codegen_artifact_from_snapshot"
                with patch.object(auto_model, "_current_device", native_query), \
                        patch.object(auto_model, "get_device_type", return_value="cuda"), \
                        patch(publisher, side_effect=RuntimeError("publication") if failure == "publication" else None):
                    try:
                        with use_codegen_model_build_overlay(session) as result:
                            self.assertEqual(auto_model._current_device(), torch.device("cuda"))
                            native_query.assert_not_called()
                            if failure == "construction":
                                raise RuntimeError("construction")
                            if failure == "publication":
                                # Exercise the exception raised by the actual publication callback.
                                HyperAutoModelForCausalLM.from_config(
                                    FamilyModelConfig(), model_factory=_MethodModel,
                                    torch_dtype=torch.float32, distributed_setup=DistributedSetup(),
                                )
                    except RuntimeError as exc:
                        self.assertEqual(str(exc), failure)
                    else:
                        if failure == "publication":
                            self.assertEqual(result.snapshot.failure.reason, "publication")
                        else:
                            self.assertIsNone(failure)
                    self.assertIsNone(result.model)
                    self.assertIs(auto_model._current_device, native_query)
                    self.assertEqual(auto_model._current_device(), torch.device("cuda", 3))
                    self.assertIs(auto_model._BaseHyperAutoModelClass.__dict__["_build_model"], original_build)

    def test_outer_factory_publishes_only_after_success_and_restores_overlay(self):
        """Factory-tail facts are included; an outer failure cannot publish an early bundle."""
        original_build = auto_model._BaseHyperAutoModelClass.__dict__["_build_model"]
        for fail in (False, True):
            with self.subTest(fail=fail):
                session = CodegenBuildSession()
                mesh = MeshContext()
                mesh.device_mesh = LogicalDeviceMesh(("dp", "cp", "tp"), (1, 1, 1))
                publisher = "hyper_parallel.codegen.model_build_overlay.ensure_codegen_artifact_from_snapshot"
                with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                        patch(publisher) as emit:
                    try:
                        with use_codegen_model_build_overlay(session, defer_publish=True) as result:
                            result.model = HyperAutoModelForCausalLM.from_config(
                                FamilyModelConfig(), model_factory=_MethodModel, torch_dtype=torch.float32,
                                distributed_setup=DistributedSetup(mesh_context=mesh),
                            )
                            self.assertIsNone(result.snapshot)
                            emit.assert_not_called()
                            _bind_factory_tail(result.model)
                            if fail:
                                raise ValueError("outer factory failure")
                    except ValueError as exc:
                        self.assertTrue(fail)
                        self.assertEqual(str(exc), "outer factory failure")
                    self.assertIsNone(result.model)
                    self.assertIs(auto_model._BaseHyperAutoModelClass.__dict__["_build_model"], original_build)
                    if fail:
                        emit.assert_not_called()
                        self.assertIsNone(result.snapshot)
                    else:
                        emit.assert_called_once()
                        self.assertIs(session.finish(), result.snapshot)
                        injection = result.snapshot._emission_view()["plan"]["injections"][0]
                        self.assertIn("+ 1", injection["observed_wrapper"]["body"])

    def test_final_factory_keeps_review_source_without_runtime_mesh(self):
        """Render an already-installed patch without rank, group or WORLD lookups."""
        model = _ObservedLayer()
        originals = {"": model.forward}
        _bind_factory_tail(model)
        session = CodegenBuildSession()
        session.record_source({"model_type": "test"})
        session.mark_not_required("pure_data_parallel")
        session.observe_final_forwards(model, originals)
        with self.assertRaisesRegex(RuntimeError, "already been observed"):
            session.observe_final_forwards(model, originals)
        snapshot = session.finish()._emission_view()
        payload = snapshot["plan"]
        rules = collect_source_observations(SimpleNamespace(injections=payload["injections"]))
        emission = emission_patches_from_meta(SimpleNamespace(
            emission_patches=emission_patches_from_snapshot_payload(snapshot),
        ))
        for imported in (False, True):
            with self.subTest(imported=imported):
                patches = build_strategy_patches(rules)
                source = "class _ObservedLayer:\n    def forward(self, value):\n        return value * 2\n"
                if imported:
                    patches.imports.append(ImportPatch(__name__, ("_ObservedLayer",)))
                    _materialize_imported_forward_hosts("", patches)
                    source = ""
                rendered = apply_patch_set(source, patches)
                verify_recorded_patch_bundle({"model.py": rendered}, patch_plan=emission)
                self.assertNotIn("get_rank", rendered)
                self.assertNotIn("get_group", rendered)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "factory_tail.py"
                    path.write_text(rendered, encoding="utf-8")
                    spec = util.spec_from_file_location("factory_tail", path)
                    module = util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    with patch.object(torch.distributed, "get_rank", side_effect=AssertionError("runtime rank lookup")):
                        self.assertEqual(module._ObservedLayer().forward(value=5), 11)
                        with self.assertRaisesRegex(TypeError, "only keyword"):
                            module._ObservedLayer().forward(5)


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

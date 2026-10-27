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
"""Regression coverage for structure-driven inline source generation."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.codegen.emit.modeling import (
    _apply_inline_modeling,
    _verify_recorded_patch_materialization,
)
from hyper_parallel.codegen.emit.recorded import materialize_recorded_forwards
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen import manager
from hyper_parallel.codegen.inline.ir import (
    ConstructorReplacePatch, ForwardExtractPatch, InlinePatchSet, InlineRule,
    ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.framework_spec import interface_component_class
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling
from hyper_parallel.codegen.inline.recognition import inner_wrapper_target
from hyper_parallel.codegen.inline.spec_bundle import StrategySpec
from hyper_parallel.codegen.inline.specs import replacement_spec, strategy_spec
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models import registry
from tests.common.mark_utils import arg_mark


QWEN_ROOT = "hyper_parallel.models.qwen3_moe.adapter."
ATTENTION = QWEN_ROOT + "conversion.module_replacement.replace_qwen3_moe_flash_attention"
ATTENTION_TYPE = "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention"
EP = QWEN_ROOT + "distributed.expert_parallel.qwen3moe_ep_compute_fn"
CP = QWEN_ROOT + "distributed.context_parallel.qwen3_moe_flash_attention_cp_wrapper"
DEEPSEEK_CP = (
    "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel"
    ".deepseek_v32_dsa_attention_cp_wrapper"
)

TOY_REPLACEMENTS = "hyper_parallel.models.toy.adapter.replacements"
TOY_NORM = TOY_REPLACEMENTS + ".replace_toy_norm"
TOY_UNKNOWN = TOY_REPLACEMENTS + ".replace_toy_unknown"
TOY_NORM_TYPE = "example.toy.modeling.ToyNorm"

SOURCE = """
class OldProjection:
    def forward(self, value):
        return value + 1

class ToyBlock:
    def __init__(self):
        self.proj = OldProjection()

    def forward(self, value):
        return self.proj.forward(value)
"""

TOY_SOURCE = """
from torch import nn
import torch


class ToyNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, value):
        return value

class ToyModel:
    def __init__(self):
        self.norm = ToyNorm(4)
"""


def _toy_rms_norm_factory(*, module, module_fqn, context):
    """Constructs the generic ``RMSNorm`` -- the shape recognition reads."""
    from hyper_parallel.components.modules import RMSNorm  # pylint: disable=C0415

    return RMSNorm(module=module, module_fqn=module_fqn, context=context)


def _toy_grouped_experts_factory(*, module, module_fqn, context):
    """Constructs the generic ``GroupedExperts`` -- a different component."""
    from hyper_parallel.components.modules import GroupedExperts  # pylint: disable=C0415

    return GroupedExperts(module=module, module_fqn=module_fqn, context=context)


class _ToyUnknownWrapper:
    """A replacement this framework has no generic component for."""


def _toy_unknown_factory(*, module, module_fqn, context):
    """Constructs an unknown component -- recognition must not guess."""
    return _ToyUnknownWrapper(module=module, module_fqn=module_fqn, context=context)


def _toy_replacements_module(factory=None) -> types.ModuleType:
    """A stand-in ``models/toy/adapter/replacements.py`` provider module."""
    module = types.ModuleType(TOY_REPLACEMENTS)
    module.replace_toy_norm = factory or _toy_rms_norm_factory
    module.replace_toy_unknown = _toy_unknown_factory
    return module


def _toy_adapter(name="toy", factory=None) -> ModelAdapterSpec:
    """A family whose only codegen-relevant input is its runtime factories."""
    module = _toy_replacements_module(factory)
    return ModelAdapterSpec(
        f"{name.title()}ForCausalLM", name, replacements=lambda: module,
    )


def _meta(factory=ATTENTION, injections=(), module_type=ATTENTION_TYPE, fqns=("blocks.0.proj",)):
    return SimpleNamespace(
        module_overrides=[
            {"factory": factory, "module_type": module_type, "fqns": list(fqns)}
        ],
        injections=list(injections),
        param_plan={"blocks.0.proj": {"params": {"linear_qkv.weight": {"tp": "S(0)"}}}},
        frozen_sharded_params=["blocks.0.proj.linear_qkv.weight", "embedding.weight"],
        source={},
    )


def _execute_source(source):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "generated.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("inline_adapter_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return vars(module)


class TestInlineAdapters(unittest.TestCase):
    """Exercise structural resolution, coverage enforcement and model-independent emission."""

    def test_recorded_local_compute_requires_generated_forward(self):
        """A trainer-installed local compute cannot remain metadata-only."""
        record = SimpleNamespace(
            module_fqn="model.layers.0.mlp",
            module_type=type("ExternalMoe", (), {}),
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": "example.compute",
            }],
        )
        with self.assertRaisesRegex(RuntimeError, "model.layers.0.mlp"):
            _verify_recorded_patch_materialization(SOURCE, meta)

    def test_recorded_local_compute_accepts_explicit_generated_forward(self):
        """The audit accepts a source forward that calls the bound compute."""
        record = SimpleNamespace(
            module_fqn="model.layers.0.mlp",
            module_type=type("ExternalMoe", (), {}),
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": "example.compute",
            }],
        )
        source = """
class ExternalMoe:
    def forward(self, hidden_states):
        return self.compute(hidden_states)
"""
        _verify_recorded_patch_materialization(source, meta)

    def test_recorded_module_function_materializes_region_forward(self):
        """An instance-bound source function becomes impl plus EP boundary."""
        def source_forward(self, hidden_states):
            return hidden_states

        record = SimpleNamespace(
            module_fqn="model.layers.0.mlp",
            module_type=type("ExternalMoe", (), {}),
            factory=lambda: None,
            original_forward=source_forward,
        )
        record.original_forward.__qualname__ = "source_forward"
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            param_plan={
                record.module_fqn: {
                    "is_boundary": True,
                    "in_src": {"hidden_states": {}},
                    "in_dst": {"hidden_states": {}},
                    "out_src": {"output": {}},
                    "out_dst": {"output": {}},
                }
            },
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": "example.compute",
            }],
            mesh_dim_names=[],
        )
        source = """
from types import MethodType

def source_forward(self, hidden_states):
    return hidden_states

class Model:
    def __init__(self, layer):
        self.layer = layer
        self.layer.forward = MethodType(source_forward, self.layer)
"""
        rendered = materialize_recorded_forwards(source, meta)
        ast.parse(rendered)
        self.assertIn("def source_forward_hp_impl", rendered)
        self.assertIn("self.compute", rendered)
        self.assertIn("self.layer._forward_impl = MethodType", rendered)

    def test_recorded_imported_subtree_host_emits_explicit_fqn_bindings(self):
        """An imported host with no constructor call binds by recorded FQN."""
        class ExternalMoe:
            def forward(self, hidden_states):
                return hidden_states

        record = SimpleNamespace(
            module_fqn="model.language_model.layers.1.mlp",
            module_type=ExternalMoe,
            factory=lambda: None,
            original_forward=ExternalMoe.forward,
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            param_plan={
                record.module_fqn: {
                    "is_boundary": True,
                    "in_src": {"hidden_states": {}},
                    "in_dst": {"hidden_states": {}},
                    "out_src": {"output": {}},
                    "out_dst": {"output": {}},
                }
            },
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": {
                    "_target_": "example.compute",
                    "fix_router": False,
                    "use_grouped_gemm": True,
                    "overlap_shared_expert": True,
                    "capacity_factor": None,
                },
            }],
            mesh_dim_names=[],
        )
        source = """
@dataclass
class OuterModel:
    def __init__(self, language_model):
        self.language_model = language_model
"""

        rendered = materialize_recorded_forwards(source, meta)

        ast.parse(rendered)
        # The host import is emitted from the recorded class, never assumed.
        self.assertIn("import ExternalMoe", rendered)
        self.assertIn("class HPCodegenExternalMoe(ExternalMoe)", rendered)
        # The insertion must not steal the first class's decorator block.
        self.assertIn("@dataclass\nclass OuterModel", rendered)
        self.assertNotIn("@dataclass\nclass HPCodegenExternalMoe", rendered)
        # One explicit binding per recorded FQN — no type registry.
        self.assertIn("def hp_bind_recorded_forwards(model):", rendered)
        self.assertIn(
            "module = model.model.language_model.layers[1].mlp", rendered)
        self.assertIn(
            "module._forward_impl = MethodType("
            "HPCodegenExternalMoe._forward_impl, module)", rendered)
        self.assertIn(
            "module.forward = MethodType(HPCodegenExternalMoe.forward, module)",
            rendered)
        self.assertIn("from types import MethodType", rendered)
        self.assertNotIn("_HP_RECORDED_FORWARD_TYPES", rendered)
        # The four EP factory parameters keep their resolved values — the
        # builder's actual local-compute config, never a YAML re-read.
        for name, value in (
                ("fix_router", "False"), ("use_grouped_gemm", "True"),
                ("overlap_shared_expert", "True"), ("capacity_factor", "None")):
            self.assertIn(f"'{name}': {value}", rendered)

    def test_recorded_imported_subtree_multi_layer_bindings_are_explicit(self):
        """Each recorded MoE layer becomes its own explicit binding target."""
        class ExternalMoe:
            def forward(self, hidden_states):
                return hidden_states

        def factory():
            return None

        layer_indices = (0, 1, 2, 10)
        fqns = tuple(
            f"model.language_model.layers.{index}.mlp" for index in layer_indices
        )
        records = tuple(
            SimpleNamespace(
                module_fqn=fqn,
                module_type=ExternalMoe,
                factory=factory,
                original_forward=ExternalMoe.forward,
            )
            for fqn in fqns
        )
        entry = {
            "is_boundary": True,
            "in_src": {"hidden_states": {}},
            "in_dst": {"hidden_states": {}},
            "out_src": {"output": {}},
            "out_dst": {"output": {}},
        }
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=records),
            param_plan={fqn: entry for fqn in fqns},
            injections=[
                {"match": fqn, "local_compute_fn": {"_target_": "example.compute"}}
                for fqn in fqns
            ],
            mesh_dim_names=[],
            materialized_patches=[],
        )
        source = """
class OuterModel:
    def __init__(self, language_model):
        self.language_model = language_model
"""

        rendered = materialize_recorded_forwards(source, meta)

        ast.parse(rendered)
        # One subclass and one binder carry every recorded layer explicitly.
        self.assertEqual(
            rendered.count("class HPCodegenExternalMoe(ExternalMoe)"), 1)
        self.assertEqual(
            rendered.count("def hp_bind_recorded_forwards(model):"), 1)
        self.assertEqual(
            rendered.count(
                "module.forward = MethodType(HPCodegenExternalMoe.forward, module)"
            ),
            len(fqns),
        )
        # Recorded layer paths appear in numeric order, keeping the binder
        # reviewable (a lexical sort would render layers.10 before layers.2).
        positions = [
            rendered.index(f"module = model.model.language_model.layers[{index}].mlp")
            for index in layer_indices
        ]
        self.assertEqual(positions, sorted(positions))
        # The audit manifest lists every recorded binding target.
        _verify_recorded_patch_materialization(rendered, meta)
        self.assertEqual(
            sorted(item["fqn"] for item in meta.materialized_patches),
            sorted(fqns),
        )

    def test_recorded_binder_binds_matching_model_and_fails_on_missing_fqn(self):
        """The explicit binder resolves real paths and raises on absent FQNs."""
        host_module = types.ModuleType("hp_recorded_binder_host")
        sys.modules[host_module.__name__] = host_module
        self.addCleanup(sys.modules.pop, host_module.__name__, None)

        class ExternalMoe:
            def forward(self, hidden_states):
                return hidden_states

        ExternalMoe.__module__ = host_module.__name__
        host_module.ExternalMoe = ExternalMoe
        record = SimpleNamespace(
            module_fqn="model.language_model.layers.1.mlp",
            module_type=ExternalMoe,
            factory=lambda: None,
            original_forward=ExternalMoe.forward,
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            param_plan={
                record.module_fqn: {
                    "is_boundary": True,
                    "in_src": {"hidden_states": {}},
                    "in_dst": {"hidden_states": {}},
                    "out_src": {"output": {}},
                    "out_dst": {"output": {}},
                }
            },
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": {"_target_": "example.compute"},
            }],
            mesh_dim_names=[],
        )
        source = """
class OuterModel:
    def __init__(self, language_model):
        self.language_model = language_model
"""

        rendered = materialize_recorded_forwards(source, meta)
        namespace = _execute_source(rendered)
        binder = namespace["hp_bind_recorded_forwards"]

        # A model matching the recorded FQN gets both MethodType bindings.
        good_model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=[None, SimpleNamespace(mlp=SimpleNamespace())]
                )
            )
        )
        binder(good_model)
        mlp = good_model.model.language_model.layers[1].mlp
        self.assertIsInstance(mlp.forward, MethodType)
        self.assertIsInstance(mlp._forward_impl, MethodType)
        self.assertIs(
            mlp.forward.__func__, namespace["HPCodegenExternalMoe"].forward)
        self.assertIsNot(mlp.forward.__func__, ExternalMoe.forward)
        # A missing recorded path must fail loudly, never silently skip.
        bad_model = SimpleNamespace(model=SimpleNamespace())
        with self.assertRaises(AttributeError):
            binder(bad_model)

    def test_recorded_host_with_digit_leading_module_imports_via_importlib(self):
        """An identifier-unsafe host module imports through a string literal.

        HF remote-code cache hashes can start with a digit
        (``transformers_modules.kimi_k3.7dc8....modeling_kimi_linear``); a
        ``from`` statement over that path is a syntax error, so the host must
        be bound through ``importlib.import_module`` instead.
        """
        class ExternalMoe:
            def forward(self, hidden_states):
                return hidden_states

        ExternalMoe.__module__ = (
            "transformers_modules.kimi_k3.7dc8cececbd11535.modeling_kimi_linear")
        record = SimpleNamespace(
            module_fqn="model.language_model.layers.1.mlp",
            module_type=ExternalMoe,
            factory=lambda: None,
            original_forward=ExternalMoe.forward,
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            param_plan={
                record.module_fqn: {
                    "is_boundary": True,
                    "in_src": {"hidden_states": {}},
                    "in_dst": {"hidden_states": {}},
                    "out_src": {"output": {}},
                    "out_dst": {"output": {}},
                }
            },
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": {"_target_": "example.compute"},
            }],
            mesh_dim_names=[],
        )
        source = """
class OuterModel:
    def __init__(self, language_model):
        self.language_model = language_model
"""

        rendered = materialize_recorded_forwards(source, meta)

        ast.parse(rendered)
        self.assertNotIn(
            "from transformers_modules.kimi_k3.7dc8", rendered)
        self.assertIn("import importlib as _hp_importlib", rendered)
        self.assertIn(
            'ExternalMoe = getattr(_hp_importlib.import_module('
            '"transformers_modules.kimi_k3.7dc8cececbd11535.modeling_kimi_linear"'
            '), "ExternalMoe")',
            rendered,
        )
        self.assertIn("class HPCodegenExternalMoe(ExternalMoe)", rendered)

    def test_hp_bind_compute_delegates_to_recorded_binder_without_type_scan(self):
        """The runtime binds recorded hosts through the generated binder only."""
        from hyper_parallel.codegen import runtime as codegen_runtime

        bound = []

        class NoScanModel:
            """A model whose module tree must never be walked by type."""

            def modules(self):
                raise AssertionError(
                    "hp_bind_compute must not scan model.modules() by type"
                )

        model = NoScanModel()
        generated_module = SimpleNamespace(
            hp_bind_recorded_forwards=lambda target: bound.append(target)
        )
        injections = [{"match": "model.language_model.layers.1.mlp"}]
        with patch.object(
            codegen_runtime, "_resolve_runtime_meshes",
            return_value=(object(), object(), object(), ()),
        ):
            codegen_runtime.hp_bind_compute(
                model, injections, mesh_context=None,
                generated_module=generated_module,
            )
            self.assertEqual(bound, [model])
            # Without a binder the per-rule loop is a no-op — no type lookup.
            codegen_runtime.hp_bind_compute(
                NoScanModel(), injections, mesh_context=None,
                generated_module=None,
            )

    def test_hp_bind_compute_uses_explicit_factory_name(self):
        """The runtime binding has the same visible name as the YAML target."""
        from hyper_parallel.codegen import runtime as codegen_runtime

        module = SimpleNamespace(
            _forward_impl=lambda: None,
            forward=lambda: None,
        )

        def compute(value):
            return value

        injections = [{
            "match": "model.layers.0.mlp",
            "local_compute_fn": {"_target_": "example.deepseek_ep_compute_fn"},
        }]
        with patch.object(
            codegen_runtime, "_resolve_runtime_meshes",
            return_value=(object(), object(), object(), ()),
        ), patch.object(
            codegen_runtime, "_injection_target",
            return_value=compute,
        ), patch(
            "hyper_parallel.distributed._builder.rule_resolver."
            "_resolve_local_compute_fn",
            return_value=compute,
        ), patch(
            "hyper_parallel.distributed._builder.parameter_sharding._resolve_module",
            return_value=module,
        ):
            codegen_runtime.hp_bind_compute(
                SimpleNamespace(), injections, mesh_context=None,
            )

        self.assertIs(module.deepseek_ep_compute_fn, compute)
        self.assertFalse(hasattr(module, "__hp_compute__"))

    def test_recorded_imported_subtree_binding_passes_materialization_audit(self):
        """The audit sees the runtime-bound form and keeps the FQN binding."""
        class ExternalMoe:
            def forward(self, hidden_states):
                return hidden_states

        record = SimpleNamespace(
            module_fqn="model.language_model.layers.1.mlp",
            module_type=ExternalMoe,
            factory=lambda: None,
            original_forward=ExternalMoe.forward,
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            param_plan={
                record.module_fqn: {
                    "is_boundary": True,
                    "in_src": {"hidden_states": {}},
                    "in_dst": {"hidden_states": {}},
                    "out_src": {"output": {}},
                    "out_dst": {"output": {}},
                }
            },
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": {"_target_": "example.compute"},
            }],
            mesh_dim_names=[],
            materialized_patches=[],
        )
        source = """
class OuterModel:
    def __init__(self, language_model):
        self.language_model = language_model
"""

        rendered = materialize_recorded_forwards(source, meta)

        _verify_recorded_patch_materialization(rendered, meta)
        self.assertEqual(meta.materialized_patches, [{
            "fqn": record.module_fqn,
            "host_class": "HPCodegenExternalMoe",
            "kind": "local_compute",
            "source_form": "subclass_fqn_binding",
        }])
        # A materialization without the explicit FQN binding must fail fast.
        broken = rendered.replace(
            "module = model.model.language_model.layers[1].mlp", "module = model")
        with self.assertRaisesRegex(RuntimeError, record.module_fqn):
            _verify_recorded_patch_materialization(broken, meta)

    def test_recorded_imported_host_with_constructor_call_skips_binder(self):
        """A retargeted construction site needs no runtime binding."""
        class ExternalMoe:
            def forward(self, hidden_states):
                return hidden_states

        record = SimpleNamespace(
            module_fqn="model.layers.0.moe",
            module_type=ExternalMoe,
            factory=lambda: None,
            original_forward=ExternalMoe.forward,
        )
        meta = SimpleNamespace(
            _build_patches=SimpleNamespace(local_computes=(record,)),
            param_plan={
                record.module_fqn: {
                    "is_boundary": True,
                    "in_src": {"hidden_states": {}},
                    "in_dst": {"hidden_states": {}},
                    "out_src": {"output": {}},
                    "out_dst": {"output": {}},
                }
            },
            injections=[{
                "match": record.module_fqn,
                "local_compute_fn": {"_target_": "example.compute"},
            }],
            mesh_dim_names=[],
            materialized_patches=[],
        )
        source = """
class HostContainer:
    def __init__(self):
        self.moe = ExternalMoe()
"""

        rendered = materialize_recorded_forwards(source, meta)

        self.assertIn("self.moe = HPCodegenExternalMoe()", rendered)
        self.assertNotIn("hp_bind_recorded_forwards", rendered)
        _verify_recorded_patch_materialization(rendered, meta)
        self.assertEqual(
            meta.materialized_patches[0]["source_form"], "subclass_forward")

    def test_recorded_patch_audit_covers_replacement_and_inner_wrapper(self):
        """The audit manifest includes every builder-recorded patch kind."""
        host_type = type("Host", (), {})
        replacement_type = type("Replacement", (), {})
        patches = SimpleNamespace(
            local_computes=(),
            module_replacements=(SimpleNamespace(
                module_fqn="model.proj",
                replacement_type=replacement_type,
            ),),
            inner_wrappers=(SimpleNamespace(
                module_fqn="model.attn",
                target_fqn="self",
                request=SimpleNamespace(target=host_type()),
            ),),
        )
        meta = SimpleNamespace(_build_patches=patches, materialized_patches=[])
        source = """
class Host:
    def forward(self, value):
        return value

class HostCP(Host):
    def forward(self, value):
        return super().forward(value)

class Replacement:
    pass

replacement = Replacement()
"""
        _verify_recorded_patch_materialization(source, meta)
        self.assertEqual(
            {entry["kind"] for entry in meta.materialized_patches},
            {"module_replacement", "inner_wrapper"},
        )

    def setUp(self):
        """Warm the lazy family-provider discovery before any test runs.

        ``registry._DISCOVERED_PROVIDERS`` is a lazy sentinel (``None``
        until ``_discover_family_providers()`` fills it), and several
        tests ``patch.dict`` the attribute directly — which requires the
        discovery to have run in this process.  A shared-process run
        warms it incidentally; the gate runs each test in a cold
        process, so warm it here (the call is idempotent).
        """
        registry._discover_family_providers()

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_inline_owned_boundary_retains_sharding_without_double_wrapping(self):
        """External-state forwards must never access an uninstalled boundary.

        Feature: structure-driven inline source generation.
        Description: External-state forwards must never access an uninstalled boundary.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        attention = {"is_boundary": True, "params": {"q_proj.weight": {"tp": "S(0)"}}}
        linear = {"is_boundary": True, "params": {"weight": {"tp": "S(0)"}}}
        meta = SimpleNamespace(
            external_state_classes=["GeneratedAttention"],
            boundary_classes={"layers.0.attn": "GeneratedAttention", "head": "Linear"},
            param_plan={"layers.0.attn": attention, "head": linear},
        )
        normalize_inline_meta(meta, ())
        self.assertFalse(attention["is_boundary"])
        self.assertEqual(attention["params"], {"q_proj.weight": {"tp": "S(0)"}})
        self.assertTrue(linear["is_boundary"])
        self.assertEqual(meta.boundary_classes, {"head": "Linear"})
        before = deepcopy(vars(meta))
        normalize_inline_meta(meta, ())
        self.assertEqual(vars(meta), before)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_legacy_and_explicit_qwen_identity(self):
        """The adapter path infers the family; every alias selects the same spec.
        ``module_type`` is the rule's own declared source type, so it travels
        with the lookup rather than being reconstructed from the family name.

        Feature: structure-driven inline source generation.
        Description: The adapter path infers the family; every alias selects the same spec.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        legacy = replacement_spec(ATTENTION, module_type=ATTENTION_TYPE)
        self.assertIsNotNone(legacy)
        self.assertIs(legacy, replacement_spec(ATTENTION, "qwen3_moe", module_type=ATTENTION_TYPE))
        self.assertIs(
            legacy, replacement_spec(ATTENTION, "Qwen3MoeForCausalLM", module_type=ATTENTION_TYPE)
        )
        # The capability spec carries no block class: the concrete class is a
        # per-model fact derived from the frozen boundary map.
        self.assertIsNone(strategy_spec(EP).target_class)
        self.assertIsNotNone(strategy_spec(EP).body_template)
        # A CP wrapper is a strategy *because of its position* in the override
        # (``inner_wrapper``); the same target as a local compute fn is not one.
        self.assertIsNone(strategy_spec(CP))
        self.assertIsNone(strategy_spec(CP, inner_wrapper=True).body_template)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_cp_wrapper_with_only_closure_forward_is_not_inlined(self):
        """A closure-only CP forward must not become an invalid import."""
        self.assertIsNone(inner_wrapper_target(CP))

    def test_deepseek_v32_cp_inline_resolves_subclass_shell_and_imports(self):
        """DeepSeek-V3.2 CP inner wrapper resolves to an inline subclass shell.

        Feature: structure-driven inline source generation.
        Description: A recognizable CP wrapper factory gets a dispatch
            subclass materialized into the generated artifact without an
            adapter-side codegen registration, plus the helper import.
        Expectation: the resolved strategy spec carries a subclass shell (no
            target class, no forward-replacing body) and the helper import.
        """
        spec = strategy_spec(DEEPSEEK_CP, "deepseek_v32", inner_wrapper=True)
        self.assertIsNone(spec.target_class)
        self.assertIsNone(spec.body_template)
        self.assertIsNotNone(spec.subclass_shell)
        self.assertIn("_deepseek_v32_dsa_cp_forward", spec.subclass_shell)
        self.assertIn("if not self.cp_enable:", spec.subclass_shell)
        self.assertEqual(spec.subclass_state_attr,
                         "_hp_deepseek_v32_dsa_cp_mesh")
        self.assertTrue(spec.imports)
        import_names = [name for patch in spec.imports for name in patch.names]
        self.assertIn("_deepseek_v32_dsa_cp_forward", import_names)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_cp_subclass_materializes_snippet_and_ctor_swap_without_forward_extract(self):
        """The CP shell materializes per host as a subclass plus a name swap.

        Feature: structure-driven inline source generation.
        Description: A CP rule materializes one in-artifact subclass per
            attention host — a module snippet carrying the ``_hp_inline_cp``
            marker plus a scoped name-mode constructor swap — and the host
            keeps its compiled boundary forward, so no forward extract exists.
        Expectation: repeated rules re-deriving the same host add nothing, and
            a rule selecting no boundary fails fast instead of emitting an
            artifact that silently lost the strategy.
        """
        rules = (
            InlineRule(("model.layers.0.self_attn",), inner_wrapper_target=DEEPSEEK_CP),
            InlineRule(("model.layers.5.self_attn",), inner_wrapper_target=DEEPSEEK_CP),
        )
        boundary = {
            "model.layers.0.self_attn": "DeepseekV32DSAAttention",
            "model.layers.5.self_attn": "DeepseekV32DSAAttention",
        }
        patches = build_strategy_patches(rules, "deepseek_v32", boundary)
        self.assertEqual(len(patches.module_snippets), 1)
        snippet = patches.module_snippets[0].text
        self.assertIn("class DeepseekV32DSAAttentionCP(DeepseekV32DSAAttention):", snippet)
        self.assertIn("_hp_inline_cp = True", snippet)
        self.assertIn('_hp_cp_state_attr = "_hp_deepseek_v32_dsa_cp_mesh"', snippet)
        self.assertIn("if not self.cp_enable:", snippet)
        self.assertIn("_deepseek_v32_dsa_cp_forward(self, *args, **kwargs)", snippet)
        self.assertEqual(patches.constructor_replaces, [
            ConstructorReplacePatch(
                old_ctor="DeepseekV32DSAAttention",
                new_ctor="DeepseekV32DSAAttentionCP",
                mode="name",
                scope_attrs=("self_attn",),
            ),
        ])
        self.assertEqual(patches.forward_extracts, [])
        with self.assertRaisesRegex(RuntimeError, "matched no boundary"):
            build_strategy_patches(rules[:1], "deepseek_v32", {"model.layers.0.mlp": "MLP"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_one_strategy_target_materializes_across_distinct_hosts(self):
        """One EP target serving dense and MoE layers materializes both.

        Feature: structure-driven inline source generation.
        Description: Rules sharing one strategy target may select different
            host classes — one EP compute target serving both dense-MLP and
            MoE layers — so every rule derives its hosts and the per-class
            bookkeeping keeps one forward extract per host.
        Expectation: three rules over two distinct hosts emit two extracts,
            and a rule re-selecting an emitted host adds no third.
        """
        rules = (
            InlineRule(("model.layers.3.mlp",), local_compute_target=EP),
            InlineRule(("model.layers.1.mlp",), local_compute_target=EP),
            InlineRule(("model.layers.7.mlp",), local_compute_target=EP),
        )
        boundary = {
            "model.layers.3.mlp": "Qwen3MoeSparseMoeBlock",
            "model.layers.1.mlp": "DenseMLP",
            "model.layers.7.mlp": "DenseMLP",
        }
        patches = build_strategy_patches(rules, "qwen3_moe", boundary)
        self.assertEqual(
            sorted(extract.class_name for extract in patches.forward_extracts),
            ["DenseMLP", "Qwen3MoeSparseMoeBlock"],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_shell_hidden_arg_token_binds_to_each_hosts_first_parameter(self):
        """The shell token resolves per host from its own forward signature.

        Feature: structure-driven inline source generation.
        Description: One strategy template carries ``__hp_hidden_arg__``
            because its hosts name their first parameter differently (a MoE
            block takes ``hidden_states``, a dense MLP takes ``x``); the
            patch engine binds the token to each host's first positional
            parameter at apply time.
        Expectation: each host's inlined body uses its own parameter name, no
            token survives anywhere, and a host without a positional parameter
            fails fast.
        """
        source = """
class DenseMLP:
    def forward(self, x):
        return x

class SparseMoe:
    def forward(self, hidden_states):
        return hidden_states

class Vararg:
    def forward(self, *args):
        return args
"""
        body = "return __hp_hidden_arg__ + 1"
        patch_set = InlinePatchSet()
        patch_set.forward_extracts.extend([
            ForwardExtractPatch(class_name="DenseMLP", method_name="forward", body=body),
            ForwardExtractPatch(class_name="SparseMoe", method_name="forward", body=body),
        ])
        rendered = apply_patch_set(source, patch_set)
        self.assertIn("return x + 1", rendered)
        self.assertIn("return hidden_states + 1", rendered)
        self.assertNotIn("__hp_hidden_arg__", rendered)
        ast.parse(rendered)
        vararg_set = InlinePatchSet()
        vararg_set.forward_extracts.append(
            ForwardExtractPatch(class_name="Vararg", method_name="forward", body=body)
        )
        with self.assertRaisesRegex(RuntimeError, "no positional parameter"):
            apply_patch_set(source, vararg_set)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_name_swap_retargets_the_callsite_a_wrap_patch_introduced(self):
        """The CP name swap runs after the wrap, on the re-parsed text.

        Feature: structure-driven inline source generation.
        Description: The DeepSeek-V3.2 attention replacement wraps with a
            component of the same name, so the wrap introduces a NEW outer
            ``DeepseekV32DSAAttention(...)`` call; the CP name swap must run
            on the re-parsed wave output to retarget that outer call —
            applying it first would consume the wrap's call site.
        Expectation: the assignment installs the CP subclass wrapping the
            component, and the inner original constructor call survives.
        """
        source = """
class D:
    def __init__(self):
        pass

class DCP(D):
    def __init__(self):
        super().__init__()

class Layer:
    def __init__(self):
        self.self_attn = D()
"""
        patch_set = InlinePatchSet()
        patch_set.constructor_replaces.extend([
            ConstructorReplacePatch(
                old_ctor="D", new_ctor="D", mode="wrap_source",
                keyword_args=("module_fqn=''",), scope_attrs=("self_attn",),
            ),
            ConstructorReplacePatch(
                old_ctor="D", new_ctor="DCP", mode="name", scope_attrs=("self_attn",),
            ),
        ])
        rendered = apply_patch_set(source, patch_set)
        self.assertIn("self.self_attn = DCP(module=D(), module_fqn='')", rendered)
        self.assertNotIn("self.self_attn = D(module=D()", rendered)
        ast.parse(rendered)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_unknown_identity_does_not_select_qwen(self):
        """An explicit unsupported identity fails loudly without mutating metadata.

        Feature: structure-driven inline source generation.
        Description: An explicit unsupported identity fails loudly without mutating metadata.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        meta = _meta()
        before = deepcopy(vars(meta))
        with self.assertRaisesRegex(RuntimeError, "no inline declaration"):
            render_inline_modeling(SOURCE, meta, "unregistered_model")
        self.assertEqual(vars(meta), before)
        self.assertIsNone(
            replacement_spec(ATTENTION, "unregistered_model", module_type=ATTENTION_TYPE)
        )
        self.assertIsNone(replacement_spec("unknown.replace"))
        self.assertIsNone(strategy_spec(None))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_missing_module_type_cannot_name_a_source_class(self):
        """Without the rule's source type there is no constructor call to target.

        Feature: structure-driven inline source generation.
        Description: Without the rule's source type there is no constructor call to target.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        self.assertIsNone(replacement_spec(ATTENTION))
        self.assertIsNone(replacement_spec(ATTENTION, "qwen3_moe"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_unknown_target_fails_and_ruleless_plan_passes_through(self):
        """Incomplete coverage fails loudly; a ruleless plan returns the source.

        Feature: structure-driven inline source generation.
        Description: Incomplete coverage fails loudly; a ruleless plan returns the source.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        meta = _meta(injections=[{"local_compute_fn": "unknown.strategy"}])
        before = deepcopy(vars(meta))
        with self.assertRaisesRegex(RuntimeError, "no inline declaration"):
            render_inline_modeling(SOURCE, meta)
        self.assertEqual(vars(meta), before)
        ruleless = _meta(injections=())
        ruleless.module_overrides = []
        self.assertEqual(render_inline_modeling(SOURCE, ruleless), SOURCE)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_both_strategy_targets_require_coverage(self):
        """Known local compute targets cannot hide unsupported inner wrappers.

        Feature: structure-driven inline source generation.
        Description: Known local compute targets cannot hide unsupported inner wrappers.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        meta = _meta(injections=[{"local_compute_fn": EP, "inner_wrapper": "unknown.wrapper"}])
        before = deepcopy(vars(meta))
        with self.assertRaisesRegex(RuntimeError, "unknown.wrapper"):
            render_inline_modeling(SOURCE, meta)
        self.assertEqual(vars(meta), before)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_conflicting_strategy_bodies_fail(self):
        """Two targets cannot silently choose different bodies for one method.

        Feature: structure-driven inline source generation.
        Description: Two targets cannot silently choose different bodies for one method.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        declarations = {
            "first": StrategySpec("one", (), "Block", body_template="return 1"),
            "second": StrategySpec("two", (), "Block", body_template="return 2"),
        }
        rules = (InlineRule((), local_compute_target="first", inner_wrapper_target="second"),)
        with patch(
            "hyper_parallel.codegen.inline.strategy_pass.strategy_spec",
            side_effect=lambda target, model_type, **kwargs: declarations.get(target),
        ):
            with self.assertRaisesRegex(ValueError, "Conflicting inline strategies"):
                build_strategy_patches(rules)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_repeated_rules_share_one_snippet_and_forward_patch(self):
        """One target matched by N rules contributes its snippets once.
        A plan freezes one rule per matched FQN, so without dedup a
        module-level snippet (the EP parallel-state accessor) is emitted once
        per layer — 61 identical copies for DeepSeek-V3.

        Feature: structure-driven inline source generation.
        Description: One target matched by N rules contributes its snippets once.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        snippet = ModuleSnippetPatch("def helper():\n    return 1")
        declaration = StrategySpec(
            "ep", (), "Block", body_template="return 1", snippets=(snippet,)
        )
        rules = tuple(
            InlineRule((f"model.layers.{index}.mlp",), local_compute_target=EP)
            for index in range(5)
        )
        with patch(
            "hyper_parallel.codegen.inline.strategy_pass.strategy_spec",
            side_effect=lambda target, model_type, **kwargs: declaration,
        ):
            patches = build_strategy_patches(rules)
        self.assertEqual(len(rules), 5)
        self.assertEqual(patches.module_snippets, [snippet])
        self.assertEqual(len(patches.forward_extracts), 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_qwen_metadata_and_source_remain_compatible(self):
        """QKV expansion is scoped, independent and stable across re-emission.

        Feature: structure-driven inline source generation.
        Description: QKV expansion is scoped, independent and stable across re-emission.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        source = """
class Qwen3MoeAttention:
    def forward(self, value):
        return value

class Qwen3MoeSparseMoeBlock:
    def forward(self, hidden_states):
        return hidden_states

class Model:
    def __init__(self):
        self.attn = Qwen3MoeAttention()
"""
        meta = _meta(
            injections=[
                {"match": "blocks.0.mlp", "local_compute_fn": EP},
                {"match": "blocks.0.attn", "inner_wrapper": CP},
            ],
            module_type="transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention",
            fqns=("blocks.0.attn",),
        )
        meta.boundary_classes = {
            "blocks.0.attn": "Qwen3MoeAttention",
            "blocks.0.mlp": "Qwen3MoeSparseMoeBlock",
        }
        expected_meta = deepcopy(meta)
        inferred = render_inline_modeling(source, meta)
        explicit = render_inline_modeling(source, expected_meta, "qwen3_moe")
        self.assertEqual(inferred, explicit)
        self.assertEqual(vars(meta), vars(expected_meta))
        self.assertIn("class GQAAttention", explicit)
        self.assertIn("def _forward_impl", explicit)
        # The removed module-level parallel-state accessor must be gone
        # everywhere: the inlined attention forward and the EP shell read the
        # instance channel the runtime binds at install time.
        self.assertNotIn("get_parallel_state", explicit)
        self.assertNotIn(
            "get_parallel_state", strategy_spec(EP, "qwen3_moe").body_template
        )
        # S4: the decoder-layer call wraps the source attention module with the
        # generated class's keyword-only ctor (matching the real component /
        # runtime replacement), so the source class is kept as the wrapper input.
        self.assertNotIn("self.attn = GQAAttention()", explicit)
        self.assertIn(
            "GQAAttention(module=Qwen3MoeAttention(), module_fqn='', context=None, "
            "attention_interface=run_qwen3_moe_flash_attention)",
            explicit,
        )
        self.assertIn("class Qwen3MoeAttention", explicit)
        # S4: the attention class is generated from the real component — the
        # kernel entry is inlined and the construction keeps the fused QKV
        # layout (no independent q/k/v projection).
        self.assertIn("def run_qwen3_moe_flash_attention", explicit)
        self.assertIn("torch_npu.npu_fusion_attention", explicit)
        self.assertIn("self.linear_qkv", explicit)
        self.assertIn("InterleaveQKV", explicit)
        self.assertNotIn("self.q_proj =", explicit)
        ast.parse(explicit)
        # The generated class name comes from the factory's component, not a table.
        self.assertIn(f"class {interface_component_class('qwen3_moe')}", explicit)
        # The fused GQA model exposes the real param ``linear_qkv`` (not the
        # source q/k/v), so the frozen plan anchors on it; checkpoint loading
        # maps source q/k/v into it via make_transforms.
        params = meta.param_plan["blocks.0.proj"]["params"]
        self.assertEqual(set(params), {"linear_qkv.weight"})
        self.assertEqual(meta.frozen_sharded_params, [
            "blocks.0.proj.linear_qkv.weight", "embedding.weight",
        ])
        self.assertEqual(render_inline_modeling(source, meta, "qwen3_moe"), explicit)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_rendered_component_boundary_is_emitted_while_the_ep_shell_stays_inline(self):
        """Only the class whose forward an inline body replaced is stripped.

        Feature: structure-driven inline source generation.
        Description: The rendered component class (the fused attention) is not
            an external-state class, so ``normalize_inline_meta`` keeps its
            boundary entry and the lowerer rewrites its forward into the
            emitted boundary form around ``_forward_impl``.  The EP shell, whose
            forward the inline strategy body replaced, is still stripped and
            keeps that body verbatim.
        Expectation: The emitted file carries exactly one boundary rewrite (the
            rendered component's) plus the shell's intact inline orchestration.
        """
        source = """
class Qwen3MoeAttention:
    def forward(self, value):
        return value

class Qwen3MoeSparseMoeBlock:
    def forward(self, hidden_states):
        return hidden_states

class Model:
    def __init__(self):
        self.attn = Qwen3MoeAttention()
        self.mlp = Qwen3MoeSparseMoeBlock()
"""
        boundary_entry = {
            "is_boundary": True,
            "in_src": {"hidden_states": {"tp": "S(1)"}},
            "in_dst": {"hidden_states": {"tp": "R"}},
            "out_src": {"output": {"tp": "P(sum)"}},
            "out_dst": {"output": {"tp": "S(1)"}},
        }
        meta = _meta(
            injections=[
                {"local_compute_fn": EP, "match": "blocks.0.mlp"},
                {"inner_wrapper": CP, "match": "blocks.0.attn"},
            ],
            module_type=ATTENTION_TYPE,
            fqns=("blocks.0.attn",),
        )
        meta.external_state_classes = ["Qwen3MoeSparseMoeBlock"]
        meta.boundary_classes = {
            "blocks.0.attn": "GQAAttention",
            "blocks.0.mlp": "Qwen3MoeSparseMoeBlock",
        }
        meta.param_plan = {
            "blocks.0.attn": dict(boundary_entry),
            "blocks.0.mlp": dict(boundary_entry),
        }
        meta.mesh_dim_names = ["tp"]

        rendered = render_inline_modeling(source, meta, "qwen3_moe")
        lowered = lower_forward_boundaries(
            rendered, meta, boundary_classes=meta.boundary_classes,
        )

        ast.parse(lowered)
        # The rendered component carries the emitted boundary form.
        self.assertEqual(lowered.count("self._hp_boundary.redistribute_inputs"), 1)
        self.assertIn("class GQAAttention", lowered)
        self.assertEqual(lowered.count("def _forward_impl"), 2)
        # The EP shell keeps the inline strategy body (and its own channel).
        self.assertIn("if not self.ep_enable:", lowered)
        self.assertIn("moe_ep_forward(", lowered)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_family_factories_alone_are_enough(self):
        """A family with only runtime factories resolves, with no codegen code.
        This is the acceptance shape: a registered family whose replacement
        provider constructs the generic ``RMSNorm`` needs no declaration at
        all -- the inline pipeline derives the component wiring, the source
        class (from the rule's ``module_type``) and the artifact itself.

        Feature: structure-driven inline source generation.
        Description: A family with only runtime factories resolves, with no codegen code.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        adapter = _toy_adapter()
        with (
            patch.dict(registry.MODEL_ADAPTER_REGISTRY),
            patch.dict(registry._FAMILY_ALIASES),
            patch.dict(registry._DISCOVERED_PROVIDERS, {"toy": adapter.model_type}),
        ):
            registry.register_model_adapter(adapter)
            meta = _meta(factory=TOY_NORM, module_type=TOY_NORM_TYPE, fqns=("blocks.0.norm",))
            meta.model_class = "ToyForCausalLM"
            emitted = _apply_inline_modeling(TOY_SOURCE, meta)
        namespace = _execute_source(emitted)
        # The factory hands the matched module over (``module=``), so the
        # generated call wraps it and the source class is kept as the wrapper
        # input; the family named neither the component nor the argument list.
        self.assertIn("RMSNorm(module=ToyNorm(4), module_fqn='', context=None)", emitted)
        self.assertIn("class ToyNorm", emitted)
        self.assertNotIn("Qwen", emitted)
        self.assertTrue(hasattr(namespace["ToyModel"](), "norm"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_unrecognized_component_is_a_coverage_gap(self):
        """A factory building a component the framework does not wire fails.

        Feature: structure-driven inline source generation.
        Description: A factory building a component the framework does not wire fails.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        adapter = _toy_adapter(factory=_toy_unknown_factory)
        with (
            patch.dict(registry.MODEL_ADAPTER_REGISTRY),
            patch.dict(registry._FAMILY_ALIASES),
            patch.dict(registry._DISCOVERED_PROVIDERS, {"toy": adapter.model_type}),
        ):
            registry.register_model_adapter(adapter)
            self.assertIsNone(
                replacement_spec(TOY_UNKNOWN, "toy", module_type=TOY_NORM_TYPE)
            )
            meta = _meta(factory=TOY_UNKNOWN, module_type=TOY_NORM_TYPE)
            with self.assertRaisesRegex(RuntimeError, "no inline declaration"):
                render_inline_modeling(TOY_SOURCE, meta, "toy")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_component_kind_selects_the_wiring(self):
        """The same source class wired through different components differs.
        Recognition reads the *factory's* component, so two factories that
        both replace ``ToyNorm`` produce different artifacts -- the family
        never names the wiring.

        Feature: structure-driven inline source generation.
        Description: The same source class wired through different components differs.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        adapter = _toy_adapter(factory=_toy_grouped_experts_factory)
        with (
            patch.dict(registry.MODEL_ADAPTER_REGISTRY),
            patch.dict(registry._FAMILY_ALIASES),
            patch.dict(registry._DISCOVERED_PROVIDERS, {"toy": adapter.model_type}),
        ):
            registry.register_model_adapter(adapter)
            spec = replacement_spec(TOY_NORM, "toy", module_type=TOY_NORM_TYPE)
        self.assertEqual(spec.new_ctor, "GroupedExperts")
        self.assertEqual(spec.mode, "wrap_source")
        self.assertFalse(spec.remove_class)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_incomplete_strategy_declarations_fail(self):
        """A partial method patch cannot silently count as supported.

        Feature: structure-driven inline source generation.
        Description: A partial method patch cannot silently count as supported.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        with self.assertRaisesRegex(ValueError, "requires a body_template"):
            StrategySpec("invalid", (), target_class="ToyBlock")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            StrategySpec("invalid", (), target_class="ToyBlock", body_template="")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_declarations_invalidate_cached_artifacts(self):
        """Changing what a target resolves to changes the artifact signature.
        The declarations are derived from the family's runtime factories
        (outside the codegen tree), so the signature must fold the *resolved*
        content in: a factory switched to a different generic component must
        regenerate rather than reuse the previous bundle.

        Feature: structure-driven inline source generation.
        Description: Changing what a target resolves to changes the artifact signature.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        projection = SimpleNamespace(
            to_dict=lambda: {
                "source": {},
                "overrides": [
                    {
                        "match": "blocks.0.proj",
                        "module_type": TOY_NORM_TYPE,
                        "replace_module": {"_target_": TOY_NORM},
                    }
                ],
                "target": {},
            }
        )
        source = SimpleNamespace(to_dict=lambda: {"architecture": "ToyForCausalLM"})
        config = SimpleNamespace(codegen=False)

        def _signature(factory):
            adapter = _toy_adapter(factory=factory)
            with (
                patch.dict(registry.MODEL_ADAPTER_REGISTRY),
                patch.dict(registry._FAMILY_ALIASES),
                patch.dict(registry._DISCOVERED_PROVIDERS, {"toy": adapter.model_type}),
                patch("hyper_parallel.codegen.spec.project.project_codegen_spec", return_value=projection),
                patch("hyper_parallel.codegen.source.resolver.resolve_model_source", return_value=source),
                patch.object(manager, "_codegen_implementation_digest", return_value="unchanged-core"),
            ):
                registry.register_model_adapter(adapter)
                return manager._compute_signature(config, None)

        self.assertNotEqual(
            _signature(_toy_rms_norm_factory), _signature(_toy_grouped_experts_factory)
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_declared_method_name_preserves_original_implementation(self):
        """Non-forward strategy methods retain an executable original method.

        Feature: structure-driven inline source generation.
        Description: Non-forward strategy methods retain an executable original method.
        Expectation: the emitted artifact and metadata satisfy the assertions in this test.
        """
        source = "class Block:\n    def compute(self, value):\n        return value * 2\n"
        patches = InlinePatchSet(forward_extracts=[ForwardExtractPatch(
            "Block", "compute", "return self._forward_impl(value) + 3",
        )])
        namespace = _execute_source(apply_patch_set(source, patches))
        self.assertEqual(namespace["Block"]().compute(4), 11)

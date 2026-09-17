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
from tempfile import TemporaryDirectory
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.codegen.emit.modeling import _apply_inline_modeling
from hyper_parallel.codegen import manager
from hyper_parallel.codegen.inline.ir import (
    ForwardExtractPatch, InlinePatchSet, InlineRule,
)
from hyper_parallel.codegen.inline.framework_spec import GENERATED_ATTENTION_CLASS
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling
from hyper_parallel.codegen.inline.spec_bundle import StrategySpec
from hyper_parallel.codegen.inline.specs import replacement_spec, strategy_spec
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models import registry


QWEN_ROOT = "hyper_parallel.models.qwen3_moe.adapter."
ATTENTION = QWEN_ROOT + "replacements.replace_qwen3_moe_flash_attention"
ATTENTION_TYPE = "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention"
EP = QWEN_ROOT + "distributed.expert_parallel.qwen3moe_ep_compute_fn"
CP = QWEN_ROOT + "distributed.context_parallel.qwen3_moe_flash_attention_cp_wrapper"

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
class ToyNorm:
    def __init__(self, hidden_size, eps=1e-5):
        self.hidden_size = hidden_size
        self.eps = eps

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


def _meta(factory=ATTENTION, injections=(), module_type=ATTENTION_TYPE):
    return SimpleNamespace(
        module_overrides=[
            {"factory": factory, "module_type": module_type, "fqns": ["blocks.0.proj"]}
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

    def test_inline_owned_boundary_retains_sharding_without_double_wrapping(self):
        """External-state forwards must never access an uninstalled boundary."""
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

    def test_legacy_and_explicit_qwen_identity(self):
        """The adapter path infers the family; every alias selects the same spec.

        ``module_type`` is the rule's own declared source type, so it travels
        with the lookup rather than being reconstructed from the family name.
        """
        legacy = replacement_spec(ATTENTION, module_type=ATTENTION_TYPE)
        self.assertIsNotNone(legacy)
        self.assertIs(legacy, replacement_spec(ATTENTION, "qwen3_moe", module_type=ATTENTION_TYPE))
        self.assertIs(
            legacy, replacement_spec(ATTENTION, "Qwen3MoeForCausalLM", module_type=ATTENTION_TYPE)
        )
        self.assertEqual(strategy_spec(EP).target_class, "Qwen3MoeSparseMoeBlock")
        # A CP wrapper is a strategy *because of its position* in the override
        # (``inner_wrapper``); the same target as a local compute fn is not one.
        self.assertIsNone(strategy_spec(CP))
        self.assertIsNone(strategy_spec(CP, inner_wrapper=True).body_template)

    def test_unknown_identity_does_not_select_qwen(self):
        """An explicit unsupported identity fails loudly without mutating metadata."""
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

    def test_missing_module_type_cannot_name_a_source_class(self):
        """Without the rule's source type there is no constructor call to target."""
        self.assertIsNone(replacement_spec(ATTENTION))
        self.assertIsNone(replacement_spec(ATTENTION, "qwen3_moe"))

    def test_unknown_target_fails_and_ruleless_plan_passes_through(self):
        """Incomplete coverage fails loudly; a ruleless plan returns the source."""
        meta = _meta(injections=[{"local_compute_fn": "unknown.strategy"}])
        before = deepcopy(vars(meta))
        with self.assertRaisesRegex(RuntimeError, "no inline declaration"):
            render_inline_modeling(SOURCE, meta)
        self.assertEqual(vars(meta), before)
        ruleless = _meta(injections=())
        ruleless.module_overrides = []
        self.assertEqual(render_inline_modeling(SOURCE, ruleless), SOURCE)

    def test_both_strategy_targets_require_coverage(self):
        """Known local compute targets cannot hide unsupported inner wrappers."""
        meta = _meta(injections=[{"local_compute_fn": EP, "inner_wrapper": "unknown.wrapper"}])
        before = deepcopy(vars(meta))
        with self.assertRaisesRegex(RuntimeError, "unknown.wrapper"):
            render_inline_modeling(SOURCE, meta)
        self.assertEqual(vars(meta), before)

    def test_conflicting_strategy_bodies_fail(self):
        """Two targets cannot silently choose different bodies for one method."""
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

    def test_qwen_metadata_and_source_remain_compatible(self):
        """QKV expansion is scoped, independent and stable across re-emission."""
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
            injections=[{"local_compute_fn": EP}, {"inner_wrapper": CP}],
            module_type="transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention",
        )
        expected_meta = deepcopy(meta)
        inferred = render_inline_modeling(source, meta)
        explicit = render_inline_modeling(source, expected_meta, "qwen3_moe")
        self.assertEqual(inferred, explicit)
        self.assertEqual(vars(meta), vars(expected_meta))
        self.assertIn("class GQAAttention", explicit)
        self.assertIn("def _forward_impl", explicit)
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
        # The generated class name comes from the component, not the family.
        self.assertIn(f"class {GENERATED_ATTENTION_CLASS}", explicit)
        # The fused GQA model exposes the real param ``linear_qkv`` (not the
        # source q/k/v), so the frozen plan anchors on it; checkpoint loading
        # maps source q/k/v into it via make_transforms.
        params = meta.param_plan["blocks.0.proj"]["params"]
        self.assertEqual(set(params), {"linear_qkv.weight"})
        self.assertEqual(meta.frozen_sharded_params, [
            "blocks.0.proj.linear_qkv.weight", "embedding.weight",
        ])
        self.assertEqual(render_inline_modeling(source, meta, "qwen3_moe"), explicit)

    def test_family_factories_alone_are_enough(self):
        """A family with only runtime factories resolves, with no codegen code.

        This is the acceptance shape: a registered family whose replacement
        provider constructs the generic ``RMSNorm`` needs no declaration at
        all -- the inline pipeline derives the component wiring, the source
        class (from the rule's ``module_type``) and the artifact itself.
        """
        adapter = _toy_adapter()
        with (
            patch.dict(registry.MODEL_ADAPTER_REGISTRY),
            patch.dict(registry._FAMILY_ALIASES),
            patch.dict(registry._DISCOVERED_PROVIDERS, {"toy": adapter.model_type}),
        ):
            registry.register_model_adapter(adapter)
            meta = _meta(factory=TOY_NORM, module_type=TOY_NORM_TYPE)
            meta.model_class = "ToyForCausalLM"
            emitted = _apply_inline_modeling(TOY_SOURCE, meta)
        namespace = _execute_source(emitted)
        # The source class is swapped 1:1 for the generic component and removed;
        # the family named neither the component nor the argument list.
        self.assertIn("RMSNorm(4)", emitted)
        self.assertNotIn("class ToyNorm", emitted)
        self.assertNotIn("Qwen", emitted)
        self.assertTrue(hasattr(namespace["ToyModel"](), "norm"))

    def test_unrecognized_component_is_a_coverage_gap(self):
        """A factory building a component the framework does not wire fails."""
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

    def test_component_kind_selects_the_wiring(self):
        """The same source class wired through different components differs.

        Recognition reads the *factory's* component, so two factories that
        both replace ``ToyNorm`` produce different artifacts -- the family
        never names the wiring.
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

    def test_incomplete_strategy_declarations_fail(self):
        """A partial method patch cannot silently count as supported."""
        with self.assertRaisesRegex(ValueError, "provided together"):
            StrategySpec("invalid", (), target_class="ToyBlock")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            StrategySpec("invalid", (), target_class="ToyBlock", body_template="")

    def test_declarations_invalidate_cached_artifacts(self):
        """Changing what a target resolves to changes the artifact signature.

        The declarations are derived from the family's runtime factories
        (outside the codegen tree), so the signature must fold the *resolved*
        content in: a factory switched to a different generic component must
        regenerate rather than reuse the previous bundle.
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
        config = SimpleNamespace(codegen=False, modeling_backend="hf")

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

    def test_declared_method_name_preserves_original_implementation(self):
        """Non-forward strategy methods retain an executable original method."""
        source = "class Block:\n    def compute(self, value):\n        return value * 2\n"
        patches = InlinePatchSet(forward_extracts=[ForwardExtractPatch(
            "Block", "compute", "return self._forward_impl(value) + 3",
        )])
        namespace = _execute_source(apply_patch_set(source, patches))
        self.assertEqual(namespace["Block"]().compute(4), 11)

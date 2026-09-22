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
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen import manager
from hyper_parallel.codegen.inline.ir import (
    ForwardExtractPatch, InlinePatchSet, InlineRule, ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.framework_spec import interface_component_class
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling
from hyper_parallel.codegen.inline.spec_bundle import StrategySpec
from hyper_parallel.codegen.inline.specs import replacement_spec, strategy_spec
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models import registry
from tests.common.mark_utils import arg_mark


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
        self.assertEqual(strategy_spec(EP).target_class, "Qwen3MoeSparseMoeBlock")
        # A CP wrapper is a strategy *because of its position* in the override
        # (``inner_wrapper``); the same target as a local compute fn is not one.
        self.assertIsNone(strategy_spec(CP))
        self.assertIsNone(strategy_spec(CP, inner_wrapper=True).body_template)

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
            injections=[{"local_compute_fn": EP}, {"inner_wrapper": CP}],
            module_type="transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention",
            fqns=("blocks.0.attn",),
        )
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
        self.assertEqual(lowered.count("self._hyper_boundary.redistribute_inputs"), 1)
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
        with self.assertRaisesRegex(ValueError, "provided together"):
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

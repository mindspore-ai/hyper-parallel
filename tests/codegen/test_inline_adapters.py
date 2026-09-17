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
"""Regression coverage for adapter-driven inline source generation."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.codegen.emit.modeling import _apply_inline_modeling
from hyper_parallel.codegen import manager
from hyper_parallel.codegen.inline.ir import (
    ForwardExtractPatch, ImportPatch, InlinePatchSet, InlineRule, ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling
from hyper_parallel.codegen.inline.spec_bundle import InlineSpecBundle, MetaNormalizer, ReplacementSpec, StrategySpec
from hyper_parallel.codegen.inline.specs import replacement_spec, strategy_spec
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models import registry


QWEN_ROOT = "hyper_parallel.models.qwen3_moe.adapter."
ATTENTION = QWEN_ROOT + "replacements.replace_qwen3_moe_flash_attention"
EP = QWEN_ROOT + "distributed.expert_parallel.qwen3moe_ep_compute_fn"
CP = QWEN_ROOT + "distributed.context_parallel.qwen3_moe_flash_attention_cp_wrapper"
CUSTOM_REPLACEMENT = "example.replace_projection"
CUSTOM_STRATEGY = "example.parallel_projection"
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


def _meta(factory=ATTENTION, injections=()):
    return SimpleNamespace(
        module_overrides=[{"factory": factory, "fqns": ["blocks.0.proj"]}],
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
    """Exercise discovery, coverage enforcement and model-independent emission."""

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
        """Legacy target lookup and architecture aliases resolve the same specs."""
        legacy = replacement_spec(ATTENTION)
        self.assertIsNotNone(legacy)
        self.assertIs(legacy, replacement_spec(ATTENTION, "qwen3_moe"))
        self.assertIs(legacy, replacement_spec(ATTENTION, "Qwen3MoeForCausalLM"))
        self.assertEqual(strategy_spec(EP).target_class, "Qwen3MoeSparseMoeBlock")
        self.assertIsNone(strategy_spec(CP).body_template)

    def test_unknown_identity_does_not_select_qwen(self):
        """An explicit unsupported identity fails loudly without mutating metadata."""
        meta = _meta()
        before = deepcopy(vars(meta))
        with self.assertRaisesRegex(RuntimeError, "no inline declaration"):
            render_inline_modeling(SOURCE, meta, "unregistered_model")
        self.assertEqual(vars(meta), before)
        self.assertIsNone(replacement_spec(ATTENTION, "unregistered_model"))
        self.assertIsNone(replacement_spec("unknown.replace"))
        self.assertIsNone(strategy_spec(None))

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
            side_effect=lambda target, model_type: declarations.get(target),
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
        meta = _meta(injections=[{"local_compute_fn": EP}, {"inner_wrapper": CP}])
        expected_meta = deepcopy(meta)
        legacy = render_inline_modeling(source, meta)
        explicit = render_inline_modeling(source, expected_meta, "qwen3_moe")
        self.assertEqual(legacy, explicit)
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
        # The fused GQA model exposes the real param ``linear_qkv`` (not the
        # source q/k/v), so the frozen plan anchors on it; checkpoint loading
        # maps source q/k/v into it via make_transforms.
        params = meta.param_plan["blocks.0.proj"]["params"]
        self.assertEqual(set(params), {"linear_qkv.weight"})
        self.assertEqual(meta.frozen_sharded_params, [
            "blocks.0.proj.linear_qkv.weight", "embedding.weight",
        ])
        self.assertEqual(render_inline_modeling(source, meta, "qwen3_moe"), explicit)

    def test_new_adapter_generates_executable_source(self):
        """A registered non-Qwen adapter supplies constructors, methods and metadata."""
        bundle = InlineSpecBundle(
            replacement_specs={
                CUSTOM_REPLACEMENT: ReplacementSpec(
                    "OldProjection", "NewProjection", (),
                    snippets=(ModuleSnippetPatch(
                        "class NewProjection:\n    def forward(self, value):\n        return value * 2"
                    ),),
                ),
            },
            strategy_specs={
                CUSTOM_STRATEGY: StrategySpec(
                    "custom_kind", (ImportPatch("math", ("sqrt",)),),
                    target_class="ToyBlock", body_template="return self._forward_impl(value) + 3",
                ),
            },
            meta_normalizers=(MetaNormalizer(
                CUSTOM_REPLACEMENT, {"linear_qkv.weight": ("projection.weight",)},
            ),),
        )
        provider = SimpleNamespace(get_render_spec=lambda: bundle)
        adapter = ModelAdapterSpec("ToyForCausalLM", "toy", inline_codegen=lambda: provider)
        with patch.dict(registry.MODEL_ADAPTER_REGISTRY), patch.dict(registry._FAMILY_ALIASES):
            registry.register_model_adapter(adapter)
            meta = _meta(CUSTOM_REPLACEMENT, [
                {"local_compute_fn": CUSTOM_STRATEGY}, {"local_compute_fn": CUSTOM_STRATEGY},
            ])
            meta.model_class = "ToyForCausalLM"
            emitted = _apply_inline_modeling(SOURCE, meta)
        namespace = _execute_source(emitted)
        self.assertEqual(namespace["ToyBlock"]().forward(4), 11)
        self.assertEqual(emitted.count("def _forward_impl"), 1)
        self.assertNotIn("Qwen", emitted)
        self.assertEqual(meta.frozen_sharded_params, ["blocks.0.proj.projection.weight", "embedding.weight"])

    def test_provider_absent_and_invalid_bundle(self):
        """Missing capabilities fall back; malformed provider results fail clearly."""
        absent = ModelAdapterSpec("BareForCausalLM", "bare")
        bad = ModelAdapterSpec(
            "BadForCausalLM", "bad",
            inline_codegen=lambda: SimpleNamespace(get_render_spec=lambda: {}),
        )
        with patch.dict(registry.MODEL_ADAPTER_REGISTRY, {"bare": absent, "bad": bad}):
            self.assertIsNone(replacement_spec(ATTENTION, "bare"))
            with self.assertRaisesRegex(TypeError, "InlineSpecBundle"):
                replacement_spec(ATTENTION, "bad")

    def test_incomplete_strategy_declarations_fail(self):
        """A partial method patch cannot silently count as supported."""
        with self.assertRaisesRegex(ValueError, "provided together"):
            StrategySpec("invalid", (), target_class="ToyBlock")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            StrategySpec("invalid", (), target_class="ToyBlock", body_template="")

    def test_adapter_declarations_invalidate_cached_artifacts(self):
        """Changing only an adapter method body changes the artifact signature."""
        projection = SimpleNamespace(to_dict=lambda: {"source": {}, "target": {}})
        source = SimpleNamespace(to_dict=lambda: {"architecture": "ToyForCausalLM"})
        config = SimpleNamespace(codegen=False, modeling_backend="hf")
        first = InlineSpecBundle({}, {"example.strategy": StrategySpec(
            "custom", (), "Block", body_template="return 1",
        )})
        second = InlineSpecBundle({}, {"example.strategy": StrategySpec(
            "custom", (), "Block", body_template="return 2",
        )})
        with (
            patch("hyper_parallel.codegen.spec.project.project_codegen_spec", return_value=projection),
            patch("hyper_parallel.codegen.source.resolver.resolve_model_source", return_value=source),
            patch.object(manager, "_codegen_implementation_digest", return_value="unchanged-core"),
            patch.object(manager, "get_inline_spec_bundle", side_effect=[first, second]),
        ):
            before = manager._compute_signature(config, None)
            after = manager._compute_signature(config, None)
        self.assertNotEqual(before, after)

    def test_declared_method_name_preserves_original_implementation(self):
        """Non-forward strategy methods retain an executable original method."""
        source = "class Block:\n    def compute(self, value):\n        return value * 2\n"
        patches = InlinePatchSet(forward_extracts=[ForwardExtractPatch(
            "Block", "compute", "return self._forward_impl(value) + 3",
        )])
        namespace = _execute_source(apply_patch_set(source, patches))
        self.assertEqual(namespace["Block"]().compute(4), 11)

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
"""Structure->archetype table: the framework single source for EP declarations.

The archetype table is what makes "native HF model + one YAML" sufficient:
the strategy kind, target class, external-state classes, boundary
sub-patterns and EP body keys are all *structure-proven* here, and the model
adapters project from it instead of hand-writing family-named literals. These
tests lock the projection so a hand-written literal cannot creep back, and pin
the framework genericity of the component replacements.
"""

import unittest

from hyper_parallel.codegen import manager
from hyper_parallel.codegen.inline.framework_spec import (
    external_state_classes_for,
    interface_component_class,
    replacement_spec_for,
    strategy_spec_for,
)
from hyper_parallel.distributed.expert_parallel.archetypes import (
    moe_archetype,
    moe_external_state_classes,
)

QWEN_FP = ("topk_router_module", "none", "batched_parameters")
DEEPSEEK_FP = ("topk_router_module", "additive", "batched_parameters")

QWEN_EP = (
    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
    "qwen3moe_ep_compute_fn"
)
DEEPSEEK_EP = (
    "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn"
)

QWEN_ATTENTION = (
    "hyper_parallel.models.qwen3_moe.adapter.replacements."
    "replace_qwen3_moe_flash_attention"
)
QWEN_ATTENTION_TYPE = (
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention"
)
QWEN_NORM = (
    "hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_rms_norm"
)
QWEN_NORM_TYPE = "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm"
QWEN_EXPERTS = (
    "hyper_parallel.models.qwen3_moe.adapter.replacements."
    "replace_qwen3_moe_grouped_experts"
)
QWEN_EXPERTS_TYPE = "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts"


class TestMoeArchetypeTable(unittest.TestCase):
    """The table owns every family-named EP literal the adapters used to write."""

    def test_kind_is_structure_keyed_not_family_named(self):
        """Both recognized MoEs share one structure-keyed strategy kind."""
        self.assertEqual(moe_archetype(*QWEN_FP).kind, "moe_ep_routed")
        self.assertEqual(moe_archetype(*DEEPSEEK_FP).kind, "moe_ep_routed")

    def test_detection_fingerprint_and_merge_key_are_distinct(self):
        """DeepSeek's gate is *detected* as a top-k router module but *merges*
        through the sigmoid-group adapter -- the two keys must not be conflated."""
        deepseek = moe_archetype(*DEEPSEEK_FP)
        self.assertEqual(deepseek.router_kind, "sigmoid_group")
        self.assertEqual(deepseek.shared, "additive")

    def test_compute_factory_anchors_are_preserved(self):
        """The importable ``_target_`` anchors the YAML/native path still use."""
        self.assertEqual(moe_archetype(*QWEN_FP).compute_factory, QWEN_EP)
        self.assertEqual(moe_archetype(*DEEPSEEK_FP).compute_factory, DEEPSEEK_EP)

    def test_external_state_classes_derive_from_the_block_class(self):
        """The block whose forward is inlined is the external-state class."""
        self.assertEqual(moe_external_state_classes(*QWEN_FP), ("Qwen3MoeSparseMoeBlock",))
        self.assertEqual(moe_external_state_classes(*DEEPSEEK_FP), ("DeepseekV3MoE",))

    def test_unrecognized_fingerprint_yields_no_archetype(self):
        """Recognition never guesses a strategy for an unknown structure."""
        self.assertIsNone(moe_archetype("softmax_topk", "none", "batched_parameters"))
        self.assertEqual(moe_external_state_classes("softmax_topk", "none", "batched_parameters"), ())

    def test_manager_factory_table_mirrors_the_archetype_table(self):
        """The manager consumes the same table -- no duplicated mapping."""
        table = manager._ep_factory_table()
        self.assertEqual(table[QWEN_FP], QWEN_EP)
        self.assertEqual(table[DEEPSEEK_FP], DEEPSEEK_EP)


class TestReplacementWiringIsReadFromTheFactory(unittest.TestCase):
    """Component wiring is read from the family's own replacement factory.

    The emitted import, the wrap-vs-swap form, the constructor keywords and the
    handed-over kernel interface all come from the factory the YAML already names
    for the native path -- there is no component-name table to extend.
    """

    def _spec(self, target: str, module_type: str):
        return replacement_spec_for(target, "qwen3_moe", module_type=module_type)

    def test_rms_norm_factory_wraps_the_source_module(self):
        """A factory that passes ``module=`` wraps, so the source class survives."""
        spec = self._spec(QWEN_NORM, QWEN_NORM_TYPE)
        self.assertEqual((spec.old_ctor, spec.new_ctor), ("Qwen3MoeRMSNorm", "RMSNorm"))
        self.assertEqual(spec.mode, "wrap_source")
        self.assertEqual(spec.keyword_args, ("module_fqn=''", "context=None"))
        self.assertFalse(spec.remove_class)
        self.assertEqual(
            [patch.module for patch in spec.imports],
            ["hyper_parallel.components.modules"],
        )

    def test_grouped_experts_factory_keeps_the_source_class_as_wrapper_input(self):
        """Batched experts wrap the source module; the class must survive."""
        spec = self._spec(QWEN_EXPERTS, QWEN_EXPERTS_TYPE)
        self.assertEqual(spec.new_ctor, "GroupedExperts")
        self.assertEqual(spec.mode, "wrap_source")
        self.assertEqual(spec.keyword_args, ("module_fqn=''", "context=None"))
        self.assertFalse(spec.remove_class)
        self.assertFalse(spec.snippets)

    def test_attention_factory_hands_over_the_family_kernel_interface(self):
        """The handed-over interface keeps the family kernel entry in the artifact."""
        spec = self._spec(QWEN_ATTENTION, QWEN_ATTENTION_TYPE)
        self.assertEqual(spec.new_ctor, "GQAAttention")
        self.assertEqual(spec.mode, "wrap_source")
        self.assertEqual(
            spec.keyword_args,
            (
                "module_fqn=''",
                "context=None",
                "attention_interface=run_qwen3_moe_flash_attention",
            ),
        )
        self.assertTrue(spec.snippets)
        self.assertEqual(interface_component_class("qwen3_moe"), "GQAAttention")


class TestDeclarationsAreAStructuralProjection(unittest.TestCase):
    """Declarations project the archetype table / factory structure.

    No per-family ``render_spec.py`` remains: every lookup is derived from the
    framework archetype table plus the family's own runtime factories, so a
    hand-written family-named literal cannot creep back in.
    """

    def test_qwen_ep_spec_equals_its_archetype(self):
        """The desugared Qwen EP spec equals its declared archetype."""
        archetype = moe_archetype(*QWEN_FP)
        spec = strategy_spec_for(QWEN_EP, "qwen3_moe")
        self.assertEqual(spec.kind, archetype.kind)
        self.assertEqual(spec.target_class, archetype.target_class)
        self.assertIn(f"router_kind={archetype.router_kind!r}", spec.body_template)
        self.assertIn(f"shared={archetype.shared!r}", spec.body_template)
        self.assertEqual(
            external_state_classes_for("qwen3_moe", ({"local_compute_fn": QWEN_EP},)),
            moe_external_state_classes(*QWEN_FP),
        )
        # Only the class whose forward an inline strategy body replaces is
        # external state.  The rendered component class keeps the component's
        # own forward plus the emitted boundary form, so the shared compiled
        # boundary path owns it.
        self.assertNotIn(
            interface_component_class("qwen3_moe"),
            external_state_classes_for("qwen3_moe", ({"local_compute_fn": QWEN_EP},)),
        )

    def test_rendered_component_alone_declares_no_external_state(self):
        """A plan that renders a component but inlines no strategy body has none."""
        self.assertEqual(interface_component_class("qwen3_moe"), "GQAAttention")
        self.assertEqual(external_state_classes_for("qwen3_moe", ()), ())

    def test_deepseek_ep_spec_equals_its_archetype(self):
        archetype = moe_archetype(*DEEPSEEK_FP)
        spec = strategy_spec_for(DEEPSEEK_EP, "deepseek_v3")
        self.assertEqual(spec.kind, archetype.kind)
        self.assertEqual(spec.target_class, archetype.target_class)
        self.assertEqual(spec.strip_boundary_subpatterns, archetype.strip_boundary_subpatterns)
        self.assertIn(f"router_kind={archetype.router_kind!r}", spec.body_template)
        self.assertEqual(
            external_state_classes_for("deepseek_v3", ({"local_compute_fn": DEEPSEEK_EP},)),
            moe_external_state_classes(*DEEPSEEK_FP),
        )

    def test_replacement_specs_come_from_the_family_factories(self):
        """The component and its wiring are read from the factory, never declared."""
        norm = replacement_spec_for(QWEN_NORM, "qwen3_moe", module_type=QWEN_NORM_TYPE)
        self.assertEqual((norm.old_ctor, norm.new_ctor), ("Qwen3MoeRMSNorm", "RMSNorm"))
        self.assertEqual(norm.mode, "wrap_source")

        attention = replacement_spec_for(
            QWEN_ATTENTION, "qwen3_moe", module_type=QWEN_ATTENTION_TYPE
        )
        self.assertEqual(attention.old_ctor, "Qwen3MoeAttention")
        self.assertEqual(attention.new_ctor, "GQAAttention")
        self.assertFalse(attention.remove_class)
        self.assertIn(
            "attention_interface=run_qwen3_moe_flash_attention", attention.keyword_args
        )


if __name__ == "__main__":
    unittest.main()

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
"""S3: a missing MoE ``local_compute_fn._target_`` is inferred from structure.

A plan override whose match resolves to a MoE boundary carries its EP compute
factory either explicitly (``local_compute_fn._target_`` — the escape hatch) or
implicitly via structure detection. These two write paths must be equivalent:
the inferred ``_target_`` written into meta must equal the explicit one, and
the generated artifact bytes must be identical. Recognition never guesses — an
unrecognized structure raises with the escape-hatch hint.
"""

import unittest

import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from hyper_parallel.codegen import manager
from hyper_parallel.distributed.expert_parallel.structure import (
    UnsupportedModuleStructure,
)

QWEN_ROOT = "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
EP_PATH = QWEN_ROOT + "qwen3moe_ep_compute_fn"


def _stub_factory(**kwargs):
    """An inert callable — ``Target.to_dict`` only serializes the path."""
    return None


def _holder_with(block: nn.Module) -> nn.Module:
    holder = nn.Module()
    holder.layers = nn.Module()
    holder.layers.mlp = block
    return holder


def _holder_with_layers(blocks: list[nn.Module]) -> nn.Module:
    """A decoder-shaped holder: ``layers.<i>.mlp`` per block."""
    holder = nn.Module()
    layers = nn.ModuleList()
    for block in blocks:
        layer = nn.Module()
        layer.mlp = block
        layers.append(layer)
    holder.layers = layers
    return holder


class TestPlanTargetInference(unittest.TestCase):
    """S3 mapping and equivalence over real Qwen3-MoE source contracts."""

    def _qwen3_block(self) -> Qwen3MoeSparseMoeBlock:
        with torch.device("meta"):
            return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig())

    def test_fingerprint_registry_exposes_qwen3_ep_factory(self):
        """The Qwen3-MoE fingerprint resolves to its model-adapter factory."""
        factory = manager._ep_factory_table()[
            ("topk_router_module", "none", "batched_parameters")
        ]
        self.assertEqual(factory, EP_PATH)

    def test_omitted_target_infers_exact_explicit_target(self):
        """An implicit (no ``_target_``) override writes the same meta target."""
        from hyper_parallel.trainer.config import (  # pylint: disable=C0415,import-outside-toplevel
            PlanOverride,
            Target,
            entries_to_plan_overrides,
        )

        explicit = entries_to_plan_overrides(
            [
                PlanOverride(
                    match="layers.mlp",
                    when="ep",
                    region_dispatch=False,
                    local_compute_fn=Target(_stub_factory, target_path=EP_PATH),
                )
            ],
            ep_size=2,
        )
        omitted = entries_to_plan_overrides(
            [PlanOverride(match="layers.mlp", when="ep", region_dispatch=False)],
            ep_size=2,
        )

        manager._fill_inferred_ep_targets(_holder_with(self._qwen3_block()), omitted)

        explicit_dict = explicit["layers.mlp"].local_compute_fn.to_dict()
        inferred_dict = omitted["layers.mlp"].local_compute_fn.to_dict()
        self.assertEqual(inferred_dict["_target_"], EP_PATH)
        # The resolved meta ``_target_`` must be byte-identical across the two
        # write paths.
        self.assertEqual(inferred_dict, explicit_dict)

    def test_inferred_and_explicit_produce_identical_artifact(self):
        """Both write paths resolve the same inlined EP body (byte-identical).

        The generated artifact inlines the EP forwarded strategy body chosen by
        ``strategy_spec(target, model_type)``. The inferred ``_target_`` and the
        explicit one must select the *same* ``StrategySpec`` — its
        ``body_template`` is the exact text inlined into the artifact, so equal
        templates prove byte-identical output without a distributed run.
        """
        from hyper_parallel.codegen.inline.specs import strategy_spec  # pylint: disable=C0415
        from hyper_parallel.trainer.config import (  # pylint: disable=C0415,import-outside-toplevel
            PlanOverride,
            entries_to_plan_overrides,
        )

        overrides = entries_to_plan_overrides(
            [PlanOverride(match="layers.mlp", when="ep", region_dispatch=False)],
            ep_size=2,
        )
        manager._fill_inferred_ep_targets(_holder_with(self._qwen3_block()), overrides)
        inferred_target = overrides["layers.mlp"].local_compute_fn.to_dict()["_target_"]

        self.assertEqual(inferred_target, EP_PATH)
        inferred_spec = strategy_spec(inferred_target, "qwen3_moe")
        explicit_spec = strategy_spec(EP_PATH, "qwen3_moe")
        self.assertIsNotNone(inferred_spec)
        self.assertIs(inferred_spec, explicit_spec)
        # The body_template is the exact EP-forward text inlined into the
        # artifact; assert on a real marker to prove it is the routed body.
        self.assertIn("get_parallel_state()", inferred_spec.body_template)
        # The dispatch literal has been retired into the framework-generic
        # ``moe_ep_forward``; the inlined body is now a thin call to it.
        self.assertIn("moe_ep_forward", inferred_spec.body_template)

    def test_non_moe_override_is_left_untouched(self):
        """An override that hits no MoE boundary gains no compute injection."""
        from hyper_parallel.trainer.config import (  # pylint: disable=C0415,import-outside-toplevel
            PlanOverride,
            entries_to_plan_overrides,
        )

        overrides = entries_to_plan_overrides(
            [PlanOverride(match="layers.0", when="ep", region_dispatch=False)],
            ep_size=2,
        )
        manager._fill_inferred_ep_targets(_holder_with(self._qwen3_block()), overrides)
        spec = overrides["layers.0"]
        self.assertIsNone(spec.local_compute_fn)

    def test_homogeneous_layers_share_one_inferred_factory(self):
        """Several structurally identical MoE layers are not an ambiguity.

        Regression: the scan used to compare ``Target`` objects, and since each
        target wraps a fresh closure (``Target`` defines no equality) any model
        with two MoE boundaries matched by one glob looked ambiguous.
        """
        from hyper_parallel.trainer.config import (  # pylint: disable=C0415,import-outside-toplevel
            PlanOverride,
            entries_to_plan_overrides,
        )

        overrides = entries_to_plan_overrides(
            [PlanOverride(match="layers.*.mlp", when="ep", region_dispatch=False)],
            ep_size=2,
        )
        holder = _holder_with_layers([self._qwen3_block(), self._qwen3_block()])
        manager._fill_inferred_ep_targets(holder, overrides)

        self.assertEqual(
            overrides["layers.*.mlp"].local_compute_fn.to_dict()["_target_"], EP_PATH
        )

    def test_conflicting_factories_still_raise(self):
        """Two different factories behind one glob must stay a hard failure."""
        from hyper_parallel.trainer.config import (  # pylint: disable=C0415,import-outside-toplevel
            PlanOverride,
            Target,
            entries_to_plan_overrides,
        )

        paths = iter(["some.pkg.first_factory", "some.pkg.second_factory"])
        original = manager._import_ep_target_for
        manager._import_ep_target_for = lambda module, match: Target(
            _stub_factory, target_path=next(paths)
        )
        self.addCleanup(setattr, manager, "_import_ep_target_for", original)

        overrides = entries_to_plan_overrides(
            [PlanOverride(match="layers.*.mlp", when="ep", region_dispatch=False)],
            ep_size=2,
        )
        holder = _holder_with_layers([self._qwen3_block(), self._qwen3_block()])
        with self.assertRaisesRegex(UnsupportedModuleStructure, "different EP compute"):
            manager._fill_inferred_ep_targets(holder, overrides)

    def test_unrecognized_structure_raises_with_escape_hatch(self):
        """An unknown MoE structure must fail hard, never guess a factory."""
        block = self._qwen3_block()
        block.shared_expert = nn.Linear(4, 4)  # ambiguous shared branch
        with self.assertRaisesRegex(UnsupportedModuleStructure, "shared expert"):
            manager._infer_ep_compute_for_match(_holder_with(block), "layers.mlp")


if __name__ == "__main__":
    unittest.main()

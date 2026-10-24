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
"""Gated shared-expert MoE structures resolve an EP compute factory.

Regression guard for the qwen2_moe fix: the structural fingerprint
``("topk_router_module", "gated", "batched_parameters")`` (shared_expert +
shared_expert_gate pair) had no archetype table entry, so EP auto-inference
failed and users had to declare an explicit ``local_compute_fn``.
"""

import unittest

from hyper_parallel.distributed.expert_parallel.archetypes import (
    ep_compute_factory_for,
    moe_archetype,
    moe_archetypes,
)

GATED_FP = ("topk_router_module", "gated", "batched_parameters")
DEEPSEEK_FACTORY = (
    "hyper_parallel.distributed.expert_parallel.recipes."
    "deepseekv3_ep_compute_fn"
)


class TestGatedArchetype(unittest.TestCase):
    """The gated fingerprint maps to the routed-EP DeepSeek-V3 factory."""

    def test_lookup_returns_entry(self):
        archetype = moe_archetype(*GATED_FP)
        self.assertIsNotNone(archetype)
        self.assertEqual(archetype.shared, "gated")
        self.assertEqual(archetype.kind, "moe_ep_routed")
        self.assertEqual(archetype.router_kind, "topk_router_module")
        self.assertEqual(archetype.compute_factory, DEEPSEEK_FACTORY)

    def test_strip_patterns_use_singular_shared_expert(self):
        archetype = moe_archetype(*GATED_FP)
        # Qwen2Moe freezes the shared branch as `shared_expert` (singular),
        # unlike DeepSeek's `shared_experts`.
        self.assertIn("shared_expert", archetype.strip_boundary_subpatterns)
        self.assertNotIn("shared_experts", archetype.strip_boundary_subpatterns)

    def test_compute_factory_helper(self):
        self.assertEqual(ep_compute_factory_for(*GATED_FP), DEEPSEEK_FACTORY)

    def test_existing_fingerprints_unchanged(self):
        table = moe_archetypes()
        self.assertEqual(len(table), 3)
        self.assertIn(("topk_router_module", "none", "batched_parameters"), table)
        self.assertIn(("topk_router_module", "additive", "batched_parameters"), table)

    def test_unknown_fingerprint_still_none(self):
        self.assertIsNone(moe_archetype("nope", "none", "batched_parameters"))


if __name__ == "__main__":
    unittest.main()

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
"""Sibling-module EP hosts defer to the runtime local-region skeleton.

Regression guard for the kimi_k3 two-file model: the EP inline patch
targeted ``KimiSparseMoeBlock``, a class defined in a sibling module
(``modeling_kimi_linear.py``) that the bundle copies verbatim, so the
main-file source index could not find it and generation raised.  A
body-bearing host the main source does not define now skips the forward
extract (and the boundary strip) and defers to the runtime local-region
skeleton (``hp_bind_compute``) — the same channel an imported-class
boundary uses — while an in-source host on the same rule keeps the
historical inline path.
"""

import ast
import unittest
from types import SimpleNamespace

from hyper_parallel.codegen.inline.ir import InlineRule
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.pipeline import render_inline_modeling
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches

DEEPSEEK_EP = (
    "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn"
)
MODEL_TYPE = "deepseek_v33"

# A two-file modeling source: the MoE block class lives in the sibling
# module the bundle copies verbatim, so only the attention class is a
# main-source class the artifact may rewrite.
SIBLING_SOURCE = """
from .modeling_sibling import SiblingMoeBlock


class InSourceAttention:
    def forward(self, hidden_states):
        return hidden_states
"""


def _rules(*matches):
    return tuple(
        InlineRule((match,), local_compute_target=DEEPSEEK_EP)
        for match in matches
    )


def _meta(boundary_classes, external_state_classes, param_plan):
    """Mirror the freeze-time meta shape: ``external_state_classes`` is what
    ``_record_inline_declarations`` recorded BEFORE the source was indexed."""
    return SimpleNamespace(
        module_overrides=[],
        injections=[
            {"match": fqn, "local_compute_fn": DEEPSEEK_EP}
            for fqn in boundary_classes
        ],
        param_plan=param_plan,
        boundary_classes=dict(boundary_classes),
        external_state_classes=list(external_state_classes),
        frozen_sharded_params=[],
        source={},
    )


class TestSiblingHostStrategyPatches(unittest.TestCase):
    """build_strategy_patches splits hosts by main-source membership."""

    def test_fully_sibling_host_emits_no_forward_extract(self):
        rules = _rules("lm.model.layers.*.block_sparse_moe")
        boundary = {
            "lm.model.layers.0.block_sparse_moe": "SiblingMoeBlock",
            "lm.model.layers.1.block_sparse_moe": "SiblingMoeBlock",
        }
        patches = build_strategy_patches(
            rules, MODEL_TYPE, boundary, None, {"InSourceAttention"}
        )
        self.assertEqual(patches.forward_extracts, [])

    def test_mixed_rule_splits_inlineable_from_sibling_hosts(self):
        rules = _rules("lm.model.layers.*.moe")
        boundary = {
            "lm.model.layers.0.moe": "SiblingMoeBlock",
            "lm.model.layers.1.moe": "InSourceBlock",
        }
        patches = build_strategy_patches(
            rules, MODEL_TYPE, boundary, None, {"InSourceBlock"}
        )
        self.assertEqual(
            [(patch.class_name, patch.method_name) for patch in patches.forward_extracts],
            [("InSourceBlock", "forward")],
        )

    def test_none_source_classes_keeps_legacy_trust(self):
        rules = _rules("lm.model.layers.*.moe")
        boundary = {
            "lm.model.layers.0.moe": "SiblingMoeBlock",
            "lm.model.layers.1.moe": "InSourceBlock",
        }
        patches = build_strategy_patches(rules, MODEL_TYPE, boundary)
        self.assertEqual(
            sorted(patch.class_name for patch in patches.forward_extracts),
            ["InSourceBlock", "SiblingMoeBlock"],
        )

    def test_deferred_host_without_a_matching_boundary_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "matched no boundary"):
            build_strategy_patches(
                _rules("lm.model.layers.*.router"),
                MODEL_TYPE,
                {"lm.model.layers.0.moe": "SiblingMoeBlock"},
                None,
                {"InSourceAttention"},
            )


class TestNormalizeKeepsSiblingBoundaries(unittest.TestCase):
    """normalize_inline_meta keeps sibling-host boundaries compiled."""

    def test_sibling_boundary_and_sub_boundaries_stay_compiled(self):
        boundary = {"lm.model.layers.0.block_sparse_moe": "SiblingMoeBlock"}
        param_plan = {
            "lm.model.layers.0.block_sparse_moe": {"is_boundary": True},
            "lm.model.layers.0.block_sparse_moe.experts": {"is_boundary": True},
            "lm.model.layers.0.block_sparse_moe.shared_experts": {"is_boundary": True},
        }
        meta = _meta(boundary, ["SiblingMoeBlock"], param_plan)
        normalize_inline_meta(
            meta,
            _rules("lm.model.layers.*.block_sparse_moe"),
            MODEL_TYPE,
            {"InSourceAttention"},
        )
        self.assertEqual(
            meta.boundary_classes,
            {"lm.model.layers.0.block_sparse_moe": "SiblingMoeBlock"},
        )
        for entry in meta.param_plan.values():
            self.assertTrue(entry["is_boundary"])
        self.assertEqual(meta.external_state_classes, [])

    def test_mixed_rule_strips_only_the_in_source_host(self):
        boundary = {
            "lm.model.layers.0.moe": "SiblingMoeBlock",
            "lm.model.layers.1.moe": "InSourceBlock",
        }
        param_plan = {
            "lm.model.layers.0.moe": {"is_boundary": True},
            "lm.model.layers.0.moe.shared_experts": {"is_boundary": True},
            "lm.model.layers.1.moe": {"is_boundary": True},
            "lm.model.layers.1.moe.shared_experts": {"is_boundary": True},
        }
        meta = _meta(boundary, ["SiblingMoeBlock", "InSourceBlock"], param_plan)
        normalize_inline_meta(
            meta, _rules("lm.model.layers.*.moe"), MODEL_TYPE, {"InSourceBlock"}
        )
        self.assertEqual(
            meta.boundary_classes,
            {"lm.model.layers.0.moe": "SiblingMoeBlock"},
        )
        self.assertTrue(meta.param_plan["lm.model.layers.0.moe"]["is_boundary"])
        self.assertTrue(
            meta.param_plan["lm.model.layers.0.moe.shared_experts"]["is_boundary"]
        )
        self.assertFalse(meta.param_plan["lm.model.layers.1.moe"]["is_boundary"])
        self.assertFalse(
            meta.param_plan["lm.model.layers.1.moe.shared_experts"]["is_boundary"]
        )
        self.assertEqual(meta.external_state_classes, ["InSourceBlock"])

    def test_in_source_host_keeps_historical_strip(self):
        boundary = {"lm.model.layers.1.moe": "InSourceBlock"}
        param_plan = {
            "lm.model.layers.1.moe": {"is_boundary": True},
            "lm.model.layers.1.moe.shared_experts": {"is_boundary": True},
        }
        meta = _meta(boundary, ["InSourceBlock"], param_plan)
        normalize_inline_meta(
            meta, _rules("lm.model.layers.*.moe"), MODEL_TYPE, {"InSourceBlock"}
        )
        self.assertEqual(meta.boundary_classes, {})
        self.assertFalse(meta.param_plan["lm.model.layers.1.moe"]["is_boundary"])
        self.assertFalse(
            meta.param_plan["lm.model.layers.1.moe.shared_experts"]["is_boundary"]
        )
        self.assertEqual(meta.external_state_classes, ["InSourceBlock"])


class TestTwoFileModelRender(unittest.TestCase):
    """render_inline_modeling carries the kk3 shape end to end."""

    def test_sibling_moe_renders_and_keeps_its_boundary(self):
        meta = _meta(
            {"lm.model.layers.0.block_sparse_moe": "SiblingMoeBlock"},
            ["SiblingMoeBlock"],
            {"lm.model.layers.0.block_sparse_moe": {"is_boundary": True}},
        )
        rendered = render_inline_modeling(SIBLING_SOURCE, meta, MODEL_TYPE)
        ast.parse(rendered)
        # The sibling class is imported, never redefined: the artifact only
        # edits the main file, and the import must survive untouched.
        self.assertIn("from .modeling_sibling import SiblingMoeBlock", rendered)
        self.assertNotIn("class SiblingMoeBlock", rendered)
        # The runtime skeleton needs the compiled boundary, and the
        # external-state record (which would strip it) is dropped.
        self.assertEqual(
            meta.boundary_classes,
            {"lm.model.layers.0.block_sparse_moe": "SiblingMoeBlock"},
        )
        self.assertTrue(
            meta.param_plan["lm.model.layers.0.block_sparse_moe"]["is_boundary"]
        )
        self.assertEqual(meta.external_state_classes, [])

    def test_in_source_moe_still_receives_the_inline_ep_body(self):
        source = """
class InSourceBlock:
    def forward(self, hidden_states):
        return hidden_states
"""
        meta = _meta(
            {"lm.model.layers.0.moe": "InSourceBlock"},
            ["InSourceBlock"],
            {"lm.model.layers.0.moe": {"is_boundary": True}},
        )
        rendered = render_inline_modeling(source, meta, MODEL_TYPE)
        ast.parse(rendered)
        self.assertIn("class InSourceBlock", rendered)
        self.assertIn("ep_enable", rendered)
        self.assertEqual(meta.external_state_classes, ["InSourceBlock"])
        self.assertEqual(meta.boundary_classes, {})


if __name__ == "__main__":
    unittest.main()

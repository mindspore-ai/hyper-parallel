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
"""Published V4.1 recipes use family-owned layouts at their declared topology."""

import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from hyper_parallel.components.modules.module_family import FamilyShardingContext, get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed._builder.family_layout import collect_owned_layouts, validate_owned_layouts
from tests.common.engram_family import small_config


SOURCE_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).with_name("fixtures") / "deepseek_v41_recipes.yaml"
RECIPES = (
    "examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml",
    "examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml",
    "examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_1M online_128die.yaml",
    "hyper_parallel/models/deepseek_v41/tools/nd_train_deepseek_v41_timing.yaml",
    ".agent/skills/sapp-parallel-strategy/templates/deepseek_v41.yaml",
)


def _family(rule: dict, config: SimpleNamespace, context: dict) -> torch.nn.Module:
    """Construct the exact public replacement requested by a recipe."""
    arguments = dict(rule["replace"]["implementation"])
    module, _, name = arguments.pop("_target_").rpartition(".")
    factory = getattr(importlib.import_module(module), name)
    return factory(config=config, context=context, **arguments)


class TestDeepseekV41Recipes(unittest.TestCase):
    """Exercise real recipe targets and final ownership without distributed initialization."""

    def test_family_layouts_cover_cp_and_ep_without_legacy_injection(self) -> None:
        """Shared-family recipes build complete ownership using their declared axis sizes."""
        for relative in RECIPES[3:]:
            with self.subTest(recipe=relative):
                recipe = self._load_recipe(relative)
                context = recipe["accelerator"]
                rules = recipe["plan_overrides"]
                selected = [rule for rule in rules if "_hc" in str(rule["match"])
                            or "engram" in str(rule["match"])]
                self.assertEqual(len(selected), 2)
                engram_rule = next(rule for rule in selected if rule["match"] == "model.layers.*.engram")
                mhc_rule = next(rule for rule in selected if rule is not engram_rule)
                for rule in selected:
                    self.assertEqual(set(rule), {"match", "replace"})
                with torch.device("meta"):
                    layer = torch.nn.Module()
                    layer.engram = _family(engram_rule, small_config(max(16, context["ep_size"])), context)
                    settings = SimpleNamespace(num_stream=4, hidden_size=4, sinkhorn_iters=20,
                                               hc_eps=1e-6, norm_eps=1e-6, use_gamma=False, init_scale=1.0)
                    layer.attn_hc = _family(mhc_rule, settings, context)
                    layer.ffn_hc = _family(mhc_rule, settings, context)
                    model = torch.nn.Module()
                    model.model = torch.nn.Module()
                    model.model.layers = torch.nn.ModuleList([layer])
                sharding_context = FamilyShardingContext(
                    ("tp", "cp", "ep"), {axis: context[f"{axis}_size"] for axis in ("tp", "cp", "ep")},
                    {name: tuple(parameter.shape) for name, parameter in model.named_parameters()},
                    context["sequence_parallel"],
                )
                owned = collect_owned_layouts(model, sharding_context)
                validate_owned_layouts(model, SimpleNamespace(modules=owned.specs), owned, sharding_context)
                self.assertEqual(len(owned.specs), 3)
                self.assertIsNotNone(get_family_build_info(layer.engram))
                self._assert_mhc_layouts(layer, owned, context)

    def test_training_recipes_preserve_fused_shifted_mhc(self) -> None:
        """Published fused recipes retain their legacy implementation and explicit layouts."""
        for relative in RECIPES[:3]:
            with self.subTest(recipe=relative):
                rules = self._load_recipe(relative)["plan_overrides"]
                mhc = [rule for rule in rules if "_hc" in str(rule["match"])]
                replacement = next(rule for rule in mhc if "replace" in rule)
                self.assertEqual(replacement["replace"]["implementation"]["_target_"],
                                 "hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc."
                                 "DeepseekV41HyperMegaMhc")
                base = next(rule for rule in mhc if "params" in rule)
                self.assertFalse(base["region_dispatch"])
                self.assertEqual(set(base["params"]), {"fn", "base", "scale"})
                self.assertEqual(base["in_dst"]["hidden_streams"]["cp"], "shard(1)")
                sequence = next(rule for rule in mhc if rule.get("when") == "sequence_parallel")
                self.assertEqual(sequence["in_dst"]["hidden_streams"]["tp"], "shard(1)")

    def test_training_engram_uses_complete_family_layout(self) -> None:
        """Host Engram owns its EP table rows without an obsolete forward injection."""
        for relative in RECIPES[:3]:
            with self.subTest(recipe=relative):
                recipe = self._load_recipe(relative)
                settings = recipe["accelerator"]
                rules = [rule for rule in recipe["plan_overrides"] if "engram" in str(rule["match"])]
                self.assertEqual(len(rules), 1)
                self.assertEqual(set(rules[0]), {"match", "replace"})
                with torch.device("meta"):
                    member = _family(rules[0], small_config(max(16, settings["ep_size"])), settings)
                context = FamilyShardingContext(
                    ("tp", "cp", "ep"), {axis: settings[f"{axis}_size"] for axis in ("tp", "cp", "ep")},
                    {}, settings["sequence_parallel"],
                )
                owned = collect_owned_layouts(torch.nn.Sequential(member), context)
                layout = owned.specs["0"]
                self.assertEqual(set(layout.params), set(dict(member.named_parameters())))
                self.assertEqual(layout.params["embed.weight"]["ep"], Shard(0))

    def _load_recipe(self, relative: str) -> dict:
        """Load packaged recipe fields and reject stale snapshots in a source checkout."""
        fixtures = yaml.safe_load(FIXTURES.read_text(encoding="utf-8"))
        self.assertEqual(set(fixtures), set(RECIPES))
        recipe = fixtures[relative]
        source = SOURCE_ROOT / relative
        if source.is_file():
            published = yaml.safe_load(source.read_text(encoding="utf-8"))
            expected = {key: published[key] for key in ("accelerator", "plan_overrides")}
            self.assertEqual(recipe, expected, f"Recipe fixture differs from source: {relative}")
        return recipe

    def _assert_mhc_layouts(self, layer, owned, context) -> None:
        """Check the two independent coefficient boundaries against the recipe topology."""
        for name in ("attn_hc", "ffn_hc"):
            member = getattr(layer, name)
            self.assertEqual(get_family_build_info(member).resolved.variant, "local")
            layout = owned.specs[f"model.layers.0.{name}"]
            self.assertEqual(set(layout.params), {"fn", "base", "scale"})
            self.assertTrue(all(value == Replicate() for named in layout.params.values()
                                for value in named.values()))
            self.assertEqual(layout.in_dst["hidden_streams"]["cp"],
                             Shard(1) if context["cp_size"] > 1 else Replicate())

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
"""Tests for the declarative DeepSeek recipe and generic optimizer builder."""

import copy
import unittest

import torch
from torch import nn
import yaml

from hyper_parallel.components.optim import ComposedOptimizer, MultiLRScheduler, Sinkhorn
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.trainer.config.optimization import OptimizerConfig
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.common.mark_utils import arg_mark


class _RecipeModel(nn.Module):
    """Expose the actual recipe names and head widths without a full model."""

    def __init__(self) -> None:
        """Build only the parameter roles consumed by the optimizer recipe."""
        super().__init__()
        attention = nn.Module()
        attention.q_b_proj = nn.Linear(4, 1024, bias=False)
        attention.kv_proj = nn.Linear(4, 512, bias=False)
        attention.indexer = nn.Module()
        attention.indexer.q_b_proj = nn.Linear(4, 256, bias=False)
        attention.indexer.wk = nn.Linear(4, 128, bias=False)
        layer = nn.Module()
        layer.self_attn = attention
        layer.engram = nn.Module()
        layer.engram.embed = nn.Embedding(8, 4)
        layer.engram.q_weight = nn.Parameter(torch.ones(2, 4))
        layer.engram.k_weight = nn.Parameter(torch.ones(2, 4))
        layer.norm = nn.LayerNorm(4)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([layer])
        self.model.vision = nn.Module()
        vision_block = nn.Module()
        vision_block.attn = nn.Module()
        vision_block.attn.wqkv = nn.Linear(4, 12)
        vision_block.attn.wo = nn.Linear(4, 4)
        vision_block.norm = nn.LayerNorm(4)
        self.model.vision.blocks = nn.ModuleList([vision_block])
        self.model.aligner = nn.Module()
        self.model.aligner.w1 = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)


def _composition_config():
    """Keep the builder contract independent of source-only training examples."""
    return yaml.safe_load(r"""
optimizer:
  _target_: hyper_parallel.components.optim.ComposedOptimizer
  fp32_main_params: true
  muon:
    lr: 2.6e-4
    weight_decay: 0.1
    momentum: 0.95
    nesterov: true
    ns_steps: 5
    ns_variant: asym5
    matched_adamw_rms: 0.18
    head_wise: true
    head_wise_patterns:
      '^model\.layers\.[0-9]+\.self_attn\.(q_b_proj|kv_proj)\.weight$': 512
      '^model\.layers\.[0-9]+\.self_attn\.indexer\.(q_b_proj|wk)\.weight$': 128
  sinkhorn:
    lr: 2.6e-4
    momentum: 0.95
    nesterov: true
    steps: 11
    tau: 1.0e-3
    eps: 1.0e-20
    correction: 0.18
    param_groups:
      - param_patterns: ['\.engram\.embed\.weight$']
        lr: 1.3e-3
  adamw:
    lr: 2.6e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
    eps: 1.0e-20
    param_patterns: ['\.engram\.[qk]_weight$']
lr_scheduler:
  _target_: hyper_parallel.components.optim.MultiLRScheduler
  lr_decay_style: cosine
  lr_config:
    lr: 3.0e-5
    lr_warmup_steps: 1
    min_lr: 0.0
""")


class TestComposedOptimizer(unittest.TestCase):
    """Exercise the public YAML target, family rates, and parameter routing."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_explicit_group_updates_and_config_immutability(self):
        """
        Feature: Explicit optimizer groups in YAML builders.
        Description: Compare absolute group rates with manually constructed runtime groups.
        Expectation: Updates match, default decay survives, and caller options stay unchanged.
        """
        model = nn.ModuleList([
            nn.Sequential(nn.Embedding(4, 3), nn.Linear(3, 3), nn.LayerNorm(3)) for _ in range(2)
        ])
        reference = copy.deepcopy(model)
        configs = {name: {"lr": 0.001, "param_groups": [{"param_patterns": [r"^1\."], "lr": 0.003}]}
                   for name in ("muon", "sinkhorn", "adamw")}
        original = copy.deepcopy(configs)
        optimizer = ComposedOptimizer(model, **configs).get_optimizer()
        baseline = get_hyper_optimizer(reference, muon={}, sinkhorn={}, adamw={})
        runtime_groups = {}
        for family, leaf in baseline.optimizers_dict.items():
            runtime_groups[f"{family}_params"] = [
                {**group, "params": [param], "lr": 0.003 if param.model_name.startswith("1.") else 0.001}
                for group in leaf.param_groups for param in group["params"]
            ]
        baseline = get_hyper_optimizer(reference, **runtime_groups)
        for _ in range(3):
            for param, ref in zip(model.parameters(), reference.parameters()):
                param.grad = torch.randn_like(param)
                ref.grad = param.grad.clone()
            optimizer.step()
            baseline.step()
        for param, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param, ref)
        self.assertEqual(configs, original)
        for leaf in optimizer:
            for group in leaf.param_groups:
                for param in group["params"]:
                    self.assertEqual(group["lr"], 0.003 if param.model_name.startswith("1.") else 0.001)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_explicit_group_selectors_reject_invalid_ownership(self):
        """
        Feature: Explicit parameter-group ownership validation.
        Description: Try unmatched, wrong-family, overlapping and malformed group declarations.
        Expectation: Invalid selectors fail before constructing optimizers.
        """
        for groups in ([{"param_patterns": ["missing"], "lr": 0.001}],
                       [{"param_patterns": ["q_b_proj"], "lr": 0.001}],
                       [{"param_patterns": ["embed"]}, {"param_patterns": ["embed"]}],
                       [{"lr": 0.001}], [{"param_patterns": ["embed"], "params": []}],
                       {"param_patterns": ["embed"]}, [0.001]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                ComposedOptimizer(_RecipeModel(), muon={}, sinkhorn={"param_groups": groups}, adamw={})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_composition_routing_and_scheduler(self):
        """
        Feature: Declarative report optimizer recipe.
        Description: Build a YAML target against representative text and vision parameters.
        Expectation: Families, head widths, norm decay and explicit group rates survive scheduling.
        """
        recipe = _composition_config()
        config = resolve_component(recipe["optimizer"], annotation=OptimizerConfig, path="optimizer")
        model = _RecipeModel()
        optimizer = config.target.build(model=model).get_optimizer()
        self.assertEqual(set(optimizer.optimizers_dict), {"muon", "sinkhorn", "adamw"})
        rows = {param.model_name: (family, group) for family, leaf in optimizer.optimizers_dict.items()
                for group in leaf.param_groups for param in group["params"]}
        self.assertEqual(len(rows), len(list(model.parameters())))
        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.layers.0.engram.embed.weight"):
            self.assertEqual(rows[name][0], "sinkhorn")
        self.assertEqual(rows["model.embed_tokens.weight"][1]["lr"], 2.6e-4)
        self.assertEqual(rows["model.layers.0.engram.embed.weight"][1]["lr"], 1.3e-3)
        for name in ("q_weight", "k_weight"):
            family, group = rows[f"model.layers.0.engram.{name}"]
            self.assertEqual(family, "adamw")
            self.assertEqual(group["weight_decay"], 0.1)
        self.assertEqual(rows["model.layers.0.norm.weight"][1]["weight_decay"], 0.1)
        self.assertEqual(rows["model.layers.0.norm.bias"][1]["weight_decay"], 0.0)
        for name in ("model.vision.blocks.0.attn.wqkv", "model.vision.blocks.0.attn.wo", "model.aligner.w1"):
            self.assertEqual(rows[f"{name}.weight"][0], "muon")
            self.assertEqual(rows[f"{name}.bias"][0], "adamw")
            self.assertEqual(rows[f"{name}.bias"][1]["weight_decay"], 0.0)
        self.assertEqual(rows["model.vision.blocks.0.norm.weight"][0], "adamw")
        self.assertEqual(rows["model.vision.blocks.0.norm.weight"][1]["weight_decay"], 0.1)
        muon = optimizer.optimizers_dict["muon"]
        # Verify the resolved geometry that drives the already-tested NS path.
        dims = muon._head_wise_dims  # pylint: disable=protected-access
        self.assertEqual(dims[model.model.layers[0].self_attn.q_b_proj.weight], 512)
        self.assertEqual(dims[model.model.layers[0].self_attn.indexer.q_b_proj.weight], 128)
        scheduler_config = recipe["lr_scheduler"]
        scheduler_config.pop("_target_")
        scheduler = MultiLRScheduler(optimizer=optimizer, train_iters=5, **scheduler_config).get_lr_scheduler()
        before = {name: param.detach().clone() for name, param in model.named_parameters()}
        for _ in range(3):
            for param in model.parameters():
                param.grad = torch.randn_like(param)
            optimizer.step()
            scheduler.step()
            regular_lr = rows["model.embed_tokens.weight"][1]["lr"]
            engram_lr = rows["model.layers.0.engram.embed.weight"][1]["lr"]
            self.assertGreater(regular_lr, 0)
            self.assertAlmostEqual(engram_lr / regular_lr, 5.0)
        for name, param in model.named_parameters():
            self.assertTrue(torch.isfinite(param).all())
            self.assertFalse(torch.equal(before[name], param))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_tied_alias_routing_and_config_immutability(self):
        """
        Feature: Tied parameters and immutable options.
        Description: Select a tied embedding through its output-head alias.
        Expectation: The tied parameter is optimized once and caller options remain intact.
        """
        model = nn.Module()
        model.embed = nn.Embedding(4, 3)
        model.lm_head = nn.Linear(3, 4, bias=False)
        model.lm_head.weight = model.embed.weight
        options = {"lr": 0.001, "param_groups": [{"param_patterns": r"^lm_head\.weight$", "lr": 0.005}]}
        original = copy.deepcopy(options)
        optimizer = ComposedOptimizer(model, sinkhorn=options).get_optimizer()
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        self.assertIs(optimizer.param_groups[0]["params"][0], model.embed.weight)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.005)
        self.assertEqual(options, original)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sinkhorn_builder_yaml_and_update(self):
        """
        Feature: Sinkhorn builder configuration and fallback.
        Description: Resolve the builders target and perform an actual mixed-family update.
        Expectation: Embeddings use Sinkhorn, other weights use AdamW, and updates are finite.
        """
        node = yaml.safe_load("""
_target_: hyper_parallel.components.optim.builders.Sinkhorn
sinkhorn_config:
  sinkhorn_lr: 0.001
  sinkhorn_steps: 3
adamw_config:
  adamw_lr: 0.002
""")
        config = resolve_component(node, annotation=OptimizerConfig, path="optimizer")
        model = _RecipeModel()
        optimizer = config.target.build(model=model).get_optimizer()
        self.assertEqual(set(optimizer.optimizers_dict), {"sinkhorn", "adamw"})
        rows = {param.model_name: family for family, leaf in optimizer.optimizers_dict.items()
                for group in leaf.param_groups for param in group["params"]}
        self.assertEqual(rows["model.embed_tokens.weight"], "sinkhorn")
        self.assertEqual(rows["lm_head.weight"], "sinkhorn")
        self.assertEqual(rows["model.layers.0.self_attn.q_b_proj.weight"], "adamw")
        before = torch.detach(model.model.embed_tokens.weight).clone()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        self.assertEqual(node["sinkhorn_config"], {"sinkhorn_lr": 0.001, "sinkhorn_steps": 3})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sinkhorn_explicit_matrix_selector(self):
        """
        Feature: Explicit Sinkhorn selection.
        Description: Select a non-embedding projection through a regex.
        Expectation: The selected matrix uses Sinkhorn and the input options are unchanged.
        """
        model = nn.Linear(4, 3)
        options = {"lr": 0.001, "param_patterns": ["^weight$"]}
        optimizer = Sinkhorn(options, {}, model).get_optimizer()
        self.assertIs(optimizer.optimizers_dict["sinkhorn"].param_groups[0]["params"][0], model.weight)
        self.assertEqual(options, {"lr": 0.001, "param_patterns": ["^weight$"]})

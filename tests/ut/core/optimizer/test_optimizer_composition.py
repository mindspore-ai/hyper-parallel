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

"""Routing, head-wise updates and three-family checkpoint integration."""

import copy
from types import SimpleNamespace
import unittest


import torch
from torch import nn

from hyper_parallel.components.modules.kimi_delta_attention import KimiRMSNormGated
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from tests.common.mark_utils import arg_mark


class TinyModel(nn.Module):
    """Independent Q/K, token table, prediction head, norm and bias parameters."""

    def __init__(self) -> None:
        """Build a small model with each optimizer parameter category."""
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_attention_heads=2)
        self.embed = nn.Embedding(9, 8)
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 9, bias=False)


def make_optimizer(model: nn.Module) -> ChainedOptimizer:
    """Create the report's three-family composition at a small test learning rate."""
    return get_hyper_optimizer(model, muon={"head_wise": True}, sinkhorn={}, adamw={})


class TestOptimizerComposition(unittest.TestCase):
    """Cover identity routing, main parameters, head scaling and checkpoint state."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_routing_and_tied_parameters(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Embedding aliases route once; norms decay and bias does not.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        model = TinyModel()
        model.lm_head.weight = model.embed.weight
        model.v_proj.weight.requires_grad_(False)
        optimizer = make_optimizer(model)
        self.assertEqual(optimizer.param_names_by_optimizer["sinkhorn"], ["embed.weight"])
        self.assertEqual(optimizer.param_names_by_optimizer["muon"], ["q_proj.weight", "k_proj.weight"])
        decays = {id(p): group["weight_decay"] for group in optimizer.optimizers_dict["adamw"].param_groups
                  for p in group["params"]}
        self.assertEqual(decays[id(model.q_proj.bias)], 0)
        self.assertEqual(decays[id(model.norm.weight)], 0.01)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_matrix_norm_weights_stay_with_adamw(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Normalization weights use AdamW even with multi-dimensional normalized shapes.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        model = nn.Module()
        model.norm = nn.LayerNorm((2, 4))
        model.gate_scale = nn.Parameter(torch.ones(4))
        optimizer = get_hyper_optimizer(model, muon={}, adamw={"weight_decay": 0.1})
        self.assertEqual(set(optimizer.optimizers_dict), {"adamw"})
        decays = {id(param): group["weight_decay"] for group in optimizer.param_groups for param in group["params"]}
        self.assertEqual(decays[id(model.norm.weight)], 0.1)
        self.assertEqual(decays[id(model.norm.bias)], 0)
        self.assertEqual(decays[id(model.gate_scale)], 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_gated_norm_weights_use_configured_decay(self):
        """
        Feature: Normalization weight decay.
        Description: Route a gated RMSNorm with default and explicit AdamW selection.
        Expectation: Zero gradients decay normalization weights but leave ordinary scales unchanged.
        """
        for options in ({}, {"param_patterns": [r"^norm\.weight$"]}):
            with self.subTest(options=options):
                model = nn.Module()
                model.norm = KimiRMSNormGated(4)
                model.gate_scale = nn.Parameter(torch.ones(4))
                optimizer = get_hyper_optimizer(
                    model, muon={}, adamw={"lr": 0.1, "weight_decay": 0.1, **options},
                )
                self.assertEqual(set(optimizer.optimizers_dict), {"adamw"})
                for param in model.parameters():
                    param.grad = torch.zeros_like(param)
                optimizer.step()
                torch.testing.assert_close(model.norm.weight, torch.full((4,), 0.99))
                torch.testing.assert_close(model.gate_scale, torch.ones(4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_regex_selection_and_errors(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Selectors override defaults but cannot claim a tied parameter twice.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        model = TinyModel()
        optimizer = get_hyper_optimizer(model, muon={}, sinkhorn={"param_patterns": [r"v_proj.weight$"]}, adamw={})
        self.assertIn("v_proj.weight", optimizer.param_names_by_optimizer["sinkhorn"])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            get_hyper_optimizer(model, muon={"param_patterns": [r"embed.weight$"]},
                                sinkhorn={"param_patterns": [r"embed.weight$"]}, adamw={})
        with self.assertRaisesRegex(ValueError, "No enabled optimizer"):
            get_hyper_optimizer(model, muon={}, sinkhorn={})
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            get_hyper_optimizer(model, muon_params=[], muon={})
        leaf = torch.optim.SGD([model.q_proj.weight], lr=0.1)
        with self.assertRaisesRegex(ValueError, "more than one"):
            ChainedOptimizer(model, {"one": leaf, "two": leaf})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_head_wise_matches_independent_heads(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Each selected head follows its own Muon trajectory and slice scaling.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        for head_dim in (1, 4):
            with self.subTest(head_dim=head_dim):
                torch.manual_seed(2)
                model = nn.Module()
                model.q_proj = nn.Linear(6, 8, bias=False)
                optimizer = get_hyper_optimizer(model, muon={"head_wise": True, "head_dim": head_dim})
                heads = [nn.Parameter(head.clone()) for head in model.q_proj.weight.detach().split(head_dim)]
                reference = Muon(heads)
                for _ in range(3):
                    gradient = torch.randn_like(model.q_proj.weight)
                    model.q_proj.weight.grad = gradient
                    for head, grad in zip(heads, gradient.split(head_dim)):
                        head.grad = grad.clone()
                    optimizer.step()
                    reference.step()
                    torch.testing.assert_close(model.q_proj.weight, torch.cat(heads))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_custom_head_patterns_and_validation(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Custom names and per-pattern widths work; ambiguous geometry fails early.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        model = nn.Linear(6, 8, bias=False)
        optimizer = get_hyper_optimizer(model, muon={"head_wise": True, "head_wise_patterns": {r"^weight$": 2}})
        self.assertEqual(optimizer.optimizer.head_wise_patterns, {r"^weight$": 2})
        with self.assertRaisesRegex(ValueError, "divisible"):
            get_hyper_optimizer(model, muon={"head_wise": True, "head_wise_patterns": {r"^weight$": 3}})
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            Muon([model.weight], head_wise=True, head_dim=4, reshape_fn=lambda name, tensor: [tensor])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_three_way_checkpoint_resume(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Flattened FQN checkpoints restore three disjoint optimizer families.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        torch.manual_seed(5)
        model = TinyModel()
        optimizer = make_optimizer(model)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        restored_model = copy.deepcopy(model)
        restored = make_optimizer(restored_model)
        restored.load_state_dict(checkpoint)
        for first, second in zip(model.parameters(), restored_model.parameters()):
            first.grad = torch.randn_like(first)
            second.grad = first.grad.clone()
        optimizer.step()
        restored.step()
        for first, second in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(first, second)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_main_parameter_replacement_retains_head_geometry(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Identity caches and Sinkhorn float32 states survive the mixed-precision wrapper.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        model = TinyModel().to(torch.bfloat16)
        chain = make_optimizer(model)
        optimizer = Float16OptimizerWithFloat16Params(chain, model)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()
        leaf = chain.optimizers_dict["sinkhorn"]
        for state in leaf.state.values():
            self.assertEqual(state["momentum_buffer"].dtype, torch.float32)
        self.assertEqual(chain.param_names_by_optimizer["muon"],
                         ["q_proj.weight", "k_proj.weight", "v_proj.weight"])

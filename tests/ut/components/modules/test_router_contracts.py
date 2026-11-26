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
"""Independent routing contracts, source ownership and model reuse tests."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from hyper_multicore.modules.mega_gate import MegaGate as CoreMegaGate

from examples.torch.module_family.config import GateConfig, MegaGateConfig
from hyper_parallel.components.modules import get_family_build_info
from hyper_parallel.components.modules.moe.router import (
    Gate, GateModule, MegaGateModule, routing_scores, select_topk_scores,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.components.modules.module_family import get_module_family
from tests.common.mark_utils import arg_mark


def _reference(weight, hidden, bias, scoring, top_k, scale, normalize):
    """Use explicit scoring and per-token Python selection, retaining differentiable unbiased weights."""
    logits = hidden.reshape(-1, weight.shape[-1]).float() @ weight.float().T
    if scoring == "sigmoid":
        scores = 1 / (1 + torch.exp(-logits))
    elif scoring == "softmax":
        shifted = logits - logits.max(-1, keepdim=True).values
        scores = shifted.exp() / shifted.exp().sum(-1, keepdim=True)
    else:
        scores = torch.log1p(logits.exp()).sqrt()
    adjusted = scores if bias is None else scores + bias
    rows = [sorted(range(weight.shape[0]), key=lambda index, current=row: current[index], reverse=True)[:top_k]
            for row in adjusted.detach().tolist()]
    indices = torch.tensor(rows, dtype=torch.long, device=hidden.device).reshape(-1, top_k)
    weights = scores.gather(1, indices)
    if normalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return logits, weights * scale, indices


class TestRouterContracts(unittest.TestCase):
    """Keep incompatible scoring/selection/state protocols explicitly separated."""

    def setUp(self) -> None:
        """Seed deterministic router inputs."""
        torch.manual_seed(20261117)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scoring_selection_and_all_gradients(self):
        """Feature: Router contracts.
        Description: Logits/weights/indices match a per-row oracle across scoring, normalization and bias shapes.
        Expectation: Scoring selection and all gradients.
        """
        for scoring in ("sigmoid", "softmax", "sqrtsoftplus"):
            for normalize in (False, True):
                for batched_bias in (False, True):
                    with self.subTest(scoring=scoring, normalize=normalize, batched=batched_bias):
                        config = GateConfig(4, 6, 2, scoring_func=scoring,
                                                      normalize_topk=normalize, sorted_indices=True,
                                                      routed_scaling_factor=1.7)
                        member = Gate(config=config)
                        weight = member.weight.detach().clone().requires_grad_()
                        left = torch.randn(2, 3, 4, requires_grad=True)
                        right = left.detach().clone().requires_grad_()
                        bias = torch.randn(6, 6) * 0.3 if batched_bias else torch.randn(6) * 0.3
                        expected = _reference(weight, right, bias, scoring, 2, 1.7, normalize)
                        actual = member(left, bias)
                        for output, target in zip(actual, expected):
                            torch.testing.assert_close(output, target)
                        upstream = torch.randn_like(actual[1])
                        (actual[0].square().sum() + (actual[1] * upstream).sum()).backward()
                        (expected[0].square().sum() + (expected[1] * upstream).sum()).backward()
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(member.weight.grad, weight.grad, atol=2e-6, rtol=2e-5)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_source_meta_empty_update_restore_and_registration(self):
        """Feature: Router contracts.
        Description: Public state survives source adoption, meta construction, an optimizer step and restore.
        Expectation: Source meta empty update restore and registration.
        """
        config = GateConfig(4, 6, 2)
        source = GateModule(config=config).eval()
        source.weight.requires_grad_(False)
        member = Gate(module=source)
        self.assertIs(member.__class__, GateModule)
        self.assertIs(member.weight, source.weight)
        self.assertFalse(member.training)
        self.assertFalse(member.weight.requires_grad)
        self.assertEqual(get_module_family(Gate).name, "gate")
        self.assertIsNotNone(get_family_build_info(member))
        self.assertFalse(getattr(GateModule, "_hp_module_replacement", False))
        with torch.device("meta"):
            trained = Gate(config=config)
        trained.to_empty(device="cpu")
        trained.reset_parameters()
        optimizer = torch.optim.AdamW(trained.parameters(), lr=0.01, foreach=False)
        values = torch.randn(3, 4)
        trained(values)[1].square().sum().backward()
        optimizer.step()
        restored = Gate(config=config)
        restored.load_state_dict(trained.state_dict())
        resumed = torch.optim.AdamW(restored.parameters(), lr=0.01, foreach=False)
        resumed.load_state_dict(deepcopy(optimizer.state_dict()))
        for current, opt in ((trained, optimizer), (restored, resumed)):
            opt.zero_grad(set_to_none=True)
            current(values)[1].square().sum().backward()
            opt.step()
        torch.testing.assert_close(trained.weight, restored.weight)
        empty = trained(torch.empty(0, 4))
        self.assertEqual(tuple(empty[0].shape), (0, 6))
        self.assertEqual(tuple(empty[1].shape), (0, 2))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_model_router_reuses_math_without_owning_model_state(self):
        """Feature: Router contracts.
        Description: Model correction bias/state stays outside the public math boundary.
        Expectation: Model router reuses math without owning model state.
        """
        config = SimpleNamespace(hidden_size=4, num_local_experts=6, num_experts_per_tok=2,
                                 scoring_func="sigmoid", routed_scaling_factor=1.5,
                                 v41_vision_enabled=True, router_aux_loss_coef=0.0)
        model = DeepseekV41TopKRouter(config).eval()
        nn.init.normal_(model.weight, std=0.2)
        model.bias.data.copy_(torch.arange(6).float() * 0.1)
        model.bias_vl.data.copy_(-torch.arange(6).float() * 0.1)
        hidden = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, False, True], [False, False, True]])
        bias = torch.where(mask.reshape(-1, 1), model.bias_vl, model.bias)
        expected = _reference(model.weight, hidden, bias, "sigmoid", 2, 1.5, True)
        actual = model(hidden, image_mask=mask)
        actual_order, target_order = actual[2].argsort(-1), expected[2].argsort(-1)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[2].gather(1, actual_order), expected[2].gather(1, target_order))
        torch.testing.assert_close(actual[1].gather(1, actual_order), expected[1].gather(1, target_order))
        self.assertIn("bias_vl", model.state_dict())
        self.assertNotIn("tokens_per_expert", model.state_dict())
        with self.assertRaises(TypeError):
            Gate(module=model)
        copied = deepcopy(model)
        self.assertIsNot(copied.weight, model.weight)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mega_gate_class_source_and_rng(self):
        """Feature: Router contracts.
        Description: The Gate mega member uses a class and source adoption does not consume initialization RNG.
        Expectation: Mega gate class source and rng.
        """
        source = CoreMegaGate(hidden_size=4, num_experts=6, top_k=2, vision_enabled=True,
                              scoring_func="sigmoid", routed_scaling_factor=1.0)
        before = torch.random.get_rng_state()
        member = Gate(module=source, implementation="mega")
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertIsInstance(member, MegaGateModule)
        self.assertFalse(getattr(MegaGateModule, "_hp_module_replacement", False))
        self.assertTrue(get_module_family(Gate).variants["mega"].endswith(".MegaGateModule"))
        for name, value in source.named_parameters():
            self.assertIs(value, member.get_parameter(name))
        adopted = Gate(module=member, implementation="mega")
        self.assertIs(adopted.weight, source.weight)
        with torch.device("meta"):
            meta = Gate(config=MegaGateConfig(4, 6, 2), implementation="mega")
        self.assertTrue(meta.weight.is_meta)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_routing_contracts(self):
        """Feature: Router contracts.
        Description: Invalid mathematical settings and unsupported bias layouts fail at boundaries.
        Expectation: Invalid routing contracts.
        """
        config = GateConfig(4, 6, 2)
        for update in ({"top_k": 0}, {"top_k": True}, {"top_k": 7}, {"scoring_func": "unknown"},
                       {"routed_scaling_factor": float("nan")}, {"normalize_topk": 1},
                       {"initializer_range": float("inf")}):
            with self.subTest(update=update), self.assertRaises((ValueError, TypeError)):
                Gate(config=replace(config, **update))
        with self.assertRaises(ValueError):
            routing_scores(torch.randn(3, 6), "unknown")
        for bias in (torch.ones(3), torch.ones(2, 6), torch.ones(6, dtype=torch.long)):
            with self.subTest(shape=bias.shape), self.assertRaises(ValueError):
                select_topk_scores(torch.rand(3, 6), top_k=2, correction_bias=bias)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_projection_preserves_enclosing_autocast_policy(self):
        """Feature: Router contracts.
        Description: The new ordinary contract retains the existing source projection's AMP dtype behavior.
        Expectation: Projection preserves enclosing autocast policy.
        """
        member = GateModule(config=GateConfig(4, 6, 2, routed_scaling_factor=1.7))
        source = CoreMegaGate(hidden_size=4, num_experts=6, top_k=2, scoring_func="sigmoid",
                              routed_scaling_factor=1.7)
        source.weight.data.copy_(member.weight)
        source.bias.data.copy_(torch.linspace(-0.2, 0.2, 6))
        values = torch.randn(2, 3, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = member(values, source.bias), source(values)
        self.assertEqual(actual[0].dtype, torch.bfloat16)
        for result, target in zip(actual, expected):
            torch.testing.assert_close(result, target, rtol=0, atol=0)

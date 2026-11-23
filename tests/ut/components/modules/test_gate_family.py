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
"""Gate family variants and shared routing adapter contracts."""
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
import unittest

import torch
from hyper_multicore.modules.mega_gate import MegaGate as SourceMegaGate

from examples.torch.module_family.config import MegaGateConfig
from hyper_parallel.components.modules import get_family_build_info
from hyper_parallel.components.modules import gate_family as gate
from hyper_parallel.components.modules.family import get_module_family
from hyper_parallel.components.modules.gate_family import Gate
from hyper_parallel.distributed.expert_parallel.routing import resolve_moe_router
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementRequest


class TestGateFamily(unittest.TestCase):
    """Preserve public router state and reuse the existing adapter implementations."""

    def test_original_mega_gate_outputs_and_gradients(self):
        """Match original scoring, TopK, modality and precision contracts with nonzero correction biases."""
        modes = ("fp32", "bf16", "autocast")
        for scoring, top_k, modality, precision in product(
                ("sqrtsoftplus", "softmax", "sigmoid"), (1, 2), ("text", "visual", "mixed"), modes):
            with self.subTest(scoring=scoring, top_k=top_k, modality=modality, precision=precision):
                torch.manual_seed(17)
                config = MegaGateConfig(8, 6, top_k, scoring_func=scoring,
                                        routed_scaling_factor=1.7, vision_enabled=modality != "text")
                member = Gate(config=config, implementation="mega")
                original = SourceMegaGate(hidden_size=8, num_experts=6, top_k=top_k,
                                          scoring_func=scoring, routed_scaling_factor=1.7,
                                          vision_enabled=config.vision_enabled)
                with torch.no_grad():
                    member.bias.copy_(torch.linspace(-0.2, 0.2, 6))
                    if member.bias_vl is not None:
                        member.bias_vl.copy_(-member.bias)
                original.load_state_dict(member.state_dict())
                dtype = torch.bfloat16 if precision == "bf16" else torch.float32
                member.to(dtype=dtype)
                original.to(dtype=dtype)
                left = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                mask = None
                if modality == "visual":
                    mask = torch.ones(2, 3, dtype=torch.bool)
                elif modality == "mixed":
                    mask = torch.tensor([[True, False, True], [False, True, False]])
                with torch.autocast("cpu", dtype=torch.bfloat16) if precision == "autocast" else nullcontext():
                    actual, expected = member(left, mask), original(right, mask)
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
                upstream = torch.randn_like(actual[1])
                (actual[0].square().sum() * 0.01 + (actual[1] * upstream).sum()).backward()
                (expected[0].square().sum() * 0.01 + (expected[1] * upstream).sum()).backward()
                torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
                for name, parameter in member.named_parameters():
                    reference = original.get_parameter(name)
                    self.assertEqual(parameter.dtype, reference.dtype)
                    if reference.grad is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0)

    def test_only_gate_family_is_registered(self):
        """Both members share one family; the removed MegaGate entry cannot register a second family."""
        self.assertFalse(hasattr(gate, "MegaGate"))
        self.assertEqual(get_module_family(Gate).name, "gate")
        self.assertEqual(set(get_module_family(Gate).variants), {"local", "mega"})

    def test_original_initialization_reset_and_empty_input(self):
        """The original and migrated member initialize identically and accept empty token batches."""
        for vision in (False, True):
            with self.subTest(vision=vision):
                torch.manual_seed(71)
                member = Gate(config=MegaGateConfig(8, 6, 2, vision_enabled=vision), implementation="mega")
                torch.manual_seed(71)
                original = SourceMegaGate(hidden_size=8, num_experts=6, top_k=2,
                                          scoring_func="sqrtsoftplus", routed_scaling_factor=1.0,
                                          vision_enabled=vision)
                self.assertEqual(tuple(member.state_dict()), tuple(original.state_dict()))
                for name, parameter in member.named_parameters():
                    torch.testing.assert_close(parameter, original.get_parameter(name), atol=0, rtol=0)
                torch.manual_seed(72)
                member.reset_parameters()
                torch.manual_seed(72)
                original.reset_parameters()
                for name, parameter in member.named_parameters():
                    torch.testing.assert_close(parameter, original.get_parameter(name), atol=0, rtol=0)
                values = torch.empty(1, 0, 8)
                mask = torch.empty(1, 0, dtype=torch.bool) if vision else None
                for actual, expected in zip(member(values, mask), original(values, mask)):
                    torch.testing.assert_close(actual, expected)

    def test_source_replacement_and_gradients(self):
        """Actual replacement preserves source values, parameter identities and mode."""
        for vision in (False, True):
            with self.subTest(vision=vision):
                source = SourceMegaGate(hidden_size=4, num_experts=4, top_k=2,
                                  scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
                                  vision_enabled=vision).eval()
                source.bias.requires_grad_(False)
                reference = SourceMegaGate(hidden_size=4, num_experts=4, top_k=2,
                                     scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
                                     vision_enabled=vision).eval()
                reference.load_state_dict(source.state_dict())
                reference.bias.requires_grad_(False)
                model = torch.nn.Sequential(source)
                model, _ = prepare_model_components(model, {"module_replacement_requests": (
                    ModuleReplacementRequest(("0",), Gate, kwargs={"implementation": "mega"}),)})
                member = model[0]
                self.assertFalse(member.training)
                self.assertEqual(get_family_build_info(member).origin, "replacement")
                for name, parameter in source.named_parameters():
                    self.assertIs(member.get_parameter(name), parameter)
                self.assertFalse(member.bias.requires_grad)
                x = torch.randn(2, 3, 4, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                mask = torch.tensor([[True, False, True], [False, True, False]]) if vision else None
                actual, expected = member(x, mask), reference(other, mask)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right)
                upstream = torch.randn_like(actual[1])
                (actual[1] * upstream).sum().backward()
                (expected[1] * upstream).sum().backward()
                torch.testing.assert_close(x.grad, other.grad)
                torch.testing.assert_close(member.weight.grad, reference.weight.grad)

    def test_direct_meta_and_scope(self):
        """Caller-owned configs construct/reset routers; other router classes are rejected."""
        config = MegaGateConfig(4, 4, 2)
        with torch.device("meta"):
            member = Gate(config=config, implementation="mega")
        member.to_empty(device="cpu")
        member.reset_parameters()
        self.assertTrue(torch.isfinite(member.weight).all())
        self.assertEqual(get_family_build_info(member).origin, "direct")
        restored = Gate(config=config, implementation="mega")
        restored.load_state_dict(member.state_dict())
        x = torch.randn(2, 3, 4)
        for actual, expected in zip(restored(x), member(x)):
            torch.testing.assert_close(actual, expected)
        with self.assertRaises(TypeError):
            Gate(module=torch.nn.Linear(4, 4), implementation="mega")
        with self.assertRaises(ValueError):
            Gate(config=config, module=member, implementation="mega")

    def test_public_adapter_reuse(self):
        """Resolved adapters preserve numerical routing contracts for public Gate members."""
        source = Gate(config=MegaGateConfig(4, 4, 2, scoring_func="sigmoid"), implementation="mega")
        x = torch.randn(2, 3, 4)
        _, weights, indices = source(x)
        actual_indices, actual_weights = resolve_moe_router("topk_router_module")(
            SimpleNamespace(router=source), x)
        torch.testing.assert_close(actual_indices, indices)
        torch.testing.assert_close(actual_weights, weights)
        with self.assertRaises(KeyError):
            resolve_moe_router("unknown_router")

    def test_softmax_and_group_adapters_against_explicit_scores(self):
        """Adapter resolution executes softmax normalization and group-limited sigmoid selection."""
        logits = torch.tensor([[4., 3., 0., -1.], [-2., -1., 2., 1.]])
        projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(4))
        owner = SimpleNamespace(router=projection, top_k=2, n_group=2, topk_group=1,
                                norm_topk_prob=True, routed_scaling_factor=1.7)
        expected_indices = torch.tensor([[0, 1], [2, 3]])
        for name, scores, scale in (("softmax_topk", logits.softmax(-1), 1.0),
                                    ("sigmoid_group", logits.sigmoid(), 1.7)):
            with self.subTest(adapter=name):
                indices, weights = resolve_moe_router(name)(owner, logits)
                order = indices.argsort(-1)
                expected = scores.gather(1, expected_indices)
                expected = expected / expected.sum(-1, keepdim=True) * scale
                torch.testing.assert_close(indices.gather(1, order), expected_indices)
                torch.testing.assert_close(weights.gather(1, order), expected)

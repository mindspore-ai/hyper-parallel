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
"""Real Family configuration and replacement contracts, with only NPU checks isolated."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.components.modules import grouped_experts
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import MXFP8GroupedSwiGLU, FusedMXFP8GroupedSwiGLU
from hyper_parallel.trainer.config.resolver import ConfigResolutionError
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.common.mark_utils import arg_mark


class TestGroupedExpertsFamily(unittest.TestCase):
    """Exercise registered production targets through configuration and replacement."""

    def setUp(self) -> None:
        """Isolate device availability without bypassing selection or constructors."""
        for name in ("validate_npu_gmm_runtime", "validate_npu_swiglu_runtime"):
            device_check = patch("hyper_parallel.components.modules.mxfp8_grouped_swiglu." + name)
            device_check.start()
            self.addCleanup(device_check.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_selection_and_layout(self) -> None:
        """
        Feature: GroupedExperts Family.
        Description: Parse all legal combinations and execute real replacement and transforms.
        Expectation: Implementation, persistent layout and loaded weights match the baseline.
        """
        source = TinyMoE()
        for selectors, expected in (({}, GroupedExperts), ({"expert_2d": True}, GroupedExperts),
                                    ({"use_mxfp8": True}, MXFP8GroupedSwiGLU),
                                    ({"use_mxfp8": True, "use_fused_mxfp8": True}, FusedMXFP8GroupedSwiGLU)):
            with self.subTest(selectors=selectors):
                model = copy.deepcopy(source)
                replace_experts(model, family_config(**selectors))
                expert = model.mlp.experts
                self.assertIs(type(expert), expected)
                self.assertIsInstance(expert, GroupedExperts)
                flat = selectors.get("expert_2d", False)
                self.assertEqual(tuple(expert.gate_up_proj.shape), (512, 512) if flat else (4, 128, 512))
                self.assertEqual(tuple(expert.down_proj.shape), (1024, 128) if flat else (4, 256, 128))
                self.assertEqual(set(dict(expert.named_parameters())), {"gate_up_proj", "down_proj"})
                torch.testing.assert_close(expert.gate_up_proj.reshape(4, 128, 512),
                                           source.mlp.experts.gate_up_proj.transpose(-2, -1))
                torch.testing.assert_close(expert.down_proj.reshape(4, 256, 128),
                                           source.mlp.experts.down_proj.transpose(-2, -1))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_selectors(self) -> None:
        """
        Feature: GroupedExperts Family.
        Description: Reject fusion without MXFP8, nonboolean selectors and constructor typos.
        Expectation: Configuration fails before any NPU constructor runs.
        """
        for selectors in ({"use_fused_mxfp8": True}, {"use_mxfp8": 1}, {"unknown": True},
                          {"use_mxfp8": True, "unknown": True}):
            with self.subTest(selectors=selectors), self.assertRaises(ConfigResolutionError):
                family_config(**selectors)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unsupported_parallelism(self) -> None:
        """
        Feature: GroupedExperts Family scope.
        Description: Parallel compute is not silently enabled by local low-precision selection.
        Expectation: Unsupported contexts fail with the module FQN during replacement.
        """
        for context in ({"ep": True}, {"ep_size": 2}, {"tp": True}, {"cp": True}, {"pp": True}):
            with self.subTest(context=context), self.assertRaisesRegex(NotImplementedError, "mlp.experts"):
                replace_experts(TinyMoE(), family_config(use_mxfp8=True), context=context)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_and_canonical_weights(self) -> None:
        """
        Feature: GroupedExperts initialization.
        Description: Eager canonical parameters keep identity and meta weights keep deferred transforms.
        Expectation: No data copy is attempted on meta tensors; checkpoint conversion remains registered.
        """
        for selectors in ({}, {"use_mxfp8": True}, {"use_mxfp8": True, "use_fused_mxfp8": True}):
            with self.subTest(selectors=selectors):
                eager = TinyMoE(canonical=True)
                weight = eager.mlp.experts.gate_up_proj
                weight.requires_grad_(False)
                replace_experts(eager, family_config(**selectors))
                self.assertIs(eager.mlp.experts.gate_up_proj, weight)
                self.assertFalse(weight.requires_grad)
                meta = TinyMoE(device="meta")
                mapping = replace_experts(meta, family_config(**selectors))
                self.assertTrue(meta.mlp.experts.gate_up_proj.is_meta)
                self.assertEqual(len(mapping), 2)


    def test_expert_2d_conversion_and_config(self) -> None:
        """Feature: 2D storage. Description: Convert both source layouts. Expectation: Exact round trip."""
        for canonical in (False, True):
            source = TinyMoE(canonical=canonical)
            model = copy.deepcopy(source)
            model.mlp.experts.config.expert_2d = True
            model._weight_conversions = replace_experts(model, family_config())  # pylint: disable=protected-access
            self.assertEqual(model.mlp.experts.gate_up_proj.ndim, 2)
            restored = revert_weight_conversion(model, model.state_dict())
            for name, value in source.state_dict().items():
                torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
            source.mlp.experts.config.expert_2d = True
            self.assertEqual(GroupedExperts(module=source.mlp.experts, expert_2d=False).gate_up_proj.ndim, 3)
        for invalid in (1, "true"):
            with self.assertRaisesRegex(ValueError, "expert_2d"):
                GroupedExperts(module=source.mlp.experts, expert_2d=invalid)
        meta = TinyMoE(device="meta")
        replace_experts(meta, family_config(expert_2d=True))
        self.assertTrue(meta.mlp.experts.gate_up_proj.is_meta)
        self.assertEqual(meta.mlp.experts.gate_up_proj.ndim, 2)

    def test_expert_2d_gradients_and_ep_entry(self) -> None:
        """Feature: 2D computation. Description: Include empty tokens and EP binding. Expectation: Equal gradients."""
        def grouped_mm(inputs: torch.Tensor, weights: torch.Tensor, *, group_list: torch.Tensor,
                       **kwargs: object) -> torch.Tensor:
            """Replace only device GEMM with its CPU reference."""
            del kwargs
            counts = torch.diff(group_list, prepend=group_list.new_zeros(1)).tolist()
            return torch.cat([part @ weight for part, weight in zip(inputs.split(counts), weights)])

        source = TinyMoE()
        source.mlp.experts.config.use_fused_swiglu = False
        reference, actual = copy.deepcopy(source), copy.deepcopy(source)
        for model, enabled in ((reference, False), (actual, True)):
            replace_experts(model, family_config(expert_2d=enabled))
        with patch.object(grouped_experts, "grouped_matmul", grouped_mm):
            for counts in ([2, 0, 3, 1], [0, 0, 0, 0]):
                left = torch.randn(sum(counts), 128, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                expected = reference.mlp.experts.forward_expert_major(left, torch.tensor(counts))
                result = actual.mlp.experts.forward_expert_major(right, torch.tensor(counts))
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                expected.sum().backward()
                result.sum().backward()
                torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
                for name, parameter in reference.named_parameters():
                    torch.testing.assert_close(actual.get_parameter(name).grad.view_as(parameter),
                                               parameter.grad, rtol=0, atol=0)
                reference.zero_grad(set_to_none=True)
                actual.zero_grad(set_to_none=True)
            expert = actual.mlp.experts
            expert.gate_up_proj = torch.nn.Parameter(expert.gate_up_proj[:128].detach().clone())
            expert.down_proj = torch.nn.Parameter(expert.down_proj[:256].detach().clone())
            bind_local_expert_forward(actual.mlp, 4)
            inputs = torch.randn(5, 128)
            expected = reference.mlp.experts.forward_expert_major(inputs, torch.tensor([5, 0, 0, 0]))
            result = expert(inputs, torch.zeros(5, dtype=torch.long))
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

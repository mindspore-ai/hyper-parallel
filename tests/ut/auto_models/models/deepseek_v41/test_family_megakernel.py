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
"""CPU contracts for V4.1 source adoption through the public grouped-expert family."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_mega_grouped_experts import EPMegaGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsModule
from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.models.deepseek_v41.components.modality_moe import (
    replace_deepseek_v41_grouped_experts,
)
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.mark_utils import arg_mark


_TARGET = "hyper_parallel.components.modules.GroupedExperts"


def _cpu_grouped_matmul(inputs: torch.Tensor, weights: torch.Tensor, *, group_list: torch.Tensor) -> torch.Tensor:
    """Replace only the accelerator GEMM boundary with differentiable CPU matrix multiplication."""
    counts = torch.diff(group_list, prepend=group_list.new_zeros(1))
    expert_ids = torch.arange(counts.numel()).repeat_interleave(counts)
    return torch.bmm(inputs.unsqueeze(1), weights.index_select(0, expert_ids)).squeeze(1)


class TestV41MegaFamily(unittest.TestCase):
    """One public YAML target consumes packed state and selects the requested backend."""

    @staticmethod
    def _config() -> DeepseekV4Config:
        return DeepseekV4Config.from_dict({
            "hidden_size": 16, "moe_intermediate_size": 24, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "num_hidden_layers": 4, "swiglu_limit": 1.5,
        })

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_public_yaml_target_preserves_source_and_selects_backend(self) -> None:
        """Feature: DeepSeek-V4.1 expert replacement.
        Description: The user-provided public target accepts V4.1 dimensions, clamp and routing facts.
        Expectation: Select either ordinary EP or Mega and retain source weights, training state and clamp.
        """
        for megakernel, expected in ((False, EPGroupedExperts), (True, EPMegaGroupedExperts)):
            with self.subTest(megakernel=megakernel):
                args = resolve_config({
                    "model": {"_target_": "torch.nn.Identity"},
                    "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
                    "plan_overrides": [{"match": "experts", "replace": {
                        "kind": "module", "implementation": {
                            "_target_": _TARGET, "megakernel": megakernel, "dispatch_mode": "push",
                        },
                    }}],
                })
                model = nn.Module()
                model.config = self._config()
                source = DeepseekV4Experts(model.config).eval()
                model.experts = source
                rules = entries_to_module_replacements(args.plan_overrides)
                apply_module_replacements(model, compile_module_replacements(model, rules),
                                          weights_mapping=[], context={"ep_size": 2})
                experts = model.experts
                self.assertIs(type(experts), expected)
                self.assertIs(experts.gate_up_proj, source.gate_up_proj)
                self.assertIs(experts.down_proj, source.down_proj)
                self.assertFalse(experts.training)
                self.assertEqual(experts.config.top_k, 2)
                self.assertEqual(experts.config.num_layers, 4)
                self.assertEqual(experts.config.routing_weight_mode, "output")
                self.assertIsNone(experts.config.activation)
                self.assertEqual(experts.config.swiglu_limit, source.limit)
                if megakernel:
                    self.assertEqual(experts._runtime.tuning["dispatch_mode"], "push")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_mega_target_warns_and_falls_back_to_ordinary_gmm_for_ep_one(self) -> None:
        """Feature: DeepSeek-V4.1 expert replacement.
        Description: EP=1 preserves the original FP32 clamped activation despite selecting the Mega target.
        Expectation: Match the legacy FP32 clamped output and gradients exactly with CPU kernel doubles.
        """
        config = self._config()
        source = DeepseekV4Experts(config).double()
        context = {"num_layers": 4, "module_configs": (config,), "ep_size": 1}
        ordinary = replace_deepseek_v41_grouped_experts(module=source, module_fqn="experts", context=context)
        with self.assertWarnsRegex(UserWarning, "EP=1 falls back"):
            fallback = GroupedExperts(
                module=source, module_fqn="experts", context=context, megakernel=True, dispatch_mode="pull",
            )
        self.assertIs(type(fallback), GroupedExpertsModule)
        self.assertIs(fallback.gate_up_proj, ordinary.gate_up_proj)
        hidden = (torch.randn(4, 16, dtype=torch.float64) * 20).requires_grad_()
        counts = torch.tensor([2, 1, 0, 1])
        prefix = "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts"
        with patch(prefix + ".grouped_matmul", side_effect=_cpu_grouped_matmul), \
                patch(prefix + ".npu_swiglu", side_effect=AssertionError("clamped path used unclamped NPU SwiGLU")):
            actual = fallback.compute_experts(hidden, counts, fallback.gate_up_proj, fallback.down_proj)
            expected = ordinary.compute_experts(hidden, counts, ordinary.gate_up_proj, ordinary.down_proj)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(actual.sum(), (hidden, source.gate_up_proj, source.down_proj))
        expected_grads = torch.autograd.grad(expected.sum(), (hidden, source.gate_up_proj, source.down_proj))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_packed_clamp_aliases_and_invalid_limits(self) -> None:
        """Feature: DeepSeek-V4.1 expert replacement.
        Description: Read source/config clamp aliases without a model-specific conversion factory.
        Expectation: Respect source precedence, warn for disabled limits and reject invalid values.
        """
        config = self._config()
        source = DeepseekV4Experts(config)
        del source.limit
        context = {"num_layers": 4, "module_configs": ({"swiglu_limit": 2.5},)}
        experts = GroupedExperts(module=source, context=context)
        self.assertEqual(experts.config.swiglu_limit, 2.5)
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                source.limit = value
                with self.assertWarnsRegex(UserWarning, "disables clamping"):
                    experts = GroupedExperts(module=source, context=context)
                self.assertIsNone(experts.config.swiglu_limit)
        for value in (True, float("nan"), float("inf"), "1.5"):
            with self.subTest(value=value):
                source.limit = value
                with self.assertRaisesRegex(ValueError, "finite number"):
                    GroupedExperts(module=source, context=context)

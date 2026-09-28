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
"""CPU unit tests for the declarative DeepSeek-V4.1 HyperMegaMhc adapter."""

import unittest

import torch
from torch import nn

from hyper_parallel.components.modules.mhc import pipelined_mhc_post
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import (
    DeepseekV41HyperMegaMhc,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41PipelinedHyperConnection,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.mark_utils import arg_mark


def _model_target():
    """Provide the minimal model target required by Trainer config resolution."""


def _optimizer_target():
    """Provide the minimal optimizer target required by Trainer config resolution."""


class _RmsNorm(nn.Module):
    """Minimal weightless RMSNorm matching the source mHC coefficient norm."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension in FP32."""
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)


class _SourceMhc(nn.Module):
    """Provide the released four-stream DeepSeek mHC parameter layout."""

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.input_norm = _RmsNorm()
        self.fn = nn.Parameter(torch.randn(24, 4 * hidden_size) * 0.01)
        self.base = nn.Parameter(torch.randn(24) * 0.01)
        self.scale = nn.Parameter(torch.ones(3))
        self.hc_mult = 4
        self.hc_sinkhorn_iters = 20
        self.hc_eps = 1.0e-6


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestDeepseekV41HyperMegaMhc(unittest.TestCase):
    """Validate replacement contracts and the hardware-free shifted protocol."""

    @staticmethod
    def _source() -> DeepseekV41PipelinedHyperConnection:
        """Build one canonical model-owned mHC coefficient module."""
        return DeepseekV41PipelinedHyperConnection(_SourceMhc())

    def test_declarative_replacement_preserves_parameter_identity(self):
        """Replace through the generic executor without changing checkpoint state."""
        model = nn.Sequential(self._source())
        original_keys = tuple(model.state_dict())
        original_parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
        rule = ModuleReplacementSpec(
            match=("0",),
            factory=DeepseekV41HyperMegaMhc,
            module_type=DeepseekV41PipelinedHyperConnection,
        )

        plan = compile_module_replacements(model, [rule])
        apply_module_replacements(model, plan, context={"ep": True})

        self.assertIsInstance(model[0], DeepseekV41HyperMegaMhc)
        self.assertEqual(tuple(model.state_dict()), original_keys)
        self.assertEqual(
            {name: id(parameter) for name, parameter in model.named_parameters()},
            original_parameters,
        )

    def test_cpu_advance_matches_shifted_reference_boundary(self):
        """Keep the model protocol numerically testable without importing multicore."""
        module = DeepseekV41HyperMegaMhc(module=self._source(), context={})
        previous_output = torch.randn(2, 3, 8)
        residual = torch.randn(2, 3, 4, 8)
        previous_pre = torch.softmax(torch.randn(2, 3, 4), dim=-1)
        previous_post = torch.randn(2, 3, 4)
        previous_residual_mix = torch.softmax(torch.randn(2, 3, 4, 4), dim=-1)
        norm_weight = torch.randn(8)

        outputs = module.advance(
            previous_output,
            residual,
            previous_pre,
            previous_post,
            previous_residual_mix,
            norm_weight,
        )

        expected_residual = pipelined_mhc_post(
            previous_output,
            residual,
            previous_post,
            previous_residual_mix,
        )
        expected_pre, expected_post, expected_mix = module(expected_residual)
        mixed_input = (previous_pre.unsqueeze(-1) * expected_residual).sum(dim=2)
        expected_input = mixed_input * torch.rsqrt(
            mixed_input.square().mean(dim=-1, keepdim=True) + module.norm_eps
        ) * norm_weight
        expected = expected_residual, expected_pre, expected_post, expected_mix, expected_input
        for actual_value, expected_value in zip(outputs, expected):
            torch.testing.assert_close(actual_value, expected_value)

    def test_yaml_target_resolves_declarative_replacement(self):
        """Resolve the HyperMegaMhc adapter through the generic YAML target channel."""
        target = (
            "hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc."
            "DeepseekV41HyperMegaMhc"
        )

        recipe = resolve_config({
            "model": {"_target_": f"{__name__}._model_target"},
            "optimizer": {"_target_": f"{__name__}._optimizer_target"},
            "plan_overrides": [{
                "match": ["model.layers.*.attn_hc", "model.layers.*.ffn_hc"],
                "module_type": (
                    "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41."
                    "DeepseekV41PipelinedHyperConnection"
                ),
                "replace_module": {"_target_": target},
            }],
        })
        rules = entries_to_module_replacements(recipe.plan_overrides)
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module()])
        model.model.layers[0].attn_hc = self._source()
        model.model.layers[0].ffn_hc = self._source()

        plan = compile_module_replacements(model, rules)
        apply_module_replacements(model, plan)

        self.assertIsInstance(model.model.layers[0].attn_hc, DeepseekV41HyperMegaMhc)
        self.assertIsInstance(model.model.layers[0].ffn_hc, DeepseekV41HyperMegaMhc)

    def test_parallel_axes_are_rejected_before_native_execution(self):
        """Fail configuration when an unsupported sharding axis is active."""
        for axis in ("tp", "cp", "pp"):
            with self.subTest(axis=axis):
                with self.assertRaisesRegex(ValueError, "TP=CP=PP=1"):
                    DeepseekV41HyperMegaMhc(module=self._source(), context={axis: True})


if __name__ == "__main__":
    unittest.main()

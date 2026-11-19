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
"""Contract and delayed-pre checks for the independent CANN mHC adapter."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.models.deepseek_v41.adapter.distributed.fused_mhc import DeepseekV41FusedMhc
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41PipelinedHyperConnection
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)


class RmsNorm(nn.Module):
    """Weightless coefficient normalization."""

    def __init__(self):
        super().__init__()
        self.variance_epsilon = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with the source epsilon."""
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)


def source(hidden: int = 8) -> DeepseekV41PipelinedHyperConnection:
    """Construct the canonical parameter layout independently of the adapter."""
    module = nn.Module()
    module.input_norm = RmsNorm()
    module.fn = nn.Parameter(torch.randn(24, 4 * hidden) * .01)
    module.base = nn.Parameter(torch.randn(24) * .01)
    module.scale = nn.Parameter(torch.tensor([.7, 1.2, .8]))
    module.hc_mult, module.hc_sinkhorn_iters, module.hc_eps = 4, 20, 1e-6
    return DeepseekV41PipelinedHyperConnection(module)


class TestFusedMhc(unittest.TestCase):
    """Keep coefficient delay, parameter identity and CPU portability explicit."""

    def test_replacement_preserves_checkpoint_parameters(self):
        """Resolve the replacement through the public executor."""
        model = nn.Sequential(source())
        parameters = {n: id(p) for n, p in model.named_parameters()}
        keys = list(model.state_dict())
        plan = compile_module_replacements(model, [ModuleReplacementSpec(match=("0",), factory=DeepseekV41FusedMhc)])
        apply_module_replacements(model, plan)
        self.assertIsInstance(model[0], DeepseekV41FusedMhc)
        self.assertEqual(parameters, {n: id(p) for n, p in model.named_parameters()})
        self.assertEqual(keys, list(model.state_dict()))
        model.bfloat16()
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_boundary_keeps_delayed_pre_and_all_gradients(self):
        """Changing the old pre must change the block input, not the new coefficients."""
        torch.manual_seed(42)
        candidate = DeepseekV41FusedMhc(module=source())
        oracle = copy.deepcopy(candidate)
        inputs = [torch.randn(*shape, requires_grad=True) for shape in (
            (2, 3, 8), (2, 3, 4, 8), (2, 3, 4), (2, 3, 4), (2, 3, 4, 4), (8,),
        )]
        expected_inputs = [t.detach().clone().requires_grad_() for t in inputs]
        with patch.dict("sys.modules", {"hyper_multicore": None}):
            actual = candidate.advance(*inputs)
        output, residual, old_pre, old_post, old_comb, weight = expected_inputs
        mixed = torch.einsum("bsij,bsih->bsjh", old_comb, residual) + old_post.unsqueeze(-1) * output.unsqueeze(-2)
        pre, post, combine = oracle(mixed)
        collapsed = torch.einsum("bsi,bsih->bsh", old_pre, mixed)
        norm = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + oracle.norm_eps) * weight
        expected = mixed, pre, post, combine, norm
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        grads = [torch.randn_like(t) for t in actual]
        actual_grad = torch.autograd.grad(actual, inputs + list(candidate.parameters()), grads)
        expected_grad = torch.autograd.grad(expected, expected_inputs + list(oracle.parameters()), grads)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)
        changed = candidate.advance(inputs[0], inputs[1], inputs[2] + 1, *inputs[3:])
        for a, b in zip(actual[:4], changed[:4]):
            torch.testing.assert_close(a, b)
        self.assertFalse(torch.allclose(actual[-1], changed[-1]))

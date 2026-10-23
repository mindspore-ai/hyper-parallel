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
"""Preserve the pre-refactor precision of routed expert contributions."""

import unittest
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.modules.grouped_experts import GroupedExperts
from tests.common.mark_utils import arg_mark


def _cpu_projection(
    inputs: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor,
    strategy: object, group_list_type: int,
) -> torch.Tensor:
    """Replace only the projection backend, isolating the shell's routing arithmetic."""
    del strategy, group_list_type
    parts = inputs.split(counts.tolist())
    return torch.cat([
        (part.float() @ expert.float().T).to(inputs.dtype)
        for part, expert in zip(parts, weight)
    ])


def _legacy_output(
    module: GroupedExperts, hidden: torch.Tensor, indices: torch.Tensor,
    probabilities: torch.Tensor, *, round_each_route: bool,
) -> torch.Tensor:
    """Express the two historical combine contracts without consulting the new policy."""
    top_k = indices.shape[1]
    order = indices.flatten().argsort()
    counts = indices.new_zeros(module.num_experts)
    counts.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten()))
    # Gather in expert order so the input-gradient accumulation order also
    # matches the historical shell; repeat_interleave would sum by route.
    routed = hidden[order // top_k]
    gate_up = _cpu_projection(routed, module.gate_up_proj, counts, None, 1)
    gate, up = gate_up.chunk(2, dim=-1)
    outputs = _cpu_projection(module.act_fn(gate) * up, module.down_proj, counts, None, 1)
    weighted = outputs * probabilities.flatten()[order, None]
    if round_each_route:
        weighted = weighted.to(hidden.dtype)
    return weighted[order.argsort()].reshape(hidden.shape[0], top_k, hidden.shape[1]).sum(dim=1)


class TestGroupedExpertsRoutingPrecision(unittest.TestCase):
    """Use real format strategies and differentiable CPU projections for routing contracts."""

    def setUp(self) -> None:
        """Keep random inputs isolated from other tests."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    @staticmethod
    def _source(dtype: torch.dtype) -> nn.Module:
        source = nn.Module()
        source.gate_up_proj = nn.Parameter(torch.randn(6, 12, 4, dtype=dtype) * 0.25)
        source.down_proj = nn.Parameter(torch.randn(6, 4, 6, dtype=dtype) * 0.25)
        source.act_fn = nn.SiLU()
        return source


    def _check_case(self, factory, dtype: torch.dtype, route_dtype: torch.dtype, top_k: int, round_routes: bool):
        """Compare one format and routing configuration with its legacy contract."""
        torch.manual_seed(20261020)
        module = GroupedExperts.from_module(self._source(dtype), fqn="experts", grouped_linear=factory())
        hidden = torch.randn(5, 4, dtype=dtype, requires_grad=True)
        indices = torch.stack([torch.arange(top_k).roll(token) for token in range(hidden.shape[0])])
        probabilities = torch.softmax(torch.randn(5, top_k), dim=-1).to(route_dtype).requires_grad_(True)
        with mock.patch(
            "hyper_parallel.components.quantization.modules.grouped_experts._GroupedLinearFunction.apply",
            side_effect=_cpu_projection,
        ):
            actual = module(hidden, indices, probabilities)
        expected = _legacy_output(module, hidden, indices, probabilities, round_each_route=round_routes)
        expected_dtype = dtype if round_routes else torch.promote_types(dtype, route_dtype)
        self.assertEqual(actual.dtype, expected_dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        parameters = (module.gate_up_proj, module.down_proj, probabilities)
        variables = (hidden, *parameters)
        actual_gradients = torch.autograd.grad(actual.float().square().sum(), variables)
        expected_gradients = torch.autograd.grad(expected.float().square().sum(), variables)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

        reference_parameters = [nn.Parameter(parameter.detach().clone()) for parameter in parameters]
        for parameter, gradient in zip(parameters, actual_gradients[1:]):
            parameter.grad = gradient
        for parameter, gradient in zip(reference_parameters, expected_gradients[1:]):
            parameter.grad = gradient
        # NPU foreach autodetection queries a device even for CPU parameters.
        torch.optim.SGD(parameters, lr=0.01, foreach=False).step()
        torch.optim.SGD(reference_parameters, lr=0.01, foreach=False).step()
        for parameter, reference_parameter in zip(parameters, reference_parameters):
            torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_outputs_gradients_and_update_keep_each_formats_original_precision(self) -> None:
        """Low-precision routed contributions return to hidden-state precision."""
        cases = (
            ("mxfp8", MXFP8GroupedLinear, True),
            ("hif8", HiFloat8GroupedLinear, True),
            ("w4a8_32", W4A8GroupedLinear, True),
            ("fake_w4a8_32", FakeW4A8GroupedLinear, True),
        )
        for name, factory, round_routes in cases:
            for dtype in (torch.bfloat16, torch.float16):
                for route_dtype in (torch.float32, dtype):
                    for top_k in (1, 2, 6):
                        with self.subTest(format=name, dtype=dtype, route_dtype=route_dtype, top_k=top_k):
                            self._check_case(factory, dtype, route_dtype, top_k, round_routes)

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
"""CPU regressions for the GMM autograd bridge, including empty groups."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

import hyper_parallel


def _grouped_kernel(x, weight, *, group_list, group_type, **kwargs):
    """Emulate the device kernel with batched products and indexed reduction."""
    del kwargs
    counts = torch.diff(group_list, prepend=group_list.new_zeros(1))
    experts = torch.arange(counts.numel()).repeat_interleave(counts)
    if group_type == 0:
        return [torch.bmm(x[0].unsqueeze(1), weight[0][experts]).squeeze(1)]
    products = x[0].T.unsqueeze(2) * weight[0].unsqueeze(1)
    output = products.new_zeros(counts.numel(), x[0].shape[0], weight[0].shape[1])
    return [output.index_add(0, experts, products)]


def _load_functional(name, backend):
    """Load the installed source with only its optional NPU kernel replaced."""
    source = Path(hyper_parallel.__file__).resolve().parent / "components/functional" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_cpu_regression", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch_npu": backend}):
        spec.loader.exec_module(module)
    return module


class TestGroupedMatmul(unittest.TestCase):
    """Check both gradients without importing a real accelerator backend."""

    def setUp(self) -> None:
        """Exercise both existing wrappers against the same CPU kernel oracle."""
        backend = ModuleType("torch_npu")
        self.kernel = Mock(side_effect=_grouped_kernel)
        backend.npu_grouped_matmul = self.kernel
        grouped = _load_functional("grouped_matmul", backend)
        swiglu = _load_functional("npu_grouped_swiglu", backend)
        with patch.dict(sys.modules, {"torch_npu": backend}):
            packed = swiglu._get_npu_grouped_matmul()
        self.bridges = {
            "grouped": lambda inputs, weight, groups: grouped.grouped_matmul(inputs, weight, group_list=groups),
            "swiglu": packed.apply,
        }

    def test_forward_and_gradients_with_empty_experts(self):
        """Include zero total tokens, unused experts and repeated assignments."""
        for name, bridge in self.bridges.items():
            for counts in ([0, 0, 0], [0, 3, 0], [2, 0, 3]):
                with self.subTest(bridge=name, counts=counts):
                    self._check_gradients(bridge, counts, compiling=False)

    def test_compiling_empty_groups_keeps_weight_gradient_kernel(self):
        """An empty eager fast path must not remove the compiled weight gradient."""
        for name, bridge in self.bridges.items():
            with self.subTest(bridge=name):
                self._check_gradients(bridge, [0, 0, 0], compiling=True)

    def _check_gradients(self, bridge, counts, compiling):
        """Compare forward, input gradient and weight gradient with indexed matmul."""
        self.kernel.reset_mock()
        inputs = torch.randn(sum(counts), 4, requires_grad=True)
        weights = torch.randn(3, 4, 6, requires_grad=True)
        gradient = torch.randn(sum(counts), 6)
        output = bridge(inputs, weights, torch.tensor(counts).cumsum(0))
        with patch("torch.compiler.is_compiling", return_value=compiling):
            actual = torch.autograd.grad(output, (inputs, weights), gradient)
        ids = torch.arange(3).repeat_interleave(torch.tensor(counts))
        expected = torch.einsum("ti,tio->to", inputs, weights[ids])
        reference = torch.autograd.grad(expected, (inputs, weights), gradient)
        torch.testing.assert_close(output, expected)
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right)
        expected_groups = [0, 0, 2] if compiling or sum(counts) else [0, 0]
        self.assertEqual([call.kwargs["group_type"] for call in self.kernel.call_args_list], expected_groups)

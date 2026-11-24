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
"""CPU contracts for grouped-kernel argument conversion and adapter autograd."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops.impl import grouped_matmul, grouped_swiglu
from tests.common.op_precision_suite import precision_cases


def _cuda_grouped_mm(x, weight, *, offs):
    """Emulate only the grouped kernel boundary to test the adapter's data flow."""
    assert offs.dtype == torch.int32, f"Expected int32 CUDA offsets, got {offs.dtype}"
    return grouped_matmul._native(x, weight, offs)  # pylint: disable=protected-access


def _npu_grouped_mm(x, weight, **kwargs):
    """Emulate cumulative-boundary NPU GEMMs without importing an NPU runtime."""
    offsets = kwargs["group_list"]
    assert offsets.dtype == torch.int64, f"Expected int64 NPU offsets, got {offsets.dtype}"
    return grouped_matmul._native(x, weight, offsets)  # pylint: disable=protected-access


def _evaluate(operation, args):
    """Evaluate the adapter using independent leaves and collect all input VJPs."""
    inputs = tuple(arg.detach().clone().requires_grad_(arg.requires_grad) if isinstance(arg, torch.Tensor)
                   else arg for arg in args)
    output = operation(*inputs)
    leaves = [arg for arg in inputs if isinstance(arg, torch.Tensor) and arg.requires_grad]
    gradients = torch.autograd.grad(output.sum(), leaves)
    return output, gradients


class TestGroupedAdapters(unittest.TestCase):
    """Validate layout, offsets, empty experts and routing gradients around kernel calls."""

    def _check_swiglu(self, operation) -> None:
        """Compare adapter results and every VJP with the independent expert oracle."""
        for case in precision_cases("grouped_swiglu", torch.float32):
            with self.subTest(case=case.name):
                expected, expected_grads = _evaluate(case.reference, case.args)
                actual, actual_grads = _evaluate(operation, case.args)
                torch.testing.assert_close(actual, expected)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_cuda_grouped_swiglu_offsets_and_gradients(self) -> None:
        """CUDA adapters use cumulative int32 counts and preserve all expert gradients."""
        with patch.object(torch, "_grouped_mm", side_effect=_cuda_grouped_mm, create=True):
            self._check_swiglu(grouped_swiglu._load_cuda())  # pylint: disable=protected-access

    def test_npu_grouped_swiglu_offsets_and_gradients(self) -> None:
        """NPU adapters preserve transpose views, routing weights and empty-expert VJPs."""
        module_name = "hyper_parallel.components.functional.grouped_matmul"
        module = ModuleType(module_name)
        module.grouped_matmul = _npu_grouped_mm
        extension = SimpleNamespace(npu_grouped_matmul=_npu_grouped_mm)
        with patch.dict(sys.modules, {module_name: module}), patch.object(
            grouped_swiglu, "require_package", return_value=extension,
        ):
            self._check_swiglu(grouped_swiglu._load_npu())  # pylint: disable=protected-access

    def test_cuda_grouped_matmul_offset_encoding(self) -> None:
        """The grouped GEMM adapter converts incoming int64 ends to CUDA int32 ends."""
        x = torch.arange(6.0).reshape(3, 2)
        weights = torch.arange(12.0).reshape(2, 2, 3)
        ends = torch.tensor([1, 3])
        with patch.object(torch, "_grouped_mm", side_effect=_cuda_grouped_mm, create=True):
            operation = grouped_matmul._load_cuda()  # pylint: disable=protected-access
            actual = operation(x, weights, ends)
        torch.testing.assert_close(actual, torch.cat((x[:1] @ weights[0], x[1:] @ weights[1])))
        self.assertEqual(ends.dtype, torch.int64)

    def test_cuda_compatibility_rejects_dtype_and_old_hardware(self) -> None:
        """Only BF16 inputs on compute capability 8.0 or later are compatible."""
        for dtype, capability, expected in ((torch.float32, (9, 0), "BF16"),
                                            (torch.bfloat16, (7, 5), "8.0"),
                                            (torch.bfloat16, (8, 0), None)):
            with self.subTest(dtype=dtype, capability=capability), patch.object(
                torch.cuda, "get_device_capability", return_value=capability,
            ):
                reason = grouped_matmul.check_torch_grouped_mm(torch.ones(1, dtype=dtype))
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)

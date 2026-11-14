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
"""CPU checks for the clipped CANN SwiGLU adapter contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    LowPrecisionCapabilityError, MXFP8NpuOps, _get_clamped_swiglu_ops,
)

from tests.common.mark_utils import arg_mark


class TestClippedSwiGLUAdapter(unittest.TestCase):
    """Keep optional registration and the forward/backward ABI explicit."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_clipped_forward_backward_contract(self) -> None:
        """
        Feature: CANN clipped MXFP8 adapter.
        Description: Translate tensor order, dtype enums, and optional forward output.
        Expectation: Both kernels see the same clip and groups, without autograd inputs or route weights.
        """
        operators = Mock()
        buffers = tuple(torch.randn(2) for _ in range(5))
        operators.swiglu_group_quant_with_dual_axis.return_value = buffers
        operators.swiglu_backward_group_quant_with_dual_axis.return_value = buffers[:4]
        adapter = object.__new__(MXFP8NpuOps)
        projected = torch.randn(4, 8, requires_grad=True)
        gradient = torch.randn(4, 4)
        groups = torch.tensor([2, 2, 4])
        with patch("hyper_parallel.components.quantization.ops.npu_mxfp8._get_clamped_swiglu_ops",
                   return_value=operators):
            adapter.validate_swiglu_fusion(clamp_limit=10.0)
            actual = adapter.swiglu_mx_quant_dual_axis(projected, group_index=groups, clamp_limit=10.0)
            backward = adapter.swiglu_backward_mx_quant_dual_axis(
                projected, gradient, group_index=groups, clamp_limit=10.0)
        self.assertEqual(len(actual), 4)
        self.assertEqual(len(backward), 4)
        forward_call = operators.swiglu_group_quant_with_dual_axis.call_args
        backward_call = operators.swiglu_backward_group_quant_with_dual_axis.call_args
        self.assertFalse(forward_call.args[0].requires_grad)
        torch.testing.assert_close(backward_call.args[0], gradient)
        torch.testing.assert_close(backward_call.args[1], projected)
        self.assertEqual(forward_call.kwargs, {
            "group_index": groups, "dst_type": 292, "quant_mode": 1, "clamp_limit": 10.0})
        self.assertEqual(backward_call.kwargs, {
            "group_index": groups, "dst_type": 36, "quant_mode": 1, "clamp_limit": 10.0})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_missing_backward_is_reported_at_setup(self) -> None:
        """
        Feature: Optional clipped SwiGLU dependency.
        Description: Model setup sees a package registering only the forward operator.
        Expectation: Missing backward is reported before training can begin.
        """
        _get_clamped_swiglu_ops.cache_clear()
        self.addCleanup(_get_clamped_swiglu_ops.cache_clear)
        namespace = SimpleNamespace(swiglu_group_quant_with_dual_axis=Mock())
        with patch.dict("sys.modules", {"cann_ops_nn": SimpleNamespace()}), patch(
                "hyper_parallel.components.quantization.ops.npu_mxfp8.torch.ops",
                SimpleNamespace(cann_ops_nn=namespace)):
            with self.assertRaisesRegex(LowPrecisionCapabilityError, "swiglu_backward_group_quant_with_dual_axis"):
                _get_clamped_swiglu_ops()

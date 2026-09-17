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
"""CPU tests for direct mixed-precision parameter copies."""

import unittest

import torch

from hyper_parallel.components.optim.mixed_precision_optimizer import _copy_tensor
from tests.common.mark_utils import arg_mark


class TestParameterCopy(unittest.TestCase):
    """Preserve parameter storage and dtype conversion across copy directions."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_parameter_copy_converts_dtype_without_replacing_storage(self) -> None:
        """Feature: Direct parameter dtype conversion.

        Description: Copy strided tensors between FP32 and FP16/BF16 without replacing storage.
        Expectation: Values are exact and the destination pointer and strides are preserved.
        """
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                source = torch.linspace(-2, 3, 30).reshape(6, 5).t()
                destination = torch.empty(6, 5, dtype=dtype).t()
                pointer = destination.data_ptr()
                _copy_tensor(destination, source)
                torch.testing.assert_close(destination, source.to(dtype), rtol=0, atol=0)
                self.assertEqual(destination.data_ptr(), pointer)
                self.assertFalse(destination.is_contiguous())
                restored = torch.empty_like(source, dtype=torch.float32)
                _copy_tensor(restored, destination)
                torch.testing.assert_close(restored, destination.float(), rtol=0, atol=0)

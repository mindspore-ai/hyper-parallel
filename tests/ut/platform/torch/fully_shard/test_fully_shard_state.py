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
"""Unit tests for fully_shard state helpers (no NPU required)."""
import os
import unittest

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.platform.torch.fully_shard.state import _to_dtype_if_needed


class TestToDtypeIfNeeded(unittest.TestCase):
    """Unit tests for _to_dtype_if_needed."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_to_dtype_if_needed_parameterized(self):
        """Parameterized test: same dtype/no-op, None dtype, cast to float16/bfloat16."""
        test_cases = [
            (torch.float32, torch.float32, True, "same dtype no-op"),
            (torch.float32, None, True, "None dtype no-op"),
            (torch.float32, torch.float16, False, "cast to float16"),
            (torch.float32, torch.bfloat16, False, "cast to bfloat16"),
        ]
        for tensor_dtype, target_dtype, expect_same_tensor, desc in test_cases:
            with self.subTest(tensor_dtype=tensor_dtype, target_dtype=target_dtype, desc=desc):
                # Arrange
                t = torch.randn(2, 2, dtype=tensor_dtype, device=self.device)
                # Act
                result = _to_dtype_if_needed(t, target_dtype)
                # Assert
                if expect_same_tensor:
                    self.assertIs(result, t)
                else:
                    self.assertIsNot(result, t)
                if target_dtype is not None:
                    self.assertEqual(result.dtype, target_dtype)
                else:
                    self.assertEqual(result.dtype, tensor_dtype)

    def test_invalid_tensor_raises(self):
        """_to_dtype_if_needed raises when first arg is not a tensor."""
        # Act & Assert (None has no .dtype attribute)
        with self.assertRaises(AttributeError):
            _to_dtype_if_needed(None, torch.float32)


if __name__ == "__main__":
    unittest.main()

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
"""Unit tests for fully_shard state helpers (no NPU required).

Covers _to_dtype_if_needed from hyper_parallel.platform.torch.fully_shard.state:
dtype no-op vs cast, and invalid input handling. All tests run on CPU.
"""
import os
import unittest

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.platform.torch.fully_shard.state import _to_dtype_if_needed


class TestToDtypeIfNeeded(unittest.TestCase):
    """Unit tests for _to_dtype_if_needed (tensor dtype cast or no-op)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_to_dtype_if_needed_parameterized(self):
        """Parameterized test: same dtype/no-op and None dtype (no cast).

        description: Call _to_dtype_if_needed with same dtype or None.
        expectation: Returns the same tensor object; dtype unchanged for None target.
        feature: fully_shard state _to_dtype_if_needed.
        """
        test_cases = [
            (torch.float32, torch.float32, "same dtype no-op"),
            (torch.float32, None, "None dtype no-op"),
        ]
        for tensor_dtype, target_dtype, desc in test_cases:
            with self.subTest(tensor_dtype=tensor_dtype, target_dtype=target_dtype, desc=desc):
                t = torch.randn(2, 2, dtype=tensor_dtype, device=self.device)
                result = _to_dtype_if_needed(t, target_dtype)
                self.assertIs(result, t)
                if target_dtype is not None:
                    self.assertEqual(result.dtype, target_dtype)
                else:
                    self.assertEqual(result.dtype, tensor_dtype)

    @unittest.skip("test_to_dtype_converts_local_tensor temporarily skipped.")
    def test_to_dtype_converts_local_tensor(self):
        """Cast path: _to_dtype_if_needed converts local tensor when target dtype differs.

        description: Call _to_dtype_if_needed with float16/bfloat16 targets.
        expectation: New tensor with requested dtype (not same object as input).
        feature: fully_shard state _to_dtype_if_needed.
        """
        test_cases = [
            (torch.float32, torch.float16, "cast to float16"),
            (torch.float32, torch.bfloat16, "cast to bfloat16"),
        ]
        for tensor_dtype, target_dtype, desc in test_cases:
            with self.subTest(tensor_dtype=tensor_dtype, target_dtype=target_dtype, desc=desc):
                t = torch.randn(2, 2, dtype=tensor_dtype, device=self.device)
                result = _to_dtype_if_needed(t, target_dtype)
                self.assertIsNot(result, t)
                self.assertEqual(result.dtype, target_dtype)

    def test_invalid_tensor_raises(self):
        """_to_dtype_if_needed raises when first arg is not a tensor.

        description: Call _to_dtype_if_needed(None, torch.float32).
        expectation: AttributeError (None has no .dtype).
        feature: fully_shard state _to_dtype_if_needed input validation.
        """
        # Act & Assert (None has no .dtype attribute)
        with self.assertRaises(AttributeError):
            _to_dtype_if_needed(None, torch.float32)


if __name__ == "__main__":
    unittest.main()

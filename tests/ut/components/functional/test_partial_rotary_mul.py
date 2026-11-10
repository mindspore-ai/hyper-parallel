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
"""CPU checks of partial-RoPE adapter gradients and storage ownership."""

import sys
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.partial_rotary_mul import (
    _load_partial_rotary_mul, _PartialRotaryMul, partial_rotary_mul,
)
from hyper_parallel.components.modules.shared_compressed_dsa_attention import _apply_v41_rope


def _reference_forward(tensor, cosine, sine, *, rotary_mode, partial_slice):
    """Stub the device entry so CPU UT checks only the adapter contract."""
    if rotary_mode != "interleave" or partial_slice[1] != tensor.shape[-1]:
        raise ValueError("Unexpected partial-RoPE operator arguments")
    tensor.copy_(_apply_v41_rope(tensor, cosine[:, :, 0, ::2], sine[:, :, 0, ::2], 2))


class TestPartialRotaryMul(unittest.TestCase):
    """Keep native execution separate from CPU-only adapter checks."""

    def test_backward_matches_reference_without_mutating_inputs(self):
        """Feature: Reference-compatible partial-RoPE gradients.
        Description: Exercise dtypes, inverse rotation and strided incoming gradients.
        Expectation: Output and gradients match exactly without input mutation.
        """
        torch.manual_seed(1031)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for head_dim in (8, 16):
                for inverse in (False, True):
                    with self.subTest(dtype=dtype, head_dim=head_dim, inverse=inverse):
                        tensor = torch.randn(2, 7, 3, head_dim, dtype=dtype, requires_grad=True)
                        angles = torch.randn(2, 7, 4)
                        cosine = angles.cos()
                        sine = -angles.sin() if inverse else angles.sin()
                        # This view also checks that backward accepts strided incoming gradients.
                        gradient = torch.randn(2, 3, 7, head_dim, dtype=dtype).transpose(1, 2)
                        original = tensor.detach().clone()
                        original_gradient = gradient.clone()
                        expected = _apply_v41_rope(tensor, cosine, sine, 2)
                        expected_gradient = torch.autograd.grad(expected, tensor, gradient)[0]
                        with patch(
                            "hyper_parallel.components.functional.partial_rotary_mul._load_partial_rotary_mul",
                            return_value=_reference_forward,
                        ):
                            actual = _PartialRotaryMul.apply(
                                tensor, cosine.repeat_interleave(2, -1).unsqueeze(2),
                                sine.repeat_interleave(2, -1).unsqueeze(2),
                            )
                            actual_gradient = torch.autograd.grad(actual, tensor, gradient)[0]
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
                        torch.testing.assert_close(gradient, original_gradient, rtol=0, atol=0)
                        self.assertNotEqual(actual.data_ptr(), tensor.data_ptr())

    def test_explicit_fusion_rejects_cpu_and_trainable_positions(self):
        """Feature: Explicit fused execution.
        Description: Request fusion on CPU or with trainable position coefficients.
        Expectation: Invalid calls raise before executing any device operator.
        """
        tensor = torch.ones(1, 7, 2, 16)
        cosine = torch.ones(1, 7, 4)
        with self.assertRaisesRegex(RuntimeError, "requires NPU inputs"):
            partial_rotary_mul(tensor, cosine, cosine, 2)
        with self.assertRaisesRegex(ValueError, "constant cos/sin"):
            partial_rotary_mul(tensor, cosine.requires_grad_(), cosine, 2)

    def test_missing_extension_preserves_cause_and_installation_hint(self):
        """Feature: Optional partial-RoPE dependency reporting.
        Description: Import the callable while its extension is unavailable.
        Expectation: The error preserves its cause and points to installation guidance.
        """
        _load_partial_rotary_mul.cache_clear()
        try:
            with patch.dict(sys.modules, {"cann_ops_transformer": None}):
                with self.assertRaisesRegex(RuntimeError, "inplace_partial_rotary_mul.*README") as caught:
                    _load_partial_rotary_mul()
            self.assertIsInstance(caught.exception.__cause__, ImportError)
        finally:
            _load_partial_rotary_mul.cache_clear()

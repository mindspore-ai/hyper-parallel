# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Dense reference attention must keep empty rows finite in forward and backward."""
import unittest

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.torch_attention import torch_attention


class TestTorchAttention(unittest.TestCase):
    """Check padding and right-aligned causal windows against an explicit SDPA oracle."""

    def test_visible_rows_and_gradients(self):
        """Feature: Portable dense attention visibility.

        Description: Compare empty padding rows and unequal-length windows, including GQA gradients.
        Expectation: Eager and SDPA outputs and gradients agree with the explicit visibility mask.
        """
        with torch.random.fork_rng():
            torch.manual_seed(19)
            for query_length, key_length, window in ((3, 3, None), (2, 5, 2), (5, 3, None)):
                query = torch.randn(2, 4, query_length, 8, requires_grad=True)
                key = torch.randn(2, 2, key_length, 8, requires_grad=True)
                value = torch.randn(2, 2, key_length, 6, requires_grad=True)
                padding = torch.ones(2, key_length, dtype=torch.bool)
                padding[0] = False
                q_positions = torch.arange(query_length) + key_length - query_length
                k_positions = torch.arange(key_length)
                visible = k_positions[None, :] <= q_positions[:, None]
                if window is not None:
                    visible &= k_positions[None, :] > q_positions[:, None] - window
                visible = padding[:, None, None, :] & visible
                reference = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                    query, key, value, attn_mask=visible, enable_gqa=True,
                ).transpose(1, 2)
                gradient = torch.randn_like(reference)
                expected_grads = torch.autograd.grad(reference, (query, key, value), gradient)
                for implementation in ("eager", "sdpa"):
                    with self.subTest(lengths=(query_length, key_length), implementation=implementation):
                        output = torch_attention(query, key, value, padding,
                                                 sliding_window=window, implementation=implementation)
                        grads = torch.autograd.grad(output, (query, key, value), gradient)
                        torch.testing.assert_close(output, reference, atol=1e-6, rtol=1e-5)
                        self.assertEqual(torch.count_nonzero(output[0]).item(), 0)
                        for actual, expected in zip(grads, expected_grads):
                            self.assertTrue(torch.isfinite(actual).all())
                            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

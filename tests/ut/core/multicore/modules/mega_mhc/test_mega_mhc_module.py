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
"""CPU unit tests for MegaMhc semantics and static validation."""

import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_mhc.module import MegaMhc
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc


class TestMegaMhc(unittest.TestCase):
    """Validate shifted mapping semantics independently of NPU kernels."""

    def test_torch_golden_uses_previous_mappings(self) -> None:
        """Use previous mappings at the shifted boundary.

        Feature: Shifted mHC semantics.
        Description: Evaluate a deterministic identity-style semantic case.
        Expectation: B/C update residuals and previous A produces the block input.
        """
        torch.manual_seed(7)
        tokens = 3
        hidden_size = 128
        residual = torch.randn(tokens, 4, hidden_size).to(torch.bfloat16)
        previous_output = torch.randn(tokens, hidden_size).to(torch.bfloat16)
        previous_pre = torch.zeros(tokens, 4)
        previous_pre[:, 0] = 1.0
        previous_post = torch.zeros(tokens, 4)
        previous_res = torch.eye(4).expand(tokens, -1, -1).clone()
        phi = torch.zeros(24, 4 * hidden_size)
        alpha = torch.ones(3)
        bias = torch.zeros(24)
        norm_weight = torch.ones(hidden_size, dtype=torch.bfloat16)

        new_residual, next_pre, next_post, next_res, block_input = torch_mega_mhc(
            previous_output,
            residual,
            previous_pre,
            previous_post,
            previous_res,
            phi,
            alpha,
            bias,
            norm_weight,
        )

        torch.testing.assert_close(new_residual, residual)
        torch.testing.assert_close(next_pre, torch.full_like(next_pre, 0.500001))
        torch.testing.assert_close(next_post, torch.ones_like(next_post))
        torch.testing.assert_close(
            next_res.sum(dim=-1),
            torch.ones_like(next_res.sum(dim=-1)),
            atol=2e-5,
            rtol=2e-5,
        )
        expected_input = residual[:, 0].float()
        expected_input = expected_input * torch.rsqrt(
            expected_input.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        torch.testing.assert_close(
            block_input,
            expected_input.to(torch.bfloat16),
            atol=0,
            rtol=0,
        )

    def test_torch_golden_preserves_leading_token_dimensions(self) -> None:
        """Preserve arbitrary leading token dimensions.

        Feature: Shifted mHC tensor shapes.
        Description: Evaluate the semantic reference with two leading dimensions.
        Expectation: Every output restores the original leading dimensions.
        """
        shape = (2, 3)
        hidden_size = 8
        residual = torch.zeros(*shape, 4, hidden_size, dtype=torch.bfloat16)
        previous_output = torch.zeros(*shape, hidden_size, dtype=torch.bfloat16)
        previous_pre = torch.full((*shape, 4), 0.25)
        previous_post = torch.ones(*shape, 4)
        previous_res = torch.eye(4).expand(*shape, 4, 4).clone()
        outputs = torch_mega_mhc(
            previous_output,
            residual,
            previous_pre,
            previous_post,
            previous_res,
            torch.zeros(24, 4 * hidden_size),
            torch.ones(3),
            torch.zeros(24),
            torch.ones(hidden_size, dtype=torch.bfloat16),
        )
        self.assertEqual(tuple(outputs[0].shape), (*shape, 4, hidden_size))
        self.assertEqual(tuple(outputs[1].shape), (*shape, 4))
        self.assertEqual(tuple(outputs[3].shape), (*shape, 4, 4))
        self.assertEqual(tuple(outputs[4].shape), (*shape, hidden_size))

    def test_module_rejects_unsupported_static_shapes(self) -> None:
        """Reject shapes outside the native static contract.

        Feature: MegaMhc input validation.
        Description: Construct modules with unsupported hidden alignment and iterations.
        Expectation: Both invalid configurations raise descriptive errors.
        """
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            MegaMhc(127)
        with self.assertRaisesRegex(ValueError, "num_iters=20"):
            MegaMhc(128, num_iters=2)
        with self.assertRaisesRegex(ValueError, "hidden_size <= 5760"):
            MegaMhc(7168)
        for token_tile in (0, -1, 1.5, True):
            with self.subTest(token_tile=token_tile), self.assertRaisesRegex(ValueError, "token_tile"):
                MegaMhc(128, token_tile=token_tile)


if __name__ == "__main__":
    unittest.main()

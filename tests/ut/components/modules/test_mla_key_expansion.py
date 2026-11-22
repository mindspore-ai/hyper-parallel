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
"""Shared-K broadcast layout preserves each source's BF16 gradient reduction."""
import unittest

import torch

from hyper_parallel.components.modules import MultiLatentAttentionModule


class TestMlaKeyExpansion(unittest.TestCase):
    """Check independent layout/gradient oracles and local/CP option propagation."""

    def test_static_default_and_both_gradient_layouts(self) -> None:
        """Both choices retain source values, storage axes and their own backward path."""
        torch.manual_seed(20261122)
        for layout in ("bsnd", "bnsd"):
            with self.subTest(layout=layout):
                content = torch.randn(2, 7, 4, 3, dtype=torch.bfloat16, requires_grad=True)
                shared = torch.randn(2, 7, 5, dtype=torch.bfloat16, requires_grad=True)
                left, right = content.detach().clone().requires_grad_(), shared.detach().clone().requires_grad_()
                if layout == "bsnd":
                    expected = torch.cat((left, right.unsqueeze(2).expand(2, 7, 4, 5)), dim=-1)
                    actual = MultiLatentAttentionModule.assemble_key(content, shared)
                else:
                    expected = torch.empty(2, 4, 7, 8, dtype=torch.bfloat16)
                    expected[..., :3] = left.transpose(1, 2)
                    expected[..., 3:] = right[:, None].expand(2, 4, 7, 5)
                    expected = expected.transpose(1, 2)
                    actual = MultiLatentAttentionModule.assemble_key(content, shared, layout=layout)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.stride(), expected.stride())
                upstream = torch.randn_like(expected)
                reference = torch.autograd.grad(expected, (left, right), upstream)
                gradients = torch.autograd.grad(actual, (content, shared), upstream)
                for value, target in zip(gradients, reference):
                    torch.testing.assert_close(value, target, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "layout"):
            MultiLatentAttentionModule.assemble_key(content, shared, layout="unknown")

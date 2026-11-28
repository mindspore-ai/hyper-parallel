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
"""Regression coverage for shared functional policies and legacy operator adapters."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.functional.moe_token_permute import moe_token_permute
from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute
from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb_bsnd
from hyper_parallel.components.functional.swiglu import local_swiglu, swiglu_activation
from hyper_parallel.components.ops.impl import rope as rope_impl


class TestOpComposition(unittest.TestCase):
    """Keep compatibility metadata, execution count and precision boundaries explicit."""

    def setUp(self) -> None:
        """Isolate process-wide backend state."""
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        operators.ops.set_backends({"rope": "native", "swiglu": "native",
                                    "moe_token_permute": "native", "moe_token_unpermute": "native"})

    def test_legacy_token_roundtrip_and_route_gradients(self) -> None:
        """Legacy int32 inverse maps retain weighted restoration and optional top-k=1."""
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                states = torch.randn(3, 4, requires_grad=True)
                indices = torch.tensor([[2, 0], [0, 1], [2, 1]])[:, :top_k]
                weights = torch.randn(3, top_k, requires_grad=True)
                permuted, positions = moe_token_permute(states, indices)
                self.assertEqual(positions.dtype, torch.int32)
                actual = moe_token_unpermute(permuted, positions, weights)
                expected = states * weights.sum(-1, keepdim=True)
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                torch.testing.assert_close(states.grad, weights.detach().sum(-1, keepdim=True).expand_as(states))
                torch.testing.assert_close(weights.grad, states.detach().sum(-1, keepdim=True).expand_as(weights))
                if top_k == 1:
                    torch.testing.assert_close(moe_token_unpermute(permuted, positions), states)

    def test_single_interleaved_runs_one_kernel(self) -> None:
        """Single-input NPU adaptation launches once and preserves the nonrotary tail."""
        states = torch.randn(2, 3, 4, 12)
        cos, sin = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
        kernel = Mock(side_effect=lambda x, *_args, **_kwargs: x)
        with patch.object(rope_impl, "require_package", return_value=SimpleNamespace(npu_rotary_mul=kernel)):
            operation = rope_impl._load_npu_interleaved_single()  # pylint: disable=protected-access
            actual = operation(states, cos, sin, unsqueeze_dim=2)
        kernel.assert_called_once()
        self.assertEqual(actual.shape, states.shape)
        torch.testing.assert_close(actual[..., 8:], states[..., 8:])

    def test_bsnd_policy_and_single_branch_gradients(self) -> None:
        """Shared BSND adapter matches independent adjacent-pair math with rounded frequencies."""
        states = torch.randn(2, 3, 4, 12, dtype=torch.float64, requires_grad=True)
        angles = torch.randn(1, 3, 4, dtype=torch.float64).repeat(1, 1, 2)
        actual = apply_rotary_pos_emb_bsnd(states, (angles.cos(), angles.sin()), interleaved=True)
        even, odd = states[..., :8:2], states[..., 1:8:2]
        cos, sin = angles[..., :4].cos().unsqueeze(2), angles[..., :4].sin().unsqueeze(2)
        expected = torch.cat((even * cos - odd * sin, odd * cos + even * sin, states[..., 8:]), -1)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertIsNotNone(states.grad)
        self.assertIs(apply_rotary_pos_emb_bsnd(states, None), states)

    def test_swiglu_rounding_and_clamp_policies(self) -> None:
        """Separate and packed policies preserve input or FP32 activation and return dtypes."""
        gate, up = torch.randn(2, 8, dtype=torch.bfloat16), torch.randn(2, 8, dtype=torch.bfloat16)
        for dtype in ("input", "float32"):
            for limit in (0.0, 0.5):
                with self.subTest(dtype=dtype, limit=limit):
                    g, u = (gate.float(), up.float()) if dtype == "float32" else (gate, up)
                    if limit > 0:
                        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
                    expected = torch.nn.functional.silu(g) * u
                    actual = swiglu_activation(gate, up, limit=limit, activation_dtype=dtype)
                    torch.testing.assert_close(actual, expected)
                    packed = local_swiglu(torch.cat((gate, up), -1), limit=limit, activation_dtype=dtype)
                    torch.testing.assert_close(packed, expected.to(gate.dtype))

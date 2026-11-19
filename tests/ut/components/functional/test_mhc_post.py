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
"""CPU coverage for the MHC CANN post interface."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.mhc_post import mhc_post, weighted_stream_sum

B, S, N, C = 2, 3, 4, 128


def _fake_npu_device():
    """Pretend every tensor lives on an NPU, as hardware-free UTs cannot own one."""
    return patch.object(torch.Tensor, "device", property(lambda _tensor: SimpleNamespace(type="npu")))


class TestMhcPost(unittest.TestCase):
    """Check the fused CANN operator receives the expected stream contract."""

    def test_cann_post_passes_stream_contract(self):
        """
        Feature: CANN MHC post contract.
        Description: Invoke mhc_post with the default CANN backend on NPU.
        Expectation: Pass residual streams, residual matrix, block output and
            post coefficients in the operator order, and return the flattened
            mixed streams.
        """
        block_output = torch.randn(B, S, C)
        residual = torch.randn(B, S, N * C)
        h_post = torch.randn(B, S, N)
        h_res = torch.randn(B, S, N, N)
        mixed = torch.randn(B, S, N, C)
        ops = ModuleType("cann_ops_transformer.ops")
        cann_post = Mock(return_value=mixed)
        ops.mhc_post = cann_post
        package = ModuleType("cann_ops_transformer")
        package.ops = ops
        with patch.dict(sys.modules, {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}), \
                _fake_npu_device():
            result = mhc_post(block_output, residual, h_post, h_res, N)

        args = cann_post.call_args.args
        self.assertEqual(args[0].shape, (B, S, N, C))
        self.assertIs(args[1], h_res)
        self.assertIs(args[2], block_output)
        self.assertIs(args[3], h_post)
        self.assertEqual(result.shape, (B, S, N * C))

    def test_cpu_eager_matches_formula(self):
        """
        Feature: Eager MHC post fallback.
        Description: Invoke mhc_post on CPU without any vendor package.
        Expectation: Match the explicit post-mapping formula for every stream.
        """
        block_output = torch.randn(B, S, C)
        residual = torch.randn(B, S, N * C)
        h_post = torch.randn(B, S, N)
        h_res = torch.randn(B, S, N, N)
        result = mhc_post(block_output, residual, h_post, h_res, N)

        expected = h_post.unsqueeze(-1) * block_output.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * residual.reshape(B, S, N, C).unsqueeze(-2), dim=2
        )
        torch.testing.assert_close(result, expected.flatten(2))

    def test_cpu_eager_preserves_output_and_gradients(self):
        """
        Feature: Eager MHC post on CPU.
        Description: Run the CPU fallback with either supported backend.
        Expectation: Bypass vendor operators and match eager outputs and gradients.
        """
        for backend in ("cann", "omni", "torch_npu"):
            with self.subTest(backend=backend):
                inputs = (
                    torch.randn(B, S, C, requires_grad=True),
                    torch.randn(B, S, N * C, requires_grad=True),
                    torch.randn(B, S, N, requires_grad=True),
                    torch.randn(B, S, N, N, requires_grad=True),
                )
                block_output, residual, h_post, h_res = inputs
                expected = (
                    torch.einsum("bsn,bsc->bsnc", h_post, block_output)
                    + torch.einsum("bsij,bsic->bsjc", h_res, residual.reshape(B, S, N, C))
                ).flatten(2)
                with patch.dict(sys.modules, {"cann_ops_transformer": None, "omni_training_custom_ops": None}):
                    result = mhc_post(*inputs, N, backend=backend)
                torch.testing.assert_close(result, expected)
                grad_output = torch.randn_like(result)
                actual_grads = torch.autograd.grad(result, inputs, grad_output)
                expected_grads = torch.autograd.grad(expected, inputs, grad_output)
                for actual, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, reference)

    def test_omni_backend_is_deprecated(self):
        """
        Feature: Legacy omni MHC post backend.
        Description: Invoke mhc_post with backend='omni' on NPU.
        Expectation: Log a deprecation warning and run the legacy post operator
            through torch.ops.custom.
        """
        block_output = torch.randn(B, S, C)
        residual = torch.randn(B, S, N * C)
        h_post = torch.randn(B, S, N)
        h_res = torch.randn(B, S, N, N)
        mixed = torch.randn(B, S, N, C)
        omni = ModuleType("omni_training_custom_ops")
        custom = Mock()
        custom.npu_ai_infra_manifold_constrained_hyper_connection_post = Mock(return_value=mixed)
        with patch.dict(sys.modules, {"omni_training_custom_ops": omni}), \
                patch.object(torch.ops, "custom", custom, create=True), \
                _fake_npu_device(), \
                patch("hyper_parallel.components.functional.mhc_post.logger") as fake_logger:
            result = mhc_post(block_output, residual, h_post, h_res, N, backend="omni")

        fake_logger.warning.assert_called_once()
        self.assertIn("deprecated", fake_logger.warning.call_args.args[0])
        self.assertEqual(result.shape, (B, S, N * C))
        args = custom.npu_ai_infra_manifold_constrained_hyper_connection_post.call_args.args
        self.assertEqual(args[0].shape, (B, S, N, C))
        self.assertIs(args[1], h_res)
        self.assertIs(args[2], block_output)
        self.assertIs(args[3], h_post)

    def test_rejects_unknown_backend(self):
        """
        Feature: MHC post backend validation.
        Description: Invoke mhc_post with an unsupported backend name.
        Expectation: Raise ValueError listing the supported backends.
        """
        with self.assertRaisesRegex(ValueError, "backend must be"):
            mhc_post(
                torch.randn(B, S, C), torch.randn(B, S, N * C), torch.randn(B, S, N),
                torch.randn(B, S, N, N), N, backend="eager",
            )


class TestWeightedStreamSum(unittest.TestCase):
    """Check arbitrary delayed pre coefficients through the native Post bridge."""

    def test_native_reduction_and_gradients(self):
        """Match an independent weighted sum including coefficient gradients."""
        def post(x: torch.Tensor, combine: torch.Tensor, output: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
            """Compute the Post formula independently of the native wrapper."""
            return (torch.einsum("bsij,bsih->bsjh", combine, x.float())
                    + output.float().unsqueeze(-2) * weights.unsqueeze(-1)).to(x.dtype)

        def backward(grad: torch.Tensor, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Use Torch autograd as the mock native derivative."""
            with torch.enable_grad():
                copies = [value.detach().requires_grad_() for value in inputs]
                return torch.autograd.grad(post(*copies), copies, grad)

        x = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.randn(2, 3, 4, requires_grad=True)
        expected = (x.float() * weights.unsqueeze(-1)).sum(-2).to(x.dtype)
        native = ModuleType("torch_npu")
        native.npu_mhc_post = post
        native.npu_mhc_post_backward = backward
        with patch.dict(sys.modules, {"torch_npu": native}), _fake_npu_device():
            actual = weighted_stream_sum(x, weights)
            upstream = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(actual, (x, weights), upstream)
        expected_grads = torch.autograd.grad(expected, (x, weights), upstream)
        torch.testing.assert_close(actual, expected)
        for value, oracle in zip(actual_grads, expected_grads):
            torch.testing.assert_close(value, oracle)

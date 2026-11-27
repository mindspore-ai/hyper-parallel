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
"""CPU coverage for configuration-driven MHC post dispatch."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.ops import ops as op_selector

B, S, N, C = 2, 3, 4, 128


def _fake_npu_device():
    """Pretend every tensor lives on an NPU, as hardware-free UTs cannot own one."""
    return patch.object(torch.Tensor, "device", property(lambda _tensor: torch.device("npu:0")))


class TestMhcPost(unittest.TestCase):
    """Check the fused CANN operator receives the expected stream contract."""

    def setUp(self):
        """Isolate backend resolution and usage between adapter mocks."""
        previous = op_selector.get_backends()
        self.addCleanup(op_selector.set_backends, previous)
        op_selector.set_backends({})

    def test_cann_post_passes_stream_contract(self):
        """
        Feature: CANN MHC post contract.
        Description: Invoke mhc_post with the default CANN backend on NPU.
        Expectation: Pass residual streams, residual matrix, block output and
            post coefficients in the operator order, and return the flattened
            mixed streams.
        """
        block_output = torch.randn(B, S, C, dtype=torch.bfloat16)
        residual = torch.randn(B, S, N * C, dtype=torch.bfloat16)
        h_post = torch.randn(B, S, N)
        h_res = torch.randn(B, S, N, N)
        mixed = torch.randn(B, S, N, C)
        ops = ModuleType("cann_ops_transformer.ops")
        ops.mhc_post = Mock(return_value=mixed)
        package = ModuleType("cann_ops_transformer")
        package.ops = ops
        with patch.dict(sys.modules, {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}), \
                _fake_npu_device():
            result = mhc_post(block_output, residual, h_post, h_res, N)

        args = getattr(ops, "mhc_post").call_args.args
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

    def test_fp32_fused_backend_rejects_without_switching(self) -> None:
        """Configured fused backends reject FP32 instead of changing the selected implementation."""
        inputs = (
            torch.randn(B, S, C), torch.randn(B, S, N * C),
            torch.randn(B, S, N), torch.randn(B, S, N, N),
        )
        extension = ModuleType("cann_ops_transformer.ops")
        extension.mhc_post = Mock()
        package = ModuleType("cann_ops_transformer")
        package.ops = extension
        custom = Mock()
        modules = {"cann_ops_transformer": package, "cann_ops_transformer.ops": extension,
                   "omni_training_custom_ops": ModuleType("omni_training_custom_ops")}
        for backend in ("cann", "omni"):
            with self.subTest(backend=backend):
                op_selector.set_backends({"mhc_post": backend})
                with patch.dict(sys.modules, modules), patch.object(torch.ops, "custom", custom), _fake_npu_device():
                    with self.assertRaisesRegex(ValueError, "FP32.*native"):
                        mhc_post(*inputs, N)
                self.assertIn(("mhc_post", "standard", backend), op_selector.used_backends())
                self.assertNotIn(("mhc_post", "standard", "native"), op_selector.used_backends())
        extension.mhc_post.assert_not_called()
        custom.npu_ai_infra_manifold_constrained_hyper_connection_post.assert_not_called()

    def test_fake_joint_graph_replays_forward_and_backward(self):
        """
        Feature: MHC joint graph replay.
        Description: Capture forward and backward with FakeTensor inputs on CPU.
        Expectation: Replay the captured graph with matching output and all input gradients.
        """
        inputs = (
            torch.randn(B, S, C, requires_grad=True),
            torch.randn(B, S, N * C, requires_grad=True),
            torch.randn(B, S, N, requires_grad=True),
            torch.randn(B, S, N, N, requires_grad=True),
        )

        def _joint(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Return the mixed streams and gradients of their sum."""
            result = mhc_post(*args, N)
            return (result, *torch.autograd.grad(result.sum(), args))

        expected = _joint(*inputs)
        graph = make_fx(_joint, tracing_mode="fake")(*inputs)
        actual = graph(*inputs)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)

    def test_configured_omni_backend(self):
        """
        Feature: Configured Omni MHC post backend.
        Description: Select Omni through the process-wide configuration on NPU.
        Expectation: Run the configured post operator through torch.ops.custom.
        """
        block_output = torch.randn(B, S, C, dtype=torch.bfloat16)
        residual = torch.randn(B, S, N * C, dtype=torch.bfloat16)
        h_post = torch.randn(B, S, N)
        h_res = torch.randn(B, S, N, N)
        mixed = torch.randn(B, S, N, C)
        op_selector.set_backends({"mhc_post": "omni"})
        omni = ModuleType("omni_training_custom_ops")
        custom = Mock()
        custom.npu_ai_infra_manifold_constrained_hyper_connection_post = Mock(return_value=mixed)
        with patch.dict(sys.modules, {"omni_training_custom_ops": omni}), \
                patch.object(torch.ops, "custom", custom, create=True), \
                _fake_npu_device():
            result = mhc_post(block_output, residual, h_post, h_res, N)

        self.assertIn(("mhc_post", "standard", "omni"), op_selector.used_backends())
        self.assertEqual(result.shape, (B, S, N * C))
        args = custom.npu_ai_infra_manifold_constrained_hyper_connection_post.call_args.args
        self.assertEqual(args[0].shape, (B, S, N, C))
        self.assertIs(args[1], h_res)
        self.assertIs(args[2], block_output)
        self.assertIs(args[3], h_post)

    def test_rejects_unknown_configured_backend(self):
        """
        Feature: MHC post backend validation.
        Description: Configure mhc_post with an unsupported backend name.
        Expectation: Raise ValueError listing the supported backends.
        """
        with self.assertRaisesRegex(ValueError, "Invalid ops.mhc_post"):
            op_selector.set_backends({"mhc_post": "eager"})

    def test_auto_uses_native_when_fused_dependencies_are_unavailable(self) -> None:
        """Unconfigured functional calls use the registry auto policy, including native fallback."""
        x = torch.ones(1, 2, 3)
        residual = torch.ones(1, 2, 6)
        post = torch.ones(1, 2, 2)
        matrix = torch.ones(1, 2, 2, 2)
        with _fake_npu_device(), patch.dict(sys.modules, {
            "cann_ops_transformer": None, "omni_training_custom_ops": None,
        }):
            actual = mhc_post(x, residual, post, matrix, 2)
        torch.testing.assert_close(actual, torch.full_like(residual, 3))
        self.assertIn(("mhc_post", "standard", "native"), op_selector.used_backends())

    def test_configured_native_controls_functional_dispatch(self) -> None:
        """Functional calls on NPU use the configured native backend without loading CANN."""
        x = torch.ones(1, 2, 3)
        residual = torch.ones(1, 2, 6)
        post = torch.ones(1, 2, 2)
        matrix = torch.ones(1, 2, 2, 2)
        op_selector.set_backends({"mhc_post": "native"})
        with _fake_npu_device(), patch.dict(sys.modules, {"cann_ops_transformer": None}):
            actual = mhc_post(x, residual, post, matrix, 2)
        torch.testing.assert_close(actual, torch.full_like(residual, 3))
        self.assertIn(("mhc_post", "standard", "native"), op_selector.used_backends())

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

from hyper_parallel.components.functional.mhc_post import mhc_post

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
        ops.mhc_post = Mock(return_value=mixed)
        package = ModuleType("cann_ops_transformer")
        package.ops = ops
        with patch.dict(sys.modules, {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}), \
                _fake_npu_device():
            result = mhc_post(block_output, residual, h_post, h_res, N)

        args = ops.mhc_post.call_args.args
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
        with self.assertRaisesRegex(ValueError, "backend must be 'cann' or 'omni'"):
            mhc_post(
                torch.randn(B, S, C), torch.randn(B, S, N * C), torch.randn(B, S, N),
                torch.randn(B, S, N, N), N, backend="eager",
            )

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
"""CPU coverage for the MHC CANN pre interface."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.mhc_pre import mhc_pre

B, S, N, C = 2, 3, 4, 128


class TestMhcPre(unittest.TestCase):
    """Check the fused CANN operator receives the expected Sinkhorn contract."""

    def test_cann_pre_returns_square_residual_matrix(self):
        """
        Feature: CANN MHC pre Sinkhorn contract.
        Description: Invoke mhc_pre with the default CANN backend.
        Expectation: Pass fp32 projections and the stream/iteration settings,
            reshape the flattened residual matrix to [B, S, N, N], and return
            the block input, post coefficients and residual buffer.
        """
        x = torch.randn(B, S, N * C)
        phi = torch.randn(N * N + 2 * N, N * C)
        alpha = torch.randn(3)
        beta = torch.randn(N * N + 2 * N)
        hin = torch.randn(B, S, C)
        h_post = torch.randn(B, S, N)
        h_res = torch.randn(B, S, N * N)
        ops = ModuleType("cann_ops_transformer.ops")
        ops.mhc_pre_sinkhorn = Mock(return_value=(hin, h_post, h_res))
        package = ModuleType("cann_ops_transformer")
        package.ops = ops
        with patch.dict(sys.modules, {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}):
            block_input, post, residual, streams = mhc_pre(
                x, phi, alpha, beta, N, sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6,
            )

        self.assertIs(block_input, hin)
        self.assertIs(post, h_post)
        self.assertEqual(residual.shape, (B, S, N, N))
        self.assertEqual(streams.shape, (B, S, N * C))
        args = ops.mhc_pre_sinkhorn.call_args.args
        self.assertEqual(args[0].shape, (B, S, N, C))
        self.assertEqual(args[1].dtype, torch.float32)
        self.assertEqual(args[4:], (N, 20, 1e-6, 1e-6))

    def test_cann_pre_folds_gamma_into_phi(self):
        """
        Feature: CANN MHC pre gamma support.
        Description: Invoke mhc_pre with an RMS normalization scale.
        Expectation: The CANN operator receives phi scaled by gamma through
            a differentiable product that keeps phi and gamma gradients exact.
        """
        x = torch.randn(B, S, N * C)
        phi = torch.randn(N * N + 2 * N, N * C)
        gamma = torch.randn(N, C)
        hin = torch.randn(B, S, C)
        ops = ModuleType("cann_ops_transformer.ops")
        ops.mhc_pre_sinkhorn = Mock(return_value=(hin, torch.randn(B, S, N), torch.randn(B, S, N * N)))
        package = ModuleType("cann_ops_transformer")
        package.ops = ops
        with patch.dict(sys.modules, {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}):
            mhc_pre(x, phi, torch.randn(3), torch.randn(N * N + 2 * N), N, gamma=gamma)

        folded = ops.mhc_pre_sinkhorn.call_args.args[1]
        torch.testing.assert_close(folded, phi * gamma.reshape(1, -1))

    def test_omni_backend_is_deprecated(self):
        """
        Feature: Legacy omni MHC pre backend.
        Description: Invoke mhc_pre with backend='omni'.
        Expectation: Log a deprecation warning and run the legacy pre and
            Sinkhorn operator pair through torch.ops.custom.
        """
        x = torch.randn(B, S, N * C)
        hin = torch.randn(B, S, C)
        h_post = torch.randn(B, S, N)
        comb = torch.randn(B, S, N, N)
        inv_rms = torch.randn(B, S, 1)
        h_mix = torch.randn(B, S, N * N + 2 * N)
        h_pre = torch.randn(B, S, N)
        sink_out = torch.randn(B, S, N, N)
        norm_out = torch.randn(B, S, N, N)
        sum_out = torch.randn(B, S, N, N)
        omni = ModuleType("omni_training_custom_ops")
        custom = Mock()
        custom.npu_manifold_constrained_hyper_connection_pre = Mock(
            return_value=(hin, h_post, comb, inv_rms, h_mix, h_pre))
        custom.npu_sinkhorn = Mock(return_value=(sink_out, norm_out, sum_out))
        with patch.dict(sys.modules, {"omni_training_custom_ops": omni}), \
                patch.object(torch.ops, "custom", custom, create=True), \
                patch("hyper_parallel.components.functional.mhc_pre.logger") as fake_logger:
            block_input, post, residual, streams = mhc_pre(
                x, torch.randn(N * N + 2 * N, N * C), torch.randn(3), torch.randn(N * N + 2 * N), N,
                backend="omni",
            )

        fake_logger.warning.assert_called_once()
        self.assertIn("deprecated", fake_logger.warning.call_args.args[0])
        self.assertIs(block_input, hin)
        self.assertEqual(post.shape, (B, S, N))
        self.assertEqual(residual.shape, (B, S, N, N))
        self.assertEqual(streams.shape, (B, S, N * C))
        custom.npu_sinkhorn.assert_called_once()

    def test_rejects_unknown_backend(self):
        """
        Feature: MHC pre backend validation.
        Description: Invoke mhc_pre with an unsupported backend name.
        Expectation: Raise ValueError listing the supported backends.
        """
        with self.assertRaisesRegex(ValueError, "backend must be 'cann' or 'omni'"):
            mhc_pre(
                torch.randn(B, S, N * C), torch.randn(N * N + 2 * N, N * C),
                torch.randn(3), torch.randn(N * N + 2 * N), N, backend="eager",
            )

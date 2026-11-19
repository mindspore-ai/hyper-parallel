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
"""CPU oracle checks for independent pipelined MHC coefficient gradients."""

import importlib
import unittest
from unittest.mock import patch

import torch

mhc = importlib.import_module("hyper_parallel.components.functional.mhc_coefficients")


def _reference(x, phi, alpha, bias, norm_eps=1e-6, hc_eps=1e-6):
    flat = x.flatten(-2).float()
    normalized = flat / torch.sqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    streams = x.shape[-2]
    mix = normalized @ phi.T
    pre, post, residual = mix.split((streams, streams, streams * streams), dim=-1)
    pre_bias, post_bias, residual_bias = bias.split((streams, streams, streams * streams))
    return (
        torch.sigmoid(pre * alpha[0] + pre_bias) + hc_eps,
        2 * torch.sigmoid(post * alpha[1] + post_bias),
        (residual * alpha[2] + residual_bias).unflatten(-1, (streams, streams)),
    )


def _mock_native_pre(x, phi, alpha, bias, *, norm_eps, hc_eps, out_flag):
    del out_flag
    flat = x.flatten(-2).float()
    inv = torch.rsqrt(flat.square().mean(-1) + norm_eps)
    raw = flat @ phi.T
    mix = raw * inv.unsqueeze(-1)
    streams = x.shape[-2]
    pre, post, residual = mix.split((streams, streams, streams * streams), -1)
    bp, bq, bc = bias.split((streams, streams, streams * streams))
    pre = torch.sigmoid(pre * alpha[0] + bp) + hc_eps
    post = 2 * torch.sigmoid(post * alpha[1] + bq)
    residual = (residual * alpha[2] + bc).unflatten(-1, (streams, streams))
    collapsed = (x.float() * pre.unsqueeze(-1)).sum(-2).to(x.dtype)
    return collapsed, post, residual, inv, raw, pre


def _mock_native_rms_backward(dy, x, gamma, inv):
    scaled = dy * gamma
    dx = inv * scaled - x * inv.pow(3) * (scaled * x).mean(-1, keepdim=True)
    return dx, torch.zeros_like(gamma)


class TestMhcCoefficients(unittest.TestCase):
    """Check independent coefficient adjoints against Torch autograd."""

    def test_independent_pre_post_and_residual_gradients_match_autograd(self):
        for dtype in (torch.float32, torch.bfloat16):
            for selected in (None, 0, 1, 2):
                for streams in (4, 6):
                    with self.subTest(dtype=dtype, selected=selected, streams=streams):
                        torch.manual_seed(670)
                        x = torch.randn(2, 3, streams, 8, dtype=dtype, requires_grad=True)
                        phi = (torch.randn(streams * (streams + 2), streams * 8) * .2).requires_grad_()
                        alpha = torch.tensor([.01, .7, 1.3], requires_grad=True)
                        bias = torch.randn(streams * (streams + 2), requires_grad=True)
                        inputs = (x, phi, alpha, bias)
                        expected_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
                        with patch.object(mhc, "_native_pre", _mock_native_pre), \
                                patch.object(mhc, "_native_rms_backward", _mock_native_rms_backward):
                            actual = mhc._MhcCoefficients.apply(*inputs, 1e-6, .03)
                            expected = _reference(*expected_inputs, hc_eps=.03)
                            upstream = [torch.randn_like(t) if selected is None or selected == i
                                        else torch.zeros_like(t) for i, t in enumerate(actual)]
                            actual_grads = torch.autograd.grad(actual, inputs, upstream)
                        expected_grads = torch.autograd.grad(expected, expected_inputs, upstream)
                        for actual_value, expected_value in zip(actual, expected):
                            torch.testing.assert_close(actual_value, expected_value, atol=1e-6, rtol=1e-5)
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=.01)

    def test_public_native_function_rejects_cpu(self):
        with self.assertRaisesRegex(ValueError, "on NPU"):
            mhc.mhc_coefficients(torch.zeros(1, 3, 4, 8), torch.zeros(24, 32),
                                 torch.ones(3), torch.zeros(24))

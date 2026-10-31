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
"""Single-NPU checks of the public fused mHC interface."""

import importlib
import os
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import


pre_module = importlib.import_module("hyper_parallel.components.functional.mhc_pre")
post_module = importlib.import_module("hyper_parallel.components.functional.mhc_post")


def _device() -> torch.device:
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    return device


def test_missing_fused_operators() -> None:
    """Both public entry points report a missing package on the target device."""
    device = _device()
    streams, hidden = 4, 128
    x = torch.zeros((1, 1, hidden), device=device, dtype=torch.bfloat16)
    residual = torch.zeros((1, 1, streams * hidden), device=device, dtype=torch.bfloat16)
    post = torch.zeros((1, 1, streams), device=device)
    mix = torch.zeros((1, 1, streams, streams), device=device)
    phi = torch.zeros(((streams + 2) * streams, streams * hidden), device=device)
    alpha = torch.zeros((3,), device=device)
    beta = torch.zeros(((streams + 2) * streams,), device=device)
    with patch.object(post_module, "omni_training_custom_ops", None):
        with pytest.raises(ImportError, match="omni_training_custom_ops"):
            post_module.mhc_post(x, residual, post, mix, streams)
    with patch.object(pre_module, "omni_training_custom_ops", None):
        with pytest.raises(ImportError, match="omni_training_custom_ops"):
            pre_module.mhc_pre(residual, phi, alpha, beta, streams)


def test_post_forward_backward() -> None:
    """Verify the fused post output and all gradients using a non-symmetric mix."""
    device = _device()
    torch.manual_seed(71)
    batch, sequence, streams, hidden = 1, 1024, 4, 5120
    stream_offsets = torch.tensor([-1.35, -0.85, 0.45, 1.25], device=device).view(1, 1, streams, 1)
    residual = (stream_offsets + 0.1 * torch.rand(batch, sequence, streams, hidden, device=device)).to(torch.bfloat16)
    x = (0.5 + 0.5 * torch.rand(batch, sequence, hidden, device=device)).to(torch.bfloat16)
    mix = torch.tensor([[0.7, 0.3, 0.0, 0.0], [0.0, 0.7, 0.3, 0.0],
                        [0.0, 0.0, 0.7, 0.3], [0.3, 0.0, 0.0, 0.7]], device=device)
    mix = mix.expand(batch, sequence, streams, streams).contiguous()
    post = torch.tensor([0.35, 0.65, 1.05, 1.4], device=device).expand(batch, sequence, streams).contiguous()
    actual_inputs = [value.detach().requires_grad_() for value in (x, residual, post, mix)]
    reference_inputs = [value.detach().clone().requires_grad_() for value in actual_inputs]
    actual_x, actual_residual, actual_post, actual_mix = actual_inputs
    ref_x, ref_residual, ref_post, ref_mix = reference_inputs
    actual = post_module.mhc_post(actual_x, actual_residual.flatten(2), actual_post, actual_mix, streams)
    expected = (torch.einsum("bsij,bsih->bsjh", ref_mix, ref_residual.float())
                + ref_post.unsqueeze(-1) * ref_x.float().unsqueeze(-2)).to(ref_x.dtype).flatten(2)
    output_scales = torch.tensor([-1.75, -0.85, 0.65, 1.45], device=device).view(1, 1, streams, 1)
    upstream = ((0.8 + 0.2 * torch.rand(batch, sequence, streams, hidden, device=device))
                * output_scales).to(actual.dtype).flatten(2)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, upstream)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
    for name, value, reference in zip(("output", "x", "residual", "post", "mix"),
                                      (actual, *actual_gradients), (expected, *reference_gradients)):
        assert torch.isfinite(value).all().item(), f"{name} contains non-finite values"
        torch.testing.assert_close(value.float(), reference.float(), rtol=2e-2, atol=2e-2, msg=name)

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
"""Check the two ShortConv input layouts used by KDA CP."""

import os

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    _run_causal_short_conv,
)


def test_kda_short_conv_local_and_halo_match_original_formulas():
    """Preserve local padding, halo concatenation, output and input gradients."""
    torch.manual_seed(20260916)
    conv = nn.Conv1d(8, 8, kernel_size=4, padding=3, groups=8)
    for use_halo in (False, True):
        tensor = torch.randn(1, 12, 8, requires_grad=True)
        halo = torch.randn(1, 3, 8, requires_grad=True) if use_halo else None
        actual = _run_causal_short_conv(tensor, conv, halo)
        if halo is None:
            reference = conv(tensor.transpose(1, 2))[:, :, : tensor.shape[1]]
        else:
            reference = F.conv1d(  # pylint: disable=not-callable
                torch.cat((halo, tensor), dim=1).transpose(1, 2),
                conv.weight,
                conv.bias,
                padding=0,
                groups=conv.groups,
            )
        reference = F.silu(reference).transpose(1, 2)
        torch.testing.assert_close(actual, reference)
        probe = torch.randn_like(actual)
        inputs = (tensor,) if halo is None else (tensor, halo)
        actual_grads = torch.autograd.grad(actual, inputs, probe, retain_graph=True)
        reference_grads = torch.autograd.grad(reference, inputs, probe)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad)

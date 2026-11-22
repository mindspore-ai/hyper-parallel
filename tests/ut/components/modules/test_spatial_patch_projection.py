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
"""Grouped patch projection source identity and post-meta initialization contracts."""
from __future__ import annotations

import unittest
from copy import deepcopy

import torch
from torch import nn

from hyper_parallel.components.modules import SpatialPatchProjection


class _PostProjection(nn.Module):
    """Independent reference with the released grouped-media projection order."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 32
        self.proj = nn.Sequential(nn.Linear(32, 32, bias=False), nn.GELU(), nn.Linear(32, 12, bias=False))
        self.post_norm = nn.RMSNorm(12, eps=1e-5)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        lengths = [item.shape[0] for item in x]
        packed = torch.cat([item.reshape(item.shape[0], -1) for item in x])
        return self.post_norm(self.proj(packed)).split(lengths)


class TestSpatialPatchProjection(unittest.TestCase):
    """Verify complete grouped-media computation and preservation of source initialization."""

    def test_source_projection_values_and_gradients(self) -> None:
        """Feature: Post-normalized spatial projection.
        Description: Use variable media lengths and both floating dtypes with independent reference equations.
        Expectation: Output, input/parameter gradients, source keys and source tensor identities are preserved.
        """
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = _PostProjection().to(dtype)
                adopted = deepcopy(source)
                target = SpatialPatchProjection(module=adopted)
                identities = dict(adopted.named_parameters())
                for name, parameter in target.named_parameters():
                    self.assertIs(parameter, identities[name])
                inputs = [torch.randn(length, 4, 8, dtype=dtype, requires_grad=True) for length in (3, 2)]
                cloned = [item.detach().clone().requires_grad_() for item in inputs]
                expected, actual = source(inputs), target(x=cloned)
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                sum(item.float().square().sum() for item in expected).backward()
                sum(item.float().square().sum() for item in actual).backward()
                for left, right in zip(inputs, cloned):
                    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
                for left, right in zip(source.parameters(), target.parameters()):
                    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
                before = {name: value.clone() for name, value in target.state_dict().items()}
                target.reset_parameters()
                for name, value in target.state_dict().items():
                    torch.testing.assert_close(value, before[name], atol=0, rtol=0)

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
"""Independent matrix-boundary, update and continuation checks for headwise Muon."""

import copy
import unittest

import torch
from torch import nn

from hyper_parallel.components.optim.headwise import build_reshape_fn
from hyper_parallel.core.optimizer.muon import Muon


class TestHeadwise(unittest.TestCase):
    """Verify the public hook without importing model-owned helpers."""


    def test_ten_updates_and_resume_match_independent_head_optimizers(self) -> None:
        """Feature: Headwise Muon numerical parity.
        Description: Compare packed head views with independently optimized heads and restored state.
        Expectation: Parameters match within tolerance and momentum matches exactly for continued updates.
        """
        generator = torch.Generator().manual_seed(20261122)
        parameter = nn.Parameter(torch.randn(6, 8, generator=generator))
        parameter.model_name = "decoder.query.weight"
        heads = [nn.Parameter(value.clone()) for value in torch.split(parameter, 2)]
        optimizer = Muon([parameter], reshape_fn=build_reshape_fn({"decoder.query.weight": 2}))
        reference = Muon(heads)
        for step in range(12):
            gradient = torch.randn(parameter.shape, generator=generator)
            parameter.grad = gradient.clone()
            for head, part in zip(heads, gradient.split(2)):
                head.grad = part.clone()
            optimizer.step()
            reference.step()
            torch.testing.assert_close(parameter, torch.cat(heads), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"],
                torch.cat([reference.state[head]["momentum_buffer"] for head in heads]),
                rtol=0,
                atol=0,
            )
            if step == 1:
                restored = nn.Parameter(torch.clone(parameter))
                restored.model_name = parameter.model_name
                restored_optimizer = Muon(
                    [restored],
                    reshape_fn=build_reshape_fn({"decoder.query.weight": 2}),
                )
                restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                parameter, optimizer = restored, restored_optimizer

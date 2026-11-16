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
"""Unit tests for the AdamW optimizer."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.adamw import AdamW


class TestAdamW(unittest.TestCase):
    """Verify the step tensors passed to the fused AdamW operator."""

    def test_fused_step_uses_float32(self) -> None:
        """
        Feature: Fused AdamW step dtype.
        Description: Capture the step tensors passed by two optimizer updates.
        Expectation: Each fused call receives a float32 step with the current count.
        """
        parameter = torch.nn.Parameter(torch.ones(2, device="cpu"))
        parameter.grad = torch.ones_like(parameter)
        optimizer = AdamW([parameter])

        with patch("hyper_parallel.core.optimizer.adamw.torch._fused_adamw_") as fused_adamw:
            optimizer.step()
            optimizer.step()

        self.assertEqual(fused_adamw.call_count, 2)
        for step, fused_call in enumerate(fused_adamw.call_args_list, start=1):
            state_steps = fused_call.args[5]
            self.assertEqual(len(state_steps), 1)
            torch.testing.assert_close(state_steps[0], torch.tensor(float(step), dtype=torch.float32, device="cpu"))

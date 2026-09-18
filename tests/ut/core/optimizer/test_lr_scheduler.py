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
"""Unit tests for optimizer learning-rate scheduler composition."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.core.optimizer.lr_scheduler import LRSchedulersContainer


class TestLRSchedulersContainer(unittest.TestCase):
    """Validate named scheduler state restoration."""

    def test_load_state_dict_restores_optimizer_learning_rate(self) -> None:
        """Apply the saved last LR before the first resumed optimizer step."""
        source_parameter = torch.nn.Parameter(torch.ones(1))
        source_optimizer = torch.optim.SGD([source_parameter], lr=1.0)
        source = LRSchedulersContainer(
            SimpleNamespace(optimizers_dict={"main": source_optimizer}),
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            ),
        )
        source_optimizer.step()
        source.step()

        target_parameter = torch.nn.Parameter(torch.ones(1))
        target_optimizer = torch.optim.SGD([target_parameter], lr=0.0)
        target = LRSchedulersContainer(
            SimpleNamespace(optimizers_dict={"main": target_optimizer}),
            scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            ),
        )

        target.load_state_dict(source.state_dict())

        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.5)
        self.assertEqual(target.get_last_lr(), [0.5])


if __name__ == "__main__":
    unittest.main()

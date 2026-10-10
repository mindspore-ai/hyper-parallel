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
"""Precision-preserving Host optimizer skeleton restore checks."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.models.deepseek_v41.adapter.optim.host_sparse import HostSparseOptimizerCoordinator
from tests.common.mark_utils import arg_mark


class TestHostOptimizerRestore(unittest.TestCase):
    """Keep lazy optimizer initialization out of persisted Host state."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_lazy_moments_not_in_checkpoint_are_pruned(self):
        """Drop unpersisted moments after dense DCP load.

        Feature: Exact dense optimizer state on Host checkpoint resume.
        Description: Prime an extra inactive parameter after the saved step.
        Expectation: Only the inactive parameter loses its artificial moments.
        """
        model = torch.nn.Module()
        model.active = torch.nn.Parameter(torch.ones(2))
        model.inactive = torch.nn.Parameter(torch.ones(2))
        leaf = torch.optim.AdamW(model.parameters(), lr=0.1)
        leaf.state[model.active] = {"exp_avg": torch.ones(2)}
        leaf.state[model.inactive] = {"exp_avg": torch.zeros(2)}
        raw = SimpleNamespace(model=model, optimizer_param_by_model_param={})
        dense = SimpleNamespace(optimizer=raw, optimizers_dict={"dense": leaf})
        coordinator = HostSparseOptimizerCoordinator(dense, None, {})
        coordinator.drop_unpersisted_dense_state(
            frozenset({"state.active.exp_avg", "param_groups.lr"}),
        )
        self.assertIn(model.active, leaf.state)
        self.assertNotIn(model.inactive, leaf.state)
        torch.testing.assert_close(leaf.state[model.active]["exp_avg"], torch.ones(2))


if __name__ == "__main__":
    unittest.main()

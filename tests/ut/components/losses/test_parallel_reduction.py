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
"""Model-parallel objective reduction against an independent global mean."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.jt_expert_parallel import _model_parallel_mean
from tests.common.mark_utils import arg_mark


class TestParallelReduction(unittest.TestCase):
    """Distinguish one global objective from independently consumed rank losses."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_partition_gradients_match_global_objective(self):
        """Feature: Replicated loss ownership.

        Description: Average unequal local losses across two or eight emulated ranks.
        Expectation: Every rank matches its slice of a dense mean's gradient, without backward communication.
        """
        for size in (2, 8):
            full = torch.arange(1., size + 1, requires_grad=True)
            expected = full.square().mean()
            (expected * 3).backward()
            for rank in range(size):
                local = full[rank].detach().clone().requires_grad_()
                group = object()
                with patch("torch.distributed.get_world_size", return_value=size), \
                        patch("torch.distributed.all_reduce") as reduce:
                    reduce.side_effect = lambda tensor, **_: tensor.copy_(full.detach().square().sum())
                    result = _model_parallel_mean(local.square(), group)
                    (result * 3).backward()
                    self.assertEqual(reduce.call_count, 1)
                    self.assertIs(reduce.call_args.kwargs["group"], group)
                torch.testing.assert_close(result, expected)
                torch.testing.assert_close(local.grad, full.grad[rank])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_no_group_is_local_even_with_default_group(self):
        """Feature: Explicit collective ownership.

        Description: Omit the model-parallel group.
        Expectation: No default group is queried and the tensor is returned unchanged.
        """
        value = torch.tensor(2., requires_grad=True)
        with patch("torch.distributed.all_reduce", side_effect=AssertionError), \
                patch("torch.distributed.get_world_size", side_effect=AssertionError):
            result = _model_parallel_mean(value)
            result.backward()
        self.assertIs(result, value)
        self.assertEqual(value.grad.item(), 1.)

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
"""Per-token vocabulary loss must include targets owned by other ranks."""

import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from hyper_parallel.components.losses import _vocab_parallel_cross_entropy as ce
from tests.common.mark_utils import arg_mark


class TestVocabParallelTokenLoss(unittest.TestCase):
    """Use dense full-vocabulary CE as an independent derivative oracle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_weighted_tokens_on_both_vocabulary_shards(self):
        """Feature: Unreduced global token losses.

        Description: Put targets on both vocabulary shards and include fractional/zero masks.
        Expectation: All ranks receive full per-token losses and correct local gradients exactly once.
        """
        torch.manual_seed(18)
        dense = torch.randn(5, 8, requires_grad=True)
        targets = torch.tensor([0, 5, -100, 7, 2])
        weights = torch.tensor([0.25, 1.5, 0., 1., 0.])
        expected = F.cross_entropy(dense, targets, reduction="none")
        (expected * weights).sum().backward()
        maximum = dense.detach().amax(-1, keepdim=True)
        denominator = (dense.detach() - maximum).exp().sum(-1, keepdim=True)
        for rank in range(2):
            mesh = Mock(ndim=1)
            mesh.get_local_rank.return_value = rank
            mesh.size.return_value = 2
            local = dense.detach()[:, rank * 4:(rank + 1) * 4].clone().requires_grad_()
            with patch.object(ce.platform, "differentiable_all_reduce", side_effect=[
                    maximum, denominator, expected.detach()]) as reduce:
                actual = ce.vocab_parallel_cross_entropy_local(
                    local, targets, vocab_size=8, mesh=mesh, reduction="none")
                (actual * weights).sum().backward()
            self.assertEqual(reduce.call_count, 3)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(local.grad, dense.grad[:, rank * 4:(rank + 1) * 4])

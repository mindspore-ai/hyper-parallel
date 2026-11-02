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
"""CPU checks for reusable MoonEP token placement metadata."""

import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep.planning import compile_plan
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestPlanning(unittest.TestCase):
    """Verify that the planned native unpermute mapping undoes send ordering."""

    def test_unpermute_indices_restore_original_topk_slots(self):
        """Feature: Reusable native unpermute indices.

        Description: Compile a route whose returned rows differ from original Top-K order.
        Expectation: Every original Top-K slot addresses its exact returned row with int32 indices.
        """
        ids = torch.tensor([2, 0, 1, 0, 3, 2], dtype=torch.int64)
        counts = torch.bincount(ids, minlength=4).reshape(1, 4)
        plan = compile_plan(counts, counts.T.contiguous(), ids, rank=0)

        self.assertEqual(plan.unpermute_indices.dtype, torch.int32)
        original = torch.arange(ids.numel())
        restored = plan.dispatch_order.index_select(0, plan.unpermute_indices.long())
        torch.testing.assert_close(restored, original)

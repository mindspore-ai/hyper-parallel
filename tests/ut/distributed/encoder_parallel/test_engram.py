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
"""Engram EP1 alignment with a TP-local decoder hidden interval."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.components.modules.engram import local_token_mask
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_engram_compute_fn,
)


class TestEngramSPAlignment(unittest.TestCase):
    """Keep EP1 hash lookup and image masking on the active TP/SP interval."""

    def test_ep1_factory_uses_tp_rank_and_local_mask(self):
        """Feature: Engram EP1 with decoder sequence parallelism.

        Description: Run EP1 fusion at TP rank two with a CP-local image mask.
        Expectation: Hash lookup receives TP coordinates and output uses local mask.
        """
        calls = []

        def _aligned_ids(hidden, input_ids, segment_starts, token_mask, **kwargs):
            del input_ids, segment_starts, token_mask
            calls.append(kwargs)
            return torch.zeros(hidden.shape[:2], dtype=torch.long)

        module = SimpleNamespace(
            hash_mapping=object(),
            embed=lambda ids: torch.ones((*ids.shape, 1, 1)),
            parallel_forward=lambda *args, **kwargs: self.fail("EP1 must use aligned local lookup"),
            _aligned_hash_ids=_aligned_ids,
            _fuse=lambda hidden, embeddings: hidden + embeddings,
        )
        tp_mesh = SimpleNamespace(size=lambda: 4, get_local_rank=lambda: 2)
        compute = deepseek_v41_engram_compute_fn(
            module=module, mesh=None, tp_mesh=tp_mesh, cp_mesh=None, ep_mesh=None,
        )
        hidden = torch.zeros(1, 2, 1, 1, requires_grad=True)
        mask = torch.tensor([[True, True, True, True, False, True, True, True]])
        output = compute(module, hidden, torch.zeros(1, 8, dtype=torch.long), token_mask=mask)
        self.assertEqual(calls, [{"cp_group": None, "cp_rank": 0, "cp_size": 1,
                                  "tp_rank": 2, "tp_size": 4}])
        self.assertTrue(torch.equal(output.flatten(), torch.tensor([0.0, 1.0])))
        output.sum().backward()
        self.assertTrue(torch.equal(hidden.grad, torch.ones_like(hidden)))

    def test_incompatible_mask_fails(self):
        """Feature: Engram token-mask geometry check.

        Description: Supply seven mask rows for a two-row TP4 hidden shard.
        Expectation: The incompatible mask fails before fusion.
        """
        with self.assertRaisesRegex(ValueError, "incompatible with TP/SP hidden"):
            local_token_mask(torch.ones(1, 7), torch.ones(1, 2, 1, 1), tp_rank=2, tp_size=4)

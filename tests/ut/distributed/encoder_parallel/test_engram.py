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
from unittest.mock import Mock

import torch

from hyper_parallel.components.modules.engram import local_token_mask
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_engram_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from tests.common.mark_utils import arg_mark


class TestEngramSPAlignment(unittest.TestCase):
    """Keep EP1 hash lookup and image masking on the active TP/SP interval."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_incompatible_mask_fails(self):
        """Feature: Engram token-mask geometry check.

        Description: Supply seven mask rows for a two-row TP4 hidden shard.
        Expectation: The incompatible mask fails before fusion.
        """
        with self.assertRaisesRegex(ValueError, "incompatible with TP/SP hidden"):
            local_token_mask(torch.ones(1, 7), torch.ones(1, 2, 1, 1), tp_rank=2, tp_size=4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_host_table_fusion_matches_full_sequence_mask(self) -> None:
        """Feature: Host Engram fusion with sequence parallelism.

        Description: Run CP1/CP2 and TP1/TP2 with mixed, all-text, or absent masks.
        Expectation: Outputs and gradients match the corresponding full-sequence slice.
        """
        full_hidden = torch.arange(2 * 4096 * 2, dtype=torch.float32).view(2, 4096, 2, 1)
        full_mask = torch.stack((torch.arange(4096) % 3 == 0, torch.arange(4096) % 5 == 0))
        for cp_size, cp_rank, tp_size, tp_rank in (
                (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 2, 1), (2, 0, 1, 0), (2, 1, 1, 0),
                (2, 0, 2, 0), (2, 0, 2, 1), (2, 1, 2, 0), (2, 1, 2, 1)):
            for token_mask in (full_mask, torch.ones_like(full_mask), None):
                with self.subTest(cp=cp_size, cp_rank=cp_rank, tp=tp_size, tp_rank=tp_rank,
                                  masked=token_mask is not None):
                    table = HostEngramTable(
                        source_weight=torch.nn.Parameter(torch.empty((4, 1), device="meta")),
                        logical_rows=4,
                    )
                    table.bind_planned_shard(ep_rank=0, ep_size=1)
                    table.weight = torch.nn.Parameter(torch.full((4, 1), 3.0))
                    hidden = full_hidden.chunk(tp_size, dim=1)[tp_rank].clone().requires_grad_(True)
                    aligned_ids = Mock(return_value=torch.zeros(hidden.shape[:2], dtype=torch.long))
                    module = SimpleNamespace(
                        hash_mapping=object(), embed=table, parallel_forward=Mock(),
                        _aligned_hash_ids=aligned_ids,
                        _fuse=lambda states, embeddings: 2 * states + embeddings.unsqueeze(-1),
                    )
                    tp_mesh = SimpleNamespace(size=lambda: tp_size, get_local_rank=lambda: tp_rank)
                    cp_group = object()
                    cp_mesh = SimpleNamespace(size=lambda: cp_size, get_local_rank=lambda: cp_rank,
                                              get_group=lambda: cp_group)
                    compute = deepseek_v41_engram_compute_fn(
                        module=module, mesh=None, tp_mesh=tp_mesh, cp_mesh=cp_mesh, ep_mesh=None,
                    )

                    output = compute(module, hidden, torch.zeros(2, 4096, dtype=torch.long), token_mask=token_mask)
                    reference_hidden = full_hidden.clone().requires_grad_(True)
                    reference_row = torch.tensor(3.0, requires_grad=True)
                    reference = 2 * reference_hidden + reference_row
                    if token_mask is not None:
                        reference = torch.where(token_mask[..., None, None], reference, reference_hidden)
                    reference = reference.chunk(tp_size, dim=1)[tp_rank]
                    torch.testing.assert_close(output, reference)
                    self.assertIs(aligned_ids.call_args.args[3], token_mask)
                    expected_coordinates = {"cp_group": cp_group, "cp_rank": cp_rank, "cp_size": cp_size,
                                            "tp_rank": tp_rank, "tp_size": tp_size}
                    self.assertEqual(aligned_ids.call_args.kwargs, expected_coordinates)
                    module.parallel_forward.assert_not_called()

                    output.sum().backward()
                    reference.sum().backward()
                    torch.testing.assert_close(hidden.grad, reference_hidden.grad.chunk(tp_size, dim=1)[tp_rank])
                    expected_table_grad = torch.zeros_like(table.weight)
                    expected_table_grad[0] = reference_row.grad
                    torch.testing.assert_close(table.coalesce_pending().to_dense(), expected_table_grad)

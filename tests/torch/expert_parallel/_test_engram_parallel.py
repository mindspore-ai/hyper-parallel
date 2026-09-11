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
"""Distributed CPU precision worker for Engram EP lookup and CP hashing."""

from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.modules.engram import EngramModule, NgramHashMapping


class _EngramSource(nn.Module):
    """Small source-layout fixture accepted by the Engram replacement."""

    def __init__(self) -> None:
        super().__init__()
        assets = {
            "layer_ids": [1],
            "max_ngram_size": 3,
            "num_heads": 2,
            "head_dim": 2,
            "primes": [[[17, 19], [23, 29]]],
            "multipliers": [[101, 103, 107]],
            "token_map": list(range(64)),
            "pad_token_id": 0,
        }
        self.layer_id = 1
        self.hidden_size = 4
        self.hc_mult = 2
        self.eps = 1.0e-6
        self.clamp_value = 1.0e-6
        self.hash_mapping = NgramHashMapping(assets, self.layer_id)
        self.logical_num_embeddings = 88
        self.padded_num_embeddings = 96
        self.embed = nn.Embedding(self.padded_num_embeddings, 2)
        self.wkv = nn.Linear(8, 12, bias=False)
        self.q_weight = nn.Parameter(torch.ones(2, 4))
        self.k_weight = nn.Parameter(torch.ones(2, 4))

    def forward(self, hidden_states, input_ids, segment_starts=None):
        """The source forward is unused by this worker."""
        del hidden_states, input_ids, segment_starts
        raise RuntimeError("fixture forward is intentionally unavailable")


def test_engram_ep_lookup_cp_hash_gloo():
    """Four ranks match a replicated table for output, input grad, and table grad."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 4
        torch.manual_seed(17)
        reference = EngramModule(module=_EngramSource())
        parallel = deepcopy(reference)
        full_weight = reference.embed.weight.detach().clone()
        rows_per_rank = parallel.padded_num_embeddings // world_size
        local_weight = full_weight[rank * rows_per_rank:(rank + 1) * rows_per_rank]
        parallel.embed.weight = nn.Parameter(local_weight.clone())

        global_ids = torch.arange(3, 19, dtype=torch.long).view(1, -1)
        local_ids = global_ids[:, rank * 4:(rank + 1) * 4].contiguous()
        torch.manual_seed(101 + rank)
        hidden = torch.randn(1, 4, 2, 4, requires_grad=True)
        hidden_reference = hidden.detach().clone().requires_grad_(True)

        output = parallel.parallel_forward(
            hidden,
            local_ids,
            ep_group=dist.group.WORLD,
            ep_rank=rank,
            ep_size=world_size,
            cp_group=dist.group.WORLD,
            cp_rank=rank,
            cp_size=world_size,
        )
        full_hashes = reference.hash_mapping(global_ids)
        local_hashes = full_hashes[:, rank * 4:(rank + 1) * 4]
        expected = reference._fuse(  # pylint: disable=protected-access
            hidden_reference,
            reference.embed(local_hashes),
        )
        torch.testing.assert_close(output, expected, rtol=1.0e-5, atol=1.0e-6)

        scale = float(rank + 1)
        (output * scale).sum().backward()
        (expected * scale).sum().backward()
        dist.all_reduce(reference.embed.weight.grad)
        expected_table_grad = reference.embed.weight.grad[
            rank * rows_per_rank:(rank + 1) * rows_per_rank
        ]
        torch.testing.assert_close(hidden.grad, hidden_reference.grad, rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(
            parallel.embed.weight.grad,
            expected_table_grad,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    finally:
        dist.destroy_process_group()

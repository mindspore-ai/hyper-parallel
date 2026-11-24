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
"""Owned Engram table layout and borrowed runtime resource contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.engram import Engram
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding


def small_config(rows: int = 16) -> SimpleNamespace:
    """Provide two independent hash columns and a physically padded table."""
    return SimpleNamespace(
        assets={
            "layer_ids": [0],
            "max_ngram_size": 2,
            "num_heads": 2,
            "token_map": list(range(6)),
            "pad_token_id": 0,
            "primes": [[[5, 7]]],
            "multipliers": [[1, 3]],
        },
        layer_id=0,
        hidden_size=4,
        hc_mult=2,
        embedding_dim=2,
        eps=1e-6,
        clamp_value=1e-6,
        padded_num_embeddings=rows,
    )


class TestEngramFamilyBinding(unittest.TestCase):
    """Exercise real family layouts and local execution without initializing distributed state."""


    def test_ep_binding_borrows_group_and_routes_original_inputs(self) -> None:
        """The bound family supplies the existing EP group to its class-owned sparse stage."""
        module = Engram(config=small_config())
        group = object()
        axis = SimpleNamespace(get_local_rank=lambda: 1, get_group=lambda: group)
        mesh = SimpleNamespace(
            ep_size=2, cp_size=1, tp_size=1, tp_rank=0, sequence_parallel=False, fsdp_moe_mesh={"ep": axis}
        )
        module.bind_parallel(ParallelBinding(MeshBinding(mesh)))
        hidden = torch.randn(1, 3, 2, 4)
        ids = torch.tensor([[1, 2, 3]])
        with patch.object(module, "parallel_forward", return_value=hidden) as stage:
            actual = module(hidden, ids)
        self.assertIs(actual, hidden)
        self.assertIs(stage.call_args.kwargs["ep_group"], group)
        self.assertEqual(stage.call_args.kwargs["ep_rank"], 1)
        self.assertEqual(stage.call_args.kwargs["ep_size"], 2)
        self.assertIs(stage.call_args.args[0], hidden)

    def test_local_binding_does_not_enter_world_collectives(self) -> None:
        """A DP-only job calls ordinary embedding lookup while preserving local outputs and gradients."""
        module = Engram(config=small_config())
        hidden = torch.randn(1, 3, 2, 4, requires_grad=True)
        ids = torch.tensor([[1, 2, 3]])
        expected = module(hidden, ids)
        mesh = SimpleNamespace(ep_size=1, cp_size=1, tp_size=1, sequence_parallel=False)
        module.bind_parallel(ParallelBinding(MeshBinding(mesh)))
        with patch("torch.distributed.all_gather", side_effect=AssertionError("Unexpected world collective")):
            actual = module(hidden, ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))

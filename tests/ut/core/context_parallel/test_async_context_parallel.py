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
"""Unit tests for async context-parallel hook argument handling."""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.context_parallel import async_context_parallel as async_cp_module  # noqa: E402


class _FakeTwoDMesh:
    """Small 2-D mesh stub exposing the subset used by the hybrid pre-hook."""

    mesh_dim_names = ("co", "ds")

    def __getitem__(self, name):
        return f"{name}_submesh"


class TestAsyncContextParallelKwargs(unittest.TestCase):
    """CPU-only tests for Q/K/V passed as keyword arguments."""

    def setUp(self):
        self.style = async_cp_module.AsyncContextParallel(
            qkv_indices=(10, 11, 12),
            qkv_kwarg_names=("query", "key", "value"),
        )
        self.query = torch.tensor([1.0])
        self.key = torch.tensor([2.0])
        self.value = torch.tensor([3.0])

    @staticmethod
    def _kwargs(query, key, value):
        return {"query": query, "key": key, "value": value, "mask": "keep"}

    def test_ulysses_pre_hook_rewrites_qkv_kwargs(self):
        """Ulysses async pre-hook should support Q/K/V in kwargs."""
        offsets = {"q": 10.0, "k": 20.0, "v": 30.0}

        def fake_wait(tensor, group, world_size, fwd_slots, key, bwd_slot):
            del group, world_size, fwd_slots, bwd_slot
            return tensor + offsets[key]

        with patch.object(self.style, "_wait_a2a", side_effect=fake_wait):
            args, kwargs = self.style._attn_pre_hook_ulysses(  # pylint: disable=protected-access
                None,
                (),
                self._kwargs(self.query, self.key, self.value),
                group="ds_group",
                world_size=2,
                fwd_slots={"q": "q_slot", "k": "k_slot", "v": "v_slot"},
                bwd_slots={"q": [], "k": [], "v": []},
            )

        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["query"], self.query + 10.0))
        self.assertTrue(torch.equal(kwargs["key"], self.key + 20.0))
        self.assertTrue(torch.equal(kwargs["value"], self.value + 30.0))
        self.assertEqual(kwargs["mask"], "keep")

    def test_colossal_pre_hook_rewrites_qkv_kwargs(self):
        """Colossal async pre-hook should support Q/K/V in kwargs."""
        from_local = MagicMock(side_effect=lambda tensor, mesh, placements: (tensor, mesh, placements))

        with patch.object(async_cp_module.DTensor, "from_local", side_effect=from_local), \
                patch.object(self.style, "_wait_allgather", side_effect=lambda tensor, *unused: tensor + 1.0):
            args, kwargs = self.style._attn_pre_hook_colossal(  # pylint: disable=protected-access
                None,
                (),
                self._kwargs(self.query, self.key, self.value),
                co_submesh="co_mesh",
                group="co_group",
                world_size=2,
                fwd_slots={"k": ("k_work", "k_out"), "v": ("v_work", "v_out")},
                bwd_slots={"k": [], "v": []},
            )

        self.assertEqual(args, ())
        self.assertEqual(kwargs["query"][1], "co_mesh")
        self.assertTrue(torch.equal(kwargs["key"][0], self.key + 1.0))
        self.assertTrue(torch.equal(kwargs["value"][0], self.value + 1.0))
        self.assertEqual(from_local.call_count, 3)

    def test_hybrid_pre_hook_rewrites_qkv_kwargs(self):
        """Hybrid async pre-hook should keep async A2A and sync K/V gather for kwargs."""
        offsets = {"q": 10.0, "k": 20.0, "v": 30.0}
        from_local = MagicMock(side_effect=lambda tensor, mesh, placements: (tensor, mesh, placements))

        def fake_wait(tensor, group, world_size, fwd_slots, key, bwd_slot):
            del group, world_size, fwd_slots, bwd_slot
            return tensor + offsets[key]

        with patch.object(async_cp_module.DTensor, "from_local", side_effect=from_local), \
                patch.object(self.style, "_wait_a2a", side_effect=fake_wait), \
                patch.object(async_cp_module, "_gather_seq", side_effect=lambda tensor, *unused: tensor + 1.0):
            args, kwargs = self.style._attn_pre_hook_hybrid(  # pylint: disable=protected-access
                None,
                (),
                self._kwargs(self.query, self.key, self.value),
                group="ds_group",
                world_size=2,
                two_d_mesh=_FakeTwoDMesh(),
                fwd_slots={"q": "q_slot", "k": "k_slot", "v": "v_slot"},
                bwd_slots={"q": [], "k": [], "v": []},
            )

        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["query"][0], self.query + 10.0))
        self.assertTrue(torch.equal(kwargs["key"][0], self.key + 21.0))
        self.assertTrue(torch.equal(kwargs["value"][0], self.value + 31.0))
        self.assertEqual(from_local.call_count, 3)


if __name__ == "__main__":
    unittest.main()

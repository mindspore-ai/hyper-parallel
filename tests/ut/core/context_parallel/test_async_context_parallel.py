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

import numpy as np
import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.context_parallel import async_context_parallel as async_cp_module  # noqa: E402
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh  # noqa: E402
from hyper_parallel.core.dtensor.dtensor import DTensor  # noqa: E402
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard  # noqa: E402
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType  # noqa: E402


class _FakeTwoDMesh:
    """Small 2-D mesh stub exposing the subset used by the hybrid pre-hook."""

    mesh_dim_names = ("co", "ds")

    def __getitem__(self, name):
        return f"{name}_submesh"


class TestAsyncContextParallelKwargs(unittest.TestCase):
    """CPU-only tests for Q/K/V passed as keyword arguments."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.style = async_cp_module.AsyncContextParallel(
            qkv_indices=(10, 11, 12),
            qkv_kwarg_names=("query", "key", "value"),
        )
        self.query = torch.tensor([1.0])
        self.key = torch.tensor([2.0])
        self.value = torch.tensor([3.0])

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

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

        def fake_wait(tensor, group, world_size, fwd_slots, key, bwd_slot, fallback_mesh, fallback_placements):
            del group, world_size, fwd_slots, bwd_slot, fallback_mesh, fallback_placements
            return tensor + offsets[key]

        with patch.object(async_cp_module.DTensor, "from_local", side_effect=from_local), \
                patch.object(self.style, "_wait_a2a_dtensor", side_effect=fake_wait), \
                patch.object(async_cp_module, "_gather_seq", side_effect=lambda tensor, *unused: tensor + 1.0):
            args, kwargs = self.style._attn_pre_hook_hybrid(  # pylint: disable=protected-access
                None,
                (),
                self._kwargs(self.query, self.key, self.value),
                group="ds_group",
                world_size=2,
                hybrid_cp_mesh=_FakeTwoDMesh(),
                fwd_slots={"q": "q_slot", "k": "k_slot", "v": "v_slot"},
                bwd_slots={"q": [], "k": [], "v": []},
            )

        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["query"], self.query + 10.0))
        self.assertTrue(torch.equal(kwargs["key"][0], self.key + 21.0))
        self.assertTrue(torch.equal(kwargs["value"][0], self.value + 31.0))
        self.assertEqual(from_local.call_count, 2)


def _setup_mock_mesh_platform(mock_platform, world_size, rank=0):
    """Configure a mocked device-mesh platform for CPU-only DTensor tests."""
    mock_platform.platform_type = PlatformType.PYTORCH
    mock_platform.get_rank.return_value = rank
    mock_platform.get_world_size.return_value = world_size
    mock_platform.tensor_to_numpy.side_effect = (
        lambda tensor: tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.array(tensor)
    )


class TestAsyncContextParallelLayoutSlots(unittest.TestCase):
    """Regression tests for preserving non-CP DTensor layout in async CP slots."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        self.style = async_cp_module.AsyncContextParallel(seq_dim=1, head_dim=2)

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @staticmethod
    def _tp_sharded_tensor(tp_mesh):
        local = torch.randn(1, 4, 4, 2)
        return DTensor.from_local(local, tp_mesh, (Shard(2),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_ulysses_slot_restores_cp_tp_composed_head_layout(self, mock_mesh_platform):
        """Async Ulysses wait should restore CP+TP layout instead of CP-only layout."""
        _setup_mock_mesh_platform(mock_mesh_platform, world_size=4)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("cp", "tp"),
            rank_list=(0, 1, 2, 3),
            init_backend=False,
        )
        cp_mesh = root["cp"]
        tp_dtensor = self._tp_sharded_tensor(root["tp"])
        slot = async_cp_module._make_async_cp_slot(  # pylint: disable=protected-access
            work="work",
            out_perm="out_perm",
            tensor=tp_dtensor,
            cp_mesh=cp_mesh,
            cp_placements=(Shard(2),),
            seq_dim=1,
        )
        local_after_a2a = torch.randn(1, 8, 2, 2)
        result = async_cp_module._wrap_async_cp_result(  # pylint: disable=protected-access
            local_after_a2a,
            slot,
            cp_mesh,
            (Shard(2),),
        )

        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(result.placements, (StridedShard(2, 2), Shard(2)))
        self.assertTrue(torch.equal(result.to_local(), local_after_a2a))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_colossal_slot_restores_tp_layout_after_async_allgather(self, mock_mesh_platform):
        """Async Colossal K/V all-gather should keep TP layout metadata."""
        _setup_mock_mesh_platform(mock_mesh_platform, world_size=4)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("cp", "tp"),
            rank_list=(0, 1, 2, 3),
            init_backend=False,
        )
        cp_mesh = root["cp"]
        tp_dtensor = self._tp_sharded_tensor(root["tp"])
        slot = async_cp_module._make_async_cp_slot(  # pylint: disable=protected-access
            work="work",
            out_perm="out_perm",
            tensor=tp_dtensor,
            cp_mesh=cp_mesh,
            cp_placements=(Replicate(),),
            seq_dim=1,
        )
        gathered = torch.randn(1, 8, 4, 2)
        result = async_cp_module._wrap_async_cp_result(  # pylint: disable=protected-access
            gathered,
            slot,
            cp_mesh,
            (Replicate(),),
        )

        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(result.placements, (Replicate(), Shard(2)))
        self.assertTrue(torch.equal(result.to_local(), gathered))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_hybrid_slot_restores_co_ds_tp_composed_layout(self, mock_mesh_platform):
        """Async Hybrid A2A should restore the full CO+DS+TP execution layout."""
        _setup_mock_mesh_platform(mock_mesh_platform, world_size=8)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("co", "ds", "tp"),
            rank_list=tuple(range(8)),
            init_backend=False,
        )
        hybrid_cp_mesh = root[("co", "ds")]
        tp_dtensor = self._tp_sharded_tensor(root["tp"])
        slot = async_cp_module._make_async_cp_slot(  # pylint: disable=protected-access
            work="work",
            out_perm="out_perm",
            tensor=tp_dtensor,
            cp_mesh=hybrid_cp_mesh,
            cp_placements=(Shard(1), Shard(2)),
            seq_dim=1,
        )
        local_after_a2a = torch.randn(1, 8, 2, 2)
        result = async_cp_module._wrap_async_cp_result(  # pylint: disable=protected-access
            local_after_a2a,
            slot,
            hybrid_cp_mesh,
            (Shard(1), Shard(2)),
        )

        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("co", "ds", "tp"))
        self.assertEqual(result.placements, (Shard(1), StridedShard(2, 2), Shard(2)))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_wait_a2a_dtensor_wraps_wait_result_with_recorded_layout(self, mock_mesh_platform):
        """_wait_a2a_dtensor should use the slot layout captured before communication."""
        _setup_mock_mesh_platform(mock_mesh_platform, world_size=4)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("cp", "tp"),
            rank_list=(0, 1, 2, 3),
            init_backend=False,
        )
        cp_mesh = root["cp"]
        tp_dtensor = self._tp_sharded_tensor(root["tp"])
        slot = async_cp_module._make_async_cp_slot(  # pylint: disable=protected-access
            work="work",
            out_perm="out_perm",
            tensor=tp_dtensor,
            cp_mesh=cp_mesh,
            cp_placements=(Shard(2),),
            seq_dim=1,
        )
        waited_local = torch.randn(1, 8, 2, 2)

        with patch.object(self.style, "_wait_a2a", return_value=waited_local) as wait_a2a:
            result = self.style._wait_a2a_dtensor(  # pylint: disable=protected-access
                tp_dtensor.to_local(),
                group="ds_group",
                world_size=2,
                fwd_slots={"q": slot},
                key="q",
                bwd_slot=[],
                fallback_mesh=cp_mesh,
                fallback_placements=(Shard(2),),
            )

        wait_a2a.assert_called_once()
        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(result.placements, (StridedShard(2, 2), Shard(2)))
        self.assertTrue(torch.equal(result.to_local(), waited_local))


if __name__ == "__main__":
    unittest.main()

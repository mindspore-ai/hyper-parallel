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
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.context_parallel import async_context_parallel as async_cp_module  # noqa: E402
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh  # noqa: E402
from hyper_parallel.core.dtensor.dtensor import DTensor  # noqa: E402
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard  # noqa: E402
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType  # noqa: E402


class _FakeTwoDMesh:
    """Small 2-D mesh stub exposing the subset used by the hybrid pre-hook."""

    mesh_dim_names = ("co", "ds")
    ndim = 2
    rank_list = (0, 1, 2, 3)
    device_type = "cpu"
    mesh = None

    def __init__(self):
        self.mesh = _FakeMeshTensor(4)
        self._submeshes = {
            "co": _FakeMesh(2, mesh_dim_names=("co",), group="co_group"),
            "ds": _FakeMesh(2, mesh_dim_names=("ds",), group="ds_group"),
        }

    def __getitem__(self, name):
        return self._submeshes[name]


class _FakeMeshTensor:
    """Minimal mesh tensor stub exposing ``numel``."""

    def __init__(self, size):
        self._size = size

    def numel(self):
        return self._size


class _FakeMesh:
    """Small mesh stub for AsyncContextParallel.apply branch tests."""

    def __init__(self, size, *, ndim=1, mesh_dim_names=("cp",), group="cp_group"):
        self.mesh = _FakeMeshTensor(size)
        self.ndim = ndim
        self.mesh_dim_names = mesh_dim_names
        self.rank_list = tuple(range(size))
        self.device_type = "cpu"
        self._group = group

    def get_group(self):
        return self._group


class _FakeWork:
    """Small async work stub recording wait calls."""

    def __init__(self):
        self.wait_called = False

    def wait(self):
        self.wait_called = True


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

    def test_launch_async_a2a_seq_to_head_shapes_and_error(self):
        """Async A2A launch should reshape by heads and reject non-divisible heads."""
        tensor = torch.randn(1, 4, 4, 2, requires_grad=True)
        out_perm = torch.empty(2, 1, 4, 2, 2)
        work = object()

        with patch.object(
                async_cp_module.platform,
                "all_to_all_single",
                return_value=(out_perm, work),
                create=True,
        ) as mock_a2a:
            result = async_cp_module._launch_async_a2a_seq_to_head(  # pylint: disable=protected-access
                tensor,
                group="group",
                world_size=2,
                head_dim=2,
            )

        self.assertEqual(result, (work, out_perm))
        call_tensor, call_shape, call_group = mock_a2a.call_args.args[:3]
        self.assertFalse(call_tensor.requires_grad)
        self.assertEqual(call_shape, [2, 1, 4, 2, 2])
        self.assertEqual(call_group, "group")
        self.assertTrue(mock_a2a.call_args.kwargs["async_op"])

        with self.assertRaisesRegex(ValueError, "num_heads \\(3\\).*world_size \\(2\\)"):
            async_cp_module._launch_async_a2a_seq_to_head(  # pylint: disable=protected-access
                torch.randn(1, 4, 3, 2),
                group="group",
                world_size=2,
                head_dim=2,
            )

    def test_move_dim_helpers_and_allgather_launch(self):
        """All-gather helpers should move arbitrary dims through the front buffer."""
        tensor = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

        front = async_cp_module._move_dim_to_front(tensor, -1)  # pylint: disable=protected-access
        restored = async_cp_module._move_dim_from_front(front, -1)  # pylint: disable=protected-access
        self.assertTrue(torch.equal(restored, tensor))
        self.assertTrue(torch.equal(
            async_cp_module._move_dim_to_front(tensor, 0),  # pylint: disable=protected-access
            tensor,
        ))
        self.assertTrue(torch.equal(
            async_cp_module._move_dim_from_front(tensor, 0),  # pylint: disable=protected-access
            tensor,
        ))

        out_perm = torch.empty(6, 2, 4)
        work = object()
        with patch.object(
                async_cp_module.platform,
                "all_gather_single",
                return_value=(out_perm, work),
                create=True,
        ) as mock_allgather:
            result = async_cp_module._launch_async_allgather_seq(  # pylint: disable=protected-access
                tensor,
                group="group",
                world_size=2,
                gather_dim=1,
            )

        self.assertEqual(result, (work, out_perm))
        self.assertEqual(mock_allgather.call_args.args[1], [6, 2, 4])
        self.assertTrue(mock_allgather.call_args.kwargs["async_op"])

    def test_detach_if_available_keeps_objects_without_detach(self):
        """Detach helper should be a no-op for backend objects without detach."""
        value = object()
        self.assertIs(async_cp_module._detach_if_available(value), value)  # pylint: disable=protected-access

    def test_apply_registers_async_colossal_and_ulysses_hooks(self):
        """Async apply should register projection and attention hooks for async modes."""
        for style, mesh in [
                (async_cp_module.AsyncContextParallel(ulysses_degree=1), _FakeMesh(2)),
                (async_cp_module.AsyncContextParallel(), _FakeMesh(2)),
        ]:
            module = nn.Identity()
            q_proj = nn.Identity()
            k_proj = nn.Identity()
            v_proj = nn.Identity()

            with patch.object(
                    async_cp_module.platform,
                    "register_forward_pre_hook",
                    create=True,
            ) as mock_pre, patch.object(
                    async_cp_module.platform,
                    "register_full_backward_pre_hook",
                    create=True,
            ) as mock_bwd:
                result = style.apply(module, mesh, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj)

            self.assertIs(result, module)
            self.assertEqual(mock_pre.call_count, 1)
            self.assertEqual(len(module._forward_hooks), 1)  # pylint: disable=protected-access
            if style.ulysses_degree == 1:
                self.assertEqual(len(q_proj._forward_hooks), 0)  # pylint: disable=protected-access
                self.assertEqual(len(k_proj._forward_hooks), 1)  # pylint: disable=protected-access
                self.assertEqual(len(v_proj._forward_hooks), 1)  # pylint: disable=protected-access
                self.assertEqual(mock_bwd.call_count, 2)
            else:
                self.assertEqual(len(q_proj._forward_hooks), 1)  # pylint: disable=protected-access
                self.assertEqual(len(k_proj._forward_hooks), 1)  # pylint: disable=protected-access
                self.assertEqual(len(v_proj._forward_hooks), 1)  # pylint: disable=protected-access
                self.assertEqual(mock_bwd.call_count, 3)

    def test_apply_registers_async_hybrid_hooks(self):
        """Async apply should register hybrid A2A hooks on a named 2-D CP mesh."""
        style = async_cp_module.AsyncContextParallel(ulysses_degree=2)
        module = nn.Identity()
        q_proj = nn.Identity()
        k_proj = nn.Identity()
        v_proj = nn.Identity()

        with patch.object(async_cp_module.platform, "register_forward_pre_hook", create=True) as mock_pre, \
                patch.object(async_cp_module.platform, "register_full_backward_pre_hook", create=True) as mock_bwd:
            result = style.apply(
                module,
                _FakeTwoDMesh(),
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
            )

        self.assertIs(result, module)
        self.assertEqual(mock_pre.call_count, 1)
        self.assertEqual(mock_bwd.call_count, 3)
        self.assertEqual(len(module._forward_hooks), 1)  # pylint: disable=protected-access

    def test_projection_post_hooks_record_async_slots(self):
        """Projection post-hooks should launch async collectives and store metadata slots."""
        output = torch.randn(1, 4, 4, 2)
        fwd_slots = {"q": None, "k": None}

        with patch.object(
                async_cp_module,
                "_launch_async_a2a_seq_to_head",
                return_value=("a2a_work", "a2a_out"),
        ) as mock_a2a:
            result = self.style._proj_post_hook(  # pylint: disable=protected-access
                None,
                (),
                output,
                key="q",
                submesh="cp_mesh",
                group="group",
                world_size=2,
                fwd_slots=fwd_slots,
                layout_mesh="layout_mesh",
                layout_placements=(Shard(2),),
            )

        self.assertIs(result, output)
        mock_a2a.assert_called_once()
        self.assertEqual(fwd_slots["q"]["work"], "a2a_work")
        self.assertEqual(fwd_slots["q"]["out_perm"], "a2a_out")
        self.assertEqual(fwd_slots["q"]["layout"], ("layout_mesh", (Shard(2),)))

        with patch.object(
                async_cp_module,
                "_launch_async_allgather_seq",
                return_value=("ag_work", "ag_out"),
        ) as mock_ag:
            result = self.style._proj_ag_post_hook(  # pylint: disable=protected-access
                None,
                (),
                output,
                key="k",
                submesh="cp_mesh",
                group="group",
                world_size=2,
                fwd_slots=fwd_slots,
            )

        self.assertIs(result, output)
        mock_ag.assert_called_once()
        self.assertEqual(fwd_slots["k"]["work"], "ag_work")
        self.assertEqual(fwd_slots["k"]["out_perm"], "ag_out")
        self.assertEqual(fwd_slots["k"]["layout"], ("cp_mesh", (Replicate(),)))

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

    def test_wait_helpers_clear_slots_and_delegate_to_platform(self):
        """Wait helpers should consume async slots and forward parameters to platform waits."""
        tensor = torch.tensor([1.0])
        fwd_slots = {"q": {"work": "work", "out_perm": "out"}}
        bwd_slot = []

        with patch.object(
                async_cp_module.platform,
                "differentiable_async_a2a_wait",
                return_value="waited",
                create=True,
        ) as mock_wait:
            result = self.style._wait_a2a(  # pylint: disable=protected-access
                tensor,
                group="group",
                world_size=2,
                fwd_slots=fwd_slots,
                key="q",
                bwd_slot=bwd_slot,
            )

        self.assertEqual(result, "waited")
        self.assertIsNone(fwd_slots["q"])
        mock_wait.assert_called_once_with(tensor, "work", "out", "group", 2, 1, 2, bwd_slot)

        with patch.object(
                async_cp_module.platform,
                "differentiable_async_allgather_wait",
                return_value="gathered",
                create=True,
        ) as mock_gather:
            result = self.style._wait_allgather(  # pylint: disable=protected-access
                tensor,
                group="group",
                world_size=2,
                work="work",
                out_perm="out",
                bwd_slot=bwd_slot,
            )

        self.assertEqual(result, "gathered")
        mock_gather.assert_called_once_with(tensor, "work", "out", "group", 2, 1, bwd_slot)

    def test_qkv_helpers_handle_positional_missing_and_kwargs(self):
        """QKV helper methods should cover positional, keyword, and missing paths."""
        style = async_cp_module.AsyncContextParallel(
            qkv_indices=(0, 10, 11),
            qkv_kwarg_names=("query", "key", "value"),
        )
        args = [torch.tensor([1.0])]
        kwargs = {"key": torch.tensor([2.0])}

        self.assertTrue(torch.equal(style._get_qkv_value(args, kwargs, 0), args[0]))  # pylint: disable=protected-access
        self.assertTrue(torch.equal(style._get_qkv_value(args, kwargs, 1), kwargs["key"]))  # pylint: disable=protected-access
        self.assertIsNone(style._get_qkv_value(args, kwargs, 2))  # pylint: disable=protected-access

        style._set_qkv_value(args, kwargs, 0, "q")  # pylint: disable=protected-access
        style._set_qkv_value(args, kwargs, 1, "k")  # pylint: disable=protected-access
        style._set_qkv_value(args, kwargs, 2, "v")  # pylint: disable=protected-access

        self.assertEqual(args[0], "q")
        self.assertEqual(kwargs["key"], "k")
        self.assertNotIn("value", kwargs)

    def test_apply_qkv_transforms_skips_missing_values(self):
        """QKV transform helper should continue over absent K/V inputs."""
        style = async_cp_module.AsyncContextParallel(qkv_indices=(0, 10, 11))
        q = torch.tensor([1.0])
        transform = MagicMock(side_effect=lambda original, local: original + local)

        args, kwargs = style._apply_qkv_transforms(  # pylint: disable=protected-access
            (q,),
            {},
            submesh="cp_mesh",
            transform_q=transform,
            transform_k=MagicMock(),
            transform_v=MagicMock(),
        )

        self.assertTrue(torch.equal(args[0], q + q))
        self.assertEqual(kwargs, {})
        transform.assert_called_once()

    def test_ulysses_pre_hook_with_dtensor_wait_path(self):
        """Ulysses pre-hook should use DTensor wrapping path when a ds submesh is supplied."""
        style = async_cp_module.AsyncContextParallel(qkv_indices=(0, 10, 11))
        q = torch.tensor([1.0])

        with patch.object(style, "_wait_a2a_dtensor", return_value=q + 1.0) as mock_wait:
            args, kwargs = style._attn_pre_hook_ulysses(  # pylint: disable=protected-access
                None,
                (q,),
                {},
                group="group",
                world_size=2,
                fwd_slots={"q": "slot", "k": None, "v": None},
                bwd_slots={"q": [], "k": [], "v": []},
                ds_submesh="ds_mesh",
            )

        self.assertTrue(torch.equal(args[0], q + 1.0))
        self.assertEqual(kwargs, {})
        mock_wait.assert_called_once()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_ulysses_layout_from_non_cp_dtensor_input(self, mock_mesh_platform):
        """Pure Ulysses layout helper should preserve incoming non-CP DTensor layout."""
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

        mesh, placements = self.style._ulysses_cp_layout_from_input(  # pylint: disable=protected-access
            tp_dtensor,
            cp_mesh,
        )

        self.assertEqual(mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(placements, (StridedShard(2, 2), Shard(2)))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_attn_post_hook_ata_reconstructs_outputs(self, mock_mesh_platform):
        """Async attention post-hook should reverse ATA for tuple and scalar outputs."""
        _setup_mock_mesh_platform(mock_mesh_platform, world_size=1)
        mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(1,),
            mesh_dim_names=("cp",),
            init_backend=False,
        )
        local = torch.randn(1, 4, 2, 2)
        seq_dtensor = DTensor.from_local(local, mesh, (Shard(1),))

        with patch.object(async_cp_module, "_gather_head_to_seq", return_value=seq_dtensor) as mock_gather:
            outputs = self.style._attn_post_hook_ata(  # pylint: disable=protected-access
                None,
                (),
                (seq_dtensor, "keep"),
                ds_submesh=mesh,
            )

        self.assertTrue(torch.equal(outputs[0], seq_dtensor.to_local()))
        self.assertEqual(outputs[1], "keep")
        mock_gather.assert_called_once()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_wrap_async_cp_result_supports_legacy_slot_tuple(self, mock_mesh_platform):
        """Legacy ``(work, out_perm)`` slots should fall back to the provided CP layout."""
        _setup_mock_mesh_platform(mock_mesh_platform, world_size=2)
        cp_mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2,),
            mesh_dim_names=("cp",),
            rank_list=(0, 1),
            init_backend=False,
        )
        local = torch.randn(1, 4, 2, 2)

        result = async_cp_module._wrap_async_cp_result(  # pylint: disable=protected-access
            local,
            ("legacy_work", "legacy_out"),
            cp_mesh,
            (Shard(1),),
        )

        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("cp",))
        self.assertEqual(result.placements, (Shard(1),))
        self.assertTrue(torch.equal(result.to_local(), local))

    def test_proj_bwd_pre_hook_waits_and_reconstructs_sequence_grad(self):
        """Projection backward pre-hook should wait the async A2A and restore seq layout."""
        work = _FakeWork()
        out_perm = torch.arange(2 * 1 * 3 * 2 * 1).reshape(2, 1, 3, 2, 1)
        expected = async_cp_module._a2a_reconstruct(  # pylint: disable=protected-access
            out_perm,
            self.style.head_dim,
        )

        grad_output = (torch.zeros_like(expected), "keep")
        result = self.style._proj_bwd_pre_hook(  # pylint: disable=protected-access
            None,
            grad_output,
            bwd_slot=[(work, out_perm)],
        )

        self.assertTrue(work.wait_called)
        self.assertTrue(torch.equal(result[0], expected))
        self.assertEqual(result[1], "keep")

    def test_proj_ag_bwd_pre_hook_waits_and_reconstructs_gather_dim(self):
        """Projection all-gather backward pre-hook should move gather dim back from front."""
        work = _FakeWork()
        out_perm = torch.randn(4, 2, 3)
        expected = async_cp_module._allgather_reconstruct(  # pylint: disable=protected-access
            out_perm,
            gather_dim=1,
        )

        result = self.style._proj_ag_bwd_pre_hook(  # pylint: disable=protected-access
            None,
            (torch.zeros_like(expected),),
            bwd_slot=[(work, out_perm, 1)],
        )

        self.assertTrue(work.wait_called)
        self.assertTrue(torch.equal(result[0], expected))

    def test_apply_rejects_non_divisible_ulysses_degree_with_async_projections(self):
        """Async apply should validate ulysses_degree before registering projection hooks."""
        style = async_cp_module.AsyncContextParallel(ulysses_degree=3)
        module = object()
        mesh = _FakeMesh(4)
        q_proj = object()
        k_proj = object()
        v_proj = object()

        with self.assertRaisesRegex(ValueError, "cp_size \\(4\\).*ulysses_degree \\(3\\)"):
            style.apply(module, mesh, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj)

    def test_apply_without_all_projections_falls_back_to_sync_context_parallel(self):
        """Async CP should reuse the sync CP path when any projection hook is missing."""
        module = object()
        mesh = object()
        q_proj = object()
        v_proj = object()

        with patch.object(
                async_cp_module.ContextParallel,
                "apply",
                autospec=True,
                return_value="fallback",
        ) as mock_apply:
            result = self.style.apply(module, mesh, q_proj=q_proj, k_proj=None, v_proj=v_proj)

        self.assertEqual(result, "fallback")
        mock_apply.assert_called_once_with(self.style, module, mesh)

    def test_apply_load_balance_uses_sync_context_parallel_even_with_projections(self):
        """Load-balanced Colossal CP keeps the synchronous implementation."""
        style = async_cp_module.AsyncContextParallel(ulysses_degree=1, load_balance=True)
        module = object()
        mesh = _FakeMesh(2)
        q_proj = object()
        k_proj = object()
        v_proj = object()

        with patch.object(
                async_cp_module.ContextParallel,
                "apply",
                autospec=True,
                return_value="fallback",
        ) as mock_apply:
            result = style.apply(module, mesh, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj)

        self.assertEqual(result, "fallback")
        mock_apply.assert_called_once_with(style, module, mesh)


if __name__ == "__main__":
    unittest.main()

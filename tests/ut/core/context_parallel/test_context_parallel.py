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
"""CPU-only unit tests for synchronous context-parallel hook wiring."""
import os
import unittest
from functools import partial
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.context_parallel import context_parallel as cp_module  # noqa: E402
from hyper_parallel.core.context_parallel.context_parallel import ContextParallel  # noqa: E402
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP  # noqa: E402
from hyper_parallel.core.dtensor.dtensor import DTensor  # noqa: E402
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard  # noqa: E402
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType  # noqa: E402


def _patch_torch_dist_rank(world_size=1):
    """Patch torch distributed rank APIs used by local DTensor redistribution."""
    return patch.multiple(
        "torch.distributed",
        get_rank=MagicMock(return_value=0),
        get_world_size=MagicMock(return_value=world_size),
    )


class _FakeMeshTensor:
    """Minimal mesh tensor stub exposing ``numel``."""

    def __init__(self, size):
        self._size = size

    def numel(self):
        return self._size


class _FakeMesh:
    """Small DeviceMesh stub for apply-branch tests."""

    def __init__(self, size, *, ndim=1, mesh_dim_names=None, rank_list=None, submeshes=None):
        self.mesh = _FakeMeshTensor(size)
        self.ndim = ndim
        self.mesh_dim_names = mesh_dim_names
        self.rank_list = tuple(range(size)) if rank_list is None else tuple(rank_list)
        self.device_type = "cpu"
        self._submeshes = submeshes or {}

    def __getitem__(self, name):
        return self._submeshes[name]


class _IdentityModule(nn.Module):
    """Leaf module whose forward can be wrapped by ContextParallel."""

    def forward(self, *args, **kwargs):
        if kwargs:
            return args, kwargs
        return args


class TestContextParallel(unittest.TestCase):
    """Unit tests for :class:`ContextParallel` without launching collectives."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @staticmethod
    def _setup_mock_platform(mock_platform, world_size=1):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.Tensor = torch.Tensor
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_mesh(self, mock_platform, mesh_shape=(1,), mesh_dim_names=("cp",)):
        self._setup_mock_platform(mock_platform, world_size=int(torch.tensor(mesh_shape).prod().item()))
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
            init_backend=False,
        )

    def test_init_rejects_load_balance_outside_colossal_mode(self):
        """Head-tail load balance is only valid for Pure Colossal CP."""
        with self.assertRaisesRegex(ValueError, "load_balance=True"):
            ContextParallel(load_balance=True)

    def test_apply_rejects_non_divisible_ulysses_degree(self):
        """Hybrid degree must divide the CP mesh size."""
        style = ContextParallel(ulysses_degree=3)
        with self.assertRaisesRegex(ValueError, "cp_size \\(4\\).*ulysses_degree \\(3\\)"):
            style.apply(_IdentityModule(), _FakeMesh(4))

    def test_apply_registers_colossal_and_ulysses_hooks(self):
        """Apply should register one pre-hook and one post-hook in sync CP modes."""
        for style, mesh in [
                (ContextParallel(ulysses_degree=1), _FakeMesh(2)),
                (ContextParallel(), _FakeMesh(2)),
        ]:
            module = _IdentityModule()
            returned = style.apply(module, mesh)

            self.assertIs(returned, module)
            self.assertEqual(len(module._forward_pre_hooks), 1)  # pylint: disable=protected-access
            self.assertEqual(len(module._forward_hooks), 1)  # pylint: disable=protected-access

    def test_apply_registers_hybrid_hooks_on_named_2d_mesh(self):
        """Hybrid CP should take the ds submesh from a named 2-D mesh."""
        ds_mesh = _FakeMesh(2)
        mesh = _FakeMesh(
            4,
            ndim=2,
            mesh_dim_names=("co", "ds"),
            submeshes={"ds": ds_mesh},
        )
        module = _IdentityModule()

        ContextParallel(ulysses_degree=2).apply(module, mesh)

        self.assertEqual(len(module._forward_pre_hooks), 1)  # pylint: disable=protected-access
        pre_hook = next(iter(module._forward_pre_hooks.values()))  # pylint: disable=protected-access
        self.assertEqual(pre_hook.keywords["ds_submesh"], ds_mesh)
        self.assertEqual(pre_hook.keywords["ds_size"], 2)

    def test_apply_load_balance_wraps_forward_with_rank_metadata(self):
        """Load-balanced Colossal CP should wrap forward with pair-rank metadata."""
        module = _IdentityModule()
        mesh = _FakeMesh(2, rank_list=(0, 1))

        with patch.object(cp_module.platform, "get_rank", return_value=0):
            ContextParallel(ulysses_degree=1, load_balance=True).apply(module, mesh)

        self.assertIsInstance(module.forward, partial)
        self.assertEqual(module.forward.keywords["local_idx"], 0)
        self.assertEqual(module.forward.keywords["target_idx"], 1)
        self.assertEqual(module.forward.keywords["ws"], 2)
        self.assertEqual(module.forward.keywords["peer_rank"], 1)

    def test_build_2d_mesh_reuses_named_2d_mesh_and_rejects_unnamed(self):
        """_build_2d_mesh should validate pre-built 2-D meshes."""
        named_mesh = _FakeMesh(4, ndim=2, mesh_dim_names=("co", "ds"))
        self.assertIs(cp_module._build_2d_mesh(named_mesh, ds=2, co=2), named_mesh)  # pylint: disable=protected-access

        unnamed_mesh = _FakeMesh(4, ndim=2, mesh_dim_names=None)
        with self.assertRaisesRegex(ValueError, "mesh_dim_names"):
            cp_module._build_2d_mesh(unnamed_mesh, ds=2, co=2)  # pylint: disable=protected-access

    def test_build_2d_mesh_tiles_1d_ranks(self):
        """_build_2d_mesh should tile adjacent 1-D ranks into co x ds rows."""
        source_mesh = _FakeMesh(4, rank_list=(10, 11, 12, 13))
        constructed = object()

        with patch.object(cp_module, "DeviceMesh", return_value=constructed) as mock_device_mesh:
            result = cp_module._build_2d_mesh(source_mesh, ds=2, co=2)  # pylint: disable=protected-access

        self.assertIs(result, constructed)
        mock_device_mesh.assert_called_once_with(
            "cpu",
            [[10, 11], [12, 13]],
            mesh_dim_names=("co", "ds"),
        )

    def test_ensure_1d_flattens_multi_dimensional_mesh(self):
        """_ensure_1d should flatten multi-dimensional meshes for Colossal CP."""
        mesh = _FakeMesh(4, ndim=2, mesh_dim_names=("dp", "cp"), rank_list=(3, 5, 7, 9))
        constructed = object()

        with patch.object(cp_module, "DeviceMesh", return_value=constructed) as mock_device_mesh:
            result = cp_module._ensure_1d(mesh)  # pylint: disable=protected-access

        self.assertIs(result, constructed)
        mock_device_mesh.assert_called_once_with("cpu", [3, 5, 7, 9], mesh_dim_names=("cp",))

    def test_build_hybrid_cp_mesh_validates_mesh_rank(self):
        """Hybrid CP only accepts 1-D or named 2-D CP meshes."""
        unnamed_2d = _FakeMesh(4, ndim=2, mesh_dim_names=None)
        with self.assertRaisesRegex(ValueError, "mesh_dim_names"):
            cp_module._build_hybrid_cp_mesh(unnamed_2d, ds=2, co=2)  # pylint: disable=protected-access

        three_d = _FakeMesh(8, ndim=3, mesh_dim_names=("a", "b", "c"))
        with self.assertRaisesRegex(ValueError, "expects a 1-D or 2-D CP mesh"):
            cp_module._build_hybrid_cp_mesh(three_d, ds=2, co=4)  # pylint: disable=protected-access

    def test_output_layout_stack_defaults_and_passthrough_helpers(self):
        """Output helper defaults should leave non-tensors and empty stacks untouched."""
        module = _IdentityModule()
        self.assertEqual(
            cp_module._pop_output_layout(None),  # pylint: disable=protected-access
            (cp_module._OUTPUT_LOCAL, None),  # pylint: disable=protected-access
        )
        self.assertEqual(
            cp_module._pop_output_layout(module),  # pylint: disable=protected-access
            (cp_module._OUTPUT_LOCAL, None),  # pylint: disable=protected-access
        )
        self.assertEqual(
            cp_module._drop_cp_from_output("keep", None, (Shard(1),)),  # pylint: disable=protected-access
            "keep",
        )
        self.assertEqual(
            cp_module._wrap_cp_output_dtensor("keep", object(), (Shard(1),)),  # pylint: disable=protected-access
            "keep",
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_wrap_cp_output_dtensor_redistributes_existing_cp_output(self, mock_mesh_platform):
        """CP output wrapping should redistribute existing DTensor placements when needed."""
        mesh = self._make_mesh(mock_mesh_platform)
        local = torch.randn(2, 4, 3, 5)
        dtensor = DTensor.from_local(local, mesh, (Replicate(),))

        result = cp_module._wrap_cp_output_dtensor(  # pylint: disable=protected-access
            dtensor,
            mesh,
            (Shard(1),),
        )

        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.to_hash(), mesh.to_hash())
        self.assertEqual(result.placements, (Shard(1),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pre_hook_colossal_rewrites_positional_and_keyword_qkv(self, mock_mesh_platform):
        """Colossal pre-hook should shard Q and gather K/V for args and kwargs."""
        mesh = self._make_mesh(mock_mesh_platform)
        style = ContextParallel(
            seq_dim=1,
            head_dim=2,
            ulysses_degree=1,
            qkv_kwarg_names=("query", "key", "value"),
        )
        q = torch.randn(2, 4, 3, 5)
        k = torch.randn(2, 4, 3, 5)
        v = torch.randn(2, 4, 3, 5)
        q_kw = torch.randn(2, 4, 3, 5)
        k_kw = torch.randn(2, 4, 3, 5)
        v_kw = torch.randn(2, 4, 3, 5)

        with _patch_torch_dist_rank():
            args, kwargs = style._pre_hook_colossal(  # pylint: disable=protected-access
                None,
                (q, k, v),
                {"query": q_kw, "key": k_kw, "value": v_kw, "mask": "keep"},
                mesh,
            )

        self.assertEqual(args[0].placements, (Shard(1),))
        self.assertEqual(args[1].placements, (Replicate(),))
        self.assertEqual(args[2].placements, (Replicate(),))
        self.assertEqual(kwargs["query"].placements, (Shard(1),))
        self.assertEqual(kwargs["key"].placements, (Replicate(),))
        self.assertEqual(kwargs["value"].placements, (Replicate(),))
        self.assertEqual(kwargs["mask"], "keep")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pre_hook_ulysses_rewrites_positional_qkv(self, mock_mesh_platform):
        """Ulysses pre-hook should apply ATA to positional Q/K/V inputs."""
        mesh = self._make_mesh(mock_mesh_platform)
        style = ContextParallel(seq_dim=1, head_dim=2)
        q = torch.randn(2, 4, 4, 5)
        k = torch.randn(2, 4, 4, 5)
        v = torch.randn(2, 4, 4, 5)

        with _patch_torch_dist_rank():
            args, kwargs = style._pre_hook_ulysses(  # pylint: disable=protected-access
                None,
                (q, k, v),
                {"mask": "keep"},
                ds_submesh=mesh,
                ds_size=1,
            )

        self.assertEqual(kwargs, {"mask": "keep"})
        self.assertEqual(args[0].placements, (Shard(2),))
        self.assertEqual(args[1].placements, (Shard(2),))
        self.assertEqual(args[2].placements, (Shard(2),))

    def test_pre_hook_ulysses_rewrites_qkv_kwargs(self):
        """Ulysses pre-hook should apply ATA to Q/K/V keyword arguments."""
        style = ContextParallel(qkv_indices=(10, 11, 12), qkv_kwarg_names=("query", "key", "value"))
        q = torch.tensor([1.0])
        k = torch.tensor([2.0])
        v = torch.tensor([3.0])

        with patch.object(
                cp_module,
                "_scatter_seq_to_head",
                side_effect=lambda tensor, *unused: tensor + 1.0,
        ) as mock_scatter:
            args, kwargs = style._pre_hook_ulysses(  # pylint: disable=protected-access
                None,
                (),
                {"query": q, "key": k, "value": v, "mask": "keep"},
                ds_submesh="ds_mesh",
                ds_size=2,
            )

        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["query"], q + 1.0))
        self.assertTrue(torch.equal(kwargs["key"], k + 1.0))
        self.assertTrue(torch.equal(kwargs["value"], v + 1.0))
        self.assertEqual(kwargs["mask"], "keep")
        self.assertEqual(mock_scatter.call_count, 3)

    def test_scatter_seq_to_head_rejects_non_divisible_heads(self):
        """ATA requires the head dimension to divide evenly by the Ulysses degree."""
        x = torch.randn(2, 4, 3, 5)
        with self.assertRaisesRegex(ValueError, "num_heads \\(3\\).*ulysses_degree \\(2\\)"):
            cp_module._scatter_seq_to_head(  # pylint: disable=protected-access
                x,
                submesh=object(),
                seq_dim=1,
                head_dim=2,
                submesh_size=2,
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_scatter_and_gather_sequence_helpers_return_dtensor(self, mock_mesh_platform):
        """CP scatter/gather helpers should return DTensors with requested placements."""
        mesh = self._make_mesh(mock_mesh_platform)
        x = torch.randn(2, 4, 4, 5)

        with _patch_torch_dist_rank():
            scattered = cp_module._scatter_seq_to_head(  # pylint: disable=protected-access
                x,
                submesh=mesh,
                seq_dim=1,
                head_dim=2,
                submesh_size=1,
            )
            gathered = cp_module._gather_head_to_seq(  # pylint: disable=protected-access
                scattered,
                submesh=mesh,
                seq_dim=1,
                head_dim=2,
            )

        self.assertEqual(scattered.placements, (Shard(2),))
        self.assertEqual(gathered.placements, (Shard(1),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_pre_hook_hybrid_rewrites_2d_placements(self, mock_mesh_platform):
        """Hybrid pre-hook should use 2-D placements for Q and gathered K/V."""
        mesh = self._make_mesh(mock_mesh_platform, mesh_shape=(1, 1), mesh_dim_names=("co", "ds"))
        style = ContextParallel(
            seq_dim=1,
            head_dim=2,
            ulysses_degree=1,
            qkv_kwarg_names=("query", "key", "value"),
        )
        q = torch.randn(2, 4, 3, 5)
        k = torch.randn(2, 4, 3, 5)
        v = torch.randn(2, 4, 3, 5)
        q_kw = torch.randn(2, 4, 3, 5)
        k_kw = torch.randn(2, 4, 3, 5)
        v_kw = torch.randn(2, 4, 3, 5)

        with _patch_torch_dist_rank():
            args, kwargs = style._pre_hook_hybrid(  # pylint: disable=protected-access
                None,
                (q, k, v),
                {"query": q_kw, "key": k_kw, "value": v_kw},
                hybrid_cp_mesh=mesh,
                ds_submesh=mesh["ds"],
                ds_size=1,
            )

        self.assertEqual(args[0].placements, (Shard(1), Shard(2)))
        self.assertEqual(args[1].placements, (Replicate(), Shard(2)))
        self.assertEqual(args[2].placements, (Replicate(), Shard(2)))
        self.assertEqual(kwargs["query"].placements, (Shard(1), Shard(2)))
        self.assertEqual(kwargs["key"].placements, (Replicate(), Shard(2)))
        self.assertEqual(kwargs["value"].placements, (Replicate(), Shard(2)))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_hybrid_layout_preserves_non_cp_dtensor_and_rejects_bad_heads(self, mock_mesh_platform):
        """Hybrid CP layout helpers should preserve TP layout and validate head divisibility."""
        self._setup_mock_platform(mock_mesh_platform, world_size=8)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("co", "ds", "tp"),
            rank_list=tuple(range(8)),
            init_backend=False,
        )
        hybrid_cp_mesh = root[("co", "ds")]
        ds_mesh = root["ds"]
        tp_mesh = root["tp"]
        tp_dtensor = DTensor.from_local(torch.randn(1, 4, 4, 2), tp_mesh, (Shard(2),))
        style = ContextParallel(seq_dim=1, head_dim=2)

        out_mesh, out_placements = style._hybrid_cp_layout_from_input(  # pylint: disable=protected-access
            tp_dtensor,
            hybrid_cp_mesh,
            (Shard(1), Shard(2)),
        )

        self.assertEqual(out_mesh.mesh_dim_names, ("co", "ds", "tp"))
        self.assertEqual(out_placements, (Shard(1), StridedShard(2, 2), Shard(2)))
        self.assertFalse(style._needs_hybrid_ata("not-a-tensor", hybrid_cp_mesh))  # pylint: disable=protected-access

        bad_heads = torch.randn(1, 4, 3, 2)
        with self.assertRaisesRegex(ValueError, "num_heads \\(3\\).*ulysses_degree \\(2\\)"):
            style._ata_scatter_to_hybrid(  # pylint: disable=protected-access
                bad_heads,
                ds_submesh=ds_mesh,
                hybrid_cp_mesh=hybrid_cp_mesh,
                ds_size=2,
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_post_hook_colossal_returns_local_outputs(self, mock_mesh_platform):
        """Colossal post-hook should convert direct DTensor outputs to locals."""
        mesh = self._make_mesh(mock_mesh_platform)
        local = torch.randn(2, 4)
        dtensor = DTensor.from_local(local, mesh, (Replicate(),))
        style = ContextParallel(ulysses_degree=1)

        outputs = style._post_hook_colossal(  # pylint: disable=protected-access
            None,
            (),
            (dtensor, "keep"),
            co_submesh=mesh,
        )

        self.assertTrue(torch.equal(outputs[0], local))
        self.assertEqual(outputs[1], "keep")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_post_hook_colossal_preserves_non_cp_dtensor_output_layout(self, mock_mesh_platform):
        """Colossal post-hook should drop CP and restore the incoming non-CP layout."""
        self._setup_mock_platform(mock_mesh_platform, world_size=4)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("cp", "tp"),
            rank_list=(0, 1, 2, 3),
            init_backend=False,
        )
        cp_mesh = root["cp"]
        tp_mesh = root["tp"]
        local_q = torch.randn(1, 4, 4, 2)
        tp_q = DTensor.from_local(local_q, tp_mesh, (Shard(2),))
        module = _IdentityModule()
        style = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)

        with _patch_torch_dist_rank(world_size=4):
            args, _ = style._pre_hook_colossal(  # pylint: disable=protected-access
                module,
                (tp_q,),
                {},
                co_submesh=cp_mesh,
            )

        self.assertEqual(args[0].device_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(args[0].placements, (Shard(1), Shard(2)))

        local_out = torch.randn(1, 4, 4, 2)
        output = style._post_hook_colossal(  # pylint: disable=protected-access
            module,
            (),
            local_out,
            co_submesh=cp_mesh,
        )

        self.assertIsInstance(output, DTensor)
        self.assertEqual(output.device_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(output.placements, (Shard(2),))
        self.assertTrue(torch.equal(output.to_local(), local_out))
        self.assertFalse(hasattr(module, cp_module._OUTPUT_LAYOUT_STACK_ATTR))  # pylint: disable=protected-access

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_post_hook_ata_reconstructs_each_tensor_output(self, mock_mesh_platform):
        """ATA post-hook should reverse head sharding and keep non-tensor outputs."""
        mesh = self._make_mesh(mock_mesh_platform)
        style = ContextParallel(seq_dim=1, head_dim=2)
        local = torch.randn(2, 4, 3, 5)
        gathered = DTensor.from_local(local, mesh, (Shard(1),))

        with patch.object(cp_module, "_gather_head_to_seq", return_value=gathered) as mock_gather:
            outputs = style._post_hook_ata(  # pylint: disable=protected-access
                None,
                (),
                (torch.tensor([1.0]), "keep"),
                ds_submesh="ds_mesh",
            )

        self.assertTrue(torch.equal(outputs[0], local))
        self.assertEqual(outputs[1], "keep")
        mock_gather.assert_called_once()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_post_hook_hybrid_output_policies(self, mock_mesh_platform):
        """Hybrid post-hook should support local, CP-DTensor, and non-CP output policies."""
        self._setup_mock_platform(mock_mesh_platform, world_size=8)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("co", "ds"),
            rank_list=(0, 1, 2, 3),
            init_backend=False,
        )
        ds_mesh = root["ds"]
        style = ContextParallel(seq_dim=1, head_dim=2)
        local = torch.randn(1, 4, 2, 2)
        seq_dtensor = DTensor.from_local(local, ds_mesh, (Shard(1),))

        with patch.object(cp_module, "_gather_head_to_seq", return_value=seq_dtensor):
            self.assertEqual(
                style._post_hook_hybrid(  # pylint: disable=protected-access
                    None,
                    (),
                    "keep",
                    hybrid_cp_mesh=root,
                    ds_submesh=ds_mesh,
                ),
                "keep",
            )
            local_out = style._post_hook_hybrid(  # pylint: disable=protected-access
                None,
                (),
                torch.randn(1, 4, 2, 2),
                hybrid_cp_mesh=root,
                ds_submesh=ds_mesh,
            )

        self.assertTrue(torch.equal(local_out, local))

        module = _IdentityModule()
        cp_module._push_output_layout(  # pylint: disable=protected-access
            module,
            (cp_module._OUTPUT_CP, root),  # pylint: disable=protected-access
        )
        with patch.object(cp_module, "_gather_head_to_seq", return_value=seq_dtensor):
            cp_out = style._post_hook_hybrid(  # pylint: disable=protected-access
                module,
                (),
                torch.randn(1, 4, 2, 2),
                hybrid_cp_mesh=root,
                ds_submesh=ds_mesh,
            )

        self.assertIsInstance(cp_out, DTensor)
        self.assertEqual(cp_out.device_mesh.mesh_dim_names, ("co", "ds"))
        self.assertEqual(cp_out.placements, (Shard(1), Replicate()))

        root_3d = init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("co", "ds", "tp"),
            rank_list=tuple(range(8)),
            init_backend=False,
        )
        tp_mesh = root_3d["tp"]
        non_cp_layout = (tp_mesh, (Shard(2),), root_3d)
        cp_module._push_output_layout(  # pylint: disable=protected-access
            module,
            (cp_module._OUTPUT_NON_CP, non_cp_layout),  # pylint: disable=protected-access
        )
        with patch.object(cp_module, "_gather_head_to_seq", return_value=seq_dtensor):
            non_cp_out = style._post_hook_hybrid(  # pylint: disable=protected-access
                module,
                (),
                [torch.randn(1, 4, 2, 2), "keep"],
                hybrid_cp_mesh=root,
                ds_submesh=ds_mesh,
            )

        self.assertIsInstance(non_cp_out[0], DTensor)
        self.assertEqual(non_cp_out[0].device_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(non_cp_out[0].placements, (Shard(2),))
        self.assertEqual(non_cp_out[1], "keep")


if __name__ == "__main__":
    unittest.main()

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
"""Unit tests for torch fully_shard parameter helper paths."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo, ShardedState
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, HSDPMeshInfo, MixedPrecisionPolicy, TPShardMetaInfo
from hyper_parallel.platform.torch.fully_shard.param import (
    AllGatherCommCtx,
    AllReduceCommCtx,
    ParameterHookMigrator,
    ReduceScatterCommCtx,
    TorchHSDPParamV2,
)


def _new_param():
    """Create an uninitialized TorchHSDPParamV2 with common test fields."""
    hsdp_param = object.__new__(TorchHSDPParamV2)
    hsdp_param.unsharded_param_buffers = []
    hsdp_param.device = torch.device("cpu")
    hsdp_param.shard_size = 2
    hsdp_param.shard_world_size = 2
    hsdp_param.dp_size = 1
    hsdp_param.replicate_world_size = 1
    hsdp_param.is_sharded = True
    hsdp_param.hsdp_placement = Shard(0)
    hsdp_param._orig_size = torch.Size((4,))
    hsdp_param._orig_param_is_dtensor = False
    hsdp_param._orig_dtensor_mesh = None
    hsdp_param._orig_dtensor_placements = None
    hsdp_param.tp_grad_info = None
    hsdp_param._sharded_param_data = torch.tensor([1.0, 2.0])
    hsdp_param.sharded_size = torch.Size((2,))
    hsdp_param.padded_sharded_param_size = torch.Size((2,))
    hsdp_param.contiguous_sharded_stride = (1,)
    hsdp_param._sharding_spec = SimpleNamespace(mesh="mesh", placements=(Shard(0),))
    hsdp_param.sharded_param = SimpleNamespace(
        shape=torch.Size((2,)),
        grad=None,
        requires_grad=True,
        _local_tensor=torch.tensor([1.0, 2.0]),
    )
    hsdp_param._unsharded_param = SimpleNamespace(grad=None)
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.mp_policy = MixedPrecisionPolicy()
    hsdp_param.orig_dtype = torch.float32
    hsdp_param.reduce_dtype = None
    hsdp_param.gradient_scaling_factor = None
    hsdp_param.offload_to_cpu = False
    hsdp_param.pin_memory = False
    hsdp_param.prefetch_handle = None
    hsdp_param.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
    hsdp_param.allgather_comm_ctx = AllGatherCommCtx()
    hsdp_param._grad = torch.ones(2)
    hsdp_param.all_reduce_comm_ctx = AllReduceCommCtx()
    hsdp_param.reduce_partial_output = None
    hsdp_param.mesh_info = object.__new__(HSDPMeshInfo)
    hsdp_param.mesh_info.shard_process_group = None
    hsdp_param.mesh_info.replicate_process_group = None
    hsdp_param.sharded_state = ShardedState.SHARDED
    return hsdp_param


class TestTorchHSDPParamHelpers(unittest.TestCase):
    """Cover parameter helper behavior without constructing real device meshes."""

    def test_reduce_comm_dtype_prefers_parameter_policy_and_falls_back_to_grad(self):
        """Effective reduction dtype should be resolved entirely by the parameter."""
        hsdp_param = _new_param()
        grad = torch.ones(2, dtype=torch.bfloat16)

        self.assertEqual(hsdp_param.reduce_comm_dtype(grad), torch.bfloat16)

        hsdp_param.reduce_dtype = torch.float16
        self.assertEqual(hsdp_param.reduce_comm_dtype(grad), torch.float16)

    def test_dim0_uneven_init_and_reset_keep_actual_view_on_padded_storage(self):
        """Parameter lifecycle should expose the actual shard while retaining zero padding."""
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.arange(15, dtype=torch.float32).view(5, 3))
        module_info = ParamModuleInfo(module, "weight", [], [])
        mesh_info = object.__new__(FSDPMeshInfo)
        with patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0):
            mesh_info.mesh = DeviceMesh(
                "cpu",
                [0, 1],
                mesh_dim_names=("fsdp",),
                _init_backend=False,
            )
        mesh_info.shard_mesh_dim = 0
        mesh_info.replicate_mesh_dim = None
        mesh_info.shard_mesh_rank = 1
        mesh_info.shard_mesh_size = 2
        mesh_info.shard_process_group = None

        hsdp_param = TorchHSDPParamV2(
            module.weight,
            module_info,
            mesh_info,
            mp_policy=MixedPrecisionPolicy(),
            device=torch.device("cpu"),
        )

        self.assertEqual(hsdp_param.sharded_size, torch.Size((2, 3)))
        self.assertEqual(hsdp_param.padded_sharded_param_size, torch.Size((3, 3)))
        self.assertEqual(hsdp_param.sharded_param.local_shape, torch.Size((2, 3)))
        self.assertEqual(hsdp_param.sharded_param.shape, (5, 3))
        torch.testing.assert_close(hsdp_param._sharded_param_data[-3:], torch.zeros(3))
        self.assertEqual(
            hsdp_param.sharded_param._local_tensor.untyped_storage().data_ptr(),
            hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
        )

        loaded_local_tensor = torch.full((2, 3), 9.0)
        loaded_dtensor = DTensor.from_local(
            loaded_local_tensor,
            hsdp_param._spmd_mesh,
            hsdp_param._spmd_placements,
            shape=(5, 3),
            stride=(3, 1),
        )
        module._parameters["weight"] = torch.nn.Parameter(loaded_dtensor)
        hsdp_param.reset_sharded_param()

        torch.testing.assert_close(hsdp_param.sharded_param.to_local(), loaded_local_tensor)
        torch.testing.assert_close(hsdp_param._sharded_param_data[:6], torch.full((6,), 9.0))
        torch.testing.assert_close(hsdp_param._sharded_param_data[6:], torch.zeros(3))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_plain_tp_local_param_builds_global_fsdp_tp_layout(self, _mock_get_rank):
        """Dual-mode TP metadata should restore the global shape before FSDP sharding."""
        root_mesh = DeviceMesh(
            "cpu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("fsdp", "tp"),
            _init_backend=False,
        )
        mesh_info = object.__new__(FSDPMeshInfo)
        mesh_info.mesh = root_mesh["fsdp"]
        mesh_info.shard_mesh_dim = 0
        mesh_info.replicate_mesh_dim = None
        mesh_info.shard_mesh_rank = 0
        mesh_info.shard_mesh_size = 2
        mesh_info.shard_process_group = None
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.randn(128, 64))

        hsdp_param = TorchHSDPParamV2(
            module.weight,
            ParamModuleInfo(module, "weight", [], []),
            mesh_info,
            mp_policy=MixedPrecisionPolicy(),
            device=torch.device("cpu"),
            tp_grad_info=TPShardMetaInfo(
                root_mesh["tp"],
                (Shard(0),),
                origin_is_dtensor=False,
            ),
        )

        self.assertEqual(hsdp_param.sharded_param.local_shape, torch.Size((64, 64)))
        self.assertEqual(hsdp_param.sharded_param.shape, (256, 64))
        self.assertEqual(hsdp_param._sharding_spec.tensor_shape, (256, 64))

    def test_dim0_smaller_than_world_size_preserves_empty_actual_shape(self):
        """Ranks past the last logical row should expose ``(0, *rest)`` over padded storage."""
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        module_info = ParamModuleInfo(module, "weight", [], [])
        mesh_info = object.__new__(FSDPMeshInfo)
        with patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0):
            mesh_info.mesh = DeviceMesh(
                "cpu",
                [0, 1, 2, 3],
                mesh_dim_names=("fsdp",),
                _init_backend=False,
            )
        mesh_info.shard_mesh_dim = 0
        mesh_info.replicate_mesh_dim = None
        mesh_info.shard_mesh_rank = 3
        mesh_info.shard_mesh_size = 4
        mesh_info.shard_process_group = None

        hsdp_param = TorchHSDPParamV2(
            module.weight,
            module_info,
            mesh_info,
            mp_policy=MixedPrecisionPolicy(),
            device=torch.device("cpu"),
        )

        self.assertEqual(hsdp_param.sharded_size, torch.Size((0, 3)))
        self.assertEqual(hsdp_param.padded_sharded_param_size, torch.Size((1, 3)))
        self.assertEqual(hsdp_param.sharded_param.local_shape, torch.Size((0, 3)))
        self.assertEqual(hsdp_param.sharded_param.shape, (2, 3))
        torch.testing.assert_close(hsdp_param._sharded_param_data, torch.zeros(3))

    def test_init_unsharded_param_buffers_reuse_and_force_recreate(self):
        """All-gather buffers should be reused unless recreation is requested."""
        hsdp_param = _new_param()
        existing = torch.empty(1)
        hsdp_param.unsharded_param_buffers = [existing]

        hsdp_param.init_unsharded_param_buffers([2], [torch.float32], 2, torch.device("cpu"))
        self.assertIs(hsdp_param.unsharded_param_buffers[0], existing)

        with self.assertRaisesRegex(RuntimeError, "stable unsharded parameter"):
            hsdp_param.init_unsharded_param_buffers(
                [2, 1], [torch.float32, torch.float16], 2, torch.device("cpu"), force_recreate=True
            )

        del hsdp_param._unsharded_param
        hsdp_param.init_unsharded_param_buffers(
            [2, 1], [torch.float32, torch.float16], 2, torch.device("cpu"), force_recreate=True
        )
        self.assertEqual([t.numel() for t in hsdp_param.unsharded_param_buffers], [4, 2])
        self.assertEqual(hsdp_param.unsharded_param_buffers[1].dtype, torch.float16)

    def test_init_unsharded_param_plain_and_dtensor(self):
        """Stable unsharded parameter initialization should restore plain and DTensor values."""
        hsdp_param = _new_param()
        del hsdp_param._unsharded_param
        hsdp_param.unsharded_param_buffers = [torch.arange(4, dtype=torch.float32)]
        hsdp_param._contiguous_orig_stride = (1,)
        hsdp_param.tp_grad_info = TPShardMetaInfo(
            "mesh",
            (Shard(0),),
            origin_is_dtensor=False,
        )

        with patch("hyper_parallel.platform.torch.fully_shard.param.DTensor.from_local") as mock_from:
            hsdp_param.init_unsharded_param()
        self.assertIsInstance(hsdp_param.unsharded_param, torch.nn.Parameter)
        self.assertNotIsInstance(hsdp_param.unsharded_param, DTensor)
        torch.testing.assert_close(hsdp_param.unsharded_param, torch.arange(4, dtype=torch.float32))
        mock_from.assert_not_called()

        del hsdp_param._unsharded_param
        hsdp_param.tp_grad_info = TPShardMetaInfo(
            "mesh",
            (Shard(0),),
            origin_is_dtensor=True,
        )
        with patch(
            "hyper_parallel.platform.torch.fully_shard.param.DTensor.from_local",
            return_value="dtensor",
        ) as mock_from:
            with patch(
                "hyper_parallel.platform.torch.fully_shard.param.nn.Parameter",
                side_effect=lambda value, **unused_kwargs: value,
            ):
                hsdp_param.init_unsharded_param()
        self.assertEqual(hsdp_param.unsharded_param, "dtensor")
        mock_from.assert_called_once()

    def test_init_unsharded_param_hides_dim0_padding(self):
        """The module parameter should expose only the logical rows from a padded all-gather buffer."""
        hsdp_param = _new_param()
        del hsdp_param._unsharded_param
        hsdp_param._orig_size = torch.Size((5, 3))
        hsdp_param._contiguous_orig_stride = (3, 1)
        hsdp_param.unsharded_param_buffers = [torch.arange(24, dtype=torch.float32)]

        hsdp_param.init_unsharded_param()

        self.assertEqual(hsdp_param.unsharded_param.shape, torch.Size((5, 3)))
        torch.testing.assert_close(
            hsdp_param.unsharded_param,
            torch.arange(15, dtype=torch.float32).view(5, 3),
        )

    def test_init_unsharded_param_requires_single_buffer(self):
        """Stable unsharded parameter initialization should require exactly one buffer."""
        hsdp_param = _new_param()
        hsdp_param.unsharded_param_buffers = []

        with self.assertRaisesRegex(AssertionError, "Expected 1 unsharded_param_buffer"):
            hsdp_param.init_unsharded_param()

    def test_output_wait_and_clear_helpers(self):
        """Output accessors should wait handles once and clear cached tensors."""
        hsdp_param = _new_param()
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = "rs"
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle = MagicMock()
        hsdp_param.all_reduce_comm_ctx.all_reduce_output = "ar"
        hsdp_param.all_reduce_comm_ctx.all_reduce_handle = MagicMock()

        self.assertEqual(hsdp_param.reduce_scatter_output(), "rs")
        self.assertEqual(hsdp_param.all_reduce_output(), "ar")
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_handle)
        hsdp_param.clear_reduce_scatter_output()
        hsdp_param.clear_all_reduce_output()
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_output)

    def test_apply_reduced_grad_assigns_and_accumulates(self):
        """Reduced gradients should assign new grads and accumulate existing grads."""
        hsdp_param = _new_param()
        hsdp_param.to_sharded_dtensor = MagicMock(
            side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor.clone())
        )
        hsdp_param._unsharded_param.grad = torch.ones(2)
        hsdp_param.orig_dtype = torch.float16

        need_sync = hsdp_param.apply_reduced_grad(torch.tensor([1.0, 2.0]))

        self.assertFalse(need_sync)
        torch.testing.assert_close(
            hsdp_param.sharded_param.grad._local_tensor,
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        self.assertIsNone(hsdp_param._unsharded_param.grad)

        hsdp_param._unsharded_param.grad = torch.ones(2)
        hsdp_param.apply_reduced_grad(torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            hsdp_param.sharded_param.grad._local_tensor,
            torch.tensor([4.0, 6.0], dtype=torch.float16),
        )

    def test_apply_reduced_grad_uses_main_grad(self):
        """Reduced gradients should accumulate into fp32 main grad when enabled."""
        hsdp_param = _new_param()
        hsdp_param.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        hsdp_param.sharded_param.main_grad = SimpleNamespace(_local_tensor=torch.ones(2))
        hsdp_param.to_sharded_dtensor = MagicMock(
            side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor.clone())
        )

        hsdp_param.apply_reduced_grad(torch.tensor([2.0, 3.0]))

        torch.testing.assert_close(hsdp_param.sharded_param.main_grad._local_tensor, torch.tensor([3.0, 4.0]))
        self.assertIsNone(hsdp_param.sharded_param.grad)

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_single_rank_and_mocked_multi_rank(self, mock_all_reduce):
        """All-reduce should skip single-rank groups and launch for multi-rank groups."""
        hsdp_param = _new_param()
        grad = torch.ones(2)
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = grad

        hsdp_param.all_reduce_grad()
        self.assertIs(hsdp_param.all_reduce_comm_ctx.all_reduce_output, grad)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_handle)
        mock_all_reduce.assert_not_called()

        hsdp_param.mesh_info.replicate_process_group = "process-group"
        hsdp_param.replicate_world_size = 2
        hsdp_param.reduce_dtype = torch.float16
        mock_all_reduce.return_value = "handle"
        hsdp_param.all_reduce_grad()
        self.assertEqual(hsdp_param.all_reduce_comm_ctx.all_reduce_handle, "handle")
        self.assertEqual(hsdp_param.all_reduce_comm_ctx.all_reduce_output.dtype, torch.float16)
        mock_all_reduce.assert_called_once()

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_skips_size_one_meshes(self, mock_reduce_scatter):
        """Size-one DP meshes should reduce a retained grad after the parameter is resharded."""
        for mesh_shape in ((1,), (1, 1)):
            with self.subTest(mesh_shape=mesh_shape):
                hsdp_param = _new_param()
                hsdp_param.shard_world_size = 1
                hsdp_param._orig_size = torch.Size((4,))
                hsdp_param.sharded_size = torch.Size((4,))
                hsdp_param._unsharded_param.grad = torch.arange(4, dtype=torch.float32)

                hsdp_param.reduce_scatter_grad()

                torch.testing.assert_close(
                    hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output,
                    torch.arange(4, dtype=torch.float32),
                )
                self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle)
        mock_reduce_scatter.assert_not_called()

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_packs_nonzero_shard_dim_with_chunk_cat(self, mock_reduce_scatter):
        """Nonzero shard dimensions should be packed explicitly with chunk and cat."""
        hsdp_param = _new_param()
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param.sharded_state = ShardedState.UNSHARDED
        hsdp_param.mesh_info.shard_process_group = "shard-group"
        grad = torch.arange(16, dtype=torch.float32).view(4, 4)
        hsdp_param._orig_size = grad.size()
        hsdp_param._unsharded_param.grad = grad
        mock_reduce_scatter.return_value = "work"

        hsdp_param.reduce_scatter_grad(async_op=False)

        expected_grad = torch.cat(torch.chunk(grad, 2, dim=1), dim=0).view(-1)
        output, packed_grad = mock_reduce_scatter.call_args.args
        self.assertIs(output, hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)
        torch.testing.assert_close(packed_grad, expected_grad)
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["op"], torch.distributed.ReduceOp.AVG)
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["group"], "shard-group")
        self.assertFalse(mock_reduce_scatter.call_args.kwargs["async_op"])

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_pads_dim0_and_applies_actual_view(self, mock_reduce_scatter):
        """Non-fused reduction should communicate padded rows and expose only the actual gradient."""
        hsdp_param = _new_param()
        hsdp_param.shard_world_size = 4
        hsdp_param._orig_size = torch.Size((5, 3))
        hsdp_param.sharded_size = torch.Size((1, 3))
        hsdp_param.padded_sharded_param_size = torch.Size((2, 3))
        hsdp_param.sharded_param.shape = hsdp_param.sharded_size
        hsdp_param.sharded_state = ShardedState.UNSHARDED
        hsdp_param.mesh_info.shard_process_group = "shard-group"
        hsdp_param._unsharded_param.grad = torch.arange(15, dtype=torch.float32).view(5, 3)
        mock_reduce_scatter.return_value = "work"

        hsdp_param.reduce_scatter_grad(async_op=False)

        reduce_output, reduce_input = mock_reduce_scatter.call_args.args
        expected_input = torch.zeros(8, 3)
        expected_input[:5].copy_(hsdp_param._unsharded_param.grad)
        torch.testing.assert_close(reduce_input.view(8, 3), expected_input)
        self.assertEqual(reduce_output.numel(), 6)

        hsdp_param.to_sharded_dtensor = MagicMock(
            side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor.clone())
        )
        hsdp_param.apply_reduced_grad(torch.arange(6, dtype=torch.float32))
        torch.testing.assert_close(
            hsdp_param.sharded_param.grad._local_tensor,
            torch.arange(3, dtype=torch.float32).view(1, 3),
        )

        hsdp_param.sharded_param.grad = None
        hsdp_param.apply_reduced_grad(torch.arange(3, dtype=torch.float32).view(1, 3))
        torch.testing.assert_close(
            hsdp_param.sharded_param.grad._local_tensor,
            torch.arange(3, dtype=torch.float32).view(1, 3),
        )

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.all_reduce")
    def test_all_reduce_tp_replicate_grad_inplace_uses_original_replicated_axes(
        self,
        mock_all_reduce,
    ):
        """TP reduction should consume the final DP output over original Replicate axes."""
        hsdp_param = _new_param()
        orig_mesh = MagicMock()
        orig_mesh.mesh_dim_names = ("tp", "sp", "ep")
        replicate_submesh = MagicMock()
        replicate_mesh = replicate_submesh.flatten.return_value
        replicate_mesh.size.return_value = 4
        replicate_mesh.get_group.return_value = "tp-replicate-group"
        orig_mesh.__getitem__.return_value = replicate_submesh
        hsdp_param.tp_grad_info = TPShardMetaInfo(
            orig_mesh,
            (Shard(0), Replicate(), Replicate()),
            origin_is_dtensor=True,
        )
        reduced_grad = torch.tensor([1.0, 2.0])

        hsdp_param.all_reduce_tp_replicate_grad_inplace(
            reduced_grad,
            torch.distributed.ReduceOp.SUM,
        )

        orig_mesh.__getitem__.assert_called_once_with(("sp", "ep"))
        replicate_submesh.flatten.assert_called_once_with()
        mock_all_reduce.assert_called_once_with(
            reduced_grad,
            op=torch.distributed.ReduceOp.SUM,
            group="tp-replicate-group",
            async_op=False,
        )

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.all_reduce")
    def test_all_reduce_tp_replicate_grad_inplace_uses_plain_parameter_metadata(self, mock_all_reduce):
        """Dual-mode gradients should all-reduce over replicated source mesh axes."""
        hsdp_param = _new_param()
        source_mesh = MagicMock()
        source_mesh.mesh_dim_names = ("tp", "ep")
        replicate_submesh = MagicMock()
        replicate_mesh = replicate_submesh.flatten.return_value
        replicate_mesh.size.return_value = 2
        replicate_mesh.get_group.return_value = "replicate-group"
        source_mesh.__getitem__.return_value = replicate_submesh
        hsdp_param.tp_grad_info = TPShardMetaInfo(
            source_mesh,
            (Shard(0), Replicate()),
            origin_is_dtensor=False,
        )
        reduced_grad = torch.tensor([1.0, 2.0])

        hsdp_param.all_reduce_tp_replicate_grad_inplace(
            reduced_grad,
            torch.distributed.ReduceOp.SUM,
        )

        source_mesh.__getitem__.assert_called_once_with(("ep",))
        mock_all_reduce.assert_called_once_with(
            reduced_grad,
            op=torch.distributed.ReduceOp.SUM,
            group="replicate-group",
            async_op=False,
        )

    def test_reset_sharded_param_meta_and_shape_mismatch(self):
        """Reset should materialize meta params and reject mismatched shard shapes."""
        hsdp_param = _new_param()
        meta_param = torch.nn.Parameter(torch.empty(2, device="meta"))
        hsdp_param._module_info = SimpleNamespace(module=SimpleNamespace(weight=meta_param), param_name="weight")
        hsdp_param.sharded_param = meta_param

        hsdp_param.reset_sharded_param()

        bad_param = torch.nn.Parameter(torch.ones(3))
        hsdp_param._module_info.module.weight = bad_param
        hsdp_param.sharded_param = SimpleNamespace(_local_tensor=torch.ones(2), _hsdp_param_initialized=True)
        hsdp_param._sharded_param_data = torch.ones(2)
        with self.assertRaisesRegex(AssertionError, "Expected sharded_size"):
            hsdp_param.reset_sharded_param()


class TestParameterHookMigrator(unittest.TestCase):
    """Verify parameter backward hooks retain their existing migration semantics."""

    def test_save_deduplicates_hooks_and_migrate_runs_once(self):
        """Repeated saves should deduplicate hooks and each target should migrate once."""
        hook_a = MagicMock()
        hook_b = MagicMock()
        source_param = torch.nn.Parameter(torch.ones(1))
        source_param._backward_hooks = {0: hook_a, 1: hook_a, 2: hook_b}
        migrator = ParameterHookMigrator()

        migrator._save_backward_hooks(source_param)
        migrator._save_backward_hooks(source_param)

        self.assertEqual(migrator._orig_param_hooks, [hook_a, hook_b])

        target_param = torch.nn.Parameter(torch.ones(1))
        migrator._migrate_backward_hooks(target_param)
        migrator._migrate_backward_hooks(target_param)

        self.assertEqual(list(target_param._backward_hooks.values()), [hook_a, hook_b])
        self.assertTrue(target_param.migrate_backward_hooks_run_once)

    def test_migrate_continues_after_registration_error_and_marks_frozen_target(self):
        """Registration errors should not stop later hooks, and frozen targets should be marked."""
        hook_a = MagicMock()
        hook_b = MagicMock()
        migrator = ParameterHookMigrator()
        source_param = torch.nn.Parameter(torch.ones(1))
        source_param._backward_hooks = {0: hook_a, 1: hook_b}
        migrator._save_backward_hooks(source_param)

        class _TargetParam:
            """Minimal parameter double for hook migration."""

            def __init__(self, requires_grad: bool) -> None:
                self.requires_grad = requires_grad
                self.register_hook = MagicMock()

        target_param = _TargetParam(requires_grad=True)
        target_param.register_hook.side_effect = [RuntimeError("cannot register"), None]

        migrator._migrate_backward_hooks(target_param)

        self.assertEqual(target_param.register_hook.call_count, 2)
        self.assertTrue(target_param.migrate_backward_hooks_run_once)

        frozen_param = _TargetParam(requires_grad=False)
        migrator._migrate_backward_hooks(frozen_param)

        frozen_param.register_hook.assert_not_called()
        self.assertTrue(frozen_param.migrate_backward_hooks_run_once)


if __name__ == "__main__":
    unittest.main()

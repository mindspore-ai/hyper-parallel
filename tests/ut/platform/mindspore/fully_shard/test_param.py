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
"""Unit tests for MindSpore fully_shard parameter lifecycle and communication."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mindspore")
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard.param import (
    AllGatherCommCtx,
    AllReduceCommCtx,
    MindSporeHSDPParamV2,
    ParameterHookMigrator,
    ReduceScatterCommCtx,
    make_contiguous_strides_for,
    set_requires_grad_if_needed,
)
from tests.ut.platform.mindspore.fully_shard.conftest import MindSporeFullyShardUnitTest


def _bare_param():
    """Build a parameter wrapper with constructor-owned contexts initialized."""
    hsdp_param = object.__new__(MindSporeHSDPParamV2)
    hsdp_param.unsharded_param_buffers = []
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param._unsharded_param = None
    hsdp_param.allgather_comm_ctx = AllGatherCommCtx()
    hsdp_param.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
    hsdp_param.all_reduce_comm_ctx = AllReduceCommCtx()
    hsdp_param._grad = None
    hsdp_param._reduce_partial_output = None
    hsdp_param.gradient_scaling_factor = None
    hsdp_param.mp_policy = MixedPrecisionPolicy()
    hsdp_param.orig_dtype = ms.float32
    hsdp_param.param_dtype = None
    hsdp_param.reduce_dtype = None
    return hsdp_param


class TestPlacementConstruction(MindSporeFullyShardUnitTest):
    """Test source-layout preservation and explicit DP placement application."""

    def test_base_placements_prefix_dp_axes_for_source_layout(self):
        """Native DTensor source placements should follow the explicit DP prefix."""
        hsdp_param = _bare_param()
        dp_mesh = MagicMock(ndim=1)
        source_mesh = MagicMock()
        hsdp_param.mesh_info = SimpleNamespace(mesh=dp_mesh)
        hsdp_param.source_shard_info = SimpleNamespace(
            mesh=source_mesh,
            placements=(Shard(1), Replicate()),
        )
        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.param.DeviceMesh.concatenate",
            return_value=MagicMock(ndim=3),
        ) as concatenate:
            placements = hsdp_param._get_base_spmd_placements()

        concatenate.assert_called_once_with([dp_mesh, source_mesh])
        self.assertTrue(placements[0].is_replicate())
        self.assertEqual(placements[1:], (Shard(1), Replicate()))

    def test_apply_data_parallel_placement_builds_strided_shard(self):
        """Same-dimension TP and FSDP sharding should use a StridedShard."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_replicate_mesh_dim = None
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=2, mesh_shape=(2, 4))
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        placements = hsdp_param._apply_data_parallel_placements(
            [Replicate(), Shard(1)],
            Shard(1),
        )

        self.assertEqual(placements[0], StridedShard(1, split_factor=4))

    def test_replicate_param_keeps_replicate_placement(self):
        """A DDP-managed plain parameter should not gain an FSDP shard placement."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._spmd_shard_mesh_dim = None
        hsdp_param._spmd_replicate_mesh_dim = 0
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=1)
        hsdp_param.mesh_info = object.__new__(DDPMeshInfo)

        placements = hsdp_param._apply_data_parallel_placements([Shard(0)], Shard(0))

        self.assertTrue(placements[0].is_replicate())

    def test_uneven_hsdp_retains_replicate_and_marks_shard(self):
        """
        Feature: Uneven HSDP placement construction.
        Description: Apply dim-0 FSDP sharding after a replicate mesh dimension.
        Expectation: Replication is preserved and the FSDP placement is marked uneven.
        """
        hsdp_param = _bare_param()
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._spmd_shard_mesh_dim = 1
        hsdp_param._spmd_replicate_mesh_dim = 0
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=2, mesh_shape=(2, 2))
        hsdp_param.shard_world_size = 2
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        hsdp_param._init_shard_placements(
            ms.Tensor(np.ones((5, 3), dtype=np.float32)),
            0,
            [Replicate(), Replicate()],
        )

        self.assertEqual(
            hsdp_param._spmd_placements,
            (Replicate(), Shard(0, uneven_shard=True)),
        )

    def test_uneven_same_dim_source_marks_strided_shard(self):
        """
        Feature: Uneven same-dimension TP and FSDP placement construction.
        Description: Apply uneven FSDP after an existing shard on the same tensor dimension.
        Expectation: The FSDP axis becomes an uneven StridedShard with the source order retained.
        """
        hsdp_param = _bare_param()
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_replicate_mesh_dim = None
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=2, mesh_shape=(2, 2))
        hsdp_param.shard_world_size = 2
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        hsdp_param._init_shard_placements(
            ms.Tensor(np.ones((5, 3), dtype=np.float32)),
            0,
            [Replicate(), Shard(0)],
        )

        self.assertEqual(
            hsdp_param._spmd_placements,
            (StridedShard(0, split_factor=2, uneven_shard=True), Shard(0)),
        )


class TestParameterHelpers(MindSporeFullyShardUnitTest):
    """Test local parameter buffers, shapes, and lifecycle state helpers."""

    def test_dim0_uneven_init_and_reset_keep_logical_shard_separate_from_padding(self):
        """
        Feature: Uneven parameter storage lifecycle.
        Description: Initialize and refresh the short rank of a five-row parameter.
        Expectation: The optimizer shard stays independent from the zero-padded communication buffer.
        """
        hsdp_param = _bare_param()
        hsdp_param.shard_world_size = 2
        hsdp_param.shard_rank = 1
        hsdp_param.offload_to_cpu = False
        hsdp_param.pin_memory = False
        full_param = ms.Tensor(np.arange(15, dtype=np.float32).reshape(5, 3))

        local_param = hsdp_param._build_sharded_param_data(
            full_param,
            shard_dim=0,
            dim_shard_size=3,
        )

        self.assertEqual(hsdp_param.sharded_size, (2, 3))
        self.assertEqual(hsdp_param.padded_sharded_param_size, (3, 3))
        self.assertEqual(local_param.shape, (2, 3))
        np.testing.assert_allclose(
            hsdp_param._sharded_param_data.asnumpy(),
            np.array([9, 10, 11, 12, 13, 14, 0, 0, 0], dtype=np.float32),
        )
        self.assertNotEqual(
            local_param.untyped_storage().data_ptr(),
            hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
        )

        loaded_local_tensor = ms.Tensor(np.full((2, 3), 9.0, dtype=np.float32))
        hsdp_param.sharded_param = MagicMock(
            _local_tensor=loaded_local_tensor,
            requires_grad=True,
        )
        with patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed"):
            hsdp_param._refresh_sharded_local_tensor(loaded_local_tensor)

        np.testing.assert_allclose(
            hsdp_param.sharded_param._local_tensor.asnumpy(),
            loaded_local_tensor.asnumpy(),
        )
        np.testing.assert_allclose(
            hsdp_param._sharded_param_data.asnumpy(),
            np.array([9, 9, 9, 9, 9, 9, 0, 0, 0], dtype=np.float32),
        )

    def test_dim0_smaller_than_world_size_preserves_empty_actual_shape(self):
        """
        Feature: Empty ceil-chunk parameter shards.
        Description: Shard two rows across four ranks and inspect the final rank.
        Expectation: Its logical tensor is empty while its communication buffer contains one zero row.
        """
        hsdp_param = _bare_param()
        hsdp_param.shard_world_size = 4
        hsdp_param.shard_rank = 3
        hsdp_param.offload_to_cpu = False
        hsdp_param.pin_memory = False

        local_param = hsdp_param._build_sharded_param_data(
            ms.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3)),
            shard_dim=0,
            dim_shard_size=1,
        )

        self.assertEqual(hsdp_param.sharded_size, (0, 3))
        self.assertEqual(hsdp_param.padded_sharded_param_size, (1, 3))
        self.assertEqual(local_param.shape, (0, 3))
        np.testing.assert_allclose(
            hsdp_param._sharded_param_data.asnumpy(),
            np.zeros(3, dtype=np.float32),
        )

    def test_uneven_non_dim0_sharding_is_rejected(self):
        """
        Feature: Uneven parameter shard validation.
        Description: Request an uneven FSDP parameter split on dimension one.
        Expectation: MindSpore rejects the same unsupported boundary as Torch.
        """
        hsdp_param = _bare_param()
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param.shard_world_size = 2
        hsdp_param._module_info = SimpleNamespace(param_name="weight")
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_replicate_mesh_dim = None
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=1, mesh_shape=(2,))
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        with self.assertRaisesRegex(NotImplementedError, "only supports uneven sharding on dim=0"):
            hsdp_param._init_shard_placements(
                ms.Tensor(np.arange(15, dtype=np.float32).reshape(3, 5)),
                1,
                [Replicate()],
            )

    def test_make_contiguous_strides_for(self):
        """Row-major and column-major stride helpers should match tensor shapes."""
        self.assertEqual(make_contiguous_strides_for((2, 3, 4)), (12, 4, 1))
        self.assertEqual(make_contiguous_strides_for((2, 3, 4), row_major=False), (12, 1, 3))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_contiguous_strides_for((2, -1))

    def test_init_unsharded_buffers_reuses_and_force_recreates(self):
        """Stable buffers should be reused unless explicit recreation is requested."""
        hsdp_param = _bare_param()
        hsdp_param.init_unsharded_param_buffers([2], [ms.float32], 2, "cpu")
        original = hsdp_param.unsharded_param_buffers[0]
        hsdp_param.init_unsharded_param_buffers([3], [ms.float16], 2, "cpu")
        self.assertIs(hsdp_param.unsharded_param_buffers[0], original)
        hsdp_param.init_unsharded_param_buffers(
            [3], [ms.float16], 2, "cpu", force_recreate=True
        )
        self.assertIsNot(hsdp_param.unsharded_param_buffers[0], original)
        self.assertEqual(hsdp_param.unsharded_param_buffers[0].dtype, ms.float16)

    def test_init_unsharded_param_restores_non_dim_zero_layout(self):
        """Per-parameter all-gather should inline chunk-cat reconstruction for dimension one."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._orig_size = (2, 4)
        hsdp_param.sharded_size = (2, 2)
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=False)
        hsdp_param.unsharded_param_buffers = [ms.mint.empty((8,), dtype=ms.float32)]
        hsdp_param.allgather_comm_ctx.allgather_output = ms.Tensor(
            [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0],
            ms.float32,
        )

        hsdp_param.init_unsharded_param()

        np.testing.assert_allclose(
            hsdp_param.unsharded_param.asnumpy(),
            np.arange(8, dtype=np.float32).reshape(2, 4),
        )
        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_output)

    def test_init_unsharded_param_hides_dim_zero_padding(self):
        """An uneven dim-0 parameter should expose only logical all-gather elements."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._orig_size = (5,)
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=False)
        hsdp_param.unsharded_param_buffers = [
            ms.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0], ms.float32)
        ]

        hsdp_param.init_unsharded_param()

        self.assertEqual(hsdp_param.unsharded_param.shape, (5,))
        np.testing.assert_allclose(
            hsdp_param.unsharded_param.asnumpy(),
            np.arange(5, dtype=np.float32),
        )

    def test_to_sharded_only_frees_distinct_unsharded_storage(self):
        """Replicate parameters must retain storage shared by sharded and unsharded views."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_param = object()
        hsdp_param._setattr_on_modules = MagicMock()
        hsdp_param.free_unsharded_param = MagicMock()
        shared_storage = object()
        hsdp_param._sharded_param_data = shared_storage
        hsdp_param.unsharded_param_buffers = [shared_storage]

        hsdp_param.to_sharded()

        hsdp_param.free_unsharded_param.assert_not_called()
        self.assertEqual(hsdp_param.sharded_state, ShardedState.SHARDED)

        hsdp_param.unsharded_param_buffers = [object()]
        hsdp_param.to_sharded()

        hsdp_param.free_unsharded_param.assert_called_once_with()

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed")
    def test_wait_for_unshard_preserves_parameter_identity(self, mock_requires_grad):
        """Waiting a prefetched all-gather should install one stable full parameter."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_state = ShardedState.SHARDED
        hsdp_param.sharded_param = MagicMock()
        hsdp_param.init_unsharded_param = MagicMock()
        hsdp_param.to_unsharded = MagicMock()
        handle = MagicMock()
        hsdp_param.allgather_comm_ctx.allgather_handle = handle

        hsdp_param.wait_for_unshard()

        handle.wait.assert_called_once_with()
        hsdp_param.init_unsharded_param.assert_called_once_with()
        hsdp_param.to_unsharded.assert_called_once_with()
        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_handle)
        mock_requires_grad.assert_not_called()

    def test_set_requires_grad_if_needed_only_updates_changes(self):
        """Requires-grad propagation should avoid redundant writes."""
        source = SimpleNamespace(requires_grad=True)
        destination = MagicMock(requires_grad=False)
        set_requires_grad_if_needed(source, destination)
        destination.requires_grad_.assert_called_once_with(True)
        destination.requires_grad = True
        destination.requires_grad_.reset_mock()
        set_requires_grad_if_needed(source, destination)
        destination.requires_grad_.assert_not_called()


class TestCommunicationContexts(MindSporeFullyShardUnitTest):
    """Test async handle waiting and mint collective routing."""

    def test_reduce_scatter_and_all_reduce_outputs_wait_once(self):
        """Each communication output accessor should consume its async handle."""
        hsdp_param = _bare_param()
        rs_handle = MagicMock()
        ar_handle = MagicMock()
        rs_output = ms.Tensor([1.0, 2.0])
        ar_output = object()
        hsdp_param._grad = ms.Tensor([3.0, 4.0])
        hsdp_param.reduce_scatter_comm_ctx = ReduceScatterCommCtx(rs_output, rs_handle)
        hsdp_param.all_reduce_comm_ctx = AllReduceCommCtx(ar_output, ar_handle)

        self.assertIs(hsdp_param.reduce_scatter_output(), rs_output)
        self.assertIs(hsdp_param.all_reduce_output(), ar_output)
        rs_handle.wait.assert_called_once_with()
        ar_handle.wait.assert_called_once_with()
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_uses_mint_string_and_context(self, mock_reduce_scatter):
        """Per-parameter RS should use parameter mesh info and cache async work."""
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0])
        )
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.is_sharded = True
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._orig_size = (4,)
        hsdp_param.padded_sharded_param_size = (2,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        handle = MagicMock()
        mock_reduce_scatter.return_value = handle

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        self.assertEqual(mock_reduce_scatter.call_args.kwargs["op"], "sum")
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["group"], "fsdp")
        self.assertIs(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle, handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_scales_gradient_view_without_materializing(
        self,
        mock_reduce_scatter,
        mock_apply_scaling,
    ):
        """An even dim-0 gradient view should retain the source storage during inplace scaling."""
        source_grad = ms.Tensor([1.0, 2.0, 3.0, 4.0])
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(grad=source_grad)
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param.padded_sharded_param_size = (2,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        hsdp_param.gradient_scaling_factor = 0.5
        mock_reduce_scatter.return_value = MagicMock()

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        scaled_grad = mock_apply_scaling.call_args.args[0]
        reduce_scatter_input = mock_reduce_scatter.call_args.args[1]
        self.assertIs(reduce_scatter_input, scaled_grad)
        self.assertEqual(
            scaled_grad.untyped_storage().data_ptr(),
            source_grad.untyped_storage().data_ptr(),
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_zero_pads_uneven_dim0_gradient(
        self,
        mock_reduce_scatter,
        mock_apply_scaling,
    ):
        """
        Feature: Per-parameter uneven reduce-scatter.
        Description: Prepare a five-element gradient for scaling and reduction over two FSDP ranks.
        Expectation: The inplace helper receives the six-element padded communication base tensor.
        """
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.is_sharded = True
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0, uneven_shard=True)
        hsdp_param._orig_size = (5,)
        hsdp_param.padded_sharded_param_size = (3,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        hsdp_param.gradient_scaling_factor = 0.5
        mock_reduce_scatter.return_value = MagicMock()
        mock_apply_scaling.return_value = ms.Tensor([-1.0], ms.float32)

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        output, packed_grad = mock_reduce_scatter.call_args.args
        self.assertEqual(output.shape, (3,))
        mock_apply_scaling.assert_called_once_with(packed_grad, 0.5)
        self.assertIsNot(packed_grad, mock_apply_scaling.return_value)
        np.testing.assert_allclose(
            packed_grad.asnumpy(),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0], dtype=np.float32),
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_consumes_rs_output_and_uses_replicate_group(self, mock_all_reduce):
        """
        Feature: HSDP all-reduce input layout.
        Description: Pass a contiguous reduce-scatter tensor into the replicate-group collective.
        Expectation: The existing tensor is reduced with the requested group and operation.
        """
        hsdp_param = _bare_param()
        output = ms.Tensor([1.0, 2.0])
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = output
        hsdp_param.replicate_world_size = 2
        hsdp_param.mesh_info = object.__new__(DDPMeshInfo)
        hsdp_param.mesh_info.replicate_process_group = "dp"
        handle = MagicMock()
        mock_all_reduce.return_value = handle

        hsdp_param.all_reduce_grad(reduce_op="avg")

        self.assertIs(hsdp_param.all_reduce_comm_ctx.all_reduce_output, output)
        self.assertIs(hsdp_param.all_reduce_comm_ctx.all_reduce_handle, handle)
        self.assertEqual(mock_all_reduce.call_args.kwargs["group"], "dp")
        self.assertEqual(mock_all_reduce.call_args.kwargs["op"], "avg")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_tp_replicate_reduce_uses_flattened_source_mesh(self, mock_all_reduce):
        """Final gradients should all-reduce across replicated source-layout axes."""
        hsdp_param = _bare_param()
        replicate_mesh = MagicMock()
        replicate_mesh.size.return_value = 2
        replicate_mesh.get_group.return_value = "tp-replicate"
        source_mesh = MagicMock(mesh_dim_names=("tp", "cp"))
        source_mesh.__getitem__.return_value.flatten.return_value = replicate_mesh
        hsdp_param.source_shard_info = SimpleNamespace(
            mesh=source_mesh,
            placements=(Shard(0), Replicate()),
        )
        grad = ms.Tensor([1.0, 2.0])

        hsdp_param.all_reduce_source_replicate_grad_inplace(grad, "sum")

        source_mesh.__getitem__.assert_called_once_with(("cp",))
        mock_all_reduce.assert_called_once_with(
            grad,
            op="sum",
            group="tp-replicate",
            async_op=False,
        )


class TestGradientApplication(MindSporeFullyShardUnitTest):
    """Test reduced-gradient assignment and source-gradient cleanup."""

    def test_apply_reduced_grad_assigns_and_clears_source(self):
        """A reduced local shard should replace the optimizer grad and release full grad."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_size = (2,)
        hsdp_param.sharded_param = SimpleNamespace(
            grad=None,
            _local_tensor=ms.Tensor([0.0, 0.0]),
        )
        hsdp_param._unsharded_param = SimpleNamespace(grad=ms.Tensor([3.0, 4.0]))
        hsdp_param.offload_to_cpu = False
        hsdp_param.pin_memory = False
        hsdp_param._sharded_param_storage_dtype = MagicMock(return_value=ms.float32)
        hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: tensor)

        need_synchronize = hsdp_param.apply_reduced_grad(ms.Tensor([1.0, 2.0], ms.float16))

        self.assertFalse(need_synchronize)
        np.testing.assert_allclose(
            hsdp_param.sharded_param.grad.asnumpy(),
            np.array([1.0, 2.0], dtype=np.float32),
        )
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_clear_output_helpers_release_context_tensors(self):
        """Explicit clear helpers should drop completed communication outputs."""
        hsdp_param = _bare_param()
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = object()
        hsdp_param.all_reduce_comm_ctx.all_reduce_output = object()
        hsdp_param.clear_reduce_scatter_output()
        hsdp_param.clear_all_reduce_output()
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_output)


class TestParameterHookMigrator(MindSporeFullyShardUnitTest):
    """Verify parameter backward hooks retain their existing migration semantics."""

    def test_save_deduplicates_hooks_and_migrate_runs_once(self):
        """Repeated saves should deduplicate hooks and each target should migrate once."""
        hook_a = MagicMock()
        hook_b = MagicMock()
        source_param = MagicMock()
        source_param.hooks.return_value = [hook_a, hook_a, hook_b]
        migrator = ParameterHookMigrator()

        migrator._save_backward_hooks(source_param)
        migrator._save_backward_hooks(source_param)

        self.assertEqual(migrator._orig_param_hooks, [hook_a, hook_b])

        class _TargetParam:
            """Minimal parameter double for hook registration."""

            requires_grad = True

            def __init__(self) -> None:
                """Initialize the mocked hook registration method."""
                self.register_hook = MagicMock()

        target_param = _TargetParam()
        migrator._migrate_backward_hooks(target_param)
        migrator._migrate_backward_hooks(target_param)

        registered_hooks = [hook_call.args[0] for hook_call in target_param.register_hook.call_args_list]
        self.assertEqual(registered_hooks, [hook_a, hook_b])
        self.assertTrue(vars(target_param)["migrate_backward_hooks_run_once"])

    def test_migrate_continues_after_registration_error_and_marks_frozen_target(self):
        """Registration errors should not stop later hooks, and frozen targets should be marked."""
        hook_a = MagicMock()
        hook_b = MagicMock()
        source_param = MagicMock()
        source_param.hooks.return_value = [hook_a, hook_b]
        migrator = ParameterHookMigrator()
        migrator._save_backward_hooks(source_param)

        class _TargetParam:
            """Minimal parameter double for hook migration."""

            def __init__(self, requires_grad: bool) -> None:
                """Initialize the gradient flag and mocked registration method."""
                self.requires_grad = requires_grad
                self.register_hook = MagicMock()

        target_param = _TargetParam(requires_grad=True)
        target_param.register_hook.side_effect = [RuntimeError("cannot register"), None]
        migrator._migrate_backward_hooks(target_param)

        self.assertEqual(target_param.register_hook.call_count, 2)
        self.assertTrue(vars(target_param)["migrate_backward_hooks_run_once"])

        frozen_param = _TargetParam(requires_grad=False)
        migrator._migrate_backward_hooks(frozen_param)

        frozen_param.register_hook.assert_not_called()
        self.assertTrue(vars(frozen_param)["migrate_backward_hooks_run_once"])


if __name__ == "__main__":
    unittest.main()

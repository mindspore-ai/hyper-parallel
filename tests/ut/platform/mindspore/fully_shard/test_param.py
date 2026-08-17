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
from unittest.mock import MagicMock, PropertyMock, patch

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
        hsdp_param.tp_grad_info = SimpleNamespace(
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


class TestParameterHelpers(MindSporeFullyShardUnitTest):
    """Test local parameter buffers, shapes, and lifecycle state helpers."""

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

    def test_to_sharded_clears_all_gather_context(self):
        """Reshard must make the next unshard launch a fresh all-gather."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_param = object()
        hsdp_param._setattr_on_modules = MagicMock()
        hsdp_param.free_unsharded_param = MagicMock()
        hsdp_param.allgather_comm_ctx.allgather_output = MagicMock()
        hsdp_param.allgather_comm_ctx.allgather_handle = MagicMock()

        hsdp_param.to_sharded()

        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_output)
        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_handle)
        self.assertEqual(hsdp_param.sharded_state, ShardedState.SHARDED)

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
        rs_output = object()
        ar_output = object()
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
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        handle = MagicMock()
        mock_reduce_scatter.return_value = handle

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        self.assertEqual(mock_reduce_scatter.call_args.kwargs["op"], "sum")
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["group"], "fsdp")
        self.assertIs(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle, handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_consumes_rs_output_and_uses_replicate_group(self, mock_all_reduce):
        """HSDP AR should reduce the current RS output on the replicate mesh."""
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
        hsdp_param.tp_grad_info = SimpleNamespace(
            mesh=source_mesh,
            placements=(Shard(0), Replicate()),
        )
        grad = ms.Tensor([1.0, 2.0])

        hsdp_param.all_reduce_tp_replicate_grad_inplace(grad, "sum")

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


if __name__ == "__main__":
    unittest.main()

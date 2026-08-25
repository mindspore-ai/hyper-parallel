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
"""Unit tests for MindSpore fully_shard fused communication buckets."""
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

from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, HSDPMeshInfo
from hyper_parallel.platform.mindspore.fully_shard.param_group import (
    AllGatherResult,
    AllReduceParamGroup,
    HSDPParamGroup,
    all_gather_copy_in,
    get_all_gather_metadata,
    reduce_scatter_copy_in,
)
from hyper_parallel.platform.mindspore.fully_shard import param_group as param_group_module
from tests.ut.platform.mindspore.fully_shard.conftest import MindSporeFullyShardUnitTest


def _mesh_info(mesh_cls=FSDPMeshInfo, *, shard_group=None, replicate_group=None):
    """Create mesh metadata without distributed initialization."""
    mesh_info = object.__new__(mesh_cls)
    mesh_info.shard_mesh_rank = 0
    mesh_info.shard_mesh_size = 2 if shard_group is not None else 1
    mesh_info.shard_process_group = shard_group
    mesh_info.replicate_mesh_rank = 0
    mesh_info.replicate_mesh_size = 2 if replicate_group is not None else 1
    mesh_info.replicate_process_group = replicate_group
    return mesh_info


@pytest.fixture(autouse=True)
def _patch_view_copy_for_cpu(monkeypatch):
    """Keep CPU UT focused on copy geometry; NPU ST validates ACLNN view-copy values."""
    monkeypatch.setattr(
        param_group_module,
        "copy_without_bumping_version",
        MagicMock(),
    )


def _fake_param(
    values,
    *,
    mesh_info=None,
    grad=None,
    param_dtype=None,
    reduce_dtype=None,
    shard_dim=0,
):
    """Create the parameter facts consumed by ``HSDPParamGroup``."""
    local_tensor = ms.Tensor(values, ms.float32)
    hsdp_param = MagicMock()
    hsdp_param.mesh_info = mesh_info or _mesh_info()
    hsdp_param.shard_world_size = (
        hsdp_param.mesh_info.shard_mesh_size
        if isinstance(hsdp_param.mesh_info, FSDPMeshInfo)
        else 1
    )
    hsdp_param.shard_rank = 0
    hsdp_param.replicate_world_size = (
        hsdp_param.mesh_info.replicate_mesh_size
        if isinstance(hsdp_param.mesh_info, DDPMeshInfo)
        else 1
    )
    hsdp_param.orig_dtype = local_tensor.dtype
    hsdp_param.param_dtype = param_dtype
    hsdp_param.reduce_dtype = reduce_dtype
    hsdp_param.sharded_size = local_tensor.shape
    hsdp_param.padded_sharded_param_size = local_tensor.shape
    hsdp_param.hsdp_placement = Shard(shard_dim)
    hsdp_param._orig_size = (local_tensor.shape[0] * hsdp_param.shard_world_size,)
    hsdp_param.sharded_param = SimpleNamespace(requires_grad=True)
    hsdp_param._sharded_param_data = local_tensor
    hsdp_param.all_gather_inputs = [
        local_tensor.to(param_dtype) if param_dtype is not None else local_tensor
    ]
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.unsharded_accumulated_grad_data = None
    hsdp_param.unsharded_param = SimpleNamespace(grad=grad)
    hsdp_param.unsharded_grad_data = grad
    hsdp_param.reduce_comm_dtype.side_effect = lambda current_grad=None: (
        reduce_dtype or (current_grad.dtype if current_grad is not None else local_tensor.dtype)
    )
    hsdp_param.reduce_partial_output = None
    hsdp_param.reduce_scatter_comm_ctx = SimpleNamespace(
        reduce_scatter_output=None,
        reduce_scatter_handle=None,
    )
    hsdp_param.all_reduce_comm_ctx = SimpleNamespace(
        all_reduce_output=None,
        all_reduce_handle=None,
    )
    return hsdp_param


class TestParamGroupHelpers(MindSporeFullyShardUnitTest):
    """Test pure fused-buffer metadata and packing helpers."""

    def test_get_all_gather_metadata_and_copy_in(self):
        """Metadata and copy-in should preserve parameter order and local values."""
        param_a = _fake_param([1.0, 2.0])
        param_b = _fake_param([3.0])
        metadata = get_all_gather_metadata([param_a, param_b])

        self.assertEqual(metadata.inp_split_sizes, [2, 1])
        self.assertEqual(metadata.total_input_numel, 3)
        output = ms.mint.zeros((6,), dtype=ms.float32)
        local_input, returned_output = all_gather_copy_in(
            [*param_a.all_gather_inputs, *param_b.all_gather_inputs],
            output,
            metadata.inp_split_sizes,
            metadata.total_input_numel,
            rank=1,
        )
        self.assertIs(returned_output, output)
        self.assertEqual(local_input.storage_offset(), 3)
        copy_calls = param_group_module.copy_without_bumping_version.call_args_list
        self.assertEqual([call.args[0].shape for call in copy_calls], [(2,), (1,)])
        np.testing.assert_allclose(copy_calls[0].args[1].asnumpy(), np.array([1.0, 2.0]))
        np.testing.assert_allclose(copy_calls[1].args[1].asnumpy(), np.array([3.0]))

    def test_get_all_gather_metadata_rejects_mixed_dtype(self):
        """One all-gather bucket must have a uniform communication dtype."""
        param_a = _fake_param([1.0])
        param_b = _fake_param([2.0], param_dtype=ms.float16)
        with self.assertRaisesRegex(ValueError, "same dtype"):
            get_all_gather_metadata([param_a, param_b])

    def test_reduce_scatter_copy_in_packs_dim_zero_gradients(self):
        """
        Feature: Fused even reduce-scatter packing.
        Description: Pack two dim-0 parameter gradients into one collective input.
        Expectation: Each reduce-scatter row retains parameter column order.
        """
        param_a = _fake_param([1.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg"))
        param_b = _fake_param([2.0, 3.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg"))
        grad_a = ms.Tensor([1.0, 2.0], ms.float32)
        grad_b = ms.Tensor([3.0, 4.0, 5.0, 6.0], ms.float32)
        output = ms.mint.empty((6,), dtype=ms.float32)

        reduce_scatter_copy_in([param_a, param_b], [grad_a, grad_b], output, world_size=2)

        copy_calls = param_group_module.copy_without_bumping_version.call_args_list
        self.assertEqual([call.args[0].shape for call in copy_calls], [(2, 1), (2, 2)])
        np.testing.assert_allclose(
            copy_calls[0].args[1].asnumpy(),
            np.array([[1.0], [2.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            copy_calls[1].args[1].asnumpy(),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        )

    def test_reduce_scatter_copy_in_packs_non_dim_zero_gradient(self):
        """A dimension-one gradient should use the inline chunk-cat RS layout."""
        hsdp_param = _fake_param(
            [[0.0, 1.0], [4.0, 5.0]],
            mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg"),
            shard_dim=1,
        )
        hsdp_param._orig_size = (2, 4)
        full_grad = ms.Tensor(np.arange(8, dtype=np.float32).reshape(2, 4))
        output = ms.mint.empty((8,), dtype=ms.float32)

        reduce_scatter_copy_in([hsdp_param], [full_grad], output, world_size=2)

        packed_grad = param_group_module.copy_without_bumping_version.call_args.args[1]
        np.testing.assert_allclose(
            packed_grad.asnumpy(),
            np.array([[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0]], dtype=np.float32),
        )

    def test_reduce_scatter_copy_in_zero_pads_uneven_dim_zero_gradients(self):
        """
        Feature: Fused uneven reduce-scatter packing.
        Description: Pack a five-element parameter gradient for two FSDP ranks.
        Expectation: Each row has the ceil-chunk size and the final element is zero padding.
        """
        hsdp_param = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg"),
        )
        hsdp_param._orig_size = (5,)
        hsdp_param.sharded_size = (2,)
        hsdp_param.padded_sharded_param_size = (3,)
        full_grad = ms.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], ms.float32)
        output = ms.mint.empty((6,), dtype=ms.float32)

        reduce_scatter_copy_in([hsdp_param], [full_grad], output, world_size=2)

        packed_grad = param_group_module.copy_without_bumping_version.call_args.args[1]
        np.testing.assert_allclose(
            packed_grad.asnumpy(),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], dtype=np.float32),
        )
        layout = HSDPParamGroup([hsdp_param])._build_gradient_bucket_layout_from([hsdp_param])
        self.assertEqual(layout.param_numels, [3])
        self.assertEqual(layout.total_numel, 3)


class TestAllGatherBuckets(MindSporeFullyShardUnitTest):
    """Test parameter-level routing into fused all-gather buckets."""

    def test_buckets_group_by_process_group_and_dtype(self):
        """Different routes or dtypes must use independent all-gather buckets."""
        param_a = _fake_param([1.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg-a"))
        param_b = _fake_param([2.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg-a"))
        param_b.param_dtype = ms.float16
        param_b.all_gather_inputs = [param_b.all_gather_inputs[0].to(ms.float16)]
        param_c = _fake_param([3.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg-b"))
        replicate_param = _fake_param([4.0], mesh_info=_mesh_info(DDPMeshInfo))
        group = HSDPParamGroup([param_a, param_b, param_c, replicate_param], device="cpu")

        group._init_all_gather_buckets()

        self.assertEqual(len(group.all_gather_buckets), 3)
        self.assertNotIn(replicate_param, [
            param
            for bucket in group.all_gather_buckets
            for param in bucket.hsdp_params
        ])

    def test_copy_out_restores_mixed_shard_dimensions(self):
        """Fused all-gather should inline dim-0 and non-dim-0 reconstruction."""
        mesh_info = _mesh_info(FSDPMeshInfo, shard_group="pg")
        dim0_param = _fake_param(
            [[0.0, 1.0], [2.0, 3.0]],
            mesh_info=mesh_info,
        )
        dim0_param._orig_size = (4, 2)
        dim0_param.unsharded_param_buffers = [ms.mint.empty((8,), dtype=ms.float32)]
        dim1_param = _fake_param(
            [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0], [12.0, 13.0]],
            mesh_info=mesh_info,
            shard_dim=1,
        )
        dim1_param._orig_size = (4, 4)
        dim1_param.unsharded_param_buffers = [ms.mint.empty((16,), dtype=ms.float32)]
        group = HSDPParamGroup([dim0_param, dim1_param], device="cpu")
        group._init_all_gather_buckets()
        all_gather_bucket = group.all_gather_buckets[0]
        rank0_input = ms.Tensor(
            [
                0.0, 1.0, 2.0, 3.0,
                0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0,
            ],
            ms.float32,
        )
        rank1_input = ms.Tensor(
            [
                4.0, 5.0, 6.0, 7.0,
                2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0,
            ],
            ms.float32,
        )
        all_gather_bucket.all_gather_result = AllGatherResult(
            all_gather_input=rank0_input,
            all_gather_output=ms.mint.cat((rank0_input, rank1_input), dim=0),
            handle=None,
        )

        all_gather_bucket.copy_out()

        copy_calls = param_group_module.copy_without_bumping_version.call_args_list
        np.testing.assert_allclose(
            copy_calls[0].args[1].reshape(4, 2).asnumpy(),
            np.arange(8, dtype=np.float32).reshape(4, 2),
        )
        np.testing.assert_allclose(
            copy_calls[1].args[1].reshape(4, 4).asnumpy(),
            np.arange(16, dtype=np.float32).reshape(4, 4),
        )
        self.assertIsNone(all_gather_bucket.all_gather_result)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.all_gather_copy_in", wraps=all_gather_copy_in)
    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.all_gather_into_tensor")
    def test_all_gather_uses_mint_and_copy_fallback(self, mock_all_gather, mock_copy_in):
        """MindSpore fused all-gather should pack copies and retain async work."""
        param = _fake_param([1.0, 2.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg"))
        param.reset_sharded_param = MagicMock()
        group = HSDPParamGroup([param], device="cpu", enable_zero_copy=False)
        handle = MagicMock()
        mock_all_gather.return_value = handle

        group.foreach_all_gather(async_op=True)

        mock_all_gather.assert_called_once()
        mock_copy_in.assert_called_once()
        bucket = group.all_gather_buckets[0]
        self.assertIs(bucket.all_gather_result.handle, handle)
        self.assertFalse(group.enable_zero_copy)
        self.assertFalse(hasattr(group, "_flat_param_buffer"))
        param.reset_sharded_param.assert_called_once_with()


class TestReduceBuckets(MindSporeFullyShardUnitTest):
    """Test fused RS/AR routing, accumulation, and output ownership."""

    def test_reduce_scatter_buckets_group_by_route_and_dtype(self):
        """RS buckets should be homogeneous in shard route and reduce dtype."""
        grad = ms.Tensor([1.0, 2.0], ms.float32)
        param_a = _fake_param(
            [1.0], mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg-a"), grad=grad
        )
        param_b = _fake_param(
            [2.0],
            mesh_info=_mesh_info(FSDPMeshInfo, shard_group="pg-a"),
            grad=grad.to(ms.float16),
            reduce_dtype=ms.float16,
        )
        replicate_param = _fake_param(
            [3.0], mesh_info=_mesh_info(DDPMeshInfo, replicate_group="dp"), grad=grad
        )
        group = HSDPParamGroup([param_a, param_b, replicate_param], device="cpu")

        buckets = group._build_reduce_scatter_buckets("avg")

        self.assertEqual(len(buckets), 3)
        self.assertTrue(all(bucket.reduce_op == "sum" for bucket in buckets))
        self.assertTrue(all(bucket.needs_avg_div for bucket in buckets))

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_foreach_reducescatter_launches_mint_sum(self, mock_reduce_scatter, mock_apply_scaling):
        """
        Feature: Fused reduce-scatter scaling and reduction.
        Description: Scale one bucket and launch AVG as a mint SUM collective.
        Expectation: The same base tensor reaches the inplace helper and SUM collective.
        """
        mesh_info = _mesh_info(FSDPMeshInfo, shard_group="pg")
        param = _fake_param(
            [1.0, 2.0],
            mesh_info=mesh_info,
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0], ms.float32),
        )
        group = HSDPParamGroup([param], device="cpu")
        group.gradient_scaling_factor = 0.5
        handle = MagicMock()
        mock_reduce_scatter.return_value = handle
        mock_apply_scaling.return_value = ms.Tensor([-1.0], ms.float32)

        group.foreach_reducescatter("avg", async_op=True)

        self.assertIs(group.comm_ctx.pre_param_group, group)
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["op"], "sum")
        reduce_scatter_input = mock_reduce_scatter.call_args.args[1]
        mock_apply_scaling.assert_called_once_with(reduce_scatter_input, 0.5)
        self.assertEqual(reduce_scatter_input.shape, (4,))
        self.assertIsNot(reduce_scatter_input, mock_apply_scaling.return_value)
        param.clear_unsharded_source_grad.assert_called_once_with()

    def test_replicate_param_uses_local_rs_then_all_reduce_bucket(self):
        """Replicate parameters should share the lifecycle using a local RS stage."""
        mesh_info = _mesh_info(DDPMeshInfo, replicate_group="dp")
        param = _fake_param([1.0, 2.0], mesh_info=mesh_info, grad=ms.Tensor([3.0, 4.0]))
        group = HSDPParamGroup([param], device="cpu")
        rs_buckets = group._build_reduce_scatter_buckets("sum")

        ar_buckets = group._build_all_reduce_buckets(rs_buckets)

        self.assertEqual(rs_buckets[0].shard_world_size, 1)
        self.assertIsNone(rs_buckets[0].shard_group)
        self.assertEqual(len(ar_buckets), 1)
        self.assertEqual(ar_buckets[0].replicate_group, "dp")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_wait_rs_issues_all_reduce_and_saves_context(
        self, mock_reduce_scatter, mock_all_reduce
    ):
        """Completed RS output should move into AR without per-parameter repacking."""
        param = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(HSDPMeshInfo, shard_group="fsdp", replicate_group="dp"),
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0]),
        )
        group = HSDPParamGroup([param], device="cpu")
        mock_reduce_scatter.return_value = MagicMock()
        group.foreach_reducescatter("sum", async_op=True)

        group.wait_reduce_scatter_and_issue_all_reduce(async_op=True)

        mock_all_reduce.assert_called_once()
        self.assertIs(group.comm_ctx.all_reduce_param_group, group)
        group.wait_all_reduce_and_save_grad()
        self.assertIsNotNone(param.all_reduce_comm_ctx.all_reduce_output)

    def test_requires_all_reduce_false_accumulates_bucket_output(self):
        """No-sync micro-steps should retain reduce outputs at bucket granularity."""
        param = _fake_param([1.0], mesh_info=_mesh_info(DDPMeshInfo), grad=ms.Tensor([2.0]))
        group = HSDPParamGroup([param], device="cpu")
        group.requires_all_reduce = False
        group.foreach_reducescatter("sum", async_op=True)

        group.wait_reduce_scatter_and_issue_all_reduce()

        self.assertEqual(len(group.reduce_partial_outputs), 1)
        self.assertEqual(group.all_reduce_buckets, [])


class TestAllReduceParamGroup(MindSporeFullyShardUnitTest):
    """Test the non-fused-RS all-reduce group used for HSDP overlap."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.all_reduce")
    def test_all_reduce_uses_sum_and_wait_exposes_views(self, mock_all_reduce):
        """Aligned buffers use SUM, with AVG restored after wait and split."""
        param = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(HSDPMeshInfo, shard_group="fsdp", replicate_group="dp"),
        )
        group = AllReduceParamGroup("dp", [param], reduce_op="avg")
        group.allocate_fused_buffer("cpu")
        group.accumulate_reduce_partial_outputs()
        reduced_grad = MagicMock()
        group.get_param_grad_view = MagicMock(return_value=reduced_grad)
        handle = MagicMock()
        mock_all_reduce.return_value = handle

        group.issue_async_allreduce()
        group.wait_and_split_grads()

        self.assertEqual(mock_all_reduce.call_args.kwargs["op"], "sum")
        handle.wait.assert_called_once_with()
        reduced_grad.div_.assert_called_once_with(2)
        self.assertIs(param.all_reduce_comm_ctx.all_reduce_output, reduced_grad)
        self.assertIsNone(group.fused_buffer)


if __name__ == "__main__":
    unittest.main()

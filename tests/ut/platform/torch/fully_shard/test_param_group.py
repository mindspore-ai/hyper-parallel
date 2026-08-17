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
# ==========================================================================
"""Unit tests for Torch HSDP fused communication buckets."""
# pylint: disable=protected-access
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, HSDPMeshInfo
from hyper_parallel.platform.torch.fully_shard.param_group import (
    AllGatherMetadata,
    AllGatherResult,
    AllReduceParamGroup,
    HSDPParamGroup,
    all_gather_copy_in,
    get_all_gather_metadata,
    reduce_scatter_copy_in,
)


class _FakeGroup:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _mesh_info(mesh_cls=FSDPMeshInfo, *, shard_group=None, replicate_group=None):
    """Create mesh metadata without constructing a DeviceMesh."""
    shard_group = shard_group or _FakeGroup()
    replicate_group = replicate_group or _FakeGroup(size=1)
    mesh_info = object.__new__(mesh_cls)
    mesh_info.shard_mesh_rank = shard_group.rank()
    mesh_info.shard_mesh_size = shard_group.size()
    mesh_info.shard_process_group = shard_group
    mesh_info.replicate_mesh_rank = replicate_group.rank()
    mesh_info.replicate_mesh_size = replicate_group.size()
    mesh_info.replicate_process_group = replicate_group
    return mesh_info


def _local_param(local_tensor, requires_grad=True):
    """Create the local parameter view consumed by parameter-group tests."""
    sharded_param = SimpleNamespace(
        _local_tensor=local_tensor,
        data=local_tensor,
        grad=None,
        main_grad=None,
        requires_grad=requires_grad,
        device=local_tensor.device,
    )
    sharded_param.requires_grad_ = lambda value=True: setattr(
        sharded_param,
        "requires_grad",
        value,
    )
    return sharded_param


def _fake_param(
    values,
    *,
    dtype=torch.float32,
    param_dtype=None,
    mesh_info=None,
    shard_dim=0,
    requires_grad=True,
):
    """Create the parameter facts consumed by HSDPParamGroup."""
    local_tensor = torch.tensor(values, dtype=dtype)
    communication_tensor = local_tensor.to(param_dtype) if param_dtype is not None else local_tensor.clone()
    sharded_param = _local_param(local_tensor.clone(), requires_grad)
    hsdp_param = MagicMock()
    hsdp_param.mesh_info = mesh_info or _mesh_info()
    hsdp_param.shard_world_size = (
        hsdp_param.mesh_info.shard_mesh_size
        if isinstance(hsdp_param.mesh_info, FSDPMeshInfo)
        else 1
    )
    hsdp_param.replicate_world_size = hsdp_param.mesh_info.replicate_mesh_size
    hsdp_param.all_gather_inputs = [communication_tensor]
    hsdp_param.orig_dtype = dtype
    hsdp_param.param_dtype = param_dtype
    hsdp_param.reduce_dtype = None
    hsdp_param.reduce_comm_dtype.side_effect = lambda grad=None: hsdp_param.reduce_dtype or (
        grad.dtype if grad is not None else dtype
    )
    hsdp_param.offload_to_cpu = False
    hsdp_param.sharded_param = sharded_param
    hsdp_param._sharded_param_data = sharded_param._local_tensor.view(-1)
    hsdp_param.sharded_size = sharded_param._local_tensor.size()
    hsdp_param.padded_sharded_param_size = hsdp_param.sharded_size
    hsdp_param.contiguous_sharded_stride = sharded_param._local_tensor.stride()
    hsdp_param.hsdp_placement = Shard(shard_dim)
    hsdp_param._orig_size = torch.Size(sharded_param._local_tensor.shape)
    hsdp_param.unsharded_param_buffers = []

    def init_unsharded_param_buffers(numels, dtypes, world_size, device):
        if not hsdp_param.unsharded_param_buffers:
            hsdp_param.unsharded_param_buffers = [
                torch.empty(numel * world_size, dtype=input_dtype, device=device)
                for numel, input_dtype in zip(numels, dtypes)
            ]

    hsdp_param.init_unsharded_param_buffers.side_effect = init_unsharded_param_buffers
    hsdp_param.alloc_unsharded_param_buffers = MagicMock()
    hsdp_param.init_unsharded_param = MagicMock()
    hsdp_param.to_unsharded = MagicMock()
    hsdp_param.unshard = MagicMock()
    hsdp_param.wait_for_unshard = MagicMock()
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.unsharded_accumulated_grad_data = None
    hsdp_param._unsharded_param = SimpleNamespace(grad=None)
    hsdp_param.unsharded_param = hsdp_param._unsharded_param
    hsdp_param.unsharded_grad_data = None
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


def _set_unsharded_grad(hsdp_param, grad):
    hsdp_param.unsharded_param.grad = grad
    hsdp_param.unsharded_grad_data = grad


class TestParamGroupHelpers(unittest.TestCase):
    """Cover copy helpers and metadata validation."""

    def test_get_all_gather_metadata(self):
        param_a = _fake_param([1.0, 2.0])
        param_b = _fake_param([3.0, 4.0, 5.0])

        metadata = get_all_gather_metadata([param_a, param_b])

        self.assertEqual(metadata.inp_split_sizes, [2, 3])
        self.assertEqual(metadata.total_input_numel, 5)
        self.assertEqual(metadata.dtype, torch.float32)

    def test_get_all_gather_metadata_rejects_mixed_dtype(self):
        with self.assertRaisesRegex(ValueError, "same dtype"):
            get_all_gather_metadata([
                _fake_param([1.0], dtype=torch.float32),
                _fake_param([1.0], dtype=torch.float16),
            ])

    def test_all_gather_copy_in(self):
        output = torch.empty(8)
        inputs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        local_input, full_output = all_gather_copy_in(inputs, output, [2, 2], 4, rank=1)

        torch.testing.assert_close(local_input, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertIs(full_output, output)

    def test_reduce_scatter_copy_in_supports_mixed_placements(self):
        """Reduce-scatter packing should support mixed shard dimensions."""
        dim0_grad = torch.arange(8, dtype=torch.float32).view(4, 2)
        dim1_grad = torch.arange(16, dtype=torch.float32).view(4, 4)
        reduce_scatter_input = torch.empty(24)
        params = [
            SimpleNamespace(
                hsdp_placement=Shard(0),
                padded_sharded_param_size=torch.Size((2, 2)),
            ),
            SimpleNamespace(
                hsdp_placement=Shard(1),
                padded_sharded_param_size=torch.Size((4, 2)),
            ),
        ]

        reduce_scatter_copy_in(
            params,
            [dim0_grad, dim1_grad],
            reduce_scatter_input,
            world_size=2,
        )

        packed_rows = reduce_scatter_input.view(2, -1)
        torch.testing.assert_close(packed_rows[:, :4], dim0_grad.view(2, 4))
        expected_dim1 = torch.cat(torch.chunk(dim1_grad, 2, dim=1), dim=0).view(2, 8)
        torch.testing.assert_close(packed_rows[:, 4:], expected_dim1)


class TestAllGatherBuckets(unittest.TestCase):
    """Cover ordered dtype buckets and all-gather buffer ownership."""

    def test_init_all_gather_buckets_groups_by_process_group_and_dtype(self):
        """All-gather buckets should group parameters by process group and dtype."""
        shard_group_a = _FakeGroup()
        shard_group_b = _FakeGroup()
        mesh_info_a = _mesh_info(shard_group=shard_group_a)
        mesh_info_b = _mesh_info(shard_group=shard_group_b)
        param_fp32_a = _fake_param([1.0, 2.0], mesh_info=mesh_info_a)
        param_fp32_b = _fake_param([3.0, 4.0], mesh_info=mesh_info_b)
        param_fp16_a = _fake_param([5.0, 6.0], dtype=torch.float16, mesh_info=mesh_info_a)
        cast_fp32 = _fake_param(
            [7.0, 8.0],
            dtype=torch.float32,
            param_dtype=torch.float16,
            mesh_info=mesh_info_a,
        )
        param_group = HSDPParamGroup(
            [param_fp32_a, param_fp32_b, param_fp16_a, cast_fp32],
            device=torch.device("cpu"),
            enable_zero_copy=True,
        )

        param_group._init_all_gather_buckets()

        self.assertEqual([bucket.dtype for bucket in param_group.all_gather_buckets], [
            torch.float32,
            torch.float32,
            torch.float16,
        ])
        self.assertEqual(
            [bucket.shard_group for bucket in param_group.all_gather_buckets],
            [shard_group_a, shard_group_b, shard_group_a],
        )
        self.assertEqual(param_group.all_gather_buckets[2].hsdp_params, [param_fp16_a, cast_fp32])
        param_group.all_gather_buckets[2].init_flat_param_buffer(param_group.device)
        self.assertIsNone(param_group.all_gather_buckets[2].flat_param_buffer)

    def test_shard_size_one_stays_in_param_group_without_all_gather_bucket(self):
        """Shard-size-one parameters should unshard without an all-gather bucket."""
        replicate_group = _FakeGroup(size=2)
        hsdp_param = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(DDPMeshInfo, replicate_group=replicate_group),
        )
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))

        param_group.foreach_all_gather(async_op=True)
        param_group.wait_for_unshard()

        self.assertEqual(param_group.all_gather_buckets, [])
        hsdp_param.unshard.assert_called_once_with(True)
        hsdp_param.wait_for_unshard.assert_called_once_with()

    def test_flat_param_buffer_rebases_homogeneous_storage(self):
        """Zero-copy all-gather should rebase homogeneous parameter storage."""
        mesh_info = _mesh_info()
        param_a = _fake_param([1.0, 2.0], mesh_info=mesh_info)
        param_b = _fake_param([3.0, 4.0], mesh_info=mesh_info)
        param_group = HSDPParamGroup(
            [param_a, param_b],
            device=torch.device("cpu"),
            enable_zero_copy=True,
        )
        param_group._init_all_gather_buckets()
        all_gather_bucket = param_group.all_gather_buckets[0]

        all_gather_bucket.init_flat_param_buffer(param_group.device)

        self.assertTrue(all_gather_bucket.is_flat_buffer_valid())
        self.assertEqual(all_gather_bucket.flat_param_buffer.numel(), 4)
        flat_storage_ptr = all_gather_bucket.flat_param_buffer.untyped_storage().data_ptr()
        self.assertEqual(param_a._sharded_param_data.untyped_storage().data_ptr(), flat_storage_ptr)
        self.assertEqual(param_b._sharded_param_data.untyped_storage().data_ptr(), flat_storage_ptr)

    def test_flat_param_buffer_preserves_uneven_shard_local_shape(self):
        """Zero-copy rebasing should preserve the N-D uneven local shard shape."""
        hsdp_param = _fake_param([[1.0, 2.0], [3.0, 4.0]])
        hsdp_param._spmd_placements = (Shard(0, uneven_shard=True),)
        param_group = HSDPParamGroup(
            [hsdp_param],
            device=torch.device("cpu"),
            enable_zero_copy=True,
        )
        param_group._init_all_gather_buckets()

        param_group.all_gather_buckets[0].init_flat_param_buffer(param_group.device)

        self.assertEqual(hsdp_param.sharded_param._local_tensor.shape, torch.Size((2, 2)))
        self.assertEqual(hsdp_param.sharded_param.data.shape, torch.Size((2, 2)))

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_gather_into_tensor")
    def test_all_gather_result_releases_temporary_references(self, mock_all_gather):
        """Waiting for unshard should release temporary all-gather references."""
        hsdp_param = _fake_param([1.0, 2.0])
        param_group = HSDPParamGroup(
            [hsdp_param],
            device=torch.device("cpu"),
            enable_zero_copy=False,
        )
        handle = MagicMock()

        def all_gather(output, input_tensor, **unused):
            output.view(2, -1)[0].copy_(input_tensor)
            output.view(2, -1)[1].copy_(input_tensor + 2)
            return handle

        mock_all_gather.side_effect = all_gather

        param_group.foreach_all_gather(async_op=True)
        all_gather_bucket = param_group.all_gather_buckets[0]
        self.assertIsNotNone(all_gather_bucket.all_gather_result.all_gather_input)
        param_group.wait_for_unshard()

        handle.wait.assert_called_once_with()
        hsdp_param.wait_for_unshard.assert_called_once_with()
        self.assertIsNone(all_gather_bucket.all_gather_result)
        torch.testing.assert_close(
            hsdp_param.unsharded_param_buffers[0],
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

    def test_copy_out_preserves_stable_buffer_version(self):
        """All-gather copy-out should preserve the persistent buffer version."""
        hsdp_param = _fake_param([1.0, 2.0])
        param_group = HSDPParamGroup(
            [hsdp_param],
            device=torch.device("cpu"),
            enable_zero_copy=False,
        )
        param_group._init_all_gather_buckets()
        all_gather_bucket = param_group.all_gather_buckets[0]
        hsdp_param.init_unsharded_param_buffers(
            [2],
            [torch.float32],
            2,
            torch.device("cpu"),
        )
        output_buffer = hsdp_param.unsharded_param_buffers[0]
        initial_version = output_buffer._version
        all_gather_bucket.all_gather_result = AllGatherResult(
            all_gather_input=torch.ones(2),
            all_gather_output=torch.arange(4, dtype=torch.float32),
            handle=None,
        )

        all_gather_bucket.copy_out()

        self.assertEqual(output_buffer._version, initial_version)

    def test_copy_out_restores_mixed_shard_dimensions(self):
        """All-gather copy-out should restore parameters sharded on mixed dimensions."""
        mesh_info = _mesh_info()
        dim0_param = _fake_param(
            [[0.0, 1.0], [2.0, 3.0]],
            mesh_info=mesh_info,
            shard_dim=0,
        )
        dim0_param._orig_size = torch.Size((4, 2))
        dim1_param = _fake_param(
            [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0], [12.0, 13.0]],
            mesh_info=mesh_info,
            shard_dim=1,
        )
        dim1_param._orig_size = torch.Size((4, 4))
        param_group = HSDPParamGroup(
            [dim0_param, dim1_param],
            device=torch.device("cpu"),
            enable_zero_copy=False,
        )
        param_group._init_all_gather_buckets()
        all_gather_bucket = param_group.all_gather_buckets[0]
        rank0_input = torch.tensor([
            0.0, 1.0, 2.0, 3.0,
            0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0,
        ])
        rank1_input = torch.tensor([
            4.0, 5.0, 6.0, 7.0,
            2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0,
        ])
        all_gather_bucket.all_gather_result = AllGatherResult(
            all_gather_input=rank0_input,
            all_gather_output=torch.cat((rank0_input, rank1_input)),
            handle=None,
        )

        all_gather_bucket.copy_out()

        torch.testing.assert_close(
            dim0_param.unsharded_param_buffers[0].view(4, 2),
            torch.arange(8, dtype=torch.float32).view(4, 2),
        )
        torch.testing.assert_close(
            dim1_param.unsharded_param_buffers[0].view(4, 4),
            torch.arange(16, dtype=torch.float32).view(4, 4),
        )
        self.assertIsNone(all_gather_bucket.all_gather_result)


class TestReduceBuckets(unittest.TestCase):
    """Cover mixed dtype RS buckets and delayed all-reduce behavior."""

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_reduce_scatter_bucket_builder_has_no_execution_side_effects(
        self,
        mock_reduce_scatter,
        mock_apply_scaling,
    ):
        """Building reduce-scatter buckets should not launch communication."""
        hsdp_param = _fake_param([1.0, 2.0])
        unsharded_grad = torch.arange(4, dtype=torch.float32)
        _set_unsharded_grad(hsdp_param, unsharded_grad)
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))

        reduce_scatter_buckets = param_group._build_reduce_scatter_buckets(dist.ReduceOp.SUM)

        self.assertEqual(len(reduce_scatter_buckets), 1)
        reduce_scatter_bucket = reduce_scatter_buckets[0]
        self.assertIsNone(reduce_scatter_bucket.reduce_scatter_input)
        self.assertIsNone(reduce_scatter_bucket.reduce_scatter_output)
        self.assertIsNone(reduce_scatter_bucket.handle)
        self.assertIs(hsdp_param.unsharded_param.grad, unsharded_grad)
        mock_apply_scaling.assert_not_called()
        mock_reduce_scatter.assert_not_called()

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_reduce_scatter_buckets_group_by_process_group_and_dtype(self, mock_reduce_scatter):
        """Reduce-scatter buckets should group by process group and dtype."""
        shard_group_a = _FakeGroup()
        shard_group_b = _FakeGroup()
        mesh_info_a = _mesh_info(shard_group=shard_group_a)
        mesh_info_b = _mesh_info(shard_group=shard_group_b)
        param_fp32_a = _fake_param([1.0, 2.0], mesh_info=mesh_info_a)
        param_fp32_b = _fake_param([1.0, 2.0], mesh_info=mesh_info_b)
        param_fp16_a = _fake_param([1.0, 2.0], dtype=torch.float16, mesh_info=mesh_info_a)
        _set_unsharded_grad(param_fp32_a, torch.arange(4, dtype=torch.float32))
        _set_unsharded_grad(param_fp32_b, torch.arange(4, dtype=torch.float32))
        _set_unsharded_grad(param_fp16_a, torch.arange(4, dtype=torch.float16))
        param_group = HSDPParamGroup(
            [param_fp32_a, param_fp32_b, param_fp16_a],
            device=torch.device("cpu"),
        )

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            reduce_scatter_input = collective_args["input"]
            output.copy_(reduce_scatter_input.view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)

        self.assertEqual(
            [bucket.dtype for bucket in param_group.reduce_scatter_buckets],
            [torch.float32, torch.float32, torch.float16],
        )
        self.assertEqual(
            [bucket.shard_group for bucket in param_group.reduce_scatter_buckets],
            [shard_group_a, shard_group_b, shard_group_a],
        )
        self.assertIs(param_group.comm_ctx.pre_param_group, param_group)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_requires_all_reduce_accumulates_fresh_reduce_outputs(self, mock_reduce_scatter):
        """Deferred all-reduce should accumulate fresh reduce-scatter outputs."""
        hsdp_param = _fake_param([1.0, 2.0])
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            reduce_scatter_input = collective_args["input"]
            output.copy_(reduce_scatter_input.view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        param_group.requires_all_reduce = False
        no_all_reduce_steps = (
            (
                torch.tensor([1.0, 2.0, 3.0, 4.0]),
                torch.tensor([4.0, 6.0]),
            ),
            (torch.ones(4), torch.tensor([6.0, 8.0])),
        )
        parked_output = None
        for unsharded_grad, expected_partial in no_all_reduce_steps:
            _set_unsharded_grad(hsdp_param, unsharded_grad)
            param_group.foreach_reducescatter(dist.ReduceOp.SUM)
            reduce_scatter_bucket = param_group.reduce_scatter_buckets[0]
            bucket_key = reduce_scatter_bucket.bucket_key
            param_group.wait_reduce_scatter_and_issue_all_reduce()
            torch.testing.assert_close(
                param_group.reduce_partial_outputs[bucket_key],
                expected_partial,
            )
            if parked_output is None:
                parked_output = param_group.reduce_partial_outputs[bucket_key]
            self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)
            self.assertIsNone(reduce_scatter_bucket.reduce_scatter_input)
            self.assertIsNone(reduce_scatter_bucket.handle)
        # Accumulation reuses the first parked buffer: later micro-steps add into
        # it and release their own reduce-scatter output.
        self.assertIs(param_group.reduce_partial_outputs[bucket_key], parked_output)
        # The whole reduce-scatter output is parked once per bucket, not once per
        # parameter: accumulation must not fall back to per-parameter views.
        self.assertEqual(len(param_group.reduce_partial_outputs), 1)
        self.assertIsNone(hsdp_param.reduce_partial_output)
        hsdp_param.all_reduce_tp_replicate_grad_inplace.assert_not_called()

        param_group.requires_all_reduce = True
        _set_unsharded_grad(hsdp_param, torch.ones(4))
        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        param_group.wait_reduce_scatter_and_issue_all_reduce()

        torch.testing.assert_close(
            hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output,
            torch.tensor([8.0, 10.0]),
        )
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_output)
        self.assertEqual(param_group.reduce_partial_outputs, {})

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_gradient_scaling_is_applied_to_packed_reduce_scatter_input(self, mock_reduce_scatter):
        """Gradient scaling should apply to the packed reduce-scatter input."""
        hsdp_param = _fake_param([1.0, 2.0])
        _set_unsharded_grad(hsdp_param, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))
        param_group.gradient_scaling_factor = 0.5

        def reduce_scatter(**collective_args):
            torch.testing.assert_close(
                collective_args["input"],
                torch.tensor([0.5, 1.0, 1.5, 2.0]),
            )
            collective_args["output"].copy_(collective_args["input"].view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        param_group.wait_reduce_scatter_and_issue_all_reduce()

        torch.testing.assert_close(
            hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output,
            torch.tensor([2.0, 3.0]),
        )

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_hsdp_all_reduce_saves_per_param_context(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """HSDP all-reduce should save its output in each parameter context."""
        replicate_group = _FakeGroup(size=2)
        mesh_info = _mesh_info(
            HSDPMeshInfo,
            shard_group=_FakeGroup(size=2),
            replicate_group=replicate_group,
        )
        hsdp_param = _fake_param([1.0, 2.0], mesh_info=mesh_info)
        _set_unsharded_grad(hsdp_param, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            reduce_scatter_input = collective_args["input"]
            output.copy_(reduce_scatter_input.view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        mock_all_reduce.return_value = MagicMock()

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        param_group.wait_reduce_scatter_and_issue_all_reduce()
        all_reduce_bucket = param_group.all_reduce_buckets[0]
        param_group.wait_all_reduce_and_save_grad()

        torch.testing.assert_close(
            hsdp_param.all_reduce_comm_ctx.all_reduce_output,
            torch.tensor([4.0, 6.0]),
        )
        self.assertEqual(param_group.reduce_scatter_buckets, [])
        self.assertEqual(param_group.all_reduce_buckets, [])
        self.assertIsNone(all_reduce_bucket.all_reduce_output)
        self.assertIsNone(all_reduce_bucket.handle)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_all_reduce_buckets_group_by_process_group_and_dtype(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """All-reduce buckets should group by replicate process group and dtype."""
        shard_group_a = _FakeGroup(size=2)
        shard_group_b = _FakeGroup(size=2)
        replicate_group_a = _FakeGroup(size=2)
        replicate_group_b = _FakeGroup(size=4)
        param_a = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(
                HSDPMeshInfo,
                shard_group=shard_group_a,
                replicate_group=replicate_group_a,
            ),
        )
        param_b = _fake_param(
            [3.0, 4.0],
            mesh_info=_mesh_info(
                HSDPMeshInfo,
                shard_group=shard_group_b,
                replicate_group=replicate_group_b,
            ),
        )
        param_c = _fake_param(
            [5.0, 6.0],
            dtype=torch.float16,
            mesh_info=_mesh_info(
                HSDPMeshInfo,
                shard_group=shard_group_a,
                replicate_group=replicate_group_a,
            ),
        )
        _set_unsharded_grad(param_a, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        _set_unsharded_grad(param_b, torch.tensor([5.0, 6.0, 7.0, 8.0]))
        _set_unsharded_grad(param_c, torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float16))
        param_group = HSDPParamGroup([param_a, param_b, param_c], device=torch.device("cpu"))

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            reduce_scatter_input = collective_args["input"]
            output.copy_(reduce_scatter_input.view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        mock_all_reduce.return_value = MagicMock()

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        param_group.wait_reduce_scatter_and_issue_all_reduce()

        self.assertEqual(
            [bucket.replicate_group for bucket in param_group.all_reduce_buckets],
            [replicate_group_a, replicate_group_b, replicate_group_a],
        )
        self.assertEqual(
            [bucket.dtype for bucket in param_group.all_reduce_buckets],
            [torch.float32, torch.float32, torch.float16],
        )
        self.assertEqual(
            [bucket.hsdp_params for bucket in param_group.all_reduce_buckets],
            [[param_a], [param_b], [param_c]],
        )
        self.assertEqual(mock_all_reduce.call_count, 3)
        param_group.wait_all_reduce_and_save_grad()

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_all_reduce_reuses_reduce_scatter_output_when_buckets_align(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """A whole RS bucket feeding one AR bucket should be all-reduced in place."""
        shard_group = _FakeGroup(size=2)
        replicate_group = _FakeGroup(size=2)
        mesh_info = _mesh_info(
            HSDPMeshInfo,
            shard_group=shard_group,
            replicate_group=replicate_group,
        )
        param_a = _fake_param([1.0, 2.0], mesh_info=mesh_info)
        param_b = _fake_param([3.0, 4.0], mesh_info=mesh_info)
        _set_unsharded_grad(param_a, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        _set_unsharded_grad(param_b, torch.tensor([5.0, 6.0, 7.0, 8.0]))
        param_group = HSDPParamGroup([param_a, param_b], device=torch.device("cpu"))

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            output.copy_(collective_args["input"].view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        mock_all_reduce.return_value = MagicMock()

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        reduce_scatter_output = param_group.reduce_scatter_buckets[0].reduce_scatter_output
        param_group.wait_reduce_scatter_and_issue_all_reduce()

        self.assertEqual(len(param_group.all_reduce_buckets), 1)
        self.assertIs(
            param_group.all_reduce_buckets[0].all_reduce_output,
            reduce_scatter_output,
        )
        self.assertIs(mock_all_reduce.call_args.args[0], reduce_scatter_output)
        self.assertIsNone(param_a.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIsNone(param_b.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIsNone(param_a.all_reduce_comm_ctx.all_reduce_output)
        self.assertIsNone(param_b.all_reduce_comm_ctx.all_reduce_output)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_all_reduce_rejects_different_routes_in_one_reduce_scatter_bucket(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """One RS bucket cannot transfer its output to different AR groups."""
        shard_group = _FakeGroup(size=2)
        replicate_group_a = _FakeGroup(size=2)
        replicate_group_b = _FakeGroup(size=4)
        param_a = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(
                HSDPMeshInfo,
                shard_group=shard_group,
                replicate_group=replicate_group_a,
            ),
        )
        param_b = _fake_param(
            [3.0, 4.0],
            mesh_info=_mesh_info(
                HSDPMeshInfo,
                shard_group=shard_group,
                replicate_group=replicate_group_b,
            ),
        )
        _set_unsharded_grad(param_a, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        _set_unsharded_grad(param_b, torch.tensor([5.0, 6.0, 7.0, 8.0]))
        param_group = HSDPParamGroup([param_a, param_b], device=torch.device("cpu"))

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            output.copy_(collective_args["input"].view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        mock_all_reduce.return_value = MagicMock()

        with self.assertRaisesRegex(ValueError, "subsequent all-reduce group"):
            param_group.foreach_reducescatter(dist.ReduceOp.SUM)

        mock_reduce_scatter.assert_not_called()
        mock_all_reduce.assert_not_called()
        self.assertIsNotNone(param_a.unsharded_param.grad)
        self.assertIsNotNone(param_b.unsharded_param.grad)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_replicate_param_uses_local_reduce_scatter_and_bucketed_all_reduce(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """Replicated parameters should bypass reduce-scatter and use all-reduce."""
        replicate_group = _FakeGroup(size=2)
        hsdp_param = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(DDPMeshInfo, replicate_group=replicate_group),
        )
        _set_unsharded_grad(hsdp_param, torch.tensor([1.0, 2.0]))
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))
        mock_all_reduce.return_value = MagicMock()

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        param_group.wait_reduce_scatter_and_issue_all_reduce()
        param_group.wait_all_reduce_and_save_grad()

        mock_reduce_scatter.assert_not_called()
        mock_all_reduce.assert_called_once()
        self.assertIs(mock_all_reduce.call_args.kwargs["group"], replicate_group)
        torch.testing.assert_close(
            hsdp_param.all_reduce_comm_ctx.all_reduce_output,
            torch.tensor([1.0, 2.0]),
        )
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_replicate_params_share_local_reduce_scatter_and_fused_all_reduce(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """Replicated parameters should share local packing and fused all-reduce."""
        replicate_group = _FakeGroup(size=2)
        mesh_info = _mesh_info(DDPMeshInfo, replicate_group=replicate_group)
        param_a = _fake_param([1.0, 2.0], mesh_info=mesh_info)
        param_b = _fake_param([3.0, 4.0], mesh_info=mesh_info)
        _set_unsharded_grad(param_a, torch.tensor([1.0, 2.0]))
        _set_unsharded_grad(param_b, torch.tensor([3.0, 4.0]))
        param_group = HSDPParamGroup([param_a, param_b], device=torch.device("cpu"))
        mock_all_reduce.return_value = MagicMock()

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        self.assertEqual(len(param_group.reduce_scatter_buckets), 1)
        reduce_scatter_bucket = param_group.reduce_scatter_buckets[0]
        self.assertIs(
            reduce_scatter_bucket.reduce_scatter_output,
            reduce_scatter_bucket.reduce_scatter_input,
        )
        reduce_scatter_output = reduce_scatter_bucket.reduce_scatter_output
        param_group.wait_reduce_scatter_and_issue_all_reduce()

        self.assertEqual(len(param_group.all_reduce_buckets), 1)
        self.assertIs(param_group.all_reduce_buckets[0].all_reduce_output, reduce_scatter_output)
        self.assertIs(mock_all_reduce.call_args.args[0], reduce_scatter_output)
        mock_reduce_scatter.assert_not_called()
        mock_all_reduce.assert_called_once()

        param_group.wait_all_reduce_and_save_grad()
        torch.testing.assert_close(param_a.all_reduce_comm_ctx.all_reduce_output, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(param_b.all_reduce_comm_ctx.all_reduce_output, torch.tensor([3.0, 4.0]))

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_hsdp_and_replicate_params_use_independent_reduce_scatter_buckets(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """Sharded and replicated parameters should use independent reduce buckets."""
        shard_group = _FakeGroup(size=2)
        hsdp_replicate_group = _FakeGroup(size=2)
        flattened_replicate_group = _FakeGroup(size=4)
        sharded_param = _fake_param(
            [1.0, 2.0],
            mesh_info=_mesh_info(
                HSDPMeshInfo,
                shard_group=shard_group,
                replicate_group=hsdp_replicate_group,
            ),
        )
        replicate_param = _fake_param(
            [3.0, 4.0],
            mesh_info=_mesh_info(
                DDPMeshInfo,
                replicate_group=flattened_replicate_group,
            ),
        )
        _set_unsharded_grad(sharded_param, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        _set_unsharded_grad(replicate_param, torch.tensor([5.0, 6.0]))
        param_group = HSDPParamGroup(
            [sharded_param, replicate_param],
            device=torch.device("cpu"),
        )

        def reduce_scatter(**collective_args):
            collective_args["output"].copy_(
                collective_args["input"].view(2, -1).sum(dim=0)
            )
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        mock_all_reduce.return_value = MagicMock()

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)

        self.assertEqual(len(param_group.reduce_scatter_buckets), 2)
        self.assertEqual(
            [bucket.shard_group for bucket in param_group.reduce_scatter_buckets],
            [shard_group, None],
        )
        param_group.wait_reduce_scatter_and_issue_all_reduce()
        self.assertEqual(
            [bucket.replicate_group for bucket in param_group.all_reduce_buckets],
            [hsdp_replicate_group, flattened_replicate_group],
        )
        mock_reduce_scatter.assert_called_once()
        self.assertEqual(mock_all_reduce.call_count, 2)

        param_group.wait_all_reduce_and_save_grad()
        torch.testing.assert_close(
            sharded_param.all_reduce_comm_ctx.all_reduce_output,
            torch.tensor([4.0, 6.0]),
        )
        torch.testing.assert_close(
            replicate_param.all_reduce_comm_ctx.all_reduce_output,
            torch.tensor([5.0, 6.0]),
        )

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_hsdp_replicate_group_size_one_still_uses_all_reduce_output(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """A size-one replicate group should still save through all-reduce output."""
        mesh_info = _mesh_info(
            HSDPMeshInfo,
            shard_group=_FakeGroup(size=2),
            replicate_group=_FakeGroup(size=1),
        )
        hsdp_param = _fake_param([1.0, 2.0], mesh_info=mesh_info)
        _set_unsharded_grad(hsdp_param, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))

        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            output.copy_(collective_args["input"].view(2, -1).sum(dim=0))
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter

        param_group.foreach_reducescatter(dist.ReduceOp.SUM)
        param_group.wait_reduce_scatter_and_issue_all_reduce()

        self.assertEqual(len(param_group.all_reduce_buckets), 1)
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)
        mock_all_reduce.assert_not_called()

        param_group.wait_all_reduce_and_save_grad()
        torch.testing.assert_close(
            hsdp_param.all_reduce_comm_ctx.all_reduce_output,
            torch.tensor([4.0, 6.0]),
        )

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_avg_and_sum_scaling_cover_shard_and_replicate_groups(
        self,
        mock_reduce_scatter,
        mock_all_reduce,
    ):
        """AVG and SUM should scale across both shard and replicate groups."""
        def reduce_scatter(**collective_args):
            output = collective_args["output"]
            reduce_scatter_input = collective_args["input"]
            output.copy_(reduce_scatter_input.view(2, -1).sum(dim=0))
            return MagicMock()

        def all_reduce(output, **unused):
            output.mul_(2)
            return MagicMock()

        mock_reduce_scatter.side_effect = reduce_scatter
        mock_all_reduce.side_effect = all_reduce

        for reduce_op, expected in (
            (dist.ReduceOp.AVG, torch.tensor([2.0, 3.0])),
            (dist.ReduceOp.SUM, torch.tensor([8.0, 12.0])),
        ):
            with self.subTest(reduce_op=reduce_op):
                mesh_info = _mesh_info(
                    HSDPMeshInfo,
                    shard_group=_FakeGroup(size=2),
                    replicate_group=_FakeGroup(size=2),
                )
                hsdp_param = _fake_param([1.0, 2.0], mesh_info=mesh_info)
                _set_unsharded_grad(
                    hsdp_param,
                    torch.tensor([1.0, 2.0, 3.0, 4.0]),
                )
                param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))

                param_group.foreach_reducescatter(reduce_op)
                param_group.wait_reduce_scatter_and_issue_all_reduce()
                param_group.wait_all_reduce_and_save_grad()

                torch.testing.assert_close(
                    hsdp_param.all_reduce_comm_ctx.all_reduce_output,
                    expected,
                )


class TestAllReduceParamGroup(unittest.TestCase):
    """Keep non-fusion fused all-reduce behavior covered."""

    def test_wait_and_split_grads_releases_fused_buffer(self):
        """Waiting and splitting gradients should release the fused buffer."""
        hsdp_param = _fake_param([1.0, 2.0])
        group = AllReduceParamGroup(_FakeGroup(size=2), [hsdp_param], dist.ReduceOp.AVG)
        group.allocate_fused_buffer(torch.device("cpu"))
        group.get_param_buffer_view(0).copy_(torch.tensor([2.0, 4.0]))

        group.wait_and_split_grads()

        torch.testing.assert_close(
            hsdp_param.all_reduce_comm_ctx.all_reduce_output,
            torch.tensor([1.0, 2.0]),
        )
        self.assertIsNone(group.fused_buffer)


class TestParamGroupReset(unittest.TestCase):
    """Cover iteration reset ownership and persistent AllGather storage."""

    def test_reset_releases_temporary_communication_state(self):
        """Reset should release temporary communication state and buffers."""
        hsdp_param = _fake_param([1.0, 2.0])
        param_group = HSDPParamGroup([hsdp_param], device=torch.device("cpu"))
        param_group._init_all_gather_buckets()
        all_gather_bucket = param_group.all_gather_buckets[0]
        flat_param_buffer = torch.ones(2)
        all_gather_bucket.flat_param_buffer = flat_param_buffer
        all_gather_result = SimpleNamespace(
            all_gather_input=torch.ones(2),
            all_gather_output=torch.ones(4),
            handle=MagicMock(),
        )
        all_gather_bucket.all_gather_result = all_gather_result
        reduce_scatter_bucket = SimpleNamespace(
            reduce_scatter_input=torch.ones(4),
            reduce_scatter_output=torch.ones(2),
            handle=MagicMock(),
        )
        all_reduce_bucket = SimpleNamespace(
            all_reduce_output=torch.ones(2),
            handle=MagicMock(),
        )
        param_group.reduce_scatter_buckets = [reduce_scatter_bucket]
        param_group.all_reduce_buckets = [all_reduce_bucket]
        param_group.comm_ctx.pre_param_group = param_group
        param_group.comm_ctx.all_reduce_param_group = param_group

        param_group.reset_iter_state()

        self.assertIsNone(all_gather_result.all_gather_input)
        self.assertIsNone(all_gather_result.all_gather_output)
        self.assertIsNone(all_gather_result.handle)
        self.assertIsNone(all_gather_bucket.all_gather_result)
        self.assertIs(all_gather_bucket.flat_param_buffer, flat_param_buffer)
        self.assertIsNone(reduce_scatter_bucket.reduce_scatter_input)
        self.assertIsNone(reduce_scatter_bucket.reduce_scatter_output)
        self.assertIsNone(reduce_scatter_bucket.handle)
        self.assertIsNone(all_reduce_bucket.all_reduce_output)
        self.assertIsNone(all_reduce_bucket.handle)
        self.assertEqual(param_group.reduce_scatter_buckets, [])
        self.assertEqual(param_group.all_reduce_buckets, [])
        self.assertIsNone(param_group.comm_ctx.pre_param_group)
        self.assertIsNone(param_group.comm_ctx.all_reduce_param_group)

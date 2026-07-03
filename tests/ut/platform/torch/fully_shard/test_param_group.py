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
"""Unit tests for torch fully_shard fused parameter-group helpers."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch

from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, HSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.torch.fully_shard import param_group as param_group_mod
from hyper_parallel.platform.torch.fully_shard.param_group import (
    AllGatherMetadata,
    AllGatherMetadataCache,
    AllGatherResult,
    HSDPParamGroup,
    PendingBucketAllReduce,
    ReplicateBucket,
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


def _mesh_info(mesh_cls, *, shard_size=2, replicate_size=1):
    """Create mesh-info objects without building a real DeviceMesh."""
    mesh_info = object.__new__(mesh_cls)
    mesh_info.shard_mesh_rank = 0
    mesh_info.shard_mesh_size = shard_size
    mesh_info.shard_process_group = _FakeGroup(shard_size, 0)
    mesh_info.replicate_mesh_rank = 0
    mesh_info.replicate_mesh_size = replicate_size
    mesh_info.replicate_process_group = _FakeGroup(replicate_size, 0)
    return mesh_info


def _tensor_dtensor(local_tensor, requires_grad=True):
    obj = SimpleNamespace(
        _local_tensor=local_tensor,
        data=local_tensor,
        grad=None,
        requires_grad=requires_grad,
        device=local_tensor.device,
    )
    obj.requires_grad_ = lambda value=True: setattr(obj, "requires_grad", value)
    return obj


def _fake_param(values, *, dtype=torch.float32, requires_grad=True):
    """Create a fake HSDP parameter backed by a local CPU tensor."""
    local = torch.tensor(values, dtype=dtype)
    sharded_param = _tensor_dtensor(local.clone(), requires_grad=requires_grad)
    hsdp_param = MagicMock()
    hsdp_param.version = 0
    hsdp_param.all_gather_inputs = [local.clone()]
    hsdp_param.all_gather_outputs = []
    hsdp_param.init_all_gather_outputs.side_effect = (
        lambda numels, dtypes, world_size, device: setattr(
            hsdp_param,
            "all_gather_outputs",
            [torch.empty(numel * world_size, dtype=dtype, device=device) for numel, dtype in zip(numels, dtypes)],
        )
    )
    hsdp_param.alloc_all_gather_outputs = MagicMock()
    hsdp_param.init_unsharded_param = MagicMock()
    hsdp_param.to_unsharded = MagicMock()
    hsdp_param.init_dtype_attrs = MagicMock()
    hsdp_param.orig_dtype = dtype
    hsdp_param.reduce_dtype = dtype
    hsdp_param.param_dtype = None
    hsdp_param.offload_to_cpu = False
    hsdp_param.sharded_param = sharded_param
    hsdp_param._sharded_param_data = sharded_param._local_tensor.reshape(-1)
    hsdp_param.sharded_size = sharded_param._local_tensor.size()
    hsdp_param.contiguous_sharded_stride = sharded_param._local_tensor.stride()
    hsdp_param.hsdp_placement = Shard(0)
    hsdp_param._orig_size = None
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.unsharded_accumulated_grad_data = None
    hsdp_param._unsharded_param = SimpleNamespace(grad=None)
    hsdp_param.unsharded_param = hsdp_param._unsharded_param
    hsdp_param.unsharded_grad_data = None
    hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor.clone()))
    hsdp_param.unsharded_group_info = GroupInfo("invalid", None, 1)
    return hsdp_param


def _new_param_group(params, *, world_size=2, enable_zero_copy=False):
    """Create an uninitialized HSDPParamGroup with common test fields."""
    group = object.__new__(HSDPParamGroup)
    group.mesh_info = SimpleNamespace(shard_process_group=_FakeGroup(world_size, 0))
    group.device = torch.device("cpu")
    group.hsdp_params = params
    group.shard_rank = 0
    group.shard_world_size = world_size
    group.shard_group = _FakeGroup(world_size, 0)
    group.replicate_group = None
    group._all_gather_output = torch.empty(0)
    group.ag_output = None
    group.metadata_cache = None
    group.mp_policy = MixedPrecisionPolicy()
    group.enable_zero_copy = enable_zero_copy
    group._result = None
    group._reduce_output = None
    group._reduce_op = None
    group._needs_avg_div = False
    group._reduce_hsdp_params = None
    group._active_replicate_buckets = {}
    group._active_param_flat_offsets = []
    group._pending_all_reduce_handles = []
    group._orig_dtype = torch.float32
    group._reduce_dtype = torch.float32
    group._flat_param_buffer = None
    group._flat_cast_buffer = None
    group.gradient_scaling_factor = None
    return group


class TestTorchParamGroupHelpers(unittest.TestCase):
    """Cover param-group helper functions without distributed initialization."""

    def setUp(self):
        AllGatherMetadataCache._cache.clear()
        param_group_mod.comm_ctx.comm_handle = None
        param_group_mod.comm_ctx.all_reduce_handle = None
        param_group_mod.comm_ctx.pre_param_group = None
        param_group_mod.comm_ctx.all_reduce_param_group = None

    def test_get_all_gather_metadata_and_cache(self):
        """All-gather metadata should include split sizes and refresh on version changes."""
        param_a = _fake_param([1.0, 2.0])
        param_b = _fake_param([3.0, 4.0, 5.0])

        metadata = get_all_gather_metadata([param_a, param_b])
        cached = AllGatherMetadataCache.get_metadata([param_a, param_b], get_all_gather_metadata)
        cached_again = AllGatherMetadataCache.get_metadata([param_a, param_b], MagicMock())
        param_b.version = 1
        refreshed = AllGatherMetadataCache.get_metadata([param_a, param_b], get_all_gather_metadata)

        self.assertEqual(metadata.inp_split_sizes, [2, 3])
        self.assertEqual(metadata.total_input_numel, 5)
        self.assertEqual(cached_again, cached)
        self.assertNotEqual(refreshed.hash_key, 0)

    def test_get_all_gather_metadata_rejects_mixed_dtype(self):
        """All-gather metadata should reject mixed input dtypes."""
        with self.assertRaisesRegex(ValueError, "uniform dtype"):
            get_all_gather_metadata([
                _fake_param([1.0], dtype=torch.float32),
                _fake_param([1.0], dtype=torch.float16),
            ])

    def test_all_gather_copy_in_and_reduce_scatter_copy_in(self):
        """Copy-in helpers should flatten all-gather and reduce-scatter tensors."""
        output = torch.empty(8)
        inputs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

        local_input, full_output = all_gather_copy_in(inputs, output, [2, 2], 4, rank=1)

        torch.testing.assert_close(local_input, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertIs(full_output, output)

        reduce_input = torch.empty(8)
        params = [SimpleNamespace(hsdp_placement=Shard(0))]
        reduce_scatter_copy_in(params, [torch.arange(8, dtype=torch.float32)], reduce_input, world_size=2)
        torch.testing.assert_close(reduce_input.view(2, 4), torch.arange(8, dtype=torch.float32).view(2, 4))
        with self.assertRaisesRegex(AssertionError, "one hsdp_param"):
            reduce_scatter_copy_in([], [torch.ones(2)], torch.empty(2), world_size=1)

    def test_metadata_hash_is_stable_for_same_layout(self):
        """Metadata hash should be stable for equivalent layouts."""
        first = AllGatherMetadata([[torch.float32]], [[4]], torch.float32, [4], 4)
        second = AllGatherMetadata([[torch.float32]], [[4]], torch.float32, [4], 4)

        self.assertEqual(first.hash_key, second.hash_key)


class TestTorchHSDPParamGroup(unittest.TestCase):
    """Cover HSDPParamGroup buffer and collective orchestration using fakes."""

    def setUp(self):
        param_group_mod.comm_ctx.comm_handle = None
        param_group_mod.comm_ctx.all_reduce_handle = None
        param_group_mod.comm_ctx.pre_param_group = None
        param_group_mod.comm_ctx.all_reduce_param_group = None

    def test_init_mp_dtypes_and_flat_buffer(self):
        """Param groups should initialize uniform dtypes and flat buffers."""
        param_a = _fake_param([1.0, 2.0])
        param_b = _fake_param([3.0, 4.0])
        param_a_obj = param_a.sharded_param
        param_b_obj = param_b.sharded_param
        old_a_ptr = param_a._sharded_param_data.untyped_storage().data_ptr()
        old_b_ptr = param_b._sharded_param_data.untyped_storage().data_ptr()
        group = _new_param_group([param_a, param_b], enable_zero_copy=True)

        HSDPParamGroup._init_mp_dtypes(group)
        HSDPParamGroup._init_flat_param_buffer(group)

        self.assertEqual(group._orig_dtype, torch.float32)
        self.assertTrue(group._is_flat_buffer_valid())
        self.assertEqual(group._flat_param_buffer.numel(), 4)
        self.assertIs(param_a.sharded_param, param_a_obj)
        self.assertIs(param_b.sharded_param, param_b_obj)
        self.assertNotEqual(param_a._sharded_param_data.untyped_storage().data_ptr(), old_a_ptr)
        self.assertNotEqual(param_b._sharded_param_data.untyped_storage().data_ptr(), old_b_ptr)
        self.assertEqual(
            param_a._sharded_param_data.untyped_storage().data_ptr(),
            group._flat_param_buffer.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            param_b._sharded_param_data.untyped_storage().data_ptr(),
            group._flat_param_buffer.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            param_a.sharded_param._local_tensor.untyped_storage().data_ptr(),
            group._flat_param_buffer.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            param_b.sharded_param._local_tensor.untyped_storage().data_ptr(),
            group._flat_param_buffer.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            param_a.sharded_param.data.untyped_storage().data_ptr(),
            group._flat_param_buffer.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            param_b.sharded_param.data.untyped_storage().data_ptr(),
            group._flat_param_buffer.untyped_storage().data_ptr(),
        )

    def test_constructor_resolves_mesh_groups_and_optional_flat_buffer(self):
        """Constructor should derive shard/replicate groups from mesh metadata."""
        fsdp_mesh = _mesh_info(FSDPMeshInfo, shard_size=2)
        hsdp_mesh = _mesh_info(HSDPMeshInfo, shard_size=2, replicate_size=4)
        ddp_mesh = _mesh_info(DDPMeshInfo, shard_size=1, replicate_size=4)

        with patch.object(HSDPParamGroup, "_init_mp_dtypes"), patch.object(
            HSDPParamGroup, "_init_flat_param_buffer"
        ) as init_flat, patch.object(
            HSDPParamGroup, "_infer_layout_replicate_group", return_value="layout-group"
        ):
            fsdp_group = HSDPParamGroup([], fsdp_mesh, device=torch.device("cpu"), enable_zero_copy=True)
            hsdp_group = HSDPParamGroup([], hsdp_mesh, device=torch.device("cpu"), enable_zero_copy=False)
            ddp_group = HSDPParamGroup([], ddp_mesh, device=torch.device("cpu"), enable_zero_copy=False)

        self.assertEqual(fsdp_group.shard_world_size, 2)
        self.assertEqual(fsdp_group.replicate_group, "layout-group")
        self.assertIs(hsdp_group.replicate_group, hsdp_mesh.replicate_process_group)
        self.assertEqual(ddp_group.shard_world_size, 1)
        self.assertIs(ddp_group.replicate_group, ddp_mesh.replicate_process_group)
        init_flat.assert_called_once_with()

    def test_init_mp_dtypes_rejects_mismatch_and_flat_buffer_skips(self):
        """Param groups should reject dtype mismatch and skip unsupported flat buffers."""
        mixed = _new_param_group([_fake_param([1.0]), _fake_param([2.0], dtype=torch.float16)])
        with self.assertRaisesRegex(AssertionError, "uniform original"):
            HSDPParamGroup._init_mp_dtypes(mixed)

        reduce_mismatch = _new_param_group([_fake_param([1.0]), _fake_param([2.0])])
        reduce_mismatch.hsdp_params[1].reduce_dtype = torch.float16
        with self.assertRaisesRegex(AssertionError, "uniform reduce"):
            HSDPParamGroup._init_mp_dtypes(reduce_mismatch)

        frozen = _new_param_group([_fake_param([1.0], requires_grad=False)])
        HSDPParamGroup._init_mp_dtypes(frozen)
        self.assertIsNone(frozen._orig_dtype)
        self.assertIsNone(frozen._reduce_dtype)

        offloaded = _fake_param([1.0])
        offloaded.offload_to_cpu = True
        group = _new_param_group([offloaded], enable_zero_copy=True)
        HSDPParamGroup._init_flat_param_buffer(group)
        self.assertIsNone(group._flat_param_buffer)

        single_rank = _new_param_group([_fake_param([1.0])], world_size=1, enable_zero_copy=True)
        HSDPParamGroup._init_flat_param_buffer(single_rank)
        self.assertIsNone(single_rank._flat_param_buffer)

    def test_flat_buffer_allocates_cast_buffer_when_param_dtype_is_set(self):
        """Zero-copy flat buffer should allocate a cast buffer for mixed param dtype."""
        param = _fake_param([1.0, 2.0])
        param.param_dtype = torch.float16
        group = _new_param_group([param], enable_zero_copy=True)

        HSDPParamGroup._init_flat_param_buffer(group)

        self.assertTrue(group._is_flat_buffer_valid())
        self.assertEqual(group._flat_cast_buffer.dtype, torch.float16)

    def test_replicate_buckets_and_bucket_pack_unpack(self):
        """Replicate buckets should pack, unpack, and update flat reduce output."""
        replica_group = object()
        param_a = _fake_param([1.0, 2.0])
        param_b = _fake_param([3.0, 4.0])
        for param in (param_a, param_b):
            param.unsharded_group_info = GroupInfo("replica", replica_group, 2)
        group = _new_param_group([param_a, param_b])
        group._reduce_output = torch.tensor([1.0, 2.0, 3.0, 4.0])
        group._reduce_hsdp_params = [param_a, param_b]
        group._active_param_flat_offsets = [0, 2]

        buckets = HSDPParamGroup._build_active_replicate_buckets(group, [param_a, param_b])
        bucket = next(iter(buckets.values()))
        group._active_replicate_buckets = buckets
        HSDPParamGroup._allocate_bucket_buffers_if_needed(group, torch.device("cpu"), torch.float32)
        packed = HSDPParamGroup._pack_bucket_from_reduce_output(group, bucket)
        packed.add_(10.0)
        HSDPParamGroup._unpack_bucket_to_reduce_output(group, bucket)

        self.assertEqual(bucket.flat_numel, 4)
        torch.testing.assert_close(group._reduce_output, torch.tensor([11.0, 12.0, 13.0, 14.0]))

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_gather_into_tensor")
    def test_foreach_all_gather_flat_and_fallback_paths(self, mock_all_gather):
        """Foreach all-gather should cover zero-copy and fallback paths."""
        mock_all_gather.return_value = "ag-handle"
        param = _fake_param([1.0, 2.0])
        group = _new_param_group([param], enable_zero_copy=True)
        group._flat_param_buffer = param._sharded_param_data

        HSDPParamGroup.foreach_all_gather(group, async_op=True)

        self.assertEqual(group._result.handle, "ag-handle")
        mock_all_gather.assert_called_once()

        fallback_param = _fake_param([3.0, 4.0])
        fallback = _new_param_group([fallback_param], enable_zero_copy=False)
        mock_all_gather.reset_mock()
        HSDPParamGroup.foreach_all_gather(fallback)
        self.assertIsNotNone(fallback._result)
        mock_all_gather.assert_called_once()

    def test_foreach_all_gather_returns_for_empty_metadata(self):
        """Empty metadata should skip allocation and communication."""
        group = _new_param_group([], enable_zero_copy=False)
        group.metadata_cache = MagicMock()
        group.metadata_cache.get_metadata.return_value = AllGatherMetadata([], [], torch.float32, [], 0)

        HSDPParamGroup.foreach_all_gather(group)

        self.assertIsNone(group.ag_output)
        self.assertIsNone(group._result)

    def test_wait_for_unshard_world_size_one_and_copy_out(self):
        """Wait-for-unshard should cover empty, single-rank, and copy-out paths."""
        empty = _new_param_group([], world_size=2)
        HSDPParamGroup.wait_for_unshard(empty)
        self.assertIsNone(empty._result)

        param = _fake_param([1.0, 2.0])
        group = _new_param_group([param], world_size=1)
        HSDPParamGroup.unshard(group)
        HSDPParamGroup.wait_for_unshard(group)

        param.init_unsharded_param.assert_called_once_with()
        param.to_unsharded.assert_called_once_with()

        param = _fake_param([1.0, 2.0])
        group = _new_param_group([param], world_size=2)
        metadata = get_all_gather_metadata([param])
        handle = MagicMock()
        group.ag_output = torch.arange(4, dtype=torch.float32)
        group._result = AllGatherResult(group.ag_output, metadata, handle)
        HSDPParamGroup.foreach_all_gather_copy_out(group)
        handle.wait.assert_called_once_with()
        self.assertIsNone(group._result)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_foreach_reduce_and_apply_reduced_grad(self, mock_reduce_scatter):
        """Foreach reduce should enqueue reduce-scatter and apply fused gradients."""
        rs_handle = MagicMock()
        mock_reduce_scatter.return_value = rs_handle
        param = _fake_param([1.0, 2.0])
        param.unsharded_param.grad = torch.arange(4, dtype=torch.float32)
        param.unsharded_grad_data = param.unsharded_param.grad
        group = _new_param_group([param], world_size=2)

        HSDPParamGroup.foreach_reduce(group, async_op=True)

        self.assertIs(param_group_mod.comm_ctx.comm_handle, rs_handle)
        self.assertIs(param_group_mod.comm_ctx.pre_param_group, group)
        group._reduce_output.copy_(torch.tensor([5.0, 6.0]))
        HSDPParamGroup.apply_fusion_reduced_grad(group)
        torch.testing.assert_close(param.sharded_param.grad._local_tensor, torch.tensor([2.5, 3.0]))

    def test_foreach_reduce_skips_no_grad_and_rejects_mixed_grad_dtype(self):
        """Fused reduce should skip empty gradients and reject heterogeneous grad dtype."""
        no_grad_param = _fake_param([1.0])
        group = _new_param_group([no_grad_param], world_size=2)
        self.assertIsNone(HSDPParamGroup.foreach_reduce(group))

        param_a = _fake_param([1.0, 2.0])
        param_b = _fake_param([3.0, 4.0])
        param_a.unsharded_param.grad = torch.ones(2, dtype=torch.float32)
        param_b.unsharded_param.grad = torch.ones(2, dtype=torch.float16)
        param_a.unsharded_grad_data = param_a.unsharded_param.grad
        param_b.unsharded_grad_data = param_b.unsharded_param.grad
        group = _new_param_group([param_a, param_b], world_size=2)

        with self.assertRaisesRegex(ValueError, "uniform grad dtype"):
            HSDPParamGroup.foreach_reduce(group)

    def test_wait_reduce_scatter_without_replicate_buckets_applies_immediately(self):
        """FSDP-only fused reduce should wait reduce-scatter and then apply grads."""
        group = _new_param_group([_fake_param([1.0, 2.0])], world_size=2)
        group._needs_avg_div = True
        group._reduce_output = torch.tensor([4.0, 8.0])
        group._active_replicate_buckets = {}
        group._apply_reduced_grad = MagicMock()
        handle = MagicMock()
        param_group_mod.comm_ctx.comm_handle = handle

        HSDPParamGroup.wait_reduce_scatter_and_issue_all_reduce(group)

        handle.wait.assert_called_once_with()
        torch.testing.assert_close(group._reduce_output, torch.tensor([2.0, 4.0]))
        group._apply_reduced_grad.assert_called_once_with()
        self.assertIsNone(param_group_mod.comm_ctx.comm_handle)

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.dist.all_reduce")
    def test_wait_all_reduce_and_main_grad_path(self, mock_all_reduce):
        """All-reduce wait and fused apply should support fp32 main grad."""
        bucket_group = object()
        param = _fake_param([1.0, 2.0])
        param.unsharded_group_info = GroupInfo("replica", bucket_group, 2)
        param.sharded_param.main_grad = SimpleNamespace(_local_tensor=torch.ones(2))
        group = _new_param_group([param], world_size=2)
        group.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        group._reduce_output = torch.tensor([2.0, 4.0])
        group._reduce_hsdp_params = [param]
        group._active_param_flat_offsets = [0]
        group._active_replicate_buckets = HSDPParamGroup._build_active_replicate_buckets(group, [param])
        HSDPParamGroup._allocate_bucket_buffers_if_needed(group, torch.device("cpu"), torch.float32)

        HSDPParamGroup.apply_fusion_reduced_grad(group)

        mock_all_reduce.assert_called_once()
        torch.testing.assert_close(param.sharded_param.main_grad._local_tensor, torch.tensor([3.0, 5.0]))
        self.assertIsNone(param.sharded_param.grad)

    def test_wait_all_reduce_and_apply_grad_waits_pending_bucket(self):
        """Deferred replicate all-reduce should wait, unpack, apply, and clear handles."""
        group = _new_param_group([_fake_param([1.0, 2.0])], world_size=2)
        bucket = ReplicateBucket(1, object(), 2, [0], 2, buffer=torch.tensor([4.0, 8.0]))
        handle = MagicMock()
        group._needs_avg_div = True
        group._active_replicate_buckets = {1: bucket}
        group._pending_all_reduce_handles = [PendingBucketAllReduce(1, handle)]
        group._unpack_bucket_to_reduce_output = MagicMock()
        group._apply_reduced_grad = MagicMock()

        HSDPParamGroup.wait_all_reduce_and_apply_grad(group)

        handle.wait.assert_called_once_with()
        torch.testing.assert_close(bucket.buffer, torch.tensor([2.0, 4.0]))
        group._unpack_bucket_to_reduce_output.assert_called_once_with(bucket)
        group._apply_reduced_grad.assert_called_once_with()
        self.assertEqual(group._pending_all_reduce_handles, [])

    @patch("hyper_parallel.platform.torch.fully_shard.param_group.torch.cuda.current_stream")
    def test_apply_reduced_grad_offload_cuda_syncs_after_cpu_transfer(self, mock_current_stream):
        """CPU offload should synchronize the owning CUDA stream after grad transfer."""
        param = _fake_param([1.0, 2.0])
        param.offload_to_cpu = True
        param.pin_memory = False
        group = _new_param_group([param], world_size=2)
        group.device = torch.device("cuda")
        group._reduce_output = torch.tensor([5.0, 6.0])
        group._reduce_hsdp_params = [param]
        stream = MagicMock()
        mock_current_stream.return_value = stream

        HSDPParamGroup._apply_reduced_grad(group)

        stream.synchronize.assert_called_once_with()
        self.assertIsNone(group._reduce_output)


if __name__ == "__main__":
    unittest.main()

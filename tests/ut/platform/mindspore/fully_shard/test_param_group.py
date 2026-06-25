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
"""Unit tests for MindSpore fully_shard parameter groups."""
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

from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, HSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard import param_group as param_group_mod
from hyper_parallel.platform.mindspore.fully_shard.param_group import (
    AllGatherMetadata,
    AllGatherMetadataCache,
    AllGatherResult,
    HSDPParamGroup,
    PendingBucketAllReduce,
    ReplicateBucket,
    _normalize_device,
    _shape_numel,
    get_all_gather_metadata,
    reduce_scatter_copy_in,
    split_with_sizes_copy,
)


def _mesh_info(mesh_cls, *, shard_size=2, replicate_size=1):
    """Create a mesh-info object without constructing a real DeviceMesh."""
    mesh_info = object.__new__(mesh_cls)
    mesh_info.shard_mesh_rank = 0
    mesh_info.shard_mesh_size = shard_size
    mesh_info.shard_process_group = "shard-group"
    mesh_info.replicate_mesh_rank = 0
    mesh_info.replicate_mesh_size = replicate_size
    mesh_info.replicate_process_group = "replicate-group"
    return mesh_info


def _new_param_group():
    """Create a bare param group for direct method tests."""
    group = object.__new__(HSDPParamGroup)
    group.hsdp_params = []
    group.mesh_info = _mesh_info(FSDPMeshInfo)
    group.device = "Ascend:0"
    group.enable_zero_copy_param_buffer = False
    group.shard_rank = 0
    group.shard_world_size = 2
    group.shard_group = "shard-group"
    group.replicate_group = None
    group.ag_output = None
    group.metadata_cache = None
    group.mp_policy = None
    group._result = None
    group._reduce_output = None
    group._reduce_op = None
    group._needs_avg_div = False
    group._reduce_hsdp_params = None
    group._active_replicate_buckets = {}
    group._active_param_flat_offsets = []
    group._pending_all_reduce_handles = []
    group._flat_param_buffer = None
    group._flat_cast_buffer = None
    group._orig_dtype = ms.float32
    group._reduce_dtype = ms.float32
    group.gradient_scaling_factor = None
    return group


def _fake_hsdp_param(name="param", *, dtype=ms.float32, requires_grad=True, shard_size=(2, 2)):
    """Build a lightweight parameter double used by HSDPParamGroup tests."""
    param = MagicMock(name=name)
    param.version = 0
    param.sharded_size = shard_size
    param.sharded_param.requires_grad = requires_grad
    param.sharded_param.device = "Ascend:0"
    param.sharded_param.grad = None
    param.orig_dtype = dtype
    param.reduce_dtype = dtype
    param.param_dtype = None
    param.hsdp_placement = SimpleNamespace(dim=0)
    param._orig_size = None
    param.offload_to_cpu = False
    param.all_gather_inputs = [ms.Tensor(np.arange(4, dtype=np.float32))]
    param._sharded_param_data = ms.Tensor(np.arange(4, dtype=np.float32))
    param._sharded_local_tensor = ms.Tensor(np.arange(4, dtype=np.float32).reshape(2, 2))
    param.unsharded_accumulated_grad = None
    param._unsharded_param = SimpleNamespace(grad=None)
    param.unsharded_grad_data = ms.Tensor(np.ones((2, 2), dtype=np.float32))
    param.unsharded_accumulated_grad_data = None
    param.unsharded_group_info = SimpleNamespace(group=None, rank_size=1)
    param.init_dtype_attrs.side_effect = lambda policy: None
    return param


class TestMindSporeParamGroupHelpers(unittest.TestCase):
    """Cover standalone helpers used by fused communication."""

    def test_normalize_device_accepts_string_and_device_like_values(self):
        """Device values should be normalized to the backend name."""
        self.assertEqual(_normalize_device("Ascend:0"), "Ascend")
        self.assertEqual(_normalize_device(0), "0")

    def test_shape_numel_multiplies_dimensions(self):
        """Shape numel should multiply integer-like dimensions."""
        self.assertEqual(_shape_numel((2, 3, 4)), 24)

    def test_get_all_gather_metadata_collects_uniform_dtype_layout(self):
        """Metadata should flatten per-param all-gather input dtypes and sizes."""
        param_a = SimpleNamespace(
            version=1,
            all_gather_inputs=[
                ms.Tensor(np.ones((2,), dtype=np.float32)),
                ms.Tensor(np.ones((3,), dtype=np.float32)),
            ],
        )
        param_b = SimpleNamespace(
            version=2,
            all_gather_inputs=[ms.Tensor(np.ones((4,), dtype=np.float32))],
        )

        metadata = get_all_gather_metadata([param_a, param_b])

        self.assertEqual(metadata.param_input_numels, [[2, 3], [4]])
        self.assertEqual(metadata.inp_split_sizes, [2, 3, 4])
        self.assertEqual(metadata.total_input_numel, 9)
        self.assertEqual(metadata.dtype, ms.float32)

    def test_get_all_gather_metadata_rejects_mixed_first_input_dtype(self):
        """Fused all-gather requires a uniform first-input dtype across params."""
        param_a = SimpleNamespace(all_gather_inputs=[ms.Tensor(np.ones((2,), dtype=np.float32))])
        param_b = SimpleNamespace(all_gather_inputs=[ms.Tensor(np.ones((2,), dtype=np.float16))])

        with self.assertRaisesRegex(ValueError, "uniform dtype"):
            get_all_gather_metadata([param_a, param_b])

    def test_all_gather_metadata_hash_changes_with_layout(self):
        """Metadata hashes should include dtypes, split sizes, and total numel."""
        metadata_a = AllGatherMetadata([[ms.float32]], [[2]], ms.float32, [2], 2)
        metadata_b = AllGatherMetadata([[ms.float32]], [[4]], ms.float32, [4], 4)

        self.assertNotEqual(metadata_a.hash_key, metadata_b.hash_key)

    def test_all_gather_metadata_cache_reuses_unchanged_param_versions(self):
        """Cache lookup should reuse metadata until a parameter version changes."""
        AllGatherMetadataCache._cache.clear()
        param = SimpleNamespace(version=1)
        metadata = AllGatherMetadata([[ms.float32]], [[2]], ms.float32, [2], 2)
        builder = MagicMock(return_value=metadata)

        first = AllGatherMetadataCache.get_metadata([param], builder)
        second = AllGatherMetadataCache.get_metadata([param], builder)
        param.version = 2
        third = AllGatherMetadataCache.get_metadata([param], builder)

        self.assertIs(first, metadata)
        self.assertIs(second, metadata)
        self.assertIs(third, metadata)
        self.assertEqual(builder.call_count, 2)

    def test_split_with_sizes_copy_supports_dim_one_only(self):
        """split_with_sizes_copy should copy dim-1 slices to the provided outputs."""
        all_gather_output = ms.Tensor(np.arange(10, dtype=np.float32).reshape(2, 5))
        dst_a = ms.Tensor(np.zeros((2, 2), dtype=np.float32))
        dst_b = ms.Tensor(np.zeros((2, 3), dtype=np.float32))

        split_with_sizes_copy(all_gather_output, [2, 3], 1, [dst_a, dst_b])

        np.testing.assert_allclose(dst_a.asnumpy(), np.array([[0.0, 1.0], [5.0, 6.0]], dtype=np.float32))
        np.testing.assert_allclose(dst_b.asnumpy(), np.array([[2.0, 3.0, 4.0], [7.0, 8.0, 9.0]], dtype=np.float32))
        with self.assertRaisesRegex(NotImplementedError, "dim=1"):
            split_with_sizes_copy(all_gather_output, [2], 0, [dst_a])

    def test_reduce_scatter_copy_in_packs_each_grad_into_row_major_buffer(self):
        """reduce_scatter_copy_in should place each packed grad in the fused input buffer."""
        hsdp_param = SimpleNamespace(hsdp_placement=SimpleNamespace(dim=0))
        grad = ms.Tensor(np.arange(8, dtype=np.float32).reshape(4, 2))
        reduce_scatter_input = ms.Tensor(np.zeros((8,), dtype=np.float32))

        reduce_scatter_copy_in([hsdp_param], [grad], reduce_scatter_input, world_size=2)

        np.testing.assert_allclose(
            reduce_scatter_input.asnumpy().reshape(2, 4),
            grad.asnumpy().reshape(2, 4),
        )

    def test_reduce_scatter_copy_in_rejects_mismatched_param_and_grad_counts(self):
        """Each fused reduce-scatter param needs one matching unsharded grad."""
        with self.assertRaisesRegex(AssertionError, "one hsdp_param per unsharded_grad"):
            reduce_scatter_copy_in([MagicMock()], [], MagicMock(), world_size=2)


class TestMindSporeParamGroup(unittest.TestCase):
    """Cover HSDPParamGroup fused communication state transitions."""

    def test_init_resolves_fsdp_and_hsdp_mesh_fields(self):
        """Constructor should derive shard and replicate groups from mesh info."""
        fsdp_mesh = _mesh_info(FSDPMeshInfo, shard_size=2)
        hsdp_mesh = _mesh_info(HSDPMeshInfo, shard_size=2, replicate_size=4)

        with patch.object(HSDPParamGroup, "_init_mp_dtypes"), patch.object(
            HSDPParamGroup, "_infer_layout_replicate_group", return_value="layout-replicate-group"
        ):
            fsdp_group = HSDPParamGroup([], fsdp_mesh)
            hsdp_group = HSDPParamGroup([], hsdp_mesh)

        self.assertEqual(fsdp_group.shard_world_size, 2)
        self.assertEqual(fsdp_group.replicate_group, "layout-replicate-group")
        self.assertEqual(hsdp_group.replicate_group, "replicate-group")

    def test_infer_layout_replicate_group_uses_first_multi_rank_group(self):
        """Layout-driven FSDP params can provide a separate replicate group."""
        group = _new_param_group()
        group.hsdp_params = [
            SimpleNamespace(unsharded_group_info=SimpleNamespace(group=None, rank_size=1), replicate_world_size=1),
            SimpleNamespace(unsharded_group_info=SimpleNamespace(group="replicate-a", rank_size=2), replicate_world_size=2),
            SimpleNamespace(unsharded_group_info=SimpleNamespace(group="replicate-b", rank_size=2), replicate_world_size=2),
        ]

        self.assertEqual(HSDPParamGroup._infer_layout_replicate_group(group), "replicate-a")

    def test_build_active_replicate_buckets_groups_params_by_process_group(self):
        """Active all-reduce buckets should group params by their replicate process group."""
        params = [
            SimpleNamespace(unsharded_group_info=SimpleNamespace(group="rep-a", rank_size=2), replicate_world_size=2, sharded_size=(2,)),
            SimpleNamespace(unsharded_group_info=SimpleNamespace(group="rep-a", rank_size=2), replicate_world_size=2, sharded_size=(3,)),
            SimpleNamespace(unsharded_group_info=SimpleNamespace(group=None, rank_size=1), replicate_world_size=1, sharded_size=(5,)),
        ]

        buckets = HSDPParamGroup._build_active_replicate_buckets(params)
        bucket = next(iter(buckets.values()))

        self.assertEqual(bucket.param_indices, [0, 1])
        self.assertEqual(bucket.flat_numel, 5)

    def test_init_mp_dtypes_rejects_mixed_trainable_orig_and_reduce_dtypes(self):
        """Fused groups require uniform original and reduce dtypes among trainable params."""
        group = _new_param_group()
        group.mp_policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float32)
        param_a = _fake_hsdp_param(dtype=ms.float32)
        param_b = _fake_hsdp_param(dtype=ms.float16)
        param_a.init_dtype_attrs.side_effect = lambda policy: None
        param_b.init_dtype_attrs.side_effect = lambda policy: None
        group.hsdp_params = [param_a, param_b]

        with self.assertRaisesRegex(AssertionError, "uniform original parameter dtype"):
            HSDPParamGroup._init_mp_dtypes(group)

    def test_init_mp_dtypes_ignores_frozen_params(self):
        """Frozen params should not determine fused reduce dtype metadata."""
        group = _new_param_group()
        group.mp_policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float32)
        group.hsdp_params = [_fake_hsdp_param(requires_grad=False)]

        HSDPParamGroup._init_mp_dtypes(group)

        self.assertIsNone(group._orig_dtype)
        self.assertIsNone(group._reduce_dtype)

    def test_init_flat_param_buffer_exits_when_disabled_or_trivial(self):
        """Flat param buffer setup should be skipped when it is not useful."""
        group = _new_param_group()
        group.enable_zero_copy_param_buffer = False
        HSDPParamGroup._init_flat_param_buffer(group)
        self.assertIsNone(group._flat_param_buffer)

        group.enable_zero_copy_param_buffer = True
        group.shard_world_size = 1
        HSDPParamGroup._init_flat_param_buffer(group)
        self.assertIsNone(group._flat_param_buffer)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.ms.mint.empty")
    def test_init_flat_param_buffer_rolls_back_on_rebase_failure(self, mock_empty):
        """Flat-buffer rebase failures should restore each param's original local tensor."""
        group = _new_param_group()
        group.enable_zero_copy_param_buffer = True
        group.hsdp_params = [_fake_hsdp_param()]
        flat_buffer = MagicMock()
        flat_buffer.narrow.return_value.copy_.side_effect = RuntimeError("copy failed")
        mock_empty.return_value = flat_buffer

        HSDPParamGroup._init_flat_param_buffer(group)

        self.assertIsNone(group._flat_param_buffer)
        self.assertIsNone(group._flat_cast_buffer)

    def test_is_flat_buffer_valid_compares_storage_pointers(self):
        """Flat buffer validity should be based on shared storage identity."""
        group = _new_param_group()
        param = _fake_hsdp_param()
        group.hsdp_params = [param]
        group._flat_param_buffer = MagicMock()
        param._sharded_param_data = MagicMock()
        param._sharded_param_data.untyped_storage().data_ptr.return_value = 7
        group._flat_param_buffer.untyped_storage().data_ptr.return_value = 7

        self.assertTrue(HSDPParamGroup._is_flat_buffer_valid(group))

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.ms.mint.empty")
    def test_allocate_bucket_buffers_if_needed_allocates_missing_bucket_buffer(self, mock_empty):
        """Replicate buckets should allocate buffers for their flat shard payloads."""
        group = _new_param_group()
        group._active_replicate_buckets = {1: ReplicateBucket(1, "pg", 2, [0], 4)}
        mock_empty.return_value = "bucket-buffer"

        HSDPParamGroup._allocate_bucket_buffers_if_needed(group, "Ascend:0", ms.float32)

        mock_empty.assert_called_once_with((4,), dtype=ms.float32, device="Ascend")
        self.assertEqual(group._active_replicate_buckets[1].buffer, "bucket-buffer")

    def test_unshard_handles_idempotent_and_single_rank_paths(self):
        """unshard should no-op when active and synthesize a local result for world_size=1."""
        group = _new_param_group()
        group._result = "active"
        HSDPParamGroup.unshard(group)
        self.assertEqual(group._result, "active")

        group._result = None
        group.shard_world_size = 1
        HSDPParamGroup.unshard(group)
        self.assertEqual(group._result, AllGatherResult(None, None, None))

    def test_wait_for_unshard_single_rank_materializes_params(self):
        """Single-rank wait_for_unshard should copy local inputs and switch params to unsharded."""
        group = _new_param_group()
        hsdp_param = _fake_hsdp_param()
        group.hsdp_params = [hsdp_param]
        group.shard_world_size = 1
        group._result = AllGatherResult(None, None, None)

        HSDPParamGroup.wait_for_unshard(group)

        hsdp_param.init_all_gather_outputs.assert_called_once()
        hsdp_param.alloc_all_gather_outputs.assert_called_once()
        hsdp_param.init_unsharded_param.assert_called_once()
        hsdp_param.to_unsharded.assert_called_once()
        self.assertIsNone(group._result)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.ms.mint.empty")
    def test_alloc_all_gather_output_allocates_or_resizes_storage(self, mock_empty):
        """All-gather output allocation should reuse dtype-compatible buffers by resizing storage."""
        group = _new_param_group()
        mock_empty.return_value = MagicMock(dtype=ms.float32)

        HSDPParamGroup.alloc_all_gather_output(group, 8, ms.float32)

        mock_empty.assert_called_once_with((8,), dtype=ms.float32, device="Ascend")

        storage = MagicMock()
        storage.size.return_value = 4
        group.ag_output.untyped_storage.return_value = storage
        group.ag_output.itemsize = 4
        HSDPParamGroup.alloc_all_gather_output(group, 8, ms.float32)
        storage.resize_.assert_called_once_with(32)

    def test_free_all_gather_output_releases_storage(self):
        """free_all_gather_output should shrink cached fused output storage."""
        group = _new_param_group()
        storage = MagicMock()
        storage.size.return_value = 8
        group.ag_output = MagicMock()
        group.ag_output.untyped_storage.return_value = storage

        HSDPParamGroup.free_all_gather_output(group)

        storage.resize_.assert_called_once_with(0)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.all_gather_into_tensor")
    def test_foreach_all_gather_uses_copy_in_path_and_records_result(self, mock_all_gather):
        """foreach_all_gather should pack local shards and launch the fused all-gather."""
        group = _new_param_group()
        group.hsdp_params = [_fake_hsdp_param()]
        group.alloc_all_gather_output = MagicMock(side_effect=lambda total, dtype: setattr(group, "ag_output", MagicMock()))
        mock_all_gather.return_value = "ag-handle"

        HSDPParamGroup.foreach_all_gather(group, async_op=True)

        group.hsdp_params[0].reset_sharded_param.assert_called_once()
        mock_all_gather.assert_called_once()
        self.assertEqual(group._result.handle, "ag-handle")

    def test_foreach_all_gather_returns_when_metadata_is_empty(self):
        """Empty metadata should skip fused all-gather launch."""
        group = _new_param_group()
        group.hsdp_params = []
        group.metadata_cache = MagicMock()
        group.metadata_cache.get_metadata.return_value = AllGatherMetadata([], [], ms.float32, [], 0)

        HSDPParamGroup.foreach_all_gather(group)

        self.assertIsNone(group._result)

    def test_foreach_all_gather_copy_out_waits_and_splits_outputs(self):
        """foreach_all_gather_copy_out should wait, allocate outputs, and free the fused buffer."""
        group = _new_param_group()
        hsdp_param = _fake_hsdp_param()
        group.hsdp_params = [hsdp_param]
        metadata = AllGatherMetadata([[ms.float32]], [[4]], ms.float32, [4], 4)
        handle = MagicMock()
        ag_output = MagicMock()
        ag_output.device = "Ascend:0"
        ag_output.view.return_value = "viewed-ag"
        group._result = AllGatherResult(ag_output, metadata, handle)
        group.free_all_gather_output = MagicMock()

        HSDPParamGroup.foreach_all_gather_copy_out(group)

        handle.wait.assert_called_once()
        hsdp_param.init_all_gather_outputs.assert_called_once()
        hsdp_param.alloc_all_gather_outputs.assert_called_once()
        group.free_all_gather_output.assert_called_once()
        self.assertIsNone(group._result)

    def test_foreach_reduce_returns_none_without_grads_and_rejects_mixed_grad_dtypes(self):
        """foreach_reduce should skip empty grads and reject mixed grad dtypes."""
        group = _new_param_group()
        group.hsdp_params = [_fake_hsdp_param()]
        group.hsdp_params[0]._unsharded_param = SimpleNamespace(grad=None)
        self.assertIsNone(HSDPParamGroup.foreach_reduce(group))

        param_a = _fake_hsdp_param()
        param_b = _fake_hsdp_param()
        param_a._unsharded_param = SimpleNamespace(grad="grad")
        param_b._unsharded_param = SimpleNamespace(grad="grad")
        param_a.unsharded_grad_data = ms.Tensor(np.ones((2, 2), dtype=np.float32))
        param_b.unsharded_grad_data = ms.Tensor(np.ones((2, 2), dtype=np.float16))
        group.hsdp_params = [param_a, param_b]
        with self.assertRaisesRegex(ValueError, "uniform grad dtype"):
            HSDPParamGroup.foreach_reduce(group)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.reduce_scatter_tensor")
    def test_foreach_reduce_launches_reduce_scatter_and_records_state(self, mock_reduce_scatter):
        """foreach_reduce should pack grads and launch reduce-scatter when a shard group exists."""
        group = _new_param_group()
        hsdp_param = _fake_hsdp_param()
        hsdp_param._unsharded_param = SimpleNamespace(grad="grad")
        group.hsdp_params = [hsdp_param]
        mock_reduce_scatter.return_value = "rs-handle"

        reduce_output = HSDPParamGroup.foreach_reduce(group, async_op=True)

        mock_reduce_scatter.assert_called_once()
        self.assertIs(reduce_output, group._reduce_output)
        self.assertEqual(param_group_mod.comm_ctx.comm_handle, "rs-handle")
        self.assertIs(param_group_mod.comm_ctx.pre_param_group, group)

    def test_wait_reduce_scatter_and_issue_all_reduce_applies_when_no_buckets(self):
        """No active replicate buckets should apply reduced grads immediately."""
        group = _new_param_group()
        group._reduce_output = MagicMock()
        group._apply_reduced_grad = MagicMock()
        param_group_mod.comm_ctx.comm_handle = MagicMock()

        HSDPParamGroup.wait_reduce_scatter_and_issue_all_reduce(group)

        group._apply_reduced_grad.assert_called_once()
        self.assertIsNone(param_group_mod.comm_ctx.comm_handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.all_reduce")
    def test_wait_reduce_scatter_and_issue_all_reduce_launches_bucket_all_reduces(self, mock_all_reduce):
        """Active replicate buckets should launch async all-reduces after reduce-scatter."""
        group = _new_param_group()
        bucket = ReplicateBucket(1, "replicate-group", 2, [0], 2, buffer=MagicMock())
        group._active_replicate_buckets = {1: bucket}
        group._pack_bucket_from_reduce_output = MagicMock(return_value=bucket.buffer)
        mock_all_reduce.return_value = "ar-handle"

        HSDPParamGroup.wait_reduce_scatter_and_issue_all_reduce(group)

        mock_all_reduce.assert_called_once_with(bucket.buffer, group="replicate-group", op=None, async_op=True)
        self.assertEqual(group._pending_all_reduce_handles, [PendingBucketAllReduce(1, "ar-handle")])
        self.assertIs(param_group_mod.comm_ctx.all_reduce_param_group, group)

    def test_wait_all_reduce_and_apply_grad_waits_and_unpacks_each_bucket(self):
        """Pending all-reduces should wait before bucket data is unpacked and applied."""
        group = _new_param_group()
        bucket = ReplicateBucket(1, "replicate-group", 2, [0], 2, buffer=ms.Tensor(np.ones((2,), dtype=np.float32)))
        handle = MagicMock()
        group._active_replicate_buckets = {1: bucket}
        group._pending_all_reduce_handles = [PendingBucketAllReduce(1, handle)]
        group._needs_avg_div = True
        group._unpack_bucket_to_reduce_output = MagicMock()
        group._apply_reduced_grad = MagicMock()

        HSDPParamGroup.wait_all_reduce_and_apply_grad(group)

        handle.wait.assert_called_once()
        group._unpack_bucket_to_reduce_output.assert_called_once_with(bucket)
        group._apply_reduced_grad.assert_called_once()
        self.assertEqual(group._pending_all_reduce_handles, [])

    @patch("hyper_parallel.platform.mindspore.fully_shard.param_group.dist.all_reduce")
    def test_apply_fusion_reduced_grad_runs_sync_bucket_all_reduce(self, mock_all_reduce):
        """Synchronous fused reduce should all-reduce buckets before applying grads."""
        group = _new_param_group()
        bucket = ReplicateBucket(1, "replicate-group", 2, [0], 2, buffer=ms.Tensor(np.ones((2,), dtype=np.float32)))
        group._active_replicate_buckets = {1: bucket}
        group._needs_avg_div = True
        group._pack_bucket_from_reduce_output = MagicMock(return_value=bucket.buffer)
        group._unpack_bucket_to_reduce_output = MagicMock()
        group._apply_reduced_grad = MagicMock()
        param_group_mod.comm_ctx.comm_handle = MagicMock()

        HSDPParamGroup.apply_fusion_reduced_grad(group)

        mock_all_reduce.assert_called_once()
        group._unpack_bucket_to_reduce_output.assert_called_once_with(bucket)
        group._apply_reduced_grad.assert_called_once()

    def test_apply_reduced_grad_writes_each_flat_grad_slice_to_param(self):
        """Reduced flat gradients should be applied to params and then clear group state."""
        group = _new_param_group()
        first_param = SimpleNamespace(sharded_size=(2,), apply_reduced_grad=MagicMock())
        second_param = SimpleNamespace(sharded_size=(3,), apply_reduced_grad=MagicMock())
        group._reduce_hsdp_params = [first_param, second_param]
        group._reduce_output = ms.Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))

        HSDPParamGroup._apply_reduced_grad(group)

        first_grad = first_param.apply_reduced_grad.call_args.args[0]
        second_grad = second_param.apply_reduced_grad.call_args.args[0]
        np.testing.assert_allclose(first_grad.asnumpy(), np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_allclose(second_grad.asnumpy(), np.array([3.0, 4.0, 5.0], dtype=np.float32))
        self.assertEqual(first_param.apply_reduced_grad.call_args.args[1], ms.float32)
        self.assertEqual(second_param.apply_reduced_grad.call_args.args[1], ms.float32)
        self.assertIsNone(group._reduce_output)
        self.assertIsNone(group._reduce_hsdp_params)
        self.assertEqual(group._active_replicate_buckets, {})


if __name__ == "__main__":
    unittest.main()

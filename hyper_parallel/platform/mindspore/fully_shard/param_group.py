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
"""MindSpore HSDP parameter group with fused communication."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, NamedTuple, Optional

import mindspore as ms
from mindspore import ops
from mindspore.common.api import _no_grad
import mindspore.mint.distributed as dist
from mindspore.ops.function.comm_func import CommHandle

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_utils import apply_gradient_scaling_factor
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, HSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import build_rs_plan, pack_for_reduce_scatter
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2


def _normalize_device(device: Any) -> str:
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device).split(":", 1)[0]


def _shape_numel(shape) -> int:
    return math.prod(int(dim) for dim in shape)


def get_all_gather_metadata(hsdp_params):
    """Collect metadata required for fused all-gather."""
    param_input_dtypes = []
    param_input_numels = []
    inp_split_sizes = []
    total_input_numel = 0
    first_dtype = None

    for hsdp_param in hsdp_params:
        inputs = hsdp_param.all_gather_inputs
        if first_dtype is None:
            first_dtype = inputs[0].dtype
        elif first_dtype != inputs[0].dtype:
            raise ValueError("All parameters in the group must have a uniform dtype.")
        param_dtypes = [t.dtype for t in inputs]
        param_numels = [t.numel() for t in inputs]
        param_input_dtypes.append(param_dtypes)
        param_input_numels.append(param_numels)
        inp_split_sizes.extend(param_numels)
        total_input_numel += sum(param_numels)

    return AllGatherMetadata(
        param_input_dtypes,
        param_input_numels,
        first_dtype,
        inp_split_sizes,
        total_input_numel,
    )


@dataclass
class AllGatherMetadata:
    """Metadata describing the fused all-gather buffer layout."""

    param_input_dtypes: list[list[Any]]
    param_input_numels: list[list[int]]
    dtype: Any
    inp_split_sizes: list[int]
    total_input_numel: int
    hash_key: int = field(init=False)

    def __post_init__(self):
        self.hash_key = hash(
            (
                tuple(tuple(d) for d in self.param_input_dtypes),
                tuple(tuple(n) for n in self.param_input_numels),
                self.dtype,
                tuple(self.inp_split_sizes),
                self.total_input_numel,
            )
        )


class AllGatherResult(NamedTuple):
    """Result of a fused all-gather operation."""

    all_gather_output: Optional[ms.Tensor]
    metadata: Optional[AllGatherMetadata]
    handle: Optional[CommHandle]


@dataclass
class CommContext:
    """Global communication context for pipelined fused reductions."""

    comm_handle: Optional[CommHandle] = None
    all_reduce_handle: Optional[CommHandle] = None
    pre_param_group = None
    all_reduce_param_group = None


comm_ctx = CommContext()


def get_comm_ctx():
    """Return the global communication context singleton."""
    return comm_ctx


@dataclass
class ReplicateBucket:
    """One fused all-reduce bucket sharing the same replicate process group."""

    key: int
    group: Any
    group_size: int
    param_indices: list[int]
    flat_numel: int
    buffer: Optional[ms.Tensor] = None


@dataclass
class PendingBucketAllReduce:
    """One in-flight async all-reduce launched for a replicate bucket."""

    bucket_key: int
    handle: Any


class AllGatherMetadataCache:
    """Cache for all-gather metadata across iterations."""

    _cache: dict[int, AllGatherMetadata] = {}

    @classmethod
    def get_metadata(cls, hsdp_params, fn):
        """Retrieve or compute all-gather metadata, caching the result by parameter identity."""
        param_key = tuple((id(p), getattr(p, "version", 0)) for p in hsdp_params)
        key = hash(param_key)
        if key in cls._cache:
            return cls._cache[key]
        metadata = fn(hsdp_params)
        cls._cache[key] = metadata
        return metadata


@_no_grad()
def all_gather_copy_in(all_gather_inputs, all_gather_output, inp_split_sizes, all_gather_input_numel, rank):
    """Copy per-parameter local shards into one fused rank-local all-gather slice."""
    all_gather_input = all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)
    offset = 0
    for src, size in zip(all_gather_inputs, inp_split_sizes):
        src_flat = src.view(-1)
        all_gather_input.narrow(0, offset, size).copy_(src_flat)
        offset += size
    return all_gather_input, all_gather_output


@_no_grad()
def split_with_sizes_copy(all_gather_output, split_sizes, dim, out):
    """Copy split views from a fused all-gather output into pre-allocated outputs."""
    if dim != 1:
        raise NotImplementedError("split_with_sizes_copy currently only supports dim=1")
    offset = 0
    for dst, size in zip(out, split_sizes):
        src = all_gather_output.narrow(dim, offset, size)
        copy_without_bumping_version(dst, src)
        offset += size


@_no_grad()
def reduce_scatter_copy_in(
    hsdp_params: List[MindSporeHSDPParamV2],
    unsharded_grads: List[ms.Tensor],
    reduce_scatter_input: ms.Tensor,
    world_size: int,
) -> None:
    """Pack all unsharded gradients into one fused reduce-scatter input buffer."""
    if len(hsdp_params) != len(unsharded_grads):
        raise AssertionError(
            "reduce_scatter_copy_in expects one hsdp_param per unsharded_grad, but got "
            f"{len(hsdp_params)} params and {len(unsharded_grads)} grads"
        )
    packed_rows = reduce_scatter_input.view(world_size, -1)
    col_offset = 0
    for hsdp_param, grad in zip(hsdp_params, unsharded_grads):
        grad = grad.contiguous()
        plan = build_rs_plan(hsdp_param, grad, world_size)
        packed_grad = pack_for_reduce_scatter(grad, plan)
        next_col_offset = col_offset + packed_grad.shape[1]
        for row_idx in range(world_size):
            packed_rows[row_idx].narrow(0, col_offset, packed_grad.shape[1]).copy_(
                packed_grad[row_idx].view(-1)
            )
        col_offset = next_col_offset
    if col_offset != packed_rows.shape[1]:
        raise AssertionError(
            "reduce_scatter_copy_in packed an unexpected number of elements: "
            f"{col_offset} != {packed_rows.shape[1]}"
        )


class HSDPParamGroup:
    """Group HSDP parameters within a module for fused collectives."""

    def __init__(
        self,
        hsdp_params,
        mesh_info: FSDPMeshInfo,
        device: Optional[str] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        enable_zero_copy_param_buffer: bool = False,
    ):
        self.mesh_info = mesh_info
        self.device = device
        self.hsdp_params = hsdp_params
        self.enable_zero_copy_param_buffer = enable_zero_copy_param_buffer
        if isinstance(self.mesh_info, (FSDPMeshInfo, HSDPMeshInfo)):
            self.shard_rank = self.mesh_info.shard_mesh_rank
            self.shard_world_size = self.mesh_info.shard_mesh_size
        else:
            self.shard_rank = 0
            self.shard_world_size = 1
        self.shard_group = self.mesh_info.shard_process_group
        self.replicate_group = None
        if isinstance(self.mesh_info, (HSDPMeshInfo, DDPMeshInfo)):
            self.replicate_group = self.mesh_info.replicate_process_group
        elif isinstance(self.mesh_info, FSDPMeshInfo):
            self.replicate_group = self._infer_layout_replicate_group()
        self.ag_output: Optional[ms.Tensor] = None
        self.metadata_cache = None
        self.mp_policy = mp_policy
        self._result = None
        self._reduce_output = None
        self._reduce_op = None
        self._reduce_hsdp_params = None
        self._active_replicate_buckets: dict[int, ReplicateBucket] = {}
        self._active_param_flat_offsets: list[int] = []
        self._pending_all_reduce_handles: list[PendingBucketAllReduce] = []
        self._flat_param_buffer: Optional[ms.Tensor] = None
        self._flat_cast_buffer: Optional[ms.Tensor] = None
        self._init_mp_dtypes()
        if self.enable_zero_copy_param_buffer:
            self._init_flat_param_buffer()
        self.gradient_scaling_factor = None

    def _infer_layout_replicate_group(self):
        replicate_groups = []
        for hsdp_param in self.hsdp_params:
            group_info = getattr(hsdp_param, "unsharded_group_info", None)
            group = getattr(group_info, "group", None)
            if group is None or getattr(hsdp_param, "replicate_world_size", 1) <= 1:
                continue
            replicate_groups.append(group)
        if not replicate_groups:
            return None
        return replicate_groups[0]

    @staticmethod
    def _build_active_replicate_buckets(hsdp_params):
        buckets: dict[int, ReplicateBucket] = {}
        for idx, hsdp_param in enumerate(hsdp_params):
            group_info = getattr(hsdp_param, "unsharded_group_info", None)
            group = getattr(group_info, "group", None)
            group_size = getattr(group_info, "rank_size", getattr(hsdp_param, "replicate_world_size", 1))
            if group is None or group_size <= 1:
                continue
            key = id(group)
            if key not in buckets:
                buckets[key] = ReplicateBucket(
                    key=key,
                    group=group,
                    group_size=group_size,
                    param_indices=[],
                    flat_numel=0,
                )
            buckets[key].param_indices.append(idx)
            buckets[key].flat_numel += _shape_numel(hsdp_param.sharded_size)
        return buckets

    def _init_flat_param_buffer(self):
        """Rebase local shards into one flat buffer when storage semantics allow it."""
        if not self.enable_zero_copy_param_buffer:
            return
        if self.shard_world_size <= 1 or len(self.hsdp_params) == 0:
            return
        if any(p.offload_to_cpu or str(p.sharded_param.device) == "meta" for p in self.hsdp_params):
            return

        total_numel = sum(hsdp_param._sharded_param_data.numel() for hsdp_param in self.hsdp_params)
        orig_dtype = self.hsdp_params[0]._sharded_param_data.dtype
        flat_buffer = ms.mint.empty((total_numel,), dtype=orig_dtype, device=_normalize_device(self.device))

        offset = 0
        original_locals = []
        try:
            for hsdp_param in self.hsdp_params:
                original_locals.append((hsdp_param, hsdp_param._sharded_param_data, hsdp_param._sharded_local_tensor))
                numel = hsdp_param._sharded_param_data.numel()
                flat_slice = flat_buffer.narrow(0, offset, numel)
                flat_slice.copy_(hsdp_param._sharded_param_data)
                hsdp_param._sharded_param_data = flat_slice
                new_local = flat_slice.view(hsdp_param.sharded_size)
                req_grad = hsdp_param.sharded_param.requires_grad
                hsdp_param.sharded_param.set_data(new_local)
                hsdp_param.sharded_param._local_tensor = new_local
                if req_grad:
                    new_local.requires_grad_(True)
                    hsdp_param.sharded_param.requires_grad_(True)
                offset += numel
        except Exception:  # pylint: disable=W0718
            for hsdp_param, orig_flat, orig_local in original_locals:
                hsdp_param._sharded_param_data = orig_flat
                hsdp_param.sharded_param.set_data(orig_local)
                hsdp_param.sharded_param._local_tensor = orig_local
            self._flat_param_buffer = None
            self._flat_cast_buffer = None
            return

        self._flat_param_buffer = flat_buffer
        has_param_dtype = any(p.param_dtype is not None for p in self.hsdp_params)
        if has_param_dtype:
            cast_dtype = next(p.param_dtype for p in self.hsdp_params if p.param_dtype is not None)
            self._flat_cast_buffer = ms.mint.empty(
                (total_numel,), dtype=cast_dtype, device=_normalize_device(self.device)
            )

    def _is_flat_buffer_valid(self):
        """Check if flat buffer still backs the params' sharded data."""
        if self._flat_param_buffer is None or len(self.hsdp_params) == 0:
            return False
        first_param = self.hsdp_params[0]
        return (
            first_param._sharded_param_data.untyped_storage().data_ptr()
            == self._flat_param_buffer.untyped_storage().data_ptr()
        )

    def _allocate_bucket_buffers_if_needed(self, device, dtype):
        normalized_device = _normalize_device(device)
        for bucket in self._active_replicate_buckets.values():
            if bucket.flat_numel == 0:
                continue
            needs_new_buffer = (
                bucket.buffer is None
                or bucket.buffer.numel() != bucket.flat_numel
                or bucket.buffer.dtype != dtype
            )
            if needs_new_buffer:
                bucket.buffer = ms.mint.empty((bucket.flat_numel,), dtype=dtype, device=normalized_device)

    def _pack_bucket_from_reduce_output(self, bucket: ReplicateBucket) -> ms.Tensor:
        if bucket.buffer is None:
            raise AssertionError("Bucket buffer must be allocated before packing from reduce output")
        if self._reduce_output is None or self._reduce_hsdp_params is None:
            raise AssertionError("Bucket packing requires an active fused reduce output")
        dst_offset = 0
        for idx in bucket.param_indices:
            hsdp_param = self._reduce_hsdp_params[idx]
            src_offset = self._active_param_flat_offsets[idx]
            numel = _shape_numel(hsdp_param.sharded_size)
            bucket.buffer.narrow(0, dst_offset, numel).copy_(
                self._reduce_output.narrow(0, src_offset, numel)
            )
            dst_offset += numel
        return bucket.buffer

    def _unpack_bucket_to_reduce_output(self, bucket: ReplicateBucket) -> None:
        if bucket.buffer is None:
            raise AssertionError("Bucket buffer must exist before unpacking to reduce output")
        if self._reduce_output is None or self._reduce_hsdp_params is None:
            raise AssertionError("Bucket unpack requires an active fused reduce output")
        src_offset = 0
        for idx in bucket.param_indices:
            hsdp_param = self._reduce_hsdp_params[idx]
            dst_offset = self._active_param_flat_offsets[idx]
            numel = _shape_numel(hsdp_param.sharded_size)
            self._reduce_output.narrow(0, dst_offset, numel).copy_(
                bucket.buffer.narrow(0, src_offset, numel)
            )
            src_offset += numel

    def unshard(self, async_op: bool = False):
        """Trigger fused all-gather for all parameters in this group."""
        if self._result is not None:
            return
        if self.shard_world_size == 1:
            self._result = AllGatherResult(None, None, None)
            return
        self.foreach_all_gather(async_op=async_op)

    def _init_mp_dtypes(self):
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[MindSporeHSDPParamV2] = [
            p for p in self.hsdp_params if p.sharded_param.requires_grad
        ]
        orig_dtypes = {p.orig_dtype for p in trainable_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform original parameter dtype but got {orig_dtypes}"
            )
        self._orig_dtype = next(iter(orig_dtypes)) if trainable_params else None
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform reduce dtype but got {reduce_dtypes}"
            )
        self._reduce_dtype = next(iter(reduce_dtypes)) if trainable_params else None

    def wait_for_unshard(self):
        """Wait for fused all-gather and materialize per-parameter unsharded views."""
        if self._result is None:
            return
        if self.shard_world_size == 1:
            for hsdp_param in self.hsdp_params:
                all_gather_input = hsdp_param.all_gather_inputs[0]
                hsdp_param.init_all_gather_outputs(
                    [all_gather_input.numel()],
                    [all_gather_input.dtype],
                    self.shard_world_size,
                    _normalize_device(self.device),
                )
                hsdp_param.alloc_all_gather_outputs()
                copy_without_bumping_version(hsdp_param.all_gather_outputs[0], all_gather_input)
            self._result = None
        else:
            self.foreach_all_gather_copy_out()
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_unsharded_param()
            hsdp_param.to_unsharded()

    def alloc_all_gather_output(self, total_output_numel, dtype):
        """Allocate or resize the fused all-gather output buffer to the specified size and dtype."""
        normalized_device = _normalize_device(self.device)
        if self.ag_output is None or self.ag_output.dtype != dtype:
            self.ag_output = ms.mint.empty((total_output_numel,), dtype=dtype, device=normalized_device)
            return
        storage = self.ag_output.untyped_storage()
        expected_size = total_output_numel * self.ag_output.itemsize
        if storage.size() != expected_size:
            storage.resize_(expected_size)

    def free_all_gather_output(self):
        """Release the fused all-gather output buffer by resizing its storage to zero."""
        if self.ag_output is None:
            return
        storage = self.ag_output.untyped_storage()
        if storage.size() != 0:
            storage.resize_(0)

    @_no_grad()
    def foreach_all_gather(self, async_op=False):
        """Perform one fused all-gather across all parameters in the group."""
        if self.metadata_cache is None:
            self.metadata_cache = AllGatherMetadataCache()
        metadata = self.metadata_cache.get_metadata(self.hsdp_params, get_all_gather_metadata)
        if metadata.total_input_numel == 0:
            return
        world_size = self.shard_world_size
        rank = self.shard_rank
        total_output_numel = metadata.total_input_numel * world_size
        self.alloc_all_gather_output(total_output_numel, metadata.dtype)
        for hsdp_param in self.hsdp_params:
            hsdp_param.reset_sharded_param()
        if self.enable_zero_copy_param_buffer and not self._is_flat_buffer_valid():
            self._init_flat_param_buffer()

        use_flat_buffer = (
            self.enable_zero_copy_param_buffer
            and self._flat_param_buffer is not None
            and self._is_flat_buffer_valid()
        )
        if use_flat_buffer:
            if self._flat_cast_buffer is not None:
                self._flat_cast_buffer.copy_(self._flat_param_buffer)
                all_gather_input = self._flat_cast_buffer
            else:
                all_gather_input = self._flat_param_buffer
        else:
            all_gather_inputs = []
            for hsdp_param in self.hsdp_params:
                all_gather_inputs.extend(hsdp_param.all_gather_inputs)
            if len(all_gather_inputs) == 0:
                return
            all_gather_input, _ = all_gather_copy_in(
                all_gather_inputs,
                self.ag_output,
                metadata.inp_split_sizes,
                metadata.total_input_numel,
                rank,
            )
        handle = dist.all_gather_into_tensor(self.ag_output, all_gather_input, self.shard_group, async_op)
        self._result = AllGatherResult(self.ag_output, metadata, handle)

    @_no_grad()
    def foreach_all_gather_copy_out(self):
        """Scatter one fused all-gather result back into per-parameter buffers."""
        ag_output, metadata, handle = self._result
        if handle is not None:
            handle.wait()
        world_size = self.shard_world_size
        split_with_sizes_out = []
        for input_numels, input_dtypes, hsdp_param in zip(
            metadata.param_input_numels, metadata.param_input_dtypes, self.hsdp_params
        ):
            hsdp_param.init_all_gather_outputs(
                input_numels,
                input_dtypes,
                world_size,
                _normalize_device(ag_output.device),
            )
            hsdp_param.alloc_all_gather_outputs()
            split_with_sizes_out.extend(hsdp_param.all_gather_outputs)
        ag_output = ag_output.view(world_size, -1)
        out = [t.view(world_size, -1) for t in split_with_sizes_out]
        split_with_sizes_copy(ag_output, metadata.inp_split_sizes, dim=1, out=out)
        self._result = None
        self.free_all_gather_output()

    @_no_grad()
    def foreach_reduce(
        self,
        reduce_scatter_reduce_op: Optional[ops.ReduceOp] = ops.ReduceOp.SUM,
        async_op: bool = True,
    ) -> Optional[ms.Tensor]:
        """Perform fused reduce-scatter and optional bucketed all-reduce."""
        hsdp_params: List[MindSporeHSDPParamV2] = []
        unsharded_grads: List[ms.Tensor] = []
        for hsdp_param in self.hsdp_params:
            if not hasattr(hsdp_param, "_unsharded_param"):
                continue
            if hsdp_param.unsharded_accumulated_grad is not None:
                hsdp_params.append(hsdp_param)
                unsharded_grads.append(hsdp_param.unsharded_accumulated_grad_data)
            elif hsdp_param._unsharded_param.grad is not None:
                hsdp_params.append(hsdp_param)
                unsharded_grads.append(hsdp_param.unsharded_grad_data)
        if not hsdp_params:
            return None
        grad_dtypes = {g.dtype for g in unsharded_grads}
        if len(grad_dtypes) != 1:
            raise ValueError(
                f"FSDP reduce-scatter expects uniform grad dtype but got {grad_dtypes}"
            )
        grad_dtype = unsharded_grads[0].dtype
        reduce_dtype = self._reduce_dtype or grad_dtype
        world_size = self.shard_world_size
        reduce_scatter_input_numel = sum(s.numel() for s in unsharded_grads)
        reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
        device = _normalize_device(unsharded_grads[0].device)
        reduce_scatter_input = ms.mint.empty((reduce_scatter_input_numel,), dtype=reduce_dtype, device=device)
        reduce_scatter_copy_in(hsdp_params, unsharded_grads, reduce_scatter_input, world_size)
        # Captured here, consumed once in _apply_reduced_grad after all collectives
        # complete. Async paths cross method boundaries, so the field is unavoidable.
        reduce_output = ms.mint.empty((reduce_scatter_output_numel,), dtype=reduce_dtype, device=device)
        self._reduce_op = reduce_scatter_reduce_op
        self._reduce_hsdp_params = hsdp_params
        self._active_param_flat_offsets = []
        flat_offset = 0
        for hsdp_param in hsdp_params:
            self._active_param_flat_offsets.append(flat_offset)
            flat_offset += _shape_numel(hsdp_param.sharded_size)
        self._active_replicate_buckets = self._build_active_replicate_buckets(hsdp_params)
        self._allocate_bucket_buffers_if_needed(reduce_output.device, reduce_output.dtype)
        self._pending_all_reduce_handles = []
        if self.shard_group is None or world_size <= 1:
            comm_ctx.comm_handle = None
            self._reduce_output = reduce_scatter_input
            if async_op:
                comm_ctx.pre_param_group = self
            else:
                self.apply_fusion_reduced_grad()
            return self._reduce_output
        apply_gradient_scaling_factor(reduce_scatter_input, self.gradient_scaling_factor)
        rs_handle = dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=self.shard_group,
            op=reduce_scatter_reduce_op,
            async_op=async_op,
        )
        comm_ctx.comm_handle = rs_handle
        self._reduce_output = reduce_output
        if async_op:
            comm_ctx.pre_param_group = self
        else:
            self.apply_fusion_reduced_grad()
        return reduce_output

    def wait_reduce_scatter_and_issue_all_reduce(self):
        """Wait for reduce-scatter and issue async all-reduces for active buckets."""
        if comm_ctx.comm_handle is not None:
            comm_ctx.comm_handle.wait()
            comm_ctx.comm_handle = None
        if not self._active_replicate_buckets:
            self._apply_reduced_grad()
            return
        self._pending_all_reduce_handles = []
        for bucket in self._active_replicate_buckets.values():
            packed = self._pack_bucket_from_reduce_output(bucket)
            ar_handle = dist.all_reduce(
                packed,
                group=bucket.group,
                op=self._reduce_op,
                async_op=True,
            )
            self._pending_all_reduce_handles.append(
                PendingBucketAllReduce(bucket_key=bucket.key, handle=ar_handle)
            )
        comm_ctx.all_reduce_param_group = self

    def wait_all_reduce_and_apply_grad(self):
        """Wait for pending bucket all-reduces and apply reduced grads."""
        for pending in self._pending_all_reduce_handles:
            bucket = self._active_replicate_buckets[pending.bucket_key]
            pending.handle.wait()
            self._unpack_bucket_to_reduce_output(bucket)
        self._pending_all_reduce_handles = []
        comm_ctx.all_reduce_handle = None
        self._apply_reduced_grad()

    def apply_fusion_reduced_grad(self):
        """Synchronous fallback: wait, all-reduce buckets, then apply grads."""
        if comm_ctx.comm_handle is not None:
            comm_ctx.comm_handle.wait()
            comm_ctx.comm_handle = None
        for bucket in self._active_replicate_buckets.values():
            packed = self._pack_bucket_from_reduce_output(bucket)
            dist.all_reduce(
                packed,
                group=bucket.group,
                op=self._reduce_op,
            )
            self._unpack_bucket_to_reduce_output(bucket)
        self._apply_reduced_grad()

    def _apply_reduced_grad(self):
        """Write reduced gradients from the fused output buffer back to params."""
        flat_grad_offset = 0
        if self._reduce_hsdp_params is None or self._reduce_output is None:
            return
        # All collectives have completed; scale once on the fused buffer right
        # before slicing it into per-parameter sharded grads.
        for hsdp_param in self._reduce_hsdp_params:
            shard_numel = _shape_numel(hsdp_param.sharded_size)
            new_sharded_grad = self._reduce_output.narrow(0, flat_grad_offset, shard_numel)
            hsdp_param.apply_reduced_grad(new_sharded_grad, self._orig_dtype)
            flat_grad_offset += shard_numel
        self._reduce_output = None
        self._reduce_hsdp_params = None
        self._active_param_flat_offsets = []
        self._active_replicate_buckets = {}
        self._pending_all_reduce_handles = []


class AllReduceParamGroup:
    """Groups HSDP parameters by replicate group for fused async all-reduce."""

    ALIGNMENT_BYTES = 512

    @staticmethod
    def _resolve_reduce_dtype(
        reduce_dtype: Any,
        hsdp_params: List[MindSporeHSDPParamV2],
        orig_dtypes: List[Any],
    ) -> Any:
        """Resolve None reduce_dtype to match ``reduce_scatter_grad``'s ``dtype or grad.dtype``."""
        if reduce_dtype is not None:
            return reduce_dtype
        for hsdp_param in hsdp_params:
            if getattr(hsdp_param, "unsharded_accumulated_grad", None) is not None:
                return hsdp_param.unsharded_accumulated_grad_data.dtype
            unsharded_param = getattr(hsdp_param, "unsharded_param", None)
            if unsharded_param is not None and getattr(unsharded_param, "grad", None) is not None:
                return hsdp_param.unsharded_grad_data.dtype
        return orig_dtypes[0] if orig_dtypes else None

    def __init__(
        self,
        replicate_group,
        hsdp_params: List[MindSporeHSDPParamV2],
        orig_dtypes: List[Any],
        reduce_dtype: Any,
        reduce_op: ops.ReduceOp,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        replicate_world_size: Optional[int] = None,
    ):
        self.replicate_group = replicate_group
        self.hsdp_params = hsdp_params
        self.orig_dtypes = orig_dtypes
        self.reduce_dtype = self._resolve_reduce_dtype(reduce_dtype, hsdp_params, orig_dtypes)
        self.reduce_op = reduce_op
        self.mp_policy = mp_policy
        if replicate_world_size is not None:
            self.replicate_world_size = replicate_world_size
        elif replicate_group is not None and hasattr(replicate_group, "rank_size"):
            self.replicate_world_size = replicate_group.rank_size
        elif hsdp_params:
            self.replicate_world_size = hsdp_params[0].unsharded_group_info.rank_size
        else:
            self.replicate_world_size = 1
        self.fused_buffer: Optional[ms.Tensor] = None
        self.param_offsets: List[int] = []
        self.param_numels: List[int] = []
        self.all_reduce_handle: Optional[CommHandle] = None

    def compute_aligned_layout(self) -> int:
        """Compute fused buffer layout with trailing 512-byte alignment."""
        self.param_offsets = []
        self.param_numels = []
        element_size = int(ms.Tensor([], dtype=self.reduce_dtype).itemsize)
        current_offset = 0
        for hsdp_param in self.hsdp_params:
            numel = _shape_numel(hsdp_param.sharded_size)
            self.param_numels.append(numel)
            self.param_offsets.append(current_offset)
            current_offset += numel
        total_bytes = current_offset * element_size
        aligned_total_bytes = (
            (total_bytes + self.ALIGNMENT_BYTES - 1) // self.ALIGNMENT_BYTES
        ) * self.ALIGNMENT_BYTES
        return aligned_total_bytes // element_size

    def allocate_fused_buffer(self, device: Any) -> None:
        """Allocate the fused buffer with computed layout."""
        total_numel = self.compute_aligned_layout()
        normalized_device = _normalize_device(device)
        self.fused_buffer = ms.mint.empty(
            (total_numel,), dtype=self.reduce_dtype, device=normalized_device
        )
        self.fused_buffer.zero_()

    def get_param_buffer_view(self, idx: int) -> ms.Tensor:
        """Return a flat view for reduce_scatter output of parameter idx."""
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer not allocated. Call allocate_fused_buffer first.")
        offset = self.param_offsets[idx]
        numel = self.param_numels[idx]
        return self.fused_buffer.narrow(0, offset, numel)

    def get_param_grad_view(self, idx: int, target_shape) -> ms.Tensor:
        """Return a reshaped view of the reduced gradient for parameter idx."""
        return self.get_param_buffer_view(idx).view(target_shape)

    def accumulate_existing_grads_to_buffer(self) -> None:
        """Accumulate existing sharded grads into fused_buffer before all-reduce."""
        if self.fused_buffer is None:
            return

        for idx, hsdp_param in enumerate(self.hsdp_params):
            existing_grad = None
            if self.mp_policy is not None and self.mp_policy.apply_grad_on_fp32_main_grad:
                if hasattr(hsdp_param.sharded_param, "main_grad"):
                    existing_grad = hsdp_param.sharded_param.main_grad
            else:
                existing_grad = hsdp_param.sharded_param.grad
            if existing_grad is not None and not hsdp_param.accumulated_allreduced_grad:
                if isinstance(existing_grad, DTensor):
                    existing_grad_local = existing_grad._local_tensor
                else:
                    existing_grad_local = existing_grad
                buffer_view = self.get_param_buffer_view(idx)
                if existing_grad_local.dtype != self.reduce_dtype:
                    existing_grad_local = existing_grad_local.to(self.reduce_dtype)
                buffer_view.add_(existing_grad_local.view_as(buffer_view))
                if self.mp_policy is not None and self.mp_policy.apply_grad_on_fp32_main_grad:
                    if hasattr(hsdp_param.sharded_param, "main_grad"):
                        hsdp_param.sharded_param.main_grad = None
                else:
                    hsdp_param.sharded_param.grad = None

    def issue_async_allreduce(self) -> None:
        """Issue async all_reduce on the fused buffer (SUM for padding correctness)."""
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer not allocated.")
        self.all_reduce_handle = dist.all_reduce(
            self.fused_buffer,
            op=ops.ReduceOp.SUM,
            group=self.replicate_group,
            async_op=True,
        )

    def wait_and_apply_grads(self) -> bool:
        """Wait for all_reduce and apply gradients to parameters."""
        if self.all_reduce_handle is not None:
            self.all_reduce_handle.wait()
            self.all_reduce_handle = None
        need_synchronize = False
        for idx, hsdp_param in enumerate(self.hsdp_params):
            reduced_grad = self.get_param_grad_view(idx, hsdp_param.sharded_size)
            # issue_async_allreduce uses SUM (so end-of-buffer padding zeros stay
            # correct), so an AVG reduce op must divide by the replicate world size
            # here. The reduce-scatter leg already averaged over the shard axis.
            if self.reduce_op == ops.ReduceOp.AVG and self.replicate_world_size > 1:
                reduced_grad = reduced_grad / self.replicate_world_size
            need_synchronize = (
                hsdp_param.apply_reduced_grad(reduced_grad, self.orig_dtypes[idx])
                or need_synchronize
            )
            hsdp_param.accumulated_allreduced_grad = True
        self.fused_buffer = None
        return need_synchronize

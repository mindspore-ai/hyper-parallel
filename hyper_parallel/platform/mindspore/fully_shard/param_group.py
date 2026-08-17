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
"""MindSpore HSDP parameter groups with fused mint collectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, NamedTuple, Optional

import mindspore as ms
from mindspore.common.api import _no_grad
import mindspore.mint.distributed as dist

from hyper_parallel.core.fully_shard.hsdp_scheduler import ParamGroupCommCtx
from hyper_parallel.core.fully_shard.hsdp_utils import apply_gradient_scaling_factor
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import build_rs_plan, pack_for_reduce_scatter
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2


def _normalize_device(device: Any) -> str:
    """Normalize a runtime device for mint allocation APIs."""
    return str(device).split(":", 1)[0]


def _shape_numel(shape) -> int:
    """Return the element count of a MindSpore shape."""
    return math.prod(int(dim) for dim in shape)


@dataclass
class AllGatherMetadata:
    """Describe the rank-local layout of one fused all-gather bucket."""

    param_input_dtypes: list[list[Any]]
    param_input_numels: list[list[int]]
    dtype: Any
    inp_split_sizes: list[int]
    total_input_numel: int
    hash_key: int = field(init=False)

    def __post_init__(self) -> None:
        self.hash_key = hash(
            (
                tuple(tuple(dtypes) for dtypes in self.param_input_dtypes),
                tuple(tuple(numels) for numels in self.param_input_numels),
                self.dtype,
                tuple(self.inp_split_sizes),
                self.total_input_numel,
            )
        )


class AllGatherResult(NamedTuple):
    """Keep fused all-gather buffers and handle alive until copy-out."""

    all_gather_input: Optional[ms.Tensor]
    all_gather_output: Optional[ms.Tensor]
    metadata: Optional[AllGatherMetadata]
    handle: Optional[Any]


class AllGatherMetadataCache:
    """Cache all-gather metadata across iterations."""

    _cache: dict[int, AllGatherMetadata] = {}

    @classmethod
    def get_metadata(cls, hsdp_params, fn):
        """Retrieve or compute metadata keyed by parameter identity and version."""
        param_key = tuple((id(param), getattr(param, "version", 0)) for param in hsdp_params)
        key = hash(param_key)
        if key not in cls._cache:
            cls._cache[key] = fn(hsdp_params)
        return cls._cache[key]


@dataclass
class AllGatherBucket:
    """Own parameters sharing one all-gather group and dtype."""

    hsdp_params: list[MindSporeHSDPParamV2]
    shard_group: Any
    shard_rank: int
    shard_world_size: int
    dtype: Any
    metadata: AllGatherMetadata
    all_gather_result: Optional[AllGatherResult] = None

    @_no_grad()
    def copy_out(self) -> None:
        """Wait the all-gather and copy each result into stable parameter buffers."""
        result = self.all_gather_result
        if result is None or result.all_gather_output is None:
            return
        if result.handle is not None:
            result.handle.wait()
        all_gather_output = result.all_gather_output
        output_buffers = []
        for input_numels, input_dtypes, hsdp_param in zip(
            self.metadata.param_input_numels,
            self.metadata.param_input_dtypes,
            self.hsdp_params,
        ):
            hsdp_param.init_unsharded_param_buffers(
                input_numels,
                input_dtypes,
                self.shard_world_size,
                _normalize_device(all_gather_output.device),
            )
            hsdp_param.alloc_unsharded_param_buffers()
            output_buffers.extend(hsdp_param.unsharded_param_buffers)
        split_with_sizes_copy(
            all_gather_output.view(self.shard_world_size, -1),
            self.metadata.inp_split_sizes,
            dim=1,
            out=[tensor.view(self.shard_world_size, -1) for tensor in output_buffers],
        )
        self.all_gather_result = None


@dataclass
class GradientBucketLayout:
    """Describe the per-parameter layout shared by an RS-to-AR bucket chain."""

    hsdp_params: list[MindSporeHSDPParamV2]
    param_offsets: list[int]
    param_numels: list[int]
    total_numel: int


@dataclass
class ReduceScatterBucket:
    """Own one fused reduce-scatter and its temporary buffers."""

    layout: GradientBucketLayout
    unsharded_grads: list[ms.Tensor]
    shard_group: Any
    shard_world_size: int
    dtype: Any
    reduce_op: str
    needs_avg_div: bool
    reduce_scatter_input: Optional[ms.Tensor] = None
    reduce_scatter_output: Optional[ms.Tensor] = None
    handle: Optional[Any] = None

    @property
    def hsdp_params(self) -> list[MindSporeHSDPParamV2]:
        """Return parameters in fused-buffer order."""
        return self.layout.hsdp_params

    @property
    def param_offsets(self) -> list[int]:
        """Return parameter offsets in the reduce-scatter output."""
        return self.layout.param_offsets

    @property
    def bucket_key(self) -> tuple:
        """Return the identity of this fusion class across micro-steps."""
        return (self.shard_group, self.dtype)

    @property
    def uses_collective(self) -> bool:
        """Whether this bucket needs an actual reduce-scatter collective."""
        return self.shard_group is not None and self.shard_world_size > 1

    def move_reduce_scatter_output(self) -> ms.Tensor:
        """Transfer exclusive ownership of the completed output."""
        if self.reduce_scatter_output is None:
            raise RuntimeError("Reduce-scatter output has already been released.")
        output = self.reduce_scatter_output
        self.reduce_scatter_output = None
        return output


@dataclass
class AllReduceBucket:
    """Own the all-reduce stage following one fused reduce-scatter bucket."""

    layout: GradientBucketLayout
    source_reduce_scatter_bucket: ReduceScatterBucket
    replicate_group: Any
    replicate_world_size: int
    dtype: Any
    needs_avg_div: bool
    all_reduce_output: Optional[ms.Tensor] = None
    handle: Optional[Any] = None

    @property
    def hsdp_params(self) -> list[MindSporeHSDPParamV2]:
        """Return parameters in fused-buffer order."""
        return self.layout.hsdp_params

    @property
    def uses_collective(self) -> bool:
        """Whether this bucket needs an actual all-reduce collective."""
        return self.replicate_group is not None and self.replicate_world_size > 1


def get_all_gather_metadata(hsdp_params) -> AllGatherMetadata:
    """Collect fused all-gather metadata for one communication dtype."""
    param_input_dtypes = []
    param_input_numels = []
    inp_split_sizes = []
    total_input_numel = 0
    dtype = None
    for hsdp_param in hsdp_params:
        inputs = hsdp_param.all_gather_inputs
        if dtype is None:
            dtype = inputs[0].dtype
        if any(tensor.dtype != dtype for tensor in inputs):
            raise ValueError("All parameters in an all-gather bucket must have the same dtype.")
        input_dtypes = [tensor.dtype for tensor in inputs]
        input_numels = [tensor.numel() for tensor in inputs]
        param_input_dtypes.append(input_dtypes)
        param_input_numels.append(input_numels)
        inp_split_sizes.extend(input_numels)
        total_input_numel += sum(input_numels)
    if dtype is None:
        raise ValueError("Cannot build all-gather metadata for an empty parameter bucket.")
    return AllGatherMetadata(
        param_input_dtypes,
        param_input_numels,
        dtype,
        inp_split_sizes,
        total_input_numel,
    )


@_no_grad()
def all_gather_copy_in(
    all_gather_inputs,
    all_gather_output,
    inp_split_sizes,
    all_gather_input_numel,
    rank,
):
    """Build a contiguous fused all-gather input without writing through views."""
    del inp_split_sizes, all_gather_input_numel, rank
    all_gather_input = ms.mint.cat(
        [tensor.reshape(-1) for tensor in all_gather_inputs],
        dim=0,
    )
    return all_gather_input, all_gather_output


@_no_grad()
def split_with_sizes_copy(all_gather_output, split_sizes, dim, out):
    """Copy dim-1 slices from a fused all-gather into stable buffers."""
    if dim != 1:
        raise NotImplementedError("split_with_sizes_copy currently only supports dim=1")
    offset = 0
    for destination, size in zip(out, split_sizes):
        copy_without_bumping_version(
            destination,
            all_gather_output.narrow(dim, offset, size),
        )
        offset += size


@_no_grad()
def reduce_scatter_copy_in(
    hsdp_params: List[MindSporeHSDPParamV2],
    unsharded_grads: List[ms.Tensor],
    reduce_scatter_input: ms.Tensor,
    world_size: int,
) -> None:
    """Pack gradients with mixed shard dimensions without writing through views."""
    if len(hsdp_params) != len(unsharded_grads):
        raise AssertionError(
            "reduce_scatter_copy_in expects one hsdp_param per unsharded_grad, but got "
            f"{len(hsdp_params)} params and {len(unsharded_grads)} grads"
        )
    packed_grads = []
    for hsdp_param, unsharded_grad in zip(hsdp_params, unsharded_grads):
        plan = build_rs_plan(hsdp_param, unsharded_grad.contiguous(), world_size)
        packed_grads.append(pack_for_reduce_scatter(unsharded_grad.contiguous(), plan))
    packed_rows = ms.mint.cat(packed_grads, dim=1)
    if packed_rows.numel() != reduce_scatter_input.numel():
        raise AssertionError(
            "reduce_scatter_copy_in packed an unexpected number of elements: "
            f"{packed_rows.numel()} != {reduce_scatter_input.numel()}"
        )
    copy_without_bumping_version(reduce_scatter_input, packed_rows.reshape(-1))


class HSDPParamGroup:
    """Fuse compatible collectives for parameters in one fully_shard unit."""

    def __init__(
        self,
        hsdp_params,
        device: Optional[str] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        enable_zero_copy: bool = False,
        comm_ctx: Optional[ParamGroupCommCtx] = None,
    ):
        self.device = device
        self.hsdp_params = hsdp_params
        self.mp_policy = mp_policy
        self.enable_zero_copy = enable_zero_copy
        self.comm_ctx = comm_ctx or ParamGroupCommCtx()
        self.gradient_scaling_factor = None
        self.requires_all_reduce = True
        self.all_gather_buckets: list[AllGatherBucket] = []
        self.reduce_scatter_buckets: list[ReduceScatterBucket] = []
        self.all_reduce_buckets: list[AllReduceBucket] = []
        self.reduce_partial_outputs: dict[tuple, ms.Tensor] = {}

    def _init_all_gather_buckets(self) -> None:
        """Build ordered all-gather buckets from per-parameter routes and dtypes."""
        params_by_bucket = {}
        bucket_groups = {}
        for hsdp_param in self.hsdp_params:
            if hsdp_param.shard_world_size <= 1:
                continue
            if not isinstance(hsdp_param.mesh_info, FSDPMeshInfo):
                raise ValueError(
                    f"Fused all-gather expects FSDPMeshInfo, got {type(hsdp_param.mesh_info)}"
                )
            shard_group = hsdp_param.mesh_info.shard_process_group
            communication_dtype = hsdp_param.param_dtype or hsdp_param.orig_dtype
            bucket_key = (shard_group, communication_dtype)
            params_by_bucket.setdefault(bucket_key, []).append(hsdp_param)
            bucket_groups[bucket_key] = shard_group

        self.all_gather_buckets = []
        for bucket_key, hsdp_params in params_by_bucket.items():
            metadata = get_all_gather_metadata(hsdp_params)
            self.all_gather_buckets.append(
                AllGatherBucket(
                    hsdp_params=hsdp_params,
                    shard_group=bucket_groups[bucket_key],
                    shard_rank=hsdp_params[0].shard_rank,
                    shard_world_size=hsdp_params[0].shard_world_size,
                    dtype=bucket_key[1],
                    metadata=metadata,
                )
            )

    def unshard(self, async_op: bool = False) -> None:
        """Launch fused all-gathers for every compatible bucket."""
        if self.all_gather_buckets and any(
            bucket.all_gather_result is not None for bucket in self.all_gather_buckets
        ):
            return
        self.foreach_all_gather(async_op)

    @_no_grad()
    def foreach_all_gather(self, async_op: bool = False) -> None:
        """Initialize ordered buckets and launch their all-gathers."""
        if not self.all_gather_buckets:
            self._init_all_gather_buckets()
        for hsdp_param in self.hsdp_params:
            if hsdp_param.shard_world_size <= 1:
                hsdp_param.unshard(async_op)
        for bucket in self.all_gather_buckets:
            if bucket.all_gather_result is not None:
                continue
            metadata = bucket.metadata
            all_gather_output = ms.mint.empty(
                (metadata.total_input_numel * bucket.shard_world_size,),
                dtype=bucket.dtype,
                device=_normalize_device(self.device),
            )
            all_gather_inputs = []
            for hsdp_param in bucket.hsdp_params:
                hsdp_param.reset_sharded_param()
                all_gather_inputs.extend(hsdp_param.all_gather_inputs)
            all_gather_input, all_gather_output = all_gather_copy_in(
                all_gather_inputs,
                all_gather_output,
                metadata.inp_split_sizes,
                metadata.total_input_numel,
                bucket.shard_rank,
            )
            handle = dist.all_gather_into_tensor(
                all_gather_output,
                all_gather_input,
                group=bucket.shard_group,
                async_op=async_op,
            )
            bucket.all_gather_result = AllGatherResult(
                all_gather_input=all_gather_input,
                all_gather_output=all_gather_output,
                metadata=metadata,
                handle=handle,
            )

    def wait_for_unshard(self) -> None:
        """Wait all all-gathers and install stable unsharded parameters."""
        for bucket in self.all_gather_buckets:
            bucket.copy_out()
        for hsdp_param in self.hsdp_params:
            hsdp_param.wait_for_unshard()

    @staticmethod
    def _build_gradient_bucket_layout(hsdp_params) -> GradientBucketLayout:
        """Build one compact output layout for the supplied parameter order."""
        param_offsets = []
        param_numels = []
        total_numel = 0
        for hsdp_param in hsdp_params:
            param_offsets.append(total_numel)
            param_numel = _shape_numel(hsdp_param.sharded_size)
            param_numels.append(param_numel)
            total_numel += param_numel
        return GradientBucketLayout(
            hsdp_params,
            param_offsets,
            param_numels,
            total_numel,
        )

    def _build_reduce_scatter_buckets(self, reduce_op: str) -> list[ReduceScatterBucket]:
        """Build ordered reduce-scatter buckets without tensor side effects."""
        params_by_bucket = {}
        grads_by_bucket = {}
        groups_by_bucket = {}
        for hsdp_param in self.hsdp_params:
            if not hsdp_param.sharded_param.requires_grad:
                continue
            if hsdp_param.unsharded_accumulated_grad is not None:
                unsharded_grad = hsdp_param.unsharded_accumulated_grad_data
            elif hsdp_param.unsharded_param.grad is not None:
                unsharded_grad = hsdp_param.unsharded_grad_data
            else:
                continue
            shard_group = (
                hsdp_param.mesh_info.shard_process_group
                if isinstance(hsdp_param.mesh_info, FSDPMeshInfo)
                else None
            )
            reduce_dtype = hsdp_param.reduce_comm_dtype(unsharded_grad)
            bucket_key = (shard_group, reduce_dtype)
            params_by_bucket.setdefault(bucket_key, []).append(hsdp_param)
            grads_by_bucket.setdefault(bucket_key, []).append(unsharded_grad)
            groups_by_bucket[bucket_key] = shard_group

        buckets = []
        for bucket_key, hsdp_params in params_by_bucket.items():
            shard_world_size = hsdp_params[0].shard_world_size
            if any(param.shard_world_size != shard_world_size for param in hsdp_params):
                raise ValueError("A reduce-scatter bucket must use one shard world size.")
            needs_avg_div = reduce_op == "avg"
            buckets.append(
                ReduceScatterBucket(
                    layout=self._build_gradient_bucket_layout(hsdp_params),
                    unsharded_grads=grads_by_bucket[bucket_key],
                    shard_group=groups_by_bucket[bucket_key],
                    shard_world_size=shard_world_size,
                    dtype=bucket_key[1],
                    reduce_op="sum" if needs_avg_div else reduce_op,
                    needs_avg_div=needs_avg_div,
                )
            )
        return buckets

    @staticmethod
    def _build_all_reduce_buckets(
        reduce_scatter_buckets: list[ReduceScatterBucket],
    ) -> list[AllReduceBucket]:
        """Build an optional all-reduce stage for each RS bucket."""
        buckets = []
        for rs_bucket in reduce_scatter_buckets:
            representative = rs_bucket.hsdp_params[0]
            if not isinstance(representative.mesh_info, DDPMeshInfo):
                if any(
                    isinstance(hsdp_param.mesh_info, DDPMeshInfo)
                    for hsdp_param in rs_bucket.hsdp_params[1:]
                ):
                    raise ValueError(
                        "A reduce-scatter bucket cannot mix parameters with and without "
                        "a subsequent all-reduce."
                    )
                continue
            replicate_group = representative.mesh_info.replicate_process_group
            replicate_world_size = representative.replicate_world_size
            for hsdp_param in rs_bucket.hsdp_params[1:]:
                if (
                    not isinstance(hsdp_param.mesh_info, DDPMeshInfo)
                    or hsdp_param.mesh_info.replicate_process_group != replicate_group
                    or hsdp_param.replicate_world_size != replicate_world_size
                ):
                    raise ValueError(
                        "All parameters in a reduce-scatter bucket must share one "
                        "subsequent all-reduce group."
                    )
            buckets.append(
                AllReduceBucket(
                    layout=rs_bucket.layout,
                    source_reduce_scatter_bucket=rs_bucket,
                    replicate_group=replicate_group,
                    replicate_world_size=replicate_world_size,
                    dtype=rs_bucket.dtype,
                    needs_avg_div=rs_bucket.needs_avg_div,
                )
            )
        return buckets

    def _issue_reduce_scatter_buckets(self, async_op: bool) -> None:
        """Prepare RS buffers, release source gradients, and launch collectives."""
        for bucket in self.reduce_scatter_buckets:
            reduce_scatter_input = ms.mint.empty(
                (bucket.layout.total_numel * bucket.shard_world_size,),
                dtype=bucket.dtype,
                device=_normalize_device(bucket.unsharded_grads[0].device),
            )
            reduce_scatter_copy_in(
                bucket.hsdp_params,
                bucket.unsharded_grads,
                reduce_scatter_input,
                bucket.shard_world_size,
            )
            apply_gradient_scaling_factor(reduce_scatter_input, self.gradient_scaling_factor)
            bucket.reduce_scatter_input = reduce_scatter_input
            if bucket.uses_collective:
                bucket.reduce_scatter_output = ms.mint.empty(
                    (bucket.layout.total_numel,),
                    dtype=bucket.dtype,
                    device=_normalize_device(reduce_scatter_input.device),
                )
            else:
                bucket.reduce_scatter_output = reduce_scatter_input
            for hsdp_param in bucket.hsdp_params:
                hsdp_param.clear_unsharded_source_grad()
            bucket.unsharded_grads = []
            if not bucket.uses_collective:
                continue
            bucket.handle = dist.reduce_scatter_tensor(
                bucket.reduce_scatter_output,
                bucket.reduce_scatter_input,
                group=bucket.shard_group,
                op=bucket.reduce_op,
                async_op=async_op,
            )

    @_no_grad()
    def foreach_reducescatter(
        self,
        reduce_scatter_reduce_op: str = "avg",
        async_op: bool = True,
    ) -> None:
        """Launch fused reduce-scatter buckets for this module's gradients."""
        self.reduce_scatter_buckets = self._build_reduce_scatter_buckets(
            reduce_scatter_reduce_op,
        )
        if not self.reduce_scatter_buckets:
            return
        self.all_reduce_buckets = self._build_all_reduce_buckets(self.reduce_scatter_buckets)
        self._issue_reduce_scatter_buckets(async_op)
        if async_op:
            self.comm_ctx.pre_param_group = self
        else:
            self.wait_reduce_scatter_and_issue_all_reduce(async_op=False)
            self.wait_all_reduce_and_save_grad()

    def _wait_reduce_scatter_buckets(self) -> None:
        """Wait RS buckets and finish shard-dimension averaging."""
        for bucket in self.reduce_scatter_buckets:
            if bucket.handle is not None:
                bucket.handle.wait()
                bucket.handle = None
            bucket.reduce_scatter_input = None
            if bucket.reduce_scatter_output is None:
                raise RuntimeError("Reduce-scatter bucket has not been prepared.")
            if bucket.needs_avg_div and bucket.shard_world_size > 1:
                bucket.reduce_scatter_output = ms.mint.div(
                    bucket.reduce_scatter_output,
                    bucket.shard_world_size,
                )

    @staticmethod
    def _issue_all_reduce_buckets(
        all_reduce_buckets: list[AllReduceBucket],
        async_op: bool,
    ) -> None:
        """Launch SUM all-reduces for completed RS outputs."""
        for bucket in all_reduce_buckets:
            if bucket.all_reduce_output is None:
                raise RuntimeError("All-reduce bucket has not received its reduce-scatter output.")
            if not bucket.uses_collective:
                continue
            bucket.handle = dist.all_reduce(
                bucket.all_reduce_output,
                group=bucket.replicate_group,
                op="sum",
                async_op=async_op,
            )

    def wait_reduce_scatter_and_issue_all_reduce(self, async_op: bool = True) -> None:
        """Wait all RS buckets and launch the resulting HSDP all-reduces."""
        self._wait_reduce_scatter_buckets()
        if not self.requires_all_reduce:
            for bucket in self.reduce_scatter_buckets:
                current_output = bucket.move_reduce_scatter_output()
                partial_output = self.reduce_partial_outputs.get(bucket.bucket_key)
                self.reduce_partial_outputs[bucket.bucket_key] = (
                    current_output
                    if partial_output is None
                    else ms.mint.add(partial_output, current_output)
                )
            self.reduce_scatter_buckets = []
            self.all_reduce_buckets = []
            return

        for bucket in self.reduce_scatter_buckets:
            partial_output = self.reduce_partial_outputs.pop(bucket.bucket_key, None)
            if partial_output is not None:
                bucket.reduce_scatter_output = ms.mint.add(
                    bucket.reduce_scatter_output,
                    partial_output,
                )
        all_reduce_by_source = {
            id(bucket.source_reduce_scatter_bucket): bucket
            for bucket in self.all_reduce_buckets
        }
        for rs_bucket in self.reduce_scatter_buckets:
            all_reduce_bucket = all_reduce_by_source.get(id(rs_bucket))
            if all_reduce_bucket is not None:
                all_reduce_bucket.all_reduce_output = rs_bucket.move_reduce_scatter_output()
                continue
            reduce_scatter_output = rs_bucket.move_reduce_scatter_output()
            for hsdp_param, param_numel, param_offset in zip(
                rs_bucket.hsdp_params,
                rs_bucket.layout.param_numels,
                rs_bucket.param_offsets,
            ):
                hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = (
                    reduce_scatter_output.narrow(0, param_offset, param_numel)
                )
        self.reduce_scatter_buckets = []
        self._issue_all_reduce_buckets(self.all_reduce_buckets, async_op)
        if self.all_reduce_buckets:
            self.comm_ctx.all_reduce_param_group = self

    def wait_all_reduce_and_save_grad(self) -> None:
        """Wait all-reduces and expose their per-parameter output views."""
        for bucket in self.all_reduce_buckets:
            if bucket.handle is not None:
                bucket.handle.wait()
                bucket.handle = None
            if bucket.all_reduce_output is None:
                raise RuntimeError("All-reduce output has already been released.")
            output = bucket.all_reduce_output
            if bucket.needs_avg_div and bucket.replicate_world_size > 1:
                output = ms.mint.div(output, bucket.replicate_world_size)
            for hsdp_param, param_numel, param_offset in zip(
                bucket.hsdp_params,
                bucket.layout.param_numels,
                bucket.layout.param_offsets,
            ):
                hsdp_param.all_reduce_comm_ctx.all_reduce_output = output.narrow(
                    0,
                    param_offset,
                    param_numel,
                )
            bucket.all_reduce_output = None
        self.all_reduce_buckets = []
        if self.comm_ctx.all_reduce_param_group is self:
            self.comm_ctx.all_reduce_param_group = None

    def reset_iter_state(self) -> None:
        """Drop communication references after a completed iteration."""
        for bucket in self.all_gather_buckets:
            bucket.all_gather_result = None
        for bucket in self.reduce_scatter_buckets:
            bucket.unsharded_grads = []
            bucket.reduce_scatter_input = None
            bucket.reduce_scatter_output = None
            bucket.handle = None
        for bucket in self.all_reduce_buckets:
            bucket.all_reduce_output = None
            bucket.handle = None
        self.reduce_scatter_buckets = []
        self.all_reduce_buckets = []
        self.reduce_partial_outputs.clear()
        if self.comm_ctx.pre_param_group is self:
            self.comm_ctx.pre_param_group = None
        if self.comm_ctx.all_reduce_param_group is self:
            self.comm_ctx.all_reduce_param_group = None


class AllReduceParamGroup:
    """Fuse per-parameter RS outputs for one HSDP replicate all-reduce."""

    ALIGNMENT_BYTES = 512

    def __init__(
        self,
        replicate_group,
        hsdp_params: List[MindSporeHSDPParamV2],
        reduce_op: str,
    ):
        self.replicate_group = replicate_group
        self.hsdp_params = hsdp_params
        self.reduce_dtype = hsdp_params[0].reduce_comm_dtype()
        self.reduce_op = reduce_op
        self.replicate_world_size = hsdp_params[0].replicate_world_size
        self.fused_buffer: Optional[ms.Tensor] = None
        self.param_offsets: List[int] = []
        self.param_numels: List[int] = []
        self.all_reduce_handle: Optional[Any] = None

    def compute_aligned_layout(self) -> int:
        """Compute packed parameter offsets and align total buffer size."""
        self.param_offsets = []
        self.param_numels = []
        element_size = int(ms.Tensor([], dtype=self.reduce_dtype).itemsize)
        current_offset = 0
        for hsdp_param in self.hsdp_params:
            numel = _shape_numel(hsdp_param.sharded_size)
            self.param_offsets.append(current_offset)
            self.param_numels.append(numel)
            current_offset += numel
        total_bytes = current_offset * element_size
        aligned_total_bytes = (
            (total_bytes + self.ALIGNMENT_BYTES - 1) // self.ALIGNMENT_BYTES
        ) * self.ALIGNMENT_BYTES
        return aligned_total_bytes // element_size

    def allocate_fused_buffer(self, device: Any) -> None:
        """Allocate and zero the fused all-reduce buffer."""
        del device
        self.fused_buffer = ms.mint.zeros(
            (self.compute_aligned_layout(),),
            dtype=self.reduce_dtype,
        )

    def get_param_buffer_view(self, index: int) -> ms.Tensor:
        """Return one parameter's communication view."""
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer not allocated. Call allocate_fused_buffer first.")
        return self.fused_buffer.narrow(
            0,
            self.param_offsets[index],
            self.param_numels[index],
        )

    def accumulate_reduce_partial_outputs(self) -> None:
        """Pack RS outputs and prior micro-step partials into one aligned AR buffer."""
        param_outputs = []
        for hsdp_param in self.hsdp_params:
            reduced_output = hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output
            if reduced_output is None:
                raise RuntimeError("All-reduce group requires one completed reduce-scatter output per parameter.")
            if reduced_output.dtype != self.reduce_dtype:
                reduced_output = reduced_output.to(self.reduce_dtype)
            partial_output = hsdp_param.reduce_partial_output
            if partial_output is not None:
                if partial_output.dtype != self.reduce_dtype:
                    partial_output = partial_output.to(self.reduce_dtype)
                reduced_output = ms.mint.add(reduced_output, partial_output)
                hsdp_param.reduce_partial_output = None
            param_outputs.append(reduced_output.reshape(-1))
            hsdp_param.clear_reduce_scatter_output()
            hsdp_param.clear_unsharded_source_grad()
        packed_output = ms.mint.cat(param_outputs, dim=0)
        total_numel = self.compute_aligned_layout()
        if packed_output.numel() < total_numel:
            padding = ms.mint.zeros(
                (total_numel - packed_output.numel(),),
                dtype=self.reduce_dtype,
            )
            packed_output = ms.mint.cat((packed_output, padding), dim=0)
        self.fused_buffer = packed_output

    def issue_async_allreduce(self) -> None:
        """Launch SUM all-reduce; AVG division is applied when splitting."""
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer not allocated.")
        self.all_reduce_handle = dist.all_reduce(
            self.fused_buffer,
            op="sum",
            group=self.replicate_group,
            async_op=True,
        )

    def wait_and_split_grads(self) -> None:
        """Wait all-reduce and expose per-parameter context views."""
        if self.all_reduce_handle is not None:
            self.all_reduce_handle.wait()
            self.all_reduce_handle = None
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer has already been released.")
        output = self.fused_buffer
        if self.reduce_op == "avg" and self.replicate_world_size > 1:
            output = ms.mint.div(output, self.replicate_world_size)
        for index, hsdp_param in enumerate(self.hsdp_params):
            hsdp_param.all_reduce_comm_ctx.all_reduce_output = output.narrow(
                0,
                self.param_offsets[index],
                self.param_numels[index],
            )
        self.fused_buffer = None


__all__ = [
    "AllGatherBucket",
    "AllGatherMetadata",
    "AllGatherMetadataCache",
    "AllGatherResult",
    "AllReduceBucket",
    "AllReduceParamGroup",
    "GradientBucketLayout",
    "HSDPParamGroup",
    "ReduceScatterBucket",
    "all_gather_copy_in",
    "get_all_gather_metadata",
    "reduce_scatter_copy_in",
    "split_with_sizes_copy",
]

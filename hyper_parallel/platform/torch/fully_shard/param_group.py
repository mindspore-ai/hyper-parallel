# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
#
# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/fsdp/_fully_shard
# ==========================================================================
"""Fused parameter communication for Torch fully_shard.

Instead of issuing one collective per parameter, the managed parameters of a
fully_shard unit are packed into contiguous buckets and communicated with a
single collective each, which cuts kernel-launch overhead and improves
bandwidth utilization.

Bucketing: parameters are not fused into one global buffer. Each collective
splits its parameters into buckets keyed by the attributes that must be uniform
within one call -- ``(process group, dtype)`` -- because a collective cannot mix
communication groups or element types:

- ``AllGatherBucket``: one fused all-gather, keyed by (shard group, param dtype).
  Owns the optional zero-copy ``flat_param_buffer`` that parameter shards are
  rebased onto, so the gather reads parameter storage directly.
- ``ReduceScatterBucket``: one fused reduce-scatter, keyed by (shard group,
  reduce dtype). Packs gradients whose shard dim may differ per parameter.
- ``AllReduceBucket``: one fused HSDP replicate all-reduce, keyed by
  (replicate group, reduced-grad dtype), built from completed RS outputs.

``CommContext`` tracks which ``HSDPParamGroup`` owns each in-flight backward
stage so the next module's backward can wait on it, giving the three-way
overlap: layer N reduce-scatter with layer N-1 backward compute, and layer N
all-reduce with layer N-1 reduce-scatter.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from hyper_parallel.core.fully_shard.hsdp_utils import apply_gradient_scaling_factor
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2


@dataclass
class AllGatherMetadata:
    """Describe the rank-local layout of one fused all-gather bucket."""

    param_input_dtypes: list[list[torch.dtype]]
    param_input_numels: list[list[int]]
    dtype: torch.dtype
    inp_split_sizes: list[int]
    total_input_numel: int


@dataclass
class AllGatherResult:
    """Keep one all-gather input, output, and handle alive until copy-out."""

    all_gather_input: Optional[torch.Tensor]
    all_gather_output: Optional[torch.Tensor]
    handle: Optional[dist.Work]


@dataclass
class AllGatherBucket:
    """Own parameters and buffers sharing one all-gather group and dtype."""

    hsdp_params: list[TorchHSDPParamV2]
    shard_group: dist.ProcessGroup
    shard_rank: int
    shard_world_size: int
    dtype: torch.dtype
    metadata: AllGatherMetadata
    all_gather_result: Optional[AllGatherResult] = None
    flat_param_buffer: Optional[torch.Tensor] = None

    def init_flat_param_buffer(self, device: Optional[torch.device]) -> None:
        """Rebase this bucket's homogeneous shards into one persistent flat buffer.

        Side effect: rebinds every parameter's ``_sharded_param_data`` and
        ``sharded_param`` onto a view of the new buffer, so the original shard
        storages are dropped. Leaves ``flat_param_buffer`` as ``None`` when the
        bucket mixes storage dtypes or holds offloaded/meta parameters, which is
        the signal that the zero-copy path does not apply.
        """
        storage_dtype = self.hsdp_params[0]._sharded_param_data.dtype
        if any(
            hsdp_param._sharded_param_data.dtype != storage_dtype
            for hsdp_param in self.hsdp_params[1:]
        ):
            self.flat_param_buffer = None
            return
        if any(
            hsdp_param.offload_to_cpu or hsdp_param.sharded_param.device.type == "meta"
            for hsdp_param in self.hsdp_params
        ):
            self.flat_param_buffer = None
            return

        total_numel = sum(hsdp_param._sharded_param_data.numel() for hsdp_param in self.hsdp_params)
        flat_param_buffer = torch.empty(total_numel, dtype=storage_dtype, device=device)
        flat_offset = 0
        for hsdp_param in self.hsdp_params:
            param_numel = hsdp_param._sharded_param_data.numel()
            flat_param_view = flat_param_buffer.narrow(0, flat_offset, param_numel)
            flat_param_view.copy_(hsdp_param._sharded_param_data)
            hsdp_param._sharded_param_data = flat_param_view
            padded_local_param = flat_param_view.view(hsdp_param.padded_sharded_param_size)
            local_param = padded_local_param.narrow(
                hsdp_param.hsdp_placement.dim,
                0,
                hsdp_param.sharded_size[hsdp_param.hsdp_placement.dim],
            )
            requires_grad = hsdp_param.sharded_param.requires_grad
            hsdp_param.sharded_param._local_tensor = local_param
            hsdp_param.sharded_param.data = local_param
            if requires_grad:
                local_param.requires_grad_(True)
                hsdp_param.sharded_param.requires_grad_(True)
            flat_offset += param_numel
        self.flat_param_buffer = flat_param_buffer

    def is_flat_buffer_valid(self) -> bool:
        """Return whether this bucket's parameter shards still use its flat storage.

        Parameter storage can be replaced behind this bucket's back (optimizer
        surgery, ``load_state_dict``), which silently invalidates the zero-copy
        rebase; comparing storage pointers is what detects that.
        """
        if self.flat_param_buffer is None:
            return False
        flat_storage_ptr = self.flat_param_buffer.untyped_storage().data_ptr()
        return all(
            hsdp_param._sharded_param_data.untyped_storage().data_ptr() == flat_storage_ptr
            for hsdp_param in self.hsdp_params
        )

    @torch.no_grad()
    def copy_out(self) -> None:
        """Wait this bucket's all-gather and copy it into stable unsharded buffers.

        Blocks on the in-flight handle, then releases the bucket's all-gather
        input/output references so the fused communication buffers can be freed
        before the next bucket runs. No-op when the bucket has no pending result.
        """
        all_gather_result = self.all_gather_result
        if all_gather_result is None or all_gather_result.all_gather_output is None:
            return
        if all_gather_result.handle is not None:
            all_gather_result.handle.wait()
            all_gather_result.handle = None
        all_gather_output = all_gather_result.all_gather_output
        metadata = self.metadata
        split_with_sizes_out = []
        for input_numels, input_dtypes, hsdp_param in zip(
            metadata.param_input_numels,
            metadata.param_input_dtypes,
            self.hsdp_params,
        ):
            hsdp_param.init_unsharded_param_buffers(
                input_numels,
                input_dtypes,
                self.shard_world_size,
                all_gather_output.device,
            )
            hsdp_param.alloc_unsharded_param_buffers()
            split_with_sizes_out.extend(hsdp_param.unsharded_param_buffers)

        gathered_rows = all_gather_output.view(self.shard_world_size, -1)
        output_rows = [
            output_buffer.view(self.shard_world_size, -1)
            for output_buffer in split_with_sizes_out
        ]
        non_inference_outputs = [output for output in output_rows if not output.is_inference()]
        # PyTorch 2.6 accepts only one tensor per context manager. no_grad does
        # not suppress version-counter bumps from the copy-out operations.
        # pylint: disable=W0212
        with ExitStack() as stack:
            for output in non_inference_outputs:
                stack.enter_context(torch.autograd._unsafe_preserve_version_counter(output))
            if all(
                hsdp_param.hsdp_placement.dim == 0
                for hsdp_param in self.hsdp_params
            ):
                torch.split_with_sizes_copy(
                    gathered_rows,
                    metadata.inp_split_sizes,
                    dim=1,
                    out=output_rows,
                )
            else:
                column_offset = 0
                for input_numels, hsdp_param in zip(
                    metadata.param_input_numels,
                    self.hsdp_params,
                ):
                    if hsdp_param.hsdp_placement.dim != 0 and len(input_numels) != 1:
                        raise NotImplementedError(
                            "Fused non-dim-0 all-gather expects one local shard tensor per parameter."
                        )
                    for input_numel, output_buffer in zip(
                        input_numels,
                        hsdp_param.unsharded_param_buffers,
                    ):
                        gathered_param = gathered_rows.narrow(1, column_offset, input_numel)
                        if hsdp_param.hsdp_placement.dim == 0:
                            output_buffer.view(self.shard_world_size, -1).copy_(gathered_param)
                        else:
                            packed_shape = list(hsdp_param.sharded_size)
                            packed_shape[0] *= self.shard_world_size
                            packed_param = gathered_param.contiguous().view(packed_shape)
                            param_chunks = torch.chunk(
                                packed_param,
                                self.shard_world_size,
                                dim=0,
                            )
                            torch.cat(
                                param_chunks,
                                dim=hsdp_param.hsdp_placement.dim,
                                out=output_buffer.view(hsdp_param._orig_size),
                            )
                        column_offset += input_numel
                if column_offset != gathered_rows.size(1):
                    raise AssertionError(
                        "Fused all-gather copy-out consumed an unexpected number of elements: "
                        f"{column_offset} != {gathered_rows.size(1)}"
                    )
        all_gather_result.all_gather_input = None
        all_gather_result.all_gather_output = None
        self.all_gather_result = None



@dataclass
class ReduceScatterBucket:
    """Own one fused reduce-scatter operation and its temporary buffers.

    ``(id(shard_group), dtype)`` is this bucket's identity across micro-steps and
    is the key under which ``HSDPParamGroup.reduce_partial_outputs`` parks the
    output of a non-synchronizing micro-step.
    """

    hsdp_params: list[TorchHSDPParamV2]
    shard_group: Optional[dist.ProcessGroup]
    shard_world_size: int
    dtype: torch.dtype
    reduce_op: dist.ReduceOp
    needs_avg_div: bool
    param_offsets: list[int]
    reduce_scatter_input: Optional[torch.Tensor]
    reduce_scatter_output: Optional[torch.Tensor]
    handle: Optional[dist.Work]

    @property
    def bucket_key(self) -> tuple:
        """Identity of this bucket's (process group, dtype) fusion class."""
        return (id(self.shard_group), self.dtype)


@dataclass
class AllReduceBucket:
    """Own one fused HSDP all-reduce operation and its temporary buffer."""

    hsdp_params: list[TorchHSDPParamV2]
    param_numels: list[int]
    replicate_group: dist.ProcessGroup
    replicate_world_size: int
    dtype: torch.dtype
    reduce_op: dist.ReduceOp
    needs_avg_div: bool
    param_offsets: list[int]
    all_reduce_output: Optional[torch.Tensor] = None
    handle: Optional[dist.Work] = None


@dataclass
class CommContext:
    """Track the two ParamGroup stages in the fused backward pipeline.

    Backward reduction is split into two stages so each can overlap the next
    module's compute:

    - ``pre_param_group``: has an in-flight fused reduce-scatter. The next
      module's post-backward waits it and issues that group's all-reduce.
    - ``all_reduce_param_group``: has an in-flight fused all-reduce. The next
      post-backward waits it and saves the per-parameter gradient views.

    Both fields are process-local and hold at most one group each, so a group
    parked here is guaranteed to be drained by either the next module's
    post-backward or the root backward hook. Leaving a group parked past the end
    of backward would leak its communication buffers, which is why
    ``reset_iter_state`` clears both.
    """

    pre_param_group: Optional["HSDPParamGroup"] = None
    all_reduce_param_group: Optional["HSDPParamGroup"] = None


comm_ctx = CommContext()


def get_comm_ctx() -> CommContext:
    """Return the process-local fused communication context."""
    return comm_ctx


def get_all_gather_metadata(hsdp_params: list[TorchHSDPParamV2]) -> AllGatherMetadata:
    """Build metadata for parameters with one all-gather communication dtype."""
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
        param_input_dtypes=param_input_dtypes,
        param_input_numels=param_input_numels,
        dtype=dtype,
        inp_split_sizes=inp_split_sizes,
        total_input_numel=total_input_numel,
    )


def all_gather_copy_in(
    all_gather_inputs: list[torch.Tensor],
    all_gather_output: torch.Tensor,
    inp_split_sizes: list[int],
    all_gather_input_numel: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy local parameter shards into this rank's fused output slice."""
    all_gather_input = all_gather_output.narrow(
        0,
        all_gather_input_numel * rank,
        all_gather_input_numel,
    )
    copy_destinations = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        # pylint: disable=W0212
        torch._foreach_copy_(copy_destinations, all_gather_inputs)
    return all_gather_input, all_gather_output


def reduce_scatter_copy_in(
    hsdp_params: list[TorchHSDPParamV2],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    """Pack gradients with mixed shard dimensions into a fused RS input."""
    if len(hsdp_params) != len(unsharded_grads):
        raise AssertionError(
            "reduce_scatter_copy_in expects one hsdp_param per unsharded_grad, but got "
            f"{len(hsdp_params)} params and {len(unsharded_grads)} grads"
        )
    packed_rows = reduce_scatter_input.view(world_size, -1)
    column_offset = 0
    with torch.no_grad():
        for hsdp_param, unsharded_grad in zip(hsdp_params, unsharded_grads):
            padded_shard_numel = hsdp_param.padded_sharded_param_size.numel()
            packed_grad_slot = packed_rows.narrow(1, column_offset, padded_shard_numel)
            shard_dim = hsdp_param.hsdp_placement.dim
            if shard_dim == 0:
                # pylint: disable=W0212
                torch._chunk_cat(
                    [unsharded_grad],
                    dim=0,
                    num_chunks=world_size,
                    out=packed_grad_slot,
                )
            else:
                grad_chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
                packed_grad = torch.cat(grad_chunks, dim=0).contiguous().view(world_size, -1)
                packed_grad_slot.copy_(packed_grad)
            column_offset += padded_shard_numel
    if column_offset != packed_rows.size(1):
        raise AssertionError(
            "reduce_scatter_copy_in packed an unexpected number of elements: "
            f"{column_offset} != {packed_rows.size(1)}"
        )


class HSDPParamGroup:
    """Fuse communication for the managed parameters in one fully_shard unit.

    Packs parameter shards into contiguous buckets so each collective is issued
    once per (group, dtype) instead of once per parameter.

    Lifecycle within one training iteration:
        1. Forward -- ``unshard()`` builds the all-gather buckets on first use
           and issues one ``all_gather_into_tensor`` per bucket.
        2. Forward (wait) -- ``wait_for_unshard()`` waits each bucket and copies
           the gathered data out into the per-parameter unsharded buffers.
        3. Backward -- ``foreach_reduce()`` packs the unsharded gradients and
           issues one fused ``reduce_scatter_tensor`` per bucket, then parks
           itself on ``CommContext`` so a later module's backward waits for it.
        4. Backward (finalize) -- ``wait_reduce_scatter_and_issue_all_reduce()``
           waits the RS buckets and launches the HSDP all-reduces;
           ``wait_all_reduce_and_save_grad()`` waits those and exposes each
           parameter's reduced-gradient view.

    Steps 3 and 4 are deliberately split across modules: that is what allows one
    module's communication to overlap the next module's backward compute. The
    root backward hook drains whatever is still parked when backward ends.

    Gradient accumulation is held at bucket granularity in
    ``reduce_partial_outputs``, keyed by the reduce-scatter bucket key. A
    micro-step with ``requires_all_reduce=False`` keeps its whole reduce-scatter
    output there and skips the all-reduce entirely; the next synchronizing
    micro-step adds it back with a single whole-buffer ``add_`` and lets the
    fused all-reduce see the total. The dict lives on the group rather than on
    the bucket because ``foreach_reduce`` rebuilds the buckets every micro-step,
    and holds whole buffers rather than per-parameter views so accumulation
    costs one kernel per bucket instead of one per parameter.
    """

    def __init__(
        self,
        hsdp_params: list[TorchHSDPParamV2],
        device: Optional[torch.device] = None,
        enable_zero_copy: bool = True,
    ) -> None:
        self.device = device
        self.hsdp_params = hsdp_params
        self.enable_zero_copy = enable_zero_copy
        self.gradient_scaling_factor = None
        self.requires_all_reduce = True
        self.all_gather_buckets: list[AllGatherBucket] = []
        self.reduce_scatter_buckets: list[ReduceScatterBucket] = []
        self.all_reduce_buckets: list[AllReduceBucket] = []
        self.reduce_partial_outputs: dict[tuple, torch.Tensor] = {}

    def _init_all_gather_buckets(self) -> None:
        """Build ordered all-gather buckets from each parameter's communication facts."""
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
            bucket_key = (id(shard_group), communication_dtype)
            if bucket_key not in params_by_bucket:
                params_by_bucket[bucket_key] = []
                bucket_groups[bucket_key] = shard_group
            params_by_bucket[bucket_key].append(hsdp_param)

        self.all_gather_buckets = []
        for bucket_key, hsdp_params in params_by_bucket.items():
            shard_group = bucket_groups[bucket_key]
            param_input_numels = [
                [hsdp_param._sharded_param_data.numel()] for hsdp_param in hsdp_params
            ]
            inp_split_sizes = [input_numels[0] for input_numels in param_input_numels]
            self.all_gather_buckets.append(
                AllGatherBucket(
                    hsdp_params=hsdp_params,
                    shard_group=shard_group,
                    shard_rank=shard_group.rank(),
                    shard_world_size=shard_group.size(),
                    dtype=bucket_key[1],
                    metadata=AllGatherMetadata(
                        param_input_dtypes=[[bucket_key[1]] for _ in hsdp_params],
                        param_input_numels=param_input_numels,
                        dtype=bucket_key[1],
                        inp_split_sizes=inp_split_sizes,
                        total_input_numel=sum(inp_split_sizes),
                    ),
                )
            )

    def unshard(self, async_op: bool = False) -> None:
        """Launch fused all-gathers for every communication bucket."""
        if self.all_gather_buckets and any(
            bucket.all_gather_result is not None for bucket in self.all_gather_buckets
        ):
            return
        self.foreach_all_gather(async_op)

    @torch.no_grad()
    def foreach_all_gather(self, async_op: bool = False) -> None:
        """Initialize ordered buckets and launch their all-gathers."""
        if not self.all_gather_buckets:
            self._init_all_gather_buckets()
        for hsdp_param in self.hsdp_params:
            if hsdp_param.shard_world_size <= 1:
                hsdp_param.unshard(async_op)
        for all_gather_bucket in self.all_gather_buckets:
            if all_gather_bucket.all_gather_result is not None:
                continue
            if self.enable_zero_copy and not all_gather_bucket.is_flat_buffer_valid():
                all_gather_bucket.init_flat_param_buffer(self.device)

            metadata = all_gather_bucket.metadata
            all_gather_output = torch.empty(
                metadata.total_input_numel * all_gather_bucket.shard_world_size,
                dtype=all_gather_bucket.dtype,
                device=self.device,
            )
            if self.enable_zero_copy and all_gather_bucket.is_flat_buffer_valid():
                if all_gather_bucket.flat_param_buffer.dtype == all_gather_bucket.dtype:
                    all_gather_input = all_gather_bucket.flat_param_buffer
                else:
                    all_gather_input = all_gather_bucket.flat_param_buffer.to(all_gather_bucket.dtype)
            else:
                all_gather_inputs = []
                for hsdp_param in all_gather_bucket.hsdp_params:
                    all_gather_inputs.extend(hsdp_param.all_gather_inputs)
                all_gather_input, all_gather_output = all_gather_copy_in(
                    all_gather_inputs,
                    all_gather_output,
                    metadata.inp_split_sizes,
                    metadata.total_input_numel,
                    all_gather_bucket.shard_rank,
                )

            handle = dist.all_gather_into_tensor(
                all_gather_output,
                all_gather_input,
                group=all_gather_bucket.shard_group,
                async_op=async_op,
            )
            all_gather_bucket.all_gather_result = AllGatherResult(
                all_gather_input=all_gather_input,
                all_gather_output=all_gather_output,
                handle=handle,
            )

    def wait_for_unshard(self) -> None:
        """Wait all fused all-gathers and install stable unsharded parameters."""
        for all_gather_bucket in self.all_gather_buckets:
            all_gather_bucket.copy_out()
        for hsdp_param in self.hsdp_params:
            if hsdp_param.shard_world_size <= 1:
                hsdp_param.wait_for_unshard()
            else:
                hsdp_param.init_unsharded_param()
                hsdp_param.to_unsharded()

    def _build_reduce_scatter_buckets(
        self,
        reduce_scatter_reduce_op: dist.ReduceOp,
        async_op: bool,
    ) -> list[ReduceScatterBucket]:
        """Pack and launch ordered reduce-scatter buckets for active gradients."""
        params_by_bucket = {}
        grads_by_bucket = {}
        bucket_groups = {}
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
            bucket_key = (id(shard_group), reduce_dtype)
            if bucket_key not in params_by_bucket:
                params_by_bucket[bucket_key] = []
                grads_by_bucket[bucket_key] = []
                bucket_groups[bucket_key] = shard_group
            params_by_bucket[bucket_key].append(hsdp_param)
            grads_by_bucket[bucket_key].append(unsharded_grad)

        reduce_scatter_buckets = []
        for bucket_key, hsdp_params in params_by_bucket.items():
            shard_group = bucket_groups[bucket_key]
            shard_world_size = hsdp_params[0].shard_world_size
            reduce_scatter_input_numel = sum(
                hsdp_param.padded_sharded_param_size.numel() * shard_world_size
                for hsdp_param in hsdp_params
            )
            reduce_scatter_input = torch.empty(
                reduce_scatter_input_numel,
                dtype=bucket_key[1],
                device=grads_by_bucket[bucket_key][0].device,
            )
            reduce_scatter_copy_in(
                hsdp_params,
                grads_by_bucket[bucket_key],
                reduce_scatter_input,
                shard_world_size,
            )
            apply_gradient_scaling_factor(reduce_scatter_input, self.gradient_scaling_factor)
            needs_avg_div = reduce_scatter_reduce_op == dist.ReduceOp.AVG
            communication_op = dist.ReduceOp.SUM if needs_avg_div else reduce_scatter_reduce_op
            if shard_group is None or shard_world_size <= 1:
                reduce_scatter_output = reduce_scatter_input
                handle = None
            else:
                reduce_scatter_output = reduce_scatter_input.new_empty(
                    reduce_scatter_input_numel // shard_world_size
                )
                handle = dist.reduce_scatter_tensor(
                    output=reduce_scatter_output,
                    input=reduce_scatter_input,
                    group=shard_group,
                    op=communication_op,
                    async_op=async_op,
                )
            param_offsets = []
            flat_offset = 0
            for hsdp_param in hsdp_params:
                param_offsets.append(flat_offset)
                flat_offset += hsdp_param.padded_sharded_param_size.numel()
                if hsdp_param.unsharded_accumulated_grad is not None:
                    hsdp_param.unsharded_accumulated_grad = None
                else:
                    hsdp_param.unsharded_param.grad = None
            reduce_scatter_buckets.append(
                ReduceScatterBucket(
                    hsdp_params=hsdp_params,
                    shard_group=shard_group,
                    shard_world_size=shard_world_size,
                    dtype=bucket_key[1],
                    reduce_op=communication_op,
                    needs_avg_div=needs_avg_div,
                    param_offsets=param_offsets,
                    reduce_scatter_input=reduce_scatter_input,
                    reduce_scatter_output=reduce_scatter_output,
                    handle=handle,
                )
            )
        return reduce_scatter_buckets

    @torch.no_grad()
    def foreach_reduce(
        self,
        reduce_scatter_reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG,
        async_op: bool = True,
    ) -> Optional[torch.Tensor]:
        """Launch fused reduce-scatter buckets for this module's gradients."""
        self.reduce_scatter_buckets = self._build_reduce_scatter_buckets(
            reduce_scatter_reduce_op,
            async_op,
        )
        if not self.reduce_scatter_buckets:
            return None
        if async_op:
            comm_ctx.pre_param_group = self
        else:
            self.wait_reduce_scatter_and_issue_all_reduce(async_op=False)
            self.wait_all_reduce_and_save_grad()
        return None

    @staticmethod
    def _reduced_grad_view(
        reduce_scatter_bucket: ReduceScatterBucket,
        param_index: int,
    ) -> torch.Tensor:
        """Return one padded parameter view from a completed RS output."""
        if reduce_scatter_bucket.reduce_scatter_output is None:
            raise RuntimeError("Reduce-scatter output has already been released.")
        hsdp_param = reduce_scatter_bucket.hsdp_params[param_index]
        return reduce_scatter_bucket.reduce_scatter_output.narrow(
            0,
            reduce_scatter_bucket.param_offsets[param_index],
            hsdp_param.padded_sharded_param_size.numel(),
        )

    def _build_all_reduce_buckets(self) -> list[AllReduceBucket]:
        """Wait RS buckets and organize completed grads by replicate group and dtype.

        A non-synchronizing micro-step (``requires_all_reduce=False``) returns no
        all-reduce buckets: each RS output is parked whole in
        ``reduce_partial_outputs`` and added back by the next synchronizing
        micro-step. Accumulation therefore costs one ``add_`` per bucket, and the
        parked buffer is the only copy of those gradients until then -- it must
        outlive this call, so it is excluded from the RS-output release below.

        The all-reduce buffer is the reduce-scatter output itself whenever a whole
        RS bucket maps onto a single AR bucket: the reduced shards are already
        contiguous there in the same order, so all-reducing in place saves one
        full-size allocation and one full copy per bucket per micro-step. A fresh
        buffer is packed only when an AR bucket draws from more than one RS bucket,
        which happens when RS grouped by ``(shard group, dtype)`` splits parameters
        that share a replicate group.

        HSDP issues both collectives at one reduce dtype, so an AR bucket inherits
        its source RS bucket's ``dtype``, ``reduce_op`` and ``needs_avg_div``
        rather than re-deriving them.
        """
        params_by_bucket = {}
        grads_by_bucket = {}
        bucket_groups = {}
        reduce_ops = {}
        needs_avg_div = {}
        source_buckets_by_key = {}
        for reduce_scatter_bucket in self.reduce_scatter_buckets:
            if reduce_scatter_bucket.handle is not None:
                reduce_scatter_bucket.handle.wait()
                reduce_scatter_bucket.handle = None
            reduce_scatter_bucket.reduce_scatter_input = None
            if reduce_scatter_bucket.needs_avg_div:
                reduce_scatter_bucket.reduce_scatter_output.div_(
                    reduce_scatter_bucket.shard_world_size
                )
            bucket_key = reduce_scatter_bucket.bucket_key
            if not self.requires_all_reduce:
                reduce_partial_output = self.reduce_partial_outputs.get(bucket_key)
                if reduce_partial_output is None:
                    self.reduce_partial_outputs[bucket_key] = (
                        reduce_scatter_bucket.reduce_scatter_output
                    )
                else:
                    reduce_partial_output.add_(reduce_scatter_bucket.reduce_scatter_output)
                continue
            reduce_partial_output = self.reduce_partial_outputs.pop(bucket_key, None)
            if reduce_partial_output is not None:
                reduce_scatter_bucket.reduce_scatter_output.add_(reduce_partial_output)
            for param_index, hsdp_param in enumerate(reduce_scatter_bucket.hsdp_params):
                reduced_grad = self._reduced_grad_view(reduce_scatter_bucket, param_index)
                replicate_group = (
                    hsdp_param.mesh_info.replicate_process_group
                    if isinstance(hsdp_param.mesh_info, DDPMeshInfo)
                    else None
                )
                if replicate_group is None or hsdp_param.replicate_world_size <= 1:
                    hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = reduced_grad
                    continue
                all_reduce_key = (id(replicate_group), reduce_scatter_bucket.dtype)
                if all_reduce_key not in params_by_bucket:
                    params_by_bucket[all_reduce_key] = []
                    grads_by_bucket[all_reduce_key] = []
                    bucket_groups[all_reduce_key] = replicate_group
                    reduce_ops[all_reduce_key] = reduce_scatter_bucket.reduce_op
                    needs_avg_div[all_reduce_key] = reduce_scatter_bucket.needs_avg_div
                    source_buckets_by_key[all_reduce_key] = []
                params_by_bucket[all_reduce_key].append(hsdp_param)
                grads_by_bucket[all_reduce_key].append(reduced_grad)
                if reduce_scatter_bucket not in source_buckets_by_key[all_reduce_key]:
                    source_buckets_by_key[all_reduce_key].append(reduce_scatter_bucket)
        drained_reduce_scatter_buckets = self.reduce_scatter_buckets
        self.reduce_scatter_buckets = []

        all_reduce_buckets = []
        for bucket_key, hsdp_params in params_by_bucket.items():
            replicate_group = bucket_groups[bucket_key]
            reduced_grads = grads_by_bucket[bucket_key]
            param_offsets = []
            flat_offset = 0
            for reduced_grad in reduced_grads:
                param_offsets.append(flat_offset)
                flat_offset += reduced_grad.numel()
            param_numels = [reduced_grad.numel() for reduced_grad in reduced_grads]
            all_reduce_output = self._reuse_reduce_scatter_output(
                source_buckets_by_key[bucket_key],
                hsdp_params,
                flat_offset,
            )
            if all_reduce_output is None:
                all_reduce_output = torch.empty(
                    flat_offset,
                    dtype=bucket_key[1],
                    device=reduced_grads[0].device,
                )
                for reduced_grad, param_offset in zip(reduced_grads, param_offsets):
                    all_reduce_output.narrow(0, param_offset, reduced_grad.numel()).copy_(reduced_grad)
            all_reduce_buckets.append(
                AllReduceBucket(
                    hsdp_params=hsdp_params,
                    param_numels=param_numels,
                    replicate_group=replicate_group,
                    replicate_world_size=replicate_group.size(),
                    dtype=bucket_key[1],
                    reduce_op=reduce_ops[bucket_key],
                    needs_avg_div=needs_avg_div[bucket_key],
                    param_offsets=param_offsets,
                    all_reduce_output=all_reduce_output,
                )
            )
        retained_output_ids = {
            id(all_reduce_bucket.all_reduce_output) for all_reduce_bucket in all_reduce_buckets
        }
        retained_output_ids.update(
            id(reduce_partial_output) for reduce_partial_output in self.reduce_partial_outputs.values()
        )
        for reduce_scatter_bucket in drained_reduce_scatter_buckets:
            if id(reduce_scatter_bucket.reduce_scatter_output) not in retained_output_ids:
                reduce_scatter_bucket.reduce_scatter_output = None
        return all_reduce_buckets

    @staticmethod
    def _reuse_reduce_scatter_output(
        source_buckets: list["ReduceScatterBucket"],
        hsdp_params: list[TorchHSDPParamV2],
        flat_offset: int,
    ) -> Optional[torch.Tensor]:
        """Return the RS output usable as this AR bucket's buffer, else ``None``.

        Reuse is only sound when one RS bucket contributed every parameter of this
        AR bucket, in the same order and covering the whole RS output: the AR then
        reduces exactly the region the RS wrote, so it can run in place and skip a
        full-size allocation plus a full copy. Any parameter routed elsewhere (no
        replicate group, or the replicate group split across RS buckets) breaks
        that correspondence and forces a freshly packed buffer.
        """
        if len(source_buckets) != 1:
            return None
        source_bucket = source_buckets[0]
        if source_bucket.hsdp_params != hsdp_params:
            return None
        reduce_scatter_output = source_bucket.reduce_scatter_output
        if reduce_scatter_output.numel() != flat_offset:
            return None
        return reduce_scatter_output

    def wait_reduce_scatter_and_issue_all_reduce(self, async_op: bool = True) -> None:
        """Wait all RS buckets and launch the resulting HSDP all-reduces."""
        self.all_reduce_buckets = self._build_all_reduce_buckets()
        for all_reduce_bucket in self.all_reduce_buckets:
            all_reduce_bucket.handle = dist.all_reduce(
                all_reduce_bucket.all_reduce_output,
                group=all_reduce_bucket.replicate_group,
                op=all_reduce_bucket.reduce_op,
                async_op=async_op,
            )
        if self.all_reduce_buckets:
            comm_ctx.all_reduce_param_group = self

    def wait_all_reduce_and_save_grad(self) -> None:
        """Wait HSDP all-reduces and expose their per-parameter output views."""
        for all_reduce_bucket in self.all_reduce_buckets:
            if all_reduce_bucket.handle is not None:
                all_reduce_bucket.handle.wait()
                all_reduce_bucket.handle = None
            if all_reduce_bucket.needs_avg_div:
                all_reduce_bucket.all_reduce_output.div_(all_reduce_bucket.replicate_world_size)
            for hsdp_param, param_numel, param_offset in zip(
                all_reduce_bucket.hsdp_params,
                all_reduce_bucket.param_numels,
                all_reduce_bucket.param_offsets,
            ):
                hsdp_param.all_reduce_comm_ctx.all_reduce_output = all_reduce_bucket.all_reduce_output.narrow(
                    0,
                    param_offset,
                    param_numel,
                )
            all_reduce_bucket.all_reduce_output = None
        self.all_reduce_buckets = []
        if comm_ctx.all_reduce_param_group is self:
            comm_ctx.all_reduce_param_group = None

    def reset_iter_state(self) -> None:
        """Drop communication references after a completed iteration."""
        for all_gather_bucket in self.all_gather_buckets:
            if all_gather_bucket.all_gather_result is not None:
                all_gather_bucket.all_gather_result.all_gather_input = None
                all_gather_bucket.all_gather_result.all_gather_output = None
                all_gather_bucket.all_gather_result.handle = None
                all_gather_bucket.all_gather_result = None
        for reduce_scatter_bucket in self.reduce_scatter_buckets:
            reduce_scatter_bucket.reduce_scatter_input = None
            reduce_scatter_bucket.reduce_scatter_output = None
            reduce_scatter_bucket.handle = None
        for all_reduce_bucket in self.all_reduce_buckets:
            all_reduce_bucket.all_reduce_output = None
            all_reduce_bucket.handle = None
        self.reduce_scatter_buckets = []
        self.all_reduce_buckets = []
        self.reduce_partial_outputs.clear()
        if comm_ctx.pre_param_group is self:
            comm_ctx.pre_param_group = None
        if comm_ctx.all_reduce_param_group is self:
            comm_ctx.all_reduce_param_group = None


class AllReduceParamGroup:
    """Group HSDP parameters by replicate group for one fused async all-reduce.

    Used by the ``comm_fusion=False`` path only: there each parameter issues its
    own reduce-scatter, and this class re-fuses the resulting shards so the
    replicate-dimension all-reduce is still a single collective. The
    ``comm_fusion=True`` path never builds this group -- it fuses at the bucket
    level via ``AllReduceBucket``.

    Zero-copy is achieved by pre-allocating one contiguous buffer with 512-byte
    alignment, having each ``reduce_scatter_grad`` write straight into an aligned
    view of it, all-reducing the whole buffer once, and slicing gradients back
    out of those views.

    Numerical-correctness constraints (each is load-bearing, do not "simplify"):

    - The all-reduce always uses SUM, never AVG. The buffer is padded for
      alignment, and averaging would divide by a world size that the zero
      padding does not participate in.
    - AVG is therefore reconstructed manually in ``wait_and_split_grads`` by
      dividing by ``replicate_world_size`` after the collective.
    - The buffer is zero-initialized so the padding regions contribute nothing
      to the SUM.

    Attributes:
        replicate_group: Process group for the replicate dimension.
        hsdp_params: Parameters sharing that replicate group.
        reduce_dtype: Uniform element type of the fused buffer.
        reduce_op: Caller-requested op (AVG or SUM); decides the final scaling.
        replicate_world_size: Size of the replicate group.
        fused_buffer: Contiguous aligned buffer for all parameters; released
            once ``wait_and_split_grads`` has handed out its views.
        param_offsets: Element offset of each parameter inside the buffer.
        param_numels: Padded element count of each parameter.
        all_reduce_handle: In-flight async all-reduce work, or None.
    """

    ALIGNMENT_BYTES = 512

    def __init__(
        self,
        replicate_group: dist.ProcessGroup,
        hsdp_params: List[TorchHSDPParamV2],
        reduce_op: dist.ReduceOp,
    ) -> None:
        self.replicate_group = replicate_group
        self.hsdp_params = hsdp_params
        self.reduce_dtype = hsdp_params[0].reduce_comm_dtype()
        self.reduce_op = reduce_op
        self.replicate_world_size = replicate_group.size() if replicate_group else 1
        self.fused_buffer: Optional[torch.Tensor] = None
        self.param_offsets: List[int] = []
        self.param_numels: List[int] = []
        self.all_reduce_handle: Optional[dist.Work] = None

    def compute_aligned_layout(self) -> int:
        """Compute packed parameter offsets and align the total buffer size."""
        self.param_offsets = []
        self.param_numels = []
        element_size = torch.tensor([], dtype=self.reduce_dtype).element_size()
        current_offset = 0
        for hsdp_param in self.hsdp_params:
            numel = hsdp_param.padded_sharded_param_size.numel()
            self.param_numels.append(numel)
            self.param_offsets.append(current_offset)
            current_offset += numel
        total_bytes = current_offset * element_size
        aligned_total_bytes = (
            (total_bytes + self.ALIGNMENT_BYTES - 1) // self.ALIGNMENT_BYTES
        ) * self.ALIGNMENT_BYTES
        return aligned_total_bytes // element_size

    def allocate_fused_buffer(self, device: torch.device) -> None:
        """Allocate and zero the fused all-reduce buffer."""
        self.fused_buffer = torch.zeros(
            self.compute_aligned_layout(),
            dtype=self.reduce_dtype,
            device=device,
        )

    def get_param_buffer_view(self, index: int) -> torch.Tensor:
        """Return one parameter's padded communication view."""
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer not allocated. Call allocate_fused_buffer first.")
        return self.fused_buffer.narrow(
            0,
            self.param_offsets[index],
            self.param_numels[index],
        )

    def get_param_grad_view(self, index: int, target_shape: torch.Size) -> torch.Tensor:
        """Return one parameter's actual-shard gradient view."""
        return self.get_param_buffer_view(index).narrow(0, 0, target_shape.numel()).view(target_shape)

    def accumulate_reduce_partial_outputs(self) -> None:
        """Merge no-all-reduce micro-step outputs into the current buffer."""
        if self.fused_buffer is None:
            return
        for index, hsdp_param in enumerate(self.hsdp_params):
            if hsdp_param.reduce_partial_output is None:
                continue
            partial_output = hsdp_param.reduce_partial_output
            if partial_output.dtype != self.reduce_dtype:
                partial_output = partial_output.to(self.reduce_dtype)
            self.get_param_buffer_view(index).add_(partial_output.view(-1))
            hsdp_param.reduce_partial_output = None

    def issue_async_allreduce(self) -> None:
        """Launch SUM all-reduce; AVG division is applied when splitting."""
        if self.fused_buffer is None:
            raise RuntimeError("Fused buffer not allocated.")
        self.all_reduce_handle = dist.all_reduce(
            self.fused_buffer,
            op=dist.ReduceOp.SUM,
            group=self.replicate_group,
            async_op=True,
        )

    def wait_and_split_grads(self) -> None:
        """Wait all-reduce and expose per-parameter context views."""
        if self.all_reduce_handle is not None:
            self.all_reduce_handle.wait()
            self.all_reduce_handle = None
        for index, hsdp_param in enumerate(self.hsdp_params):
            reduced_grad = self.get_param_grad_view(index, hsdp_param.sharded_size)
            if self.reduce_op == dist.ReduceOp.AVG and self.replicate_world_size > 1:
                reduced_grad.div_(self.replicate_world_size)
            hsdp_param.all_reduce_comm_ctx.all_reduce_output = reduced_grad
        self.fused_buffer = None

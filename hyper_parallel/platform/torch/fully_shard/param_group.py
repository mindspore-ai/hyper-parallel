# Copyright 2025 Huawei Technologies Co., Ltd
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

# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
# enhanced with fully_shard parameter management
# ============================================================================
"""HSDP parameter group.

This module implements fused communication for HSDP (Hybrid Shard Data Parallel) parameters.
Instead of issuing one all-gather / reduce-scatter per parameter, ``HSDPParamGroup`` packs all
parameters within a module into a single contiguous buffer and performs one collective operation,
which reduces kernel launch overhead and improves bandwidth utilization.

Key components:
- ``HSDPParamGroup``: Groups all HSDP parameters in a module for fused all-gather (forward)
  and fused reduce-scatter + all-reduce (backward).
- ``AllGatherMetadata`` / ``AllGatherMetadataCache``: Caches per-group metadata (dtypes, numels,
  split sizes) to avoid recomputation across iterations.
- ``CommContext``: Global context that tracks the in-flight async communication handle and the
  param group that owns it, enabling pipelined overlap between communication and computation.
"""
from typing import List, Optional, NamedTuple
from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    FSDPMeshInfo,
    DDPMeshInfo,
    HSDPMeshInfo,
)
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2


def get_all_gather_metadata(hsdp_params):
    """Collect metadata required for fused all-gather from all HSDP parameters.

    Iterates over each parameter's local shard inputs and records their dtypes and
    element counts. All parameters must share the same dtype (heterogeneous dtypes
    are not yet supported).

    Args:
        hsdp_params: List of ``TorchHSDPParamV2`` whose ``all_gather_inputs`` will
            be inspected.

    Returns:
        AllGatherMetadata: Aggregated metadata used by ``foreach_all_gather`` to
            allocate the fused output buffer and perform copy-in/copy-out.

    Raises:
        ValueError: If parameters have different dtypes.
    """
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
        total_input_numel
    )


@dataclass
class AllGatherMetadata:
    """Metadata describing the fused all-gather buffer layout.

    Attributes:
        param_input_dtypes: Per-parameter list of input tensor dtypes.
        param_input_numels: Per-parameter list of input tensor element counts.
        dtype: Uniform dtype of all inputs (used to allocate the fused buffer).
        inp_split_sizes: Flat list of element counts for each input tensor across
            all parameters, used by ``torch.split`` / ``split_with_sizes_copy`` to
            slice the fused buffer back into per-parameter chunks.
        total_input_numel: Total number of elements from all local shards (one rank's
            contribution); the full all-gather output has ``total_input_numel * world_size``
            elements.
        hash_key: Computed in ``__post_init__`` for use as a cache key.
    """
    param_input_dtypes: list[list[torch.dtype]]
    param_input_numels: list[list[int]]
    dtype: torch.dtype
    inp_split_sizes: list[int]
    total_input_numel: int
    hash_key: int = field(init=False)

    def __post_init__(self):
        self.hash_key = hash((
            tuple(tuple(d) for d in self.param_input_dtypes),
            tuple(tuple(n) for n in self.param_input_numels),
            self.dtype,
            tuple(self.inp_split_sizes),
            self.total_input_numel
        ))


class AllGatherResult(NamedTuple):
    """Result of a fused all-gather operation.

    Attributes:
        all_gather_output: The contiguous output buffer holding gathered data from all ranks.
        metadata: The ``AllGatherMetadata`` used to interpret the buffer layout.
        handle: Async work handle from ``dist.all_gather_into_tensor``; ``None`` when
            the operation was synchronous or when ``shard_world_size == 1``.
    """
    all_gather_output: torch.Tensor
    metadata: AllGatherMetadata
    handle: dist.distributed_c10d.Work | None


@dataclass
class CommContext:
    """Global communication context for pipelining fused gradient reduction.

    For FSDP (shard-only), the reduce-scatter handle is stored in ``comm_handle``
    and the next module's backward hook waits on it before issuing its own reduction.

    For HSDP (shard + replicate), a two-phase pipeline is used:
        Phase 1 (``wait_reduce_scatter_and_issue_all_reduce``): wait for
            reduce-scatter, then issue an async all-reduce and store the handle
            in ``all_reduce_handle``.
        Phase 2 (``wait_all_reduce_and_apply_grad``): wait for all-reduce and
            write reduced gradients back.

    This allows three-way overlap:
        Layer N reduce_scatter ↔ Layer N-1 backward compute
        Layer N all_reduce     ↔ Layer N-1 reduce_scatter
    """
    comm_handle: dist.distributed_c10d.Work | None = None
    all_reduce_handle: dist.distributed_c10d.Work | None = None
    pre_param_group = None
    # Param group whose all_reduce has been issued but grad not yet applied
    all_reduce_param_group = None


comm_ctx = CommContext()


def get_comm_ctx():
    """Return the global ``CommContext`` singleton."""
    return comm_ctx


class AllGatherMetadataCache:
    """Cache for ``AllGatherMetadata`` to avoid recomputation across iterations.

    The cache key is derived from ``(id(param), param.version)`` tuples so that
    it invalidates automatically when parameters are re-sharded or replaced.
    """
    _cache: dict[int, AllGatherMetadata] = {}

    @classmethod
    def get_metadata(cls, hsdp_params, fn):
        """Return cached metadata or compute via *fn* and cache the result."""
        param_key = tuple((id(p), getattr(p, 'version', 0)) for p in hsdp_params)
        key = hash(param_key)

        if key in cls._cache:
            return cls._cache[key]
        metadata = fn(hsdp_params)
        cls._cache[key] = metadata
        return metadata


def all_gather_copy_in(all_gather_inputs, all_gather_output, inp_split_sizes, all_gather_input_numel, rank):
    """Copy per-parameter local shards into the fused all-gather input buffer.

    The fused output buffer has shape ``(total_input_numel * world_size,)``. Each rank
    writes its local shards into the slice ``[input_numel * rank : input_numel * (rank+1)]``
    using ``torch._foreach_copy_`` for efficient batched copy.

    Args:
        all_gather_inputs: Flat list of local shard tensors from all parameters.
        all_gather_output: The pre-allocated fused output buffer.
        inp_split_sizes: Element counts for splitting the rank-local slice.
        all_gather_input_numel: Total elements for one rank's local shards.
        rank: This rank's index within the shard process group.

    Returns:
        Tuple of (rank-local input slice, full output buffer).
    """
    all_gather_input = all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        # pylint: disable=W0212
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


def reduce_scatter_copy_in(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    """Pack unsharded gradients into the fused reduce-scatter input buffer.

    Uses ``torch._chunk_cat`` to interleave chunks from each gradient tensor so that
    the buffer layout matches what ``dist.reduce_scatter_tensor`` expects: the buffer
    is viewed as ``(world_size, total_numel // world_size)`` where row *i* contains
    the slice destined for rank *i* after reduction.

    Args:
        unsharded_grads: Full (unsharded) gradients from all parameters.
        reduce_scatter_input: Pre-allocated flat buffer of size ``sum(g.numel() for g in unsharded_grads)``.
        world_size: Number of ranks in the shard process group.
    """
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    # pylint: disable=W0212
    torch._chunk_cat(unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input)


class HSDPParamGroup:
    """Groups all HSDP parameters within a module for fused collective communication.

    Instead of issuing per-parameter all-gather (forward) and reduce-scatter (backward),
    this class packs all parameter shards into a single contiguous buffer and performs one
    fused collective, reducing NCCL/HCCL kernel launch overhead.

    Lifecycle within one training iteration:
        1. **Forward** — ``unshard()`` → ``foreach_all_gather()`` packs local shards into
           ``ag_output`` and issues a single ``all_gather_into_tensor``.
        2. **Forward (wait)** — ``wait_for_unshard()`` → ``foreach_all_gather_copy_out()``
           waits on the handle and scatters gathered data back to per-parameter buffers.
        3. **Backward** — ``foreach_reduce()`` packs unsharded gradients, issues fused
           ``reduce_scatter_tensor`` (+ optional ``all_reduce`` for HSDP replicate dim),
           and stores the handle in ``CommContext`` for pipelined overlap.
        4. **Backward (apply)** — ``apply_fusion_reduced_grad()`` waits on the handle and
           writes reduced gradient slices back to each parameter's ``.grad`` or ``.main_grad``.

    Args:
        hsdp_params: List of ``TorchHSDPParamV2`` belonging to this module.
        mesh_info: Mesh info providing shard/replicate process groups.
        device: Target device for buffer allocation.
        mp_policy: Mixed-precision policy controlling reduce dtype and grad dtype.
    """

    def __init__(
        self,
        hsdp_params,
        mesh_info: FSDPMeshInfo,
        device: Optional[torch.device] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None
    ):
        self.mesh_info = mesh_info
        self.device = device
        self.hsdp_params = hsdp_params
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
        self.device = device
        self._all_gather_output = torch.empty(0, device=self.device)
        self.ag_output = None  # Fused all-gather output buffer, lazily allocated
        self.metadata_cache = None
        self.mp_policy = mp_policy
        self._result = None  # Pending AllGatherResult from async all-gather
        self._reduce_output = None  # Fused reduce-scatter output, consumed by apply_fusion_reduced_grad
        self._reduce_op = None  # Reduce op saved from foreach_reduce for use in apply_fusion_reduced_grad
        self._needs_avg_div = False  # Whether AVG was split into SUM + deferred div
        self._reduce_hsdp_params = None
        self._init_mp_dtypes()
        self._flat_param_buffer = None  # Contiguous buffer holding all params' sharded data
        self._flat_cast_buffer = None  # Cast buffer for mixed precision (param_dtype)
        self._init_flat_param_buffer()

    def _init_flat_param_buffer(self):
        """Initialize a contiguous flat buffer and rebase all params' sharded data into it.

        This enables zero-copy all-gather by making all local shards contiguous in memory,
        so they can be passed directly to ``all_gather_into_tensor`` without ``foreach_copy_``.
        When mixed-precision casting is needed, a separate cast buffer is also allocated.
        """
        if self.shard_world_size <= 1:
            return
        if len(self.hsdp_params) == 0:
            return
        if any(p.offload_to_cpu or p.sharded_param.device.type == "meta" for p in self.hsdp_params):
            return

        total_numel = sum(p._sharded_param_data.numel() for p in self.hsdp_params)
        orig_dtype = self.hsdp_params[0]._sharded_param_data.dtype
        flat_buffer = torch.empty(total_numel, dtype=orig_dtype, device=self.device)

        offset = 0
        for hsdp_param in self.hsdp_params:
            numel = hsdp_param._sharded_param_data.numel()
            flat_slice = flat_buffer.narrow(0, offset, numel)
            flat_slice.copy_(hsdp_param._sharded_param_data)
            # Rebase _sharded_param_data to be a view into the flat buffer
            hsdp_param._sharded_param_data = flat_slice
            # Rebase DTensor's local tensor so optimizer in-place updates write to flat buffer
            new_local = flat_slice.view(hsdp_param.sharded_size)
            req_grad = hsdp_param.sharded_param.requires_grad
            hsdp_param.sharded_param._local_tensor = new_local
            hsdp_param.sharded_param.data = new_local
            if req_grad:
                new_local.requires_grad_(True)
                hsdp_param.sharded_param.requires_grad_(True)
            offset += numel

        self._flat_param_buffer = flat_buffer

        # Allocate cast buffer for mixed precision if needed
        has_param_dtype = any(p.param_dtype is not None for p in self.hsdp_params)
        if has_param_dtype:
            cast_dtype = next(p.param_dtype for p in self.hsdp_params if p.param_dtype is not None)
            self._flat_cast_buffer = torch.empty(total_numel, dtype=cast_dtype, device=self.device)

    def _is_flat_buffer_valid(self):
        """Check if the flat buffer is still backing the params' sharded data.

        The flat buffer becomes invalid after ``load_state_dict`` triggers
        ``reset_sharded_param``, which re-assigns ``_sharded_param_data``.
        """
        if self._flat_param_buffer is None or len(self.hsdp_params) == 0:
            return False
        return self.hsdp_params[0]._sharded_param_data.data_ptr() == self._flat_param_buffer.data_ptr()

    def unshard(self, async_op: bool = False):
        """Trigger fused all-gather to reconstruct full parameters from shards.

        If a prefetch has already been issued (``_result is not None``), this is a no-op.
        For ``shard_world_size == 1`` (no sharding), skips the collective entirely.

        Args:
            async_op: If True, the all-gather runs asynchronously and must be
                completed later via ``wait_for_unshard()``.
        """
        # Already prefetched — skip
        if self._result is not None:
            return
        if self.shard_world_size == 1:
            self._result = AllGatherResult(self._all_gather_output, None, None)
            return
        self.foreach_all_gather(async_op=async_op)

    def _init_mp_dtypes(self):
        """Initialize and validate mixed-precision dtypes across all trainable parameters.

        All trainable parameters in the group must have a uniform ``orig_dtype`` and
        ``reduce_dtype``; heterogeneous dtypes would cause incorrect buffer slicing.
        """
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[TorchHSDPParamV2] = [
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
        """Wait for the async all-gather to complete and scatter data to per-parameter buffers.

        For ``shard_world_size == 1``, simply copies the local shard as the full parameter.
        Otherwise, calls ``foreach_all_gather_copy_out`` to split the fused buffer and
        write each parameter's all-gather output. Finally, initializes unsharded parameters.
        """
        if self._result is None:
            return
        if self.shard_world_size == 1:
            for hsdp_param in self.hsdp_params:
                all_gather_input = hsdp_param.all_gather_inputs[0]
                hsdp_param.init_all_gather_outputs(
                    [all_gather_input.numel()],
                    [all_gather_input.dtype],
                    self.shard_world_size,
                    self.device
                )
                hsdp_param.alloc_all_gather_outputs()
                # pylint: disable=W0212
                with torch.autograd._unsafe_preserve_version_counter(hsdp_param.all_gather_outputs[0]):
                    # pylint: disable=W0212
                    hsdp_param.all_gather_outputs[0].copy_(all_gather_input)
        else:
            self.foreach_all_gather_copy_out()
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_unsharded_param()
            hsdp_param.to_unsharded()

    def alloc_all_gather_output(self, total_output_numel):
        """Resize the fused all-gather buffer storage to fit ``total_output_numel`` elements.

        Uses ``untyped_storage().resize_()`` to avoid reallocating the tensor object,
        enabling storage reuse across iterations.
        """
        storage = self.ag_output.untyped_storage()
        expected_size = total_output_numel * self.ag_output.itemsize
        if storage.size() != expected_size:
            storage.resize_(expected_size)

    def free_all_gather_output(self):
        """Release device memory of the fused all-gather buffer by resizing storage to 0."""
        storage = self.ag_output.untyped_storage()
        if storage.size() != 0:
            storage.resize_(0)

    @torch.no_grad()
    def foreach_all_gather(self, async_op=False):
        """Perform a fused all-gather for all parameters in the group.

        When a flat parameter buffer is available (see ``_init_flat_param_buffer``),
        the local shards are already contiguous and can be passed directly to
        ``all_gather_into_tensor`` without any copy-in.  Otherwise falls back to
        the ``all_gather_copy_in`` path.

        Args:
            async_op: If True, the collective runs asynchronously.
        """
        if self.metadata_cache is None:
            self.metadata_cache = AllGatherMetadataCache()
        # pylint: disable=W0108
        metadata = self.metadata_cache.get_metadata(self.hsdp_params, lambda p: get_all_gather_metadata(p))
        if metadata.total_input_numel == 0:
            return
        world_size, rank = self.shard_group.size(), self.shard_group.rank()
        total_output_numel = metadata.total_input_numel * world_size
        if self.ag_output is None:
            self.ag_output = torch.empty(size=(total_output_numel,),
                                         dtype=metadata.dtype, device=self.device)
        else:
            self.alloc_all_gather_output(total_output_numel)

        if not self._is_flat_buffer_valid():
            self._init_flat_param_buffer()
        use_flat_buffer = self._flat_param_buffer is not None
        if use_flat_buffer:
            # Zero-copy path: flat buffer already holds contiguous shard data
            if self._flat_cast_buffer is not None:
                # Mixed precision: single contiguous cast instead of N small copies
                self._flat_cast_buffer.copy_(self._flat_param_buffer)
                all_gather_input = self._flat_cast_buffer
            else:
                all_gather_input = self._flat_param_buffer
        else:
            # Fallback: collect inputs and copy into the rank-local slice of ag_output
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
                rank
            )
            del all_gather_inputs  # Free references to individual shard tensors

        handle = dist.all_gather_into_tensor(self.ag_output, all_gather_input, self.shard_group, async_op)
        self._result = AllGatherResult(self.ag_output, metadata, handle)

    @torch.no_grad()
    def foreach_all_gather_copy_out(self):
        """Wait for the fused all-gather and scatter results back to per-parameter buffers.

        After the collective completes, the fused output is viewed as ``(world_size, -1)``
        and split along dim=1 according to ``inp_split_sizes``. Each slice is copied into
        the corresponding parameter's ``all_gather_outputs`` buffer using
        ``split_with_sizes_copy`` for zero-extra-allocation copy-out.

        Version counters are preserved via ``_unsafe_preserve_version_counter`` to avoid
        triggering autograd version checks on parameter tensors that alias these buffers.
        """
        (ag_output, metadata, _) = self._result
        if self._result.handle is not None:
            self._result.handle.wait()
        device = ag_output.device
        world_size = self.shard_group.size()
        split_with_sizes_out = []
        for input_numels, input_dtypes, hsdp_param in zip(
            metadata.param_input_numels, metadata.param_input_dtypes, self.hsdp_params
        ):
            hsdp_param.init_all_gather_outputs(input_numels, input_dtypes, world_size, device)
            hsdp_param.alloc_all_gather_outputs()
            split_with_sizes_out.extend(hsdp_param.all_gather_outputs)
        ag_output = ag_output.view(world_size, -1)
        out = [t.view(world_size, -1) for t in split_with_sizes_out]
        non_inference_outs = [o for o in out if not o.is_inference()]
        if len(non_inference_outs) > 0:
            # pylint: disable=W0212
            with torch.autograd._unsafe_preserve_version_counter(tuple(non_inference_outs)):
                torch.split_with_sizes_copy(ag_output, metadata.inp_split_sizes, dim=1, out=out)
        else:
            torch.split_with_sizes_copy(ag_output, metadata.inp_split_sizes, dim=1, out=out)
        self._result = None
        self.free_all_gather_output()  # Immediately release fused buffer memory

    @torch.no_grad()
    def foreach_reduce(
        self,
        reduce_scatter_reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG,
        async_op: bool = True
    ) -> Optional[torch.Tensor]:
        """Perform fused gradient reduction (reduce-scatter + optional all-reduce).

        Collects unsharded gradients from all parameters, packs them into a single
        contiguous buffer, and issues one ``reduce_scatter_tensor``. For HSDP (2D mesh),
        a follow-up ``all_reduce`` across the replicate dimension is also performed.

        When ``async_op=True``, the communication handle is stored in the global
        ``CommContext`` so that the next module's backward hook can overlap computation
        with this reduction. The actual gradient write-back is deferred to
        ``apply_fusion_reduced_grad()``.

        Args:
            reduce_scatter_reduce_op: Reduction operator (default: AVG).
            async_op: If True, run collectives asynchronously for compute-comm overlap.
        """
        # Collect unsharded gradients (from accumulated grad or .grad)
        hsdp_params: List[TorchHSDPParamV2] = []
        unsharded_grads: List[torch.Tensor] = []
        for hsdp_param in self.hsdp_params:
            if not hasattr(hsdp_param, '_unsharded_param'):
                continue
            if hsdp_param.unsharded_accumulated_grad_data is not None:
                hsdp_params.append(hsdp_param)
                unsharded_grads.append(hsdp_param.unsharded_accumulated_grad_data)
            elif hsdp_param._unsharded_param.grad is not None:  # pylint: disable=W0212
                hsdp_params.append(hsdp_param)
                # pylint: disable=W0212
                unsharded_grads.append(hsdp_param._unsharded_param.grad)
        if not hsdp_params:
            return
        grad_dtypes = {g.dtype for g in unsharded_grads}
        if len(grad_dtypes) != 1:
            raise ValueError(
                f"FSDP reduce-scatter expects uniform grad dtype but got {grad_dtypes}"
            )
        grad_dtype = unsharded_grads[0].dtype
        reduce_dtype = self._reduce_dtype or grad_dtype
        world_size = self.shard_group.size()
        reduce_scatter_input_numel = sum(s.numel() for s in unsharded_grads)
        reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
        device = unsharded_grads[0].device
        # Pack all gradients into a contiguous buffer for fused reduce-scatter
        reduce_scatter_input = torch.empty((reduce_scatter_input_numel,), dtype=reduce_dtype, device=device)
        reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
        unsharded_grads.clear()  # Release references to full gradients
        reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
        self._needs_avg_div = reduce_scatter_reduce_op == dist.ReduceOp.AVG
        comm_op = dist.ReduceOp.SUM if self._needs_avg_div else reduce_scatter_reduce_op
        self._reduce_op = comm_op
        self._reduce_hsdp_params = hsdp_params
        rs_handle = dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=self.shard_group,
            op=comm_op,
            async_op=async_op
        )
        comm_ctx.comm_handle = rs_handle
        # Step 2 (HSDP only): All-reduce is deferred to apply_fusion_reduced_grad()
        self._reduce_output = reduce_output
        if async_op:
            # Register this group for deferred grad application by the next backward hook
            comm_ctx.pre_param_group = self
        else:
            self.apply_fusion_reduced_grad()

    def wait_reduce_scatter_and_issue_all_reduce(self):
        """Phase 1 of pipelined HSDP gradient reduction.

        Waits for the async reduce-scatter to complete, then issues an async
        all-reduce across the replicate dimension. The all-reduce handle is
        stored in ``comm_ctx.all_reduce_handle`` so that it can overlap with
        the next layer's reduce-scatter (Phase 2 is deferred).

        For FSDP (no replicate group), skips the all-reduce and directly
        applies gradients since there is nothing further to pipeline.
        """
        if comm_ctx.comm_handle is not None:
            comm_ctx.comm_handle.wait()
            comm_ctx.comm_handle = None
        # Deferred div for AVG: apply after RS completes, before AR
        if self._needs_avg_div:
            self._reduce_output.div_(self.shard_world_size)
        if self.replicate_group is not None and self.replicate_group.size() > 1:
            ar_handle = dist.all_reduce(
                self._reduce_output,
                group=self.replicate_group,
                op=self._reduce_op,
                async_op=True,
            )
            comm_ctx.all_reduce_handle = ar_handle
            comm_ctx.all_reduce_param_group = self
        else:
            # No replicate group — no all-reduce needed, apply grads immediately
            self._apply_reduced_grad()

    def wait_all_reduce_and_apply_grad(self):
        """Phase 2 of pipelined HSDP gradient reduction.

        Waits for the async all-reduce issued in Phase 1 and writes reduced
        gradients back to sharded parameters.
        """
        if comm_ctx.all_reduce_handle is not None:
            comm_ctx.all_reduce_handle.wait()
            comm_ctx.all_reduce_handle = None
        # Deferred div for AVG: apply after AR completes
        if self._needs_avg_div and self.replicate_group is not None:
            self._reduce_output.div_(self.replicate_group.size())
        self._apply_reduced_grad()

    def apply_fusion_reduced_grad(self):
        """Full synchronous reduction path (used for final drain and sync mode).

        Waits for reduce-scatter, performs synchronous all-reduce if needed,
        and applies gradients — all in one call without pipelining.
        """
        if comm_ctx.comm_handle is not None:
            comm_ctx.comm_handle.wait()
            comm_ctx.comm_handle = None
        # Deferred div for AVG after RS
        if self._needs_avg_div:
            self._reduce_output.div_(self.shard_world_size)
        if self.replicate_group is not None and self.replicate_group.size() > 1:
            dist.all_reduce(
                self._reduce_output,
                group=self.replicate_group,
                op=self._reduce_op,
            )
            # Deferred div for AVG after AR
            if self._needs_avg_div:
                self._reduce_output.div_(self.replicate_group.size())
        self._apply_reduced_grad()

    def _apply_reduced_grad(self):
        """Write reduced gradients from ``_reduce_output`` back to sharded parameters.

        Slices the fused ``_reduce_output`` buffer into per-parameter sharded gradients
        using ``torch.as_strided`` (zero-copy view), then either accumulates into the
        existing ``.grad`` / ``.main_grad`` or assigns a new DTensor gradient.

        Handles:
            - Mixed-precision: casts reduced gradient to ``_orig_dtype`` if needed.
            - CPU offload: transfers gradient to CPU (``non_blocking`` when possible).
            - Gradient accumulation: adds to existing grad when present.
            - Memory cleanup: nulls out unsharded grad references to free memory.
        """
        flat_grad_offset = 0
        world_size = self.shard_group.size()
        if self._reduce_hsdp_params is None:
            return
        for hsdp_param in self._reduce_hsdp_params:
            # Determine target gradient tensor (regular .grad or fp32 main_grad)
            sharded_grad = None
            if not self.mp_policy.apply_grad_on_fp32_main_grad:
                sharded_grad = hsdp_param.sharded_param.grad
            else:
                if not hasattr(hsdp_param.sharded_param, "main_grad"):
                    hsdp_param.sharded_param.main_grad = None
                sharded_grad = hsdp_param.sharded_param.main_grad
            # pylint: disable=W0212
            unsharded_size = hsdp_param._unsharded_param.size()
            shard_size = hsdp_param.sharded_size
            # Zero-copy view into the fused reduce output for this parameter's shard
            new_sharded_grad = torch.as_strided(
                self._reduce_output,
                size=shard_size,
                stride=hsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            # Cast to original dtype if reduce was done in a different precision
            if not self.mp_policy.apply_grad_on_fp32_main_grad and new_sharded_grad.dtype != self._orig_dtype:
                new_sharded_grad = new_sharded_grad.to(self._orig_dtype)
            need_synchronize = False
            if hsdp_param.offload_to_cpu:
                non_blocking = hsdp_param.pin_memory and sharded_grad is None
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                need_synchronize = True
            # Accumulate or assign gradient
            if sharded_grad is not None:
                if not self.mp_policy.apply_grad_on_fp32_main_grad:
                    hsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
                else:
                    hsdp_param.sharded_param.main_grad._local_tensor += new_sharded_grad
                    hsdp_param.sharded_param.grad = None
            else:
                if not self.mp_policy.apply_grad_on_fp32_main_grad:
                    hsdp_param.sharded_param.grad = hsdp_param.to_sharded_dtensor(new_sharded_grad)
                else:
                    hsdp_param.sharded_param.main_grad = hsdp_param.to_sharded_dtensor(new_sharded_grad)
                    hsdp_param.sharded_param.grad = None
            flat_grad_offset += unsharded_size.numel() // world_size
            # Release unsharded gradient references to free memory
            if hsdp_param.unsharded_accumulated_grad_data is not None:
                hsdp_param.unsharded_accumulated_grad = None
            elif hsdp_param.unsharded_param.grad is not None:
                hsdp_param.unsharded_param.grad = None

            if need_synchronize:
                if self.device.type == "npu":
                    torch.npu.current_stream().synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.current_stream().synchronize()
                else:
                    raise NotImplementedError(f"Unsupported device type {self.device} for \
                                              synchronization after CPU offload.")
        self._reduce_output = None  # Release fused reduce buffer
        self._reduce_hsdp_params = None

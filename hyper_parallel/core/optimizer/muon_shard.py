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

"""Shard function of muon."""

__all__ = [
    "build_param_shard_metadata_for_group",
    "build_pad_ns_inputs",
    "chunk_update_by_layout",
    "fused_allgather_dtensor_params",
]

import math
import logging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Native Torch is required by the current core rules; the legacy backend-dispatch lint is inapplicable.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.sharding_category import (
    HSDPCommGroup,
    ParamLayoutSpec,
    ParamShardMeta,
    ParamShardStageMeta,
)

logger = logging.getLogger(__name__)


def make_param_shard_meta(
        param: torch.Tensor,
        stage_metas: List[ParamShardStageMeta],
) -> ParamShardMeta:
    """Construct shard metadata from real local-shape plans.

    Args:
        param: Parameter whose local and global shapes are described.
        stage_metas: Per-axis padding and split metadata.
    """
    global_shape = tuple(param.shape)
    local_shape = tuple(to_local_if_dtensor(param.data).shape)
    return ParamShardMeta(
        global_shape=global_shape,
        global_numel=math.prod(global_shape),
        local_shape=local_shape,
        local_numel=math.prod(local_shape),
        stage_metas=tuple(stage_metas),
    )


def _get_mesh_shape(device_mesh) -> Tuple[int, ...]:
    """Get mesh shape."""
    mesh = getattr(device_mesh, "mesh", None)
    if mesh is not None:
        return tuple(mesh.shape)
    mesh_shape = getattr(device_mesh, "mesh_shape", None)
    if mesh_shape is not None:
        return tuple(mesh_shape)
    return ()


def _get_local_shape_kinds(global_shape: Tuple[int, ...], shard_meta: ParamShardMeta) -> List[Tuple[int, ...]]:
    """Collect all distinct local shapes implied by the shard metadata."""
    dim_size_options: Dict[int, List[int]] = {}
    for stage_meta in shard_meta.stage_metas:
        tensor_dim = stage_meta.tensor_dim
        seen_sizes = []
        for split_size in stage_meta.split_sizes:
            if split_size not in seen_sizes:
                seen_sizes.append(split_size)
        dim_size_options[tensor_dim] = seen_sizes

    shape_kinds = [list(global_shape)]
    for tensor_dim, size_options in sorted(dim_size_options.items()):
        next_shape_kinds = []
        for shape_kind in shape_kinds:
            for size in size_options:
                next_shape = list(shape_kind)
                next_shape[tensor_dim] = size
                next_shape_kinds.append(next_shape)
        shape_kinds = next_shape_kinds

    unique_shape_kinds = []
    seen = set()
    for shape_kind in shape_kinds:
        shape_kind = tuple(shape_kind)
        if shape_kind in seen:
            continue
        seen.add(shape_kind)
        unique_shape_kinds.append(shape_kind)
    return unique_shape_kinds


def _debug_param_shard_metadata(hsdp_group, param_to_meta: Dict[torch.nn.Parameter, ParamShardMeta]) -> None:
    """Log shard metadata for the group on rank 0 when distributed is ready."""
    if not dist.is_available() or not dist.is_initialized():
        return

    rank = dist.get_rank()
    if rank != 0:
        return

    for record in hsdp_group.records:
        param = record.param
        shard_meta = param_to_meta.get(param)
        if shard_meta is None:
            continue
        local_shape_kinds = _get_local_shape_kinds(shard_meta.global_shape, shard_meta)
        logger.debug_rank0(
            "name=%s, placements=%s, mesh_shape=%s, global_shape=%s, local_shapes=%s",
            getattr(param, "model_name", f"param_{record.index}"),
            [str(placement) for placement in param.placements],
            _get_mesh_shape(param.device_mesh),
            shard_meta.global_shape,
            local_shape_kinds,
        )


def _allgather_shapes_for_group(
        local_shapes: List[Tuple[int, ...]],
        shard_pg: Optional[dist.ProcessGroup],
        shard_size: int,
        device: torch.device,
) -> List[List[Tuple[int, ...]]]:
    """All-gather per-parameter shapes within one shard process group."""
    if shard_pg is None or shard_size <= 1:
        return [[tuple(shape) for shape in local_shapes]]

    shape_tensor = torch.tensor(local_shapes, dtype=torch.int64, device=device)
    gathered = torch.empty((shard_size, *shape_tensor.shape), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered.view(-1), shape_tensor.contiguous().view(-1), group=shard_pg)
    gathered_cpu = gathered.cpu().tolist()
    return [
        [tuple(shape) for shape in rank_shapes]
        for rank_shapes in gathered_cpu
    ]


def _build_shard_stage(gathered_shapes, record_idx, tensor_dim, shard_size, cur_rank):
    """Construct one padding stage and its reconstructed shape."""
    rank_shapes = tuple(tuple(gathered_shapes[rank_idx][record_idx]) for rank_idx in range(shard_size))
    tensor_dim_norm = tensor_dim % len(rank_shapes[0])
    split_sizes = tuple(shape[tensor_dim_norm] for shape in rank_shapes)
    pad_shape = tuple(max(shape[dim] for shape in rank_shapes) for dim in range(len(rank_shapes[0])))
    stage = ParamShardStageMeta(
        tensor_dim=tensor_dim_norm, shard_size=shard_size, cur_rank_in_shard_group=cur_rank,
        rank_shapes=rank_shapes, split_sizes=split_sizes, pad_shape=pad_shape,
    )
    output_shape = list(rank_shapes[cur_rank])
    if shard_size > 1:
        output_shape[tensor_dim_norm] = sum(split_sizes)
    return stage, tuple(output_shape)


def build_param_shard_metadata_for_group(
        hsdp_group: HSDPCommGroup,
) -> Dict[torch.Tensor, ParamShardMeta]:
    """Build per-parameter shard metadata before HSDP batching.

    Args:
        hsdp_group: Communication group and its parameter records.
    """
    # pylint: disable=too-many-locals
    layout_spec = hsdp_group.layout_spec
    shard_pgs = hsdp_group.shard_pgs
    if layout_spec is None or not layout_spec.shard_axes or not hsdp_group.records:
        return {}

    current_shapes = [tuple(to_local_if_dtensor(record.param.data).shape) for record in hsdp_group.records]
    stage_metas_per_param: List[List[ParamShardStageMeta]] = [[] for _ in hsdp_group.records]
    local_device = to_local_if_dtensor(hsdp_group.records[0].param.data).device

    for (_, tensor_dim), shard_pg in zip(layout_spec.shard_axes, shard_pgs):
        shard_size = dist.get_world_size(shard_pg) if shard_pg is not None else 1
        gathered_shapes = _allgather_shapes_for_group(current_shapes, shard_pg, shard_size, local_device)
        cur_rank = dist.get_rank(shard_pg) if shard_pg is not None and shard_size > 1 else 0
        next_shapes: List[Tuple[int, ...]] = []

        for record_idx, _ in enumerate(hsdp_group.records):
            stage, output_shape = _build_shard_stage(gathered_shapes, record_idx, tensor_dim, shard_size, cur_rank)
            stage_metas_per_param[record_idx].append(stage)
            next_shapes.append(output_shape)

        current_shapes = next_shapes

    param_to_meta: Dict[torch.Tensor, ParamShardMeta] = {}
    updated_records = []
    for record, stage_metas in zip(hsdp_group.records, stage_metas_per_param):
        shard_meta = make_param_shard_meta(record.param, stage_metas)
        param_to_meta[record.param] = shard_meta
        updated_records.append(record.__class__(index=record.index, param=record.param, shard_meta=shard_meta))

    hsdp_group.records = updated_records
    return param_to_meta


def build_pad_ns_inputs(
        comm_params: List[torch.Tensor],
        param_to_ns_input: Dict[torch.Tensor, torch.Tensor],
        param_shard_metadata: Optional[Dict[torch.Tensor, ParamShardMeta]] = None,
) -> List[torch.Tensor]:
    """Build aligned local NS inputs for all communication params.

    Args:
        comm_params: Parameters in communication order.
        param_to_ns_input: Prepared local momentum inputs.
        param_shard_metadata: Optional cached per-parameter shape metadata.
    """
    if not comm_params:
        return []

    ref_tensor = next(iter(param_to_ns_input.values()), None)
    default_dtype = ref_tensor.dtype if ref_tensor is not None else to_local_if_dtensor(comm_params[0].data).dtype
    local_inputs: List[torch.Tensor] = []

    for param in comm_params:
        ns_input = param_to_ns_input.get(param)
        if ns_input is not None:
            local_inputs.append(ns_input)
            continue

        shard_meta = None
        if param_shard_metadata is not None:
            shard_meta = param_shard_metadata.get(param)
        if shard_meta is None and hasattr(param, "shard_meta"):
            shard_meta = getattr(param, "shard_meta")
        if shard_meta is None:
            local_shape = tuple(to_local_if_dtensor(param.data).shape)
        else:
            local_shape = shard_meta.local_shape

        local_inputs.append(
            torch.zeros(
                local_shape,
                dtype=default_dtype,
                device=to_local_if_dtensor(param.data).device,
            )
        )

    return local_inputs


def chunk_update_by_layout(
        global_update: torch.Tensor,
        param: torch.Tensor,
        layout_spec: ParamLayoutSpec,
        param_shard_meta: Optional[ParamShardMeta] = None,
) -> torch.Tensor:
    """Slice a full update back to the local shard using narrow.

    Args:
        global_update: Full orthogonalized parameter update.
        param: Parameter whose local and global shapes are described.
        layout_spec: Shard axes used to restore the local update.
        param_shard_meta: Optional cached shape metadata for this parameter.
    """
    if not hasattr(param, "device_mesh") or layout_spec is None or not layout_spec.shard_axes:
        return global_update

    device_mesh = param.device_mesh
    mesh_coordinates = device_mesh.get_coordinate()

    shard_axes = layout_spec.shard_axes
    local_update = global_update

    # ``stage_metas`` are built while all-gathering local shards back to the
    # global tensor shape, so slicing a global update back to the local shard
    # must reverse that order.
    for axis_idx, (mesh_dim, tensor_dim) in reversed(list(enumerate(shard_axes))):
        num_chunks = device_mesh.size(mesh_dim)

        if num_chunks <= 1:
            continue

        local_rank = mesh_coordinates[mesh_dim]
        if param_shard_meta is not None and axis_idx < len(param_shard_meta.stage_metas):
            split_sizes = param_shard_meta.stage_metas[axis_idx].split_sizes
        else:
            full_chunk_size = (local_update.size(tensor_dim) + num_chunks - 1) // num_chunks
            split_sizes = tuple(
                min(full_chunk_size, max(local_update.size(tensor_dim) - rank * full_chunk_size, 0))
                for rank in range(num_chunks)
            )
        local_update = local_update.narrow(tensor_dim, sum(split_sizes[:local_rank]), split_sizes[local_rank])

    if not local_update.is_contiguous():
        local_update = local_update.contiguous()

    return local_update


def _get_or_alloc_buffer(
        cache: Optional[Dict],
        key: Any,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    """Return a cached buffer with at least `numel` elements."""
    if cache is not None and key in cache:
        buf = cache[key]
        if buf.numel() >= numel:
            return buf
        buf = torch.empty(numel, dtype=dtype, device=device)
        cache[key] = buf
        return buf

    buf = torch.empty(numel, dtype=dtype, device=device)
    if cache is not None:
        cache[key] = buf
    return buf


def _early_return_tensors(
        local_tensors: List[torch.Tensor],
        keep_indices: Optional[set],
) -> List[Optional[torch.Tensor]]:
    """Return tensors directly when no communication is needed."""
    if keep_indices is None:
        return list(local_tensors)
    return [t if i in keep_indices else None for i, t in enumerate(local_tensors)]


@dataclass(frozen=True)
class _GatherParamMeta:
    offset: int
    actual_numel: int
    padded_numel: int
    rest_shape: Tuple[int, ...]
    split_sizes: Tuple[int, ...]


def _prepare_gather_inputs(
        current_tensors: List[torch.Tensor],
        tensor_dim: int,
        alignment_elements: int,
        shard_size: int,
        stage_metas: Optional[List[Optional[ParamShardStageMeta]]] = None,
) -> Tuple[List[torch.Tensor], List[_GatherParamMeta], int]:
    """Move shard dim to dim0 and compute padding metadata."""
    gather_inputs: List[torch.Tensor] = []
    param_meta: List[_GatherParamMeta] = []
    total_padded_numel = 0

    for idx, gather_input in enumerate(current_tensors):
        normalized_dim = tensor_dim % gather_input.dim()
        if normalized_dim != 0 or not gather_input.is_contiguous():
            gather_input = gather_input.movedim(normalized_dim, 0).contiguous()
        stage_meta = stage_metas[idx] if stage_metas is not None else None
        if stage_meta is None:
            split_sizes = tuple(gather_input.shape[0] for _ in range(shard_size))
            padded_numel_raw = gather_input.numel()
        else:
            split_sizes = stage_meta.split_sizes
            pad_shape = list(stage_meta.pad_shape)
            if normalized_dim != 0:
                pad_shape[0], pad_shape[normalized_dim] = pad_shape[normalized_dim], pad_shape[0]
            padded_numel_raw = math.prod(pad_shape)
        param_meta.append(
            _GatherParamMeta(
                offset=total_padded_numel,
                actual_numel=gather_input.numel(),
                padded_numel=(
                    (padded_numel_raw + alignment_elements - 1) // alignment_elements * alignment_elements
                ),
                rest_shape=tuple(gather_input.shape[1:]),
                split_sizes=split_sizes,
            )
        )
        gather_inputs.append(gather_input)
        total_padded_numel += param_meta[-1].padded_numel

    return gather_inputs, param_meta, total_padded_numel


def _pack_gather_buffers(
        gather_inputs: List[torch.Tensor],
        param_meta: List[_GatherParamMeta],
        total_padded_numel: int,
        axis_idx: int,
        dtype: torch.dtype,
        device: torch.device,
        shard_size: int,
        buffer_cache: Optional[Dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack local shards and allocate the matching receive buffer."""
    pack_buffer = _get_or_alloc_buffer(
        buffer_cache, ("fused_allgather", axis_idx, dtype, device), total_padded_numel,
        dtype, device,
    )[:total_padded_numel]

    pack_buffer.zero_()
    for gi, meta in zip(gather_inputs, param_meta):
        pack_buffer[meta.offset:meta.offset + meta.actual_numel].copy_(gi.view(-1))

    gathered_buffer = _get_or_alloc_buffer(
        buffer_cache,
        ("fused_allgather_out", axis_idx, dtype, device),
        total_padded_numel * shard_size,
        dtype, device,
    )[:total_padded_numel * shard_size]

    return pack_buffer, gathered_buffer


def _pack_and_allgather(
        gather_inputs: List[torch.Tensor],
        param_meta: List[_GatherParamMeta],
        total_padded_numel: int,
        axis_idx: int,
        dtype: torch.dtype,
        device: torch.device,
        shard_pg: dist.ProcessGroup,
        shard_size: int,
        buffer_cache: Optional[Dict],
) -> torch.Tensor:
    """Pack local shards into one buffer, all-gather, return gathered view."""
    pack_buffer, gathered_buffer = _pack_gather_buffers(
        gather_inputs, param_meta, total_padded_numel, axis_idx,
        dtype, device, shard_size, buffer_cache,
    )
    dist.all_gather_into_tensor(gathered_buffer, pack_buffer, group=shard_pg)
    return gathered_buffer.view(shard_size, total_padded_numel)


def _unpack_gathered_results(
        gathered_view: torch.Tensor,
        gather_inputs: List[torch.Tensor],
        param_meta: List[_GatherParamMeta],
        current_tensors: List[torch.Tensor],
        tensor_dim: int,
        shard_size: int,
        n_params: int,
        is_last_axis: bool,
        keep_indices: Optional[set],
) -> List[Optional[torch.Tensor]]:
    """Slice gathered buffer back to per-parameter full tensors."""
    new_tensors: List[Optional[torch.Tensor]] = []
    for i in range(n_params):
        if is_last_axis and keep_indices is not None and i not in keep_indices:
            new_tensors.append(None)
            continue

        meta = param_meta[i]
        rest_shape = meta.rest_shape
        rest_numel = math.prod(rest_shape) if rest_shape else 1
        param_slice = gathered_view[:, meta.offset:meta.offset + meta.padded_numel]

        param_chunks = []
        for rank in range(shard_size):
            valid_numel = meta.split_sizes[rank] * rest_numel
            if valid_numel == 0:
                continue
            param_chunks.append(param_slice[rank, :valid_numel].contiguous().view(-1))

        if param_chunks:
            param_data = torch.cat(param_chunks, dim=0)
        else:
            param_data = gather_inputs[i].new_empty((0,))

        result = param_data.view(sum(meta.split_sizes), *rest_shape)
        tensor_dim_norm = tensor_dim % current_tensors[i].dim()
        if tensor_dim_norm == 0:
            new_tensors.append(result)
        else:
            new_tensors.append(result.movedim(0, tensor_dim_norm))

    return new_tensors


def fused_allgather_dtensor_params(
        local_tensors: List[torch.Tensor],
        shard_pgs: Sequence[dist.ProcessGroup],
        layout_spec: ParamLayoutSpec,
        param_shard_metadata: Optional[List[Optional[ParamShardMeta]]] = None,
        buffer_cache: Optional[Dict] = None,
        keep_indices: Optional[set] = None,
) -> List[Optional[torch.Tensor]]:
    """Fuse many parameter shards into one all-gather per shard axis.

    Args:
        local_tensors: Local NS inputs to gather.
        shard_pgs: Process groups for each shard axis.
        layout_spec: Shard axes used to restore the local update.
        param_shard_metadata: Optional cached per-parameter shape metadata.
        buffer_cache: Optional reusable communication buffers.
        keep_indices: Indices of updates reconstructed on this compute owner.
    """
    if not shard_pgs or not local_tensors:
        return _early_return_tensors(local_tensors, keep_indices)

    n_params = len(local_tensors)
    device = local_tensors[0].device
    dtype = local_tensors[0].dtype
    alignment_bytes = 512
    element_size = local_tensors[0].element_size()
    alignment_elements = max(1, alignment_bytes // element_size)

    active_axes = []
    for axis_idx, ((_, tensor_dim), shard_pg) in enumerate(zip(layout_spec.shard_axes, shard_pgs)):
        if shard_pg is None:
            continue
        shard_size = dist.get_world_size(shard_pg)
        if shard_size <= 1:
            continue
        active_axes.append((axis_idx, tensor_dim, shard_pg, shard_size))

    if not active_axes:
        return _early_return_tensors(local_tensors, keep_indices)

    current_tensors: List[torch.Tensor] = list(local_tensors)

    for active_pos, (axis_idx, tensor_dim, shard_pg, shard_size) in enumerate(active_axes):
        is_last_axis = active_pos == len(active_axes) - 1
        stage_metas: Optional[List[Optional[ParamShardStageMeta]]] = None
        if param_shard_metadata is not None:
            stage_metas = []
            for meta in param_shard_metadata:
                if meta is None or axis_idx >= len(meta.stage_metas):
                    stage_metas.append(None)
                else:
                    stage_metas.append(meta.stage_metas[axis_idx])

        gather_inputs, param_meta, total_padded_numel = _prepare_gather_inputs(
            current_tensors, tensor_dim, alignment_elements, shard_size, stage_metas,
        )

        gathered_view = _pack_and_allgather(
            gather_inputs, param_meta, total_padded_numel,
            axis_idx, dtype, device, shard_pg, shard_size, buffer_cache,
        )

        new_tensors = _unpack_gathered_results(
            gathered_view, gather_inputs, param_meta,
            current_tensors, tensor_dim, shard_size,
            n_params, is_last_axis, keep_indices,
        )

        if is_last_axis:
            return new_tensors

        current_tensors = new_tensors  # type: ignore[assignment]

    return current_tensors


@dataclass
class PendingMuonGather:
    """Own asynchronous gather storage until the consumer has finished using it."""

    work: Any
    send_buffer: torch.Tensor
    gathered_view: torch.Tensor
    inputs: List[torch.Tensor]
    metadata: List[_GatherParamMeta]
    tensor_dim: int
    shard_size: int
    keep_indices: set

    def finish(self) -> List[Optional[torch.Tensor]]:
        """Wait on the caller's stream and reconstruct only its owner inputs."""
        self.work.wait()
        return _unpack_gathered_results(
            self.gathered_view, self.inputs, self.metadata, self.inputs,
            self.tensor_dim, self.shard_size, len(self.inputs), True, self.keep_indices,
        )


@dataclass
class PreparedMuonGather:
    """Packed input independent of the reusable receive slot."""

    send_buffer: torch.Tensor
    inputs: List[torch.Tensor]
    metadata: List[_GatherParamMeta]
    tensor_dim: int
    shard_size: int
    keep_indices: set

    def launch(self, shard_pg: dist.ProcessGroup, buffer_cache: Dict) -> PendingMuonGather:
        """Submit the collective after input packing has been enqueued.

        Args:
            shard_pg: Process group for this shard axis.
            buffer_cache: Reusable receive buffers for this pipeline slot.
        """
        source = self.send_buffer
        output = _get_or_alloc_buffer(
            buffer_cache, ("fused_allgather_out", 0, source.dtype, source.device),
            source.numel() * self.shard_size, source.dtype, source.device,
        )[:source.numel() * self.shard_size]
        work = dist.all_gather_into_tensor(output, source, group=shard_pg, async_op=True)
        return PendingMuonGather(
            work, source, output.view(self.shard_size, source.numel()),
            self.inputs, self.metadata, self.tensor_dim, self.shard_size, self.keep_indices,
        )


def prepare_muon_gather(
        local_tensors: List[torch.Tensor],
        shard_pg: dist.ProcessGroup,
        tensor_dim: int,
        stage_metas: List[ParamShardStageMeta],
        keep_indices: set,
        buffer_cache: Optional[Dict] = None,
        plan_cache: Optional[Dict] = None,
) -> PreparedMuonGather:
    """Pack one bucket, reusing static layout metadata and bounded send storage.

    Args:
        local_tensors: Local NS inputs in parameter order.
        shard_pg: Process group for the shard axis.
        tensor_dim: Tensor dimension partitioned across the group.
        stage_metas: Padding and split metadata for each parameter.
        keep_indices: Parameters reconstructed by this rank after gathering.
        buffer_cache: Optional reusable send storage and views.
        plan_cache: Optional static layout plan cache.
    """
    shard_size = dist.get_world_size(shard_pg)
    first = local_tensors[0]
    signature = (tensor_dim, shard_size, first.dtype,
                 tuple((tuple(tensor.shape), tensor.stride()) for tensor in local_tensors),
                 tuple((tuple(meta.pad_shape), tuple(meta.split_sizes)) for meta in stage_metas))
    if plan_cache is not None and plan_cache.get("signature") == signature:
        metadata, padded_numel = plan_cache["metadata"], plan_cache["padded_numel"]
        gather_inputs = [
            tensor if tensor_dim % tensor.dim() == 0 and tensor.is_contiguous()
            else tensor.movedim(tensor_dim % tensor.dim(), 0).contiguous()
            for tensor in local_tensors
        ]
    else:
        gather_inputs, metadata, padded_numel = _prepare_gather_inputs(
            local_tensors, tensor_dim, max(1, 512 // first.element_size()), shard_size, stage_metas,
        )
        if plan_cache is not None:
            plan_cache.update(signature=signature, metadata=metadata, padded_numel=padded_numel)
    source = _get_or_alloc_buffer(
        buffer_cache, ("pipeline_send", first.dtype, first.device), padded_numel, first.dtype, first.device,
    )[:padded_numel]
    source.zero_()
    view_key = (source.data_ptr(), signature)
    cached_views = buffer_cache.get("pipeline_views") if buffer_cache is not None else None
    if cached_views is None or cached_views[0] != view_key:
        destinations = [source[meta.offset:meta.offset + meta.actual_numel].view_as(tensor)
                        for tensor, meta in zip(gather_inputs, metadata)]
        if buffer_cache is not None:
            buffer_cache["pipeline_views"] = (view_key, destinations)
    else:
        destinations = cached_views[1]
    # Torch exposes batched tensor copies only through its private foreach API.
    torch._foreach_copy_(destinations, gather_inputs)  # pylint: disable=protected-access
    return PreparedMuonGather(source, local_tensors, metadata, tensor_dim, shard_size, keep_indices)


def launch_muon_gather(
        local_tensors: List[torch.Tensor],
        shard_pg: dist.ProcessGroup,
        tensor_dim: int,
        stage_metas: List[ParamShardStageMeta],
        keep_indices: set,
        buffer_cache: Dict,
) -> PendingMuonGather:
    """Prepare and submit a single-axis asynchronous gather.

    Args:
        local_tensors: Local NS inputs in parameter order.
        shard_pg: Process group for the shard axis.
        tensor_dim: Tensor dimension partitioned across the group.
        stage_metas: Padding and split metadata for each parameter.
        keep_indices: Parameters reconstructed by this rank after gathering.
        buffer_cache: Reusable receive buffers for this pipeline slot.
    """
    return prepare_muon_gather(local_tensors, shard_pg, tensor_dim, stage_metas, keep_indices).launch(
        shard_pg, buffer_cache,
    )

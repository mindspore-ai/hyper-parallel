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
"""Optimizer-state helpers for graph-mode FSDP.

A checkpoint restored before the lazy graph compile materializes optimizer
moments at the unsharded parameter shapes. Once FSDPPass shards the live
parameters, those tensors must follow the same padded chunk geometry or the
first optimizer step sees a shape mismatch.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from hyper_parallel.core.distributed_checkpoint.metadata import CHUNK_INFO
from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.dtensor.placement_types import StridedShard

from .checkpoint_metadata import canonical_parameter_names, has_strided_placement

_LOG = logging.getLogger(__name__)


def iter_optimizer_leaves(optimizer: Any) -> Iterator[Any]:
    """Yield the leaf optimizers that own ``state`` tensors.

    Args:
        optimizer: Optimizer, chained optimizer, or wrapper to descend into.
    """
    children = getattr(optimizer, "chained_optimizers", None)
    if children:
        for child in children:
            yield from iter_optimizer_leaves(child)
        return
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer and hasattr(inner, "state"):
        yield from iter_optimizer_leaves(inner)
        return
    yield optimizer


def reshard_state_tensor(
    tensor: Any,
    full_shape: Sequence[int],
    shard_index: int,
    degree: int,
    n: int,
    n_pad: int,
) -> Optional[torch.Tensor]:
    """Return one rank's padded shard of an unsharded optimizer-state tensor.

    Args:
        tensor: Dense optimizer-state tensor to shard.
        full_shape: Shape of the unsharded parameter.
        shard_index: This rank's position along the shard axis.
        degree: Number of ranks the parameter is sharded across.
        n: True size of dim 0 before padding.
        n_pad: Padded size of dim 0 used by FSDP storage.

    Returns:
        The local padded shard, or None when the tensor does not map.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    if tuple(tensor.shape) != tuple(full_shape) or tensor.ndim == 0:
        return None
    if degree <= 1 or n_pad < n or n_pad % degree or not 0 <= shard_index < degree:
        return None
    shard_size = n_pad // degree
    if shard_size <= 0:
        return None
    start = shard_index * shard_size
    tail_shape = (shard_size, *tensor.shape[1:])
    if start >= n:
        # This rank owns only padding rows; keep the padded local geometry.
        return tensor.new_zeros(tail_shape)
    end = start + shard_size
    if end <= n:
        return tensor[start:end].clone()
    # The shard straddles the true extent: copy the real rows and zero-fill the
    # padding tail without materializing a full padded copy of the tensor.
    shard = tensor.new_zeros(tail_shape)
    real = n - start
    shard[:real] = tensor[start:n]
    return shard


def reshard_optimizer_state(compiler: Any, optimizers: Any) -> None:
    """Re-shard pre-compile optimizer state onto the live FSDP shards.

    Checkpoint restore runs before the lazy graph compile, so a resumed run
    materializes dense moments at the unsharded parameter shapes. Once
    ``FSDPPass`` shards the parameters, the moments must follow the same padded
    chunk geometry; leaving them unsharded makes the first step see a
    full-shaped state against a sharded gradient.

    Only state tensors shaped exactly like the parameter (AdamW ``exp_avg``,
    Muon ``momentum_buffer``, ...) are re-sharded. Scalar bookkeeping is left
    alone; any other tensor shape (factored moments and the like) is left
    untouched and reported in a single warning, because such state has no
    general mapping onto one rectangular shard.

    Args:
        compiler: Compiled graph compiler owning the sharded live model.
        optimizers: One optimizer or a list of optimizers whose leaf ``state``
            tensors should be re-sharded in place.
    """
    if compiler.joint_graph is None:
        raise RuntimeError(
            "reshard_optimizer_state() requires a completed compile: the "
            "live parameters are not sharded yet, so there is no FSDP "
            "shard geometry to map the dense optimizer state onto"
        )
    compiler.attach_shard_metadata(strict_strided=False)
    layouts = compiler.parameter_layouts or {}
    if not layouts:
        return
    names = canonical_parameter_names(compiler.model)
    padded = getattr(compiler.joint_graph.graph_module, "fsdp_padded_params", {})
    plans, strided, off_axis, unmappable = _optimizer_reshard_plans(
        compiler, layouts, names, padded
    )
    if not plans and not strided and not off_axis and not unmappable:
        return
    entries = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
    unresharded = _reshard_restored_state(
        entries, plans, strided, off_axis, unmappable, names
    )
    _warn_unresharded(unresharded)


def _optimizer_reshard_plans(
    compiler: Any,
    layouts: Dict[Any, Any],
    names: Dict[Any, str],
    padded: Dict[str, Tuple[int, int]],
) -> Tuple[
    Dict[Any, Tuple[Tuple[int, ...], int, int, int, int]],
    Dict[Any, Any],
    Dict[Any, Any],
    Dict[Any, Any],
]:
    """Classify live parameters into reshard plans and rejection cases."""
    plans: Dict[Any, Tuple[Tuple[int, ...], int, int, int, int]] = {}
    strided: Dict[Any, Tuple[str, Tuple[int, ...]]] = {}
    off_axis: Dict[Any, Tuple[str, Tuple[int, ...]]] = {}
    unmappable: Dict[Any, Tuple[str, Tuple[int, ...]]] = {}
    for parameter, layout in layouts.items():
        name = names.get(parameter)
        full_shape = compiler.parameter_shapes.get(name)
        if name is None or full_shape is None or not parameter.shape:
            continue
        if tuple(parameter.shape) == tuple(full_shape):
            continue
        entry = (name, tuple(full_shape))
        if has_strided_placement(layout):
            strided[parameter] = entry
            continue
        if not _is_plain_dim0_shard(layout):
            off_axis[parameter] = entry
            continue
        plan = _optimizer_reshard_plan(parameter, name, full_shape, padded, layout)
        if plan is None:
            unmappable[parameter] = entry
        else:
            plans[parameter] = plan
    return plans, strided, off_axis, unmappable


def _optimizer_reshard_plan(
    parameter: torch.nn.Parameter,
    name: str,
    full_shape: Tuple[int, ...],
    padded: Dict[str, Tuple[int, int]],
    layout: Any,
) -> Optional[Tuple[Tuple[int, ...], int, int, int, int]]:
    """Return the chunk plan for one live shard, or None when unmappable."""
    chunk_info = getattr(parameter, CHUNK_INFO, None)
    local_size = int(parameter.shape[0])
    n, n_pad = padded.get(name, (int(full_shape[0]), int(full_shape[0])))
    degree = n_pad // local_size if local_size else 0
    if chunk_info is None or degree <= 1 or degree * local_size != n_pad:
        return None
    # Derive the physical shard index from the unpadded FSDP layout, not the DCP
    # chunk offset: ``attach_shard_metadata`` clips the chunk to the true global
    # extent, so a rank that owns only padding rows reports an empty logical
    # chunk at ``offset == true_dim0`` (e.g. 5 for a [8, ...] pad of a [5, ...]
    # param), whose offset is not a multiple of the local shard size. That rank
    # is still a valid physical shard and must map to an all-padding slice.
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rank_list = tuple(layout.rank_list)
    if rank not in rank_list:
        return None
    area = infer_slice_area_by_layout(
        layout, rank_list.index(rank), tuple(layout.tensor_shape)
    )
    offset = int(area[0][0])
    if offset % local_size:
        return None
    return (tuple(full_shape), offset // local_size, degree, n, n_pad)


def _reshard_restored_state(
    optimizers: List[Any],
    plans: Dict[Any, Tuple[Tuple[int, ...], int, int, int, int]],
    strided: Dict[Any, Any],
    off_axis: Dict[Any, Any],
    unmappable: Dict[Any, Any],
    names: Dict[Any, str],
) -> List[str]:
    """Re-shard each leaf optimizer's dense slots and collect leftovers."""
    unresharded: List[str] = []
    for optimizer in optimizers:
        for leaf in iter_optimizer_leaves(optimizer):
            state = getattr(leaf, "state", None)
            if not isinstance(state, dict):
                continue
            for parameter, slots in state.items():
                unresharded.extend(
                    _reshard_parameter_slots(
                        parameter, slots, plans, strided, off_axis, unmappable, names
                    )
                )
    return unresharded


def _reshard_parameter_slots(
    parameter: torch.nn.Parameter,
    slots: Any,
    plans: Dict[Any, Tuple[Tuple[int, ...], int, int, int, int]],
    strided: Dict[Any, Any],
    off_axis: Dict[Any, Any],
    unmappable: Dict[Any, Any],
    names: Dict[Any, str],
) -> List[str]:
    """Re-shard one parameter's slots; return leftover descriptions."""
    if parameter in strided:
        _reject_strided_resume(strided[parameter], slots)
        return []
    if parameter in off_axis:
        _reject_offaxis_resume(off_axis[parameter], slots)
        return []
    if parameter in unmappable:
        _reject_unmappable_resume(unmappable[parameter], slots)
        return []
    plan = plans.get(parameter)
    if plan is None or not isinstance(slots, dict):
        return []
    full_shape, local_shape = plan[0], tuple(parameter.shape)
    leftovers = []
    for key, value in _reshard_slots(slots, plan):
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and tuple(value.shape) not in (full_shape, local_shape)
        ):
            leftovers.append(f"{names.get(parameter)}.{key}: {tuple(value.shape)}")
    return leftovers


def _warn_unresharded(unresharded: List[str]) -> None:
    """Warn once about state tensors matching neither dense nor shard shape."""
    if not unresharded:
        return
    preview = ", ".join(unresharded[:8])
    if len(unresharded) > 8:
        preview += ", ..."
    _LOG.warning(
        "Left %d optimizer-state tensor(s) un-resharded: their shape "
        "matches neither the dense parameter nor its local shard. Exotic "
        "state (for example factored moments) is not supported in graph "
        "FSDP resume: %s",
        len(unresharded),
        preview,
    )


def _reshard_slots(
    slots: Dict[Any, Any],
    plan: Tuple[Tuple[int, ...], int, int, int, int],
) -> List[Tuple[Any, Any]]:
    """Re-shard one parameter's slots; return the slots left untouched."""
    left = []
    for key, value in list(slots.items()):
        reshaped = reshard_state_tensor(value, *plan)
        if reshaped is not None:
            slots[key] = reshaped
        else:
            left.append((key, value))
    return left


def _reject_strided_resume(
    entry: Tuple[str, Tuple[int, ...]],
    slots: Any,
) -> None:
    """Fail loudly when dense state cannot map onto a strided dim-0 shard."""
    name, full_shape = entry
    if not isinstance(slots, dict):
        return
    if not _has_dense_slot(slots, full_shape):
        return
    raise RuntimeError(
        f"Cannot reshard restored optimizer state for {name!r}: its dim-0 "
        "FSDP shard is strided (for example TP and FSDP shard the same axis), "
        "so the dense state has no single contiguous slice to map onto. "
        "Resume with a layout that does not combine strided dim-0 shards."
    )


def _reject_offaxis_resume(
    entry: Tuple[str, Tuple[int, ...]],
    slots: Any,
) -> None:
    """Fail loudly when dense state maps to a non-contiguous dim-0 shard."""
    name, full_shape = entry
    if not isinstance(slots, dict):
        return
    if not _has_dense_slot(slots, full_shape):
        return
    raise RuntimeError(
        f"Cannot reshard restored optimizer state for {name!r}: its FSDP "
        "shard is not a single contiguous dim-0 block (it shards another "
        "axis or spans several mesh dims), so the dense state has no single "
        "contiguous slice to map onto. Resume with a layout whose FSDP shard "
        "uses one non-strided dim-0 placement."
    )


def _reject_unmappable_resume(
    entry: Tuple[str, Tuple[int, ...]],
    slots: Any,
) -> None:
    """Fail loudly when a sharded param has dense state but no shard geometry."""
    name, full_shape = entry
    if not isinstance(slots, dict):
        return
    if not _has_dense_slot(slots, full_shape):
        return
    raise RuntimeError(
        f"Cannot reshard restored optimizer state for {name!r} (dense shape "
        f"{tuple(full_shape)}): the parameter is FSDP-sharded but its shard "
        "geometry is unavailable or inconsistent with the padded layout. "
        "This can happen with an uneven (unpadded) shard or when the sharding "
        "metadata name does not match the parameter. Resume from a checkpoint "
        "written by a compatible run."
    )


def _has_dense_slot(slots: Dict[Any, Any], full_shape: Tuple[int, ...]) -> bool:
    """Whether any slot tensor still has the unsharded parameter shape."""
    return any(
        isinstance(value, torch.Tensor) and tuple(value.shape) == full_shape
        for value in slots.values()
    )


def _is_plain_dim0_shard(layout: Any) -> bool:
    """Whether dim 0 is sharded by exactly one non-strided placement."""
    placements = tuple(layout.placements or ())
    shard_axes = [
        index for index, placement in enumerate(placements) if placement.is_shard(0)
    ]
    return len(shard_axes) == 1 and not isinstance(
        placements[shard_axes[0]], StridedShard
    )

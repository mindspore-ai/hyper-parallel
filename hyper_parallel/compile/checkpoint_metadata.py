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
"""Build and copy graph-mode shard metadata for checkpoint consumers.

Graph sharding replaces ``param.data`` without creating a ``DTensor``, so the
checkpoint and HF-export consumers need the final layout copied onto the
tensors they read. This module owns both halves of that bridge:

- ``attach_*_shard_metadata`` describe the live parameters (final FSDP storage,
  or source TP/EP storage before the lazy compile) as one rectangular chunk.
- ``annotate_*_state_dict`` copy that metadata onto the tensors a checkpoint or
  HF export actually reads.

The helpers stay inactive unless a graph compiler is attached to the trainer.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch._prims_common import make_contiguous_strides_for

from hyper_parallel.core.distributed_checkpoint.metadata import (
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
)
from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.dtensor.placement_types import StridedShard


def _compiler_layouts(
    trainer: Any, *, strict_strided: bool = True
) -> Optional[Dict[Any, Any]]:
    """Attach the applicable metadata and return the live parameter layouts.

    A compiled compiler owns the final FSDP layouts. Before compilation the
    model is still source-local (for example TP/EP), so annotate it with the
    source layouts instead; checkpoint consumers then see the true global shape
    even when the checkpoint is written or read before the first forward.

    Args:
        trainer: Trainer owning the graph compiler (if any).
        strict_strided: Raise on a strided shard when describing final layouts.
            Model export keeps it on because such a weight cannot be written
            correctly; the optimizer path relaxes it so an unrelated strided
            parameter does not block saving state that maps cleanly.

    Returns:
        The applicable parameter -> layout mapping, or None when the trainer has
        no graph compiler or the compiler reports no layouts.
    """
    compiler = getattr(trainer, "graph_compiler", None)
    if compiler is None:
        return None
    if compiler.is_compiled:
        attach = getattr(compiler, "attach_shard_metadata", None)
        layouts = getattr(compiler, "parameter_layouts", None)
    else:
        attach = getattr(compiler, "attach_source_shard_metadata", None)
        layouts = getattr(compiler, "source_parameter_layouts", None)
    if callable(attach):
        if compiler.is_compiled:
            attach(strict_strided=strict_strided)
        else:
            attach()
    return layouts or None


def _layouts_by_name(
    layouts: Dict[Any, Any], model: torch.nn.Module
) -> Dict[str, Tuple[Any, Any]]:
    """Return graph layouts keyed by model parameter FQN.

    Aliases are kept (``remove_duplicate=False``) so a tied parameter's other
    state-dict keys map to the same layout as its canonical name; deduplicating
    here would leave those keys unannotated and exported as local shards.
    """
    return {
        name: (parameter, layouts[parameter])
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter in layouts
    }


def _copy_shard_metadata(source: Any, target: Any) -> None:
    """Copy the live parameter's final layout onto a checkpoint tensor."""
    spec = getattr(source, "_sharding_spec", None)
    if spec is None or not isinstance(target, torch.Tensor):
        return
    setattr(target, "_sharding_spec", spec)
    setattr(target, "_layout", getattr(source, "_layout", spec))
    placements = getattr(source, "_placements", None)
    if placements is None:
        placements = tuple(spec.placements or ())
    setattr(target, "_placements", placements)
    chunk_info = getattr(source, CHUNK_INFO, None)
    if chunk_info is not None:
        setattr(target, CHUNK_INFO, chunk_info)


def _is_sharded_layout(layout: Any, parameter: Any) -> bool:
    """Whether ``layout`` describes a tensor larger than the local parameter."""
    tensor_shape = tuple(getattr(layout, "tensor_shape", None) or ())
    return bool(tensor_shape) and tensor_shape != tuple(parameter.shape)


def _reject_unmappable_optimizer_slot(
    name: str, parameter: Any, layout: Any, slots: Any
) -> None:
    """Fail loudly when sharded state has no attachable rectangular metadata.

    A strided/off-axis shard is skipped by ``attach_shard_metadata`` (no
    ``_sharding_spec``) because it has no single rectangular chunk. If such a
    parameter still owns dense state shaped like its local shard, silently
    exporting it would write a corrupt checkpoint, so reject it explicitly.
    """
    if getattr(parameter, "_sharding_spec", None) is not None:
        return
    if not _is_sharded_layout(layout, parameter):
        return
    local_shape = tuple(parameter.shape)
    if not any(
        isinstance(value, torch.Tensor) and tuple(value.shape) == local_shape
        for value in slots
    ):
        return
    raise RuntimeError(
        f"Cannot export optimizer state for {name!r}: its FSDP shard has no "
        "single rectangular chunk (for example a strided or off-axis shard), "
        "so the state cannot be written correctly. Save with a layout that "
        "shards one contiguous dim-0 block, or exclude this optimizer state."
    )


def annotate_model_state_dict(trainer: Any, state_dict: Dict[str, Any]) -> None:
    """Attach graph shard metadata so HF export and DCP see global tensors.

    Args:
        trainer: Trainer owning the graph compiler (if any).
        state_dict: Model state dictionary to annotate in place.
    """
    layouts = _compiler_layouts(trainer, strict_strided=True)
    if not layouts:
        return
    by_name = _layouts_by_name(layouts, trainer.model)
    for name, value in state_dict.items():
        entry = by_name.get(name)
        if entry is not None:
            _copy_shard_metadata(entry[0], value)


def annotate_optimizer_state_dict(trainer: Any, payload: Any) -> Any:
    """Attach graph shard metadata to dense optimizer-state tensors.

    Unrelated strided parameters are skipped here instead of aborting the save:
    only a strided parameter that actually owns dense state raises, because that
    state cannot be mapped onto one rectangular chunk.

    Args:
        trainer: Trainer owning the graph compiler (if any).
        payload: Nested optimizer state dictionary to annotate in place.

    Returns:
        The same ``payload`` object, annotated where a layout applies.
    """
    layouts = _compiler_layouts(trainer, strict_strided=False)
    if not layouts:
        return payload
    by_name = _layouts_by_name(layouts, trainer.model)

    def annotate_slot(slot: torch.Tensor, name: str, entry: Tuple[Any, Any]) -> None:
        """Annotate one state tensor that is shaped like its local parameter.

        Args:
            slot: Optimizer-state tensor for one slot.
            name: Canonical parameter FQN the slot belongs to.
            entry: ``(parameter, layout)`` pair from the live model.
        """
        parameter, layout = entry
        if tuple(slot.shape) != tuple(parameter.shape):
            return
        _reject_unmappable_optimizer_slot(name, parameter, layout, (slot,))
        _copy_shard_metadata(parameter, slot)

    def apply_state(state: Dict[Any, Any], name: str, entry: Tuple[Any, Any]) -> None:
        """Copy metadata onto every slot shaped like the local parameter.

        Args:
            state: One parameter's optimizer-state slot mapping.
            name: Canonical parameter FQN the slots belong to.
            entry: ``(parameter, layout)`` pair from the live model.
        """
        parameter, layout = entry
        _reject_unmappable_optimizer_slot(name, parameter, layout, state.values())
        local_shape = tuple(parameter.shape)
        for value in state.values():
            if isinstance(value, torch.Tensor) and tuple(value.shape) == local_shape:
                _copy_shard_metadata(parameter, value)

    def visit(value: Any) -> None:
        """Walk nested optimizer state and annotate matching entries.

        Args:
            value: Current node in the nested optimizer-state payload.
        """
        if isinstance(value, dict):
            for key, item in value.items():
                entry = by_name.get(key) if isinstance(key, str) else None
                if entry is not None and isinstance(item, dict):
                    apply_state(item, key, entry)
                    continue
                if isinstance(key, str) and key.startswith("state."):
                    # Flattened format is ``state.<FQN>.<slot>``; split on the
                    # last dot instead of scanning every parameter name.
                    fqn, sep, _ = key[len("state.") :].rpartition(".")
                    entry = by_name.get(fqn) if sep else None
                    if entry is not None and isinstance(item, torch.Tensor):
                        annotate_slot(item, fqn, entry)
                    continue
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return payload


def canonical_parameter_names(model: torch.nn.Module) -> Dict[Any, str]:
    """Map each live parameter to the canonical FQN used for shard metadata.

    ``_parameter_shapes``, ``fsdp_padded_params`` and the tracer's state FQNs all
    come from the deduplicated ``named_parameters()``, so a tied parameter
    resolves to its first name. Resolving the same way here keeps tied
    parameters aligned with the pass metadata instead of picking up a later
    alias that has no recorded shape or padding entry.
    """
    return {parameter: name for name, parameter in model.named_parameters()}


def has_strided_placement(layout: Any) -> bool:
    """Whether any axis uses a strided shard (interleaved, not rectangular)."""
    return any(
        isinstance(placement, StridedShard) for placement in (layout.placements or ())
    )


def trim_padded_dim0(
    global_shape: Tuple[int, ...], n: int, n_pad: int
) -> Optional[Tuple[int, ...]]:
    """Replace the padded FSDP dim 0 with the parameter's true global size."""
    if n_pad <= 0 or not global_shape or global_shape[0] % n_pad:
        return None
    true_dim0 = n * (global_shape[0] // n_pad)
    if true_dim0 == global_shape[0]:
        return None
    return (true_dim0, *global_shape[1:])


def shard_chunk_metadata(
    layout: Any,
    rank_index: int,
    global_shape: Tuple[int, ...],
    padded_entry: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, ...], ChunkStorageMetadata]:
    """Describe one rank's chunk, clipped to the parameter's true global extent.

    The offsets follow the padded geometry the live parameter is physically
    stored in. A pad-sharded parameter reports the true (unpadded) global shape
    and clips its chunk to it, so the metadata never claims rows past the real
    tensor while staying aligned with the local storage.
    """
    area = infer_slice_area_by_layout(layout, rank_index, global_shape)
    offsets = tuple(begin for begin, _ in area)
    sizes = tuple(end - begin for begin, end in area)
    if padded_entry is not None:
        trimmed = trim_padded_dim0(global_shape, *padded_entry)
        if trimmed is not None:
            global_shape = trimmed
            clamped_offsets = []
            clamped_sizes = []
            for size, extent, offset in zip(sizes, global_shape, offsets):
                # Keep the padded offset geometry but never let a chunk start
                # or end past the true extent. A rank that owns only padding
                # rows collapses to a zero-size chunk at the boundary.
                start = min(offset, extent)
                end = min(offset + size, extent)
                clamped_offsets.append(start)
                clamped_sizes.append(max(0, end - start))
            offsets = tuple(clamped_offsets)
            sizes = tuple(clamped_sizes)
    return global_shape, ChunkStorageMetadata(offsets=offsets, sizes=sizes)


def _set_shard_metadata(
    parameter: torch.nn.Parameter,
    layout: Any,
    chunk: ChunkStorageMetadata,
    global_shape: Tuple[int, ...],
) -> None:
    """Attach one parameter's layout and chunk description.

    ``_sharding_spec`` aliases are only set for layouts that actually shard a
    tensor dim, matching the eager path so consumers do not wrap a replicated
    parameter in a sharded DTensor.
    """
    placements = tuple(layout.placements or ())
    if any(placement.is_shard() for placement in placements):
        setattr(parameter, "_sharding_spec", layout)
        setattr(parameter, "_layout", layout)
        setattr(parameter, "_placements", placements)
    setattr(
        parameter,
        CHUNK_INFO,
        ChunkInfo(chunk=chunk, global_shape=global_shape),
    )


def attach_shard_metadata(compiler: Any, *, strict_strided: bool = True) -> None:
    """Annotate graph-sharded parameters for HF export and DCP consumers.

    Called by checkpoint/export paths after compilation; keeping it out of
    ``compile`` avoids building layouts for callers that never checkpoint.

    Args:
        compiler: Compiled graph compiler owning the sharded live model.
        strict_strided: Raise on a strided FSDP shard (export cannot describe
            it). Optimizer resharding passes False so an unrelated strided
            parameter does not block resharding of parameters that do have
            optimizer state.
    """
    if compiler.joint_graph is None:
        return
    layouts = compiler.parameter_layouts
    if not layouts:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    names = canonical_parameter_names(compiler.model)
    padded = getattr(compiler.joint_graph.graph_module, "fsdp_padded_params", {})
    for parameter, layout in layouts.items():
        global_shape = tuple(layout.tensor_shape or ())
        if not global_shape:
            continue
        if has_strided_placement(layout):
            if strict_strided:
                raise RuntimeError(
                    f"Cannot describe checkpoint metadata for {names.get(parameter)!r}: "
                    "its FSDP shard is strided (for example TP and FSDP shard the same "
                    "axis), so the local shard is not a single rectangular chunk. "
                    "Checkpoint export and optimizer-state resharding of strided "
                    "shards are not supported yet."
                )
            continue
        rank_list = tuple(layout.rank_list)
        if rank not in rank_list:
            continue
        global_shape, chunk = shard_chunk_metadata(
            layout,
            rank_list.index(rank),
            global_shape,
            padded.get(names.get(parameter)),
        )
        _set_shard_metadata(parameter, layout, chunk, global_shape)


def attach_source_shard_metadata(compiler: Any) -> None:
    """Annotate source-local parameters for pre-compile checkpoint load/save.

    Restore happens before lazy compilation, when TP/EP parameters are still
    plain local tensors. Without this, DCP would treat every rank as the first
    source shard and reject a global checkpoint whose size differs from the
    local shape.
    """
    if compiler.joint_graph is not None:
        return
    layouts = compiler.source_parameter_layouts
    if not layouts:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    for parameter, layout in layouts.items():
        global_shape = tuple(layout.tensor_shape or ())
        if not global_shape or has_strided_placement(layout):
            continue
        rank_list = tuple(layout.rank_list)
        if rank not in rank_list:
            continue
        _, chunk = shard_chunk_metadata(layout, rank_list.index(rank), global_shape)
        _set_shard_metadata(parameter, layout, chunk, global_shape)


def build_source_parameter_layouts(compiler: Any, parameters: dict) -> Optional[dict]:
    """Describe source TP/EP storage before FSDP shards the live model."""
    if compiler.source_shard_info is None and compiler.mesh_context is None:
        return None
    compiler.ensure_expert_param_names()
    layouts = {}
    for name, parameter in parameters.items():
        layout = compiler.source_global_layout(name, len(parameter.shape))
        if layout is None:
            continue
        placements = tuple(layout.placements or ())
        if not placements or not any(placement.is_shard() for placement in placements):
            continue
        if any(isinstance(placement, StridedShard) for placement in placements):
            raise RuntimeError(
                f"Cannot describe checkpoint metadata for {name!r}: its source "
                "TP/EP shard is strided and cannot be represented as one "
                "rectangular chunk."
            )
        global_shape = layout.get_global_shape(tuple(parameter.shape))
        layout.set_tensor_meta(
            tuple(global_shape),
            make_contiguous_strides_for(global_shape),
            parameter.dtype,
        )
        layouts[parameter] = layout
    return layouts or None

# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""RaggedShard geometry adapters for distributed checkpoint planners."""
from math import prod
from typing import Any, NamedTuple

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    WriteItem,
    WriteItemType,
)
from hyper_parallel.core.dtensor._ragged_utils import _compute_ragged_slice
from hyper_parallel.core.dtensor.dtensor import DTensor


class RaggedCheckpointBox(NamedTuple):
    """One regular global box backed by a contiguous local flat interval."""

    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    local_flat_start: int
    local_flat_end: int


def _decompose_flat_interval(
    shape: tuple[int, ...],
    flat_start: int,
    flat_end: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Decompose a row-major flat interval into ordered N-D boxes."""
    total_numel = prod(shape)
    if not shape:
        raise ValueError("Ragged checkpoint geometry requires a non-scalar global shape")
    if flat_start < 0 or flat_end < flat_start or flat_end > total_numel:
        raise ValueError(
            "Invalid flat interval for Ragged checkpoint geometry, "
            f"got interval=({flat_start}, {flat_end}), shape={shape!r}"
        )
    if flat_start == flat_end:
        return ()

    boxes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _decompose_axis(
        axis: int,
        start: int,
        end: int,
        prefix_offsets: tuple[int, ...],
    ) -> None:
        if axis == len(shape) - 1:
            boxes.append(
                (
                    prefix_offsets + (start,),
                    (1,) * len(prefix_offsets) + (end - start,),
                )
            )
            return

        stride = prod(shape[axis + 1:])
        start_block, start_remainder = divmod(start, stride)
        end_block = (end - 1) // stride
        if start_block == end_block:
            _decompose_axis(
                axis + 1,
                start_remainder,
                start_remainder + end - start,
                prefix_offsets + (start_block,),
            )
            return

        complete_start = start_block
        if start_remainder:
            _decompose_axis(
                axis + 1,
                start_remainder,
                stride,
                prefix_offsets + (start_block,),
            )
            complete_start += 1

        complete_end, end_remainder = divmod(end, stride)
        if complete_start < complete_end:
            boxes.append(
                (
                    prefix_offsets
                    + (complete_start,)
                    + (0,) * (len(shape) - axis - 1),
                    (1,) * len(prefix_offsets)
                    + (complete_end - complete_start,)
                    + shape[axis + 1:],
                )
            )

        if end_remainder:
            _decompose_axis(
                axis + 1,
                0,
                end_remainder,
                prefix_offsets + (complete_end,),
            )

    _decompose_axis(0, flat_start, flat_end, ())
    return tuple(boxes)


def compute_ragged_boxes(tensor: DTensor) -> tuple[RaggedCheckpointBox, ...]:
    """Return ordered N-D boxes covering one RaggedShard local flat tensor."""
    layout = tensor.layout
    if layout.ragged_shard is None:
        raise ValueError("compute_ragged_boxes requires a RaggedShard DTensor")

    global_shape = tuple(tensor.shape)
    ragged_slice = _compute_ragged_slice(global_shape, layout)
    raw_boxes = _decompose_flat_interval(
        global_shape,
        ragged_slice.flat_start,
        ragged_slice.flat_end,
    )
    boxes: list[RaggedCheckpointBox] = []
    local_flat_offset = 0
    for offsets, sizes in raw_boxes:
        box_numel = prod(sizes)
        boxes.append(
            RaggedCheckpointBox(
                offsets=offsets,
                sizes=sizes,
                local_flat_start=local_flat_offset,
                local_flat_end=local_flat_offset + box_numel,
            )
        )
        local_flat_offset += box_numel

    if local_flat_offset != ragged_slice.local_numel:
        raise ValueError(
            "Ragged checkpoint boxes do not cover the local flat tensor, "
            f"covered={local_flat_offset}, expected={ragged_slice.local_numel}"
        )
    return tuple(boxes)


def create_ragged_write_items(fqn: str, tensor: DTensor) -> list[WriteItem]:
    """Create one standard N-D checkpoint write item per RaggedShard box."""
    local_tensor = tensor.to_local()
    dtype_str = str(local_tensor.dtype) if hasattr(local_tensor, "dtype") else "unknown"
    properties = TensorProperties(dtype=dtype_str)
    items: list[WriteItem] = []
    for box in compute_ragged_boxes(tensor):
        chunk = ChunkStorageMetadata(offsets=box.offsets, sizes=box.sizes)
        items.append(
            WriteItem(
                index=MetadataIndex(fqn=fqn, offset=box.offsets, index=None),
                type=WriteItemType.TENSOR,
                tensor_data={
                    "chunk": chunk,
                    "properties": properties,
                    "size": tuple(tensor.shape),
                },
            )
        )
    return items


def get_ragged_box_tensor(tensor: DTensor, index: MetadataIndex) -> Any:
    """Return the local flat view corresponding to one global N-D box."""
    requested_offset = tuple(index.offset) if index.offset is not None else None
    for box in compute_ragged_boxes(tensor):
        if box.offsets == requested_offset:
            local_flat = tensor.to_local().reshape((-1,))
            return local_flat[
                box.local_flat_start:box.local_flat_end
            ].reshape(box.sizes)
    raise ValueError(
        "Ragged checkpoint box was not found in the local DTensor, "
        f"fqn={index.fqn!r}, offset={index.offset!r}"
    )

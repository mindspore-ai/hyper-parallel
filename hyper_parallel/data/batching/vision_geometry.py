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
"""CPU media geometry prepared after collation, separate from language position IDs."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isqrt
from typing import Any

from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from hyper_parallel.components.modules.vision.learned_axial_vision_transformer import (
    frame_indices,
    temporal_merge_indices,
)
from hyper_parallel.data.batching.build_collate_fn import build_omni_collate_fn


@dataclass(frozen=True, eq=False)
class HostVisionGeometry(Mapping[str, Any]):
    """Picklable CPU control metadata retained through ordinary device batch preparation."""

    fields: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Read one precomputed field without moving its tensor storage."""
        return self.fields[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate descriptor names for the vision family's local kwargs copy."""
        return iter(self.fields)

    def __len__(self) -> int:
        """Return the number of geometry fields."""
        return len(self.fields)


@dataclass
class AxialVisionCollator:
    """Attach immutable host geometry to a complete collated image or video batch.

    The descriptor travels as metadata through ordinary TP transport and is
    consumed inside the complete vision family. Language position_ids remain
    independent. Mixed image/video batches require separate descriptors and
    are rejected until their model orchestration declares that distinction.
    """

    collate: Callable
    num_grid_per_side: int
    spatial_merge_size: int

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Precompute indices and packed clip boundaries while processor grids remain on CPU."""
        batch = dict(self.collate(samples))
        grids = [batch[name] for name in ("image_grid_thw", "video_grid_thw") if batch.get(name) is not None]
        if not grids:
            return batch
        if len(grids) != 1:
            raise ValueError("Axial geometry requires separate batches for images and videos")
        grid = grids[0]
        if grid.device.type != "cpu":
            raise ValueError("Vision geometry must be prepared before device batch transport")
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, self.num_grid_per_side, align_corners=True, spatial_merge_size=self.spatial_merge_size,
        )
        cumulative = get_vision_cu_seqlens(grid)
        lengths = tuple(cumulative.diff().tolist())
        batch["vision_geometry"] = HostVisionGeometry({
            "grid_thw": grid,
            "position_ids": get_vision_position_ids(grid, self.spatial_merge_size),
            "interp_indices": indices, "interp_weights": weights,
            "cu_seqlens": cumulative, "max_seqlen": max(lengths), "vision_attention_lengths": lengths,
        })
        return batch


def build_axial_vision_collate_fn(
    mesh_context: Any = None, *, num_position_embeddings: int, spatial_merge_size: int,
    pad_to_length: int | None = None,
) -> AxialVisionCollator:
    """Build normal Omni collation plus CPU geometry for an axial vision family."""
    side = isqrt(num_position_embeddings)
    if side < 1 or side * side != num_position_embeddings:
        raise ValueError("Vision learned positions must form a positive square grid")
    if isinstance(spatial_merge_size, bool) or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be a positive integer")
    collate = build_omni_collate_fn(mesh_context=mesh_context, pad_to_length=pad_to_length)
    return AxialVisionCollator(collate, side, spatial_merge_size)


@dataclass
class LearnedVisionCollator:
    """Prepare clip-level geometry for learned spatial/time vision before device movement."""

    collate: Callable
    num_grid_per_side: int
    merge_kernel_size: tuple[int, int]

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Keep vision positions distinct from decoder positions in a host descriptor."""
        batch = dict(self.collate(samples))
        grid = batch.get("image_grid_thw")
        if grid is None:
            return batch
        if grid.device.type != "cpu":
            raise ValueError("Learned vision geometry must be prepared before device transport")
        cumulative = get_vision_cu_seqlens(grid, merge_temporal=True)
        lengths = tuple(cumulative.diff().tolist())
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, self.num_grid_per_side, mode="bicubic", align_corners=False,
        )
        batch["vision_geometry"] = HostVisionGeometry({
            "grid_thw": grid, "position_ids": get_vision_position_ids(grid, spatial_merge_size=1),
            "interp_indices": indices, "interp_weights": weights,
            "frame_index": frame_indices(grid, {}),
            "temporal_merge_index": temporal_merge_indices(grid, self.merge_kernel_size, {}),
            "cu_seqlens": cumulative, "max_seqlen": max(lengths), "vision_attention_lengths": lengths,
        })
        return batch


def build_learned_vision_collate_fn(
    mesh_context: Any = None, *, num_grid_per_side: int, merge_kernel_size: list[int],
    pad_to_length: int | None = None,
) -> LearnedVisionCollator:
    """Build Omni collation plus independent spatial, temporal and packed-attention geometry."""
    if (isinstance(num_grid_per_side, bool) or not isinstance(num_grid_per_side, int) or num_grid_per_side < 1
            or len(merge_kernel_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in merge_kernel_size)):
        raise ValueError("Learned vision requires a positive table side and two positive integer merge dimensions")
    return LearnedVisionCollator(build_omni_collate_fn(mesh_context=mesh_context, pad_to_length=pad_to_length),
                                 num_grid_per_side, tuple(merge_kernel_size))

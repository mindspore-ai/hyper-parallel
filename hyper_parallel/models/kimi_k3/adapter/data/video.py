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
"""Deterministic local-video sampling for explicitly rendered K3 training records."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

# The project is Torch-only; the legacy platform lint does not apply.
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.omni.video import decode_video_frames


def preprocess_rendered_medias(medias: Sequence[Mapping[str, Any]], *, processor: Any,
                               source_directory: str | None = None, max_video_frames: int = 128,
                               patch_size: int = 14) -> tuple[torch.Tensor, torch.Tensor, tuple[dict[str, Any], ...]]:
    """Reuse the released image preprocessor for frame groups and still images.

    Args:
        medias: Ordered image/video descriptors paired with pre-rendered media prompts.
            A video contributes one prompt per group of up to four sampled frames.
        processor: K3 processor exposing ``media_processor.preprocess``.
        source_directory: Directory for relative image and video paths.
        max_video_frames: Maximum number of sampled frames in one video.
        patch_size: Patch geometry used for an empty media list.

    Returns:
        Concatenated pixels, temporal grids and one provenance record per grid.
        Time metadata is diagnostic only and does not add text or position embeddings.
    """
    if not isinstance(medias, Sequence) or isinstance(medias, (str, bytes)):
        raise ValueError("Rendered K3 medias must be an ordered sequence of descriptors")
    media_processor = getattr(processor, "media_processor", None)
    if media_processor is None:
        raise ValueError("Raw K3 medias require the released processor.media_processor")
    all_pixels, all_grids, metadata = [], [], []
    for media_index, media in enumerate(medias):
        frames, provenance = _load_media(media, source_directory, max_video_frames)
        for group_start in range(0, len(frames), 4):
            group = frames[group_start:group_start + 4]
            encoded = media_processor.preprocess([{"type": "image", "image": frame} for frame in group],
                                                 return_tensors="pt")
            grids = torch.as_tensor(encoded["grid_thws"])
            if grids.shape != (len(group), 3) or not bool((grids[:, 0] == 1).all()):
                raise ValueError("K3 frame preprocessing must return one still-image grid per frame")
            if not bool((grids[:, 1:] == grids[0, 1:]).all()):
                raise ValueError("Frames in one K3 temporal group must have matching processed spatial grids")
            grid = grids[:1].clone()
            grid[0, 0] = len(group)
            all_grids.append(grid)
            all_pixels.append(torch.as_tensor(encoded["pixel_values"]))
            item = dict(provenance, source_media_index=media_index, frame_start=group_start,
                        frame_count=len(group), sample_index=0)
            for field in ("requested_timestamps", "frame_timestamps"):
                if field in item:
                    item[field] = item[field][group_start:group_start + len(group)]
            metadata.append(item)
    pixels = torch.cat(all_pixels) if all_pixels else torch.empty(0, 3, patch_size, patch_size)
    grids = torch.cat(all_grids) if all_grids else torch.empty(0, 3, dtype=torch.long)
    return pixels, grids, tuple(metadata)


def _load_media(media: Mapping[str, Any], source_directory: str | None, max_frames: int
                ) -> tuple[list[Any], dict[str, Any]]:
    """Resolve one source without mutating the dataset's descriptors."""
    if not isinstance(media, Mapping):
        raise ValueError("Each rendered K3 media must be an image or video descriptor")
    if media.get("type") == "video":
        return decode_video_frames(media, source_directory=source_directory, max_frames=max_frames)
    if media.get("type") != "image" or "image" not in media:
        raise ValueError("Rendered K3 medias must contain image/image or video/video descriptors")
    image = media["image"]
    if isinstance(image, (str, os.PathLike)):
        image = os.fspath(image)
        if (source_directory is not None and not os.path.isabs(image)
                and not image.startswith(("data:", "http://", "https://"))):
            image = os.path.join(source_directory, image)
    return [image], {"type": "image", "source": image if isinstance(image, str) else None}

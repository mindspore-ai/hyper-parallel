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

import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

# The project is Torch-only; the legacy platform lint does not apply.
import torch  # pylint: disable=forbidden-backend-import


def _seconds(value: Any, name: str) -> float:
    """Reject invalid timestamps before opening a decoder."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative time in seconds")
    return float(value)


def _uniform_sampling_times(spec: Mapping[str, Any], duration: float | None, limit: int) -> list[float]:
    """Generate bounded timestamps in an explicitly selected half-open interval."""
    count = spec.get("num_frames", 4)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= limit:
        raise ValueError(f"video num_frames must be an integer in [1, {limit}]")
    start = _seconds(spec.get("start_time", 0.0), "video start_time")
    end = spec.get("end_time", duration)
    if end is None:
        raise ValueError("video duration is unavailable; provide timestamps or end_time")
    end = _seconds(end, "video end_time")
    if end <= start or (duration is not None and end > duration + 1e-6):
        raise ValueError("video interval must satisfy 0 <= start_time < end_time <= duration")
    return [start + (end - start) * index / count for index in range(count)]


def _sampling_times(spec: Mapping[str, Any], duration: float | None, limit: int) -> list[float]:
    """Choose explicit timestamps or uniformly spaced times in a half-open clip."""
    if "timestamps" in spec:
        if any(name in spec for name in ("num_frames", "start_time", "end_time")):
            raise ValueError("video timestamps cannot be combined with num_frames/start_time/end_time")
        timestamps = spec["timestamps"]
        if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):
            raise ValueError("video timestamps must be a sequence of seconds")
        times = [_seconds(value, "video timestamp") for value in timestamps]
    else:
        times = _uniform_sampling_times(spec, duration, limit)
    if not 1 <= len(times) <= limit:
        raise ValueError(f"a video must select between 1 and {limit} frames")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("video timestamps must be strictly increasing")
    if duration is not None and times[-1] >= duration:
        raise ValueError("video timestamps must be strictly below the video duration")
    return times


def _sample_decoded_frames(container: Any, stream: Any, times: Sequence[float]) -> tuple[list[Any], list[float]]:
    """Select nearest PTS frames in one streaming pass, breaking ties toward the earlier frame."""
    origin = (stream.start_time or 0) * stream.time_base
    frames, actual_times = [], []
    previous_frame, previous_time = None, None
    for frame in container.decode(stream):
        if frame.pts is None:
            raise ValueError("video frames require presentation timestamps for deterministic sampling")
        current_time = float(frame.pts * frame.time_base - origin)
        if previous_time is not None and current_time < previous_time:
            raise ValueError("video decoder produced non-monotonic presentation timestamps")
        while len(frames) < len(times) and times[len(frames)] <= current_time:
            requested = times[len(frames)]
            # PTS conversion can round exact midpoints to opposite sides in float64.
            earlier = previous_time is not None and (
                requested - previous_time <= current_time - requested
                or math.isclose(requested - previous_time, current_time - requested, rel_tol=0.0, abs_tol=1e-12)
            )
            selected, timestamp = (previous_frame, previous_time) if earlier else (frame, current_time)
            frames.append(selected.to_image())
            actual_times.append(timestamp)
        if len(frames) == len(times):
            break
        previous_frame, previous_time = frame, current_time
    _finish_frame_sampling(frames, actual_times, times, previous_frame, previous_time, origin)
    return frames, actual_times


def _finish_frame_sampling(frames: list[Any], actual_times: list[float], times: Sequence[float],
                           previous_frame: Any, previous_time: float | None, origin: Any) -> None:
    """Validate observed EOF bounds and select the final frame for remaining timestamps."""
    if previous_frame is None and not frames:
        raise ValueError("video contains no decodable frames")
    if len(frames) < len(times) and previous_frame.duration is not None and previous_frame.duration > 0:
        duration = float((previous_frame.pts + previous_frame.duration) * previous_frame.time_base - origin)
        if times[-1] >= duration:
            raise ValueError("video timestamps must be strictly below the video duration")
    while len(frames) < len(times):
        frames.append(previous_frame.to_image())
        actual_times.append(previous_time)


def _scan_video_duration(container: Any, stream: Any) -> float:
    """Find a reliable local presentation end without retaining decoded frames."""
    last_frame = None
    for frame in container.decode(stream):
        last_frame = frame
    if last_frame is None:
        raise ValueError("video contains no decodable frames")
    if last_frame.pts is None or last_frame.duration is None or last_frame.duration <= 0:
        raise ValueError("video duration is unavailable; provide timestamps or end_time")
    origin = (stream.start_time or 0) * stream.time_base
    duration = float((last_frame.pts + last_frame.duration) * last_frame.time_base - origin)
    if duration <= 0:
        raise ValueError("video duration is unavailable; provide timestamps or end_time")
    return duration


def decode_video_frames(spec: Mapping[str, Any], *, source_directory: str | None = None,
                        max_frames: int = 128) -> tuple[list[Any], dict[str, Any]]:
    """Decode bounded frames from a local video without imposing a text template.

    Args:
        spec: A video descriptor with ``video`` path and either ``timestamps`` or
            ``num_frames`` (default four), ``start_time`` and ``end_time`` in seconds.
            Uniform samples include the start and exclude the end. Each requested
            time selects its nearest presentation frame; ties select the earlier one.
        source_directory: Directory used to resolve relative source paths.
        max_frames: Maximum requested frames in this video.

    Returns:
        RGB PIL frames and provenance with requested/actual timestamps. Repeated
        actual frames are retained when several times select the same source frame.
    """
    allowed = {"type", "video", "timestamps", "num_frames", "start_time", "end_time"}
    if set(spec) - allowed:
        raise ValueError(f"Unsupported K3 video fields: {sorted(set(spec) - allowed)}")
    if "video" not in spec:
        raise ValueError("K3 video descriptors require a video file path")
    path = os.fspath(spec["video"])
    if "://" in path or path.startswith("data:"):
        raise ValueError("K3 video decoding requires a local file path")
    if not os.path.isabs(path) and source_directory is not None:
        path = os.path.join(source_directory, path)
    path = os.path.abspath(path)
    # Video is optional; image/text preprocessing must not require PyAV.
    try:
        import av  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("Raw K3 video input requires PyAV; install the optional 'av' package") from exc
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("K3 video input contains no video stream")
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        scan_duration = duration is None and "timestamps" not in spec and "end_time" not in spec
        if scan_duration:
            duration = _scan_video_duration(container, stream)
        times = _sampling_times(spec, duration, max_frames)
        if scan_duration:
            with av.open(path) as sampling_container:
                frames, actual = _sample_decoded_frames(sampling_container, sampling_container.streams.video[0], times)
        else:
            frames, actual = _sample_decoded_frames(container, stream, times)
    return frames, {"type": "video", "source": path, "requested_timestamps": tuple(times),
                    "frame_timestamps": tuple(actual), "duration": duration,
                    "sampling": "nearest_pts_earlier_tie"}


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

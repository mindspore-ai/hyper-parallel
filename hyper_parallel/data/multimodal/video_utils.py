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
"""Video decoding + temporal frame sampling for the Qwen3-Omni data
pipeline.

Decode goes through PyAV (``_decode_video_container``) with the output
contract (T, C, H, W) float tensor + ``fps`` + ``total_frames``; audio
resampling reuses the scipy resampler from ``audio_utils``.

Frame values stay raw: decoded RGB uint8 → float32 on the 0..255 scale, handed
to the processor **before** rescale/normalize — HF 4.57.1's
``Qwen2VLVideoProcessor`` (via ``BaseVideoProcessor``) does its own resize +
rescale_and_normalize + patch packing, so the transform only owes it
pre-sampled (T, C, H, W) tensors.
"""

import io
import math
from typing import ByteString, Dict, List, Optional, Tuple, Union

import av
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, functional

from .audio_utils import extract_audio_from_video, resample_audio

DEFAULT_FPS = 2.0

VideoInput = Union[
    List["torch.Tensor"],
    Dict[str, "np.ndarray"],
    ByteString,
    str,
]


def _align_to_factor_remainder_floor(n: float, factor: int, remainder: int) -> int:
    """Round down n to the nearest value of the form factor * k + remainder."""
    adjusted = n - remainder
    return int(adjusted // factor) * factor + remainder


def _align_to_factor_remainder_ceil(n: float, factor: int, remainder: int) -> int:
    """Round up n to the nearest value of the form factor * k + remainder."""
    adjusted = n - remainder
    return math.ceil(adjusted / factor) * factor + remainder


def calculate_frame_indices(
    total_frames: int,
    video_fps: Union[int, float],
    fps: float = DEFAULT_FPS,
    frame_factor: int = None,
    frame_factor_remainder: int = 0,
    min_frames: int = None,
    max_frames: int = None,
    **kwargs,
) -> Tuple[List[int], int]:
    """Calculate frame indices to sample and padding count.

    Uniform sampling with min/max frame clamps and frame_factor alignment
    (remainder supported).
    """
    r = frame_factor_remainder
    if frame_factor is not None:
        if frame_factor <= 0:
            raise ValueError(f"frame_factor must be a positive integer, got {frame_factor}")
        if not 0 <= r < frame_factor:
            raise ValueError(f"frame_factor_remainder must be in [0, {frame_factor}), got {r}")

    nframes = total_frames / video_fps * fps

    if min_frames is not None:
        if frame_factor is not None:
            min_frames = _align_to_factor_remainder_ceil(min_frames, frame_factor, r)
        nframes = max(min_frames, nframes)

    if max_frames is not None:
        if frame_factor is not None:
            max_frames = _align_to_factor_remainder_floor(max_frames, frame_factor, r)
        nframes = min(max_frames, nframes)

    if frame_factor is not None:
        nframes = _align_to_factor_remainder_floor(nframes, frame_factor, r)
        min_valid = r if r > 0 else frame_factor
        nframes = max(nframes, min_valid)

    nframes = int(max(1, nframes))

    pad_count = 0
    if nframes > total_frames:
        pad_count = nframes - total_frames
        sample_count = total_frames
    else:
        sample_count = nframes

    if sample_count > 0:
        indices = np.linspace(0, total_frames - 1, sample_count).round().astype(int).tolist()
    else:
        indices = []

    return indices, pad_count


def smart_video_nframes(
    video: torch.Tensor,
    video_fps: Union[int, float],
    fps: float = DEFAULT_FPS,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Union[float, int, torch.Tensor]]]:
    """Sample video frames with fps and alignment constraints.

    Returns (video, metadata): metadata['fps'] / ['total_num_frames']
    describe the input BEFORE sampling; ['frames_indices'] maps each output
    frame (including repeated padding) back to the input timeline.
    """
    total_frames = video.shape[0]

    if "frames" in kwargs:
        target_frames = kwargs["frames"]
        indices, pad_count = calculate_frame_indices(
            total_frames=total_frames,
            video_fps=video_fps,
            fps=fps,
            frame_factor=frame_factor,
            min_frames=target_frames,
            max_frames=target_frames,
        )
    else:
        indices, pad_count = calculate_frame_indices(
            total_frames=total_frames,
            video_fps=video_fps,
            fps=fps,
            frame_factor=frame_factor,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    video = video[indices]

    if pad_count > 0:
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)
        video = torch.cat([video, last_frame], dim=0)
        indices = indices + [indices[-1]] * pad_count

    return video, {
        "fps": video_fps,
        "total_num_frames": total_frames,
        "frames_indices": torch.tensor(indices, dtype=torch.long),
    }


def smart_resize(
    video: torch.Tensor,
    scale_factor: int = None,
    video_min_pixels: int = None,
    video_max_pixels: int = None,
    max_ratio: int = None,
    **kwargs,
) -> torch.Tensor:
    """Aspect-preserving resize with pixel/alignment constraints."""
    if video.ndim != 4:
        raise ValueError(f"video must be 4-dim, but got {video.ndim}")
    _, _, height, width = video.shape

    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if video_max_pixels is not None and h_bar * w_bar > video_max_pixels:
        beta = math.sqrt((height * width) / video_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)

    if video_min_pixels is not None and h_bar * w_bar < video_min_pixels:
        beta = math.sqrt(video_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)

    return functional.resize(
        video,
        [h_bar, w_bar],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()


def smart_audio_nframes(
    audio: Optional[np.ndarray], audio_fps: Optional[int], sample_rate: int = 16000, **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, Union[float, int]]]:
    """Resample extracted video audio to 16 kHz (scipy instead of V's librosa)."""
    if audio is not None and audio_fps is not None and audio_fps != sample_rate:
        audio = resample_audio(audio.astype(np.float32), int(audio_fps), sample_rate)
    num_frames = len(audio) if audio is not None else 0
    return audio, {"fps": sample_rate, "total_num_frames": num_frames}


def _decode_video_container(video_input: Union[str, ByteString]) -> Tuple[torch.Tensor, float, int]:
    """Decode a str/bytes video container with PyAV (replaces torchcodec).

    Returns (video, fps, total_frames) where video is (T, C, H, W) float32
    RGB on the 0..255 scale (normalize happens downstream in the HF video
    processor, matching the uint8→float decode output convention).
    """
    if isinstance(video_input, str):
        if "http://" in video_input or "https://" in video_input:
            raise ValueError(
                "URL video input is not supported in the offline NPU env; "
                f"download {video_input} locally first."
            )
        container = av.open(video_input)
    else:
        container = av.open(io.BytesIO(video_input))
    try:
        stream = container.streams.video[0]
        frames = []
        for frame in container.decode(stream):
            # to_ndarray → (H, W, C) uint8 for rgb24
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError("video container has no decodable frames")
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
        avg_fps = float(stream.average_rate) if stream.average_rate else DEFAULT_FPS
        total_frames = video.shape[0]
    finally:
        container.close()
    return video, avg_fps, total_frames


def _pil_images_to_tensor(images: List) -> torch.Tensor:
    """Convert PIL images to (T, C, H, W) float tensor."""
    tensors = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensors.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
    return torch.stack(tensors).float()


def _dict_to_video_audio(video_dict: Dict[str, "np.ndarray"], default_video_fps: float = DEFAULT_FPS):
    """Convert a paired-A/V dict to (video tensor, video_fps, audio, audio_fps).

    Accepts ``{"video": (T,H,W,C)|(T,C,H,W) ndarray,
    "audio": ndarray?, "video_fps"?, "audio_fps"?}``.
    """
    if "video" not in video_dict:
        raise ValueError("Dict input must contain a 'video' ndarray (offline frames-bytes layout is not supported)")
    video_np = video_dict["video"]
    if video_np.ndim != 4:
        raise ValueError(f"Video array must be 4D, got shape {video_np.shape}")
    if video_np.shape[-1] == 3:
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)
    else:
        video = torch.from_numpy(video_np)
    video_fps = video_dict.get("video_fps", 30.0)
    audio = video_dict.get("audio", None)
    audio_fps = video_dict.get("audio_fps", None)
    return video, video_fps, audio, audio_fps


def _load_and_process_video(video_input: VideoInput, use_audio_in_video: bool = True, **kwargs):
    """Load + temporally sample one video (replaces V's torchcodec branch).

    Returns (video, audio, audio_fps, metadata): video is the sampled
    (T', C, H, W) tensor (spatial resize is left to the HF video processor —
    4.57.1 does its own smart_resize; V 5.x is fed the same convention via
    ``do_sample_frames=False``), audio the extracted 16 kHz mono track or
    None, metadata the source-fps frame-index mapping.
    """
    if isinstance(video_input, (list, torch.Tensor)) and not isinstance(video_input, dict):
        # Pre-decoded frame stack: (T, C, H, W) tensor or list of per-frame arrays.
        video = video_input if isinstance(video_input, torch.Tensor) else torch.from_numpy(
            np.stack(video_input)
        )
        if video.dim() != 4:
            raise ValueError(f"video must be 4-dim (T, C, H, W), got {video.dim()}")
        video_fps = kwargs.get("fps", DEFAULT_FPS)
        video, metadata = smart_video_nframes(video, video_fps, **kwargs)
        return video, None, None, metadata

    if isinstance(video_input, dict):
        video, video_fps, audio, audio_fps = _dict_to_video_audio(
            video_input, default_video_fps=kwargs.get("fps", DEFAULT_FPS)
        )
        video, metadata = smart_video_nframes(video, video_fps, **kwargs)
        if audio is not None and audio_fps is not None:
            audio, _ = smart_audio_nframes(audio, audio_fps, **kwargs)
        else:
            audio, audio_fps = None, None
        return video, audio, audio_fps, metadata

    # str path / bytes container — PyAV decode (torchcodec absent in H env).
    video, container_fps, total_frames = _decode_video_container(video_input)
    indices, pad_count = calculate_frame_indices(
        total_frames=total_frames, video_fps=container_fps, **kwargs
    )
    video = video[indices]
    if pad_count > 0:
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)
        video = torch.cat([video, last_frame], dim=0)
        padded_indices = indices + [indices[-1]] * pad_count
    else:
        padded_indices = indices

    audio, audio_fps = None, None
    if use_audio_in_video:
        # Reuse the audio decoder (PyAV demux+decode), then resample its
        # native-rate output to 16 kHz (the AV-interleave convention).
        raw_audio, native_fps = extract_audio_from_video(video_input)
        if raw_audio is not None and native_fps is not None:
            audio = resample_audio(raw_audio.astype(np.float32), int(native_fps))
            audio_fps = 16000

    metadata = {
        "fps": container_fps,
        "total_num_frames": total_frames,
        "frames_indices": torch.tensor(padded_indices, dtype=torch.long),
    }
    return video, audio, audio_fps, metadata


def fetch_videos(
    videos: List[VideoInput], use_audio_in_video: bool = False, **kwargs
) -> Tuple[List[torch.Tensor], List[Optional[np.ndarray]]]:
    """Fetch + temporally sample videos (audio optional; AV interleave feeds it back in).

    Returns (video_tensors, audios): video_tensors[i] is the sampled
    (T_i, C, H, W) float tensor for video i; audios[i] is the extracted
    16 kHz mono track or None (only populated when ``use_audio_in_video``).
    """
    video_inputs, audios = [], []
    for i, video in enumerate(videos):
        try:
            processed, audio, audio_fps, _ = _load_and_process_video(
                video, use_audio_in_video=use_audio_in_video, **kwargs
            )
            video_inputs.append(processed)
            audios.append(audio if audio_fps is not None else None)
        except Exception as e:
            raise RuntimeError(f"Failed to process video {i}: {e}") from e
    return video_inputs, audios


def fetch_videos_metadata(
    videos: List[VideoInput], use_audio_in_video: bool = False, **kwargs
):
    """Fetch + sample videos with full metadata (V API-compatible).

    Returns (videos, video_metadata, audios, audio_metadata); the metadata
    uses source fps/total frames and maps
    sampled output frames (incl. padding repeats) back to the source timeline.
    """
    video_inputs, video_metadata_list, audios, audio_metadata_list = [], [], [], []
    for i, video in enumerate(videos):
        try:
            processed, audio, audio_fps, video_meta = _load_and_process_video(
                video, use_audio_in_video=use_audio_in_video, **kwargs
            )
            video_inputs.append(processed)
            video_metadata_list.append(video_meta)
            if audio is not None and audio_fps is not None:
                audios.append(audio)
                audio_metadata_list.append({"fps": audio_fps, "total_num_frames": len(audio)})
            else:
                audios.append(None)
                audio_metadata_list.append(None)
        except Exception as e:
            raise RuntimeError(f"Failed to process video {i}: {e}") from e
    return video_inputs, video_metadata_list, audios, audio_metadata_list


def load_video_from_path(video_path: str, use_audio_in_video: bool = False, **kwargs):
    """Load video from file path — V compatibility wrapper."""
    videos, video_meta, audios, audio_meta = fetch_videos_metadata(
        [video_path], use_audio_in_video=use_audio_in_video, **kwargs
    )
    return videos[0], video_meta[0], audios[0], audio_meta[0]


def load_video(video: VideoInput, **kwargs):
    """Unified video loading interface — V compatibility wrapper."""
    videos, video_meta, audios, audio_meta = fetch_videos_metadata([video], **kwargs)
    return videos[0], video_meta[0], audios[0], audio_meta[0]


__all__ = [
    "calculate_frame_indices",
    "fetch_videos",
    "fetch_videos_metadata",
    "load_video",
    "load_video_from_path",
    "smart_audio_nframes",
    "smart_resize",
    "smart_video_nframes",
]

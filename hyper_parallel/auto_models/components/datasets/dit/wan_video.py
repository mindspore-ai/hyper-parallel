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
"""Wan video Dataset transform and DiT batching utilities."""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional

from hyper_parallel.auto_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_parallel.auto_models.components.datasets.llm.online_dataset import build_online_dataset
from hyper_parallel.auto_models.components.datasets.llm.transform_dataset import apply_llm_data_transform
from hyper_parallel.auto_models.components.datasets.parallel import (
    DataLoaderParallelContext,
    create_dataloader_parallel_context,
)

logger = get_dataset_logger(__name__)


def _align_to_factor_remainder_floor(value: float, factor: int, remainder: int) -> int:
    adjusted = value - remainder
    return int(adjusted // factor) * factor + remainder


def _align_to_factor_remainder_ceil(value: float, factor: int, remainder: int) -> int:
    adjusted = value - remainder
    return math.ceil(adjusted / factor) * factor + remainder


def _calculate_frame_indices(
    *,
    total_frames: int,
    video_fps: float,
    fps: float,
    frame_factor: int | None,
    frame_factor_remainder: int,
    min_frames: int | None,
    max_frames: int | None,
) -> tuple[list[int], int]:
    if total_frames <= 0:
        raise ValueError("video must contain at least one frame")
    if frame_factor is not None and frame_factor <= 0:
        raise ValueError("frame_factor must be positive")

    target_frames = total_frames / max(video_fps, 1e-6) * fps
    if min_frames is not None:
        if frame_factor is not None:
            min_frames = _align_to_factor_remainder_ceil(min_frames, frame_factor, frame_factor_remainder)
        target_frames = max(min_frames, target_frames)
    if max_frames is not None:
        if frame_factor is not None:
            max_frames = _align_to_factor_remainder_floor(max_frames, frame_factor, frame_factor_remainder)
        target_frames = min(max_frames, target_frames)
    if frame_factor is not None:
        target_frames = _align_to_factor_remainder_floor(target_frames, frame_factor, frame_factor_remainder)
        target_frames = max(target_frames, frame_factor_remainder or frame_factor)
    target_frames = int(max(1, target_frames))

    sample_count = min(target_frames, total_frames)
    pad_count = max(0, target_frames - total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count).round().astype(int).tolist()
    return indices, pad_count


def _normalize_video_tensor(video: Any) -> torch.Tensor:
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    if not torch.is_tensor(video):
        raise TypeError(f"Expected decoded video tensor/array, got {type(video).__name__}")
    if video.ndim == 3:
        video = video.unsqueeze(0)
    if video.ndim != 4:
        raise ValueError(f"Video tensor must be 4D, got shape {tuple(video.shape)}")
    if video.shape[1] in (1, 3):
        return video
    if video.shape[-1] in (1, 3):
        return video.permute(0, 3, 1, 2)
    raise ValueError(f"Could not infer channel dimension for video shape {tuple(video.shape)}")


def _pil_frames_to_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
    tensors = []
    for frame in frames:
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        tensors.append(torch.from_numpy(np.array(frame)).permute(2, 0, 1))
    return torch.stack(tensors)


def _sample_video(
    video: torch.Tensor,
    *,
    video_fps: float,
    fps: float,
    frame_factor: int | None,
    frame_factor_remainder: int,
    min_frames: int | None,
    max_frames: int | None,
) -> torch.Tensor:
    indices, pad_count = _calculate_frame_indices(
        total_frames=int(video.shape[0]),
        video_fps=video_fps,
        fps=fps,
        frame_factor=frame_factor,
        frame_factor_remainder=frame_factor_remainder,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    video = video[indices]
    if pad_count:
        video = torch.cat([video, video[-1:].expand(pad_count, -1, -1, -1)], dim=0)
    return video


def _smart_resize(
    video: torch.Tensor,
    *,
    scale_factor: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    max_ratio: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    del kwargs
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


def _apply_dynamic_video_max_pixels(frame_count: int, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    video_total_pixels = kwargs.get("video_total_pixels")
    if video_total_pixels is None or frame_count <= 0:
        return dict(kwargs)

    temporal_merge_factor = kwargs.get("frame_factor", 2) or 2
    video_max_pixels = kwargs.get("video_max_pixels")
    video_min_pixels = kwargs.get("video_min_pixels")
    dynamic_max = video_total_pixels / frame_count * temporal_merge_factor
    if video_max_pixels is not None:
        dynamic_max = min(dynamic_max, video_max_pixels)
    if video_min_pixels is not None:
        dynamic_max = max(dynamic_max, video_min_pixels * 1.05)
    updated = dict(kwargs)
    updated["video_max_pixels"] = int(dynamic_max)
    return updated


def _load_video_with_torchcodec(video_input: Any, **kwargs: Any) -> tuple[torch.Tensor, float] | None:
    try:
        from torchcodec.decoders import VideoDecoder  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    try:
        decoder = VideoDecoder(video_input, device="cpu", num_ffmpeg_threads=0)
        metadata = decoder.metadata
        total_frames = int(metadata.num_frames or 0)
        if total_frames <= 0:
            raise ValueError("torchcodec did not report a positive frame count")
        video_fps = float(metadata.average_fps or kwargs.get("fps", 24.0))
        indices, pad_count = _calculate_frame_indices(
            total_frames=total_frames,
            video_fps=video_fps,
            fps=float(kwargs.get("fps", 24.0)),
            frame_factor=kwargs.get("frame_factor"),
            frame_factor_remainder=int(kwargs.get("frame_factor_remainder", 0)),
            min_frames=kwargs.get("min_frames"),
            max_frames=kwargs.get("max_frames"),
        )
        frame_count = len(indices) + pad_count
        resize_kwargs = _apply_dynamic_video_max_pixels(frame_count, kwargs)
        video = _smart_resize(_normalize_video_tensor(decoder.get_frames_at(indices).data), **resize_kwargs)
        if pad_count:
            video = torch.cat([video, video[-1:].expand(pad_count, -1, -1, -1)], dim=0)
        return video, video_fps
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.debug("torchcodec video decode failed: %s", error)
        return None


def _load_video_with_imageio(video_input: Any) -> tuple[torch.Tensor, float]:
    try:
        import imageio.v3 as iio  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("Decoding video paths/bytes requires torchcodec or imageio") from exc
    source = io.BytesIO(video_input) if isinstance(video_input, (bytes, bytearray)) else video_input
    frames = iio.imread(source, index=None)
    return _normalize_video_tensor(frames), 24.0


def _load_video(video_input: Any, *, data_dir: str | None = None, **kwargs: Any) -> torch.Tensor:
    if torch.is_tensor(video_input) or isinstance(video_input, np.ndarray):
        video = _normalize_video_tensor(video_input)
        video_fps = float(kwargs.get("source_fps", kwargs.get("fps", 24.0)))
    elif isinstance(video_input, list):
        if not video_input:
            raise ValueError("video frame list is empty")
        frames = []
        for frame in video_input:
            if isinstance(frame, (bytes, bytearray)):
                frame = Image.open(io.BytesIO(frame)).convert("RGB")
            frames.append(frame)
        video = _pil_frames_to_tensor(frames)
        video_fps = float(kwargs.get("source_fps", kwargs.get("fps", 24.0)))
    elif isinstance(video_input, dict):
        if "frames" in video_input:
            frames = []
            for frame in video_input["frames"]:
                if isinstance(frame, (bytes, bytearray)):
                    frame = Image.open(io.BytesIO(frame)).convert("RGB")
                frames.append(frame)
            video = _pil_frames_to_tensor(frames)
            video_fps = float(video_input.get("video_fps", kwargs.get("fps", 24.0)))
            indices, pad_count = _calculate_frame_indices(
                total_frames=int(video.shape[0]),
                video_fps=video_fps,
                fps=float(kwargs.get("fps", 24.0)),
                frame_factor=kwargs.get("frame_factor"),
                frame_factor_remainder=int(kwargs.get("frame_factor_remainder", 0)),
                min_frames=kwargs.get("min_frames"),
                max_frames=kwargs.get("max_frames"),
            )
            resize_kwargs = _apply_dynamic_video_max_pixels(len(indices) + pad_count, kwargs)
            return _sample_video(
                _smart_resize(video, **resize_kwargs),
                video_fps=video_fps,
                fps=float(kwargs.get("fps", 24.0)),
                frame_factor=kwargs.get("frame_factor"),
                frame_factor_remainder=int(kwargs.get("frame_factor_remainder", 0)),
                min_frames=kwargs.get("min_frames"),
                max_frames=kwargs.get("max_frames"),
            )
        if "video" not in video_input:
            raise ValueError("video dict must contain 'video' or 'frames'")
        video = _normalize_video_tensor(video_input["video"])
        video_fps = float(video_input.get("video_fps", kwargs.get("fps", 24.0)))
    else:
        source = video_input
        if isinstance(source, str):
            if data_dir and not source.startswith(("http://", "https://")):
                source = str(Path(data_dir) / source)
            if source.startswith(("http://", "https://")):
                import requests  # pylint: disable=import-outside-toplevel

                source = requests.get(source, timeout=120).content
        decoded = _load_video_with_torchcodec(source, **kwargs)
        if decoded is not None:
            return decoded[0]
        decoded = _load_video_with_imageio(source)
        video, video_fps = decoded

    indices, pad_count = _calculate_frame_indices(
        total_frames=int(video.shape[0]),
        video_fps=video_fps,
        fps=float(kwargs.get("fps", 24.0)),
        frame_factor=kwargs.get("frame_factor"),
        frame_factor_remainder=int(kwargs.get("frame_factor_remainder", 0)),
        min_frames=kwargs.get("min_frames"),
        max_frames=kwargs.get("max_frames"),
    )
    resize_kwargs = _apply_dynamic_video_max_pixels(len(indices) + pad_count, kwargs)
    return _sample_video(
        _smart_resize(video, **resize_kwargs),
        video_fps=video_fps,
        fps=float(kwargs.get("fps", 24.0)),
        frame_factor=kwargs.get("frame_factor"),
        frame_factor_remainder=int(kwargs.get("frame_factor_remainder", 0)),
        min_frames=kwargs.get("min_frames"),
        max_frames=kwargs.get("max_frames"),
    )


def _load_image(image_input: Any, *, data_dir: str | None = None) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if torch.is_tensor(image_input):
        frame = image_input
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0)
        if frame.dtype.is_floating_point:
            frame = (frame * (255.0 if frame.max() <= 1.0 else 1.0)).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(frame.cpu().numpy()).convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, str):
        source = image_input
        if data_dir and not source.startswith(("http://", "https://")):
            source = str(Path(data_dir) / source)
        if source.startswith(("http://", "https://")):
            import requests  # pylint: disable=import-outside-toplevel

            return Image.open(io.BytesIO(requests.get(source, timeout=120).content)).convert("RGB")
        return Image.open(source).convert("RGB")
    raise TypeError(f"Unsupported image input type {type(image_input).__name__}")


def _first_available(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)) and len(value) == 0:
            continue
        if isinstance(value, Mapping) and len(value) == 0:
            continue
        if torch.is_tensor(value) and value.numel() == 0:
            continue
        if isinstance(value, np.ndarray) and value.size == 0:
            continue
        return value
    return None


class WanVideoTransform:
    """Convert raw records into VeOmni-style Wan online DiT samples."""

    def __init__(
        self,
        *,
        task: str = "t2v",
        source_name: str | None = None,
        data_dir: str | None = None,
        fps: float = 24.0,
        min_frames: int | None = None,
        max_frames: int | None = 81,
        frame_factor: int | None = None,
        frame_factor_remainder: int = 0,
        use_first_video_frame_as_image: bool = True,
        prompt_keys: Sequence[str] = ("prompt", "text", "caption", "input", "inputs"),
        video_keys: Sequence[str] = ("video_bytes", "video", "video_path", "target_video", "target_video_path"),
        image_keys: Sequence[str] = ("image", "image_bytes", "image_path", "input_image", "reference_image", "first_frame"),
        **kwargs: Any,
    ) -> None:
        if task not in {"t2v", "i2v"}:
            raise ValueError("WanVideoTransform task must be either 't2v' or 'i2v'")
        self.task = task
        self.source_name = source_name
        self.data_dir = data_dir
        self.kwargs = {
            **kwargs,
            "fps": fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "frame_factor": frame_factor,
            "frame_factor_remainder": frame_factor_remainder,
        }
        self.use_first_video_frame_as_image = use_first_video_frame_as_image
        self.prompt_keys = tuple(prompt_keys)
        self.video_keys = tuple(video_keys)
        self.image_keys = tuple(image_keys)

    def _preprocess(self, example: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[Any], list[Any]]:
        if (
            self.source_name == "Tom-and-Jerry-VideoGeneration-Dataset"
            and "prompt" in example
            and "video_bytes" in example
        ):
            return str(example["prompt"]), {}, [], [example["video_bytes"]]

        prompt = _first_available(example, self.prompt_keys)
        video = _first_available(example, self.video_keys)
        image = _first_available(example, self.image_keys)
        if prompt is None:
            raise ValueError(f"Wan data requires one prompt key from {self.prompt_keys}")
        if video is None:
            raise ValueError(f"Wan data requires one video key from {self.video_keys}")
        images = [] if image is None else [image]
        return str(prompt), {}, images, [video]

    def __call__(self, example: Mapping[str, Any]) -> list[dict[str, Any]]:
        prompt, outputs, images, videos = self._preprocess(example)
        processed_videos = [
            _load_video(video, data_dir=self.data_dir, **self.kwargs)
            for video in videos
        ]
        processed_images = [
            _load_image(image, data_dir=self.data_dir)
            for image in images
        ]
        if self.task == "i2v" and not processed_images and self.use_first_video_frame_as_image:
            processed_images = [_load_image(processed_videos[0][0])]
        return [
            {
                "inputs": prompt,
                "outputs": outputs,
                "images": processed_images,
                "videos": processed_videos,
            }
        ]


def build_wan_video_transform(**kwargs: Any) -> WanVideoTransform:
    """YAML target for the Wan online video transform."""
    return WanVideoTransform(**kwargs)


def _build_dataloader_context(mesh_context: Any, data_config: Mapping[str, Any]) -> DataLoaderParallelContext | None:
    if mesh_context is None:
        return None
    return create_dataloader_parallel_context(
        mesh_context,
        data_index_cache=bool(data_config.get("data_index_cache", False)),
        shared_storage=not bool(data_config.get("no_shared_storage", False)),
    )


def build_wan_video_dataset(
    *,
    data_config: Mapping[str, Any],
    data_path: str | Sequence[str] | None = None,
    transform: Callable[[Any], Any] | None = None,
    dataloader_context: DataLoaderParallelContext | None = None,
    mesh_context: Any = None,
    training_config: Any = None,
    **kwargs: Any,
) -> Any:
    """Build an online HuggingFace video dataset and apply the Wan transform."""
    del kwargs
    dataset_config = dict(data_config)
    has_data_path = bool(data_path)
    if isinstance(data_path, str):
        has_data_path = bool(data_path.strip())
    if has_data_path and dataset_config.get("hf_dataset_name") is not None:
        logger.warning(
            "Wan dataset data_path is set; ignoring hf_dataset_name=%s and loading local data_path instead.",
            dataset_config["hf_dataset_name"],
        )
        dataset_config["hf_dataset_name"] = None
    training_seed = getattr(training_config, "seed", None)
    dataset_config["random_seed"] = 42 if training_seed is None else int(training_seed)
    if dataloader_context is None:
        dataloader_context = _build_dataloader_context(mesh_context, dataset_config)
    online_dataset = build_online_dataset(
        data_path=data_path,
        data_config=dataset_config,
        dataloader_context=dataloader_context,
    )
    return apply_llm_data_transform(online_dataset, transform, skip_invalid_samples=False)


def _normalize_model_sample(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        if len(sample) != 1 or not isinstance(sample[0], Mapping):
            raise ValueError("Wan fixed micro-batches expect each source item to produce one mapping sample")
        return sample[0]
    raise ValueError("Wan DataLoader samples must be mappings")


class DiTCollator:
    """List-preserving collator for per-sample DiT tensors and raw media."""

    def __call__(self, model_samples: Sequence[Any]) -> dict[str, Any]:
        if not model_samples:
            raise ValueError("DiT micro-batch must contain at least one sample")
        samples = [_normalize_model_sample(sample) for sample in model_samples]
        keys = []
        for sample in samples:
            for key in sample:
                if key not in keys:
                    keys.append(key)
        return {key: [sample.get(key) for sample in samples] for key in keys}


def build_dit_collate_fn() -> DiTCollator:
    """YAML target for DiT collation."""
    return DiTCollator()


class DiTBatch:
    """Read one collated DiT micro-batch from the DataLoader iterator."""

    def __init__(
        self,
        mesh_context: Any,
        device: Any,
        pp_shared_data: bool = False,
    ) -> None:
        del device
        self.mesh_context = mesh_context
        self.pp_shared_data = pp_shared_data
        self._logged = False

    def __call__(self, data_iterator: Any) -> dict[str, Any]:
        if self.pp_shared_data:
            raise NotImplementedError("Wan DiT get_batch does not yet support pipeline-parallel shared data")
        batch = next(data_iterator)
        if not self._logged:
            logger.debug(
                "Wan DiT get_batch active on dp_rank=%s, tp_rank=%s, cp_rank=%s",
                getattr(self.mesh_context, "dp_rank", 0),
                getattr(self.mesh_context, "tp_rank", 0),
                getattr(self.mesh_context, "cp_rank", 0),
            )
            self._logged = True
        return batch

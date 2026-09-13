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
"""DeepSeek-V4.1 native image-text Online dataset transformation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pylint: disable=forbidden-backend-import
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from hyper_parallel.data.constants import IGNORE_INDEX

TEXT = -1
IMAGE_START = 0
IMAGE = 1
IMAGE_NEWLINE = 2
IMAGE_END = 3

_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
_USER_TOKEN = "<｜User｜>"
_ASSISTANT_TOKEN = "<｜Assistant｜>"
_EOS_TOKEN = "<｜end▁of▁sentence｜>"
_THINKING_END_TOKEN = "</think>"
_IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def _llm_grid(height: int, width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    """Return the post-aligner grid dimensions for a resized image."""
    return (
        math.ceil((height // patch_size) / downsample_ratio),
        math.ceil((width // patch_size) / downsample_ratio),
    )


def _num_image_tokens(grid_height: int, grid_width: int) -> int:
    """Return V4.1's start/image/newline/end span length."""
    return grid_height * (grid_width + 1) + 2


def _solve_resize_ratio(
        height: int,
        width: int,
        patch_size: int,
        downsample_ratio: int,
        max_tokens: int,
) -> tuple[int, int]:
    """Find the released aspect-ratio-preserving resize under the token cap."""
    aspect_ratio = height / width
    max_width = math.sqrt((max_tokens - 2) / aspect_ratio + 0.25) - 0.5
    max_height = max_width * aspect_ratio
    cell_size = patch_size * downsample_ratio
    if max_width < 1.0:
        return (max_tokens - 2) // 2 * cell_size, cell_size
    if max_height < 1.0:
        return cell_size, (max_tokens - 3) * cell_size
    scale = min(
        math.floor(max_width) * cell_size / width,
        math.floor(max_height) * cell_size / height,
    )
    return math.floor(height * scale / patch_size) * patch_size, math.floor(width * scale / patch_size) * patch_size


class DeepseekV41ImageProcessor:
    """Pure-PyTorch image processor preserving V4.1 resize and patch semantics."""

    def __init__(
            self,
            *,
            patch_size: int,
            downsample_ratio: int,
            max_image_tokens: int,
            min_pixels: int,
            max_wh_ratio: float | None,
    ) -> None:
        """Store the released image preprocessing parameters."""
        if patch_size <= 0 or downsample_ratio <= 0 or max_image_tokens < 4:
            raise ValueError("invalid V4.1 image preprocessing configuration")
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio

    def _plan(self, image: Image.Image) -> tuple[int, int, int, int]:
        """Compute LLM and ViT grid dimensions for one source image."""
        width, height = image.size
        if self.max_wh_ratio is not None and width > height * self.max_wh_ratio:
            width = int(height * self.max_wh_ratio)
        if 0 < width * height < self.min_pixels:
            ratio = (self.min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / self.patch_size) * self.patch_size
        best_height = math.ceil(height / self.patch_size) * self.patch_size
        llm_height, llm_width = _llm_grid(
            best_height,
            best_width,
            self.patch_size,
            self.downsample_ratio,
        )
        if _num_image_tokens(llm_height, llm_width) > self.max_image_tokens:
            best_height, best_width = _solve_resize_ratio(
                height,
                width,
                self.patch_size,
                self.downsample_ratio,
                self.max_image_tokens,
            )
            llm_height, llm_width = _llm_grid(
                best_height,
                best_width,
                self.patch_size,
                self.downsample_ratio,
            )
        if _num_image_tokens(llm_height, llm_width) > self.max_image_tokens:
            raise ValueError("V4.1 image resize failed to meet max_image_tokens")
        return best_height, best_width, llm_height, llm_width

    def __call__(self, image_path: str) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """Load a local image and return BF16 patches plus ViT/LLM grids."""
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        best_height, best_width, llm_height, llm_width = self._plan(image)
        if self.max_wh_ratio is not None and image.width >= self.max_wh_ratio * image.height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
        pixel_values = torch.from_numpy(np.asarray(image, dtype="float32")).permute(2, 0, 1) / 255.0
        pixel_values = ((pixel_values - 0.5) / 0.5).to(torch.bfloat16)
        vit_height = best_height // self.patch_size
        vit_width = best_width // self.patch_size
        patches = pixel_values.reshape(3, vit_height, self.patch_size, vit_width, self.patch_size)
        patches = patches.permute(1, 3, 0, 2, 4).reshape(vit_height * vit_width, 3, self.patch_size, self.patch_size)
        return patches, (vit_height, vit_width), (llm_height, llm_width)


class DeepseekV41VLMTransform:
    """Encode one supervised V4.1 image-text conversation into a fixed sample.

    The intentionally narrow contract is the standard SFT form used by the
    downloaded data: one user message containing text/image blocks followed by
    one final assistant text answer. More elaborate tool and multi-turn
    rendering remains model-specific and should use the upstream encoder.
    """

    def __init__(
            self,
            config_path: str,
            *,
            max_seq_len: int = 4096,
            max_image_tokens: int | None = None,
    ) -> None:
        """Load only local V4.1 configuration/tokenizer assets.

        Args:
            config_path: Local V4.1 model directory containing config/tokenizer.
            max_seq_len: Expanded LLM sequence length, including image spans.
            max_image_tokens: Optional validation cap for one image's LLM
                token span. ``None`` retains the released model limit.
        """
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.config_path = Path(config_path).expanduser().resolve()
        config_file = self.config_path / "config.json"
        if not config_file.is_file():
            raise ValueError(f"DeepSeek-V4.1 config is missing: {config_file}")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if config.get("model_type") != "deepseek_v41":
            raise ValueError("config_path must point to a DeepSeek-V4.1 model directory")
        self.max_seq_len = max_seq_len
        text_config = config["text_config"]
        compression_ratios = [
            int(ratio) for ratio in text_config.get("compress_ratios", []) if int(ratio) > 1
        ]
        self.compression_alignment = math.lcm(*compression_ratios) if compression_ratios else 1
        if self.max_seq_len % self.compression_alignment:
            raise ValueError(
                "max_seq_len must align with V4.1 compressed-attention groups: "
                f"{self.max_seq_len} is not divisible by {self.compression_alignment}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_path, local_files_only=True, trust_remote_code=False)
        self.image_token_id = int(config["image_token_id"])
        placeholder_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_PLACEHOLDER)
        if placeholder_id is not None and placeholder_id != self.tokenizer.unk_token_id:
            if placeholder_id != self.image_token_id:
                raise ValueError("tokenizer image placeholder does not match config.image_token_id")
        vision_config = config["vision_config"]
        released_max_image_tokens = int(vision_config["max_image_tokens"])
        resolved_max_image_tokens = (
            released_max_image_tokens if max_image_tokens is None else int(max_image_tokens)
        )
        if not 4 <= resolved_max_image_tokens <= released_max_image_tokens:
            raise ValueError(
                "max_image_tokens must be between 4 and the released limit "
                f"{released_max_image_tokens}, got {resolved_max_image_tokens}"
            )
        self.image_processor = DeepseekV41ImageProcessor(
            patch_size=int(vision_config["patch_size"]),
            downsample_ratio=int(vision_config["downsample_ratio"]),
            max_image_tokens=resolved_max_image_tokens,
            min_pixels=int(vision_config["min_pixels"]),
            max_wh_ratio=vision_config["max_wh_ratio"],
        )
        self.pad_token_id = int(config["pad_token_id"])

    @staticmethod
    def _message_content(message: dict[str, Any]) -> tuple[str, list[str]]:
        """Render user content blocks and return image paths in placeholder order."""
        content = message.get("content")
        if isinstance(content, str):
            return content, []
        if not isinstance(content, list):
            raise ValueError("V4.1 user content must be a string or a list of content blocks")
        text_parts = []
        image_paths = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("V4.1 content blocks must be mappings")
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type in ("image", "image_url"):
                if block_type == "image_url":
                    image_url = block.get("image_url")
                    image_path = image_url if isinstance(image_url, str) else (image_url or {}).get("url")
                else:
                    image_path = block.get("url") or block.get("image") or block.get("source")
                if not isinstance(image_path, str) or not image_path:
                    raise ValueError("V4.1 image blocks must carry a local URL")
                text_parts.append(_IMAGE_PLACEHOLDER)
                image_paths.append(image_path)
            else:
                raise ValueError(f"unsupported V4.1 SFT content block type: {block_type!r}")
        return "\n\n".join(text_parts), image_paths

    def _render_sft(self, messages: Any) -> tuple[str, str, list[str]]:
        """Return the exact chat-mode prefix/completion for one two-turn SFT record."""
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("V4.1 Online SFT requires exactly one user message and one assistant answer")
        user_message, assistant_message = messages
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            raise ValueError("V4.1 Online SFT messages must be ordered user then assistant")
        user_text, image_paths = self._message_content(user_message)
        answer = assistant_message.get("content")
        if not isinstance(answer, str) or not answer:
            raise ValueError("V4.1 Online SFT assistant content must be a non-empty string")
        prefix = _BOS_TOKEN + _USER_TOKEN + user_text + _ASSISTANT_TOKEN + _THINKING_END_TOKEN
        return prefix, answer + _EOS_TOKEN, image_paths

    @staticmethod
    def _image_token_types(grid_height: int, grid_width: int) -> list[int]:
        """Create one start/image/newline/end V4.1 image span."""
        return [IMAGE_START] + ([IMAGE] * grid_width + [IMAGE_NEWLINE]) * grid_height + [IMAGE_END]

    def __call__(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Transform one canonical JSONL record into V4.1 model inputs."""
        prefix, completion, image_paths = self._render_sft(record["messages"])
        prefix_ids = self.tokenizer.encode(prefix)
        full_ids = self.tokenizer.encode(prefix + completion)
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("V4.1 tokenizer prefix does not align with the full supervised prompt")
        raw_labels = [IGNORE_INDEX] * len(prefix_ids) + full_ids[len(prefix_ids):]
        expanded_ids: list[int] = []
        expanded_labels: list[int] = []
        token_types: list[int] = []
        pixel_values = []
        patch_offsets = [0]
        vit_grids = []
        llm_grids = []
        image_token_starts = []
        image_iterator = iter(image_paths)
        for token_id, label in zip(full_ids, raw_labels):
            if token_id != self.image_token_id:
                if len(expanded_ids) >= self.max_seq_len:
                    break
                expanded_ids.append(token_id)
                expanded_labels.append(label)
                token_types.append(TEXT)
                continue
            try:
                image_path = next(image_iterator)
            except StopIteration as exc:
                raise ValueError("tokenizer emitted more image placeholders than the record contains") from exc
            patches, vit_grid, llm_grid = self.image_processor(image_path)
            span_types = self._image_token_types(*llm_grid)
            if len(expanded_ids) + len(span_types) > self.max_seq_len:
                break
            image_token_starts.append(len(expanded_ids))
            expanded_ids.extend([self.image_token_id] * len(span_types))
            expanded_labels.extend([IGNORE_INDEX] * len(span_types))
            token_types.extend(span_types)
            pixel_values.append(patches)
            patch_offsets.append(patch_offsets[-1] + patches.shape[0])
            vit_grids.append(vit_grid)
            llm_grids.append(llm_grid)
        if next(image_iterator, None) is not None:
            raise ValueError("record contains more images than the tokenizer image placeholders")
        if not any(label != IGNORE_INDEX for label in expanded_labels):
            raise ValueError("max_seq_len truncation removed the complete assistant target")
        sequence_length = len(expanded_ids)
        alignment_padding = (-sequence_length) % self.compression_alignment
        if sequence_length + alignment_padding > self.max_seq_len:
            raise ValueError(
                "max_seq_len leaves no room for V4.1 compressed-attention alignment padding"
            )
        if alignment_padding:
            expanded_ids.extend([self.pad_token_id] * alignment_padding)
            expanded_labels.extend([IGNORE_INDEX] * alignment_padding)
            token_types.extend([TEXT] * alignment_padding)
            sequence_length += alignment_padding
        padding = self.max_seq_len - sequence_length
        expanded_ids.extend([self.pad_token_id] * padding)
        expanded_labels.extend([IGNORE_INDEX] * padding)
        token_types.extend([TEXT] * padding)
        attention_mask = [1] * sequence_length + [0] * padding
        patch_tensor = (
            torch.cat(pixel_values, dim=0)
            if pixel_values
            else torch.empty(
                (0, 3, self.image_processor.patch_size, self.image_processor.patch_size),
                dtype=torch.bfloat16,
            )
        )
        return {
            "input_ids": torch.tensor(expanded_ids, dtype=torch.long),
            "labels": torch.tensor(expanded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_types": torch.tensor(token_types, dtype=torch.long),
            "pixel_values": patch_tensor,
            "image_patch_offsets": torch.tensor(patch_offsets, dtype=torch.long),
            "image_vit_grid_hw": torch.tensor(vit_grids, dtype=torch.long).reshape(-1, 2),
            "image_llm_grid_hw": torch.tensor(llm_grids, dtype=torch.long).reshape(-1, 2),
            "image_token_starts": torch.tensor(image_token_starts, dtype=torch.long),
        }


def build_deepseek_v41_vlm_data_transform(
        *,
        config_path: str,
        max_seq_len: int = 4096,
        max_image_tokens: int | None = None,
        processor: Any = None,
        **transform_options: Any,
) -> DeepseekV41VLMTransform:
    """Build the native V4.1 image-text SFT transform.

    Args:
        config_path: Local DeepSeek-V4.1 model directory.
        max_seq_len: Expanded LLM token length including image spans.
        max_image_tokens: Optional validation cap for one image span.
        processor: Ignored; V4.1 does not have a Transformers AutoProcessor.
        **transform_options: Reserved model-specific transform options.

    Returns:
        A transform that accepts canonical OpenAI-style V4.1 records.
    """
    del processor, transform_options
    return DeepseekV41VLMTransform(
        config_path,
        max_seq_len=max_seq_len,
        max_image_tokens=max_image_tokens,
    )


__all__ = [
    "DeepseekV41VLMTransform",
    "build_deepseek_v41_vlm_data_transform",
]

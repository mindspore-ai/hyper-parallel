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
"""K3 image pretraining samples in final decoder-token coordinates.

The released processor emits one media-pad token per image. Expansion belongs
here, before batching or eventual CP sharding; model forward only replaces the
expanded positions. Transform labels remain unshifted; collation shifts global
targets for CP/packing, while the default CP1 causal loss shifts once.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# The project is Torch-only; the legacy platform lint does not apply.
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.omni.omni_transform import OmniDataTransform
from hyper_parallel.models.kimi_k3.adapter.data.video import preprocess_rendered_medias


def validate_media_grid(grid_thws: torch.Tensor, *, max_frames: int = 4) -> torch.Tensor:
    """Validate sampled-media grids and return temporally pooled token counts.

    Args:
        grid_thws: Integer [media,3] rows (frames,height,width).
        max_frames: Maximum frames in one temporal-pooling group.

    Returns:
        One spatial feature count height*width/4 per media item.
    """
    if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
        raise ValueError("grid_thws must have shape [media_count,3]")
    if grid_thws.dtype not in (torch.int32, torch.int64):
        raise ValueError("grid_thws must contain integer t,h,w values")
    if bool((grid_thws <= 0).any()) or bool((grid_thws[:, 0] > max_frames).any()):
        raise ValueError("K3 media grids require positive dimensions and at most max_frames frames")
    if bool((grid_thws[:, 1:] % 2 != 0).any()) or bool((grid_thws[:, 1:] > 512).any()):
        raise ValueError("K3 media spatial grids must be even and at most 512")
    return grid_thws[:, 1] * grid_thws[:, 2] // 4


def validate_image_grid(grid_thws: torch.Tensor) -> torch.Tensor:
    """Validate the legacy still-image contract.

    Args:
        grid_thws: Integer still-image grids, each with one frame.

    Returns:
        One decoder feature count per image.
    """
    return validate_media_grid(grid_thws, max_frames=1)


def patchify_sampled_video(frames: torch.Tensor, *, patch_size: int = 14,
                           max_frames: int = 4) -> dict[str, torch.Tensor]:
    """Normalize pre-sampled RGB frames and use the official frame-major patch order.

    This boundary does not decode video files, choose timestamps, resize frames,
    or invent a conversation template. Those decisions belong to the dataset.

    Args:
        frames: Uint8 RGB [time,height,width,3] with 1..max_frames frames.
        patch_size: Spatial patch edge; image dimensions must divide 2*patch_size.
        max_frames: Maximum temporal group supported by the vision position table.

    Returns:
        FP32 pixel_values and one grid_thws row, ready for encoded sample input.
    """
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("sampled video must be uint8 RGB [time,height,width,3]")
    time, height, width, _ = frames.shape
    if patch_size < 1 or height % (2 * patch_size) or width % (2 * patch_size):
        raise ValueError("sampled frame sizes must divide the 2x2 spatial patch merger")
    grid = torch.tensor([[time, height // patch_size, width // patch_size]], device=frames.device)
    validate_media_grid(grid, max_frames=max_frames)
    values = frames.float().div(255).sub(0.5).mul(2)
    values = values.reshape(time, height // patch_size, patch_size, width // patch_size, patch_size, 3)
    return {"pixel_values": values.permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, patch_size, patch_size),
            "grid_thws": grid}


def expand_media_tokens(
        input_ids: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        media_token_id: int = 163605,
        labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Expand each raw image placeholder without changing image order.

    Args:
        input_ids: Unpadded, unbatched raw processor token IDs.
        grid_thws: Image grids in processor/media encounter order.
        media_token_id: Token replaced by the model's visual features.
        labels: Unshifted labels on the raw token sequence, or ``None`` for IDs.

    Returns:
        Expanded IDs and labels, a boolean visual-feature mask, media spans
        ``[media_count, 3]`` with ``(sample_index=0, start, exclusive_end)``,
        and patch offsets. Coordinates refer to the final global sequence.
        Media spans identify feature slots, excluding surrounding prompt tokens.

    Raises:
        ValueError: If shapes or the placeholder-to-image correspondence differ.
    """
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("input_ids must be a nonempty unbatched sequence")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain integer token IDs")
    labels = input_ids.clone() if labels is None else labels
    if labels.shape != input_ids.shape:
        raise ValueError("unshifted labels must have the same shape as input_ids")
    counts = validate_media_grid(grid_thws)
    positions = (input_ids == media_token_id).nonzero().flatten().tolist()
    if len(positions) != len(counts):
        raise ValueError("K3 processor must emit exactly one media_pad placeholder per grid_thws row")

    repeats = torch.ones_like(input_ids)
    if positions:
        repeats[positions] = counts.to(device=repeats.device, dtype=repeats.dtype)
    expanded_ids = input_ids.repeat_interleave(repeats)
    expanded_labels = labels.repeat_interleave(repeats)
    media_mask = expanded_ids == media_token_id
    expanded_labels[media_mask] = IGNORE_INDEX
    spans = []
    added_tokens = 0
    for raw_position, count in zip(positions, counts.tolist()):
        start = raw_position + added_tokens
        spans.append((0, start, start + count))
        added_tokens += count - 1
    patch_counts = grid_thws.prod(dim=-1)
    return {
        "input_ids": expanded_ids,
        "labels": expanded_labels,
        "media_token_mask": media_mask,
        "media_spans": input_ids.new_tensor(spans).reshape(-1, 3),
        "media_patch_offsets": torch.cat((patch_counts.new_zeros(1), patch_counts.cumsum(0))),
    }


def _single_sequence(value: Any, name: str) -> torch.Tensor:
    """Read a processor's sole token sequence without flattening real batches."""
    tensor = torch.as_tensor(value)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"processor {name} must be one integer sequence, [S] or [1, S]")
    return tensor.to(torch.long)


class KimiK3OmniTransform(OmniDataTransform):  # pylint: disable=abstract-method
    """Encode still images and text with a deliberate pretraining label policy.

    ``all_text`` supervises all rendered text, including conversation scaffold,
    except complete media prompts. ``final_assistant`` masks the prompt and
    supervises only the final assistant suffix. Neither policy shifts labels.
    Oversized records fail instead of truncating an image or its supervision.
    Rendered input_ids support sampled media_frames or ordered raw medias. Local
    video files use explicit sampling controls and groups of up to four frames;
    the released message processor itself only supports still images.
    """

    def __init__(
            self,
            *,
            processor: Any,
            max_seq_len: int = 8192,
            label_policy: str | None = None,
            training_mode: str | None = None,
            pixel_dtype: str = "float32",
            patch_size: int = 14,
            media_token_id: int = 163605,
            media_begin_id: int = 163602,
            media_end_id: int = 163604,
            max_video_frames: int = 128,
    ) -> None:
        """Store the processor and explicit image/text training contract."""
        if training_mode is None:
            training_mode = "sft" if label_policy == "final_assistant" else "pretrain"
        expected_policy = "all_text" if training_mode == "pretrain" else "final_assistant"
        label_policy = expected_policy if label_policy is None else label_policy
        super().__init__(max_seq_len=max_seq_len, processor=processor, training_mode=training_mode)
        if label_policy != expected_policy:
            raise ValueError("K3 training_mode and label_policy must select the same supervision contract")
        if label_policy not in ("all_text", "final_assistant"):
            raise ValueError("label_policy must be 'all_text' or 'final_assistant'")
        if pixel_dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("pixel_dtype must be float32, bfloat16, or float16")
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if isinstance(max_video_frames, bool) or not isinstance(max_video_frames, int) or max_video_frames < 1:
            raise ValueError("max_video_frames must be a positive integer")
        self.max_video_frames = max_video_frames
        self.label_policy = label_policy
        self.pixel_dtype = getattr(torch, pixel_dtype)
        self.patch_size = patch_size
        self.media_token_id = media_token_id
        self.media_begin_id = media_begin_id
        self.media_end_id = media_end_id

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Keep rendered-token records or valid conversations for lazy encoding.

        Args:
            sample: Encoded record or conversation awaiting transformation.
        """
        if "input_ids" in sample:
            return True
        return super().is_valid_sample(sample)

    def _prepare_k3_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve inherited Omni paths and normalize explicit image placeholders."""
        if sample.get("videos") or sample.get("audios"):
            raise ValueError("K3 messages support images/text; videos require rendered input_ids with medias")
        messages = self.prepare_messages(sample)
        images = list(sample.get("images") or [])
        source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
        source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else None
        consumed = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                if "<video>" in content or "<audio>" in content:
                    raise ValueError("K3 messages support images/text; videos require rendered input_ids with medias")
                if "<image>" not in content:
                    continue
                content, consumed = self._expand_image_placeholders(content, images, consumed, source_directory)
                message["content"] = content
            if isinstance(content, list):
                self._validate_content(message, content)
        if consumed != len(images):
            raise ValueError("Top-level images must each correspond to an explicit <image> placeholder")
        return messages

    def _expand_image_placeholders(self, content: str, images: list[Any], consumed: int,
                                   source_directory: str | None) -> tuple[list[dict[str, Any]], int]:
        """Expand explicit markers without changing top-level image order."""
        blocks = []
        for part_index, part in enumerate(content.split("<image>")):
            if part_index:
                if consumed >= len(images):
                    raise ValueError("Each <image> placeholder requires a top-level images entry")
                image = images[consumed]
                if isinstance(image, str) and source_directory is not None:
                    image = self._resolve_media_path(image, source_directory)
                blocks.append({"type": "image", "image": image})
                consumed += 1
            if part:
                blocks.append({"type": "text", "text": part})
        return blocks, consumed

    @staticmethod
    def _validate_content(message: Mapping[str, Any], content: list[Any]) -> None:
        """Reject media the released processor would silently ignore."""
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type not in ("text", "image", "image_url"):
                raise ValueError(f"Unsupported K3 message content type: {block_type!r}; images and text only")
            if block_type in ("image", "image_url"):
                if message.get("role") != "user":
                    raise ValueError("The released K3 processor only extracts images from user messages")
                if block.get(block_type) is None:
                    raise ValueError(f"K3 {block_type} content must carry a {block_type} value")
                if block_type == "image_url" and isinstance(block[block_type], dict):
                    image_url = block[block_type].get("url")
                    if not isinstance(image_url, str):
                        raise ValueError("K3 image_url mappings must contain a string url")
                    block[block_type] = image_url

    def _labels(self, input_ids: torch.Tensor, messages: list[dict[str, Any]]) -> torch.Tensor:
        """Mask complete media prompts and optionally the final-assistant prefix."""
        labels = input_ids.clone()
        if self.label_policy == "final_assistant":
            self._mask_assistant_prompt(input_ids, labels, messages)
        start = None
        placeholder_count = 0
        for index, token_id in enumerate(input_ids.tolist()):
            if token_id == self.media_begin_id:
                if start is not None:
                    raise ValueError("Nested K3 media prompts are invalid")
                start = index
                placeholder_count = 0
            if token_id == self.media_token_id:
                if start is None:
                    raise ValueError("K3 media_pad must occur inside a complete media prompt")
                placeholder_count += 1
            if token_id == self.media_end_id:
                if start is None:
                    raise ValueError("K3 media_end lacks a matching media_begin")
                if placeholder_count != 1:
                    raise ValueError("Each raw K3 media prompt must contain exactly one media_pad")
                labels[start:index + 1] = IGNORE_INDEX
                start = None
        if start is not None:
            raise ValueError("K3 media prompt is incomplete; truncating media is unsupported")
        # The first token has no predecessor. Mask it so generic token-weighted
        # accumulation counts the same targets as the model's single shift.
        if labels.numel():
            labels[0] = IGNORE_INDEX
        return labels

    def _mask_assistant_prompt(self, input_ids: torch.Tensor, labels: torch.Tensor,
                               messages: list[dict[str, Any]]) -> None:
        """Apply the final-assistant prefix only after checking its exact token identity."""
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("final_assistant requires a final assistant message and a preceding prompt")
        prompt = self.processor(messages=messages[:-1], return_tensors="pt", add_generation_prompt=True)
        prompt_ids = _single_sequence(prompt["input_ids"], "input_ids")
        if len(prompt_ids) >= len(input_ids) or not input_ids[:len(prompt_ids)].equal(prompt_ids):
            raise ValueError("K3 assistant prompt must be an exact prefix of the complete conversation")
        labels[:len(prompt_ids)] = IGNORE_INDEX

    def _preprocessed_media(self, sample: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Resolve raw media, pre-sampled frames or an explicit patch payload."""
        if "medias" in sample:
            if any(field in sample for field in ("media_frames", "pixel_values", "grid_thws")):
                raise ValueError("Choose medias, media_frames, or an explicit pixel/grid payload")
            source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
            source_directory = os.path.dirname(os.path.abspath(source_path)) if source_path else None
            return preprocess_rendered_medias(
                sample["medias"], processor=self.processor, source_directory=source_directory,
                max_video_frames=self.max_video_frames, patch_size=self.patch_size,
            )
        if "media_frames" not in sample:
            return torch.as_tensor(sample["pixel_values"]), torch.as_tensor(sample["grid_thws"]), None
        if "pixel_values" in sample or "grid_thws" in sample:
            raise ValueError("Choose media_frames or an explicit pixel/grid payload")
        clips = [patchify_sampled_video(torch.as_tensor(frames), patch_size=self.patch_size)
                 for frames in sample["media_frames"]]
        grids = (torch.cat([clip["grid_thws"] for clip in clips]) if clips
                 else torch.empty(0, 3, dtype=torch.long))
        pixels = (torch.cat([clip["pixel_values"] for clip in clips]) if clips
                  else torch.empty(0, 3, self.patch_size, self.patch_size))
        return pixels, grids, None

    def _encode_preprocessed(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Read rendered tokens and preserve sampled-video provenance outside model math."""
        input_ids = _single_sequence(sample["input_ids"], "input_ids")
        pixels, grids, metadata = self._preprocessed_media(sample)
        validate_media_grid(grids)
        if self.label_policy == "final_assistant" and "labels" not in sample:
            raise ValueError("pre-rendered final_assistant samples require explicit raw labels")
        labels = (_single_sequence(sample["labels"], "labels") if "labels" in sample
                  else self._labels(input_ids, []))
        labels = labels.clone()
        labels[0] = IGNORE_INDEX
        result = expand_media_tokens(input_ids, grids, media_token_id=self.media_token_id, labels=labels)
        length = result["input_ids"].numel()
        if length > self.max_seq_len:
            raise ValueError("Expanded sampled-media record exceeds max_seq_len")
        if pixels.shape != (int(grids.prod(-1).sum()), 3, self.patch_size, self.patch_size):
            raise ValueError("sampled-media pixel patches must exactly match grid_thws")
        if not bool(result["labels"][1:].ne(IGNORE_INDEX).any()):
            raise ValueError("sampled-media record requires a valid language target")
        result.update(attention_mask=torch.ones_like(result["input_ids"]),
                      position_ids=torch.arange(length), grid_thws=grids.to(torch.long),
                      pixel_values=pixels.to(self.pixel_dtype))
        if metadata is not None:
            result["media_metadata"] = metadata
        return result

    @staticmethod
    def _validate_message_media_fields(sample: Mapping[str, Any]) -> None:
        """Reject pre-rendered media payloads that the conversation processor cannot consume."""
        fields = {"medias", "media_frames", "pixel_values", "grid_thws", "media_spans",
                  "media_patch_offsets", "media_token_mask", "media_metadata"}.intersection(sample)
        if fields:
            raise ValueError(
                f"K3 pre-rendered media fields {sorted(fields)} require input_ids; "
                "messages accept image content or top-level images, not pre-rendered media payloads"
            )

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Encode one source record, preserving image order and unshifted labels.

        Args:
            sample: Source conversation or text record accepted by the transform,
                with image content in processor encounter order. Prepacked records
                are not accepted. Rendered input_ids may accompany normalized patches
                or uint8 media_frames. Ordered ``medias`` descriptors also accept
                local video files with explicit sampling controls. Each group of up
                to four frames requires its own rendered media prompt; no video
                conversation template is synthesized.

        Returns:
            One unpadded example containing expanded input_ids, unshifted labels,
            attention_mask, position_ids, media spans and masks, image grids, patch
            offsets and flattened pixel_values [sum(t*h*w), 3, patch_size, patch_size].
            Image prompt/feature tokens are excluded from language-model targets.

        Raises:
            ValueError: If media cannot be aligned, the expanded sequence exceeds
                the budget, or no valid next-token target remains.
        """
        if "cu_seq_lens" in sample or "packing_length" in sample:
            raise ValueError("K3 transforms accept one document at a time; enable packing on the K3 collator")
        if "input_ids" in sample:
            return self._encode_preprocessed(sample)
        self._validate_message_media_fields(sample)
        messages = self._prepare_k3_messages(sample)
        encoded = dict(self.processor(messages=messages, return_tensors="pt", add_generation_prompt=False))
        input_ids = _single_sequence(encoded["input_ids"], "input_ids")
        if "attention_mask" in encoded:
            attention_mask = _single_sequence(encoded["attention_mask"], "attention_mask")
            if attention_mask.shape != input_ids.shape or not bool((attention_mask == 1).all()):
                raise ValueError("K3 processor must return unpadded tokens; padding belongs to the collator")
        grid = torch.as_tensor(encoded.get("grid_thws", torch.empty(0, 3, dtype=torch.long)))
        validate_media_grid(grid)
        media_count = sum(
            isinstance(block, dict) and block.get("type") in ("image", "image_url")
            for message in messages if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        if len(grid) != media_count:
            raise ValueError("The K3 processor must preserve every input image in grid_thws")
        result = expand_media_tokens(
            input_ids, grid, media_token_id=self.media_token_id, labels=self._labels(input_ids, messages),
        )
        length = len(result["input_ids"])
        if length > self.max_seq_len:
            raise ValueError(
                f"Expanded K3 sequence length {length} exceeds max_seq_len={self.max_seq_len}; "
                "reduce the media budget or split the source document without cutting media"
            )
        pixels = torch.as_tensor(encoded.get(
            "pixel_values", torch.empty(0, 3, self.patch_size, self.patch_size),
        ))
        expected_rows = int(grid.prod(dim=-1).sum())
        if pixels.shape != (expected_rows, 3, self.patch_size, self.patch_size):
            raise ValueError("pixel_values must be [sum(t*h*w), 3, patch_size, patch_size] in grid order")
        result.update({
            "attention_mask": torch.ones_like(result["input_ids"]),
            "position_ids": torch.arange(length, dtype=torch.long),
            "grid_thws": grid.to(torch.long),
            "pixel_values": pixels.to(self.pixel_dtype),
        })
        if not bool((result["labels"][1:] != IGNORE_INDEX).any()):
            raise ValueError("K3 sample must contain at least one valid next-token target after media masking")
        return result


def build_kimi_k3_omni_transform(*, processor: Any, **options: Any) -> KimiK3OmniTransform:
    """Build the model-owned image/text Omni transform from recipe options.

    Args:
        processor: Callable K3 processor producing tokens, flattened patches and
            grid_thws from structured messages.
        **options: KimiK3OmniTransform settings, including sequence budget,
            patch geometry, pixel dtype and supervision policy.

    Returns:
        Transform that expands image placeholders and constructs unshifted labels.
    """
    return KimiK3OmniTransform(processor=processor, **options)

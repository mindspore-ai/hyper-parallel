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
"""Kimi-K2.x multimodal Omni transform (native ``kimi_k25`` protocol).

The native transformers processor renders one image as
``<|media_begin|>image<|media_content|>`` + ``N x <|media_pad|>`` +
``<|media_end|>`` (``N = grid_h*grid_w/4``) and outputs
``pixel_values`` / ``image_grid_thw`` (same names as Qwen3-VL). Unlike
Qwen3-VL it does **not** emit ``mm_token_type_ids`` by default, so this
transform rebuilds the modal-run map itself by scanning the ``<|media_begin|>``
.. ``<|media_end|>`` spans (falling back to bare ``<|media_pad|>`` runs), which
keeps the run<->``image_grid_thw`` row alignment used by truncation bookkeeping
in the shared VLM pipeline. Image positions (inside spans) are excluded from
the loss with ``IGNORE_INDEX``; only the final assistant text is the target.

This module hosts the Omni data lifecycle for Kimi on top of the shared
:class:`~hyper_parallel.data.omni.omni_transform.OmniDataTransform` contract:

* :class:`KimiOmniTransform` encodes eagerly (``encode_sample``), which is the
  path used by the packing loader because the candidate cost can be derived
  from the encoded ``input_ids``.
* :class:`KimiPackingOmniTransform` implements the deferred hooks
  (``preencode_sample`` + ``postencode_sample``) for pipelines that must select
  candidates before the final per-sample encoding. ``preencode_sample`` renders
  the conversation once and reports its exact ``packing_length``;
  ``postencode_sample`` then applies length truncation. Sequence padding is left
  to the packing collator, because packing concatenates whole samples.

Both wrappers delegate the actual rendering to :class:`KimiVLMChatTransform`,
which stays importable for callers that drive it directly.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.omni.omni_transform import OmniDataTransform

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")

# Accepted ``truncate_mode`` values: ``"proportional"`` splits the ``max_seq_len``
# budget between prompt and response, ``"head"`` is the historical pure-head cut.
_TRUNCATE_MODES = ("proportional", "head")


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Split a ``cutoff_len`` budget between a source and a target region.

    Ported verbatim from MindSpeed-MM's ``infer_seqlen``
    (``mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py``), which resolves
    the same problem for supervised fine-tuning data: a pure head cut throws the
    supervised region away whenever the prompt alone fills the window. The three
    branches are:

    * ``target_len * 2 < cutoff_len`` -- the response is small relative to the
      budget, so it is allowed everything it asks for (``max_target_len =
      cutoff_len``) and the prompt keeps the remainder.
    * ``source_len * 2 < cutoff_len`` -- the prompt is small, so the response gets
      the whole remainder of the budget (``cutoff_len - source_len``).
    * otherwise both sides are large and the budget is split proportionally to
      their sizes, ``int(cutoff_len * target_len / (source_len + target_len))``
      (truncated, not rounded).

    In every branch the two results are then clamped to their own lengths, so a
    region shorter than its share is never padded and the sum never exceeds
    ``cutoff_len``.

    Args:
        source_len: Length of the source (prompt/image) region.
        target_len: Length of the target (supervised) region.
        cutoff_len: Total budget the two regions must fit into.

    Returns:
        The ``(new_source_len, new_target_len)`` pair, with
        ``new_source_len + new_target_len <= cutoff_len``.
    """
    if target_len * 2 < cutoff_len:  # the response is small: keep all of it
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # the prompt is small: target takes the rest
        max_target_len = cutoff_len - source_len
    else:  # both are large: split the budget proportionally
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


class KimiVLMChatTransform:
    """Encode one Kimi multimodal conversation into one padded model sample.

    A sample longer than ``max_seq_len`` is truncated with the policy selected by
    ``truncate_mode``: ``"proportional"`` (the default) reserves a share of the
    window for the supervised response, ``"head"`` keeps the leading tokens only.
    Images a truncation would cut are dropped either way.
    """

    def __init__(self, processor: Any, *, max_seq_len: int = 1024,
                 pixel_dtype: str = "bfloat16",
                 pad_granularity: int | None = None,
                 image_max_pixels: int | None = None,
                 truncate_mode: str = "proportional") -> None:
        """Store the processor, the target length, and the media-id remap.

        Args:
            processor: Native ``kimi_k25`` processor used to render conversations.
            max_seq_len: Hard cap; longer samples are truncated (dropping images
                a truncation would cut) and shorter ones padded.
            pixel_dtype: Dtype name for ``pixel_values``.
            pad_granularity: When set, pad each sample only up to the next
                multiple of this value (instead of always ``max_seq_len``), so
                the collator can pad the batch to its own longest sample. This
                removes most pad-only compute; choose a multiple of the
                parallel sequence alignment (``cp_size * tp_size``).
            image_max_pixels: When set, downscale each media image to at most
                this many pixels. This is an *area* (pixel-count) budget with
                MindSpeed-MM's ``_preprocess_image`` semantics: both sides are
                scaled by ``sqrt(image_max_pixels / (width * height))`` and
                truncated with ``int()``, so the aspect ratio is preserved and
                an image already within budget is never upscaled. ``None`` keeps
                the processor's own budget, which for this checkpoint is
                ``in_patch_limit=16384`` / ``patch_limit_on_one_side=512`` in
                *patch* units -- so a 1024x1024 image is accepted unresized at
                74x74 patches (1369 media tokens after the 2x2 merge) instead of
                the 361 a 512x512 area budget yields (the processor rounds 512
                up to its 28-pixel grid, giving 38x38 patches). Set this to
                match a pipeline that resizes by pixels (MindSpeed's kimi_k3
                config uses ``image_max_pixels: 262144``, i.e. 512x512),
                otherwise the two feed different image-token budgets and are not
                comparable: unresized, ten 1024x1024 images occupy 13690 of the
                12288-slot window and truncate the assistant turn away entirely,
                while at ``262144`` they take 3610.
            truncate_mode: How a sample longer than ``max_seq_len`` is cut, either
                ``"proportional"`` (default) or ``"head"``. The supervised region
                sits at the *end* of the stream, so a pure head cut keeps the
                prompt and images and silently discards the whole assistant turn
                whenever they fill the window -- the sample then reaches the loss
                with no supervised token at all and the step trains on nothing.
                ``"proportional"`` mirrors MindSpeed-MM's ``infer_seqlen``: the
                ``max_seq_len`` budget is split between the prompt (everything
                before the first supervised token) and the response (that token
                to the end of the stream), so the response survives a truncation.
                ``"head"`` is the historical behaviour, kept for configs that
                were tuned against it.

        Raises:
            ValueError: If ``pad_granularity`` is not a positive integer,
                ``image_max_pixels`` is not a positive integer or ``None``, or
                ``truncate_mode`` is not one of ``"proportional"`` / ``"head"``.
        """
        if pad_granularity is not None and (
                isinstance(pad_granularity, bool) or not isinstance(pad_granularity, int)
                or pad_granularity <= 0):
            raise ValueError(
                f"pad_granularity must be a positive integer or None, got {pad_granularity!r}"
            )
        if image_max_pixels is not None and (
                isinstance(image_max_pixels, bool) or not isinstance(image_max_pixels, int)
                or image_max_pixels <= 0):
            raise ValueError(
                f"image_max_pixels must be a positive integer or None, got {image_max_pixels!r}"
            )
        if not isinstance(truncate_mode, str) or truncate_mode not in _TRUNCATE_MODES:
            raise ValueError(
                f"truncate_mode must be one of {_TRUNCATE_MODES}, got {truncate_mode!r}"
            )
        self.image_max_pixels = image_max_pixels
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.pad_granularity = pad_granularity
        self.truncate_mode = truncate_mode
        self.pixel_dtype = getattr(torch, pixel_dtype)
        # Canonical (hub tokenizer_config / model config) media token ids.
        self._canonical = {
            "begin": 163602, "content": 163603, "pad": 163605, "end": 163604,
        }
        self._remap: dict[int, int] | None = None

    # -- media token id helpers ----------------------------------------------

    def _media_ids(self) -> dict[str, int]:
        """Return the canonical media ids after any native-id renumbering.

        transformers 5.x ``TokenizersBackend`` reassigns *sparse* added-token
        ids contiguously: for this repo the media tokens (hub ids 163602-163605)
        come back as 163599-163602 because the decoder table has gaps at
        163589/163592/163600. The native model locates image placeholders with
        ``config.image_token_id == 163605``, so the token stream must be
        renumbered back to the canonical ids (also the ones a real checkpoint
        vocabulary uses).
        """
        if self._remap is not None:
            return self._canonical
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Kimi processor must expose a tokenizer")
        remap: dict[int, int] = {}
        for name, canonical in self._canonical.items():
            text = {
                "begin": "<|media_begin|>",
                "content": "<|media_content|>",
                "pad": "<|media_pad|>",
                "end": "<|media_end|>",
            }[name]
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not encoded:
                raise ValueError(f"tokenizer cannot encode {text!r}")
            native = int(encoded[-1])
            if native != canonical:
                remap[native] = canonical
        self._remap = remap
        return self._canonical

    def _renumber_media_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map native media ids back to canonical hub ids when they drift."""
        self._media_ids()
        if not self._remap:
            return input_ids
        # Simultaneous mapping: masks are computed on the *original* ids so a
        # mapped value can never be re-matched by a later key (chain collision).
        out = input_ids.clone()
        for native, canonical in self._remap.items():
            out[input_ids == native] = canonical
        return out

    def _build_mm(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build ``mm_token_type_ids``: 1 inside media spans, 0 elsewhere."""
        begin = self._canonical["begin"]
        pad = self._canonical["pad"]
        end = self._canonical["end"]
        mm = torch.zeros_like(input_ids)
        in_media = False
        for pos in range(int(input_ids.shape[0])):
            token = int(input_ids[pos])
            if token == begin:
                in_media = True
            if in_media or token == pad:
                mm[pos] = 1
            if token == end:
                in_media = False
        return mm

    # -- shared encoding helpers ---------------------------------------------

    @staticmethod
    def _normalize_messages(messages: Any, images: Any = None) -> Any:
        """Split ``<image>``/``<video>`` string placeholders into content items."""
        if images is None:
            images = []
        image_queue = list(images)
        normalized = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                parts = []
                for token in re.split(r"(<image>|<video>)", content):
                    if token in ("<image>", "<video>"):
                        if not image_queue:
                            raise ValueError(
                                f"message content contains a {token} placeholder but no matching media is available"
                            )
                        media_type = "image" if token == "<image>" else "video"
                        parts.append({"type": media_type, "url": image_queue.pop(0)})
                    elif token:
                        parts.append({"type": "text", "text": token})
                content = parts
            normalized.append({"role": message.get("role"), "content": content})
        return normalized

    def _resize_media(self, media: Any) -> Any:
        """Downscale one media item to the ``image_max_pixels`` area budget.

        The contract is MindSpeed-MM's ``_preprocess_image``: when the pixel
        count exceeds the budget, both sides are scaled by
        ``sqrt(image_max_pixels / (width * height))`` and truncated with ``int()``
        (floored at one pixel), so the aspect ratio is preserved and the result
        covers the budget from below. No rounding to the processor's patch grid
        is applied here -- the processor rounds each side up to a multiple of
        ``patch_size * merge_size`` itself, so an area budget of 262144 turns a
        1024x1024 image into 512x512, which the checkpoint renders as a 38x38
        patch grid -> 361 media tokens. Leaving the processor's patch-unit budget
        in charge instead accepts the image at 74x74 (1369 tokens) and ten of
        them then overflow the sequence window. ``image_max_pixels=None`` (the
        ``_resize_media_images`` no-op path) and images already within budget are
        returned unresized.

        A ``str`` path is opened and a PIL image is used as-is; any other type
        (an already-loaded tensor/array) is returned unchanged.
        """
        from PIL import Image  # pylint: disable=C0415

        if isinstance(media, str):
            image = Image.open(media)
        elif isinstance(media, Image.Image):
            image = media
        else:
            return media  # an already-loaded tensor/array: leave it alone
        image = image.convert("RGB")
        width, height = image.size
        if self.image_max_pixels is None or width * height <= self.image_max_pixels:
            return image
        scale = math.sqrt(self.image_max_pixels / (width * height))
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(size, Image.BICUBIC)  # pylint: disable=no-member

    def _resize_media_images(self, messages: Any) -> Any:
        """Swap media items for downscaled images (a no-op when not configured)."""
        if self.image_max_pixels is None:
            return messages
        resized = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                resized.append(message)
                continue
            items = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    item = {**item, "url": self._resize_media(item.get("url"))}
                items.append(item)
            resized.append({**message, "content": items})
        return resized

    def _encode(self, messages: Any, *, add_generation_prompt: bool) -> dict[str, Any]:
        """Render and encode one conversation with the (native) processor."""
        chat_template = getattr(self.processor, "chat_template", None)
        if not chat_template:
            tokenizer = getattr(self.processor, "tokenizer", None)
            chat_template = getattr(tokenizer, "chat_template", None)
        return self.processor.apply_chat_template(
            self._resize_media_images(messages),
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )

    # -- render / truncation bookkeeping (mirrors VLMChatTransform) ----------

    def _render(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Render one record into an unpacked, untruncated model sample.

        Shared by the eager ``__call__`` path and the deferred Omni hooks so a
        deferred pipeline renders a conversation exactly once.
        """
        messages = self._normalize_messages(record["messages"], record.get("images"))
        full = self._encode(messages, add_generation_prompt=False)
        prompt = self._encode(messages[:-1], add_generation_prompt=True)
        prompt_len = len(prompt["input_ids"][0])

        input_ids = torch.tensor(full["input_ids"][0], dtype=torch.long)
        input_ids = self._renumber_media_ids(input_ids)
        mm = self._build_mm(input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        labels[mm == 1] = IGNORE_INDEX  # never train on media positions

        sample = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(full["attention_mask"][0], dtype=torch.long),
            "mm_token_type_ids": mm,
            "labels": labels,
            # Vision-tower parameters are bf16 while the processor normalizes to
            # fp32; cast so conv/linear inputs match the parameter dtype.
            "pixel_values": full["pixel_values"].to(self.pixel_dtype),
            "image_grid_thw": full["image_grid_thw"],
        }
        return sample

    @staticmethod
    def _modal_runs(mm: torch.Tensor) -> list[tuple[int, int]]:
        """Return inclusive ``(start, end)`` positions of contiguous media runs."""
        is_modal = mm == 1
        runs: list[tuple[int, int]] = []
        start: Optional[int] = None
        for pos in range(int(mm.shape[0])):
            if bool(is_modal[pos]):
                if start is None:
                    start = pos
            elif start is not None:
                runs.append((start, pos - 1))
                start = None
        if start is not None:
            runs.append((start, int(mm.shape[0]) - 1))
        return runs

    def _drop_truncated_images(
            self, sample: dict[str, torch.Tensor], max_len: int
    ) -> dict[str, torch.Tensor]:
        """Drop images whose media-token run extends past ``max_len``."""
        grid = sample["image_grid_thw"]
        if grid.numel() == 0:
            return sample

        runs = self._modal_runs(sample["mm_token_type_ids"])
        keep = 0
        for run_idx, (_, end) in enumerate(runs):
            if run_idx >= int(grid.shape[0]) or end >= max_len:
                break
            keep += 1
        if keep == int(grid.shape[0]):
            return sample

        mm = sample["mm_token_type_ids"].clone()
        input_ids = sample["input_ids"].clone()
        attention_mask = sample["attention_mask"].clone()
        labels = sample["labels"].clone()
        for start, end in runs[keep:]:
            mm[start:end + 1] = 0
            input_ids[start:end + 1] = 0
            attention_mask[start:end + 1] = 0
            labels[start:end + 1] = IGNORE_INDEX
        sample["mm_token_type_ids"] = mm
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["labels"] = labels

        pixel_rows = sum(int(g[0]) * int(g[1]) * int(g[2]) for g in grid[:keep])
        sample["pixel_values"] = sample["pixel_values"][:pixel_rows]
        sample["image_grid_thw"] = grid[:keep]
        return sample

    def _pad_target(self, seq_len: int) -> int:
        """Return this sample's post-pad length.

        Without ``pad_granularity`` this is always ``max_seq_len`` (the historical
        fixed-length contract). With it, the sample grows only to the next
        multiple of the granularity, so a batch of short samples no longer pays
        ``max_seq_len`` worth of compute; the collator then pads the batch to its
        longest member.
        """
        if self.pad_granularity is None:
            return self.max_seq_len
        granularity = self.pad_granularity
        rounded = -(-seq_len // granularity) * granularity
        return min(self.max_seq_len, max(rounded, granularity))

    def _truncate(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Truncate one sample to ``max_seq_len``, dropping cut images.

        With ``truncate_mode="proportional"`` the truncation keeps a share of the
        window for the supervised region, which sits at the end of the stream;
        with ``"head"`` (or when there is nothing supervised to preserve) only the
        leading ``max_seq_len`` tokens survive. No padding is applied, so the
        packing path can concatenate whole samples.
        """
        max_len = self.max_seq_len
        seq_len = int(sample["input_ids"].shape[0])
        if seq_len <= max_len:
            return sample

        # Read the supervised span off the pre-cut labels: the image drop below
        # rewrites them.
        supervised = sample["labels"] != IGNORE_INDEX
        positions = supervised.nonzero().flatten()
        if self.truncate_mode == "proportional" and positions.numel():
            resp_start = int(positions[0])
            resp_end = int(positions[-1]) + 1
            # The target region runs from the first supervised token to the end
            # of the stream, so a trailing end-of-turn scaffold is part of it.
            source_len = resp_start
            target_len = seq_len - resp_start
            new_source_len, new_target_len = _infer_seqlen(source_len, target_len, max_len)
            if new_source_len + new_target_len > max_len:
                raise ValueError(
                    f"truncation kept {new_source_len + new_target_len} tokens "
                    f"for a {max_len}-token budget (source {source_len}, target "
                    f"{target_len}, supervised span {resp_start}:{resp_end})"
                )
            # Images are dropped against the new prompt boundary, not max_len.
            sample = self._drop_truncated_images(sample, new_source_len)
            for field in _SEQ_FIELDS:
                sample[field] = torch.cat([
                    sample[field][:new_source_len],
                    sample[field][resp_start:resp_start + new_target_len],
                ])
            return sample

        sample = self._drop_truncated_images(sample, max_len)
        for field in _SEQ_FIELDS:
            sample[field] = sample[field][:max_len]
        return sample

    def _truncate_and_pad(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Truncate (dropping cut images) and pad sequence fields to the pad target."""
        sample = self._truncate(sample)
        seq_len = int(sample["input_ids"].shape[0])
        target_len = self._pad_target(seq_len)
        for field in _SEQ_FIELDS:
            value = sample[field]
            if value.shape[0] < target_len:
                pad_value = IGNORE_INDEX if field == "labels" else 0
                sample[field] = torch.cat([
                    value,
                    torch.full((target_len - value.shape[0],), pad_value, dtype=value.dtype),
                ])
        return sample

    def __call__(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode one record into one padded model sample."""
        sample = self._render(record)
        return self._truncate_and_pad(sample)


class _KimiOmniTransformBase(OmniDataTransform):
    """Bind the Kimi renderer to the shared Omni transform lifecycle."""

    def __init__(
            self,
            *,
            processor: Any,
            max_seq_len: int = 1024,
            pixel_dtype: str = "bfloat16",
            pad_granularity: int | None = None,
            image_max_pixels: int | None = None,
            truncate_mode: str = "proportional",
    ) -> None:
        """Store the shared renderer and validate the sequence limit."""
        if processor is None:
            raise ValueError("processor is required for the Kimi Omni data transform")
        super().__init__(max_seq_len=max_seq_len, processor=processor)
        self.chat_transform = KimiVLMChatTransform(
            processor,
            max_seq_len=max_seq_len,
            pixel_dtype=pixel_dtype,
            pad_granularity=pad_granularity,
            image_max_pixels=image_max_pixels,
            truncate_mode=truncate_mode,
        )

    def _prepare_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Resolve media paths and hand the raw record to the renderer."""
        record = dict(sample)
        messages = self.prepare_messages(sample)
        source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
        if source_path is not None:
            source_directory = os.path.dirname(os.path.abspath(source_path))
            self._resolve_message_media(messages, source_directory)
        record["messages"] = messages
        record["images"] = self._resolve_record_images(sample, source_path)
        return record

    def _resolve_message_media(self, messages: Any, source_directory: str) -> None:
        """Anchor every path-bearing content field to the source directory.

        ``prepare_messages`` resolves the ``image``/``video``/``audio`` and
        ``image_url`` fields. The Kimi content-list contract also carries media
        under ``url``, which is resolved here so a JSON-array source (loaded
        through the ``datasets`` json builder) keeps the same semantics as a
        JSONL source.
        """
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for content_block in content:
                self._resolve_content_block(content_block, source_directory)

    def _resolve_content_block(self, content_block: Any, source_directory: str) -> None:
        """Resolve the path-bearing keys of one content block in place."""
        if not isinstance(content_block, dict):
            return

        for key in ("url", "image", "video", "audio"):
            value = content_block.get(key)
            if isinstance(value, str):
                content_block[key] = self._resolve_media_path(value, source_directory)

        image_url = content_block.get("image_url")
        if isinstance(image_url, str):
            content_block["image_url"] = self._resolve_media_path(image_url, source_directory)
        elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            image_url["url"] = self._resolve_media_path(image_url["url"], source_directory)

    def _resolve_record_images(self, sample: dict[str, Any], source_path: str | None) -> Any:
        """Resolve the top-level ``images`` list against the source file directory.

        The historical VLM records carry a top-level ``images`` list consumed by
        ``<image>`` string placeholders, which must be anchored like the
        content-list media; without this a JSON-array source loaded through the
        ``datasets`` json builder would hand the processor a bare relative filename.
        """
        images = sample.get("images")
        if not images or source_path is None:
            return images

        source_directory = os.path.dirname(os.path.abspath(source_path))
        resolved_images = []
        for image in images:
            if isinstance(image, str):
                image = self._resolve_media_path(image, source_directory)
            resolved_images.append(image)
        return resolved_images


class KimiOmniTransform(_KimiOmniTransformBase):
    """Eager Kimi-K2.x Omni transform.

    ``encode_sample`` fully encodes one record before candidate selection, which
    is what the packing pipeline needs to derive each sample's cost from its
    encoded ``input_ids``. Use :class:`KimiPackingOmniTransform` when the render
    must be deferred until after selection.
    """

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Encode one Omni conversation into one padded model sample."""
        return self.chat_transform(self._prepare_record(sample))


class KimiPackingOmniTransform(_KimiOmniTransformBase):
    """Deferred Kimi-K2.x Omni transform for packing pipelines.

    ``preencode_sample`` renders the conversation exactly once and attaches the
    exact ``packing_length`` used by the packing selector, while
    ``postencode_sample`` applies length truncation after selection. Padding is
    intentionally *not* applied per sample: the packing path concatenates whole
    samples and pads only the packed window.
    """

    def preencode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Render one conversation and report its exact packing length."""
        rendered = self.chat_transform._render(self._prepare_record(sample))  # pylint: disable=protected-access
        rendered["packing_length"] = int(rendered["input_ids"].shape[0])
        return rendered

    def postencode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Truncate one selected sample to ``max_seq_len`` without padding."""
        rendered = {field: value for field, value in sample.items() if field != "packing_length"}
        return self.chat_transform._truncate(rendered)  # pylint: disable=protected-access


def build_kimi_omni_transform(
        *,
        processor: Any,
        max_seq_len: int = 1024,
        pixel_dtype: str = "bfloat16",
        pad_granularity: int | None = None,
        image_max_pixels: int | None = None,
        truncate_mode: str = "proportional",
        deferred: bool = False,
        **transform_options: Any,
) -> OmniDataTransform:
    """Build the Kimi-K2.x Omni sample transform.

    Args:
        processor: Native ``kimi_k25`` processor used to render conversations.
        max_seq_len: Target sequence length for truncation and padding.
        pixel_dtype: Dtype for ``pixel_values`` (must match model params).
        pad_granularity: Pad each sample only to the next multiple of this value
            instead of always ``max_seq_len``. Pair with a collator that pads the
            micro-batch to its own longest sample. Use a multiple of
            ``cp_size * tp_size`` when CP/TP sequence parallelism is enabled.
        image_max_pixels: Downscale each media image to at most this many pixels
            before the processor sees it, using MindSpeed-MM's area semantics
            (``sqrt(image_max_pixels / (width * height))`` on both sides, aspect
            ratio preserved, never upscaled). ``None`` keeps the processor's
            patch-unit budget.
        truncate_mode: Truncation policy for samples longer than ``max_seq_len``,
            ``"proportional"`` (default) or ``"head"``.
        deferred: Return the deferred ``preencode_sample``/``postencode_sample``
            transform instead of the eager ``encode_sample`` one. Use it when the
            pipeline must select packing candidates before the final encoding.
        **transform_options: Reserved model-specific transform options.

    Returns:
        The configured Omni Kimi transform.

    Raises:
        ValueError: If ``processor`` is not provided, or a transform option is
            invalid.
    """
    del transform_options
    transform_type = KimiPackingOmniTransform if deferred else KimiOmniTransform
    return transform_type(
        processor=processor,
        max_seq_len=max_seq_len,
        pixel_dtype=pixel_dtype,
        pad_granularity=pad_granularity,
        image_max_pixels=image_max_pixels,
        truncate_mode=truncate_mode,
    )


def build_kimi_vlm_data_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 1024,
        pixel_dtype: str = "bfloat16",
        pad_granularity: int | None = None,
        image_max_pixels: int | None = None,
        truncate_mode: str = "proportional",
        **transform_options: Any,
) -> KimiVLMChatTransform:
    """Build the raw Kimi-K2.x sample transform.

    Retained for callers that drive :class:`KimiVLMChatTransform` directly.
    Prefer :func:`build_kimi_omni_transform` for the Omni data lifecycle.

    Args:
        processor: Native ``kimi_k25`` processor used to render conversations.
        max_seq_len: Target sequence length for truncation and padding.
        pixel_dtype: Dtype for ``pixel_values`` (must match model params).
        pad_granularity: Pad each sample only to the next multiple of this value
            instead of always ``max_seq_len``.
        image_max_pixels: Area budget for per-image downscaling.
        truncate_mode: Truncation policy, ``"proportional"`` or ``"head"``.
        **transform_options: Reserved model-specific transform options.

    Returns:
        The configured :class:`KimiVLMChatTransform`.

    Raises:
        ValueError: If ``processor`` is not provided.
    """
    del transform_options
    if processor is None:
        raise ValueError("processor is required for the Kimi VLM data transform")
    return KimiVLMChatTransform(processor, max_seq_len=max_seq_len,
                                pixel_dtype=pixel_dtype,
                                pad_granularity=pad_granularity,
                                image_max_pixels=image_max_pixels,
                                truncate_mode=truncate_mode)


__all__ = [
    "KimiOmniTransform",
    "KimiPackingOmniTransform",
    "KimiVLMChatTransform",
    "build_kimi_omni_transform",
    "build_kimi_vlm_data_transform",
]

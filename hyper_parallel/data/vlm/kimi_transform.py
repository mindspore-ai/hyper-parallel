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
"""Kimi-K2.x multimodal sample transform (native ``kimi_k25`` protocol).

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
"""

import re
from typing import Any, Optional

import torch

from hyper_parallel.data.constants import IGNORE_INDEX

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")


class KimiVLMChatTransform:
    """Encode one Kimi multimodal conversation into one padded model sample."""

    def __init__(self, processor: Any, *, max_seq_len: int = 1024,
                 pixel_dtype: str = "bfloat16") -> None:
        """Store the processor, the target length, and the media-id remap."""
        self.processor = processor
        self.max_seq_len = max_seq_len
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

    def _encode(self, messages: Any, *, add_generation_prompt: bool) -> dict[str, Any]:
        """Render and encode one conversation with the (native) processor."""
        chat_template = getattr(self.processor, "chat_template", None)
        if not chat_template:
            tokenizer = getattr(self.processor, "tokenizer", None)
            chat_template = getattr(tokenizer, "chat_template", None)
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )

    # -- truncation bookkeeping (mirrors VLMChatTransform) -------------------

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

    def _truncate_and_pad(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Truncate (dropping cut images) and pad sequence fields to ``max_seq_len``."""
        max_len = self.max_seq_len
        seq_len = int(sample["input_ids"].shape[0])
        if seq_len > max_len:
            sample = self._drop_truncated_images(sample, max_len)
            for field in _SEQ_FIELDS:
                sample[field] = sample[field][:max_len]

        for field in _SEQ_FIELDS:
            value = sample[field]
            if value.shape[0] < max_len:
                pad_value = IGNORE_INDEX if field == "labels" else 0
                sample[field] = torch.cat([
                    value,
                    torch.full((max_len - value.shape[0],), pad_value, dtype=value.dtype),
                ])
        return sample

    def __call__(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode one record into a padded model sample."""
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
        return self._truncate_and_pad(sample)


def build_kimi_vlm_data_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 1024,
        pixel_dtype: str = "bfloat16",
        **transform_options: Any,
) -> KimiVLMChatTransform:
    """Build the Kimi-K2.x multimodal sample transform.

    Args:
        processor: Native ``kimi_k25`` processor used to render conversations.
        max_seq_len: Target sequence length for truncation and padding.
        pixel_dtype: Dtype for ``pixel_values`` (must match model params).
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
                                pixel_dtype=pixel_dtype)


__all__ = ["KimiVLMChatTransform", "build_kimi_vlm_data_transform"]

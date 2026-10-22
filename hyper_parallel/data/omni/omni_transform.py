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
# =====================================================================
"""Omni transform lifecycle and the default AutoProcessor implementation."""

from __future__ import annotations

import copy
import io
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from transformers import AutoProcessor

from hyper_parallel.data.constants import IGNORE_INDEX, OMNI_CP_TOKEN_FIELDS, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.online.format import (
    convert_sharegpt_to_messages,
    normalize_chat_media_fields,
    resolve_media_path,
)

logger = get_dataset_logger(__name__)


def build_labels_from_supervised_start(
        input_ids: torch.Tensor,
        supervised_start: int,
        *,
        media_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise text from a token position through the end of one sample.

    Args:
        input_ids: One encoded sample with shape [sequence].
        supervised_start: First supervised token after the prompt or BOS.
        media_mask: Media positions excluded from supervision.

    Returns:
        Labels retaining supervised text and masking other positions with IGNORE_INDEX.
    """
    if input_ids.ndim != 1:
        raise ValueError("Omni label construction requires one-dimensional input_ids")

    if not 0 <= supervised_start <= input_ids.numel():
        raise ValueError(f"Invalid supervised start {supervised_start} for length {input_ids.numel()}")

    # [prompt..., reply, EOS] -> [-100..., reply, EOS]; next-token shifting belongs to the model loss.
    labels = input_ids.clone()
    labels[:supervised_start] = IGNORE_INDEX

    if media_mask is not None:
        if media_mask.shape != input_ids.shape:
            raise ValueError("media_mask must have the same shape as input_ids")

        labels[media_mask.bool()] = IGNORE_INDEX

    return labels


def build_auto_processor(
        *,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        **processor_kwargs: Any,
) -> Any:
    """Build the Transformers processor used by the default Omni transform.

    Args:
        pretrained_model_name_or_path: Model directory or Hub identifier.
        trust_remote_code: Whether Transformers may load model-owned code.
        **processor_kwargs: Additional ``AutoProcessor.from_pretrained`` options.

    Returns:
        Processor with an attached tokenizer and resolved chat template.
    """
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        **processor_kwargs,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor must provide a tokenizer")

    chat_template = getattr(processor, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is not None:
            processor.chat_template = chat_template

    return processor


class OmniDataTransform:
    """Define model-specific sample and batch encoding hooks.

    Eager transforms implement ``encode_sample`` to return model inputs and labels.
    Deferred transforms provide ``preencode_sample`` and ``postencode_sample``;
    offline metadata needs only ``postencode_sample``. The pipeline detects
    these hook combinations automatically.

    Packing selection and concatenation belong to the optional packing path
    rather than this transform.
    """

    def __init__(
            self, *, max_seq_len: int, processor: Any, training_mode: str = "sft",
    ) -> None:
        """Store the sequence limit and externally built processor."""
        if training_mode not in ("sft", "pretrain"):
            raise ValueError(f"Unsupported Omni training mode: {training_mode!r}")

        self.training_mode = training_mode
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if processor is None:
            raise ValueError("processor is required")

        self.max_seq_len = max_seq_len
        self.processor = processor

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Return whether one source record contains conversation messages."""
        messages = sample.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Omni sample messages must be a list")

        valid_sample = bool(messages)
        return valid_sample

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode a complete sample with unshifted labels, or return None to skip it."""
        del sample
        raise NotImplementedError(f"{type(self).__name__} does not implement sample encoding")

    def preencode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Prepare metadata used by a packing selector without full encoding."""
        del sample
        raise NotImplementedError(f"{type(self).__name__} does not implement preencode_sample")

    def postencode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Encode selected online or offline metadata into a model-ready sample."""
        del sample
        raise NotImplementedError(f"{type(self).__name__} does not implement postencode_sample")

    def encode_batch(self, batch: Any) -> Any:
        """Apply an optional model-specific batch encoding stage.

        Args:
            batch: Collated model inputs.
        """
        return batch

    def prepare_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Bind external images in placeholder order and resolve source-relative paths.

        Args:
            sample: Messages, optional external images, and source path metadata.
        """
        messages = sample.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Omni sample must contain non-empty messages")

        prepared_messages = copy.deepcopy(messages)
        self._bind_external_images(prepared_messages, sample.get("images"))
        source_path = sample.get(ONLINE_SOURCE_PATH_KEY)
        if source_path is None:
            return prepared_messages

        source_directory = os.path.dirname(os.path.abspath(source_path))
        for message in prepared_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for content_block in content:
                self._resolve_media_block(content_block, source_directory)
        return prepared_messages

    def _bind_external_images(self, messages: list, images: Any) -> None:
        """Consume external images only after checking the complete placeholder count."""
        if images is None:
            images = []
        if not isinstance(images, list):
            raise TypeError("Omni sample images must be a list")
        placeholders = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and (images or "<image>" in content):
                content = [{"type": "text", "text": content}]
            if isinstance(content, list):
                message["content"] = self._expand_image_placeholders(content, placeholders)
        if len(placeholders) != len(images):
            raise ValueError(f"Image placeholder count {len(placeholders)} does not match images count {len(images)}")
        for index, (block, image) in enumerate(zip(placeholders, images)):
            block["image"] = self._decode_external_image(image, index)

    @staticmethod
    def _expand_image_placeholders(content: list, placeholders: list) -> list:
        """Preserve text around each placeholder and leave explicit media untouched."""
        expanded = []
        for block in content:
            if not isinstance(block, dict):
                expanded.append(block)
                continue
            text_content = block.get("text")
            if (block.get("type") == "text"
                    and isinstance(text_content, str) and "<image>" in text_content):
                for index, text in enumerate(block["text"].split("<image>")):
                    if index:
                        image_block = {"type": "image"}
                        expanded.append(image_block)
                        placeholders.append(image_block)
                    if text:
                        expanded.append({**block, "text": text})
            else:
                expanded.append(block)
                if block.get("type") == "image":
                    # Arrow structs can materialize absent source fields as nulls.
                    source_fields = ("image", "url", "path", "base64")
                    for field in source_fields:
                        if block.get(field) is None:
                            block.pop(field, None)
                    if not any(field in block for field in source_fields):
                        placeholders.append(block)
        return expanded

    @staticmethod
    def _decode_external_image(image: Any, index: int) -> Any:
        """Prefer embedded bytes over path, returning a fully loaded RGB image."""
        if isinstance(image, str) and image:
            return image
        if isinstance(image, Mapping):
            payload = image.get("bytes")
            if payload is not None:
                if not isinstance(payload, (bytes, bytearray, memoryview)):
                    raise TypeError(f"images[{index}].bytes must be binary image data")
                # Pillow is optional for configurations that only use text/path messages.
                from PIL import Image  # pylint: disable=C0415

                try:
                    with io.BytesIO(payload) as buffer, Image.open(buffer) as decoded:
                        return decoded.convert("RGB")
                except (OSError, ValueError) as error:
                    raise ValueError(f"Cannot decode images[{index}].bytes") from error
            path = image.get("path")
            if isinstance(path, str) and path:
                return path
        raise ValueError(f"images[{index}] must be a path or a mapping with bytes/path")

    @classmethod
    def _resolve_media_block(cls, content_block: Any, source_directory: str) -> None:
        """Resolve path-bearing fields without changing the processor's message format."""
        if not isinstance(content_block, dict):
            return

        block_type = content_block.get("type")
        if block_type == "image_url":
            image_url = content_block.get("image_url")
            if isinstance(image_url, str):
                content_block["image_url"] = cls._resolve_media_path(image_url, source_directory)
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                image_url["url"] = cls._resolve_media_path(image_url["url"], source_directory)
            return

        if block_type in ("image", "video", "audio"):
            for field in (block_type, "url", "path"):
                media_field = content_block.get(field)
                if isinstance(media_field, str):
                    content_block[field] = cls._resolve_media_path(media_field, source_directory)

    @staticmethod
    def _resolve_media_path(media_path: str, source_directory: str) -> str:
        """Delegate media path resolution for existing model adapters."""
        resolved_path = resolve_media_path(media_path, source_directory)
        return resolved_path


class AutoProcessorTransform(OmniDataTransform):
    """Encode a conversation and supervise only its final assistant response."""

    def __init__(
            self, *, max_seq_len: int, processor: Any, training_mode: str = "sft",
            sample_format: str = "messages",
    ) -> None:
        """Select messages/OpenAI/HF or fixed ShareGPT input explicitly."""
        super().__init__(max_seq_len=max_seq_len, processor=processor, training_mode=training_mode)
        if sample_format not in ("messages", "openai", "hf", "sharegpt"):
            raise ValueError(f"Unsupported AutoProcessor sample format: {sample_format!r}")

        self.sample_format = sample_format

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Filter the configured conversation column before source construction."""
        message_key = "conversations" if self.sample_format == "sharegpt" else "messages"
        messages = sample.get(message_key)
        if not isinstance(messages, list):
            raise ValueError(f"Omni {self.sample_format} sample requires a {message_key} list")

        valid_sample = bool(messages)
        return valid_sample

    def _assistant_prompt_length(self, messages: list, input_ids: Any, prompt_ids: Any) -> int:
        """Locate the prompt in processor output without retokenizing media spans.

        A token crossing the source/target character boundary is masked in full.
        This preserves native processor inputs, unlike separately encoding and
        concatenating the source and target, which changes sequence-aligned fields.
        """
        if len(prompt_ids) < len(input_ids) and input_ids[:len(prompt_ids)].equal(prompt_ids):
            return len(prompt_ids)

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError("Assistant generation prompt must be a text prefix of the full conversation")

        full_tokens = input_ids.tolist()
        prompt_tokens = prompt_ids.tolist()
        common_length = 0
        for full_token, prompt_token in zip(full_tokens, prompt_tokens):
            if full_token != prompt_token:
                break
            common_length += 1
        return self._merged_prompt_length(full_tokens, prompt_tokens, common_length)

    def _merged_prompt_length(self, full_tokens: list[int], prompt_tokens: list[int], common_length: int) -> int:
        """Accept only text-equivalent retokenization after the shared token prefix."""
        decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        tokenizer = self.processor.tokenizer
        full_tail = tokenizer.decode(full_tokens[common_length:], **decode_kwargs)
        prompt_tail = tokenizer.decode(prompt_tokens[common_length:], **decode_kwargs)
        # Reject lossy decoding rather than guessing a boundary inside UTF-8 bytes.
        if not prompt_tail or "\ufffd" in full_tail or "\ufffd" in prompt_tail or not full_tail.startswith(prompt_tail):
            raise ValueError("Assistant prompt tokens differ beyond a lossless text boundary merge")
        for end in range(common_length + 1, len(full_tokens) + 1):
            decoded = tokenizer.decode(full_tokens[common_length:end], **decode_kwargs)
            if full_tail.startswith(decoded) and decoded.startswith(prompt_tail):
                return end
        raise ValueError("Cannot locate assistant supervision boundary in processor tokens")

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Build model inputs and assistant-only labels from processor messages."""
        if self.training_mode != "sft":
            raise NotImplementedError("Image-caption pretraining requires a model-specific encoding method")

        if self.sample_format == "sharegpt":
            messages = convert_sharegpt_to_messages(sample)
        else:
            prepared_sample = dict(sample, messages=self.prepare_messages(sample))
            messages = normalize_chat_media_fields(prepared_sample)
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("Omni training sample must end with a final assistant message")

        encoded_sample = dict(self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ))
        # The generation prompt includes the assistant header: [user, header | reply, EOS].
        prompt_sample = self.processor.apply_chat_template(
            messages[:-1],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        input_ids = encoded_sample["input_ids"].squeeze(0)
        prompt_ids = prompt_sample["input_ids"].squeeze(0)
        if len(input_ids) > self.max_seq_len:
            raise ValueError("Omni sample exceeds max_seq_len; refusing to truncate a possible media span")
        prompt_length = self._assistant_prompt_length(messages, input_ids, prompt_ids)

        media_mask = None
        if "mm_token_type_ids" in encoded_sample:
            media_mask = encoded_sample["mm_token_type_ids"].squeeze(0) != 0
        labels = build_labels_from_supervised_start(
            input_ids, prompt_length, media_mask=media_mask,
        )

        # Single-sample token fields: [1, L] -> [L]; media tensors keep the processor layout.
        for field in OMNI_CP_TOKEN_FIELDS.intersection(encoded_sample):
            encoded_sample[field] = encoded_sample[field].squeeze(0)
        encoded_sample["labels"] = labels
        return encoded_sample


SampleEncoder = Callable[[Any], Any]


@dataclass(frozen=True)
class _OmniTransformStrategy:
    """Resolve the encoding stages implemented by an Omni transform."""

    sample_encoder: SampleEncoder | None
    selected_sample_encoder: SampleEncoder | None

    @classmethod
    def from_transform(cls, transform: OmniDataTransform) -> "_OmniTransformStrategy":
        """Select eager or deferred encoding from the implemented pipeline hooks."""
        transform_type = type(transform)
        encode_sample = transform_type.encode_sample is not OmniDataTransform.encode_sample
        preencode_sample = transform_type.preencode_sample is not OmniDataTransform.preencode_sample
        postencode_sample = transform_type.postencode_sample is not OmniDataTransform.postencode_sample

        if encode_sample and (preencode_sample or postencode_sample):
            raise TypeError(
                f"{type(transform).__name__} cannot combine encode_sample with "
                "preencode_sample or postencode_sample"
            )

        if preencode_sample and not postencode_sample:
            raise TypeError(
                f"{type(transform).__name__} must pair preencode_sample with postencode_sample"
            )

        if not encode_sample and not postencode_sample:
            raise TypeError(
                f"{type(transform).__name__} must implement encode_sample or postencode_sample"
            )

        # encode_sample                     -> encode before candidate selection
        # preencode_sample + postencode_sample -> prepare metadata, then defer full encoding
        # postencode_sample only            -> preserve offline metadata, then defer encoding
        sample_encoder = None
        if preencode_sample:
            sample_encoder = transform.preencode_sample
        elif encode_sample:
            sample_encoder = transform.encode_sample

        selected_sample_encoder = None
        if postencode_sample:
            selected_sample_encoder = transform.postencode_sample

        transform_strategy = cls(
            sample_encoder=sample_encoder,
            selected_sample_encoder=selected_sample_encoder,
        )
        return transform_strategy

    def encode_selected_sample(self, sample: Any) -> Any:
        """Finish encoding a sample after packing selection.

        Args:
            sample: Selected candidate.
        """
        encoded_sample = sample
        if self.selected_sample_encoder is not None:
            encoded_sample = self.selected_sample_encoder(sample)
        return encoded_sample

    def prepare_samples(self, raw_sample: Any) -> list[Any]:
        """Return encoded, preencoded, or unchanged samples for candidate selection."""
        prepared_sample = raw_sample
        if self.sample_encoder is not None:
            prepared_sample = self.sample_encoder(raw_sample)
        if prepared_sample is None:
            logger.warning("Skipping sample rejected by transform")
            return []

        return [prepared_sample]


__all__ = [
    "build_auto_processor",
    "OmniDataTransform",
    "AutoProcessorTransform",
]

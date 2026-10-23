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
"""Omni transform lifecycle and the default AutoProcessor implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from transformers import AutoProcessor

from hyper_parallel.data.constants import IGNORE_INDEX, OMNI_CP_TOKEN_FIELDS
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.online.format import (
    convert_sharegpt_to_messages,
    normalize_chat_media_fields,
    resolve_chat_media_paths,
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
        """Apply an optional model-specific batch encoding stage."""
        return batch

    def prepare_messages(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Copy messages and resolve media paths relative to their JSONL file."""
        prepared_messages = resolve_chat_media_paths(sample)
        return prepared_messages

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

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Build model inputs and assistant-only labels from processor messages."""
        if self.training_mode != "sft":
            raise NotImplementedError("Image-caption pretraining requires a model-specific encoding method")

        if self.sample_format == "sharegpt":
            messages = convert_sharegpt_to_messages(sample)
        else:
            messages = normalize_chat_media_fields(sample)
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
        prompt_length = len(prompt_ids)
        if prompt_length >= len(input_ids) or not input_ids[:prompt_length].equal(prompt_ids):
            raise ValueError("Assistant prompt tokens must be a prefix of the full conversation")

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
        """Finish encoding a sample after packing selection."""
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

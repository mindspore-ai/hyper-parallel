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
"""Build LLM plaintext, conversation, and pretokenized data transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch

from hyper_models.components.datasets.contracts import ModelSample, RawSample, SampleTransform

LLMDataType = Literal["plaintext", "conversation", "pretokenized"]
TextKeys: TypeAlias = str | Sequence[str]


def _get_record_value(sample: Mapping[str, Any], keys: TextKeys) -> Any:
    if isinstance(keys, str):
        try:
            return sample[keys]
        except KeyError as exc:
            raise ValueError(f"Sample does not contain field {keys!r}") from exc
    for key in keys:
        if key in sample:
            return sample[key]
    raise ValueError(f"Sample does not contain any configured text fields: {list(keys)!r}")


class IdentityDataTransform:
    """Return each input sample unchanged."""

    def __init__(self, tokenizer: Any = None, chat_template: Any = None) -> None:
        """Retain optional upstream assets for target compatibility.

        Args:
            tokenizer: Optional tokenizer built by the LLM Trainer.
            chat_template: Optional chat template built from model assets.
        """
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def __call__(self, sample: Any) -> Any:
        """Return the input sample without modification."""
        return sample


@dataclass
class PlaintextTransform:
    """Tokenize plaintext records into one or more model samples."""

    tokenizer: Any
    max_seq_len: int
    text_keys: TextKeys = "text"

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for plaintext data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def __call__(self, sample: RawSample) -> list[dict[str, Any]]:
        """Tokenize and chunk one plaintext record."""
        text = _get_record_value(sample, self.text_keys)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            token_ids = [*token_ids, eos_token_id]

        transformed = []
        for start in range(0, len(token_ids), self.max_seq_len):
            input_ids = torch.tensor(token_ids[start:start + self.max_seq_len], dtype=torch.long)
            model_sample = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids.clone(),
            }
            transformed.append(model_sample)
        return transformed


@dataclass
class ConversationTransform:
    """Encode conversation records with a configured chat template."""

    chat_template: Any
    max_seq_len: int
    text_keys: TextKeys = "conversation"

    def __post_init__(self) -> None:
        if self.chat_template is None:
            raise ValueError("chat_template is required for conversation data")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def __call__(self, sample: RawSample) -> list[dict[str, Any]]:
        """Encode one conversation record."""
        messages = _get_record_value(sample, self.text_keys)
        encoded = self.chat_template.encode_messages(messages, max_seq_len=self.max_seq_len)
        model_sample = {field: torch.as_tensor(value) for field, value in encoded.items()}
        return [model_sample]


@dataclass
class PretokenizedTransform:
    """Normalize one pretokenized indexed GPT record.

    Indexed Dataset code has already shifted labels. It may also provide loss
    masks, position IDs, and an attention mask, omits those
    fields when ``create_attention_mask=False``. Runtime ``get_batch`` owns
    their final reconstruction for both forms.
    """

    max_seq_len: int | None = None

    def __post_init__(self) -> None:
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive or None")

    def __call__(self, sample: RawSample) -> ModelSample:
        """Convert one pretokenized indexed record into one model sample.

        Args:
            sample: Record containing ``tokens`` or ``input_ids`` and shifted
                ``labels``. Runtime mask fields are optional.

        Returns:
            The normalized model sample. A missing attention mask remains
            missing so kernels can construct it when configured to do so.

        Raises:
            ValueError: If required fields are missing, sequence shapes are
                inconsistent, or the indexed sequence exceeds ``max_seq_len``.
        """
        # Keep the source record unchanged and expose source ``tokens`` through
        # the model-facing ``input_ids`` name.
        normalized = dict(sample)
        if "input_ids" not in normalized:
            if "tokens" not in normalized:
                raise ValueError("Pretokenized samples must contain 'input_ids' or 'tokens'")
            normalized["input_ids"] = normalized.pop("tokens")

        # Tokens and shifted labels.
        # GPTDataset may additionally return masks and positions, while GPTFromMRDataset
        # intentionally omits them when attention-mask creation is disabled.
        required_fields = (
            "input_ids",
            "labels",
        )
        missing_fields = [field for field in required_fields if field not in normalized]
        if missing_fields:
            raise ValueError(f"Pretokenized samples must contain fields: {missing_fields!r}")

        # Normalize source dtypes without recomputing any field.
        field_dtypes = {
            "input_ids": torch.long,
            "labels": torch.long,
            "loss_mask": torch.float,
            "position_ids": torch.long,
            "text_position_ids": torch.long,
        }
        for field, dtype in field_dtypes.items():
            if field in normalized:
                normalized[field] = torch.as_tensor(normalized[field], dtype=dtype)

        # Indexed data should already have the configured sequence length. Do
        # not silently truncate because labels and masks must remain aligned.
        input_ids = normalized["input_ids"]
        if input_ids.ndim != 1:
            raise ValueError(f"Pretokenized field 'input_ids' must be one-dimensional, got {input_ids.shape}")
        sequence_length = input_ids.shape[0]
        if self.max_seq_len is not None and sequence_length > self.max_seq_len:
            raise ValueError(
                f"Pretokenized sequence length ({sequence_length}) exceeds max_seq_len ({self.max_seq_len})"
            )

        for field in ("labels", "loss_mask", "position_ids", "text_position_ids"):
            if field not in normalized:
                continue
            value = normalized[field]
            if value.ndim != 1 or value.shape[0] != sequence_length:
                raise ValueError(
                    f"Pretokenized field {field!r} must have shape ({sequence_length},), got {tuple(value.shape)}"
                )

        # ``attention_mask`` is optional when the attention kernel creates it.
        # When present, preserve the [1, sequence, sequence] causal mask.
        if "attention_mask" in normalized:
            attention_mask = torch.as_tensor(normalized["attention_mask"], dtype=torch.bool)
            expected_shape = (1, sequence_length, sequence_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"Pretokenized field 'attention_mask' must have shape {expected_shape}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            normalized["attention_mask"] = attention_mask
        return normalized


def build_llm_data_transform(
        data_type: LLMDataType,
        *,
        tokenizer: Any = None,
        chat_template: Any = None,
        max_seq_len: int,
        text_keys: TextKeys = "text",
) -> SampleTransform:
    """Build the transform selected by the LLM data type.

    Args:
        data_type: Plaintext, conversation, or pretokenized input format.
        tokenizer: Tokenizer used by plaintext transforms.
        chat_template: Chat template used by conversation transforms.
        max_seq_len: Maximum model sequence length.
        text_keys: Field or candidate fields containing the source text.

    Returns:
        The configured LLM sample transform.

    Raises:
        ValueError: If ``data_type`` is unsupported.
    """
    if data_type == "plaintext":
        data_transform = PlaintextTransform(tokenizer, max_seq_len, text_keys)
    elif data_type == "conversation":
        data_transform = ConversationTransform(chat_template, max_seq_len, text_keys)
    elif data_type == "pretokenized":
        data_transform = PretokenizedTransform(max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Unsupported LLM data type: {data_type!r}")
    return data_transform


__all__ = ["build_llm_data_transform"]

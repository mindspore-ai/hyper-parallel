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
"""Model-owned plaintext transform aligning V4.1 text documents to DSA blocks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.text.text_transform import PlaintextTransform

logger = get_dataset_logger(__name__)


class DeepseekV41TextTransform(PlaintextTransform):
    """Tokenize plaintext with per-document alignment for DSA compression.

    The V4.1 DSA compressor divides each document into fixed-size blocks; a
    document whose length is not a multiple of the active compress ratios
    would make block shapes, and with them the captured static graph, depend
    on the data. Mirroring the Omni/VLM transform contract, every packed
    document is padded to a ``sample_alignment_size`` multiple here, at the
    transform boundary and before packing selection, so the loader token
    budget and eager/static-graph attention share one compact shape contract.
    """

    def __init__(
            self,
            tokenizer: Any,
            max_seq_len: int,
            text_keys: str | Sequence[str] = "text",
            drop_ragged_tail: bool = False,
            sample_alignment_size: int = 1,
    ) -> None:
        """Bind the plaintext encoding policy to the alignment contract.

        Args:
            tokenizer: Tokenizer encoding raw plaintext records.
            max_seq_len: Maximum model sequence length per emitted sample.
            text_keys: Source field(s) of the plaintext record.
            drop_ragged_tail: Drop trailing partial plaintext windows.
            sample_alignment_size: Pad every emitted sample to a multiple of
                this value (use the lcm of the model's active compress ratios).
                Default 1 keeps natural sample lengths.
        """
        super().__init__(tokenizer, max_seq_len, text_keys, drop_ragged_tail)
        if (isinstance(sample_alignment_size, bool)
                or not isinstance(sample_alignment_size, int)
                or sample_alignment_size <= 0):
            raise ValueError(f"sample_alignment_size must be a positive integer, got {sample_alignment_size!r}")
        self.sample_alignment_size = sample_alignment_size

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Tokenize, chunk, and align one plaintext record."""
        aligned_samples = []
        for model_sample in super().__call__(sample):
            aligned_sample = self._align_sample(model_sample)
            if aligned_sample is not None:
                aligned_samples.append(aligned_sample)
        return aligned_samples

    def _align_sample(self, model_sample: dict[str, Any]) -> dict[str, Any] | None:
        """Pad one encoded sample to a complete-block multiple for DSA compression.

        Padding appends ignored tokens at the document end: ``labels`` receive
        ``IGNORE_INDEX`` so the loss skips them, later documents keep causal
        isolation, and real tokens keep their relative positions. A sample
        whose aligned length exceeds ``max_seq_len`` is dropped with a warning,
        matching the Omni/VLM alignment policy.

        Args:
            model_sample: One encoded sample with ``input_ids`` and ``labels``.

        Returns:
            The same sample, a padded copy, or None when the aligned length
            exceeds ``max_seq_len``.
        """
        if self.sample_alignment_size <= 1:
            return model_sample

        sequence_length = model_sample["input_ids"].numel()
        pad_length = (-sequence_length) % self.sample_alignment_size
        if not pad_length:
            return model_sample

        if sequence_length + pad_length > self.max_seq_len:
            logger.warning(
                "Skipping DeepSeek-V4.1 sample: aligned_length=%d exceeds max_seq_len=%d",
                sequence_length + pad_length, self.max_seq_len,
            )
            return None

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Sample alignment requires a tokenizer pad_token_id or eos_token_id")

        padded = dict(model_sample)
        padded["input_ids"] = torch.nn.functional.pad(
            model_sample["input_ids"], (0, pad_length), value=pad_token_id,
        )
        if "labels" in model_sample:
            padded["labels"] = torch.nn.functional.pad(
                model_sample["labels"], (0, pad_length), value=IGNORE_INDEX,
            )
        return padded


def build_deepseek_v41_text_transform(
        data_type: str,
        *,
        tokenizer: Any = None,
        max_seq_len: int,
        text_keys: str | Sequence[str] | None = None,
        drop_ragged_tail: bool = False,
        sample_alignment_size: int = 1,
) -> Callable[[Any], Any]:
    """Build the model-owned V4.1 text transform.

    Args:
        data_type: Text dataset format; plaintext pretraining data is
            supported first, conversation formats raise for now.
        tokenizer: Tokenizer built by the LLM Trainer from model assets.
        max_seq_len: Maximum model sequence length.
        text_keys: Source field(s); defaults to text.
        drop_ragged_tail: Drop trailing partial plaintext windows.
        sample_alignment_size: Pad every emitted sample to a multiple of this
            value (use the lcm of the model's active compress ratios) so DSA
            compression blocks never cross a document boundary.

    Returns:
        The configured V4.1 text sample transform.

    Raises:
        ValueError: If ``data_type`` is not plaintext or the configuration is
            invalid.
    """
    if text_keys is None:
        text_keys = "text"
    if data_type != "plaintext":
        raise ValueError(
            f"DeepSeek-V4.1 text transform currently supports plaintext pretraining data, got {data_type!r}"
        )

    data_transform = DeepseekV41TextTransform(
        tokenizer, max_seq_len, text_keys, drop_ragged_tail,
        sample_alignment_size=sample_alignment_size,
    )
    logger.debug(
        "Built DeepSeek-V4.1 text transform: alignment=%d", sample_alignment_size,
    )
    return data_transform

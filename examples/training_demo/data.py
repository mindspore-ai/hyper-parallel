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
"""WikiText data components used only by the Trainer end-to-end demo."""

from collections.abc import Mapping, Sequence
from typing import Any


class PlainTextDataTransform:
    """Tokenize and group batched plain text into fixed-length LM samples."""

    def __init__(
        self,
        tokenizer: Any,
        text_column: str = "text",
        seq_len: int = 128,
        append_eos: bool = True,
    ) -> None:
        """Initialize the plain-text transform.

        Args:
            tokenizer: Hugging Face tokenizer used to encode each document.
            text_column: Dataset column containing source text.
            seq_len: Number of tokens in every output training sample.
            append_eos: Whether to append an EOS token after every document.

        Raises:
            ValueError: If the column name or sequence length is invalid, or
                the tokenizer has no EOS token while EOS appending is enabled.
        """
        if not isinstance(text_column, str) or not text_column.strip():
            raise ValueError("text_column must be a non-empty string")
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if append_eos and eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is required when append_eos is enabled")

        self.tokenizer = tokenizer
        self.text_column = text_column
        self.seq_len = seq_len
        self.append_eos = append_eos
        self.eos_token_id = eos_token_id

    def __call__(self, examples: Mapping[str, Sequence[str]]) -> dict[str, list[list[int]]]:
        """Convert a batched dataset mapping into fixed-length causal-LM rows.

        Args:
            examples: Batched examples containing the configured text column.

        Returns:
            Batched ``input_ids``, ``attention_mask``, and ``labels`` rows.

        Raises:
            ValueError: If the configured text column is missing or contains a
                non-string value.
        """
        texts = examples.get(self.text_column)
        if texts is None:
            raise ValueError(f"Dataset batch is missing text column {self.text_column!r}")

        all_token_ids = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Dataset column {self.text_column!r} must contain strings, "
                    f"but item {index} has type {type(text).__name__}"
                )
            if not text.strip():
                continue
            token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
            if self.append_eos:
                token_ids.append(self.eos_token_id)
            all_token_ids.extend(token_ids)

        usable_length = len(all_token_ids) - len(all_token_ids) % self.seq_len
        input_ids = [
            all_token_ids[start:start + self.seq_len]
            for start in range(0, usable_length, self.seq_len)
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * self.seq_len for _ in input_ids],
            "labels": [list(token_ids) for token_ids in input_ids],
        }


def load_tokenized_text_dataset(
    path: str,
    name: str,
    split: str,
    transform: PlainTextDataTransform,
) -> Any:
    """Load a Hugging Face text dataset and lazily cache tokenized rows.

    Args:
        path: Hugging Face dataset repository ID.
        name: Dataset subset/configuration name.
        split: Dataset split to load.
        transform: Batched tokenizer and fixed-length grouping transform.

    Returns:
        A map-style Hugging Face dataset containing model-ready rows.

    Raises:
        ValueError: If ``transform`` is not callable.
    """
    if not callable(transform):
        raise ValueError("transform must be callable")

    # datasets is an optional dependency used only by this training demo.
    from datasets import load_dataset  # pylint: disable=C0415

    dataset = load_dataset(path, name=name, split=split)
    tokenized_dataset = dataset.map(
        transform,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {path}/{name}:{split}",
    )
    return tokenized_dataset.with_format("torch")


__all__ = ["PlainTextDataTransform", "load_tokenized_text_dataset"]

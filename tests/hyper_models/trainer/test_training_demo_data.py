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
"""Tests for the WikiText components used by the Trainer demo."""

import sys
from types import SimpleNamespace

import pytest

from examples.training_demo.data import PlainTextDataTransform, load_tokenized_text_dataset


class _Tokenizer:
    """Small deterministic tokenizer used by data-transform tests."""

    eos_token_id = 99

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode whitespace-separated integer tokens."""
        if add_special_tokens:
            raise ValueError("The test tokenizer does not add special tokens")
        return [int(token) for token in text.split()]


class _Dataset:
    """Record the map call made by the demo dataset adapter."""

    column_names = ["text"]

    def __init__(self) -> None:
        self.map_kwargs = None
        self.mapped_result = None
        self.format_type = None

    def map(self, transform, **kwargs):
        """Record and execute the configured batched transform."""
        self.map_kwargs = kwargs
        self.mapped_result = transform({"text": ["1 2 3", "4 5 6"]})
        return self

    def with_format(self, format_type: str):
        """Record the requested sample format and return this dataset."""
        self.format_type = format_type
        return self


def test_plain_text_transform_filters_empty_text_and_groups_fixed_length_rows() -> None:
    """Concatenate documents with EOS separators and drop an incomplete tail."""
    transform = PlainTextDataTransform(_Tokenizer(), seq_len=4)

    result = transform({"text": ["1 2", "", "3 4 5 6"]})

    assert result == {
        "input_ids": [[1, 2, 99, 3], [4, 5, 6, 99]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
        "labels": [[1, 2, 99, 3], [4, 5, 6, 99]],
    }


def test_plain_text_transform_validates_required_eos_token() -> None:
    """Reject EOS appending when the tokenizer has no EOS token ID."""
    tokenizer = SimpleNamespace(eos_token_id=None)

    with pytest.raises(ValueError, match="eos_token_id"):
        PlainTextDataTransform(tokenizer, append_eos=True)


def test_dataset_adapter_loads_named_split_and_maps_transform(monkeypatch) -> None:
    """Delegate dataset loading and apply the transform in batched map mode."""
    dataset = _Dataset()
    calls = []

    def _load_dataset(path: str, *, name: str, split: str):
        calls.append((path, name, split))
        return dataset

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=_load_dataset),
    )
    transform = PlainTextDataTransform(_Tokenizer(), seq_len=4)

    result = load_tokenized_text_dataset(
        path="Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        split="train",
        transform=transform,
    )

    assert calls == [("Salesforce/wikitext", "wikitext-2-raw-v1", "train")]
    assert dataset.map_kwargs["batched"] is True
    assert dataset.map_kwargs["remove_columns"] == ["text"]
    assert dataset.format_type == "torch"
    assert dataset.mapped_result["input_ids"] == [[1, 2, 3, 99], [4, 5, 6, 99]]
    assert result is dataset

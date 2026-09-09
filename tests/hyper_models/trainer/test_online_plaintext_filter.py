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
"""Regression tests for zero-target Online plaintext samples."""

import pytest
import torch
from torch.utils.data import IterableDataset

from hyper_parallel.auto_models.components.datasets.llm.build_data_transform import PlaintextTransform
from hyper_parallel.auto_models.components.datasets.llm.transform_dataset import apply_llm_data_transform
from hyper_parallel.auto_models.components.datasets.contracts import is_iterable_dataset


class _Tokenizer:
    """Whitespace tokenizer with an EOS token."""

    eos_token_id = 99

    def __init__(self) -> None:
        self.calls = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode integer tokens without adding tokenizer-owned specials."""
        if add_special_tokens:
            raise ValueError("Special tokens are not expected")
        self.calls.append(text)
        return [int(token) for token in text.split()]


class _RawIterableDataset:
    """Yield raw records without implementing map-style access."""

    def __iter__(self):
        yield {"text": "1 2"}


class _WikiTextIterableDataset:
    """Yield empty and non-empty WikiText-style records."""

    def __iter__(self):
        yield {"text": ""}
        yield {"text": " "}
        yield {"text": "1 2"}


class _HuggingFaceLikeIterableDataset(_WikiTextIterableDataset):
    """Expose a Hugging Face-style map method that filtering must bypass."""

    def map(self, transform, **kwargs):
        del transform, kwargs
        raise ValueError("Online invalid-sample filtering must use one transform pass")


def test_iterable_transform_wrapper_uses_pytorch_dataset_contract() -> None:
    """Expose the wrapper as iterable to PyTorch and Trainer dataset logic."""
    dataset = apply_llm_data_transform(
        _RawIterableDataset(),
        PlaintextTransform(_Tokenizer(), max_seq_len=8),
    )

    assert isinstance(dataset, IterableDataset)
    assert is_iterable_dataset(dataset)
    assert next(iter(dataset))["input_ids"].tolist() == [1, 2, 99]


def test_online_iterable_skips_records_without_causal_targets() -> None:
    """Skip empty streaming records without applying the tokenizer twice."""
    tokenizer = _Tokenizer()
    dataset = apply_llm_data_transform(
        _WikiTextIterableDataset(),
        PlaintextTransform(tokenizer, max_seq_len=8),
        skip_invalid_samples=True,
    )

    samples = list(dataset)

    assert tokenizer.calls == ["", " ", "1 2"]
    assert len(samples) == 1
    assert samples[0]["input_ids"].tolist() == [1, 2, 99]


def test_online_huggingface_iterable_filters_in_one_transform_pass() -> None:
    """Bypass Hugging Face map so filtering reuses each transformed result."""
    tokenizer = _Tokenizer()
    dataset = apply_llm_data_transform(
        _HuggingFaceLikeIterableDataset(),
        PlaintextTransform(tokenizer, max_seq_len=8),
        skip_invalid_samples=True,
    )

    assert [sample["input_ids"].tolist() for sample in dataset] == [[1, 2, 99]]
    assert tokenizer.calls == ["", " ", "1 2"]


def test_online_mapping_skips_empty_records_once_per_candidate() -> None:
    """Tokenize each candidate once and return the first trainable result."""
    source_dataset = [
        {"text": ""},
        {"text": "   "},
        {"text": "1"},
        {"text": "2 3"},
    ]
    tokenizer = _Tokenizer()
    transform = PlaintextTransform(tokenizer, max_seq_len=8)

    dataset = apply_llm_data_transform(
        source_dataset,
        transform,
        skip_invalid_samples=True,
    )

    assert len(dataset) == 4
    first_sample = dataset[0]
    assert first_sample["input_ids"].tolist() == [1, 99]
    assert tokenizer.calls == ["", "   ", "1"]
    shifted_labels = first_sample["labels"][1:]
    assert shifted_labels.numel() == 1
    assert torch.isfinite(shifted_labels.float()).all()


def test_plaintext_transform_preserves_single_token_tail_chunk() -> None:
    """Keep the transform behavior unchanged outside the mapping wrapper."""
    transform = PlaintextTransform(_Tokenizer(), max_seq_len=3)

    samples = transform({"text": "1 2 3"})

    assert len(samples) == 2
    assert samples[0]["input_ids"].tolist() == [1, 2, 3]
    assert samples[1]["input_ids"].tolist() == [99]


def test_plaintext_transform_preserves_eos_only_empty_text() -> None:
    """Keep VeOmni-style EOS representation outside the mapping wrapper."""
    transform = PlaintextTransform(_Tokenizer(), max_seq_len=8)

    samples = transform({"text": ""})

    assert len(samples) == 1
    assert samples[0]["input_ids"].tolist() == [99]


def test_online_mapping_raises_when_every_record_is_invalid() -> None:
    """Stop after one full scan instead of retrying invalid records forever."""
    transform = PlaintextTransform(_Tokenizer(), max_seq_len=8)
    dataset = apply_llm_data_transform(
        [{"text": ""}, {"text": " "}],
        transform,
        skip_invalid_samples=True,
    )

    with pytest.raises(ValueError, match="no samples with trainable labels"):
        dataset[0]


def test_default_mapping_path_preserves_strict_source_index() -> None:
    """Keep Offline and non-Online mapping behavior unchanged by default."""
    transform = PlaintextTransform(_Tokenizer(), max_seq_len=8)
    dataset = apply_llm_data_transform([{"text": ""}, {"text": "1"}], transform)

    assert dataset[0]["input_ids"].tolist() == [99]

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
"""Unit tests for the Online plaintext Dataset pipeline."""

import sys
from types import SimpleNamespace

import pytest
import torch

from hyper_parallel.auto_models.components.datasets.llm.build_data_transform import PlaintextTransform
from hyper_parallel.auto_models.components.datasets.llm.build_dataset import build_llm_dataset


class _Tokenizer:
    """Small tokenizer exposing the Hugging Face encode contract."""

    eos_token_id = 9

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode whitespace-delimited integers."""
        if add_special_tokens:
            raise ValueError("Special tokens are not expected")
        self.calls += 1
        return [int(token) for token in text.split()]


def test_online_mapping_plaintext_matches_direct_tokenization(monkeypatch) -> None:
    """Run the real Online builder and compare it with direct tokenizer output."""
    raw_dataset = [{"text": "1 2 3"}]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            disable_progress_bars=lambda: None,
            enable_progress_bars=lambda: None,
            load_dataset=lambda *args, **kwargs: raw_dataset,
        ),
    )
    tokenizer = _Tokenizer()
    dataset = build_llm_dataset(
        data_config={
            "source_type": "online",
            "dataset_type": "mapping",
            "hf_dataset_name": "test/plaintext",
            "namespace": "train",
        },
        transform=PlaintextTransform(tokenizer, max_seq_len=8),
    )

    assert tokenizer.calls == 0
    sample = dataset[0]
    expected_ids = [1, 2, 3, tokenizer.eos_token_id]
    assert tokenizer.calls == 1
    assert sample["input_ids"].tolist() == expected_ids
    assert sample["attention_mask"].tolist() == [1] * len(expected_ids)
    assert sample["labels"].tolist() == expected_ids
    assert sample["input_ids"].dtype == torch.long


def test_online_mapping_transform_is_lazy(monkeypatch) -> None:
    """Do not tokenize Online mapping records until ``__getitem__`` is called."""
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            disable_progress_bars=lambda: None,
            enable_progress_bars=lambda: None,
            load_dataset=lambda *args, **kwargs: [{"text": "4 5"}],
        ),
    )
    tokenizer = _Tokenizer()
    dataset = build_llm_dataset(
        data_config={
            "source_type": "online",
            "dataset_type": "mapping",
            "hf_dataset_name": "test/plaintext",
            "namespace": "train",
        },
        transform=PlaintextTransform(tokenizer, max_seq_len=8),
    )

    assert tokenizer.calls == 0
    assert dataset[0]["input_ids"].tolist() == [4, 5, 9]
    assert tokenizer.calls == 1


def test_online_iterable_hub_source_does_not_require_data_path(monkeypatch) -> None:
    """Allow a streaming Hub Dataset to use only ``hf_dataset_name``."""
    captured = {}

    def _build_online_dataset(**kwargs):
        captured.update(kwargs)
        return iter([{"input_ids": [1, 2], "labels": [1, 2]}])

    monkeypatch.setattr(
        "hyper_parallel.auto_models.components.datasets.llm.build_dataset.build_online_dataset",
        _build_online_dataset,
    )
    dataset = build_llm_dataset(
        data_config={
            "source_type": "online",
            "dataset_type": "iterable",
            "hf_dataset_name": "Salesforce/wikitext",
        },
        transform=None,
    )

    assert captured["data_path"] is None
    assert next(iter(dataset))["input_ids"] == [1, 2]


def test_online_local_source_requires_data_path() -> None:
    """Require data_path only when an Online Hub name is absent."""
    with pytest.raises(ValueError, match="data_path is required"):
        build_llm_dataset(
            data_config={"source_type": "online", "dataset_type": "mapping"},
            transform=None,
        )


def test_offline_source_requires_data_path() -> None:
    """Keep indexed Offline Dataset paths mandatory."""
    with pytest.raises(ValueError, match="Offline LLM Datasets require data_path"):
        build_llm_dataset(
            data_config={"source_type": "offline"},
            transform=None,
            train_valid_test_num_samples=(1, 0, 0),
        )

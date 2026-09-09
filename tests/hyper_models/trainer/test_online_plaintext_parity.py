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

from hyper_parallel.auto_models.components.datasets.llm.build_data_transform import (
    PlaintextTransform,
    build_llm_data_transform,
)
from hyper_parallel.auto_models.components.datasets.llm.build_dataset import build_online_text_dataset


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
    dataset = build_online_text_dataset(
        data_config={
            "source_type": "online",
            "dataset_type": "mapping",
            "hf_dataset_name": "test/plaintext",
        },
        transform=PlaintextTransform(tokenizer, max_seq_len=8),
    )

    assert tokenizer.calls == 0
    sample = dataset[0]
    expected_ids = [1, 2, 3, tokenizer.eos_token_id]
    assert tokenizer.calls == 1
    assert sample["input_ids"].tolist() == expected_ids
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
    dataset = build_online_text_dataset(
        data_config={
            "source_type": "online",
            "dataset_type": "mapping",
            "hf_dataset_name": "test/plaintext",
        },
        transform=PlaintextTransform(tokenizer, max_seq_len=8),
    )

    assert tokenizer.calls == 0
    assert dataset[0]["input_ids"].tolist() == [4, 5, 9]
    assert tokenizer.calls == 1


@pytest.mark.parametrize(
    ("sample", "template", "expected"),
    [
        (
            {"instruction": "1", "input": "2", "output": "3"},
            "{instruction} {input} {output}",
            [1, 2, 3, 9],
        ),
        (
            {"instruction": "1", "output": "3"},
            "{instruction} {output}",
            [1, 3, 9],
        ),
    ],
)
def test_plaintext_transform_renders_instruction_records(sample, template, expected) -> None:
    """Render Alpaca-like records before using the normal tokenizer path."""
    transform = build_llm_data_transform(
        "plaintext",
        tokenizer=_Tokenizer(),
        max_seq_len=16,
        text_template=template,
    )

    assert transform(sample)[0]["input_ids"].tolist() == expected


def test_plaintext_transform_rejects_missing_template_fields() -> None:
    """Report the missing source field instead of tokenizing an invalid record."""
    transform = PlaintextTransform(
        _Tokenizer(),
        max_seq_len=16,
        text_template="{instruction} {input} {output}",
    )

    with pytest.raises(ValueError, match="text_template field 'input'"):
        transform({"instruction": "1", "output": "3"})

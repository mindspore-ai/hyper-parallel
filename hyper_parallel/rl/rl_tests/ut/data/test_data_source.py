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
"""CPU unit tests for prompt loading, tokenization, collation, and records."""

from pathlib import Path
from typing import Any

import pytest

from rl.dataset import data_source
from rl.dataset.data_source import (
    PromptDataset,
    build_padded_evaluation_batches,
    build_prompt_records,
    collate_prompt_samples,
)


class _Tokenizer:
    """Encode prompt words into deterministic increasing token IDs."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, list[int]]:
        """Return one token per normalized prompt word."""
        assert kwargs["add_generation_prompt"]
        length = len(messages[0]["content"].split())
        tokens = list(range(1, length + 1))
        return {"input_ids": tokens, "attention_mask": [1] * length}


def test_prompt_loading_and_normalization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raw and structured prompts normalize once with deterministic answer priority."""
    instruction = 'Return the answer after "####".'
    parquet_path = tmp_path / "prompts.parquet"
    parquet_path.touch()
    frame = data_source.pd.DataFrame(
        [
            {
                "prompt": "What is two plus two?",
                "extra_info": {"answer": "#### 4"},
                "reward_model": {"ground_truth": "999"},
            },
            {
                "prompt": [{"role": "user", "content": f"What is three plus three? {instruction}"}],
                "extra_info": {"answer": "$6"},
                "reward_model": {"ground_truth": "999"},
            },
        ]
    )
    monkeypatch.setattr(data_source.pd, "read_parquet", lambda _path: frame)

    dataset = PromptDataset(
        str(parquet_path),
        _Tokenizer(),
        max_prompt_length=32,
        answer_column="extra_info",
        prompt_instruction=instruction,
    )
    first = dataset[0]
    second = dataset[1]

    assert first["source_prompt"] == "What is two plus two?"
    assert first["prompt"] == f"What is two plus two? {instruction}"
    assert first["ground_truth"] == "4"
    assert second["source_prompt"] == f"What is three plus three? {instruction}"
    assert second["prompt"].count(instruction) == 1
    assert second["ground_truth"] == "6"
    inferred = data_source._select_answer_source(  # pylint: disable=protected-access
        {"answer": "column", "reward_model": {"ground_truth": "reward"}},
        "answer",
        0,
        answer_column_is_explicit=False,
    )
    assert inferred == "reward"


def test_tokenize_padding_collate_and_prompt_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Samples left-pad in order while PromptRecord stores only valid prompt tokens."""
    parquet_path = tmp_path / "prompts.parquet"
    parquet_path.touch()
    frame = data_source.pd.DataFrame(
        [
            {"prompt": "short prompt", "answer": "1"},
            {"prompt": "a somewhat longer prompt", "answer": "2"},
        ]
    )
    monkeypatch.setattr(data_source.pd, "read_parquet", lambda _path: frame)
    dataset = PromptDataset(str(parquet_path), _Tokenizer(), max_prompt_length=16)

    batch = collate_prompt_samples([dataset[0], dataset[1]], pad_token_id=0)
    records = build_prompt_records(batch, batch["input_ids"], batch["attention_mask"])

    assert batch["input_ids"].tolist() == [[0, 0, 1, 2], [1, 2, 3, 4]]
    assert batch["attention_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert batch["sample_indices"] == [0, 1]
    assert [record.prompt_id for record in records] == ["0", "1"]
    assert [record.messages[0].content for record in records] == ["short prompt", "a somewhat longer prompt"]
    assert [record.ground_truth for record in records] == ["1", "2"]
    assert records[0].metadata["input_ids"].tolist() == [1, 2]
    assert records[1].metadata["input_ids"].tolist() == [1, 2, 3, 4]


def test_evaluation_partition_pads_only_rows_beyond_the_global_sample_limit() -> None:
    """Distributed evaluation emits equal local batches with explicit padding flags."""
    rank_zero = build_padded_evaluation_batches(10, 2, 0, batch_size=2, max_samples=7)
    rank_one = build_padded_evaluation_batches(10, 2, 1, batch_size=2, max_samples=7)

    assert rank_zero == [[(0, True), (1, True)], [(4, True), (5, True)]]
    assert rank_one == [[(2, True), (3, True)], [(6, True), (0, False)]]

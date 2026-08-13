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
"""CPU unit tests for prompt parquet conversion and DP partitioning."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import torch
from torch.utils.data import DistributedSampler

from rl.dataset import (
    PROMPT_INSTRUCTION,
    PromptDataset,
    build_padded_evaluation_batches,
    collate_prompt_samples,
)


class FakeTokenizer:
    """Deterministic tokenizer stub used by dataset unit tests."""

    def __init__(self) -> None:
        """Initialize an empty chat-template call record."""
        self.chat_template_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        add_generation_prompt: bool,
        tokenize: bool,
        truncation: bool,
        max_length: int,
        return_dict: bool,
        tokenizer_kwargs: dict[str, Any],
    ) -> Dict[str, Any]:
        """Encode a chat-formatted user message as stable token IDs."""
        self.chat_template_calls.append(
            {
                "conversation": conversation,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "truncation": truncation,
                "max_length": max_length,
                "return_dict": return_dict,
                "tokenizer_kwargs": tokenizer_kwargs,
            }
        )
        text = conversation[0]["content"]
        token_count = min(len(text.split()), max_length)
        input_ids = list(range(1, token_count + 1))
        return {"input_ids": input_ids, "attention_mask": [1] * token_count}


def _write_prompt_parquet(path: Path, rows: int = 4) -> None:
    """Create a tiny local prompt/answer parquet fixture."""
    frame = pd.DataFrame(
        {
            "question": [f"Question {index}?" for index in range(rows)],
            "answer": [f"Reasoning {index}\n#### {index:,}" for index in range(rows)],
        }
    )
    frame.to_parquet(path)


def test_dataset_converts_prompt_answer_and_index(tmp_path: Path) -> None:
    """Verify parquet fields are converted into the trainer contract."""
    parquet_path = tmp_path / "train.parquet"
    _write_prompt_parquet(parquet_path)
    tokenizer = FakeTokenizer()
    dataset = PromptDataset(str(parquet_path), tokenizer, max_prompt_length=128)
    sample = dataset[2]
    assert sample["sample_index"] == 2, f"Unexpected index: expected=2, got={sample['sample_index']}"
    assert sample["ground_truth"] == "2", (
        f"Unexpected ground truth: expected=2, got={sample['ground_truth']}"
    )
    assert sample["prompt"].endswith(PROMPT_INSTRUCTION), (
        f"Prompt instruction missing: expected_suffix={PROMPT_INSTRUCTION}, got={sample['prompt']}"
    )
    assert sample["input_ids"].dtype == torch.long, (
        f"Unexpected input dtype: expected={torch.long}, got={sample['input_ids'].dtype}"
    )
    expected_calls = 1
    assert len(tokenizer.chat_template_calls) == expected_calls, (
        f"Unexpected chat-template call count: expected={expected_calls}, "
        f"got={len(tokenizer.chat_template_calls)}"
    )
    call = tokenizer.chat_template_calls[0]
    assert call["conversation"] == [{"role": "user", "content": sample["prompt"]}]
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is True
    assert call["truncation"] is True
    assert call["max_length"] == 128
    assert call["return_dict"] is True
    assert call["tokenizer_kwargs"] == {"return_attention_mask": True}


def test_dataset_supports_configurable_parquet_columns(tmp_path: Path) -> None:
    """Verify the generic dataset is not tied to question/answer column names."""
    parquet_path = tmp_path / "custom.parquet"
    pd.DataFrame(
        {
            "input_text": ["Custom prompt?"],
            "expected_text": ["Reasoning\n#### 7"],
        }
    ).to_parquet(parquet_path)
    dataset = PromptDataset(
        str(parquet_path),
        FakeTokenizer(),
        max_prompt_length=128,
        prompt_column="input_text",
        answer_column="expected_text",
    )
    sample = dataset[0]
    assert sample["source_prompt"] == "Custom prompt?"
    assert sample["ground_truth"] == "7"


def test_dataset_supports_historical_gsm8k_chat_schema(tmp_path: Path) -> None:
    """The retained parity dataset should preserve its existing chat instruction."""
    parquet_path = tmp_path / "historical.parquet"
    historical_prompt = f"Historical question? {PROMPT_INSTRUCTION}"
    pd.DataFrame(
        {
            "prompt": [[{"role": "user", "content": historical_prompt}]],
            "extra_info": [{"answer": "Reasoning\n#### 42", "split": "train", "index": 0}],
        }
    ).to_parquet(parquet_path)
    dataset = PromptDataset(
        str(parquet_path),
        FakeTokenizer(),
        max_prompt_length=128,
        prompt_column="prompt",
        answer_column="extra_info",
    )

    sample = dataset[0]

    assert sample["source_prompt"] == historical_prompt
    assert sample["prompt"] == historical_prompt
    assert sample["prompt"].count(PROMPT_INSTRUCTION) == 1
    assert sample["ground_truth"] == "42"


def test_dataset_can_limit_training_prefix_for_smoke(tmp_path: Path) -> None:
    """A smoke config should avoid exposing unused rows to the sampler."""
    parquet_path = tmp_path / "train.parquet"
    _write_prompt_parquet(parquet_path, rows=5)

    dataset = PromptDataset(
        str(parquet_path),
        FakeTokenizer(),
        max_prompt_length=128,
        max_samples=2,
    )

    assert len(dataset) == 2


def test_collate_left_pads_prompts(tmp_path: Path) -> None:
    """Verify collate produces aligned left-padded tensors and metadata lists."""
    parquet_path = tmp_path / "train.parquet"
    _write_prompt_parquet(parquet_path, rows=2)
    dataset = PromptDataset(str(parquet_path), FakeTokenizer(), max_prompt_length=128)
    first = dataset[0]
    second = dict(dataset[1])
    second["input_ids"] = torch.tensor([8, 9], dtype=torch.long)
    second["attention_mask"] = torch.ones(2, dtype=torch.long)
    batch = collate_prompt_samples([first, second], pad_token_id=0)
    expected_shape = (2, first["input_ids"].numel())
    assert tuple(batch["input_ids"].shape) == expected_shape, (
        f"Unexpected batch shape: expected={expected_shape}, got={tuple(batch['input_ids'].shape)}"
    )
    expected_padding = first["input_ids"].numel() - 2
    actual_padding = int((batch["attention_mask"][1] == 0).sum())
    assert actual_padding == expected_padding, (
        f"Unexpected left padding: expected={expected_padding}, got={actual_padding}"
    )


def test_distributed_sampler_assigns_different_prompts(tmp_path: Path) -> None:
    """Verify FSDP data-parallel ranks receive disjoint prompt indices."""
    parquet_path = tmp_path / "train.parquet"
    _write_prompt_parquet(parquet_path, rows=6)
    dataset = PromptDataset(str(parquet_path), FakeTokenizer(), max_prompt_length=128)
    rank_zero = list(DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False, drop_last=True))
    rank_one = list(DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False, drop_last=True))
    overlap = set(rank_zero) & set(rank_one)
    assert not overlap, f"DP prompt partitions must be disjoint, overlap={sorted(overlap)}"
    combined = sorted(rank_zero + rank_one)
    expected = list(range(6))
    assert combined == expected, f"Unexpected DP partition union: expected={expected}, got={combined}"


def test_evaluation_partition_pads_collectives_without_counting_duplicates() -> None:
    """Verify every rank runs equal shapes while valid test rows remain unique."""
    rank_zero = build_padded_evaluation_batches(
        dataset_size=11,
        num_replicas=2,
        rank=0,
        batch_size=3,
    )
    rank_one = build_padded_evaluation_batches(
        dataset_size=11,
        num_replicas=2,
        rank=1,
        batch_size=3,
    )
    assert len(rank_zero) == len(rank_one) == 2
    assert all(len(batch) == 3 for batch in rank_zero + rank_one)
    valid_indices = sorted(
        sample_index
        for batches in (rank_zero, rank_one)
        for batch in batches
        for sample_index, is_valid in batch
        if is_valid
    )
    assert valid_indices == list(range(11))
    padded_count = sum(
        not is_valid
        for batches in (rank_zero, rank_one)
        for batch in batches
        for _, is_valid in batch
    )
    assert padded_count == 1


def test_evaluation_partition_honors_max_samples() -> None:
    """Verify bounded validation uses a stable prefix of the test split."""
    partitions = [
        build_padded_evaluation_batches(
            dataset_size=20,
            num_replicas=2,
            rank=rank,
            batch_size=2,
            max_samples=5,
        )
        for rank in range(2)
    ]
    valid_indices = sorted(
        sample_index
        for batches in partitions
        for batch in batches
        for sample_index, is_valid in batch
        if is_valid
    )
    assert valid_indices == list(range(5))


def test_dataset_rejects_missing_schema(tmp_path: Path) -> None:
    """Verify malformed parquet input fails at the data boundary."""
    parquet_path = tmp_path / "bad.parquet"
    pd.DataFrame({"question": ["Missing answer"]}).to_parquet(parquet_path)
    with pytest.raises(ValueError, match="missing required columns"):
        PromptDataset(str(parquet_path), FakeTokenizer(), max_prompt_length=128)

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
"""Tests for prompt data-source normalization."""
# Tests intentionally exercise prompt normalization as a module-internal contract.
# pylint: disable=protected-access

from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.dataset import data_source
from rl.dataset import PROMPT_INSTRUCTION
from rl.dataset.data_source import PromptDataset, _select_answer_source


def test_prompt_instruction_remains_a_compatible_public_export() -> None:
    """The package-level GSM8K instruction export must remain importable."""
    assert 'after "####"' in PROMPT_INSTRUCTION


def test_prompt_normalization_does_not_inject_task_instruction() -> None:
    """Generic prompt loading must not add GSM8K text to unrelated tasks."""
    source, prompt = data_source._normalize_prompt("Who wrote Hamlet?", index=0)

    assert source == "Who wrote Hamlet?"
    assert prompt == source


def test_structured_prompt_receives_configured_instruction_once() -> None:
    """Configured instructions apply equally to structured dataset prompts."""
    instruction = 'Output the final answer after "####".'
    source = [{"role": "user", "content": "What is 1 + 1?"}]

    source_prompt, prompt = data_source._normalize_prompt(
        source,
        index=0,
        prompt_instruction=instruction,
    )
    _, repeated = data_source._normalize_prompt(
        [{"role": "user", "content": prompt}],
        index=0,
        prompt_instruction=instruction,
    )

    assert source_prompt == "What is 1 + 1?"
    assert prompt == f"{source_prompt} {instruction}"
    assert repeated == prompt


def test_explicit_answer_column_takes_precedence_over_reward_model() -> None:
    """An explicitly selected answer column must remain authoritative."""
    record = {
        "extra_info": {"answer": "configured-answer"},
        "reward_model": {"ground_truth": "inferred-answer"},
    }

    answer_source = _select_answer_source(
        record,
        "extra_info",
        index=0,
        answer_column_is_explicit=True,
    )

    assert answer_source == {"answer": "configured-answer"}


def test_reward_model_is_used_when_answer_column_is_not_configured() -> None:
    """Megatron/verl records use reward_model as the automatic fallback."""
    record = {
        "extra_info": {"answer": "inferred-column-answer"},
        "reward_model": {"ground_truth": "reward-model-answer"},
    }

    answer_source = _select_answer_source(
        record,
        "extra_info",
        index=0,
        answer_column_is_explicit=False,
    )

    assert answer_source == "reward-model-answer"


def test_inferred_answer_column_is_used_without_reward_model() -> None:
    """Hugging Face answer columns remain the final automatic fallback."""
    record = {"answer": "inferred-column-answer"}

    answer_source = _select_answer_source(
        record,
        "answer",
        index=0,
        answer_column_is_explicit=False,
    )

    assert answer_source == "inferred-column-answer"


def test_prompt_dataset_preserves_explicit_answer_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public dataset path must preserve an explicitly configured answer."""
    parquet_path = tmp_path / "prompts.parquet"
    parquet_path.touch()
    frame = data_source.pd.DataFrame(
        [
            {
                "prompt": "What is 1 + 1?",
                "extra_info": {"answer": "configured-answer"},
                "reward_model": {"ground_truth": "inferred-answer"},
            }
        ]
    )
    monkeypatch.setattr(data_source.pd, "read_parquet", lambda _path: frame)
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *_args, **_kwargs: {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
        }
    )

    dataset = PromptDataset(
        str(parquet_path),
        tokenizer,
        max_prompt_length=16,
        answer_column="extra_info",
    )

    assert dataset[0]["ground_truth"] == "configured-answer"

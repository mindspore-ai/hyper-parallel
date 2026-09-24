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
"""Structured task data contracts for the single-turn code integration."""

from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
import yaml

import rl

import rl.config as config_module

from rl.dataset.contracts import Message, PromptRecord
from rl.dataset.data_source import (
    PromptDataset, build_prompt_records, collate_prompt_samples, validate_row_adapter_config,
)




class _Tokenizer:
    """Capture chat inputs and assign one deterministic token per word."""

    def __init__(self) -> None:
        """Initialize capture fields without loading a model tokenizer."""
        self.messages = None
        self.truncation = None

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, list[int]]:
        """Record exactly which task data reaches the model."""
        self.messages = messages
        self.truncation = kwargs["truncation"]
        tokens = list(range(1, 1 + sum(len(message["content"].split()) for message in messages)))
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def adapt_row(row: Mapping[str, Any], index: int) -> PromptRecord:
    """Adapt the fixture while requiring recursive conversion of parquet arrays."""
    if not isinstance(row["private_tests"]["inputs"], list):
        raise ValueError("Nested arrays must become plain lists before invoking adapters")
    return PromptRecord(
        prompt_id=row["task_id"],
        messages=tuple(Message(message["role"], message["content"]) for message in row["chat"]),
        ground_truth=row["private_tests"],
        metadata={"language": "python", "source_row": index, "input_ids": "untrusted"},
    )


@pytest.fixture(name="task_data")
def make_task_data(tmp_path: Path) -> tuple[Path, _Tokenizer]:
    """Write a real parquet round-trip with private structured tests."""
    path = tmp_path / "code.parquet"
    pd.DataFrame([{
        "task_id": "code:stable-id",
        "chat": [{"role": "system", "content": "Write Python code"},
                 {"role": "user", "content": "Print the sum"}],
        "private_tests": {"inputs": np.array(["7 9", "5 6"]), "outputs": np.array(["16", "11"])},
    }]).to_parquet(path)
    return path, _Tokenizer()


def test_adapter_preserves_private_tests_messages_and_id(task_data: tuple) -> None:
    """Labels remain structured and isolated from chat through collation and rollout records."""
    path, tokenizer = task_data
    dataset = PromptDataset(str(path), tokenizer, 32, row_adapter=f"{__name__}:adapt_row")
    sample = dataset[0]
    batch = collate_prompt_samples([sample], pad_token_id=0)
    record = build_prompt_records(batch, batch["input_ids"], batch["attention_mask"])[0]
    assert record.prompt_id == "code:stable-id"
    assert record.messages == (Message("system", "Write Python code"), Message("user", "Print the sum"))
    assert record.ground_truth == {"inputs": ["7 9", "5 6"], "outputs": ["16", "11"]}
    assert record.metadata["language"] == "python"
    assert record.metadata["source_row"] == 0
    assert record.metadata["input_ids"].tolist() == list(range(1, 7))
    assert tokenizer.messages == [{"role": item.role, "content": item.content} for item in record.messages]
    assert tokenizer.truncation is False
    assert "7 9" not in sample["prompt"]


def test_adapter_rejects_overlong_prompt_without_truncating(task_data: tuple) -> None:
    """Task semantics must not change silently when the prompt budget is too small."""
    path, tokenizer = task_data
    dataset = PromptDataset(str(path), tokenizer, 3, row_adapter=f"{__name__}:adapt_row")
    with pytest.raises(ValueError, match="exceeds max_prompt_length"):
        _ = dataset[0]
    assert tokenizer.truncation is False


@pytest.mark.parametrize("overrides", [
    {"row_adapter": "invalid"}, {"row_adapter": "module:"}, {"row_adapter": ":function"},
    {"prompt_column": "prompt"}, {"answer_column": "answer"}, {"prompt_instruction": "extra"},
])
def test_adapter_configuration_rejects_ambiguous_inputs(overrides: dict) -> None:
    """The task adapter owns messages and labels independently of text-column inference."""
    with pytest.raises(ValueError, match="data.row_adapter"):
        validate_row_adapter_config({"row_adapter": f"{__name__}:adapt_row", **overrides})


@pytest.mark.parametrize("changes", [
    {"prompt_id": " "}, {"messages": ()}, {"messages": (Message("user", " "),)},
    {"metadata": []},
])
def test_adapter_rejects_invalid_records(task_data: tuple, monkeypatch: pytest.MonkeyPatch, changes: dict) -> None:
    """Malformed adapter output fails before generation starts."""
    original = adapt_row
    monkeypatch.setattr(sys.modules[__name__], "adapt_row", lambda row, index: replace(original(row, index), **changes))
    path, tokenizer = task_data
    dataset = PromptDataset(str(path), tokenizer, 32, row_adapter=f"{__name__}:adapt_row")
    with pytest.raises(ValueError, match="data.row_adapter"):
        _ = dataset[0]


def test_adapter_must_resolve_to_callable(task_data: tuple) -> None:
    """A valid import path still cannot select a non-callable module attribute."""
    path, tokenizer = task_data
    with pytest.raises(ValueError, match="resolve to a callable"):
        PromptDataset(str(path), tokenizer, 32, row_adapter=f"{__name__}:__doc__")


@pytest.mark.parametrize("runner", ["codex", "deepseek"])
def test_external_runner_does_not_require_internal_environment(runner: str) -> None:
    """External harness configuration remains independent of internal environments."""
    config_dir = Path(rl.__file__).resolve().parent.parent / "examples/gsm8k/configs"
    agentic = yaml.safe_load((config_dir / f"{runner}_multi_turn.yaml").read_text())["agentic"]
    agentic.pop("environment")
    agentic.pop("module_path")
    config_module._validate_agentic(agentic)  # pylint: disable=protected-access

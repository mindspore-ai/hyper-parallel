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
"""Unit tests for offline source-record normalization."""

import argparse
import json
from pathlib import Path

import pytest

from hyper_parallel.data.tools import offline_preparation
from hyper_parallel.data.tools.offline_record_transform import (
    OfflineRecordTransform,
    parse_role_map,
)
from tests.common.mark_utils import arg_mark


_CPU_UT_MARK = arg_mark(
    plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _ChatTokenizer:
    """Small tokenizer stub exposing rendered chat messages."""

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        assert not add_generation_prompt
        self.messages = messages
        return "rendered conversation"


@_CPU_UT_MARK
def test_instruction_transform_renders_template() -> None:
    transform = OfflineRecordTransform(
        output_key="text",
        text_template="Instruction: {instruction}\nOutput: {output}",
    )
    actual = transform({"instruction": "classify", "output": "positive", "id": 1}, None)
    assert actual["text"] == "Instruction: classify\nOutput: positive"
    assert actual["id"] == 1


@_CPU_UT_MARK
def test_instruction_transform_reports_missing_field() -> None:
    transform = OfflineRecordTransform(output_key="text", text_template="{instruction} {input}")
    with pytest.raises(ValueError, match="text_template field 'input'"):
        transform({"instruction": "classify"}, None)


@_CPU_UT_MARK
def test_sharegpt_transform_normalizes_roles() -> None:
    tokenizer = _ChatTokenizer()
    transform = OfflineRecordTransform(
        output_key="text",
        conversation_key="conversations",
        role_key="from",
        content_key="value",
    )
    actual = transform(
        {"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "hi"}]},
        tokenizer,
    )
    assert tokenizer.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert actual["text"] == "rendered conversation"


@_CPU_UT_MARK
def test_materialize_normalized_jsonl(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "instruction.jsonl"
    source.write_text(
        json.dumps({"instruction": "classify", "output": "positive"}) + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(output_prefix=str(tmp_path / "offline"))
    transform = OfflineRecordTransform(output_key="text", text_template="{instruction}: {output}")
    monkeypatch.setattr(offline_preparation, "build_tokenizer", lambda _: object())
    normalized = offline_preparation._materialize_normalized_jsonl(args, [str(source)], transform)
    assert Path(normalized).read_text(encoding="utf-8") == '{"text": "classify: positive"}\n'


@pytest.mark.parametrize("value", ["[]", '{"human": 1}', "not-json"])
@_CPU_UT_MARK
def test_parse_role_map_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="role_map"):
        parse_role_map(value)

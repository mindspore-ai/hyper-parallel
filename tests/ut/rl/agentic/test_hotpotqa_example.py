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
"""Unit tests for the HotpotQA Codex Harness Search MVP."""

import json

from examples.search_r1.reward import normalize_answer, score_hotpotqa_answer
from rl.dataset.contracts import Message, PromptRecord


def _prompt() -> PromptRecord:
    """Build one HotpotQA prompt with known evidence."""
    context = {
        "title": ["Arthur's Magazine", "First for Women", "Distractor"],
        "sentences": [
            ["Arthur's Magazine was first published in 1844."],
            ["First for Women began publishing in 1989."],
            ["This unrelated sentence mentions publishing."],
        ],
    }
    supporting = {
        "title": ["Arthur's Magazine", "First for Women"],
        "sent_id": [0, 0],
    }
    return PromptRecord(
        prompt_id="hotpot-0",
        messages=(Message(role="user", content="Which magazine started first?"),),
        ground_truth="Arthur's Magazine",
        metadata={
            "question_id": "hotpot-0",
            "context_json": json.dumps(context),
            "supporting_facts_json": json.dumps(supporting),
        },
    )


def test_hotpotqa_reward_scores_answer_and_supporting_facts() -> None:
    """Exact answer plus both gold sentences receives the full reward."""
    answer = (
        "ANSWER: Arthur's Magazine\n"
        "SOURCES: Arthur's Magazine:0; First for Women:0"
    )
    result = score_hotpotqa_answer(answer, _prompt())

    assert normalize_answer("The Arthur's Magazine.") == "arthurs magazine"
    assert result.value == 1.0
    assert result.components == {
        "answer_f1": 1.0,
        "answer_em": 1.0,
        "support_f1": 1.0,
        "answer_format": 1.0,
    }
    flexible = score_hotpotqa_answer(
        "ANSWER: Arthur's Magazine\nSOURCES: Arthur's Magazine:sent=0; First for Women:sent0",
        _prompt(),
    )
    assert flexible.components["support_f1"] == 1.0

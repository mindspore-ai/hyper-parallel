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
"""Answer and supporting-fact rewards for HotpotQA Agent GRPO."""

import json
import re
import string
from collections import Counter
from typing import Any, Mapping

from rl.agentic.core.types import RewardResult
from rl.dataset.contracts import PromptRecord

_ANSWER = re.compile(r"^ANSWER\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SOURCES = re.compile(r"^SOURCES\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def normalize_answer(value: str) -> str:
    """Apply the article, punctuation, and whitespace normalization used by HotpotQA."""
    lowered = value.casefold()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _answer_f1(prediction: str, expected: str) -> float:
    """Compute token overlap F1 after HotpotQA answer normalization."""
    predicted_tokens = normalize_answer(prediction).split()
    expected_tokens = normalize_answer(expected).split()
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)
    common = Counter(predicted_tokens) & Counter(expected_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _supporting_facts(value: Any) -> set[tuple[str, int]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        return {
            (str(title).casefold(), int(sent_id))
            for title, sent_id in zip(value.get("title", []), value.get("sent_id", []))
        }
    return {
        (str(item[0]).casefold(), int(item[1]))
        for item in value or []
        if isinstance(item, (list, tuple)) and len(item) == 2
    }


def _predicted_sources(answer: str) -> set[tuple[str, int]]:
    match = _SOURCES.search(answer)
    if not match:
        return set()
    sources = set()
    for item in match.group(1).split(";"):
        source_match = re.match(r"^(.*?):(?:sent(?:ence)?\s*=?\s*)?(\d+)\s*$", item.strip(), re.I)
        if source_match:
            sources.add((source_match.group(1).strip().casefold(), int(source_match.group(2))))
    return sources


def _set_f1(predicted: set[Any], expected: set[Any]) -> float:
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if not overlap else 2.0 * precision * recall / (precision + recall)


def score_hotpotqa_answer(answer: str, prompt: PromptRecord) -> RewardResult:
    """Score final answer quality and cited sentence-level supporting facts."""
    answer_match = _ANSWER.search(answer)
    prediction = answer_match.group(1).strip() if answer_match else answer.strip().splitlines()[0]
    expected = str(prompt.ground_truth)
    answer_f1 = _answer_f1(prediction, expected)
    answer_em = float(normalize_answer(prediction) == normalize_answer(expected))
    expected_sources = _supporting_facts(prompt.metadata.get("supporting_facts_json", []))
    predicted_sources = _predicted_sources(answer)
    support_f1 = _set_f1(predicted_sources, expected_sources)
    format_reward = float(answer_match is not None and _SOURCES.search(answer) is not None)
    value = round(
        0.70 * answer_f1 + 0.20 * answer_em + 0.08 * support_f1 + 0.02 * format_reward,
        12,
    )
    return RewardResult(
        value=value,
        components={
            "answer_f1": answer_f1,
            "answer_em": answer_em,
            "support_f1": support_f1,
            "answer_format": format_reward,
        },
        metadata={
            "predicted_answer": prediction,
            "expected_answer": expected,
            "predicted_sources": sorted(predicted_sources),
            "expected_sources": sorted(expected_sources),
        },
    )

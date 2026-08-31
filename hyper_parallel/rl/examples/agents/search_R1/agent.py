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
"""Search-R1 protocol, reward, and multi-turn environment example."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from examples.agents.search_R1.tools import LocalBM25Retriever, build_search_registry
from rl.agentic.core.chat_template import CHAT_TEMPLATE_MESSAGES
from rl.agentic.core.types import (
    Action,
    EpisodeContext,
    InteractionMode,
    Observation,
    ToolCall,
    ToolResult,
    Transition,
    TurnContext,
)
from rl.agentic.envs.environment import ENVIRONMENTS, ToolEnvironment
from rl.agentic.tools import ToolExecutor, ToolRegistry
from rl.agentic.tools.protocol import INTERACTION_PROTOCOLS, ParsedAction
from rl.dataset.contracts import PromptRecord


_SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_DEFAULT_SYSTEM_PROMPT = """You are a retrieval agent. Use the local search tool to answer the user.
To search, output exactly one tag per turn, such as:
<search>concise search query</search>
The environment replies with <information> containing retrieved passages. You may search again.
When enough evidence is available, output only:
<answer>short final answer</answer>
Never invent search results. Do not put a final answer inside a search tag."""


class SearchR1Protocol:
    """Parse Search-R1 tags and format retrieval observations."""

    def parse_action(self, action: Action, context: TurnContext) -> ParsedAction:
        """Interpret search tags as tool calls and an answer tag as terminal."""
        answers = _ANSWER_PATTERN.findall(action.content)
        searches = _SEARCH_PATTERN.findall(action.content)
        if answers and searches:
            raise ValueError("Agent action cannot contain search and answer tags together")
        if len(answers) > 1:
            raise ValueError("Agent action must contain at most one answer tag")
        if answers:
            answer = answers[0].strip()
            if not answer:
                raise ValueError("answer tag must contain non-empty text")
            return ParsedAction(final_answer=answer)
        calls = []
        for index, query in enumerate(searches):
            query = query.strip()
            if not query:
                raise ValueError("search tag must contain a non-empty query")
            calls.append(
                ToolCall(
                    call_id=f"search-{context.turn_index + 1}-{index + 1}",
                    name="search",
                    arguments={"query": query},
                )
            )
        if not calls:
            raise ValueError("Agent action must contain <search> or <answer> tags")
        return ParsedAction(tool_calls=tuple(calls))

    def format_tool_results(
        self,
        results: Sequence[ToolResult],
        context: TurnContext,
    ) -> str:
        """Render retrieval results inside model-visible information tags."""
        del context
        blocks = []
        for result in results:
            if result.is_error:
                blocks.append(f"Search error: {result.content}")
                continue
            try:
                payload = json.loads(result.content)
            except json.JSONDecodeError as error:
                raise ValueError("Search tool returned invalid JSON") from error
            if not isinstance(payload, Mapping):
                raise ValueError("Search tool result must be a JSON object")
            passages = payload.get("results", [])
            if not isinstance(passages, Sequence) or isinstance(passages, (str, bytes)):
                raise ValueError("Search tool results must contain a passage sequence")
            lines = [f"Query: {payload.get('query', '')}"]
            if not passages:
                lines.append("No matching passages found.")
            for index, passage in enumerate(passages, start=1):
                if not isinstance(passage, Mapping):
                    raise ValueError("Every search passage must be a JSON object")
                lines.extend((f"[{index}] {passage.get('title', '')}", str(passage.get("text", ""))))
            blocks.append("\n".join(lines))
        return "<information>\n" + "\n\n".join(blocks) + "\n</information>"

    def format_error(self, message: str, context: TurnContext) -> str:
        """Return parser feedback using the Search-R1 observation tag."""
        del context
        return f"<information>\nProtocol error: {message}\n</information>"


def normalize_qa_answer(value: str) -> str:
    """Normalize an English open-domain answer for comparison."""
    if not isinstance(value, str):
        raise ValueError("QA answer must be text")
    lowered = value.casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in lowered
    )
    return " ".join(_ARTICLE_PATTERN.sub(" ", without_punctuation).split())


def parse_answer_aliases(value: Any) -> tuple[str, ...]:
    """Convert common dataset answer layouts into distinct textual aliases."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Ground truth must contain at least one answer alias")
        if "|||" in stripped:
            aliases = tuple(alias.strip() for alias in stripped.split("|||") if alias.strip())
            if aliases:
                return tuple(dict.fromkeys(aliases))
        if stripped[:1] in {"[", "{"}:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if decoded is not None:
                return parse_answer_aliases(decoded)
        return (stripped,)
    if isinstance(value, Mapping):
        for key in ("aliases", "answers", "answer", "ground_truth", "value"):
            candidate = value.get(key)
            if candidate is not None:
                return parse_answer_aliases(candidate)
        raise ValueError("Ground-truth mapping does not contain a supported answer field")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases = []
        for candidate in value:
            aliases.extend(parse_answer_aliases(candidate))
        distinct = tuple(dict.fromkeys(alias for alias in aliases if alias.strip()))
        if distinct:
            return distinct
    raise ValueError("Ground truth must contain at least one textual answer alias")


def compute_qa_reward(
    prediction: str,
    ground_truth: Any,
    partial_f1_weight: float = 0.25,
) -> float:
    """Score exact match and bounded token F1 across accepted aliases."""
    if not 0.0 <= partial_f1_weight <= 1.0:
        raise ValueError("partial_f1_weight must be between 0 and 1")
    normalized_prediction = normalize_qa_answer(prediction)
    if not normalized_prediction:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    best_f1 = 0.0
    for alias in parse_answer_aliases(ground_truth):
        normalized_alias = normalize_qa_answer(alias)
        if normalized_prediction == normalized_alias:
            return 1.0
        alias_tokens = normalized_alias.split()
        if not alias_tokens:
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(alias_tokens)).values())
        if overlap:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(alias_tokens)
            best_f1 = max(best_f1, 2.0 * precision * recall / (precision + recall))
    return partial_f1_weight * best_f1


class SearchR1Environment(ToolEnvironment):
    """Run Search-R1 turns against a fixed local corpus."""

    def __init__(self, *args: Any, retrieval_hit_reward: float = 0.1, **kwargs: Any) -> None:
        """Initialize one episode and its one-time dense retrieval reward."""
        super().__init__(*args, **kwargs)
        if not math.isfinite(retrieval_hit_reward) or retrieval_hit_reward < 0.0:
            raise ValueError("retrieval_hit_reward must be finite and non-negative")
        self.retrieval_hit_reward = float(retrieval_hit_reward)
        self._retrieval_rewarded = False

    async def reset(self, context: EpisodeContext) -> Observation:
        """Render Search-R1 instructions and the open-domain question."""
        self._validate_episode(context)
        question = self.prompt.messages[-1].content
        system_prompt = str(context.settings.get("search_system_prompt", _DEFAULT_SYSTEM_PROMPT))
        return context.encode_observation(
            f"{system_prompt}\n\nQuestion:\n{question}",
            role="system",
            metadata={
                "protocol": "search_r1",
                "tools": ("search",),
                CHAT_TEMPLATE_MESSAGES: (
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ),
            },
        )

    def _retrieval_hit(self, content: str) -> bool:
        """Return whether evidence contains a normalized answer alias."""
        evidence = "\n".join(
            line for line in content.splitlines() if not line.casefold().startswith("query:")
        )
        normalized_content = normalize_qa_answer(evidence)
        return any(
            len(alias) > 2 and alias not in {"yes", "no"} and alias in normalized_content
            for alias in (
                normalize_qa_answer(value)
                for value in parse_answer_aliases(self.prompt.ground_truth)
            )
        )

    async def step(self, action: Action, context: TurnContext) -> Transition:
        """Add a one-time retrieval-hit reward to a generic tool transition."""
        transition = await super().step(action, context)
        if (
            transition.done
            or self._retrieval_rewarded
            or self.retrieval_hit_reward == 0.0
            or int(transition.info.get("tool_success_count", 0)) == 0
            or not self._retrieval_hit(transition.observation.content)
        ):
            return transition
        self._retrieval_rewarded = True
        info = dict(transition.info)
        components = dict(info.get("reward_components", {}))
        components["retrieval_hit"] = self.retrieval_hit_reward
        info.update({"reward_components": components, "retrieval_hit": True})
        return replace(transition, reward=transition.reward + self.retrieval_hit_reward, info=info)


@lru_cache(maxsize=8)
def _load_retriever(
    path: str,
    modification_time_ns: int,
    size_bytes: int,
    max_documents: Optional[int],
) -> LocalBM25Retriever:
    """Reuse immutable indexes and invalidate changed corpus files."""
    del modification_time_ns, size_bytes
    return LocalBM25Retriever.from_jsonl(path, max_documents=max_documents)


def _positive_int(settings: Mapping[str, Any], name: str, default: int) -> int:
    """Read one positive integer from example settings."""
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"agentic.{name} must be a positive integer")
    return value


def _retriever_from_settings(settings: Mapping[str, Any]) -> LocalBM25Retriever:
    """Resolve and cache the configured local corpus."""
    corpus_value = settings.get("search_corpus_path")
    if not isinstance(corpus_value, str) or not corpus_value.strip():
        raise ValueError("agentic.search_corpus_path must be a non-empty JSONL path")
    corpus_path = Path(corpus_value).expanduser().resolve()
    if not corpus_path.is_file():
        raise ValueError(f"Search corpus does not exist: {corpus_path}")
    max_documents = None
    if settings.get("search_max_documents") is not None:
        max_documents = _positive_int(settings, "search_max_documents", 1)
    stat = corpus_path.stat()
    return _load_retriever(str(corpus_path), stat.st_mtime_ns, stat.st_size, max_documents)


@lru_cache(maxsize=16)
def _cached_search_registry(
    retriever: LocalBM25Retriever,
    top_k: int,
    max_query_chars: int,
    max_document_chars: int,
) -> ToolRegistry:
    """Reuse immutable tool definitions across episodes in one batch."""
    return build_search_registry(
        retriever,
        top_k=top_k,
        max_query_chars=max_query_chars,
        max_document_chars=max_document_chars,
    )


def build_search_r1_environment(context: EpisodeContext) -> SearchR1Environment:
    """Build one bounded Search-R1 episode from generic runner settings."""
    if context.interaction_mode is not InteractionMode.MULTI_TURN:
        raise ValueError("Search-R1 requires interaction_mode=multi_turn")
    settings = context.settings
    retriever = _retriever_from_settings(settings)
    registry = _cached_search_registry(
        retriever,
        _positive_int(settings, "search_top_k", 3),
        _positive_int(settings, "search_max_query_chars", 256),
        _positive_int(settings, "search_max_document_chars", 1200),
    )
    timeout_value = settings.get("tool_timeout_seconds", 5.0)
    timeout_seconds = None if timeout_value is None else float(timeout_value)
    partial_f1_weight = float(settings.get("search_partial_f1_weight", 0.25))

    def score(answer: str, prompt: PromptRecord) -> float:
        """Score final answers against every configured alias."""
        return compute_qa_reward(answer, prompt.ground_truth, partial_f1_weight)

    return SearchR1Environment(
        context=context,
        protocol=SearchR1Protocol(),
        executor=ToolExecutor(
            registry,
            timeout_seconds=timeout_seconds,
            max_concurrency=_positive_int(settings, "tool_max_concurrency", 4),
            max_calls_per_turn=_positive_int(settings, "tool_max_calls_per_turn", 4),
        ),
        reward_function=score,
        invalid_action_reward=float(settings.get("invalid_action_reward", -0.05)),
        retrieval_hit_reward=float(settings.get("search_retrieval_hit_reward", 0.1)),
        tool_observation_role="environment",
    )


INTERACTION_PROTOCOLS.register("search_r1")(SearchR1Protocol)
ENVIRONMENTS.register("search_r1_local")(build_search_r1_environment)

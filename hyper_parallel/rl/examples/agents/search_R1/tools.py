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
"""Dependency-free local BM25 search tool for the Search-R1 example."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rl.agentic.tools import ToolRegistry


_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_SEARCH_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
        "was", "were", "what", "when", "where", "which", "who", "why", "with",
    }
)


@dataclass(frozen=True)
class SearchDocument:
    """One immutable document in a local retrieval corpus."""

    document_id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        """Reject corpus rows that cannot produce useful observations."""
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("Search document_id must be non-empty")
        if not isinstance(self.title, str):
            raise ValueError(f"Search document title must be text: {self.document_id}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"Search document text must be non-empty: {self.document_id}")


class LocalBM25Retriever:
    """Small dependency-free BM25 index intended for offline RL smoke runs."""

    def __init__(self, documents: Sequence[SearchDocument]) -> None:
        """Build term and document frequencies once for shared episode use."""
        if not documents:
            raise ValueError("LocalBM25Retriever requires at least one document")
        self.documents = tuple(documents)
        self._term_frequencies = tuple(
            Counter(self._tokenize(f"{document.title} {document.title} {document.text}"))
            for document in self.documents
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_length = sum(self._document_lengths) / len(self._document_lengths)
        if self._average_length <= 0.0:
            raise ValueError("Search corpus must contain at least one searchable word")
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        document_count = len(self.documents)
        self._inverse_document_frequency = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        """Tokenize Unicode text and remove common query-only stop words."""
        return [
            token
            for token in _WORD_PATTERN.findall(value.casefold())
            if token not in _SEARCH_STOP_WORDS
        ]

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        max_documents: Optional[int] = None,
    ) -> "LocalBM25Retriever":
        """Load document_id, title, and text rows from JSONL."""
        corpus_path = Path(path)
        if not corpus_path.is_file():
            raise ValueError(f"Search corpus does not exist: {corpus_path}")
        if max_documents is not None and (
            isinstance(max_documents, bool)
            or not isinstance(max_documents, int)
            or max_documents <= 0
        ):
            raise ValueError("max_documents must be positive or null")
        documents = []
        with corpus_path.open(encoding="utf-8") as corpus_file:
            for line_number, line in enumerate(corpus_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid search corpus JSON at {corpus_path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(row, Mapping):
                    raise ValueError(f"Search corpus row {line_number} must be a JSON object")
                document_id = str(row.get("document_id", row.get("id", line_number)))
                title = str(row.get("title", ""))
                text = row.get("text", row.get("contents"))
                if not isinstance(text, str):
                    raise ValueError(
                        f"Search corpus row {line_number} must define textual 'text'"
                    )
                documents.append(SearchDocument(document_id, title, text))
                if max_documents is not None and len(documents) >= max_documents:
                    break
        return cls(documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
        max_document_chars: int = 1200,
    ) -> dict[str, Any]:
        """Return the highest-scoring passages for one textual query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be non-empty text")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("search top_k must be positive")
        if (
            isinstance(max_document_chars, bool)
            or not isinstance(max_document_chars, int)
            or max_document_chars <= 0
        ):
            raise ValueError("search max_document_chars must be positive")
        query_terms = tuple(dict.fromkeys(self._tokenize(query)))
        scored = []
        for index, frequencies in enumerate(self._term_frequencies):
            document_length = self._document_lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * document_length / self._average_length
                )
                score += self._inverse_document_frequency[term] * frequency * 2.5 / denominator
            if score > 0.0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.documents[item[1]].document_id))
        results = []
        for score, index in scored[:top_k]:
            document = self.documents[index]
            results.append(
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "text": document.text[:max_document_chars],
                    "score": round(score, 6),
                }
            )
        return {"query": query.strip(), "results": results}


def build_search_registry(
    retriever: LocalBM25Retriever,
    *,
    top_k: int,
    max_query_chars: int,
    max_document_chars: int,
) -> ToolRegistry:
    """Build one bounded search registry around a shared immutable index."""
    def search(query: str) -> dict[str, Any]:
        """Search the configured evidence corpus."""
        if not isinstance(query, str) or len(query) > max_query_chars:
            raise ValueError(
                f"search query must be text with at most {max_query_chars} characters"
            )
        return retriever.search(
            query,
            top_k=top_k,
            max_document_chars=max_document_chars,
        )

    registry = ToolRegistry()
    registry.register(
        "search",
        description="Search the local evidence corpus",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )(search)
    return registry

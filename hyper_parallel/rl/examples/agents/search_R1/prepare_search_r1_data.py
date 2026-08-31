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
"""Prepare small HotpotQA or NQ subsets for local Search-R1 training."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd


_QUESTION_COLUMNS = ("question", "prompt", "input", "query")
_ANSWER_COLUMNS = ("answers", "answer", "reward_model", "extra_info")
_CONTEXT_COLUMNS = ("context", "contexts", "documents", "passages")


def _to_builtin(value: Any) -> Any:
    """Recursively remove NumPy/Pandas container wrappers."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_builtin(item) for item in value]
    return value


def _is_missing(value: Any) -> bool:
    """Return whether one scalar field is absent or NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _iter_jsonl_records(path: Path, row_label: str) -> Any:
    """Yield mapping rows from one JSON Lines file."""
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{row_label} {line_number} must be an object")
            yield _to_builtin(value)


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read records from Parquet, JSON, or JSONL without network access."""
    if not path.is_file():
        raise ValueError(f"Input dataset does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        records = frame.to_dict("records")
    elif suffix in {".jsonl", ".ndjson"}:
        records = list(_iter_jsonl_records(path, "JSONL row"))
    elif suffix == ".json":
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
        if isinstance(value, Mapping):
            value = value.get("data", value.get("records"))
        if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
            raise ValueError("JSON input must be a list of objects or contain data/records")
        records = [dict(row) for row in value]
    else:
        raise ValueError(f"Unsupported dataset suffix {path.suffix!r}; use parquet, json, or jsonl")
    if not records:
        raise ValueError(f"Input dataset contains no records: {path}")
    return [_to_builtin(record) for record in records]


def _pick_value(record: Mapping[str, Any], columns: Sequence[str], label: str) -> Any:
    """Return the first present value among accepted source columns."""
    for column in columns:
        value = record.get(column)
        if not _is_missing(value):
            return value
    raise ValueError(f"Dataset row is missing {label}; expected one of {tuple(columns)}")


def _question(value: Any) -> str:
    """Normalize raw text or a chat-message sequence into one question."""
    value = _to_builtin(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("question", "text", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for message in reversed(value):
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("Question must be non-empty text or contain one user chat message")


def _answer_aliases(value: Any) -> tuple[str, ...]:
    """Flatten common HotpotQA, NQ, and Search-R1 answer layouts."""
    value = _to_builtin(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped[:1] in {"[", "{"}:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if decoded is not None:
                aliases = _answer_aliases(decoded)
                if aliases:
                    return aliases
        return (stripped,)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (str(value),)
    if isinstance(value, Mapping):
        aliases = []
        for key in (
            "aliases",
            "answers",
            "answer",
            "ground_truth",
            "value",
            "text",
            "short_answers",
        ):
            candidate = value.get(key)
            if candidate is not None:
                aliases.extend(_answer_aliases(candidate))
        return tuple(dict.fromkeys(aliases))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        aliases = []
        for candidate in value:
            aliases.extend(_answer_aliases(candidate))
        return tuple(dict.fromkeys(aliases))
    return ()


def _text(value: Any) -> str:
    """Join sentence lists or return one stripped passage."""
    value = _to_builtin(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return ""


def _document(title: Any, text: Any, fallback_id: str, max_chars: int) -> Optional[dict[str, str]]:
    """Build one stable local-corpus row when passage text is available."""
    normalized_title = str(title).strip() if title is not None else ""
    normalized_text = _text(text)[:max_chars]
    if not normalized_text:
        return None
    identity = sha256(f"{normalized_title}\0{normalized_text}".encode("utf-8")).hexdigest()[:20]
    return {
        "document_id": identity or fallback_id,
        "title": normalized_title,
        "text": normalized_text,
    }


def _parallel_context_documents(
    titles: Sequence[Any],
    sentences: Any,
    task_id: str,
    max_chars: int,
) -> list[dict[str, str]]:
    """Combine parallel title and passage arrays into corpus rows."""
    if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
        raise ValueError("Parallel context titles require a sentence/text sequence")
    documents = []
    for index, (title, passage) in enumerate(zip(titles, sentences), start=1):
        document = _document(title, passage, f"{task_id}-{index}", max_chars)
        if document is not None:
            documents.append(document)
    return documents


def _mapping_context_documents(
    value: Mapping[str, Any],
    task_id: str,
    max_chars: int,
) -> list[dict[str, str]]:
    """Normalize a context mapping or parallel-array mapping."""
    titles = value.get("title", value.get("titles"))
    sentences = value.get("sentences", value.get("text", value.get("contents")))
    if isinstance(titles, Sequence) and not isinstance(titles, (str, bytes)):
        return _parallel_context_documents(titles, sentences, task_id, max_chars)
    document = _document(
        value.get("title", ""),
        value.get("text", value.get("contents", value.get("sentences"))),
        f"{task_id}-1",
        max_chars,
    )
    return [] if document is None else [document]


def _sequence_context_documents(
    value: Sequence[Any],
    task_id: str,
    max_chars: int,
) -> list[dict[str, str]]:
    """Normalize a heterogeneous sequence of passage representations."""
    documents = []
    for index, passage in enumerate(value, start=1):
        fallback_id = f"{task_id}-{index}"
        if isinstance(passage, Mapping):
            document = _document(
                passage.get("title", ""),
                passage.get("text", passage.get("contents", passage.get("sentences"))),
                fallback_id,
                max_chars,
            )
        elif (
            isinstance(passage, Sequence)
            and not isinstance(passage, (str, bytes))
            and len(passage) == 2
        ):
            document = _document(passage[0], passage[1], fallback_id, max_chars)
        else:
            document = _document("", passage, fallback_id, max_chars)
        if document is not None:
            documents.append(document)
    return documents


def _context_documents(value: Any, task_id: str, max_chars: int) -> list[dict[str, str]]:
    """Normalize HotpotQA parallel arrays and common passage collections."""
    value = _to_builtin(value)
    if _is_missing(value):
        return []
    if isinstance(value, str):
        document = _document("", value, f"{task_id}-1", max_chars)
        return [] if document is None else [document]
    if isinstance(value, Mapping):
        return _mapping_context_documents(value, task_id, max_chars)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _sequence_context_documents(value, task_id, max_chars)
    return []


def _task_id(record: Mapping[str, Any], source_index: int, dataset: str) -> str:
    """Resolve a stable row identity without depending on DataFrame indices."""
    for key in ("task_id", "id", "_id", "question_id"):
        value = record.get(key)
        if not _is_missing(value) and str(value).strip():
            return str(value).strip()
    return f"{dataset}-{source_index}"


def _source_rows(
    records: Sequence[Mapping[str, Any]],
    dataset: str,
    offset: int,
    max_samples: int,
    max_document_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Convert selected task rows and their embedded evidence passages."""
    selected = records[offset : offset + max_samples]
    if not selected:
        raise ValueError("The requested offset and max_samples select no dataset rows")
    tasks = []
    documents = []
    for relative_index, record in enumerate(selected):
        source_index = offset + relative_index
        task_id = _task_id(record, source_index, dataset)
        question = _question(_pick_value(record, _QUESTION_COLUMNS, "question"))
        aliases = _answer_aliases(_pick_value(record, _ANSWER_COLUMNS, "answer"))
        if not aliases:
            raise ValueError(f"Dataset row {source_index} contains no textual answer alias")
        tasks.append(
            {
                "prompt": [{"role": "user", "content": question}],
                # A plain delimiter preserves multiple QA aliases end-to-end.
                "answer": " ||| ".join(aliases),
                "data_source": dataset,
                "task_id": task_id,
            }
        )
        context = next(
            (
                record.get(column)
                for column in _CONTEXT_COLUMNS
                if not _is_missing(record.get(column))
            ),
            None,
        )
        documents.extend(_context_documents(context, task_id, max_document_chars))
    return tasks, documents


def _external_documents(
    paths: Sequence[Path],
    max_documents: Optional[int],
    max_document_chars: int,
) -> list[dict[str, str]]:
    """Load optional Wikipedia or retrieval-corpus records."""
    documents = []
    for path in paths:
        suffix = path.suffix.casefold()
        if suffix in {".jsonl", ".ndjson"}:
            if not path.is_file():
                raise ValueError(f"Input corpus does not exist: {path}")
            corpus_records = _iter_jsonl_records(path, "Corpus row")
        else:
            corpus_records = iter(_read_records(path))
        for index, record in enumerate(corpus_records, start=1):
            document = _document(
                record.get("title", ""),
                record.get("text", record.get("contents", record.get("passage"))),
                f"external-{index}",
                max_document_chars,
            )
            if document is not None:
                documents.append(document)
            if max_documents is not None and len(documents) >= max_documents:
                return documents
    return documents


def _deduplicate_documents(documents: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Preserve first occurrence order while removing duplicate passages."""
    deduplicated = []
    seen = set()
    for document in documents:
        identity = document["document_id"]
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(document)
    return deduplicated


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one generated or source artifact."""
    digest = sha256()
    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(
    args: argparse.Namespace,
    tasks: Sequence[dict[str, Any]],
    documents: Sequence[dict[str, str]],
) -> None:
    """Write train/test Parquet, local corpus JSONL, and a manifest."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suggested_test_count = max(1, len(tasks) // 10) if len(tasks) > 1 else 0
    test_count = min(args.test_samples, suggested_test_count, max(0, len(tasks) - 1))
    train_tasks = tasks[:-test_count] if test_count else tasks
    test_tasks = tasks[-test_count:] if test_count else tasks
    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    corpus_path = args.output_dir / "corpus.jsonl"
    pd.DataFrame(train_tasks).to_parquet(train_path, index=False)
    pd.DataFrame(test_tasks).to_parquet(test_path, index=False)
    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for document in documents:
            corpus_file.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "dataset": args.dataset,
        "source": str(args.input),
        "source_sha256": _file_sha256(args.input),
        "corpus_sources": [
            {"path": str(path), "sha256": _file_sha256(path)}
            for path in args.corpus_input
        ],
        "offset": args.offset,
        "requested_max_samples": args.max_samples,
        "max_corpus_documents": args.max_corpus_documents,
        "train_records": len(train_tasks),
        "test_records": len(test_tasks),
        "corpus_documents": len(documents),
        "artifacts": {
            "train.parquet": _file_sha256(train_path),
            "test.parquet": _file_sha256(test_path),
            "corpus.jsonl": _file_sha256(corpus_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    """Parse local, bounded data-conversion arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hotpotqa", "nq"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--corpus-input", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=32)
    parser.add_argument("--max-corpus-documents", type=int, default=None)
    parser.add_argument("--max-document-chars", type=int, default=4000)
    args = parser.parse_args()
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    for name in ("max_samples", "max_document_chars"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.test_samples < 0:
        raise ValueError("test-samples must be non-negative")
    if args.max_corpus_documents is not None and args.max_corpus_documents <= 0:
        raise ValueError("max-corpus-documents must be positive or omitted")
    return args


def main() -> None:
    """Prepare one deterministic Search-R1 dataset without downloading data."""
    args = _parse_args()
    records = _read_records(args.input)
    tasks, embedded_documents = _source_rows(
        records,
        args.dataset,
        args.offset,
        args.max_samples,
        args.max_document_chars,
    )
    external_documents = _external_documents(
        args.corpus_input,
        args.max_corpus_documents,
        args.max_document_chars,
    )
    documents = _deduplicate_documents((*embedded_documents, *external_documents))
    if not documents:
        raise ValueError(
            "No retrieval passages were found. HotpotQA should provide context; "
            "for NQ, pass one or more --corpus-input files with title/text fields."
        )
    if args.max_corpus_documents is not None:
        documents = documents[: args.max_corpus_documents]
    _write_outputs(args, tasks, documents)


if __name__ == "__main__":
    main()

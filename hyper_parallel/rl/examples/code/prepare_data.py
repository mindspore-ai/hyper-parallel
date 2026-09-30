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
"""Prepare a pinned, deduplicated Eurus stdio subset and adapt its rows for RL."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from rl.dataset.contracts import Message, PromptRecord
from examples.code.judge import validate_tests

DATASET = "PRIME-RL/Eurus-2-RL-Data"
DEFAULT_REVISION = "9776b13264b5aaa0b16495fcf086a0a8d86fd655"
SYSTEM_PROMPT = (
    "Solve the programming problem in Python 3. Read input from standard input and write the required "
    "answer to standard output. Return a complete runnable program in one ```python code block. "
    "Do not call external services or read files. /no_think"
)
_UNSUPPORTED_TASK = re.compile(
    r"\b(?:interactive (?:problem|task)|output[- ]only|(?:custom|special) (?:checker|judge)|"
    r"(?:find|print|output) any|any (?:valid|correct) (?:answer|solution)|"
    r"(?:multiple|several) (?:valid )?(?:answers|solutions)|"
    r"(?:absolute|relative) (?:or (?:absolute|relative) )?error|decimal places|floating[- ]point)\b",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt_row(row: Mapping[str, Any], index: int) -> "PromptRecord":
    """Preserve the prepared messages, private tests and stable task identity."""
    del index
    metadata = dict(row["extra_info"])
    task_id = metadata.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Prepared code rows require a non-empty stable task_id")
    validate_tests(row["reward_model"]["ground_truth"])
    if metadata.get("task_type") != "code_stdio" or metadata.get("language") != "python":
        raise ValueError("Prepared code rows must declare task_type=code_stdio and language=python")
    return PromptRecord(
        prompt_id=metadata["task_id"],
        messages=tuple(Message(message["role"], message["content"]) for message in row["prompt"]),
        ground_truth=row["reward_model"]["ground_truth"],
        metadata={**metadata, "data_source": row["data_source"]},
    )


def _prepare_tests(truth: Any) -> tuple[Any, str]:
    """Validate stdio test pairs without changing their order or contents."""
    try:
        truth = json.loads(truth) if isinstance(truth, str) else truth
    except json.JSONDecodeError:
        return None, "invalid_test_json"
    if not isinstance(truth, dict):
        return None, "invalid_tests"
    if any(truth.get(key) for key in (
        "fn_name", "assert_case", "checker", "special_judge", "is_interactive", "interactive",
    )):
        return None, "unsupported_test_mode"
    inputs, outputs = truth.get("inputs"), truth.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        return None, "invalid_test_pairs"
    if not all(isinstance(value, str) for value in inputs + outputs):
        return None, "non_string_test_case"
    return {"inputs": inputs, "outputs": outputs}, "accepted"


def prepare_row(row: Mapping[str, Any], split: str, index: int, revision: str) -> tuple[Any, str]:
    """Return one stdio row, or an explicit filtering reason; never repair tests."""
    if row.get("ability") != "code":
        return None, "not_code"
    tests, reason = _prepare_tests(row.get("reward_model", {}).get("ground_truth"))
    if tests is None:
        return None, reason
    messages = row.get("prompt")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        return None, "invalid_messages"
    users = [message.get("content") for message in messages if message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0], str) or not users[0].strip():
        return None, "invalid_user_prompt"
    if re.match(r"\s*(?:Examples?|Sample (?:Input|Output))\b", users[0], re.IGNORECASE):
        return None, "missing_problem_statement"
    if _UNSUPPORTED_TASK.search(users[0]):
        return None, "unsupported_problem_statement"
    data_source = row.get("data_source")
    if data_source not in ("apps", "codecontests", "codeforces", "taco"):
        return None, "unknown_code_source"
    test_hash = hashlib.sha256(json.dumps(tests, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    task_id = f"eurus2:{revision[:12]}:{data_source}:{split}:{index}"
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": users[0]}],
        "ability": "code", "data_source": data_source,
        "reward_model": {"style": "rule", "ground_truth": {**tests, "test_version": test_hash}},
        "extra_info": {
            "task_id": task_id, "task_type": "code_stdio", "language": "python",
            "source_revision": revision, "source_split": split, "source_index": index,
        },
    }, "accepted"


def _budget_rejection(
    prepared: Mapping[str, Any], tokenizer: Any, max_prompt_tokens: int, max_test_bytes: int, max_test_cases: int,
) -> str:
    """Classify resource limits before accepting a prepared task."""
    tests = prepared["reward_model"]["ground_truth"]
    if len(tests["inputs"]) > max_test_cases:
        return "too_many_test_cases"
    if any(len(value.encode("utf-8")) > max_test_bytes for value in tests["inputs"] + tests["outputs"]):
        return "test_case_exceeds_byte_budget"
    if tokenizer is not None:
        encoded = tokenizer.apply_chat_template(
            prepared["prompt"], tokenize=True, add_generation_prompt=True, return_dict=True,
        )
        if len(encoded["input_ids"]) > max_prompt_tokens:
            return "prompt_exceeds_token_budget"
    return "accepted"


def _validate_preparation_limits(
    revision: str, limits: Mapping[str, int], max_prompt_tokens: int, max_test_bytes: int, max_test_cases: int,
) -> None:
    """Validate reproducibility and resource bounds before preparing any files."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character commit hash")
    if any(limit <= 0 for limit in limits.values()):
        raise ValueError("Split limits must be positive")
    if min(max_prompt_tokens, max_test_bytes, max_test_cases) <= 0:
        raise ValueError("Preparation limits must be positive")


def prepare_dataset(
    source_dir: Path, output_dir: Path, revision: str, limits: Mapping[str, int], *,
    tokenizer: Any = None, max_prompt_tokens: int = 2048, max_test_bytes: int = 16384,
    max_test_cases: int = 64, manual_exclusions: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Write reproducible development splits without crossing source split boundaries."""
    _validate_preparation_limits(revision, limits, max_prompt_tokens, max_test_bytes, max_test_cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET, "revision": revision, "prompt_template": SYSTEM_PROMPT, "splits": {}}
    manifest["manual_exclusions"] = dict(manual_exclusions or {})
    manifest["unsupported_problem_pattern"] = _UNSUPPORTED_TASK.pattern
    manifest["filter_limitations"] = (
        "Statement checks are conservative English heuristics, not proof that a task has a unique exact output. "
        "Review development tasks and establish correct-solution baselines before using their rewards."
    )
    manifest["limits"] = {
        "max_prompt_tokens": max_prompt_tokens, "max_test_bytes": max_test_bytes, "max_test_cases": max_test_cases,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
    }
    seen_prompts = set()
    for split, limit in limits.items():
        source = source_dir / f"{split}.parquet"
        counts, records = Counter(), []
        scanned = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                prepared, reason = prepare_row(row, split, scanned, revision)
                if f"{split}:{scanned}" in manifest["manual_exclusions"]:
                    prepared, reason = None, "manual_exclusion"
                scanned += 1
                if prepared is not None:
                    reason = _budget_rejection(prepared, tokenizer, max_prompt_tokens, max_test_bytes, max_test_cases)
                    if reason != "accepted":
                        prepared = None
                if prepared is not None:
                    question = prepared["prompt"][-1]["content"]
                    prompt_hash = hashlib.sha256(" ".join(question.split()).encode()).hexdigest()
                    if prompt_hash in seen_prompts:
                        reason = "duplicate_prompt"
                    else:
                        seen_prompts.add(prompt_hash)
                        records.append(prepared)
                counts[reason] += 1
                if len(records) == limit:
                    break
            if len(records) == limit:
                break
        if not records:
            raise ValueError(f"No supported stdio tasks found in {source}")
        destination = output_dir / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(records), destination)
        manifest["splits"][split] = {
            "path": str(destination), "source_sha256": _sha256(source), "sha256": _sha256(destination),
            "source_rows": pq.ParquetFile(source).metadata.num_rows, "scanned_rows": scanned,
            "selected_rows": len(records), "filter_counts": dict(counts),
            "task_ids": [record["extra_info"]["task_id"] for record in records],
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Prepare a small stdio subset from locally available pinned Eurus parquet files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer used to enforce the prompt budget")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--manual-exclusions", type=Path, help="JSON mapping source split:index to review reason")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-test-bytes", type=int, default=16384)
    parser.add_argument("--max-test-cases", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=128)
    parser.add_argument("--max-validation", type=int, default=32)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prepare_dataset(
        args.source_dir, args.output_dir, args.revision, {"train": args.max_train, "validation": args.max_validation},
        tokenizer=tokenizer, max_prompt_tokens=args.max_prompt_tokens,
        max_test_bytes=args.max_test_bytes, max_test_cases=args.max_test_cases,
        manual_exclusions=(json.loads(args.manual_exclusions.read_text(encoding="utf-8"))
                           if args.manual_exclusions else None),
    )


if __name__ == "__main__":
    main()

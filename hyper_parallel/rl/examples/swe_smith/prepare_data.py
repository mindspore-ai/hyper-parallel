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
"""Prepare selected or full SWE-Smith splits through the same public data contract."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from rl.data.contracts import Message, PromptRecord


_SCHEMA = pa.schema([
    ("task_id", pa.string()), ("replica_index", pa.int64()),
    ("ground_truth", pa.struct([("instance_id", pa.string()), ("record_sha256", pa.string())])),
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
])


def iter_instances(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSONL incrementally, keeping one task record in memory."""
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a task object at {path}:{number}")
                yield row


def record_path(directory: Path, instance_id: str) -> Path:
    """Resolve an instance without trusting its ID as a filesystem path."""
    return Path(directory) / "instances" / (hashlib.sha256(instance_id.encode()).hexdigest() + ".json")


def _validate_test_nodes(value: Any, field: str) -> None:
    """Require distinct, nonempty pytest nodes without option-like entries."""
    if (not isinstance(value, list) or any(not isinstance(node, str) or not node or "\x00" in node
                                         or node.startswith("-") for node in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"SWE-Smith {field} must contain distinct test node strings")


def validate_instance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project the Agent Lightning fields without exposing private tests to Codex."""
    if not isinstance(row, Mapping):
        raise ValueError("SWE-Smith task records must be objects")
    result = {}
    for field in ("instance_id", "problem_statement", "image_name", "repo"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"SWE-Smith requires a nonempty {field}")
        result[field] = value
    identity = result["instance_id"]
    if identity.startswith("-") or any(char.isspace() for char in identity) or ".." in identity:
        raise ValueError("Invalid SWE-Smith branch identity")
    for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = row.get(field)
        _validate_test_nodes(value, field)
        result[field] = value
    if not result["FAIL_TO_PASS"]:
        raise ValueError("SWE-Smith requires at least one FAIL_TO_PASS test")
    if set(result["FAIL_TO_PASS"]) & set(result["PASS_TO_PASS"]):
        raise ValueError("FAIL_TO_PASS and PASS_TO_PASS must be disjoint")
    return result


def adapt_row(row: Mapping[str, Any], index: int) -> PromptRecord:
    """Reconstruct public prompts with controller-owned task identity."""
    del index
    truth = row.get("ground_truth")
    if (not isinstance(truth, Mapping) or set(truth) != {"instance_id", "record_sha256"}
            or not isinstance(truth["instance_id"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(truth["record_sha256"]))):
        raise ValueError("SWE-Smith rows require instance_id and record_sha256")
    task_id = "swe-smith:" + truth["instance_id"]
    if row.get("task_id") != task_id:
        raise ValueError("SWE-Smith public and trusted task identities differ")
    replica = row.get("replica_index", 0)
    if not isinstance(replica, int) or isinstance(replica, bool) or replica < 0:
        raise ValueError("replica_index must be a nonnegative integer")
    prompt_id = task_id if replica == 0 else f"{task_id}:replica:{replica}"
    messages = tuple(Message(item["role"], item["content"]) for item in row["prompt"])
    return PromptRecord(prompt_id, messages, dict(truth),
                        {"task_type": "swe_smith", "task_id": task_id, "replica_index": replica})


def _selection_ids(splits: Mapping[str, Path], instance_ids: Sequence[str] | None,
                   max_instances: int | None, repeats: int) -> set[str] | None:
    """Validate selection controls before writing any prepared data."""
    for name, value in (("max_instances", max_instances), ("repeats", repeats)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if not splits or set(splits) - {"train", "validation"}:
        raise ValueError("Supply train and/or validation split paths")
    requested = None if instance_ids is None else set(instance_ids)
    if requested is not None and (not requested or len(requested) != len(instance_ids)):
        raise ValueError("The requested ID list must be nonempty and distinct")
    return requested


def prepare_data(splits: Mapping[str, Path], output: Path, *, instance_ids: Sequence[str] | None = None,
                 max_instances: int | None = None, repeats: int = 1) -> dict[str, Any]:
    """Select records before writing bounded public batches and private per-instance files.

    Selection applies across the supplied splits in their declared order. Repeats
    are explicit functional sampling replicas, never additional benchmark tasks.
    Missing requested IDs are errors even when a quantity limit selects fewer IDs.
    """
    requested = _selection_ids(splits, instance_ids, max_instances, repeats)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty prepared-data directory")
    (output / "instances").mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "requested_ids": None if requested is None else sorted(requested),
                "max_instances": max_instances, "repeats": repeats, "splits": {}}
    seen, found, selected = set(), set(), []
    with (output / "instances.jsonl").open("w", encoding="utf-8") as instances:
        for split, source in splits.items():
            rows, split_ids = [], []
            source_digest = hashlib.sha256()
            with pq.ParquetWriter(output / f"{split}.parquet", _SCHEMA) as writer, Path(source).open("rb") as stream:
                for line in stream:
                    source_digest.update(line)
                    if not line.strip():
                        continue
                    row = validate_instance(json.loads(line))
                    identity = row["instance_id"]
                    if identity in seen:
                        raise ValueError(f"Duplicate or overlapping split instance: {identity}")
                    seen.add(identity)
                    if requested is not None and identity not in requested:
                        continue
                    found.add(identity)
                    if max_instances is not None and len(selected) >= max_instances:
                        continue
                    raw = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
                    record_path(output, identity).write_bytes(raw)
                    instances.write(json.dumps({**row, "split": split}, ensure_ascii=False) + "\n")
                    selected.append(identity)
                    split_ids.append(identity)
                    issue = ("Repair the repository in /workspace for the issue below. Inspect relevant source, "
                             "implement a focused fix and run relevant existing tests. Preserve tests, "
                             "project configuration and Git metadata. Use /tmp for scratch. The repository "
                             "is offline.\n\n" + row["problem_statement"])
                    for replica in range(repeats):
                        rows.append({"task_id": "swe-smith:" + identity, "replica_index": replica,
                                     "ground_truth": {"instance_id": identity,
                                                      "record_sha256": hashlib.sha256(raw).hexdigest()},
                                     "prompt": [{"role": "user", "content": issue}]})
                        if len(rows) == 128:
                            writer.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA))
                            rows.clear()
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA))
            manifest["splits"][split] = {"source": str(Path(source).resolve()),
                                         "sha256": source_digest.hexdigest(), "instance_ids": split_ids}
    if requested is not None and requested - found:
        raise ValueError(f"Requested IDs not present: {sorted(requested - found)}")
    if not selected:
        raise ValueError("Selection contains no SWE-Smith instances")
    (output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Prepare full splits by default, or apply explicit reproducible selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--validation-jsonl", type=Path)
    parser.add_argument("--instance-ids-file", type=Path)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {name: path for name, path in (("train", args.train_jsonl), ("validation", args.validation_jsonl)) if path}
    identities = None if args.instance_ids_file is None else [
        line.strip() for line in args.instance_ids_file.read_text().splitlines() if line.strip()]
    prepare_data(splits, args.output, instance_ids=identities, max_instances=args.max_instances, repeats=args.repeats)


if __name__ == "__main__":
    main()

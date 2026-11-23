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
"""Validated input and single-writer atomic output on a shared filesystem."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from hyper_parallel.predict.types import PredictionRequest


def file_digest(path: str | Path) -> str:
    """Hash file contents without retaining a second copy in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_messages(messages: Any, ordinal: int) -> None:
    """Accept text conversation history while preserving template-specific fields."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"input record {ordinal}: messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"input record {ordinal}: unsupported message role")
        content = message.get("content")
        if not isinstance(content, str) and not (
                content is None and message["role"] == "assistant" and message.get("tool_calls")):
            raise ValueError(f"input record {ordinal}: only text messages are supported")


def _request(value: Any, ordinal: int) -> PredictionRequest:
    if not isinstance(value, dict):
        raise ValueError(f"input record {ordinal} must be an object")
    input_id = value.get("id", ordinal)
    if type(input_id) not in (str, int) or input_id == "":
        raise ValueError(f"input record {ordinal}: id must be a non-empty string or integer")
    if ("prompt" in value) == ("messages" in value):
        raise ValueError(f"input record {ordinal}: supply exactly one of prompt or messages")
    prompt, messages = value.get("prompt"), value.get("messages")
    if "prompt" in value:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"input record {ordinal}: prompt must be non-empty text")
    else:
        _validate_messages(messages, ordinal)
    return PredictionRequest(input_id, ordinal, prompt, messages)


def read_requests(path: str | Path) -> list[PredictionRequest]:
    """Read txt lines or JSON/JSONL records and reject duplicate ids before generation."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".txt":
            values = [{"prompt": line.rstrip("\r\n")} for line in handle]
        elif path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix.lower() == ".json":
            values = json.load(handle)
        else:
            raise ValueError("prediction.input must be a .txt, .jsonl, or .json file")
    if not isinstance(values, list) or not values:
        raise ValueError("prediction input must contain a non-empty list of requests")
    requests = [_request(value, ordinal) for ordinal, value in enumerate(values)]
    seen = set()
    for request in requests:
        if request.input_id in seen:
            raise ValueError(f"duplicate prediction id: {request.input_id!r}")
        seen.add(request.input_id)
    return requests


def write_json(path: Path, value: Any) -> None:
    """Write and flush a JSON control record before publishing it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class PredictionOutput:
    """Per-run staging directory and one coordinator for final publication.

    Shards survive failures for diagnosis. The JSONL is the atomic commit point;
    the adjacent manifest carries its digest, so readers can verify the pair.
    """

    def __init__(self, output_file: str, run_id: str, overwrite: bool) -> None:
        """Describe the output without touching the filesystem."""
        self.path = Path(output_file).absolute()
        self.run_id = run_id
        self.overwrite = overwrite
        self.directory = self.path.parent / f".{self.path.name}.{run_id}"
        self.manifest_path = self.path.with_name(self.path.name + ".manifest.json")
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._owns_lock = False

    def prepare(self) -> None:
        """Reserve the destination; only the coordinator calls this method."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock_path.open("x", encoding="utf-8") as handle:
                self._owns_lock = True
                handle.write(self.run_id)
        except FileExistsError as exc:
            raise FileExistsError(
                f"prediction output is locked by another run: {self.lock_path}. "
                "If no prediction process is writing this output, a previous run "
                "may have crashed; inspect the recorded run id and remove the "
                "stale lock only after confirming that process has stopped."
            ) from exc
        if not self.overwrite and (self.path.exists() or self.manifest_path.exists()):
            raise FileExistsError(f"prediction output already exists: {self.path}")
        self.directory.mkdir()

    def release(self) -> None:
        """Release this coordinator's reservation without deleting staged evidence."""
        if self._owns_lock:
            self.lock_path.unlink()
            self._owns_lock = False

    def write_shard(self, rank: int, records: Iterable[dict[str, Any]]) -> None:
        """Close one lane's shard before publishing its completion manifest."""
        path = self.directory / f"rank-{rank}.jsonl"
        count = 0
        with path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        write_json(self.directory / f"rank-{rank}.manifest.json", {
            "run_id": self.run_id, "rank": rank, "num_samples": count, "sha256": file_digest(path),
        })

    def _read_shards(self, requests: list[PredictionRequest], world_size: int) -> list[dict[str, Any]]:
        records = {}
        for rank in range(world_size):
            path = self.directory / f"rank-{rank}.jsonl"
            manifest = json.loads((self.directory / f"rank-{rank}.manifest.json").read_text(encoding="utf-8"))
            if (manifest["run_id"], manifest["rank"], manifest["sha256"]) != (self.run_id, rank, file_digest(path)):
                raise ValueError(f"invalid prediction shard manifest for rank {rank}")
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    ordinal = record.get("ordinal")
                    if (not isinstance(ordinal, int) or isinstance(ordinal, bool)) or not 0 <= ordinal < len(requests):
                        raise ValueError("prediction shard has an invalid ordinal")
                    if ordinal % world_size != rank or ordinal in records:
                        raise ValueError("duplicate prediction or incorrect lane ownership")
                    if record.get("id") != requests[ordinal].input_id or record.get("sample_idx") != 0:
                        raise ValueError("prediction shard has an unexpected (input_id, sample_idx)")
                    records[ordinal] = record
                    count += 1
            if count != manifest["num_samples"]:
                raise ValueError("prediction shard count does not match its manifest")
        if len(records) != len(requests):
            raise ValueError("prediction shards do not cover every input exactly once")
        return [records[ordinal] for ordinal in range(len(requests))]

    def commit(self, requests: list[PredictionRequest], world_size: int, metadata: dict[str, Any]) -> None:
        """Verify all shards, restore input order, then atomically publish JSONL."""
        records = self._read_shards(requests, world_size)
        merged = self.directory / "predictions.jsonl"
        with merged.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            **metadata, "run_id": self.run_id, "world_size": world_size,
            "num_requests": len(requests), "num_samples": len(records),
            "output_sha256": file_digest(merged),
        }
        staged_manifest = self.directory / "manifest.json"
        write_json(staged_manifest, manifest)
        if self.overwrite:
            os.replace(merged, self.path)
        else:
            # A hard link gives atomic no-clobber semantics on the shared filesystem.
            os.link(merged, self.path)
        os.replace(staged_manifest, self.manifest_path)

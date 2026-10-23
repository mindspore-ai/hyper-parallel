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
"""Versioned CPU wire data, configuration and durable asynchronous snapshots."""
import hashlib
import json
import logging
import math
import os
import pickle
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rl.config import AsyncTrainingConfig, validate_async_config, _require_async as _require, _async_integer as _integer
from rl.dataset.episodes import episode_rows

__all__ = ["AsyncTrainingConfig", "validate_async_config", "SamplingTask", "SampleGroup", "ExperienceBuffer",
           "AsyncCheckpoint", "behavior_segments", "validate_behavior", "decode_group", "configuration_fingerprint"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingTask:
    """An admitted identity bound to committed inference weights."""
    identity: int
    policy_version: int
    source_update: int


def behavior_segments(row: Any, task: SamplingTask) -> tuple[dict, ...]:
    """Retain exact segments, including the legacy single-version representation."""
    segments = row.metadata.get("behavior_segments")
    if segments is None:
        return ({"start": 0, "end": int(row.action_mask.sum()),
                 "policy_version": task.policy_version, "source_update": task.source_update},)
    _require(isinstance(segments, (list, tuple)), "Behavior segments must be a sequence")
    return tuple(segments)


def validate_behavior(row: Any, task: SamplingTask) -> None:
    """Require complete, monotonic action coverage and committed worker identity."""
    end, previous = 0, task.policy_version
    segments = behavior_segments(row, task)
    for segment in segments:
        _require(isinstance(segment, Mapping), "Behavior segment must be a mapping")
        _require(all(_integer(segment.get(key), 0) for key in ("start", "end", "policy_version", "source_update")),
                 "Behavior segments must contain integer boundaries and versions")
        _require(segment["start"] == end and segment["end"] > end and segment["policy_version"] >= previous
                 and segment["policy_version"] == segment["source_update"] + 1,
                 "Behavior segments have gaps, overlaps or invalid policy identities")
        end, previous = segment["end"], segment["policy_version"]
    _require(bool(segments) and end == int(row.action_mask.sum())
             and row.policy_version == row.worker_policy_version == previous,
             "Trajectory behavior segments do not cover actions or match the worker policy")


@dataclass(frozen=True)
class SampleGroup:
    """Complete candidate episodes with immutable behavior probabilities."""
    task: SamplingTask
    trajectories: tuple[Any, ...]
    generation_seconds: float
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def source_updates(self) -> tuple[int, ...]:
        """Return every actual behavior age rather than the admission age."""
        return tuple(part["source_update"] for row in self.trajectories for part in behavior_segments(row, self.task))

    def validate(self, candidates: int) -> None:
        """Validate complete identities, rewards, detached CPU storage and action scores."""
        _require(all(math.isfinite(v) for v in (self.generation_seconds, self.started_at, self.finished_at))
                 and self.generation_seconds >= 0 and 0 <= self.started_at <= self.finished_at,
                 "Invalid generation timing")
        _require(self.task.identity >= 0 and self.task.source_update >= 0
                 and self.task.policy_version == self.task.source_update + 1,
                 "Invalid sample identity or policy version")
        _require(len(episode_rows(self.trajectories)) == candidates
                 and not any(row.metadata.get("dp_padding", False) for row in self.trajectories),
                 "Incomplete GRPO candidate group")
        _require(len({row.trajectory_id for row in self.trajectories}) == len(self.trajectories)
                 and {row.group_id for row in self.trajectories} == {str(self.task.identity)},
                 "Duplicate candidates or inconsistent group identity")
        for row in self.trajectories:
            validate_behavior(row, self.task)
            _require(math.isfinite(row.reward) and row.metadata.get("reward_status") != "pending",
                     "Group rewards must be complete and finite")
            _require(row.rollout_log_probs is not None, "Sampling probabilities are required")
            _require(all(v.device.type == "cpu" and not v.requires_grad
                         for v in (row.token_ids, row.attention_mask, row.action_mask, row.rollout_log_probs)),
                     "Wire tensors must be detached CPU tensors")
            scores = row.rollout_log_probs[row.action_mask[1:].bool()]
            _require(bool(scores.isfinite().all()) and not bool((scores > 1e-5).any()),
                     "Invalid sampled action probabilities")

    def encode(self, candidates: int, max_bytes: int) -> bytes:
        """Own serialized CPU storage independently of multiprocessing tensor handles."""
        self.validate(candidates)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        _require(len(payload) <= max_bytes, "Sample group exceeds max_group_bytes")
        return payload


def decode_group(payload: bytes, candidates: int, max_bytes: int) -> SampleGroup:
    """Revalidate trusted local-process data at the receiving boundary."""
    _require(isinstance(payload, bytes) and len(payload) <= max_bytes, "Invalid or oversized group payload")
    group = pickle.loads(payload)
    _require(isinstance(group, SampleGroup), "Unexpected group payload")
    group.validate(candidates)
    return group


def configuration_fingerprint(config: Mapping[str, Any]) -> str:
    """Keep learning/data identity while allowing relocation and longer resumed runs."""
    value = deepcopy(dict(config))
    asynchronous = value.get("async_training", {})
    asynchronous.pop("audit_dir", None)
    if asynchronous.get("max_candidate_retries", 1) == 1:
        asynchronous.pop("max_candidate_retries", None)
    for key in ("logging", "evaluation"):
        value.pop(key, None)
    for key in ("train_path", "test_path"):
        path = value.get("data", {}).get(key)
        if path and Path(path).is_file():
            with Path(path).open("rb") as handle:
                digest = hashlib.sha256()
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            value["data"][key + "_sha256"] = digest.hexdigest()
    for key in ("checkpoint", "max_steps"):
        value["train"].pop(key, None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class AsyncCheckpoint:
    """Checksum-linked RL state committed after existing model/optimizer persistence."""
    schema = 1

    @staticmethod
    def resolve_path(config: Mapping[str, Any]) -> str | None:
        """Honor explicit restore; otherwise skip incomplete, but not corrupt, snapshots."""
        checkpoint = config["train"]["checkpoint"]
        if checkpoint.get("load_path") is not None:
            return str(checkpoint["load_path"])
        paths = (p for p in Path(checkpoint["output_dir"]).glob("step_*") if p.is_dir() and p.name[5:].isdigit())
        for path in sorted(paths, key=lambda p: int(p.name[5:]), reverse=True):
            if all((path / name).is_file() for name in
                   ("checkpoint_complete.json", "async_complete.json", "async_state.pkl")):
                logger.info("Automatically resuming asynchronous training from %s", path)
                return str(path)
            logger.warning("Skipping incomplete asynchronous checkpoint: %s", path)
        logger.info("No complete asynchronous checkpoint in %s", checkpoint["output_dir"])
        return None

    @classmethod
    def load(cls, directory: str | None, config: Mapping[str, Any]) -> dict | None:
        """Reject corrupt commits, changed learning/data identity and invalid restored steps."""
        if directory is None:
            return None
        root = Path(directory)
        manifest, data = json.loads((root / "async_complete.json").read_text()), (root / "async_state.pkl").read_bytes()
        _require(manifest.get("schema") == cls.schema and manifest.get("sha256") == hashlib.sha256(data).hexdigest(),
                 "Incomplete or corrupted asynchronous checkpoint")
        _require(manifest.get("configuration") == configuration_fingerprint(config),
                 "Asynchronous checkpoint learning/data configuration differs")
        state = pickle.loads(data)
        _require(state["learner_update"] == manifest["learner_update"], "Checkpoint step identity differs")
        _require(config["train"].get("max_steps", state["learner_update"]) >= state["learner_update"],
                 "Resume max_steps is below the checkpoint update")
        _require(state["policy_version"] == state["source_update"] + 1, "Invalid restored policy/source mapping")
        return state

    @staticmethod
    def invalidate(directory: Path) -> None:
        """Hide the old commit before overwriting any model component at this step."""
        (directory / "async_complete.json").unlink(missing_ok=True)

    @classmethod
    def commit(cls, directory: Path, state: dict, config: Mapping[str, Any]) -> None:
        """Publish durable state, then its final manifest; incomplete model state cannot commit."""
        if not (directory / "checkpoint_complete.json").is_file():
            raise RuntimeError("Cannot commit async state before the model checkpoint")
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        manifest = {"schema": cls.schema, "sha256": hashlib.sha256(data).hexdigest(),
                    "learner_update": state["learner_update"], "configuration": configuration_fingerprint(config)}
        payloads = (("async_state.pkl", data), ("async_complete.json", json.dumps(manifest, sort_keys=True).encode()))
        for name, payload in payloads:
            path = directory / name
            pending = path.with_suffix(path.suffix + ".tmp")
            with pending.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            pending.replace(path)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class ExperienceBuffer:
    """Own all admitted identities until consumption or explicit stale eviction."""

    def __init__(self, settings: AsyncTrainingConfig, candidates: int) -> None:
        self.settings = settings
        self.candidates = candidates
        self.next_identity = self.consumed_groups = self.dropped_groups = 0
        self.reserved: dict[int, SamplingTask] = {}
        self.ready: dict[int, bytes] = {}
        self.selected: tuple[int, ...] = ()
        self.completed_intervals: deque = deque(maxlen=1024)

    def reserve(self, version: int, source_update: int) -> SamplingTask | None:
        """Reserve worst-case storage before the producer reads another prompt."""
        occupied = len(self.reserved) + len(self.ready)
        if occupied >= self.settings.max_buffer_groups or len(self.reserved) >= self.settings.max_inflight_groups:
            return None
        task = SamplingTask(self.next_identity, version, source_update)
        self.next_identity += 1
        self.reserved[task.identity] = task
        return task

    def accept(self, payload: bytes) -> None:
        """Accept exactly one result for a live reservation."""
        group = decode_group(payload, self.candidates, self.settings.max_group_bytes)
        if self.reserved.get(group.task.identity) != group.task:
            raise ValueError("Unreserved, duplicated or incorrectly versioned group")
        del self.reserved[group.task.identity]
        self.ready[group.task.identity] = payload
        self.completed_intervals.append((group.started_at, group.finished_at))

    def discard(self, task: SamplingTask) -> None:
        """Release a failed complete reservation without creating a zero-reward sample."""
        if self.reserved.get(task.identity) != task:
            raise ValueError("Unreserved or duplicate failed group")
        del self.reserved[task.identity]
        self.dropped_groups += 1

    def evict_stale(self, update: int) -> int:
        """Evict whole groups; selected data cannot change during optimization."""
        if self.selected:
            raise RuntimeError("Cannot evict during an uncommitted training update")
        stale = []
        for identity, payload in self.ready.items():
            group = decode_group(payload, self.candidates, self.settings.max_group_bytes)
            ages = [update - source for source in group.source_updates]
            if min(ages) < 0:
                raise ValueError("Experience comes from a future training update")
            if max(ages) > self.settings.max_policy_lag_updates:
                stale.append(identity)
        for identity in stale:
            del self.ready[identity]
        self.dropped_groups += len(stale)
        return len(stale)

    def select(self, count: int) -> tuple[SampleGroup, ...]:
        """Lease a complete global batch without releasing its capacity."""
        if self.selected or len(self.ready) < count:
            raise RuntimeError("No complete batch or an update is already selected")
        self.selected = tuple(list(self.ready)[:count])
        return tuple(decode_group(self.ready[key], self.candidates, self.settings.max_group_bytes)
                     for key in self.selected)

    def commit(self) -> None:
        """Release selected identities only after all training ranks succeed."""
        if not self.selected:
            raise RuntimeError("No selected update to commit")
        for identity in self.selected:
            del self.ready[identity]
        self.consumed_groups += len(self.selected)
        self.selected = ()

    def overlap_seconds(self, started: float, finished: float) -> float:
        """Return a lower bound on task/update overlap from received completion timestamps."""
        overlap = max((max(0.0, min(end, finished) - max(begin, started))
                       for begin, end in self.completed_intervals), default=0.0)
        self.completed_intervals.clear()
        return overlap

    def state_dict(self) -> dict[str, Any]:
        """Export a quiescent buffer; no in-flight or partial update is resumable."""
        if self.reserved or self.selected:
            raise RuntimeError("Checkpoint requires a drained producer and committed update")
        return {"next_identity": self.next_identity, "ready": list(self.ready.values()),
                "consumed_groups": self.consumed_groups, "dropped_groups": self.dropped_groups}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot and validate capacity, identities and accounting."""
        if self.reserved or self.ready or self.selected:
            raise RuntimeError("Restore requires an empty buffer")
        next_identity = state["next_identity"]
        ready = {}
        for payload in state["ready"]:
            group = decode_group(payload, self.candidates, self.settings.max_group_bytes)
            identity = group.task.identity
            if identity in ready or identity >= next_identity:
                raise ValueError("Invalid restored group identity")
            ready[identity] = payload
        consumed, dropped = state["consumed_groups"], state["dropped_groups"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (next_identity, consumed, dropped)):
            raise ValueError("Invalid restored buffer counters")
        if len(ready) > self.settings.max_buffer_groups or next_identity != consumed + dropped + len(ready):
            raise ValueError("Restored buffer violates capacity or identity conservation")
        self.next_identity, self.ready = next_identity, ready
        self.consumed_groups, self.dropped_groups = consumed, dropped

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
"""Synchronous row-interval sidecars committed before dense DCP save."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader

from hyper_parallel.models.external_state import CheckpointRuntime

_VERSION = 1
_DIR = "engram_host"


@dataclass(frozen=True)
class HostSidecarLoadPlan:
    """Validated sidecar data carried through one DCP restore attempt.

    The schema dictionary is read-only by convention; freezing this object does
    not make nested JSON values immutable.
    """

    step_dir: str
    persisted_optimizer_keys: frozenset[str]
    schema: dict[str, Any]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_digest(state) -> str:
    path = getattr(getattr(state.model, "config", None), "v41_engram_assets_path", None)
    return _sha_file(Path(path)) if path else ""


def _topology(mesh) -> dict[str, int]:
    return {name: int(getattr(mesh, f"{name}_size", 1)) for name in
            ("dp", "cp", "tp", "ep", "pp", "edp_shard")}


def _sparse_moment_snapshot(state, table) -> dict[str, Any]:
    """Read moments for saving without creating optimizer state."""
    sparse = state.sparse_optimizer
    if sparse is None:
        raise RuntimeError("Host table has no SparseAdam optimizer")
    moments = sparse.state.get(table.weight, {})
    exp_avg = moments.get("exp_avg")
    exp_avg_sq = moments.get("exp_avg_sq")
    return {
        "step": moments.get("step", 0),
        "exp_avg": exp_avg if exp_avg is not None else torch.zeros_like(table.weight),
        "exp_avg_sq": exp_avg_sq if exp_avg_sq is not None else torch.zeros_like(table.weight),
    }


def _initialize_sparse_moments(state, table) -> dict[str, Any]:
    """Create owner-local moments only for restore or explicit reset."""
    sparse = state.sparse_optimizer
    if sparse is None:
        raise RuntimeError("Host table has no SparseAdam optimizer")
    moments = sparse.state[table.weight]
    moments.setdefault("step", 0)
    if "exp_avg" not in moments:
        moments["exp_avg"] = torch.zeros_like(table.weight)
    if "exp_avg_sq" not in moments:
        moments["exp_avg_sq"] = torch.zeros_like(table.weight)
    return moments


def _fingerprint(table, moments) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(moments["step"])).encode())
    for tensor in (table.weight, moments["exp_avg"], moments["exp_avg_sq"]):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("Host sidecar tensors must be CPU FP32")
        digest.update(memoryview(tensor.detach().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def _all_rank_objects(value):
    if not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _validate_files(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check owner replicas agree and keep one writer record per owner."""
    by_owner = {}
    for record in records:
        key = (record["stage"], record["fqn"], record["ep_rank"])
        previous = by_owner.setdefault(key, record)
        if previous["fingerprint"] != record["fingerprint"]:
            raise ValueError(f"Host owner replicas disagree: {key}")
        if previous["start"] != record["start"] or previous["rows"] != record["rows"]:
            raise ValueError(f"Host owner replica interval mismatch: {key}")
        if record.get("file"):
            if previous.get("file") and previous is not record:
                raise ValueError(f"Host owner has multiple sidecar writers: {key}")
            by_owner[key] = record
    files = sorted(by_owner.values(), key=lambda entry: (entry["stage"], entry["fqn"], entry["start"]))
    by_layer = {}
    for entry in files:
        by_layer.setdefault((entry["stage"], entry["fqn"]), []).append(entry)
        if not entry.get("file"):
            raise ValueError("Host owner has no sidecar file")
    for (stage, fqn), entries in by_layer.items():
        cursor = 0
        for entry in entries:
            if entry["start"] != cursor:
                raise ValueError(f"Host row coverage gap or overlap: {stage}:{fqn}")
            cursor += entry["rows"]
        if cursor != entries[0]["physical_rows"]:
            raise ValueError(f"Host row coverage incomplete: {stage}:{fqn}")
    return files


def _write_owner_record(state: Any, root: Path, stage: int,
                        fqn: str, table: Any) -> dict[str, Any]:
    """Write one owner file and return its replica fingerprint and interval."""
    moments = _sparse_moment_snapshot(state, table)
    record = {
        "stage": stage, "fqn": fqn, "ep_rank": table.ep_rank,
        "ep_size": table.ep_size, "start": table.global_row_start,
        "rows": table.rows_per_owner, "physical_rows": table.physical_rows,
        "logical_rows": table.logical_rows, "width": table.width,
        "fingerprint": _fingerprint(table, moments),
    }
    if state.is_sparse_norm_representative():
        name = f"pp_{stage:04d}_{fqn.replace('.', '_')}_ep_{table.ep_rank:05d}.pt"
        path = root / name
        temporary = root / f".{name}.{os.getpid()}.tmp"
        torch.save({
            "weight": table.weight.detach(),
            "step": int(moments["step"]),
            "exp_avg": moments["exp_avg"],
            "exp_avg_sq": moments["exp_avg_sq"],
        }, temporary)
        os.replace(temporary, path)
        record.update({"file": name, "sha256": _sha_file(path)})
    return record


def _consistent_sparse_hparams(gathered: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose nonempty optimizer settings and verify all Host stages agree."""
    settings = [entry["sparse_hparams"] for entry in gathered if entry["sparse_hparams"]]
    if settings and any(item != settings[0] for item in settings):
        raise ValueError("Host sparse optimizer settings differ between ranks")
    return settings[0] if settings else {}


def _publish_schema(root: Path, schema: dict[str, Any]) -> None:
    """Atomically publish schema on rank zero and align write failures."""
    schema_error = None
    try:
        if not dist.is_initialized() or dist.get_rank() == 0:
            temporary = root / f".schema.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(schema, sort_keys=True), encoding="utf-8")
            os.replace(temporary, root / "schema.json")
    except (OSError, TypeError, ValueError) as exc:
        schema_error = str(exc)
    errors = [message for message in _all_rank_objects(schema_error) if message]
    if errors:
        raise RuntimeError(f"Host sidecar schema write failed: {errors[0]}")


def before_checkpoint_save(state: Any, runtime: CheckpointRuntime) -> None:
    """Write one file per owner, validate all replicas, then publish schema.

    Args:
        state: Trainer or external model state.
        runtime: Checkpoint runtime and topology context.
    """
    if not runtime.include_optimizer or not runtime.include_train_state:
        raise ValueError("Host checkpoint requires optimizer and train state")
    root = Path(runtime.step_dir) / _DIR
    root.mkdir(parents=True, exist_ok=True)
    mesh = runtime.mesh_context
    stage = int(getattr(mesh, "pp_rank", 0))
    records = []
    error = None
    try:
        for fqn, table in state.tables.items():
            records.append(_write_owner_record(state, root, stage, fqn, table))
    except (OSError, RuntimeError, ValueError) as exc:
        error = str(exc)
    hparams = state.sparse_optimizer.param_groups[0].copy() if state.sparse_optimizer else {}
    hparams.pop("params", None)
    gathered = _all_rank_objects({
        "records": records, "error": error, "sparse_hparams": hparams,
        "optimizer_keys": sorted(runtime.persisted_optimizer_keys),
    })
    errors = [entry["error"] for entry in gathered if entry["error"]]
    if errors:
        raise RuntimeError(f"Host sidecar save failed: {errors[0]}")
    files = _validate_files([record for entry in gathered for record in entry["records"]])
    hparams = _consistent_sparse_hparams(gathered)
    optimizer_keys_by_rank = [entry["optimizer_keys"] for entry in gathered]
    schema = {
        "format_version": _VERSION,
        "global_step": runtime.global_step,
        "assets_digest": _asset_digest(state),
        "topology": _topology(mesh),
        "files": files,
        "sparse_hparams": hparams,
        "current_lr": hparams.get("lr"),
        "dense_optimizer_keys": sorted({key for keys in optimizer_keys_by_rank for key in keys}),
        "dense_optimizer_keys_by_rank": optimizer_keys_by_rank,
    }
    _publish_schema(root, schema)


def _validate_dense_metadata(step_dir: str, expected: list[str]) -> None:
    """Require DCP optimizer leaves to equal the sidecar's exact manifest."""
    if not expected:
        return
    metadata = FileSystemReader(step_dir).read_metadata()
    persisted = {
        key.removeprefix("optimizer.") for key in metadata.state_dict_metadata
        if key.startswith("optimizer.")
    }
    if persisted != set(expected):
        raise ValueError("Dense optimizer DCP metadata differs from Host sidecar manifest")


def _validate_sparse_optimizer_hparams(state: Any, schema: dict[str, Any]) -> None:
    """Require the same SparseAdam settings apart from the scheduled LR."""
    if state.sparse_optimizer is None:
        return
    current = state.sparse_optimizer.param_groups[0].copy()
    current.pop("params", None)
    current.pop("lr", None)
    saved = schema["sparse_hparams"].copy()
    saved.pop("lr", None)
    if json.loads(json.dumps(current, sort_keys=True)) != saved:
        raise ValueError("Host SparseAdam hyperparameters differ from checkpoint")


def _validate_load_schema(state: Any, runtime: CheckpointRuntime,
                          root: Path, schema: dict[str, Any]) -> None:
    """Validate topology, dense metadata, owner checksums and local intervals."""
    if schema["format_version"] != _VERSION or schema["assets_digest"] != _asset_digest(state):
        raise ValueError("Host sidecar version or asset digest mismatch")
    if schema["topology"] != _topology(runtime.mesh_context):
        raise ValueError("Host sidecar topology mismatch")
    _validate_sparse_optimizer_hparams(state, schema)
    files = _validate_files(schema["files"])
    _validate_dense_metadata(runtime.step_dir, schema["dense_optimizer_keys"])
    for entry in files:
        path = root / entry["file"]
        if _sha_file(path) != entry["sha256"]:
            raise ValueError(f"Host sidecar checksum mismatch: {path}")
    for fqn, table in state.tables.items():
        matches = [entry for entry in files if entry["fqn"] == fqn
                   and entry["stage"] == getattr(runtime.mesh_context, "pp_rank", 0)
                   and entry["ep_rank"] == table.ep_rank]
        if len(matches) != 1 or matches[0]["start"] != table.global_row_start:
            raise ValueError(f"Host sidecar owner interval mismatch: {fqn}")


def _local_optimizer_keys(schema: dict[str, Any], rank: int,
                          world_size: int) -> frozenset[str]:
    """Validate every rank's optimizer-key manifest before collective success."""
    by_rank = schema["dense_optimizer_keys_by_rank"]
    if not isinstance(by_rank, list) or len(by_rank) != world_size or not 0 <= rank < world_size:
        raise ValueError("Host sidecar dense optimizer rank manifest mismatch")
    for keys in by_rank:
        if (not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
                or len(keys) != len(set(keys))):
            raise ValueError("Host sidecar dense optimizer rank keys are invalid")
    return frozenset(by_rank[rank])


def before_checkpoint_load(state: Any, runtime: CheckpointRuntime) -> HostSidecarLoadPlan:
    """Validate complete sidecar coverage, checksums and exact topology.

    Args:
        state: Trainer or external model state.
        runtime: Checkpoint runtime and topology context.
    """
    if runtime.include_optimizer != runtime.include_train_state:
        raise ValueError("Host checkpoint restore requires both optimizer and train state or neither")
    root = Path(runtime.step_dir) / _DIR
    error = None
    schema = None
    local_keys = frozenset()
    try:
        schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
        _validate_load_schema(state, runtime, root, schema)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_keys = _local_optimizer_keys(schema, rank, world_size)
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        error = str(exc)
    errors = [message for message in _all_rank_objects(error) if message]
    if errors:
        raise RuntimeError(f"Host sidecar load failed: {errors[0]}")
    return HostSidecarLoadPlan(runtime.step_dir, local_keys, schema)


def _restore_owner_payload(state: Any, runtime: CheckpointRuntime, root: Path,
                           schema: dict[str, Any], fqn: str, table: Any) -> None:
    """Restore one owner row interval and its optional optimizer moments."""
    entry = next(item for item in schema["files"] if item["fqn"] == fqn
                 and item["stage"] == getattr(runtime.mesh_context, "pp_rank", 0)
                 and item["ep_rank"] == table.ep_rank)
    payload = torch.load(root / entry["file"], map_location="cpu", weights_only=True)
    if tuple(payload["weight"].shape) != tuple(table.weight.shape):
        raise ValueError(f"Host sidecar row shape mismatch: {fqn}")
    with torch.no_grad():
        table.weight.copy_(payload["weight"])
    if runtime.include_optimizer:
        moments = _initialize_sparse_moments(state, table)
        moments["step"] = int(payload["step"])
        for name in ("exp_avg", "exp_avg_sq"):
            if tuple(payload[name].shape) != tuple(table.weight.shape):
                raise ValueError(f"Host sidecar moment shape mismatch: {fqn}:{name}")
            moments[name].copy_(payload[name])
    table.clear_step()


def after_checkpoint_load(state: Any, runtime: CheckpointRuntime,
                          load_plan: HostSidecarLoadPlan) -> None:
    """Restore this rank's owner-local rows and optional SparseAdam moments.

    Args:
        state: Trainer or external model state.
        runtime: Checkpoint runtime and topology context.
        load_plan: Validated schema from this restore attempt.
    """
    root = Path(runtime.step_dir) / _DIR
    plan_error = None
    try:
        if not isinstance(load_plan, HostSidecarLoadPlan):
            raise TypeError("Host sidecar load plan has an invalid type")
        if load_plan.step_dir != runtime.step_dir:
            raise ValueError("Host sidecar load plan belongs to a different checkpoint")
        if not isinstance(load_plan.schema, dict):
            raise TypeError("Host sidecar load plan schema has an invalid type")
    except (TypeError, ValueError) as exc:
        plan_error = str(exc)
    errors = [message for message in _all_rank_objects(plan_error) if message]
    if errors:
        raise RuntimeError(f"Host sidecar load failed: {errors[0]}")

    error = None
    try:
        schema = load_plan.schema
        if runtime.include_optimizer and state.sparse_optimizer is not None:
            state.sparse_optimizer.param_groups[0]["lr"] = schema["current_lr"]
        for fqn, table in state.tables.items():
            _restore_owner_payload(state, runtime, root, schema, fqn, table)
    except (EOFError, OSError, KeyError, RuntimeError, StopIteration,
            TypeError, ValueError, pickle.UnpicklingError) as exc:
        error = str(exc)
    errors = [message for message in _all_rank_objects(error) if message]
    if errors:
        raise RuntimeError(f"Host sidecar load failed: {errors[0]}")


def after_weights_only_load(state: Any) -> None:
    """Discard sparse optimizer moments without modifying restored weights.

    Args:
        state: Trainer or external model state.
    """
    if state.sparse_optimizer is not None:
        state.sparse_optimizer.state.clear()
        for table in state.tables.values():
            _initialize_sparse_moments(state, table)

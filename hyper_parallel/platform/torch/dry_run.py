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
"""No-accelerator memory analysis for the Torch Trainer path."""
import copy
import csv
import json
import logging
import os
import re
import tempfile
import traceback
import weakref
from pathlib import Path
from typing import Any, Dict, Optional


_SCHEMA_VERSION = 2
_GIB = 2 ** 30
_PERSISTENT_END_INDEX = 2 ** 63 - 1
_SMALL_ALLOCATION_BYTES = 2 ** 20
_VIRTUAL_ADDRESS_BASE = 0xF00000000000
_VIRTUAL_ADDRESS_ALIGNMENT = 512
_CSV_FIELDS = (
    "start_time_stamp",
    "end_time_stamp",
    "device_addr",
    "stream_id",
    "pool_type",
    "size",
    "actual_used_memory",
    "actual_peak_memory",
    "file_name",
    "line_num",
    "type",
    "producer_task",
    "task_name",
    "node_name",
    "graph_name",
    "user_tasks",
    "last_user_task",
    "python_stack",
    "is_persistent",
    "is_small",
)
logger = logging.getLogger(__name__)
_LIMITATIONS = [
    "Reports logical live tensor bytes, not allocator reserved memory or fragmentation.",
    "Kernel workspaces and opaque third-party fused-operator allocations are not tracked.",
    "Checkpoint loading, dataloaders, callbacks, and gradient accumulation are not simulated.",
    "Pipeline parallelism and data-dependent dynamic shapes are not supported in this version.",
]


def _report_limitations(metadata: Optional[Dict[str, Any]] = None) -> list[str]:
    """Return static limitations plus run-specific simulation caveats."""
    limitations = list(_LIMITATIONS)
    metadata = metadata or {}
    target = metadata.get("device_type")
    simulation = metadata.get("simulation_device_type")
    if target and simulation and target != simulation:
        limitations.append(
            f"The logical {target} run was simulated with {simulation} FakeTensors; "
            "target-backend custom operators, device guards, and allocator behavior "
            "were not exercised."
        )
    return limitations


def _category_name(category: Any) -> str:
    """Return a stable JSON key for a MemTracker category."""
    if isinstance(category, str):
        return category
    value = getattr(category, "value", None)
    return str(value if value is not None else category)


def _normalize_snapshot(snapshot: Dict[Any, Dict[Any, int]]) -> Dict[str, Dict[str, int]]:
    """Convert device/category objects in a MemTracker snapshot to JSON keys."""
    normalized = {}
    for device, categories in snapshot.items():
        normalized[str(device)] = {
            _category_name(category): int(value)
            for category, value in categories.items()
        }
    return normalized


def _snapshot_total(snapshot: Dict[str, Dict[str, int]], device_type: str) -> int:
    """Sum total bytes for devices matching ``device_type``."""
    return sum(
        categories.get("Total", 0)
        for device, categories in snapshot.items()
        if device.split(":", maxsplit=1)[0] == device_type
    )


def _snapshot_breakdown(snapshot: Dict[str, Dict[str, int]], device_type: str) -> Dict[str, int]:
    """Aggregate category bytes for devices matching ``device_type``."""
    breakdown: Dict[str, int] = {}
    for device, categories in snapshot.items():
        if device.split(":", maxsplit=1)[0] != device_type:
            continue
        for category, value in categories.items():
            if category == "Total":
                continue
            breakdown[category] = breakdown.get(category, 0) + int(value)
    return dict(sorted(breakdown.items()))


def _module_local_peak(module_stats: Any, device_type: str) -> int:
    """Return a module's largest local peak on the target device type."""
    peaks = [
        int(value)
        for device, value in getattr(module_stats, "local_peak", {}).items()
        if str(device).split(":", maxsplit=1)[0] == device_type
    ]
    return max(peaks, default=0)


def _serialize_module(module_stats: Any, device_type: str) -> Dict[str, Any]:
    """Serialize one MemTracker module-stat object without private enum types."""
    snapshots = {}
    for state, snapshot_list in getattr(module_stats, "snapshots", {}).items():
        snapshots[_category_name(state)] = [
            _normalize_snapshot(snapshot)
            for snapshot in snapshot_list
        ]
    return {
        "fqn": str(module_stats.mod_fqn),
        "parameter_bytes": int(getattr(module_stats, "parameter_mem", 0)),
        "buffer_bytes": int(getattr(module_stats, "buffer_mem", 0)),
        "input_bytes": int(getattr(module_stats, "input_mem", 0)),
        "output_bytes": int(getattr(module_stats, "output_mem", 0)),
        "local_peak_bytes": _module_local_peak(module_stats, device_type),
        "snapshots": snapshots,
    }


def _capture_python_stack() -> Dict[str, Any]:
    """Capture a formatted stack and its leaf caller location."""
    frames = traceback.extract_stack(limit=64)
    kept_frames = []
    for frame in frames:
        normalized_path = frame.filename.replace("\\", "/")
        if normalized_path == __file__ and frame.name in (
                "__torch_dispatch__", "_capture_python_stack"
        ):
            continue
        if normalized_path.endswith(("/torch/_compile.py", "/torch/_dynamo/eval_frame.py")):
            continue
        kept_frames.append(frame)
    python_stack = "|".join(
        (
            f"File:{frame.filename};Line:{frame.lineno};Function:{frame.name}"
        )
        for frame in kept_frames
    )
    leaf_frame = kept_frames[-1] if kept_frames else None
    return {
        "python_stack": python_stack,
        "file_name": leaf_frame.filename if leaf_frame else "",
        "line_num": leaf_frame.lineno if leaf_frame else 0,
    }


def _operator_phase(tracker: Any) -> str:
    """Resolve the current training phase from MemTracker state."""
    if getattr(tracker, "_in_opt", False):
        return "optimizer"
    from hyper_parallel.core.activation_checkpoint.recompute_state import (  # pylint: disable=C0415
        is_recomputing,
    )
    if is_recomputing():
        return "recompute"
    module_tracker = getattr(tracker, "_mod_tracker", None)
    if getattr(module_tracker, "is_bw", False):
        return "backward"
    return "forward"


def _active_module_fqn(tracker: Any) -> str:
    """Return the deepest active module as optional operator context."""
    module_tracker = getattr(tracker, "_mod_tracker", None)
    parents = getattr(module_tracker, "parents", ())
    candidates = [str(parent) for parent in parents if str(parent) not in ("", "Global")]
    return max(candidates, key=lambda fqn: (fqn.count("."), len(fqn)), default="")


def _create_operator_trace_mode(tracker: Any, device_type: str) -> Any:
    """Create a lazy TorchDispatchMode that records logical memory blocks."""
    import torch  # pylint: disable=C0415
    from torch.distributed._tools.mem_tracker import get_untyped_storages  # pylint: disable=C0415
    from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=C0415
    from torch.utils._pytree import tree_flatten  # pylint: disable=C0415

    def _storage_map(values: Any) -> Dict[int, Dict[str, Any]]:
        """Return storage identifiers, logical devices, and byte sizes."""
        flat_values, _ = tree_flatten(values)
        storages = {}
        for value in flat_values:
            if not isinstance(value, torch.Tensor):
                continue
            try:
                tensor_storages = get_untyped_storages(value)
            except (RuntimeError, TypeError):
                continue
            for storage in tensor_storages:
                storage_key = int(getattr(storage, "_cdata", id(storage)))
                storages[storage_key] = {
                    "device": str(value.device),
                    "size": int(storage.size()),
                    "storage": storage,
                }
        return storages

    def _tracked_storage_info() -> Dict[int, Dict[str, Any]]:
        """Read live category/device metadata from MemTracker storage refs."""
        tracked = {}
        for storage, (winfo, storage_ref) in tracker._WINFO.items():
            storage_key = int(getattr(storage, "_cdata", id(storage)))
            tracked[storage_key] = {
                "device": str(winfo.device),
                "size": int(winfo.mem_consumed),
                "type": _category_name(winfo.reftype),
                "storage_ref": storage_ref,
            }
        return tracked

    class _OperatorTraceMode(TorchDispatchMode):
        """Track producer, consumer, and lifetime data for fake storages."""

        def __init__(self) -> None:
            """Initialize ordered logical task and storage state."""
            super().__init__()
            self.memory_blocks: list[Dict[str, Any]] = []
            self._active_blocks: Dict[int, Dict[str, Any]] = {}
            self._active_storage_refs: Dict[int, Any] = {}
            self._active_storage_sizes: Dict[int, int] = {}
            self._memtracker_storage_keys: set[int] = set()
            self._released_storage_keys: set[int] = set()
            self._next_task_index = 0
            self._next_virtual_address = _VIRTUAL_ADDRESS_BASE

        def _close_blocks(self, storage_keys: set[int], end_index: int) -> None:
            """Close active blocks and discard their lifetime bookkeeping."""
            for storage_key in storage_keys & set(self._active_blocks):
                block = self._active_blocks.pop(storage_key)
                self._active_storage_refs.pop(storage_key)
                self._active_storage_sizes.pop(storage_key)
                self._memtracker_storage_keys.discard(storage_key)
                self._released_storage_keys.discard(storage_key)
                block["end_time_stamp"] = end_index

        def _close_released_blocks(self, end_index: int) -> None:
            """Close blocks whose storage weak refs disappeared."""
            released_keys = self._released_storage_keys & set(self._active_blocks)
            self._close_blocks(released_keys, end_index)

        def _allocate_virtual_address(self, size: int) -> str:
            """Return a deterministic aligned address in a reserved fake range."""
            address = self._next_virtual_address
            aligned_size = max(
                _VIRTUAL_ADDRESS_ALIGNMENT,
                (
                    (size + _VIRTUAL_ADDRESS_ALIGNMENT - 1)
                    // _VIRTUAL_ADDRESS_ALIGNMENT
                ) * _VIRTUAL_ADDRESS_ALIGNMENT,
            )
            self._next_virtual_address += aligned_size
            return f"0x{address:x}"

        def finalize(self) -> None:
            """Close released blocks and mark remaining storages persistent."""
            self._close_released_blocks(self._next_task_index)
            tracked_storages = _tracked_storage_info()
            for storage_key, block in self._active_blocks.items():
                tracked_storage = tracked_storages.get(storage_key)
                if tracked_storage is None or tracked_storage["size"] <= 0:
                    block["end_time_stamp"] = self._next_task_index
                    continue
                block["type"] = tracked_storage["type"]
                block["end_time_stamp"] = _PERSISTENT_END_INDEX
                block["is_persistent"] = 1
            self._active_blocks.clear()
            self._active_storage_refs.clear()
            self._active_storage_sizes.clear()
            self._memtracker_storage_keys.clear()
            self._released_storage_keys.clear()

        def _classify_output_keys(
                self,
                input_storages: Dict[int, Dict[str, Any]],
                output_storages: Dict[int, Dict[str, Any]],
                active_input_keys: set[int],
        ) -> tuple[set[int], set[int]]:
            """Return resized inputs and newly allocated target-device outputs."""
            resized_output_keys = {
                storage_key
                for storage_key in active_input_keys & set(output_storages)
                if output_storages[storage_key]["size"] != self._active_storage_sizes[storage_key]
            }
            new_output_keys = {
                storage_key
                for storage_key, storage in output_storages.items()
                if (
                    storage_key not in input_storages
                    and storage_key not in self._active_blocks
                    and storage["device"].split(":", maxsplit=1)[0] == device_type
                    and storage["size"] > 0
                )
            }
            new_output_keys.update(
                storage_key
                for storage_key in resized_output_keys
                if (
                    output_storages[storage_key]["device"].split(":", maxsplit=1)[0] == device_type
                    and output_storages[storage_key]["size"] > 0
                )
            )
            return resized_output_keys, new_output_keys

        def _inactive_tracked_keys(
                self,
                tracked_storages: Dict[int, Dict[str, Any]],
        ) -> set[int]:
            """Return MemTracker storages that were removed or resized to zero."""
            return {
                storage_key
                for storage_key in self._memtracker_storage_keys
                if storage_key not in tracked_storages or tracked_storages[storage_key]["size"] <= 0
            }

        def _record_input_users(self, active_input_keys: set[int], task_index: int) -> None:
            """Append the task index to every active input storage's user list."""
            for storage_key in active_input_keys:
                user_tasks = self._active_blocks[storage_key]["user_tasks"]
                if not user_tasks or user_tasks[-1] != task_index:
                    user_tasks.append(task_index)

        def _track_new_outputs(
                self,
                output_keys: set[int],
                output_storages: Dict[int, Dict[str, Any]],
                tracked_storages: Dict[int, Dict[str, Any]],
                context: Dict[str, Any],
        ) -> None:
            """Create logical memory blocks and weak lifetime refs for outputs."""
            for storage_key in sorted(output_keys):
                output_storage = output_storages[storage_key]
                tracked_storage = tracked_storages.get(storage_key, {})
                device = output_storage["device"]
                size = int(tracked_storage.get("size", output_storage["size"]))
                block = {
                    "start_time_stamp": context["task_index"],
                    "end_time_stamp": _PERSISTENT_END_INDEX,
                    "device_addr": self._allocate_virtual_address(size),
                    "stream_id": 0,
                    "pool_type": "FakeTensorLogicalMemoryPool",
                    "size": size,
                    "actual_used_memory": context["after"].get(device, {}).get("Total", 0),
                    "actual_peak_memory": context["running_peak"].get(device, {}).get("Total", 0),
                    "file_name": context["stack"]["file_name"],
                    "line_num": context["stack"]["line_num"],
                    "type": tracked_storage.get("type", "Activation"),
                    "producer_task": context["task_index"],
                    "task_name": context["phase"],
                    "node_name": context["operator_name"],
                    "graph_name": context["module_fqn"],
                    "user_tasks": [],
                    "python_stack": context["stack"]["python_stack"],
                    "is_persistent": 0,
                    "is_small": int(size < _SMALL_ALLOCATION_BYTES),
                }
                self.memory_blocks.append(block)
                self._active_blocks[storage_key] = block
                self._active_storage_sizes[storage_key] = output_storage["size"]
                storage_ref = tracked_storage.get("storage_ref")
                if storage_ref is not None:
                    self._memtracker_storage_keys.add(storage_key)
                storage = storage_ref() if storage_ref is not None else None
                if storage is None:
                    storage = output_storage["storage"]
                self._active_storage_refs[storage_key] = weakref.ref(
                    storage,
                    lambda _, key=storage_key: self._released_storage_keys.add(key),
                )

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self._close_released_blocks(self._next_task_index)
            input_storages = _storage_map((args, kwargs))
            active_input_keys = set(input_storages) & set(self._active_blocks)
            phase = _operator_phase(tracker)
            module_fqn = _active_module_fqn(tracker)
            result = func(*args, **(kwargs or {}))
            after = _normalize_snapshot(tracker.get_tracker_snapshot("current"))
            running_peak = _normalize_snapshot(tracker.get_tracker_snapshot("peak"))
            output_storages = _storage_map(result)
            resized_output_keys, new_output_keys = self._classify_output_keys(
                input_storages,
                output_storages,
                active_input_keys,
            )
            tracked_storages = _tracked_storage_info()
            inactive_tracked_keys = self._inactive_tracked_keys(tracked_storages)
            if not active_input_keys and not new_output_keys and not resized_output_keys:
                self._close_blocks(inactive_tracked_keys, self._next_task_index)
                return result
            task_index = self._next_task_index
            self._next_task_index += 1
            self._record_input_users(active_input_keys, task_index)
            self._close_blocks(
                resized_output_keys | inactive_tracked_keys,
                task_index,
            )
            self._track_new_outputs(
                new_output_keys,
                output_storages,
                tracked_storages,
                {
                    "task_index": task_index,
                    "after": after,
                    "running_peak": running_peak,
                    "stack": _capture_python_stack(),
                    "phase": phase,
                    "operator_name": str(func),
                    "module_fqn": module_fqn,
                },
            )
            return result

    return _OperatorTraceMode()


def build_memory_report(
        tracker: Any,
        metadata: Dict[str, Any],
        device_type: str,
        module_depth: int,
        top_modules: int,
        device_memory_gib: Optional[float] = None,
        memory_blocks: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the stable JSON report from a completed MemTracker run.

    Args:
        tracker: Completed memory-tracker instance.
        metadata: Run metadata to include verbatim.
        device_type: Device type used for FakeTensor simulation. This is the
            logical accelerator type when its backend is available, otherwise
            ``"cpu"`` for the no-backend fallback.
        module_depth: Maximum module FQN depth, or zero for unlimited.
        top_modules: Maximum number of hottest modules, or zero for all.
        device_memory_gib: Optional target capacity for OOM-risk reporting.
        memory_blocks: Optional logical output-storage lifetime records.

    Returns:
        JSON-serializable report dictionary.

    Raises:
        ValueError: If the tracker observed no memory on the simulation device.
    """
    peak = _normalize_snapshot(tracker.get_tracker_snapshot("peak"))
    current = _normalize_snapshot(tracker.get_tracker_snapshot("current"))
    peak_bytes = _snapshot_total(peak, device_type)
    if peak_bytes <= 0:
        raise ValueError(
            f"MemTracker observed no {device_type!r} tensor memory; "
            f"tracked devices are {sorted(peak)}"
        )

    modules = []
    for module_stats in tracker.memory_tracking.values():
        fqn = str(module_stats.mod_fqn)
        depth = fqn.count(".") + 1
        if module_depth and depth > module_depth:
            continue
        modules.append(_serialize_module(module_stats, device_type))
    modules.sort(key=lambda item: (-item["local_peak_bytes"], item["fqn"]))
    if top_modules:
        modules = modules[:top_modules]

    capacity_bytes = None
    headroom_bytes = None
    oom_risk = None
    if device_memory_gib is not None:
        capacity_bytes = int(device_memory_gib * _GIB)
        headroom_bytes = capacity_bytes - peak_bytes
        oom_risk = headroom_bytes < 0

    devices = {}
    for device in sorted(set(peak) | set(current)):
        devices[device] = {
            "peak": peak.get(device, {}),
            "current": current.get(device, {}),
        }
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "ok",
        "metadata": metadata,
        "summary": {
            "peak_bytes": peak_bytes,
            "current_bytes": _snapshot_total(current, device_type),
            "peak_gib": peak_bytes / _GIB,
            "capacity_bytes": capacity_bytes,
            "headroom_bytes": headroom_bytes,
            "oom_risk": oom_risk,
            "peak_breakdown_bytes": _snapshot_breakdown(peak, device_type),
            "current_breakdown_bytes": _snapshot_breakdown(current, device_type),
        },
        "devices": devices,
        "modules": modules,
        "memory_blocks": memory_blocks or [],
        "limitations": _report_limitations(metadata),
    }


def write_memory_report(report: Dict[str, Any], output_path: str) -> str:
    """Atomically write a memory report.

    Args:
        report: JSON-serializable report dictionary.
        output_path: Destination JSON path.

    Returns:
        Absolute destination path.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, ensure_ascii=False, indent=2, sort_keys=True)
            report_file.write("\n")
        os.replace(temporary_path, destination)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return str(destination)


def build_memory_csv_rows(report: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Build MindSpore-compatible logical memory-block CSV rows.

    Args:
        report: Successful report returned by :func:`build_memory_report`.

    Returns:
        One row per new non-aliasing operator output storage. No summary rows.

    Raises:
        ValueError: If ``report`` does not describe a successful run.
    """
    if report.get("status") != "ok":
        raise ValueError("CSV output requires a successful HyperDryRun report")
    rows = []
    for block in report.get("memory_blocks", []):
        user_tasks = list(block.get("user_tasks", []))
        row = {
            field: block.get(field, "")
            for field in _CSV_FIELDS
        }
        row["user_tasks"] = "{" + "-".join(str(task) for task in user_tasks) + "}"
        row["last_user_task"] = user_tasks[-1] if user_tasks else ""
        rows.append(row)
    return rows


def write_memory_csv(report: Dict[str, Any], output_path: str) -> str:
    """Atomically write the tabular companion to a JSON memory report.

    Args:
        report: Successful JSON report dictionary.
        output_path: Destination CSV path.

    Returns:
        Absolute destination path.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = build_memory_csv_rows(report)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, destination)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return str(destination)


class TorchDryRunRunner:
    """Drive one fake Torch Trainer step and emit its memory report.

    Args:
        args: Parsed ``HyperTrainerConfig``.
    """

    def __init__(self, args: Any) -> None:
        """Initialize the runner without touching distributed or device state."""
        self.args = args
        self._stage = "validation"

    @staticmethod
    def _check_torch_version(torch_module: Any) -> None:
        """Require the MemTracker fake-collective integration from Torch 2.7."""
        match = re.match(r"^(\d+)\.(\d+)", torch_module.__version__)
        version = tuple(int(part) for part in match.groups()) if match else (0, 0)
        if version < (2, 7):
            raise RuntimeError(
                "HyperDryRun requires PyTorch >= 2.7 because PyTorch 2.6 "
                "MemTracker does not track fake collectives and DTensor dispatch; "
                f"found {torch_module.__version__}"
            )

    @staticmethod
    def _resolve_world_size(args: Any) -> int:
        """Resolve the logical world size from dry-run and parallel config."""
        dry_run_cfg = args.train.dry_run
        accelerator = args.train.accelerator
        if dry_run_cfg.world_size is not None:
            return int(dry_run_cfg.world_size)
        dp_shard = accelerator.dp_shard
        if dp_shard in (None, -1):
            dp_shard = accelerator.dp
        if dp_shard in (None, -1):
            raise ValueError(
                "train.dry_run.world_size is required when dp_shard is automatic"
            )
        return int(
            accelerator.dp_replicate
            * dp_shard
            * accelerator.cp
            * accelerator.tp
            * accelerator.pp
            * accelerator.ep
        )

    def _prepare_args(self, torch_module: Any) -> Any:
        """Validate user input and return an isolated dry-run config copy."""
        from hyper_parallel.trainer.parallel_dims import ParallelDims  # pylint: disable=C0415

        self._check_torch_version(torch_module)
        if self.args.train.backend != "torch":
            raise NotImplementedError(
                "HyperDryRun currently supports train.backend='torch' only, "
                f"got {self.args.train.backend!r}"
            )
        if self.args.train.accelerator.pp > 1:
            raise NotImplementedError(
                "HyperDryRun does not support pipeline parallelism in this version; "
                f"got pp={self.args.train.accelerator.pp}"
            )
        unsupported_degrees = {
            name: int(getattr(self.args.train.accelerator, name))
            for name in ("dp_replicate", "tp", "cp", "ep", "etp")
            if int(getattr(self.args.train.accelerator, name)) > 1
        }
        if unsupported_degrees:
            raise NotImplementedError(
                "HyperDryRun Phase 1 supports FSDP (dp_shard) only; "
                f"unsupported parallel degrees are {unsupported_degrees}"
            )
        if self.args.train.accelerator.cpu_offload:
            raise NotImplementedError(
                "HyperDryRun Phase 1 does not support FSDP CPU offload"
            )
        activation_checkpoint = (
            self.args.train.gradient_checkpointing.activation_checkpoint
        )
        if activation_checkpoint not in (
                "off", "none", "full", None, False, ""
        ):
            raise NotImplementedError(
                "HyperDryRun supports full activation checkpointing only; "
                f"got {activation_checkpoint!r}"
            )

        prepared = copy.deepcopy(self.args)
        world_size = self._resolve_world_size(prepared)
        rank = int(prepared.train.dry_run.rank)
        if rank >= world_size:
            raise ValueError(
                f"dry_run.rank ({rank}) must be smaller than world_size ({world_size})"
            )
        ParallelDims.from_config(prepared.train.accelerator, world_size=world_size)
        prepared.train.dry_run.world_size = world_size
        prepared.train.init_device = "meta"
        prepared.train.local_rank = 0
        return prepared

    @staticmethod
    def _build_fake_batch(torch_module: Any, base: Any) -> tuple[Dict[str, Any], int]:
        """Build the fixed-shape LM batch used by the simulated step."""
        micro_batch_size = int(base.args.train.micro_batch_size)
        sequence_length = int(base.args.data.max_seq_len)
        if micro_batch_size < 1:
            raise ValueError(
                f"train.micro_batch_size must be >= 1, got {micro_batch_size}"
            )
        if sequence_length < 2:
            raise ValueError(
                "data.max_seq_len must be >= 2 for causal-LM dry-run, "
                f"got {sequence_length}"
            )
        shape = (micro_batch_size, sequence_length)
        batch = {
            "input_ids": torch_module.empty(shape, dtype=torch_module.int64, device=base.device),
            "labels": torch_module.empty(shape, dtype=torch_module.int64, device=base.device),
        }
        prepare_batch_fn = getattr(base.spec, "prepare_batch_fn", None)
        if prepare_batch_fn is not None:
            batch = prepare_batch_fn(batch, base.model)
        batch = base._shard_micro_batches_for_cp([batch])[0]
        local_sequence = int(batch["input_ids"].shape[1])
        token_count = micro_batch_size * max(local_sequence - 1, 1)
        return batch, token_count

    @staticmethod
    def _metadata(
            torch_module: Any,
            base: Any,
            original_init_device: str,
            target_device_type: str,
    ) -> Dict[str, Any]:
        """Build reproducibility metadata for a successful report."""
        dims = base.parallel_dims
        parallel = {
            name: int(getattr(dims, name))
            for name in ("dp_replicate", "dp_shard", "tp", "cp", "ep", "pp", "etp")
        }
        return {
            "model": base.args.model.name,
            "torch_version": torch_module.__version__,
            "rank": int(base.args.train.dry_run.rank),
            "world_size": int(base.args.train.dry_run.world_size),
            "device_type": target_device_type,
            "simulation_device_type": base.device.type,
            "original_init_device": original_init_device,
            "weights_loaded": False,
            "gradient_accumulation_simulated": False,
            "parallel": parallel,
            "reshard_after_forward": bool(base.args.train.accelerator.reshard_after_forward),
            "comm_fusion": bool(base.args.train.accelerator.comm_fusion),
        }

    def _error_report(self, error: Exception) -> Dict[str, Any]:
        """Build a minimal, explicitly unsuccessful report."""
        dry_run_cfg = self.args.train.dry_run
        return {
            "schema_version": _SCHEMA_VERSION,
            "status": "error",
            "stage": self._stage,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "metadata": {
                "model": self.args.model.name,
                "rank": dry_run_cfg.rank,
                "world_size": dry_run_cfg.world_size,
            },
            "limitations": _report_limitations(),
        }

    @staticmethod
    def _initialize_flat_buffers(base: Any) -> None:
        """Materialize enabled FSDP flat shards before tracker registration."""
        for hsdp_state in base._iter_hsdp_states():
            param_group = getattr(hsdp_state, "param_group", None)
            if param_group is not None and param_group.enable_zero_copy:
                param_group._init_flat_param_buffer()

    def _execute_fake_step(
            self,
            torch_module: Any,
            fake_mode_type: Any,
            mem_tracker_type: Any,
            base: Any,
            prepared: Any,
            original_init_device: str,
            target_device_type: str,
    ) -> Dict[str, Any]:
        """Materialize the fake model, execute one step, and build its report."""
        fake_mode = fake_mode_type(allow_non_fake_inputs=True)
        with fake_mode:
            self._stage = "materialize"
            base._post_parallelize()
            base._build_optimizer()
            base._build_training_context()
            batch, token_count = self._build_fake_batch(torch_module, base)

            # Register only after FSDP creates its long-lived flat shard, so
            # MemTracker categorizes the storage as a Parameter.
            self._initialize_flat_buffers(base)

            self._stage = "memory_tracking"
            tracker = mem_tracker_type()
            tracker.track_external(base.model, base.optimizer, batch)
            operator_trace = _create_operator_trace_mode(tracker, base.device.type)
            with tracker, operator_trace:
                if hasattr(base.model, "set_requires_gradient_sync"):
                    base.model.set_requires_gradient_sync(True)
                if hasattr(base.model, "set_is_last_backward"):
                    base.model.set_is_last_backward(True)
                global_tokens = token_count * base.parallel_dims.dp_size * base.parallel_dims.cp
                base.forward_backward_step(batch, token_count, global_tokens)
                base._run_post_fsdp_grad_reduce()
                base.optimizer.step()
                base.optimizer.zero_grad(set_to_none=True)
            operator_trace.finalize()

            self._stage = "report_generation"
            return build_memory_report(
                tracker=tracker,
                metadata=self._metadata(
                    torch_module,
                    base,
                    original_init_device,
                    target_device_type,
                ),
                device_type=base.device.type,
                module_depth=prepared.train.dry_run.module_depth,
                top_modules=prepared.train.dry_run.top_modules,
                device_memory_gib=prepared.train.dry_run.device_memory_gib,
                memory_blocks=operator_trace.memory_blocks,
            )

    def run(self) -> str:
        """Execute one fake training step and return the report path.

        Returns:
            Absolute JSON report path.

        Raises:
            RuntimeError: If validation, model construction, fake execution,
                tracking, or report generation fails.
        """
        self._stage = "validation"
        output_path = os.path.join(
            self.args.train.dry_run.output_dir,
            f"rank_{self.args.train.dry_run.rank}.json",
        )
        original_init_device = self.args.train.init_device
        dist = None
        had_existing_process_group = False
        try:
            if self.args.train.backend != "torch":
                raise NotImplementedError(
                    "HyperDryRun currently supports train.backend='torch' only, "
                    f"got {self.args.train.backend!r}"
                )

            # pylint: disable=C0415
            import torch
            import torch.distributed as dist
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.distributed._tools.mem_tracker import MemTracker
            from hyper_parallel.trainer.base import BaseTrainer

            had_existing_process_group = dist.is_initialized()
            if had_existing_process_group:
                raise RuntimeError(
                    "HyperDryRun must start before a real or fake distributed "
                    "process group is initialized"
                )
            prepared = self._prepare_args(torch)
            output_path = os.path.join(
                prepared.train.dry_run.output_dir,
                f"rank_{prepared.train.dry_run.rank}.json",
            )
            base = BaseTrainer(prepared)
            self._stage = "distributed_setup"
            base._setup()
            target_device_type = (
                prepared.train.dry_run.device_type or base.device.type
            )
            device_handle = getattr(torch, target_device_type, None)
            if device_handle is None or not device_handle.is_available():
                # CPU FakeTensors preserve shape/dtype/lifecycle and avoid
                # constructing a real CUDA/NPU autograd device guard in wheels
                # compiled without that backend. The report records both the
                # logical target and this simulation device.
                base.device = torch.device("cpu")

            self._stage = "model_build"
            # Build real meta tensors before entering FakeTensorMode. The
            # repository's init_empty_weights preserves arbitrary Parameter
            # subclasses via ``type(param)(...)``; a FakeTensor Parameter has
            # a private constructor and cannot be rebuilt that way. Real meta
            # tensors allocate no storage and are safely converted when
            # parallelization materializes them under the fake mode below.
            base._build_model()
            base._freeze_model()
            # Apply storage dtype policy while parameters are ordinary meta
            # tensors. Wrapper-subclass FakeTensors cannot change their outer
            # dtype through ``Parameter.data`` after FSDP materialization, and
            # casting here preserves the same configured dtype and byte count.
            base._maybe_downcast_frozen_params()
            base._maybe_cast_trainable_params()
            self._stage = "parallelize"
            # Build HyperParallel DTensor/FSDP metadata while parameters are
            # ordinary meta tensors. DeviceMesh owns real CPU rank metadata;
            # deepcopying that metadata inside FakeTensorMode would incorrectly
            # mix fake meta storage with its real CPU storage.
            if base.spec.parallelize_fn is None:
                raise ValueError(
                    f"Model spec {base.spec.name!r} must define parallelize_fn "
                    "for HyperDryRun"
                )
            base.model = base.spec.parallelize_fn(base.model, base.mesh, base.args)
            report = self._execute_fake_step(
                torch,
                FakeTensorMode,
                MemTracker,
                base,
                prepared,
                original_init_device,
                target_device_type,
            )
            json_path = write_memory_report(report, output_path)
            csv_path = os.path.splitext(json_path)[0] + "_memory.csv"
            write_memory_csv(report, csv_path)
            logger.info("HyperDryRun tabular report: %s", csv_path)
            return json_path
        except Exception as error:
            try:
                write_memory_report(self._error_report(error), output_path)
            except Exception as report_error:
                logger.warning(
                    "Failed to write HyperDryRun error report %s: %s",
                    output_path,
                    report_error,
                )
            raise RuntimeError(
                f"HyperDryRun failed during {self._stage}: {error}"
            ) from error
        finally:
            if (
                    dist is not None
                    and not had_existing_process_group
                    and dist.is_initialized()
            ):
                dist.destroy_process_group()

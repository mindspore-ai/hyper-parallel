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
"""Offline merger for framework and standalone MegaKernel Chrome traces."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


DEFAULT_KERNEL_PATTERN = r"(?i)mega[_]?kernel"
INTERNAL_EVENT_CATEGORY = "MegaKernelInternal"
MERGE_SCHEMA_VERSION = 1
SUPPORTED_CYCLE_TRACE_SCHEMA_VERSION = 1

TraceObject = dict[str, Any]
TraceInput = str | Path | TraceObject | list[dict[str, Any]]


@dataclass(frozen=True)
class _InvocationTrace:
    """Internal events and display metadata for one MegaKernel invocation."""

    invocation_id: Any
    events: list[dict[str, Any]]
    kernel_name: str
    rank: Any
    device_id: Any
    direction: str | None

    @property
    def anchor_us(self) -> float:
        """Return the first internal event timestamp."""
        return min(_number(event.get("ts"), "ts") for event in self.events)

    @property
    def span_us(self) -> float:
        """Return the relative end-to-end internal event span."""
        anchor = self.anchor_us
        return max(
            _number(event.get("ts"), "ts") + _number(event.get("dur"), "dur") - anchor
            for event in self.events
        )


def _read_trace(source: TraceInput, trace_name: str) -> TraceObject | list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Unable to read {trace_name} from {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {trace_name} {path}: {exc}") from exc
    if isinstance(source, (dict, list)):
        return source
    raise TypeError(
        f"{trace_name} must be a path, Chrome Trace object, or event list, "
        f"got {type(source).__name__}"
    )


def _normalize_framework_trace(trace: Any) -> TraceObject:
    if isinstance(trace, list):
        if not all(isinstance(event, dict) for event in trace):
            raise ValueError("framework_trace event list must contain only objects")
        return {"traceEvents": copy.deepcopy(trace)}
    if isinstance(trace, dict):
        events = trace.get("traceEvents")
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise ValueError("framework_trace must contain a traceEvents object list")
        return copy.deepcopy(trace)
    raise ValueError("framework_trace must be an event list or an object containing traceEvents")


def _events_from_trace(trace: Any, trace_name: str) -> list[dict[str, Any]]:
    if isinstance(trace, list):
        events = trace
    elif isinstance(trace, dict):
        events = trace.get("traceEvents")
    else:
        events = None
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ValueError(f"{trace_name} must contain a traceEvents object list")
    return events


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Trace event {field_name} must be numeric, got {value!r}")
    try:
        resolved_value = float(value)
    except ValueError as exc:
        raise ValueError(f"Trace event {field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(resolved_value):
        raise ValueError(f"Trace event {field_name} must be finite, got {value!r}")
    return resolved_value


def _is_device_kernel_event(event: dict[str, Any]) -> bool:
    """Return whether an outer event carries evidence that it is a Device kernel."""
    category = str(event.get("cat", "")).lower()
    args = event.get("args", {})
    task_type = ""
    if isinstance(args, dict):
        for key, value in args.items():
            normalized_key = str(key).strip().lower().replace("_", " ")
            if normalized_key == "task type":
                task_type = str(value).lower()
                break
    category_is_device = any(token in category for token in ("kernel", "npu", "ascend", "aicore"))
    args_are_device_task = bool(task_type) and any(
        token in task_type for token in ("kernel", "ai_core", "aicore", "aic", "aiv")
    )
    return category_is_device or args_are_device_task


def _kernel_score(event: dict[str, Any]) -> int:
    name = str(event.get("name", "")).lower()
    category = str(event.get("cat", "")).lower()
    args_text = json.dumps(event.get("args", {}), ensure_ascii=False).lower()
    score = 0
    if _is_device_kernel_event(event):
        score += 6
    if "pyboostlaunchaclnn" in name:
        score += 2
    elif "pynativelaunchtask" in name:
        score += 1
    if "getworkspacesize" in name or "tiling" in name:
        score -= 3
    if "_op_" in name or "kernel" in name:
        score += 5
    if any(token in category for token in ("kernel", "npu", "ascend", "aicore")):
        score += 3
    if any(token in args_text for token in ("kernel", "task type", "ai_core", "aicore")):
        score += 2
    if any(token in category for token in ("cpu", "python")):
        score -= 4
    return score


def _complete_events(
        events: list[dict[str, Any]],
        *,
        outer_pid: str | None,
) -> list[dict[str, Any]]:
    complete_events = []
    for event in events:
        if event.get("ph") not in (None, "X"):
            continue
        if outer_pid is not None and str(event.get("pid")) != outer_pid:
            continue
        try:
            timestamp = _number(event.get("ts"), "ts")
            duration = _number(event.get("dur"), "dur")
        except ValueError:
            continue
        if timestamp >= 0 and duration > 0:
            complete_events.append(event)
    return complete_events


def _kernel_candidates(
        events: list[dict[str, Any]],
        kernel_pattern: str,
        outer_pid: str | None,
) -> list[dict[str, Any]]:
    try:
        name_pattern = re.compile(kernel_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid kernel regex {kernel_pattern!r}: {exc}") from exc

    candidates = [
        event
        for event in _complete_events(events, outer_pid=outer_pid)
        if name_pattern.search(str(event.get("name", "")))
    ]
    if not candidates:
        raise ValueError(
            f"No complete MegaKernel event matched pattern {kernel_pattern!r}"
            + (f" and pid {outer_pid!r}" if outer_pid is not None else "")
        )

    best_score = max(_kernel_score(event) for event in candidates)
    return sorted(
        (event for event in candidates if _kernel_score(event) == best_score),
        key=lambda event: _number(event.get("ts"), "ts"),
    )


def _fallback_kernel_candidates(
        events: list[dict[str, Any]],
        invocations: list[_InvocationTrace],
        outer_pid: str | None,
        kernel_index: int,
) -> list[dict[str, Any]]:
    device_events = [
        event
        for event in _complete_events(events, outer_pid=outer_pid)
        if _is_device_kernel_event(event)
    ]
    if kernel_index + len(invocations) > len(device_events):
        raise ValueError(
            "Not enough complete device-kernel events are available for approximate MegaKernel alignment"
            + (f" with pid {outer_pid!r}" if outer_pid is not None else "")
        )
    device_events.sort(key=lambda event: _number(event.get("ts"), "ts"))
    device_events = device_events[kernel_index:]
    invocation_spans = [invocation.span_us for invocation in invocations]

    def window_distance(start_index: int) -> tuple[float, float]:
        """Rank a contiguous Device-kernel window by duration similarity."""
        selected = device_events[start_index:start_index + len(invocations)]
        duration_distance = 0.0
        score = 0
        for event, internal_span in zip(selected, invocation_spans):
            duration = _number(event.get("dur"), "dur")
            shorter_penalty = 1.0 if duration < internal_span else 0.0
            duration_distance += shorter_penalty + abs(duration - internal_span) / max(internal_span, 1.0)
            score += _kernel_score(event)
        return duration_distance, -float(score)

    last_start = len(device_events) - len(invocations)
    best_start = min(range(last_start + 1), key=window_distance)
    return device_events[best_start:best_start + len(invocations)]


def _camel_case_pattern(kernel_name: str) -> str:
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", kernel_name)
    return r"[_]?".join(re.escape(word) for word in words) if words else re.escape(kernel_name)


def _default_kernel_pattern(invocations: list[_InvocationTrace]) -> str:
    patterns = []
    for invocation in invocations:
        pattern = _camel_case_pattern(invocation.kernel_name)
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    if not patterns:
        return DEFAULT_KERNEL_PATTERN
    return rf"(?i)(?:{'|'.join(patterns)})"


def _invocation_metadata(metadata: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    raw_invocations = metadata.get("invocations", [])
    if raw_invocations is None:
        return {}
    if not isinstance(raw_invocations, list):
        raise ValueError("megaKernelCycleTrace.invocations must be a list")
    resolved = {}
    for item in raw_invocations:
        if not isinstance(item, dict) or "invocationId" not in item:
            raise ValueError("Each megaKernelCycleTrace invocation must contain invocationId")
        invocation_id = item["invocationId"]
        if invocation_id in resolved:
            raise ValueError(f"Duplicate MegaKernel invocationId {invocation_id!r}")
        resolved[invocation_id] = item
    return resolved


def _load_invocations(mega_kernel_trace: Any) -> list[_InvocationTrace]:
    if not isinstance(mega_kernel_trace, dict):
        raise ValueError("mega_kernel_trace must be a Chrome Trace object")
    metadata = mega_kernel_trace.get("megaKernelCycleTrace")
    if not isinstance(metadata, dict):
        raise ValueError("mega_kernel_trace is missing megaKernelCycleTrace metadata")
    schema_version = metadata.get("schemaVersion")
    if schema_version != SUPPORTED_CYCLE_TRACE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported MegaKernel cycle trace schemaVersion: "
            f"expected {SUPPORTED_CYCLE_TRACE_SCHEMA_VERSION}, got {schema_version!r}"
        )

    internal_events = [
        event
        for event in _events_from_trace(mega_kernel_trace, "mega_kernel_trace")
        if event.get("ph") == "X" and event.get("cat") == INTERNAL_EVENT_CATEGORY
    ]
    if not internal_events:
        raise ValueError("mega_kernel_trace contains no complete MegaKernel internal events")

    event_args = []
    for event in internal_events:
        args = event.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("MegaKernel internal event args must be an object")
        event_args.append(args)

    has_invocation_id = ["invocation_id" in args for args in event_args]
    if any(has_invocation_id) and not all(has_invocation_id):
        raise ValueError("MegaKernel internal events must either all contain invocation_id or all omit it")

    grouped_events: dict[Any, list[dict[str, Any]]] = {}
    if all(has_invocation_id):
        for event, args in zip(internal_events, event_args):
            invocation_id = args["invocation_id"]
            grouped_events.setdefault(invocation_id, []).append(event)
    else:
        grouped_events[None] = internal_events

    metadata_by_invocation = _invocation_metadata(metadata)
    invocations = []
    for invocation_id, events in grouped_events.items():
        invocation_item = metadata_by_invocation.get(invocation_id, {})
        kernel_name = str(
            invocation_item.get("kernelName")
            or metadata.get("kernelName")
            or "MegaKernel"
        )
        raw_direction = invocation_item.get("direction")
        invocations.append(
            _InvocationTrace(
                invocation_id=invocation_id,
                events=events,
                kernel_name=kernel_name,
                rank=invocation_item.get("rank", metadata.get("rank")),
                device_id=invocation_item.get("deviceId", metadata.get("deviceId")),
                direction=None if raw_direction is None else str(raw_direction),
            )
        )
    return invocations


def _core_key(event: dict[str, Any]) -> tuple[str, int]:
    args = event.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("MegaKernel internal event args must be an object")
    core_type = str(args.get("core_type", "UnknownCore"))
    block_id = args.get("block_id", -1)
    if isinstance(block_id, bool):
        raise ValueError(f"MegaKernel internal block_id must be an integer, got {block_id!r}")
    try:
        resolved_block_id = int(block_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MegaKernel internal block_id must be an integer, got {block_id!r}") from exc
    return core_type, resolved_block_id


def _next_thread_id(events: list[dict[str, Any]], pid: Any) -> int:
    numeric_thread_ids = [
        event["tid"]
        for event in events
        if event.get("pid") == pid
        and isinstance(event.get("tid"), int)
        and not isinstance(event.get("tid"), bool)
    ]
    return max(numeric_thread_ids, default=0) + 1


def _initial_thread_sort_indexes(
        events: list[dict[str, Any]],
        selected_candidates: list[dict[str, Any]],
        invocations: list[_InvocationTrace],
) -> dict[Any, int]:
    thread_counts = {}
    for outer_event, invocation in zip(selected_candidates, invocations):
        pid = outer_event.get("pid", 0)
        core_keys = {_core_key(event) for event in invocation.events}
        thread_counts[pid] = thread_counts.get(pid, 0) + len(core_keys)

    initial_indexes = {}
    for pid, thread_count in thread_counts.items():
        existing_indexes = []
        for event in events:
            if event.get("pid") != pid or event.get("name") != "thread_sort_index":
                continue
            args = event.get("args", {})
            if not isinstance(args, dict):
                continue
            sort_index = args.get("sort_index")
            if isinstance(sort_index, int) and not isinstance(sort_index, bool):
                existing_indexes.append(sort_index)
        initial_indexes[pid] = min(existing_indexes, default=0) - thread_count
    return initial_indexes


def _align_invocation(
        events: list[dict[str, Any]],
        invocation: _InvocationTrace,
        outer_event: dict[str, Any],
        first_sort_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    outer_timestamp = _number(outer_event.get("ts"), "ts")
    outer_duration = _number(outer_event.get("dur"), "dur")
    outer_pid = outer_event.get("pid", 0)
    core_keys = sorted({_core_key(event) for event in invocation.events})
    first_thread_id = _next_thread_id(events, outer_pid)
    thread_ids = {
        core_key: first_thread_id + index
        for index, core_key in enumerate(core_keys)
    }

    metadata_events = []
    for index, (core_key, thread_id) in enumerate(thread_ids.items()):
        metadata_events.extend([
            {
                "name": "thread_name",
                "ph": "M",
                "pid": outer_pid,
                "tid": thread_id,
                "args": {"name": f"{invocation.kernel_name}/{core_key[0]}/{core_key[1]}"},
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": outer_pid,
                "tid": thread_id,
                "args": {"sort_index": first_sort_index + index},
            },
        ])

    aligned_events = []
    invocation_anchor = invocation.anchor_us
    for internal_event in invocation.events:
        internal_args = copy.deepcopy(internal_event.get("args", {}))
        aligned_event = copy.deepcopy(internal_event)
        aligned_event["pid"] = outer_pid
        aligned_event["tid"] = thread_ids[_core_key(internal_event)]
        aligned_event["ts"] = (
            outer_timestamp
            + _number(internal_event.get("ts"), "ts")
            - invocation_anchor
        )
        internal_args.update({
            "parent_kernel": outer_event.get("name"),
            "parent_kernel_ts": outer_timestamp,
            "parent_kernel_dur": outer_duration,
            "cycle_trace_rank": invocation.rank,
            "cycle_trace_device_id": invocation.device_id,
        })
        aligned_event["args"] = internal_args
        aligned_events.append(aligned_event)

    internal_end = max(
        _number(event.get("ts"), "ts") + _number(event.get("dur"), "dur")
        for event in aligned_events
    )
    outer_end = outer_timestamp + outer_duration
    alignment = {
        "invocationId": invocation.invocation_id,
        "direction": invocation.direction,
        "kernelName": invocation.kernel_name,
        "rank": invocation.rank,
        "deviceId": invocation.device_id,
        "outerKernelName": outer_event.get("name"),
        "outerKernelPid": outer_pid,
        "outerKernelTid": outer_event.get("tid"),
        "outerKernelTs": outer_timestamp,
        "outerKernelDur": outer_duration,
        "outerKernelScore": _kernel_score(outer_event),
        "outerKernelIsDevice": _is_device_kernel_event(outer_event),
        "internalEventCount": len(aligned_events),
        "internalSpanUs": invocation.span_us,
        "exceedsOuterKernelUs": max(internal_end - outer_end, 0.0),
    }
    return metadata_events, aligned_events, alignment, first_sort_index + len(core_keys)


def _merge_trace_objects(
        framework_trace: Any,
        mega_kernel_trace: Any,
        *,
        kernel_pattern: str | None,
        kernel_index: int,
        outer_pid: str | int | None,
        allow_host_wrapper: bool,
) -> TraceObject:
    if not isinstance(kernel_index, int) or isinstance(kernel_index, bool) or kernel_index < 0:
        raise ValueError(f"kernel_index must be a non-negative integer, got {kernel_index!r}")
    if kernel_pattern is not None and not isinstance(kernel_pattern, str):
        raise TypeError(f"kernel_pattern must be a string or None, got {type(kernel_pattern).__name__}")
    if kernel_pattern == "":
        raise ValueError("kernel_pattern must not be empty")
    if not isinstance(allow_host_wrapper, bool):
        raise TypeError(f"allow_host_wrapper must be bool, got {type(allow_host_wrapper).__name__}")

    merged_trace = _normalize_framework_trace(framework_trace)
    outer_events = merged_trace["traceEvents"]
    invocations = _load_invocations(mega_kernel_trace)
    resolved_outer_pid = None if outer_pid is None else str(outer_pid)
    resolved_kernel_pattern = kernel_pattern or _default_kernel_pattern(invocations)
    used_fallback_selection = False
    warnings = []
    try:
        candidates = _kernel_candidates(outer_events, resolved_kernel_pattern, resolved_outer_pid)
        if kernel_index + len(invocations) > len(candidates):
            raise ValueError(
                "Not enough outer MegaKernel events: requested indexes "
                f"{kernel_index}..{kernel_index + len(invocations) - 1}, found {len(candidates)}"
            )
        selected_candidates = candidates[kernel_index:kernel_index + len(invocations)]
    except ValueError as exc:
        if kernel_pattern is not None or "No complete MegaKernel event matched" not in str(exc):
            raise
        selected_candidates = _fallback_kernel_candidates(
            outer_events,
            invocations,
            resolved_outer_pid,
            kernel_index,
        )
        used_fallback_selection = True
        warnings.append(
            f"No outer kernel name matched inferred pattern {resolved_kernel_pattern!r}; "
            "selected a contiguous device-kernel window by duration similarity, so alignment is approximate"
        )

    host_candidates = [event for event in selected_candidates if not _is_device_kernel_event(event)]
    if host_candidates and not allow_host_wrapper:
        host_names = sorted({str(event.get("name", "")) for event in host_candidates})
        raise ValueError(
            "Refusing to align MegaKernel internal events to host-only outer event(s) "
            f"{host_names}. Recollect the framework profile with Device activities, or pass "
            "allow_host_wrapper=True only for an explicitly approximate diagnostic timeline."
        )
    if host_candidates:
        warnings.append(
            "Selected host-only outer event(s); alignment is approximate because Device kernel events were unavailable"
        )

    next_sort_indexes = _initial_thread_sort_indexes(outer_events, selected_candidates, invocations)
    report = {
        "schemaVersion": MERGE_SCHEMA_VERSION,
        "kernelPattern": resolved_kernel_pattern,
        "kernelIndex": kernel_index,
        "usedFallbackKernelSelection": used_fallback_selection,
        "allowHostWrapperAlignment": allow_host_wrapper,
        "mergedInvocationCount": len(invocations),
        "mergedEventCount": 0,
        "alignments": [],
        "warnings": warnings,
    }

    for invocation, outer_event in zip(invocations, selected_candidates):
        outer_pid_value = outer_event.get("pid", 0)
        metadata_events, aligned_events, alignment, next_sort_index = _align_invocation(
            outer_events,
            invocation,
            outer_event,
            next_sort_indexes[outer_pid_value],
        )
        next_sort_indexes[outer_pid_value] = next_sort_index
        outer_events.extend(metadata_events)
        outer_events.extend(aligned_events)
        report["mergedEventCount"] += len(aligned_events)
        report["alignments"].append(alignment)
        if alignment["exceedsOuterKernelUs"] > 0:
            report["warnings"].append(
                f"Invocation {invocation.invocation_id!r} internal events exceed outer kernel "
                f"by {alignment['exceedsOuterKernelUs']:.3f} us"
            )

    merged_trace["megaKernelTraceMerge"] = report
    return merged_trace


def merge_chrome_traces(
        framework_trace: TraceInput,
        mega_kernel_trace: TraceInput,
        output: str | Path,
        *,
        kernel_pattern: str | None = None,
        kernel_index: int = 0,
        outer_pid: str | int | None = None,
        allow_host_wrapper: bool = False,
) -> TraceObject:
    """Merge standalone MegaKernel events into a framework Chrome Trace.

    Each standalone invocation is aligned to one outer Device kernel in
    timestamp order. The merge is pure Host post-processing and does not
    require Torch, MindSpore, or NPU runtime imports.

    Args:
        framework_trace: Framework Chrome Trace path, object, or event list.
        mega_kernel_trace: Standalone MegaKernel Chrome Trace path or object.
        output: Destination path for the merged Chrome Trace object.
        kernel_pattern: Optional outer-kernel name regular expression. By
            default it is inferred from standalone invocation metadata.
        kernel_index: First matching outer-kernel occurrence to consume.
        outer_pid: Optional outer process ID filter.
        allow_host_wrapper: Allow approximate alignment to matching Host
            wrapper events when Device kernel events are unavailable.

    Returns:
        Merged Chrome Trace object. Alignment diagnostics are stored in its
        megaKernelTraceMerge metadata.

    Raises:
        TypeError: If an option or in-memory trace has an invalid type.
        ValueError: If a trace is malformed or cannot be aligned safely.
    """
    framework_trace_object = _read_trace(framework_trace, "framework_trace")
    mega_kernel_trace_object = _read_trace(mega_kernel_trace, "mega_kernel_trace")
    merged_trace = _merge_trace_objects(
        framework_trace_object,
        mega_kernel_trace_object,
        kernel_pattern=kernel_pattern,
        kernel_index=kernel_index,
        outer_pid=outer_pid,
        allow_host_wrapper=allow_host_wrapper,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_trace, ensure_ascii=False), encoding="utf-8")
    return merged_trace


__all__ = ["merge_chrome_traces"]

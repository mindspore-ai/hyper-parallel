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
"""Stable source, recovery and impact metrics for tool-protocol failures."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


JSON_ERROR_CATEGORIES = (
    "model_invalid_json",
    "parser_error",
    "serialization_error",
    "unknown_json_error",
)

JSON_ERROR_TYPES = (
    "invalid_escape",
    "unterminated_string",
    "invalid_control_character",
    "missing_delimiter",
    "extra_data",
    "other_json_error",
)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _collect_protocol_events(completions):
    """Collect failure events and category counts from model completions."""
    events = []
    categories: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    error_positions: Counter[str] = Counter()
    tool_protocol_errors = 0
    tool_capable_calls = 0
    for fallback_ordinal, completion in enumerate(completions):
        metadata = dict(completion.get("metadata", {}))
        if completion.get("request", {}).get("tools"):
            tool_capable_calls += 1
        reason = metadata.get("failure_reason")
        if not reason:
            continue
        tool_protocol_errors += 1
        category = metadata.get("error_category")
        if category in JSON_ERROR_CATEGORIES:
            categories[category] += 1
        error_type = metadata.get("json_error_type")
        parser_error = metadata.get("parser_error") or {}
        if error_type in JSON_ERROR_TYPES:
            error_types[error_type] += 1
        if error_type and parser_error.get("line") is not None:
            position_key = f"{error_type}:line{parser_error['line']}:col{parser_error.get('column')}"
            error_positions[position_key] += 1
        events.append({
            "event_id": metadata.get("protocol_event_id"),
            "session_id": metadata.get("session_id"),
            "call_index": int(completion.get("ordinal", fallback_ordinal)),
            "policy_version": metadata.get("policy_version"),
            "failure_stage": metadata.get("failure_stage"),
            "failure_origin": metadata.get("failure_origin"),
            "failure_reason": reason,
            "error_category": category,
            "json_error_type": error_type,
            "json_error_position": {
                "line": parser_error.get("line"),
                "column": parser_error.get("column"),
                "char": parser_error.get("position"),
            } if error_type else None,
            "raw_tool_call_sha256": metadata.get("raw_tool_call_sha256"),
            "trainable": metadata.get("trainable"),
            "feedback_delivered": bool(metadata.get("feedback_delivered")),
            "recovered_with_valid_action": bool(metadata.get("recovered_with_valid_action")),
            "recovered_with_valid_tool_call": bool(metadata.get("recovered_with_valid_tool_call")),
            "recovery_call_index": metadata.get("recovery_call_index"),
        })

    return events, categories, error_types, error_positions, tool_protocol_errors, tool_capable_calls


def summarize_tool_protocol(
    completions: Sequence[Mapping[str, Any]],
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize unique model responses without changing captured evidence."""
    final = dict(outcome or {})
    events, categories, error_types, error_positions, tool_protocol_errors, tool_capable_calls = (
        _collect_protocol_events(completions)
    )
    json_error_count = sum(categories.values())
    recovered = sum(event["recovered_with_valid_action"] for event in events)
    recovered_tool = sum(event["recovered_with_valid_tool_call"] for event in events)
    affected = bool(events)
    completed = bool(final.get("harness_completed"))
    task_success = bool(final.get("task_success"))
    final_origin = final.get("failure_origin")
    reward_evaluated = bool(final.get("reward_evaluated"))
    training_rows = int(final.get("training_rows_emitted", 0) or 0)
    contamination = int(
        final_origin in {"infrastructure", "unknown"} and (reward_evaluated or training_rows > 0)
    )
    return {
        "complete": bool(outcome is not None),
        "model_calls": len(completions),
        "tool_capable_model_calls": tool_capable_calls,
        "tool_protocol_error_count": tool_protocol_errors,
        "json_error_count": json_error_count,
        **{name: categories[name] for name in JSON_ERROR_CATEGORIES},
        **{name: error_types[name] for name in JSON_ERROR_TYPES},
        "json_error_positions": dict(error_positions),
        "json_error_rate": _ratio(json_error_count, tool_capable_calls),
        "json_error_per_model_call": _ratio(json_error_count, len(completions)),
        "tool_error_recovery_count": recovered,
        "tool_error_recovery_rate": _ratio(recovered, tool_protocol_errors),
        "valid_tool_call_after_error_count": recovered_tool,
        "affected_trajectories": int(affected),
        "trajectory_completed_after_error": int(affected and completed),
        "post_error_task_success": int(affected and task_success),
        "post_error_task_success_rate": float(task_success) if affected else 0.0,
        "final_failure_origin": final_origin,
        "final_failure_reason": final.get("failure_reason"),
        "reward_evaluated": reward_evaluated,
        "training_rows_emitted": training_rows,
        "infra_reward_contamination_count": contamination,
        "events": events,
    }


def aggregate_tool_protocol_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate episode summaries with explicit, reproducible denominators."""
    totals: Counter[str] = Counter()
    for summary in summaries:
        for key in (
            "model_calls", "tool_capable_model_calls", "tool_protocol_error_count", "json_error_count",
            *JSON_ERROR_CATEGORIES, "tool_error_recovery_count", "valid_tool_call_after_error_count",
            *JSON_ERROR_TYPES,
            "affected_trajectories", "trajectory_completed_after_error", "post_error_task_success",
            "infra_reward_contamination_count", "training_rows_emitted",
        ):
            totals[key] += int(summary.get(key, 0) or 0)
    result = dict(totals)
    positions: Counter[str] = Counter()
    for summary in summaries:
        positions.update(summary.get("json_error_positions", {}))
    result.update(
        complete=bool(summaries) and all(bool(item.get("complete")) for item in summaries),
        trajectories=len(summaries),
        json_error_rate=_ratio(totals["json_error_count"], totals["tool_capable_model_calls"]),
        json_error_per_model_call=_ratio(totals["json_error_count"], totals["model_calls"]),
        tool_error_recovery_rate=_ratio(
            totals["tool_error_recovery_count"], totals["tool_protocol_error_count"]
        ),
        post_error_task_success_rate=_ratio(
            totals["post_error_task_success"], totals["affected_trajectories"]
        ),
        json_error_positions=dict(positions),
    )
    return result

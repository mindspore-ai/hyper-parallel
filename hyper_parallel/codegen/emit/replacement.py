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
"""Translate native replacement observations for source emission."""
from __future__ import annotations

from typing import Any


def replacement_records_from_snapshot(snapshot: Any) -> tuple[dict[str, Any], ...]:
    """Translate native replacement observations to the legacy emitter shape.

    This is a transition-only adapter. It consumes immutable snapshot data and
    never recompiles rules or inspects a live model.
    """
    payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
    if not isinstance(payload, dict):
        raise TypeError("replacement snapshot must be a CodegenSnapshot or dict")
    replacements = payload.get("replacements", {})
    declarations = replacements.get("declarations", ())
    applications = replacements.get("applications", ())
    declarations_by_id = {item["rule_id"]: item for item in declarations}
    if len(declarations_by_id) != len(declarations):
        raise ValueError("replacement snapshot contains duplicate rule_id values")
    records = []
    for application in applications:
        rule_id = application.get("rule_id")
        declaration = declarations_by_id.get(rule_id)
        if declaration is None:
            raise ValueError(
                f"replacement application references unknown rule_id {rule_id!r}"
            )
        source_type = declaration["source_type"].replace(":", ".", 1)
        record = {
            "match": list(declaration["match"]),
            "fqn": application["primary_fqn"],
            "fqns": list(application["module_fqns"]),
            "module_type": source_type,
            "factory": declaration["replacement_target"],
            "exact_type": bool(declaration["exact_type"]),
        }
        if declaration.get("static_arguments"):
            record["target_config"] = dict(declaration["static_arguments"])
        records.append(record)
    return tuple(records)


__all__ = ["replacement_records_from_snapshot"]

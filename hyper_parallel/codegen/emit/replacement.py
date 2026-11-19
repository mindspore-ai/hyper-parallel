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


PREPARED_FAMILY_CONSTRUCTOR = (
    "hyper_parallel.components.modules.module_family:build_prepared_family_replacement"
)


def replacement_factory(family: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Translate one prepared factory record into a source constructor and arguments."""
    selection = family.get("selection")
    if selection is not None:
        return PREPARED_FAMILY_CONSTRUCTOR, {"prepared": selection}
    return family["factory"], {**family["kwargs"], "context": family["context"]}


def replacement_records_from_snapshot(snapshot: Any) -> tuple[dict[str, Any], ...]:
    """Translate native replacement observations into emission records.

    The conversion consumes immutable observation data and never recompiles
    rules or inspects a live model.

    Args:
        snapshot: Resolved native build snapshot containing replacement observations.
    """
    payload = snapshot._emission_view() if hasattr(snapshot, "_emission_view") else snapshot  # pylint: disable=protected-access
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
        source_identity = application.get("source_type") or declaration.get("source_type")
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError(
                f"replacement application {rule_id!r} has no resolved source_type"
            )
        source_type = source_identity.replace(":", ".", 1)
        record = {
            "match": list(declaration["match"]),
            "fqn": application["primary_fqn"],
            "fqns": list(application["module_fqns"]),
            "module_type": source_type,
            "factory": application["replacement_type"].replace(":", ".", 1),
            "replacement_target": declaration["replacement_target"],
            "exact_type": bool(declaration["exact_type"]),
        }
        if declaration.get("static_arguments"):
            record["target_config"] = dict(declaration["static_arguments"])
        if application.get("opaque_legacy"):
            record["opaque_legacy"] = True
        family = application.get("family_request")
        if family is not None:
            factory, arguments = replacement_factory(family)
            record["prepared_family"] = True
            record["factory"] = factory.replace(":", ".", 1)
            record["replacement_target"] = record["factory"]
            record["target_config"] = arguments
        records.append(record)
    return tuple(records)


__all__ = ["replacement_factory", "replacement_records_from_snapshot"]

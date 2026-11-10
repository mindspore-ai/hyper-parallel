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
"""Project native replacement and injection observations into source rules."""

from __future__ import annotations

from typing import Any, Iterable

from hyper_parallel.codegen.inline.ir import SourceObservation
from hyper_parallel.codegen.inline.targets import projected_target_path


def collect_source_observations(meta: Any) -> tuple[SourceObservation, ...]:
    """Return normalized source rules from the already observed native plan."""
    rules: list[SourceObservation] = []
    for record in getattr(meta, "module_overrides", None) or []:
        target = _replacement_target(record)
        if target:
            rules.append(
                SourceObservation(
                    match=_matches(record),
                    module_type=record.get("module_type"),
                    replace_target=target,
                )
            )
    for record in getattr(meta, "injections", None) or []:
        local = projected_target_path(record.get("local_compute_fn"))
        inner = projected_target_path(record.get("inner_wrapper"))
        if local is not None or inner is not None:
            rules.append(
                SourceObservation(
                    match=_matches(record),
                    inner_wrapper_target=inner,
                    local_compute_target=local,
                )
            )
    return tuple(rules)


def _replacement_target(record: dict[str, Any]) -> Any:
    """Prefer model-owned declarations and otherwise use the observed factory."""
    declared = record.get("replacement_target")
    parts = str(declared or "").split(".")
    if len(parts) >= 4 and parts[:2] == ["hyper_parallel", "models"] and parts[3] == "adapter":
        return declared
    return record.get("factory")


def _matches(record: dict[str, Any]) -> tuple[str, ...]:
    """Normalize one observation's selected FQNs."""
    value = record.get("fqns") or record.get("match") or record.get("fqn")
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


__all__ = ["collect_source_observations"]

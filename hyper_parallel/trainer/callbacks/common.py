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
"""Shared helpers for trainer callback implementations."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Optional


def _train_cfg(args: Any) -> Any:
    """Return args.train when present."""
    return getattr(args, "train", None)


def _sub_cfg(args: Any, name: str) -> Any:
    """Return args.train.<name> or a top-level fallback when present."""
    train_cfg = _train_cfg(args)
    if train_cfg is not None and hasattr(train_cfg, name):
        return getattr(train_cfg, name)
    return getattr(args, name, None)


def _should_trigger(step: int, every: int) -> bool:
    """Return whether step is a positive multiple of every."""
    return every > 0 and step > 0 and step % every == 0


def _as_float(value: Any) -> Optional[float]:
    """Best-effort scalar conversion for tensors and numbers."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """Best-effort integer conversion for tensors and numbers."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable_config(args: Any) -> dict[str, Any]:
    """Convert nested dataclass or mapping config into JSON-safe values."""
    if dataclasses.is_dataclass(args):
        return dataclasses.asdict(args)
    if isinstance(args, Mapping):
        return dict(args)
    return {}


def _format_console_value(value: Any) -> str:
    """Format one metric value for rank-0 console logging."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.4e}"
    return str(value)


def _state_snapshot(state: Any) -> dict[str, Any]:
    """Serialize trainer state through the state object's contract."""
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(state, Mapping):
        return dict(state)
    return dict(vars(state))


def _restore_state(state: Any, values: Mapping[str, Any]) -> None:
    """Mutate an existing trainer state object from serialized values."""
    from_dict = getattr(type(state), "from_dict", None)
    if callable(from_dict):
        restored = from_dict(values)
        for key, value in vars(restored).items():
            setattr(state, key, value)
        return
    update = getattr(state, "update", None)
    if callable(update):
        update(**values)
        return
    for key, value in values.items():
        setattr(state, key, value)


__all__ = [
    "_train_cfg",
    "_sub_cfg",
    "_should_trigger",
    "_as_float",
    "_as_int",
    "_jsonable_config",
    "_format_console_value",
    "_state_snapshot",
    "_restore_state",
]

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
"""Delayed-invocation target wrapper and config serialization helper.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5):
``Target`` is the YAML transport wrapper; ``_serialize_config_value`` is
shared by every nested config's ``to_dict`` and therefore lives beside it
(the plan lists it under ``trainer.py``; placing it here avoids a circular
import, the public surface is unchanged).
"""

import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Generic, TypeVar

_T = TypeVar("_T")


def _serialize_config_value(value: Any) -> Any:
    """Convert one target argument to a plain serializable value."""
    if inspect.isroutine(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return {
            config_field.name: _serialize_config_value(
                getattr(value, config_field.name)
            )
            for config_field in fields(value)
        }
    if isinstance(value, dict):
        return {
            key: _serialize_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(item) for item in value]
    return value


class Target(Generic[_T]):
    """Configuration for one callable whose invocation is delayed until runtime."""

    def __init__(
            self,
            _target_: Callable[..., _T],  # pylint: disable=invalid-name
            *,
            target_path: str,
            **kwargs: Any,
    ) -> None:
        """Store the resolved callable, its source path, and configured arguments."""
        if not callable(_target_):
            raise TypeError("_target_ must be callable")
        if not isinstance(target_path, str) or not target_path.strip():
            raise ValueError("target_path must be a non-empty string")

        self._target_ = _target_
        self._target_path = target_path
        self._kwargs = dict(kwargs)

    def __getattr__(self, name: str) -> Any:
        kwargs = object.__getattribute__(self, "_kwargs")
        try:
            return kwargs[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def build(self, **runtime_kwargs: Any) -> _T:
        """Invoke the target with configured and applicable runtime arguments."""
        signature = inspect.signature(self._target_)
        if not any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
        ):
            runtime_kwargs = {
                name: value
                for name, value in runtime_kwargs.items()
                if name in signature.parameters
            }

        kwargs = {**self._kwargs, **runtime_kwargs}
        return self._target_(**kwargs)

    def replace(self, **changes: Any) -> "Target[_T]":
        """Return a new target with selected configured arguments replaced."""
        kwargs = dict(self._kwargs)
        kwargs.update(changes)
        return type(self)(
            self._target_,
            target_path=self._target_path,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this target back to its YAML-compatible form."""
        return {
            "_target_": self._target_path,
            **{
                name: _serialize_config_value(value)
                for name, value in self._kwargs.items()
            },
        }

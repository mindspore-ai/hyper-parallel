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
"""Small typed registry shared by public extension points."""
from collections.abc import Iterator
from typing import Any, Callable, Generic, TypeVar
T = TypeVar("T")


class Registry(Generic[T]):
    """Register named implementations with consistent validation."""

    def __init__(self, kind: str) -> None:
        """Create an empty registry for one implementation kind."""
        self.kind = kind
        self._items: dict[str, T] = {}

    @staticmethod
    def _key(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Registry name must be non-empty")
        return key

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers one implementation."""
        key = self._key(name)

        def decorator(item: T) -> T:
            """Store one implementation under the normalized key."""
            if key in self._items:
                raise ValueError(f"{self.kind.capitalize()} is already registered: {key}")
            self._items[key] = item
            return item
        return decorator

    def get(self, name: str) -> T:
        """Return a named implementation or report available names."""
        key = self._key(name)
        try:
            return self._items[key]
        except KeyError as error:
            raise ValueError(
                f"Unknown {self.kind} '{name}'; available={list(self.names)}"
            ) from error

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a registered builder."""
        builder = self.get(name)
        if not callable(builder):
            raise TypeError(f"Registered {self.kind} '{name}' is not callable")
        return builder(*args, **kwargs)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names in deterministic order."""
        return iter(self.names)

    def __getitem__(self, name: str) -> T:
        """Support direct lookup for callers treating a registry as a mapping."""
        return self.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._items))
__all__ = ["Registry"]

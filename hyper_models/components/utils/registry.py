# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterator, List, MutableMapping, Optional, Type, Union


class Registry(MutableMapping):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    registry = []

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: Human-readable name of the registered category, used in
                error messages.
        """
        self._name = name
        self.registry.append(name)
        self._local_mapping = {}
        self._global_mapping = {}

    def __getitem__(self, key: str) -> Union[Type, Callable]:
        """Look up a registered class or function by key.

        Args:
            key: Registration key.

        Returns:
            The locally overridden value when present, otherwise the globally
            registered value.

        Raises:
            ValueError: If ``key`` is not registered.
        """
        # First check if instance has a local override
        if key not in self.valid_keys():
            raise ValueError(f"Unknown {self._name} name: {key}. No {self._name} registered for this source.")
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key: str, value: Union[Type, Callable]) -> None:
        """Set a local override for ``key`` without affecting other instances."""
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key: str) -> None:
        """Delete the local override for ``key``."""
        del self._local_mapping[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over all valid keys, local overrides taking precedence."""
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self) -> int:
        """Return the number of distinct registered keys."""
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    def register(
        self,
        key: str,
        cls_or_func: Optional[Union[Type, Callable]] = None,
    ) -> Union[Type, Callable]:
        """Register a class or function under ``key``.

        Can be used either as a plain call or as a decorator::

            registry.register("name", MyClass)

            @registry.register("name")
            class MyClass: ...

        Args:
            key: Registration key.
            cls_or_func: The class or function to register. When ``None``, a
                decorator is returned instead.

        Returns:
            The registered class/function, or a decorator when ``cls_or_func``
            is ``None``.

        Raises:
            ValueError: If ``key`` is already registered (decorator form).
        """
        if cls_or_func is not None:
            self._global_mapping[key] = cls_or_func
            return cls_or_func

        def decorator(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
            """Register the decorated class or function under ``key``."""
            if key in self._global_mapping:
                raise ValueError(
                    f"{self._name} for '{key}' is already registered. Cannot register duplicate {self._name}."
                )
            self._global_mapping.update({key: cls_or_func})
            return cls_or_func

        return decorator

    def valid_keys(self) -> List[str]:
        """Return the list of all registered keys."""
        return list(self.keys())

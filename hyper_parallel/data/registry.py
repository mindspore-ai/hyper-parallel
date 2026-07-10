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
"""Dataset registry — single ``data.type`` name → builder dispatch.

The registry is the only piece of dataset-related code aware of the full
catalogue of supported formats. Each format lives in its own module and
registers a builder via ``@DATASET_REGISTRY.register("<name>")``; the trainer
then calls :func:`build_dataset` with the ``data.type`` from the YAML config
and never needs to know about specific formats. New formats plug in by adding
one module-level registration instead of modifying trainer code.
"""
from typing import Any, Callable, Dict, Iterator, List


BuilderFn = Callable[..., Any]


class DatasetRegistry:
    """Map ``data.type`` names to builder callables.

    Each builder takes keyword arguments ``base`` (the :class:`BaseTrainer`
    instance), ``args`` (the parsed config) plus any extras the builder
    needs (``tokenizer``, ``data_transform`` ...). The registry never holds
    state about a specific run — builders are looked up fresh each time.

    Example:
        >>> @DATASET_REGISTRY.register("dummy")
        ... def build_dummy(*, base, args, **_):
        ...     return _DummyDataset(...)
    """

    def __init__(self) -> None:
        self._builders: Dict[str, BuilderFn] = {}

    def register(self, name: str) -> Callable[[BuilderFn], BuilderFn]:
        """Decorator that registers ``fn`` under ``name``.

        Raises:
            ValueError: When ``name`` is already registered. Re-registration
                is treated as a programming error (typo / duplicate import)
                rather than silently overwritten.
        """
        if not name:
            raise ValueError("Dataset registry name must be a non-empty string")

        def decorator(fn: BuilderFn) -> BuilderFn:
            if name in self._builders:
                raise ValueError(
                    f"Dataset type '{name}' is already registered; "
                    f"check for duplicate registrations / circular imports"
                )
            self._builders[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> BuilderFn:
        """Return the builder for ``name`` or raise with a helpful message."""
        try:
            return self._builders[name]
        except KeyError as exc:
            available = sorted(self._builders)
            raise ValueError(
                f"Unknown data.type '{name}'. Available: {available}. "
                f"Add a new format by registering a builder under "
                f"hyper_parallel.data."
            ) from exc

    def names(self) -> List[str]:
        """Return registered names in deterministic (sorted) order."""
        return sorted(self._builders)

    def __contains__(self, name: str) -> bool:
        return name in self._builders

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


DATASET_REGISTRY = DatasetRegistry()


def build_dataset(name: str, **kwargs: Any) -> Any:
    """Build a dataset by ``data.type`` name.

    Args:
        name: Dataset type identifier — must match a registered builder.
        **kwargs: Forwarded to the builder. The trainer passes ``base``
            (the :class:`BaseTrainer` instance) and ``args`` (the parsed
            config) plus, when available, ``tokenizer`` / ``data_transform``.

    Returns:
        A ``torch.utils.data.Dataset``-compatible object suitable for the
        trainer's distributed sampler + ``StatefulDataLoader``.
    """
    return DATASET_REGISTRY.get(name)(**kwargs)

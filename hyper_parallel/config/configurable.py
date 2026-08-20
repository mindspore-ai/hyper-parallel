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
"""Declarative configuration base types for HyperParallel components."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


def _build_from_config(
    owner: type["Configurable"],
    config: "Configurable.Config",
    **kwargs: Any,
) -> "Configurable":
    """Instantiate *owner* from a config snapshot.

    Thin wrapper around ``owner(config=config, **kwargs)`` so static analyzers
    can resolve the constructor call site.

    Args:
        owner: Configurable subclass to construct (from ``Config._owner``).
        config: Config instance passed to ``owner.__init__``.
        **kwargs: Extra keyword arguments forwarded to ``owner.__init__``.

    Returns:
        A new instance of *owner*.
    """
    return owner(config=config, **kwargs)


class Configurable:
    """Base class for components built from nested dataclass configs.

    Subclasses declare a nested ``Config`` and implement ``__init__(self, config)``.
    ``Config.build()`` constructs the outer class; ``__init_subclass__`` wires
    ``Config._owner`` when the nested class is defined on the subclass.

    ``build()`` supports two forms:

    - No extra kwargs: ``owner(config=replace(self))``.
    - With kwargs: runtime-only values (not stored on the config) are forwarded
      to ``owner.__init__``; names must not overlap config fields.

    Module configs additionally require ``@dataclass(kw_only=True, slots=True)``
    via :func:`enforce_module_config_slots` in
    :mod:`hyper_parallel.dmodule.module`.
    """

    Config: type["Configurable.Config"]

    @dataclass(kw_only=True, slots=True)
    class Config:
        """Base dataclass for component configuration trees."""

        _owner: ClassVar[type["Configurable"] | None] = None

        def replace(self, **kwargs) -> "Configurable.Config":
            """Return a shallow copy with selected fields replaced.

            Args:
                **kwargs: Field names and new values (same semantics as
                    :func:`dataclasses.replace`).

            Returns:
                A new config instance; the original is unchanged.
            """
            return replace(self, **kwargs)

        def to_dict(self) -> dict:
            """Serialize this config tree to a JSON-friendly dict.

            Recurses into nested configs and containers. Callable fields (for
            example ``param_init`` hooks) are converted with ``repr()``; other
            non-JSON types log a warning and fall back to ``repr()``.

            Fields whose names start with ``_`` (such as ``_owner``) are omitted.

            Returns:
                Plain ``dict`` suitable for ``json.dumps``.
            """

            def _convert(val):
                if hasattr(val, "to_dict"):
                    return val.to_dict()
                if dataclasses.is_dataclass(val):
                    return dataclasses.asdict(val)
                if isinstance(val, (list, tuple)):
                    return type(val)(_convert(v) for v in val)
                if isinstance(val, dict):
                    return {k: _convert(v) for k, v in val.items()}
                if isinstance(val, (str, int, float, bool, type(None))):
                    return val
                if callable(val):
                    return repr(val)
                logger.warning(
                    "Config field value of type %s may not be JSON serializable",
                    type(val).__name__,
                )
                return repr(val)

            return {
                f.name: _convert(getattr(self, f.name))
                for f in fields(self)
                if not f.name.startswith("_")
            }

        def traverse(
            self,
            config_cls: type,
            *,
            _prefix: str = "",
        ) -> Iterator[tuple[str, "Configurable.Config", object, str | int]]:
            """Walk nested configs and yield nodes matching *config_cls*.

            Traverses dataclass fields and list elements recursively. Each
            yielded fully-qualified name (*fqn*) mirrors the attribute path
            under the root config (for example ``"layers.0.attention"``).

            Args:
                config_cls: Config type to match (for example a specific
                    :class:`~hyper_parallel.dmodule.module.Module` nested
                    ``Config`` subclass).
                _prefix: Internal path prefix used during recursion.

            Yields:
                Tuples ``(fqn, config, parent, field_name)`` where *parent* is
                the object holding *config* (another config or a ``list``), and
                *field_name* is the attribute name or list index. Callers can
                replace nodes in place, for example via ``setattr(parent, attr,
                new_cfg)`` or ``parent[attr] = new_cfg`` when *parent* is a list.
            """
            for field in fields(self):
                val = getattr(self, field.name)
                fqn = f"{_prefix}.{field.name}" if _prefix else field.name
                if isinstance(val, config_cls):
                    yield fqn, val, self, field.name
                elif isinstance(val, list):
                    for index, item in enumerate(val):
                        item_fqn = f"{fqn}.{index}"
                        if isinstance(item, config_cls):
                            yield item_fqn, item, val, index
                        elif hasattr(item, "traverse"):
                            yield from item.traverse(config_cls, _prefix=item_fqn)
                elif hasattr(val, "traverse"):
                    yield from val.traverse(config_cls, _prefix=fqn)

        def build(self, **kwargs):
            """Construct the :class:`Configurable` subclass bound to this config.

            Uses ``Config._owner`` set by :meth:`Configurable.__init_subclass__`.
            The config is copied with :func:`dataclasses.replace` before
            construction so the template instance is not mutated.

            Args:
                **kwargs: Runtime-only arguments for ``owner.__init__``. Must not
                    name fields already present on this config.

            Returns:
                New instance of ``self._owner``.

            Raises:
                NotImplementedError: If ``_owner`` was never wired (config not
                    defined inside a :class:`Configurable` subclass).
                ValueError: If any ``kwargs`` key overlaps a config field name.
            """
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            owner_cls: type[Configurable] = self._owner
            built_config = replace(self)
            if not kwargs:
                return _build_from_config(owner_cls, built_config)

            config_field_names = {f.name for f in fields(self)}
            overlap = config_field_names & kwargs.keys()
            if overlap:
                raise ValueError(
                    f"build() kwargs {overlap} overlap with config fields. "
                    "Put these values in the Config, not in build() kwargs."
                )
            return _build_from_config(owner_cls, built_config, **kwargs)

    def __init_subclass__(cls, **kwargs):
        """Bind nested ``Config._owner`` when a subclass defines its own ``Config``."""
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if not (isinstance(config_cls, type) and issubclass(config_cls, Configurable.Config)):
                raise TypeError(
                    f"{cls.__name__}.Config must be a Configurable.Config subclass, "
                    f"got {config_cls!r}"
                )
            # Read _owner from the Config class's own __dict__: a subclassed
            # Config inherits the parent's binding and must be (re)bound,
            # while an aliased Config (Config = OtherOwner.Config) carries its
            # own binding and must be rejected.
            if config_cls.__dict__.get("_owner") is not None:
                raise TypeError(
                    f"{cls.__name__}.Config is already bound to "
                    f"{config_cls._owner.__name__}; subclass the Config instead of "
                    "aliasing it"
                )
            config_cls._owner = cls


def enforce_module_config_slots(config_cls: type, owner_name: str) -> None:
    """Require ``@dataclass(kw_only=True, slots=True)`` on a Module nested config.

    Called from :class:`~hyper_parallel.dmodule.module.Module` ``__init_subclass__``
    for configs declared directly on Module subclasses.

    Args:
        config_cls: Nested config class to validate.
        owner_name: Outer class name (used in error messages).

    Raises:
        TypeError: If *config_cls* lacks ``slots=True`` or has positional-init
            fields.
    """
    if "__slots__" not in config_cls.__dict__:
        raise TypeError(
            f"{owner_name}.Config must use @dataclass(kw_only=True, slots=True)"
        )
    for field in fields(config_cls):
        if field.init and not field.kw_only:
            raise TypeError(
                f"{owner_name}.Config field '{field.name}' must be keyword-only"
            )


__all__ = ["Configurable", "enforce_module_config_slots"]

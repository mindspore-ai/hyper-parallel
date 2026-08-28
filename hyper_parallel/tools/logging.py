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
"""Centralized logging for hyper_parallel.

Every hyper_parallel component (FSDP, DTensor, ...) logs through a single
component-aware system instead of configuring ``logging`` by hand. Records are
rendered with a uniform prefix so a line tells you the level, which component,
when it was emitted and the exact call site::

    [DEBUG] [HP-FSDP]: 2026-06-23 11:20:31 [hsdp_state.py: 102] action=reshard ...

Rank is intentionally not in the prefix -- emit it from the message when needed.

Three concerns are deliberately decoupled so each can change independently:

* **Format** -- one ``_LOG_FORMAT`` constant plus :class:`_ContextFilter`, which
  stamps the component label onto every record. Change the look of all logs in
  one place via :func:`set_format`; the components are unaffected.
* **Components** -- a module declares its component with an explicit label,
  ``logger = get_logger("FSDP")``. The label -- not the file's import path -- is
  what ties a module to a component, so moving or renaming files never changes
  where their logs land. Onboarding a new component (e.g. ``DTensor``) needs no
  registration here: just call ``get_logger("DTensor")`` and start logging.
* **Configuration** -- per-component levels come from the ``HP_LOG_CONFIG`` env
  var (``export HP_LOG_CONFIG=FSDP:INFO,DTensor:DEBUG``) or programmatically via
  :func:`set_level` / :func:`configure` / the :data:`logger` manager. Component
  names are case-insensitive for known components (:data:`_KNOWN_COMPONENTS`); an
  unrecognised name still works but warns once, to catch typos that would
  otherwise silently produce no logs.

Logging is *off by default* (each component logger starts at ``WARNING``): the
stdout handler is always installed, but ``debug``/``info`` calls stay silent
until a component is enabled by env var or code. Output goes to ``stdout``.

Usage::

    from hyper_parallel.tools.logging import get_logger
    log = get_logger("FSDP")
    log.debug("hook=forward_pre module=%s", name)

    # or drive configuration from code:
    from hyper_parallel.tools.logging import logger
    logger.set_level("FSDP", "DEBUG")
"""
__all__ = [
    "HP_LOG_CONFIG_ENV",
    "configure",
    "get_logger",
    "logger",
    "logging_enabled",
    "set_format",
    "set_level",
]

import logging
import os
import sys
from typing import Dict, Optional, Union

# Env var that enables/levels components, e.g. "FSDP:INFO,DTensor:DEBUG".
HP_LOG_CONFIG_ENV = "HP_LOG_CONFIG"

# Logger namespace; each component lives at ``hyper_parallel.<component>``.
_NAMESPACE = "hyper_parallel"

# Components stay silent until explicitly enabled.
_DEFAULT_LEVEL = logging.WARNING

# Component label used when ``get_logger`` is called without one.
_DEFAULT_COMPONENT = "HP"

# Known component labels. This list is NOT a gate -- unknown labels still work --
# it exists only to (a) give case-insensitive matching its canonical spelling and
# (b) warn on a likely typo (e.g. ``FDSP`` for ``FSDP``), which would otherwise
# silently never match. Only FSDP is wired up today; whoever adds a new component
# (DTensor, CP, EP, ...) appends its name here (one line) to make it
# case-insensitive and silence the typo warning.
_KNOWN_COMPONENTS = (_DEFAULT_COMPONENT, "FSDP")
_CANONICAL = {name.upper(): name for name in _KNOWN_COMPONENTS}
_warned_unknown = set()

# A component listed in HP_LOG_CONFIG without an explicit level is just enabled.
_DEFAULT_ENABLED_LEVEL = logging.INFO

# ---------------------------------------------------------------------------
# Format concern -- the single place that decides how a record looks.
# ---------------------------------------------------------------------------

# The ``hp_component`` field is supplied by _ContextFilter; ``filename``/``lineno``
# are standard LogRecord fields pointing at the ``logger.debug(...)`` call site.
_LOG_FORMAT = "[%(levelname)s] [HP-%(hp_component)s]: %(asctime)s [%(filename)s: %(lineno)d] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ContextFilter(logging.Filter):
    """Stamp the component label onto every record.

    Keeping this out of the format string lets :data:`_LOG_FORMAT` stay a pure,
    declarative template -- the only thing :func:`set_format` ever needs to touch.
    """

    def __init__(self, component: str):
        super().__init__()
        self._component = component

    def filter(self, record: logging.LogRecord) -> bool:
        record.hp_component = self._component
        return True


class _HPStreamHandler(logging.StreamHandler):
    """Stream handler owned by the HyperParallel component logger."""


def _build_formatter() -> logging.Formatter:
    """Return a formatter for the current global format settings."""
    return logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


# ---------------------------------------------------------------------------
# Component registry concern -- lazily create one stdout logger per component.
# ---------------------------------------------------------------------------

_registry: Dict[str, logging.Logger] = {}


def _normalize_level(level: Union[int, str]) -> int:
    """Convert a level name or value to the integer level."""
    if isinstance(level, int):
        return level
    level_value = logging.getLevelName(str(level).upper())
    if isinstance(level_value, int):
        return level_value
    raise ValueError(f"Invalid logging level: {level!r}")


def _canonical_component(component: str) -> str:
    """Return the canonical label for ``component`` (case-insensitive).

    A known label is returned in its registered spelling, so ``fsdp`` / ``Fsdp`` /
    ``FSDP`` all resolve to ``FSDP`` and share one logger. An unknown label is
    returned unchanged but triggers a one-time stderr warning -- it still works,
    but a typo such as ``FDSP`` would otherwise silently never match
    ``get_logger("FSDP")`` and no logs would appear.
    """
    canonical = _CANONICAL.get(component.upper())
    if canonical is not None:
        return canonical
    if component not in _warned_unknown:
        _warned_unknown.add(component)
        logging.getLogger(__name__).warning(
            "unknown component %r; known components: %s. It still works but won't "
            "match a registered component -- check for a typo (e.g. 'FDSP' vs 'FSDP').",
            component,
            ", ".join(_KNOWN_COMPONENTS),
        )
    return component


def _parse_config(spec: str) -> Dict[str, int]:
    """Parse ``"FSDP:INFO,DTensor:DEBUG"`` into ``{component: level}``.

    A bare component name (``"FSDP"``) enables it at ``_DEFAULT_ENABLED_LEVEL``.
    Names are canonicalized (case-insensitive) so the config matches the labels
    used by :func:`get_logger`.
    """
    levels: Dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, level = item.partition(":")
        name = name.strip()
        if not name:
            continue
        level = level.strip()
        parsed = _normalize_level(level) if (sep and level) else _DEFAULT_ENABLED_LEVEL
        levels[_canonical_component(name)] = parsed
    return levels


def _env_levels() -> Dict[str, int]:
    """Per-component levels parsed from the ``HP_LOG_CONFIG`` env var."""
    return _parse_config(os.environ.get(HP_LOG_CONFIG_ENV, ""))


def _make_handler(component: str) -> logging.StreamHandler:
    """Build the stdout handler for ``component`` with HP formatting."""
    handler = _HPStreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_build_formatter())
    handler.addFilter(_ContextFilter(component))
    return handler


def get_logger(component: str = _DEFAULT_COMPONENT) -> logging.Logger:
    """Return the logger for ``component``, registering it lazily.

    ``component`` is an explicit label (``"FSDP"``, ``"DTensor"``, ...) -- the same
    string used in ``HP_LOG_CONFIG`` and :func:`set_level`. A module just declares
    ``logger = get_logger("FSDP")``; the label is independent of the file's path,
    so moving or renaming modules never changes which component they log under.
    All callers passing the same label share one ``[HP-<component>]`` logger, level
    and handler. The first call installs a stdout handler with HP formatting and
    applies any ``HP_LOG_CONFIG`` level for it. Matching is case-insensitive for
    known components; an unrecognised label still works but warns once (typo guard).
    """
    component = _canonical_component(component)
    existing = _registry.get(component)
    if existing is not None:
        return existing
    component_logger = logging.getLogger(f"{_NAMESPACE}.{component}")
    # Own a single stdout handler; never propagate to the root logger so HP logs
    # are not duplicated by an app-level root handler.
    component_logger.handlers = [_make_handler(component)]
    component_logger.propagate = False
    component_logger.setLevel(_env_levels().get(component, _DEFAULT_LEVEL))
    _registry[component] = component_logger
    return component_logger


# ---------------------------------------------------------------------------
# Configuration concern -- env var and programmatic entry points.
# ---------------------------------------------------------------------------


def set_level(component: str, level: Union[int, str]) -> logging.Logger:
    """Set ``component``'s level (registering it if needed) and return its logger."""
    component_logger = get_logger(component)
    component_logger.setLevel(_normalize_level(level))
    return component_logger


def configure(spec: str) -> None:
    """Apply a ``HP_LOG_CONFIG``-style spec programmatically.

    Example: ``configure("FSDP:INFO,DTensor:DEBUG")``.
    """
    for component, level in _parse_config(spec).items():
        _ = set_level(component, level)


def set_format(fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
    """Override the global log format and refresh every registered handler.

    Use ``%(hp_component)s`` in ``fmt`` for the component label, alongside any
    standard ``LogRecord`` field (``%(levelname)s``, ``%(filename)s``, ...).
    """
    global _LOG_FORMAT, _DATE_FORMAT
    if fmt is not None:
        _LOG_FORMAT = fmt
    if datefmt is not None:
        _DATE_FORMAT = datefmt
    for component_logger in _registry.values():
        for handler in component_logger.handlers:
            if isinstance(handler, _HPStreamHandler):
                handler.setFormatter(_build_formatter())


def logging_enabled(component: str, level: int = logging.DEBUG) -> bool:
    """Whether ``component`` would emit a record at ``level``."""
    return get_logger(component).isEnabledFor(level)


# ---------------------------------------------------------------------------
# Manager facade -- ``from hyper_parallel.tools.logging import logger``.
# ---------------------------------------------------------------------------


class _LoggingManager:
    """Thin object facade over the module-level configuration functions.

    Lets callers drive the logging system from code without importing each
    function separately::

        from hyper_parallel.tools.logging import logger
        logger.set_level("FSDP", "DEBUG")
        log = logger.get_logger("FSDP")
    """

    get_logger = staticmethod(get_logger)
    set_level = staticmethod(set_level)
    configure = staticmethod(configure)
    set_format = staticmethod(set_format)
    enabled = staticmethod(logging_enabled)


logger = _LoggingManager()

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
"""Distributed-aware logging helpers for the trainer.

Design (matches industry consensus across , Megatron-LM, DeepSpeed,
HF Transformers, ):

- ``logger.info / warning / error`` always fire on **every rank** so that
  rank-local failures (OOM on rank 7, NCCL timeout on rank 3) are never
  silently dropped.
- Rank-0-only progress / status messages call **explicit helpers**:
  ``logger.info_rank0(...)`` / ``logger.warning_rank0(...)``.
- One-shot dedup helpers ``info_once`` / ``warning_once`` use ``lru_cache``
  so the same message never spams every step.

Design rejected: a global ``logging.Filter`` returning ``False`` on
non-rank-0 records — that's what hyper had before, and it silently dropped
rank-local errors. Don't reintroduce it.

The functions here are also installed as bound methods on
``logging.Logger`` so any module that already does
``logger = logging.getLogger(__name__)`` gets the helpers for free.
"""
import functools
import logging
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Rank lookup — uses platform when available, falls back to env var.
# Going through platform avoids importing torch / mindspore here directly.
# ---------------------------------------------------------------------------


def _get_rank() -> int:
    """Best-effort rank lookup.

    Order:
      1. ``hyper_parallel.platform.get_platform().get_rank()`` if the process
         group is initialised.
      2. ``RANK`` env var (set by ``torchrun`` / ``msrun``).
      3. ``LOCAL_RANK`` env var.
      4. ``0`` as the last resort.
    """
    try:
        # pylint: disable=C0415
        from hyper_parallel import get_platform
        platform = get_platform()
        return int(platform.get_rank())
    except (ImportError, RuntimeError, ValueError, AttributeError):
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

# ---------------------------------------------------------------------------
# Configuration entry points.
# ---------------------------------------------------------------------------

_DEFAULT_FORMAT = (
    "[%(asctime)s][rank%(rank)s][%(levelname)s] %(name)s: %(message)s"
)
_DATE_FORMAT = "%H:%M:%S"


class _RankInjector(logging.Filter):
    """Inject the current rank into every record as ``record.rank``.

    Unlike a "drop non-rank-0 record" filter, this **never returns False** —
    every rank's record is preserved. The format string can use
    ``%(rank)s`` to display it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rank"):
            record.rank = _get_rank()
        return True


def init_logger(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DATE_FORMAT,
    stream=None,
) -> None:
    """Configure the root logger with rank-injecting formatter.

    Idempotent: calling twice replaces handlers cleanly so re-importing or
    testing doesn't double-print.

    Args:
        level: Root logger level (default ``INFO``).
        fmt: Format string. Must include ``%(rank)s`` if you want the rank
            displayed.
        datefmt: ``%(asctime)s`` format.
        stream: Output stream (default ``sys.stdout``).
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Replace handlers so re-init in tests / notebooks doesn't double-log.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.addFilter(_RankInjector())
    root.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the rank-aware helpers attached.

    Equivalent to ``logging.getLogger(name)`` plus binding
    ``info_rank0`` / ``warning_rank0`` / ``info_once`` / ``warning_once``
    on the ``logging.Logger`` class (idempotent).
    """
    _install_logger_methods()
    return logging.getLogger(name)

# ---------------------------------------------------------------------------
# Standalone module-level functions.
# ---------------------------------------------------------------------------


def info_rank0(self, msg, *args, **kwargs) -> None:
    """``logger.info`` that fires only on rank 0."""
    if _get_rank() == 0:
        kwargs.setdefault("stacklevel", 2)
        self.info(msg, *args, **kwargs)


def warning_rank0(self, msg, *args, **kwargs) -> None:
    """``logger.warning`` that fires only on rank 0."""
    if _get_rank() == 0:
        kwargs.setdefault("stacklevel", 2)
        self.warning(msg, *args, **kwargs)


@functools.lru_cache(maxsize=None)
def _info_once_cached(name: str, msg: str) -> None:
    """LRU-cached one-shot info; key = (logger name, message)."""
    if _get_rank() == 0:
        logging.getLogger(name).info(msg)


def info_once(self, msg, *args, **kwargs) -> None:  # pylint: disable=W0613
    """``logger.info`` that fires at most once across the whole run."""
    if args:
        msg = msg % args
    _info_once_cached(self.name, str(msg))


@functools.lru_cache(maxsize=None)
def _warning_once_cached(name: str, msg: str) -> None:
    if _get_rank() == 0:
        logging.getLogger(name).warning(msg)


def warning_once(self, msg, *args, **kwargs) -> None:  # pylint: disable=W0613
    """``logger.warning`` that fires at most once across the whole run."""
    if args:
        msg = msg % args
    _warning_once_cached(self.name, str(msg))

# ---------------------------------------------------------------------------
# Method installation — bind helpers onto Logger so existing
# ``logging.getLogger(__name__)`` consumers get them for free.
# ---------------------------------------------------------------------------

_INSTALLED = False


def _install_logger_methods() -> None:
    """Idempotently install rank-aware helpers on ``logging.Logger``."""
    global _INSTALLED
    if _INSTALLED:
        return
    logging.Logger.info_rank0 = info_rank0
    logging.Logger.warning_rank0 = warning_rank0
    logging.Logger.info_once = info_once
    logging.Logger.warning_once = warning_once
    _INSTALLED = True

# Install at import time so any module that imports this package gets
# the methods automatically.
_install_logger_methods()

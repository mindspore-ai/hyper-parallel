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
"""Dataset logging configuration with optional per-record rank selection."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, Literal

from hyper_parallel.platform import get_platform

RankCondition = bool | Callable[[], bool]
DatasetLogLevel = Literal["debug", "info", "warn"]
_DATASET_LOGGER_NAME = "hyper_models.components.datasets"
_DATASET_LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] "
    "[rank:%(rank_id)d] \t> %(message)s"
)
_DATASET_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
platform = get_platform()


def _get_rank() -> int:
    try:
        rank = int(platform.get_rank())
    except (RuntimeError, ValueError):
        rank = 0
    return rank


class _DatasetDebugRankFilter(logging.Filter):
    """Filter Dataset DEBUG records by distributed rank."""

    def __init__(self) -> None:
        """Default Dataset DEBUG output to rank zero."""
        super().__init__()
        self.ranks: frozenset[int] | None = frozenset({0})

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep regular logs and Dataset DEBUG logs from selected ranks."""
        rank = _get_rank()
        record.rank_id = rank
        if record.levelno != logging.DEBUG or not record.name.startswith(_DATASET_LOGGER_NAME):
            return True
        rank_enabled = getattr(record, "dataset_rank_enabled", None)
        if rank_enabled is not None:
            return bool(rank_enabled)
        return self.ranks is None or rank in self.ranks


class _DatasetLogFormatter(logging.Formatter):
    """Format Dataset log timestamps with microsecond precision."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # pylint: disable=C0103
        """Format a log record timestamp in local time."""
        record_time = datetime.fromtimestamp(record.created).astimezone()
        return record_time.strftime(datefmt) if datefmt else record_time.isoformat(timespec="microseconds")


_DEBUG_RANK_FILTER = _DatasetDebugRankFilter()


class DatasetLogger(logging.LoggerAdapter):
    """Add an optional rank condition to Dataset DEBUG records."""

    def debug(
            self, message: object, *args: object, enabled: RankCondition | None = None, **kwargs: Any,
    ) -> None:
        """Log on default ranks, or on ranks selected by ``enabled`` when provided."""
        if not self.isEnabledFor(logging.DEBUG):
            return
        if enabled is not None:
            rank_enabled = enabled() if callable(enabled) else enabled
            extra = dict(kwargs.get("extra", {}))
            extra["dataset_rank_enabled"] = rank_enabled
            kwargs["extra"] = extra
        kwargs.setdefault("stacklevel", 2)
        self.logger.debug(message, *args, **kwargs)


def get_dataset_logger(name: str) -> DatasetLogger:
    """Return a Dataset logger supporting the ``enabled`` DEBUG argument."""
    return DatasetLogger(logging.getLogger(name), {})


def enable_dataset_logging(level: DatasetLogLevel, ranks: Iterable[int] | None = (0,)) -> None:
    """Enable uniformly formatted Dataset logs at the configured level.

    Args:
        level: Minimum Dataset log level: ``debug``, ``info``, or ``warn``.
        ranks: Global ranks allowed to emit Dataset DEBUG records. Pass
            ``None`` to allow every rank. Dataset records above DEBUG remain
            visible on every rank.

    Raises:
        ValueError: If ``level`` is unsupported, or ``ranks`` is invalid.
    """
    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING}
    if level not in log_levels:
        raise ValueError(f"Unsupported Dataset log level: {level!r}")

    if ranks is None:
        selected_ranks = None
    else:
        selected_ranks = frozenset(ranks)
        if not selected_ranks or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                                     for rank in selected_ranks):
            raise ValueError("ranks must contain non-negative integers, or be None for all ranks")

    _DEBUG_RANK_FILTER.ranks = selected_ranks
    dataset_logger = logging.getLogger(_DATASET_LOGGER_NAME)
    dataset_logger.setLevel(log_levels[level])
    dataset_logger.propagate = False
    dataset_handlers = [handler for handler in dataset_logger.handlers if _DEBUG_RANK_FILTER in handler.filters]
    if not dataset_handlers:
        dataset_handler = logging.StreamHandler()
        dataset_handler.setFormatter(_DatasetLogFormatter(_DATASET_LOG_FORMAT, datefmt=_DATASET_DATE_FORMAT))
        dataset_handler.addFilter(_DEBUG_RANK_FILTER)
        dataset_logger.addHandler(dataset_handler)
        dataset_handlers.append(dataset_handler)
    for dataset_handler in dataset_handlers:
        dataset_handler.setLevel(log_levels[level])

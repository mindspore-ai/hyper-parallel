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
"""Persist rank failures before potentially blocking distributed cleanup."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
import traceback

logger = logging.getLogger(__name__)


def report_failure(error: BaseException, *, stage: str, step: int | None = None) -> None:
    """Log the original exception and optionally publish an atomic host-visible record.

    Set HYPER_RL_FAILURE_DIR to a fresh per-attempt shared directory. Reporting
    performs no device operations or collectives; a host supervisor owns process
    termination. The first record for each rank is retained across cleanup errors.
    File errors are logged without replacing the original training exception.
    """
    logger.error("RL failure during %s", stage, exc_info=(type(error), error, error.__traceback__))
    directory = os.environ.get("HYPER_RL_FAILURE_DIR")
    if not directory:
        return
    rank = os.environ.get("RANK", "0")
    if not rank.isascii() or not rank.isdecimal():
        rank = "unknown"
    destination = Path(directory) / f"failure-rank-{rank}.json"
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    try:
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": time.time(), "rank": rank, "pid": os.getpid(), "stage": stage, "step": step,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
        temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
    except OSError:
        logger.exception("Could not persist the original RL failure to %s", destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary failure record %s", temporary)

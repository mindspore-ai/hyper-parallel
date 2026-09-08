# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
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

"""Trainer-side logging helpers.

Split out of the former ``auto_models/components/utils/helper.py`` in stage 7
(05 §10.4); function names and signatures are unchanged.
"""

import logging as builtin_logging
import os
import sys
import warnings

import torch.distributed as dist
import transformers


logger = builtin_logging.getLogger(__name__)


def _is_rank_0() -> bool:
    """True if global rank 0 (or distributed not initialized)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_logging() -> None:
    """Setup logging with rank filter."""
    builtin_logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=builtin_logging.INFO,
    )


def _info_rank0(message: str) -> None:
    """Log an informational message on global rank zero."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)


def create_logger(name: str | None = None) -> "builtin_logging._Logger":
    """
    Creates a pretty logger for the third-party program.
    """
    new_logger = builtin_logging.getLogger(name)
    formatter = builtin_logging.Formatter(
        fmt="[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = builtin_logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    new_logger.addHandler(handler)
    new_logger.setLevel(builtin_logging.INFO)
    new_logger.propagate = False
    return new_logger


def enable_third_party_logging() -> None:
    """
    Enables explicit logger of the third-party libraries.
    """
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()


def disable_warning() -> None:
    """
    Enables warning filter.
    """
    # pyiceberg is an optional dependency; import lazily so that importing this
    # module does not require it.
    from pyiceberg.metrics import LoggingMetricsReporter  # pylint: disable=import-outside-toplevel

    builtin_logging.basicConfig(level=builtin_logging.ERROR)
    warnings.simplefilter("ignore")
    LoggingMetricsReporter()
    LoggingMetricsReporter._logger = builtin_logging.getLogger(LoggingMetricsReporter.__name__)
    LoggingMetricsReporter._logger.setLevel(builtin_logging.WARNING)
    LoggingMetricsReporter._logger.propagate = False


if os.getenv("DISABLE_WARNINGS", "0").lower() in ["true", "1"]:
    disable_warning()


__all__ = [
    "create_logger",
    "disable_warning",
    "enable_third_party_logging",
    "setup_logging",
]

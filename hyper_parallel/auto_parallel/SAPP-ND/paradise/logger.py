# Copyright 2025 Huawei Technologies Co., Ltd
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
"""logger for parall"""
import logging
from typing import cast
from memory_estimation.logger import logger as memo_logger

DEFAULT_STDOUT_FORMAT = (
    "[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] - %(message)s"
)
PPB_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PARALL_FORMAT = (
    "%(name)s [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"
)

FORMATTER = logging.Formatter(PARALL_FORMAT)
OUTPUT_LEVEL_NUM = logging.CRITICAL
logging.addLevelName(OUTPUT_LEVEL_NUM, "OUTPUT")


class MyLogger(logging.Logger):
    """Inherit from standard Logger and add level OUTPUT."""

    def output(self, msg, *args, **kwargs):
        """Log 'msg % args' with severity 'OUTPUT'.

        Args:
            msg (str): The message to log.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        if self.isEnabledFor(OUTPUT_LEVEL_NUM):
            self._log(OUTPUT_LEVEL_NUM, msg, args, **kwargs)


def setup_logger(name: str, level: int = logging.DEBUG):
    """Set up a logger.

    Args:
        name (str): Logger name.
        level (int, optional): Logging level. Default: logging.DEBUG.

    Returns:
        MyLogger: Configured logger.
    """
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(FORMATTER)

    previous_logger_cls = logging.getLoggerClass()
    logging.setLoggerClass(MyLogger)
    try:
        paradise_logger = cast(MyLogger, logging.getLogger(name))
    finally:
        logging.setLoggerClass(previous_logger_cls)
    paradise_logger.setLevel(level)
    paradise_logger.addHandler(ch)

    return paradise_logger


logger = setup_logger("Paradise")
perf_logger = setup_logger("Perf. estim.")


def set_verbose_level(level):
    """Assign level to each logger from a global level"""
    memo_logger.disabled = True
    perf_logger.disabled = True
    logger.disabled = True
    if level >= 1:
        logger.disabled = False
        logger.setLevel(logging.CRITICAL)
    if level >= 2:
        logger.setLevel(logging.ERROR)
    if level >= 3:
        logger.setLevel(logging.INFO)
    if level >= 4:
        memo_logger.disabled = False
        perf_logger.disabled = False
        memo_logger.setLevel(logging.CRITICAL)
        perf_logger.setLevel(logging.CRITICAL)
    if level >= 5:
        memo_logger.setLevel(logging.INFO)
        perf_logger.setLevel(logging.INFO)
        logger.setLevel(logging.DEBUG)
    if level >= 6:
        memo_logger.setLevel(logging.DEBUG)
        perf_logger.setLevel(logging.DEBUG)

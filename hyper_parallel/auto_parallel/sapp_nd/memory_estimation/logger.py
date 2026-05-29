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


def setup_logger(name: str, level: int = logging.DEBUG):
    """Set up a logger.

    Args:
        name (str): Logger name.
        level (int, optional): Logging level. Default: logging.DEBUG.

    Returns:
        logging.Logger: Configured logger.
    """
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(FORMATTER)

    logging.Logger.output = logging.Logger.critical
    memory_logger = logging.getLogger(name)
    memory_logger.setLevel(level)
    memory_logger.addHandler(ch)

    return memory_logger


logger = setup_logger("memory_estimation")

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
import os
import logging


def _get_rank_id():
    """Try to get rank from environment variables set by torchrun / torch.distributed."""
    # Priority: RANK (global rank) > LOCAL_RANK (node-local rank)
    # If not found, assume single process with rank 0
    rank = os.getenv("RANK")
    if rank is not None:
        return int(rank)
    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)
    return 0


def setup_logger(log_dir=None, log_level_env="0"):
    """
    Configure the root logger.

    Args:
        log_dir (str, optional): Directory to save per-rank log files. 
                                 If None, only console output is used.
        log_level_env (str): Log level string ("0": WARNING, "1": INFO, "2": DEBUG)
    """
    level_map = {"0": logging.WARNING, "1": logging.INFO, "2": logging.DEBUG}
    log_level = level_map.get(log_level_env, logging.WARNING)

    # Get rank ID
    rank_id = _get_rank_id()

    # Create formatter
    formatter = logging.Formatter(
        fmt=f"[PID %(process)d | Rank {rank_id}] %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%H:%M:%S"
    )

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (if log_dir specified)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rank_{rank_id}.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

# -----------------------------
# Default initialization
# -----------------------------


# Read LOG_LEVEL from env (for backward compatibility)
log_level_env = os.getenv("LOG_LEVEL", "0")

# Optional: read LOG_DIR from environment variable
log_dir = os.getenv("LOG_DIR")  # e.g., export LOG_DIR=./logs

# Setup logger
setup_logger(log_dir=log_dir, log_level_env=log_level_env)

# Get logger instance
logger = logging.getLogger(__name__)

# Convenience functions with stacklevel=2


def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, stacklevel=2, **kwargs)


def info(msg, *args, **kwargs):
    logger.info(msg, *args, stacklevel=2, **kwargs)


def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, stacklevel=2, **kwargs)

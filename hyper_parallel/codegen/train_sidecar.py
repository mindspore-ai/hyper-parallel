# Copyright 2026 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Best-effort Codegen sidecar for the standard training entry point."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

_DISTRIBUTED_ENV_NAMES = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
)


def _codegen_environment() -> dict[str, str]:
    """Return a child environment without torchrun distributed identity."""
    child_env = os.environ.copy()
    for name in _DISTRIBUTED_ENV_NAMES:
        child_env.pop(name, None)
    return child_env


def _monitor_sidecar(process: subprocess.Popen, config_path: str) -> None:
    """Reap the Codegen process and report failure without stopping training."""
    return_code = process.wait()
    if return_code == 0:
        logger.info("Codegen sidecar completed for %s", config_path)
        return
    logger.warning(
        "Codegen sidecar failed for %s with exit code %s; native training continues",
        config_path,
        return_code,
    )


def start_codegen_sidecar(
    config_path: str | Path,
    overrides: Sequence[str] = (),
) -> Optional[subprocess.Popen]:
    """Start device-free bundle generation on global rank 0.

    The child receives a copy of the parent environment with torchrun identity
    removed. The parent environment and native training process are unchanged.
    Startup and generation failures are best effort and never block training.

    Args:
        config_path: Training YAML used by the native Trainer.
        overrides: Trainer CLI overrides forwarded unchanged to generation.

    Returns:
        The rank-0 Codegen child process, or ``None`` on nonzero ranks and when
        the child cannot be started.
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return None

    resolved_config = str(Path(config_path).resolve())
    command = [
        sys.executable,
        "-m",
        "hyper_parallel.codegen",
        "--verbose",
        "generate",
        "--config",
        resolved_config,
        *overrides,
    ]
    try:
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            env=_codegen_environment(),
            close_fds=True,
        )
    except OSError as exc:
        logger.warning(
            "Codegen sidecar could not start for %s: %s; native training continues",
            resolved_config,
            exc,
        )
        return None

    monitor = threading.Thread(
        target=_monitor_sidecar,
        args=(process, resolved_config),
        name="hyper-parallel-codegen-sidecar",
        daemon=True,
    )
    monitor.start()
    logger.info("Started Codegen sidecar process %s for %s", process.pid, resolved_config)
    return process


def run_training_with_codegen(
    parse_training_args: Callable[[], Any],
    trainer_type: Callable[[Any], Any],
    argv: Optional[Sequence[str]] = None,
) -> None:
    """Start rank-zero generation before running the standard Trainer.

    Args:
        parse_training_args: Native Trainer configuration parser.
        trainer_type: Native Trainer class or compatible factory.
        argv: Complete training command arguments. Defaults to ``sys.argv``.
    """
    arguments = sys.argv if argv is None else argv
    config = parse_training_args()
    start_codegen_sidecar(config._yaml_path, arguments[2:])  # pylint: disable=protected-access
    trainer_type(config).train()


__all__ = ["run_training_with_codegen", "start_codegen_sidecar"]

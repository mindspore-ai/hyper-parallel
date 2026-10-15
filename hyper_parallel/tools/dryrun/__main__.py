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
"""Run one configured logical training step without entering the real Trainer."""

import copy
import logging
import os
from collections.abc import Mapping

from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.tools.dryrun import HyperModelsDryRunRunner
from hyper_parallel.tools.dryrun.config import DryRunConfig
from hyper_parallel.tools.dryrun.runtime import DryRunRuntime

logger = logging.getLogger(__name__)


def _resolve_config(value: object) -> DryRunConfig:
    """Normalize the trainer's generic dry-run mapping."""
    if isinstance(value, DryRunConfig):
        return value
    if isinstance(value, Mapping):
        return DryRunConfig.from_mapping(value)
    raise TypeError("The training config must contain a dry_run mapping")


def _torchrun_identity() -> tuple[int, int, int] | None:
    """Return rank identity when the CLI was launched by torchrun."""
    names = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(name in os.environ for name in names):
        return None
    return tuple(int(os.environ[name]) for name in names)


def _configured_ranks(dry_run: DryRunConfig, world_size: int) -> list[int]:
    """Resolve and validate the ranks selected by the config."""
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("dry_run.virtual_world_size must be a positive integer")
    ranks = dry_run.ranks if dry_run.ranks is not None else [0]
    if (
            not ranks
            or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)
            or any(rank >= world_size for rank in ranks)
    ):
        raise ValueError(
            "dry_run.ranks must be a non-empty list of unique non-negative "
            f"integers smaller than world size {world_size}"
        )
    return ranks


def main() -> None:
    """Run dry-run from a resolved training YAML and optional CLI overrides."""
    config = parse_training_args()
    dry_run = _resolve_config(config.dry_run)
    if not dry_run.enabled:
        logger.info("Dry-run is disabled by dry_run.enabled=false")
        return
    config.dry_run = dry_run
    torchrun_identity = _torchrun_identity()
    if torchrun_identity is not None:
        rank, world_size, _ = torchrun_identity
        if dry_run.virtual_world_size is not None and dry_run.virtual_world_size != world_size:
            raise ValueError(
                "dry_run.virtual_world_size must match torchrun WORLD_SIZE: "
                f"{dry_run.virtual_world_size} versus {world_size}"
            )
        if rank not in _configured_ranks(dry_run, world_size):
            logger.info("Skipping dry-run rank %s; dry_run.ranks=%s", rank, dry_run.ranks)
            return
        HyperModelsDryRunRunner(config).run()
        return

    world_size = dry_run.virtual_world_size or 1
    for rank in _configured_ranks(dry_run, world_size):
        logger.info("Running dry-run rank %s/%s", rank, world_size)
        rank_config = copy.deepcopy(config)
        HyperModelsDryRunRunner(
            rank_config,
            DryRunRuntime(rank=rank, world_size=world_size, local_rank=0),
        ).run()


if __name__ == "__main__":
    main()

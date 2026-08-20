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
"""Persistent Actor checkpoint lifecycle for synchronous RL training."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from rl.utils.monitoring.config import sanitize_config

from hyper_parallel import get_platform
from hyper_parallel.core.distributed_checkpoint import (
    load as dcp_load,
    save as dcp_save,
)

platform = get_platform()
logger = logging.getLogger(__name__)


class RLCheckpointManager:
    """Compose HyperModels checkpoint IO with RL completion metadata."""

    def __init__(
        self,
        trainer: Any,
        config: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        run_synchronized: Callable[[str, Callable[[], None]], None],
    ) -> None:
        """Store role state and initialize the checkpoint output directory."""
        self.trainer = trainer
        self.config = config
        self.resolved_config = resolved_config
        self.run_synchronized = run_synchronized
        self.output_dir = Path(str(config["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def save_steps(self) -> int:
        """Return the periodic checkpoint interval."""
        return int(self.config.get("save_steps", 0))

    @property
    def load_path(self) -> Optional[str]:
        """Return an explicitly configured resume path."""
        value = self.config.get("load_path")
        return None if value is None else str(value)

    def directory(self, step: int) -> Path:
        """Return the distributed checkpoint directory for one step."""
        return self.output_dir / f"step_{step}"

    def will_save(self, step: int) -> bool:
        """Return whether periodic or final policy state is due."""
        periodic = self.save_steps > 0 and step % self.save_steps == 0
        final = bool(self.config.get("save_final", True)) and step == self.trainer.state.max_steps
        return periodic or final

    def validate_resume(self) -> None:
        """Reject incomplete or topology-incompatible checkpoints before loading."""
        if not self.load_path:
            return

        def validate_files() -> None:
            """Validate artifacts visible to the current rank."""
            checkpoint_dir = Path(self.load_path)
            if not checkpoint_dir.is_dir():
                raise RuntimeError(f"Checkpoint directory does not exist: {checkpoint_dir}")
            manifest_path = checkpoint_dir / "checkpoint_complete.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Checkpoint completion manifest is missing: {manifest_path}"
                )
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            world_size = platform.get_world_size()
            if int(manifest.get("world_size", -1)) != world_size:
                raise RuntimeError(
                    "Checkpoint world size does not match the active job: "
                    f"checkpoint={manifest.get('world_size')}, active={world_size}"
                )
            rank_state = checkpoint_dir / f"rank_{platform.get_rank()}"
            if not rank_state.is_dir():
                raise RuntimeError(
                    f"Checkpoint rank-local state is missing: {rank_state}"
                )

        self.run_synchronized("checkpoint resume preflight", validate_files)

    def begin(self, state: Any) -> None:
        """Restore policy, optimizer, scheduler, and progress state."""
        if not self.load_path:
            return
        checkpoint_dir = Path(self.load_path)

        def load() -> None:
            """Load distributed and rank-local state into live runtimes."""
            checkpoint_state = {"model": self.trainer.model.state_dict()}
            dcp_load(
                checkpoint_state,
                checkpoint_id=checkpoint_dir,
                use_collectives=True,
            )
            self.trainer.model.load_state_dict(checkpoint_state["model"])
            serialized_rank_state = {"runtime": b""}
            dcp_load(
                serialized_rank_state,
                checkpoint_id=checkpoint_dir / f"rank_{platform.get_rank()}",
                use_collectives=False,
            )
            rank_state_value = serialized_rank_state["runtime"]
            if not isinstance(rank_state_value, Mapping):
                raise ValueError(
                    "Checkpoint rank-local runtime state must deserialize to a mapping"
                )
            rank_state = dict(rank_state_value)
            platform.set_rng_state(rank_state["cpu_rng"])
            platform.set_rng_state(
                rank_state["device_rng"],
                self.trainer.device,
                self.trainer.device_handle,
            )
            self.trainer.train_dataloader.load_state_dict(rank_state["dataloader"])
            if self.trainer.optimizer is not None:
                self.trainer.optimizer.load_state_dict(rank_state["optimizer"])
            if self.trainer.lr_scheduler is not None:
                self.trainer.lr_scheduler.load_state_dict(rank_state["scheduler"])

            with (checkpoint_dir / "extra_state.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            state.global_step = int(metadata.get("global_step", 0))
            state.epoch = int(metadata.get("epoch", 0))
            state.consumed_samples = int(metadata.get("consumed_samples", 0))
            state.consumed_tokens = int(metadata.get("consumed_tokens", 0))

        self.run_synchronized("checkpoint resume", load)

    def invalidate(self, step: int) -> None:
        """Remove a stale completion marker before overwriting a checkpoint."""
        def remove_manifest() -> None:
            """Remove the marker on the metadata owner rank."""
            if platform.get_rank() == 0:
                (self.directory(step) / "checkpoint_complete.json").unlink(
                    missing_ok=True
                )

        self.run_synchronized("checkpoint manifest invalidation", remove_manifest)

    def complete_step(self, state: Any, *, loss: float, grad_norm: float) -> None:
        """Save periodic policy state after a successfully published update."""
        del loss, grad_norm
        step = int(state.global_step)
        if self.save_steps <= 0 or step % self.save_steps != 0:
            return
        self._save(state)

    def finalize(self, state: Any) -> None:
        """Save and optionally reload the final Actor checkpoint."""
        if not bool(self.config.get("save_final", True)):
            return
        step = int(state.global_step)
        if self.save_steps <= 0 or step % self.save_steps != 0:
            self._save(state)
        if bool(self.config.get("verify_reload", False)):
            self._verify_reload(self.directory(step))

    def _save(self, state: Any) -> None:
        """Persist distributed model state and then publish RL completion metadata."""
        step = int(state.global_step)
        self.invalidate(step)
        checkpoint_dir = self.directory(step)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_state = {"model": self.trainer.model.state_dict()}
        dcp_save(
            checkpoint_state,
            checkpoint_id=checkpoint_dir,
            use_collectives=True,
        )
        def save_rank_state() -> None:
            """Persist RNG, scheduler, and dataloader state independently per rank."""
            rank_state = {
                "cpu_rng": platform.get_rng_state(),
                "device_rng": platform.get_rng_state(
                    self.trainer.device,
                    self.trainer.device_handle,
                ),
                "dataloader": self.trainer.train_dataloader.state_dict(),
            }
            if self.trainer.lr_scheduler is not None:
                rank_state["scheduler"] = self.trainer.lr_scheduler.state_dict()
            if self.trainer.optimizer is not None:
                rank_state["optimizer"] = self.trainer.optimizer.state_dict()
            dcp_save(
                {"runtime": pickle.dumps(rank_state)},
                checkpoint_id=checkpoint_dir / f"rank_{platform.get_rank()}",
                use_collectives=False,
            )

        self.run_synchronized("checkpoint rank-state save", save_rank_state)
        self._write_metadata(checkpoint_dir, state)

    def _write_metadata(self, checkpoint_dir: Path, state: Any) -> None:
        """Persist resolved configuration and publish the completion marker."""
        def write() -> None:
            """Atomically publish rank-zero metadata after all state is durable."""
            if platform.get_rank() != 0:
                return
            step = int(state.global_step)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with (checkpoint_dir / "resolved_config.yaml").open(
                "w", encoding="utf-8"
            ) as handle:
                yaml.safe_dump(
                    sanitize_config(self.resolved_config),
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
            with (checkpoint_dir / "extra_state.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "global_step": step,
                        "epoch": int(state.epoch),
                        "consumed_samples": int(state.consumed_samples),
                        "consumed_tokens": int(state.consumed_tokens),
                    },
                    handle,
                )
            manifest_path = checkpoint_dir / "checkpoint_complete.json"
            temporary = checkpoint_dir / f".{manifest_path.name}.{os.getpid()}.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"step": step, "world_size": platform.get_world_size()},
                    handle,
                )
            os.replace(temporary, manifest_path)

        self.run_synchronized("checkpoint config write", write)

    def _verify_reload(self, checkpoint_dir: Path) -> None:
        """Reload final model state through distributed checkpoint primitives."""
        def reload_checkpoint() -> None:
            """Reload the just-written policy into the live model."""
            if not checkpoint_dir.is_dir():
                raise RuntimeError(
                    f"Final checkpoint directory was not created: {checkpoint_dir}"
                )
            checkpoint_state = {"model": self.trainer.model.state_dict()}
            dcp_load(
                checkpoint_state,
                checkpoint_id=checkpoint_dir,
                use_collectives=True,
            )
            self.trainer.model.load_state_dict(checkpoint_state["model"])

        self.run_synchronized("checkpoint reload verification", reload_checkpoint)
        logger.info(
            "rank=%d verified checkpoint reload from %s",
            platform.get_rank(),
            checkpoint_dir,
        )


__all__ = ["RLCheckpointManager"]

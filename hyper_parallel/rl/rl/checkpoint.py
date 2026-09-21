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
"""Top-level persistent checkpoint lifecycle for synchronous RL training."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist
import yaml

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.distributed_checkpoint import load as dcp_load
from hyper_parallel.core.distributed_checkpoint import save as dcp_save
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager
from rl.utils.monitoring.config import sanitize_config

logger = logging.getLogger(__name__)


def _clone_shared_checkpoint_tensors(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    """Detach duplicate tensor storage while preserving every checkpoint key."""
    storage_owners: dict[tuple[str, int], str] = {}
    cloned_aliases = []
    for name, value in state_dict.items():
        data_ptr = getattr(value, "data_ptr", None)
        numel = getattr(value, "numel", None)
        clone = getattr(value, "clone", None)
        if not callable(data_ptr) or not callable(numel) or not callable(clone):
            continue
        if int(numel()) == 0:
            continue
        pointer = int(data_ptr())
        if pointer == 0:
            continue
        storage_key = (str(getattr(value, "device", "unknown")), pointer)
        owner = storage_owners.setdefault(storage_key, name)
        if owner == name:
            continue
        state_dict[name] = clone()
        cloned_aliases.append((name, owner))
    return state_dict, tuple(cloned_aliases)


class RLCheckpointManager:
    """Compose HyperAutoModel checkpoint IO with RL completion metadata."""

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
        self._last_saved_step: Optional[int] = None
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
            world_size = dist.get_world_size()
            if int(manifest.get("world_size", -1)) != world_size:
                raise RuntimeError(
                    "Checkpoint world size does not match the active job: "
                    f"checkpoint={manifest.get('world_size')}, active={world_size}"
                )
            if bool(manifest.get("critic", False)) != (getattr(self.trainer, "critic", None) is not None):
                raise RuntimeError("Checkpoint Critic ownership does not match the active algorithm")
            rank_state = checkpoint_dir / f"rank_{dist.get_rank()}"
            if not rank_state.is_dir():
                raise RuntimeError(f"Checkpoint rank-local state is missing: {rank_state}")
            if not (checkpoint_dir / "extra_state.json").is_file():
                raise RuntimeError(f"Checkpoint training progress is missing: {checkpoint_dir}")

        self.run_synchronized("checkpoint resume preflight", validate_files)

    def begin(self, state: Any) -> None:
        """Restore policy, optimizer, scheduler, RNG, dataloader, and progress state."""
        if not self.load_path:
            return
        checkpoint_dir = Path(self.load_path)

        def load() -> None:
            """Load distributed and rank-local state into live runtimes."""
            checkpoint_state = {"model": self.trainer.model.state_dict()}
            critic = getattr(self.trainer, "critic", None)
            if critic is not None:
                checkpoint_state["critic"] = critic.critic_model.state_dict()
            dcp_load(
                checkpoint_state,
                checkpoint_id=checkpoint_dir,
                use_collectives=True,
            )
            self.trainer.model.load_state_dict(checkpoint_state["model"])
            if critic is not None:
                critic.critic_model.load_state_dict(checkpoint_state["critic"])
            serialized_rank_state = {"runtime": b""}
            dcp_load(
                serialized_rank_state,
                checkpoint_id=checkpoint_dir / f"rank_{dist.get_rank()}",
                use_collectives=False,
            )
            rank_state_value = serialized_rank_state["runtime"]
            if not isinstance(rank_state_value, Mapping):
                raise ValueError(
                    "Checkpoint rank-local runtime state must deserialize to a mapping"
                )
            rank_state = dict(rank_state_value)
            self.trainer.train_dataloader.load_state_dict(rank_state["dataloader"])
            if self.trainer.optimizer is not None:
                with SkipDTensorDispatch():
                    self.trainer.optimizer.load_state_dict(rank_state["optimizer"])
            if self.trainer.lr_scheduler is not None:
                self.trainer.lr_scheduler.load_state_dict(rank_state["scheduler"])
            if critic is not None:
                with SkipDTensorDispatch():
                    critic.optimizer.load_state_dict(rank_state["critic_optimizer"])
                if critic.lr_scheduler is not None:
                    critic.lr_scheduler.load_state_dict(rank_state["critic_scheduler"])

            with (checkpoint_dir / "extra_state.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            state.global_step = int(metadata.get("global_step", 0))
            state.epoch = int(metadata.get("epoch", 0))
            state.consumed_samples = int(metadata.get("consumed_samples", 0))
            state.consumed_tokens = int(metadata.get("consumed_tokens", 0))
            # Restore generators last so loading other state cannot advance them.
            torch.set_rng_state(rank_state["cpu_rng"])
            self.trainer.device_handle.set_rng_state(
                rank_state["device_rng"], self.trainer.device,
            )

        self.run_synchronized("checkpoint resume", load)

    def invalidate(self, step: int) -> None:
        """Remove a stale completion marker before overwriting a checkpoint."""
        def remove_manifest() -> None:
            """Remove the marker on the metadata owner rank."""
            if dist.get_rank() == 0:
                (self.directory(step) / "checkpoint_complete.json").unlink(
                    missing_ok=True
                )

        self.run_synchronized("checkpoint manifest invalidation", remove_manifest)
        if self._last_saved_step == step:
            self._last_saved_step = None

    def complete_step(self, state: Any, *, loss: float, grad_norm: float) -> None:
        """Save periodic policy state after a successfully published update."""
        del loss, grad_norm
        step = int(state.global_step)
        if self.save_steps <= 0 or step % self.save_steps != 0:
            return
        self._save(state)

    def finalize(self, state: Any) -> None:
        """Save resumable state, verify it if requested, and export final HF weights."""
        if not bool(self.config.get("save_final", True)):
            return
        step = int(state.global_step)
        if self._last_saved_step != step:
            self._save(state)
        if bool(self.config.get("verify_reload", False)):
            self._verify_reload(self.directory(step))
        self._export_hf(self.directory(step))

    def _save(self, state: Any) -> None:
        """Persist distributed model state and then publish RL completion metadata."""
        step = int(state.global_step)
        self.invalidate(step)
        checkpoint_dir = self.directory(step)
        checkpoint_state = {}

        def prepare_model_state() -> None:
            """Prepare local model shards before any rank enters collective IO."""
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_state, cloned_aliases = _clone_shared_checkpoint_tensors(
                self.trainer.model.state_dict()
            )
            checkpoint_state["model"] = model_state
            critic = getattr(self.trainer, "critic", None)
            if critic is not None:
                checkpoint_state["critic"] = critic.critic_model.state_dict()
            if cloned_aliases and dist.get_rank() == 0:
                logger.info("Cloned shared checkpoint storage for tied aliases: %s", cloned_aliases)

        def save_model_state() -> None:
            """Persist model shards on all ranks before runtime metadata."""
            dcp_save(checkpoint_state, checkpoint_id=checkpoint_dir, use_collectives=True)

        self.run_synchronized("checkpoint model preparation", prepare_model_state)
        self.run_synchronized("checkpoint model save", save_model_state)
        del checkpoint_state

        def save_rank_state() -> None:
            """Persist RNG, scheduler, optimizer, and dataloader state per rank."""
            rank_state = {
                "cpu_rng": torch.get_rng_state(),
                "device_rng": self.trainer.device_handle.get_rng_state(self.trainer.device),
                "dataloader": self.trainer.train_dataloader.state_dict(),
            }
            if self.trainer.lr_scheduler is not None:
                rank_state["scheduler"] = self.trainer.lr_scheduler.state_dict()
            if self.trainer.optimizer is not None:
                rank_state["optimizer"] = self._optimizer_state_dict()
            critic = getattr(self.trainer, "critic", None)
            if critic is not None:
                with SkipDTensorDispatch():
                    rank_state["critic_optimizer"] = critic.optimizer.state_dict()
                if critic.lr_scheduler is not None:
                    rank_state["critic_scheduler"] = critic.lr_scheduler.state_dict()
            dcp_save(
                {"runtime": pickle.dumps(rank_state)},
                checkpoint_id=checkpoint_dir / f"rank_{dist.get_rank()}",
                use_collectives=False,
            )

        self.run_synchronized("checkpoint rank-state save", save_rank_state)
        self._write_metadata(checkpoint_dir, state)
        self._last_saved_step = step

    def _export_hf(self, checkpoint_dir: Path) -> None:
        """Gather final weights through HyperAutoModel and save tokenizer assets."""
        export_dir = checkpoint_dir / "hf"

        def export() -> None:
            """All ranks gather; only the writing rank saves tokenizer files."""
            model_config = self.trainer.model.config
            try:
                wrote_weights = CheckpointManager(self.trainer.model).save_pretrained(
                    export_dir,
                    save_original_format=True,
                )
            finally:
                # Transformers serializes the dynamic HSDP class name; inference
                # loaders need the original architecture registered for this model.
                model_config.architectures = [self.trainer.model_registration.hf_architecture]
            if wrote_weights:
                model_config.save_pretrained(export_dir)
                self.trainer.tokenizer.save_pretrained(str(export_dir))

        self.run_synchronized("checkpoint HF export", export)
        logger.info("rank=%d exported final HF checkpoint to %s", dist.get_rank(), export_dir)

    def _optimizer_state_dict(self) -> dict[str, Any]:
        """Export optimizer state without dispatching its fused initialization step."""
        with SkipDTensorDispatch():
            return self.trainer.optimizer.state_dict()

    def _write_metadata(self, checkpoint_dir: Path, state: Any) -> None:
        """Persist resolved configuration and publish the completion marker."""
        def write() -> None:
            """Atomically publish rank-zero metadata after all state is durable."""
            if dist.get_rank() != 0:
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
                    {"step": step, "world_size": dist.get_world_size(),
                     "critic": getattr(self.trainer, "critic", None) is not None},
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
            dist.get_rank(),
            checkpoint_dir,
        )


__all__ = ["RLCheckpointManager"]

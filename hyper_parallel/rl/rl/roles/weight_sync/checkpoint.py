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
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import yaml
from hyper_parallel import get_platform
from hyper_parallel.core.distributed_checkpoint import load as dcp_load
from hyper_parallel.trainer.callbacks.base import CheckpointCallback
from rl.utils.monitoring.config import sanitize_config
platform = get_platform()
logger = logging.getLogger(__name__)
class RLCheckpointManager:
    """Compose Hyper-Parallel checkpoint IO with RL completion metadata."""
    def __init__(
        self,
        trainer: Any,
        config: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        run_synchronized: Callable[[str, Callable[[], None]], None],
    ) -> None:
        self.trainer = trainer
        self.config = config
        self.resolved_config = resolved_config
        self.run_synchronized = run_synchronized
        self.output_dir = Path(str(config["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.callback = CheckpointCallback(trainer)
    @property
    def save_steps(self) -> int:
        return int(self.config.get("save_steps", 0))
    @property
    def load_path(self) -> Optional[str]:
        return self.callback.load_path
    def directory(self, step: int) -> Path:
        return self.output_dir / f"step_{step}"
    def will_save(self, step: int) -> bool:
        periodic = self.save_steps > 0 and step % self.save_steps == 0
        final = bool(self.config.get("save_final", True)) and step == self.trainer.state.max_steps
        return periodic or final
    def validate_resume(self) -> None:
        """Reject incomplete or topology-incompatible checkpoints before loading."""
        if not self.load_path:
            return
        def validate_files() -> None:
            checkpoint_dir = Path(self.load_path)
            manifest_path = checkpoint_dir / "checkpoint_complete.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"Checkpoint completion manifest is missing: {manifest_path}")
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            world_size = platform.get_world_size()
            if int(manifest.get("world_size", -1)) != world_size:
                raise RuntimeError(
                    "Checkpoint world size does not match the active job: "
                    f"checkpoint={manifest.get('world_size')}, active={world_size}"
                )
            rank = platform.get_rank()
            required = [
                checkpoint_dir / "extra_state.json",
                checkpoint_dir / f"rng_rank{rank}.pt",
            ]
            if self.trainer.optimizer is not None:
                required.append(checkpoint_dir / f"optimizer_rank{rank}.pt")
            if self.trainer.lr_scheduler is not None:
                required.append(checkpoint_dir / "scheduler.pt")
            if hasattr(self.trainer.train_dataloader, "state_dict"):
                required.append(checkpoint_dir / f"dataloader_rank{rank}.pt")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"Checkpoint is incomplete; missing artifacts={missing}")
        self.run_synchronized("checkpoint resume preflight", validate_files)
    def begin(self, state: Any) -> None:
        """Start the callback and surface asynchronous load errors on every rank."""
        self.callback.on_train_begin(state)
        self.run_synchronized("checkpoint resume", self.callback.raise_if_load_failed)
    def invalidate(self, step: int) -> None:
        def remove_manifest() -> None:
            if platform.get_rank() == 0:
                (self.directory(step) / "checkpoint_complete.json").unlink(missing_ok=True)
        self.run_synchronized("checkpoint manifest invalidation", remove_manifest)
    def complete_step(self, state: Any, *, loss: float, grad_norm: float) -> None:
        """Run periodic callback bookkeeping and finalize saved artifacts."""
        step = int(state.global_step)
        periodic = self.save_steps > 0 and step % self.save_steps == 0
        callback_saves = (
            self.callback.save_steps > 0
            and step % self.callback.save_steps == 0
        )
        if callback_saves:
            self.invalidate(step)
        self.callback.on_step_end(state, loss=loss, grad_norm=grad_norm)
        if callback_saves:
            self._ensure_saved(step)
        if periodic:
            self._write_metadata(self.directory(step), step)
    def finalize(self, state: Any) -> None:
        """Save, annotate, and optionally reload the final Actor checkpoint."""
        if not bool(self.config.get("save_final", True)):
            return
        step = int(state.global_step)
        self.invalidate(step)
        self.callback.save_now(state)
        self._ensure_saved(step)
        checkpoint_dir = self.directory(step)
        self._write_metadata(checkpoint_dir, step)
        if bool(self.config.get("verify_reload", False)):
            self._verify_reload(checkpoint_dir)
    def _ensure_saved(self, step: int) -> None:
        self.run_synchronized(
            "checkpoint save",
            lambda: self.callback.ensure_saved(step),
        )
    def _write_metadata(self, checkpoint_dir: Path, step: int) -> None:
        """Persist checkpoint metadata after model files are durable."""
        def write() -> None:
            if platform.get_rank() != 0:
                return
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
        """Reload a final checkpoint to verify its serialized state."""
        def reload_checkpoint() -> None:
            if not checkpoint_dir.is_dir():
                raise RuntimeError(
                    f"Final checkpoint directory was not created: {checkpoint_dir}"
                )
            model_state = self.trainer.model.state_dict()
            dcp_load(
                model_state,
                checkpoint_id=str(checkpoint_dir),
                use_collectives=False,
            )
            self.trainer.model.load_state_dict(model_state)
        self.run_synchronized("checkpoint reload verification", reload_checkpoint)
        logger.info(
            "rank=%d verified checkpoint reload from %s",
            platform.get_rank(),
            checkpoint_dir,
        )
__all__ = ["RLCheckpointManager"]

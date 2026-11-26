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
"""Default full-state VeRL checkpoints through the native HyperParallel DCP interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from safetensors.torch import save
from torch import Tensor
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

from hyper_parallel.components.checkpoint import build_checkpointer, initialize_optimizer_state
from hyper_parallel.core.distributed_checkpoint import filesystem_storage
from hyper_parallel.models.external_state import get_model_external_state
from hyper_parallel.trainer.runtime.device import synchronize


def _save_safetensors(tensors: dict[str, Tensor], file_path: str) -> None:
    """Serialize one shard in memory and write it without Rust file publication."""
    Path(file_path).write_bytes(save(tensors))


class VeRLCheckpointAdapter(BaseCheckpointManager):
    """Save default VeRL training state directly through HyperParallel's DCP interface."""

    def __init__(self, trainer: Any, **kwargs: Any) -> None:
        """Bind native training components for checkpointing."""
        if get_model_external_state(trainer.model) is not None:
            raise NotImplementedError("VeRL checkpoints do not support external parameter providers")
        super().__init__(**kwargs)
        self.trainer = trainer
        self.checkpointer = build_checkpointer(extra_state_per_rank=True)

    def save_checkpoint(
        self,
        local_path: str | None,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> None:
        """Synchronously save model, optimizer and per-rank continuation state."""
        if hdfs_path is not None or max_ckpt_to_keep is not None:
            raise NotImplementedError("Checkpoint remote paths and custom retention are not supported")
        if local_path is None:
            return
        if self.optimizer is None:
            raise NotImplementedError("Default full-state checkpoints require a trainable actor role")
        if self.rank == 0:
            destination = Path(local_path)
            if any(destination.glob("*.metadata")):
                raise FileExistsError(f"Checkpoint already exists: {local_path}")
            assets = destination / "huggingface"
            self.model.config.save_pretrained(assets)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(assets)
        # Optimizer state export may read CPU tensors from a pending offload.
        synchronize()
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "extra_state": {
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "rng_state": self.get_rng_state(),
            },
        }
        self.trainer.model_integration.capture_checkpoint_payload("before_save", local_path, state)
        # Restore the native writer even if the synchronous VeRL save fails.
        with patch.object(filesystem_storage, "save_checkpoint_file", _save_safetensors):
            self.checkpointer.save(
                local_path,
                state,
                global_step=global_step,
                save_async=False,
            )
        if self.rank == 0:
            self.register_checkpoint(local_path, None)  # pylint: disable=no-member

    def load_checkpoint(
        self,
        local_path: str | None,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> None:
        """Load DCP state and install it into the live training objects."""
        if hdfs_path is not None or del_local_after_load:
            raise NotImplementedError("Checkpoint remote paths and deletion after load are not supported")
        if local_path is None:
            return
        if (Path(local_path) / "hp_checkpoint.json").is_file():
            raise ValueError("Legacy hp_checkpoint.json requires the original engine to restore")
        if self.optimizer is None:
            raise NotImplementedError("Default full-state checkpoints require a trainable actor role")
        if not initialize_optimizer_state(self.optimizer):
            raise RuntimeError("Cannot initialize optimizer state for checkpoint restoration")
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self.checkpointer.load(
            local_path,
            state,
            extra_state_skeleton={
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "rng_state": self.get_rng_state(),
            },
        )
        self.trainer.model_integration.capture_checkpoint_payload("after_load", local_path, state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        extra = state["extra_state"]
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        self.load_rng_state(extra["rng_state"])

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
"""Checkpointer — stub for DCP-based checkpoint save/load engine.

Following design doc 04_checkpoint.md §5.
Full implementation requires PyTorch Distributed Checkpointing (DCP).
"""

import logging
import os
from typing import Any, Optional

import torch

from hyper_models.components.checkpoint.config import CheckpointingConfig

logger = logging.getLogger(__name__)


class Checkpointer:
    """Checkpoint save/load engine.

    Stub — implements basic torch.save/load instead of DCP.
    Full DCP implementation will use torch.distributed.checkpoint.
    """

    def __init__(self, config: CheckpointingConfig):
        self.config = config

    def save_model(self, model, path: str, **kwargs) -> None:
        """Save model weights (stub)."""
        os.makedirs(path, exist_ok=True)
        if isinstance(model, list):
            for i, part in enumerate(model):
                torch.save(part.state_dict(), os.path.join(path, f"model_part_{i}.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    def save_optimizer(self, model_ref, optimizer, path: str, **kwargs) -> None:
        """Save optimizer state (stub)."""
        os.makedirs(path, exist_ok=True)
        if isinstance(optimizer, list):
            state = {f"opt_{i}": opt.state_dict() for i, opt in enumerate(optimizer)}
        else:
            state = {"optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(path, "optimizer.pt"))

    def load_model(self, model, path: str) -> None:
        """Load model weights (stub)."""
        state_dict = torch.load(os.path.join(path, "model.pt"), weights_only=True)
        model.load_state_dict(state_dict)

    def load_optimizer(self, model_ref, optimizer, path: str) -> None:
        """Load optimizer state (stub)."""
        state = torch.load(os.path.join(path, "optimizer.pt"), weights_only=True)
        if isinstance(optimizer, list):
            for i, opt in enumerate(optimizer):
                key = f"opt_{i}"
                if key in state:
                    opt.load_state_dict(state[key])
        else:
            optimizer.load_state_dict(state["optimizer"])

    def save_on_dp_ranks(self, obj, name: str, path: str) -> None:
        """Save per-DP-rank state (stub)."""
        rank = int(os.environ.get("RANK", 0))
        torch.save(obj.state_dict(), os.path.join(path, f"{name}_dp_rank_{rank}.pt"))

    def load_on_dp_ranks(self, obj, name: str, path: str) -> None:
        """Load per-DP-rank state (stub)."""
        rank = int(os.environ.get("RANK", 0))
        state = torch.load(os.path.join(path, f"{name}_dp_rank_{rank}.pt"), weights_only=True)
        obj.load_state_dict(state)

    def async_wait(self) -> None:
        """Wait for any in-flight async checkpoint (stub)."""
        pass

    def maybe_wait_for_staging(self) -> None:
        """Wait for staging to complete (stub)."""
        pass

    def close(self) -> None:
        """Cleanup resources (stub)."""
        pass
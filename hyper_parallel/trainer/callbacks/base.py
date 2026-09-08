# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from typing import TYPE_CHECKING, Any, Dict, List

from hyper_parallel.trainer.state import TrainerState


if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer


class Callback:
    def __init__(self, trainer: "BaseTrainer") -> None:
        """Bind the callback to its owning trainer."""
        self.trainer = trainer
        self.mesh = trainer.mesh

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Hook invoked at the start of each training step."""

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs: Any
    ) -> None:
        """Hook invoked at the end of each training step."""

    def on_micro_step_begin(self, state: TrainerState, micro_batch: Dict[str, Any], **kwargs: Any) -> None:
        """Hook invoked at the start of each gradient-accumulation micro step."""

    def on_micro_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the end of each gradient-accumulation micro step."""

    def on_epoch_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the start of each epoch."""

    def on_epoch_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the end of each epoch."""

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the start of training."""

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the end of training."""

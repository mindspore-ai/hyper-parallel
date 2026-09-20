# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
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

"""Base callback hooks for Trainer lifecycle events."""

import weakref
from typing import TYPE_CHECKING, Any, Dict, List

from hyper_parallel.trainer.state import TrainerState


if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer


class Callback:
    """Base callback bound to a Trainer through a weak reference."""

    def __init__(self, trainer: "BaseTrainer") -> None:
        """Bind the callback to its owning trainer."""
        self.trainer = weakref.proxy(trainer)
        self.mesh = trainer.mesh

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Hook invoked at the start of each training step.

        Args:
            state: Current Trainer state.
            micro_batches: Micro batches participating in this step.
            **kwargs: Additional callback-specific values.
        """

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs: Any
    ) -> None:
        """Hook invoked at the end of each training step.

        Args:
            state: Current Trainer state.
            loss: Aggregate loss for this step.
            loss_dict: Named loss values for this step.
            grad_norm: Gradient norm for this step.
            **kwargs: Additional callback-specific values.
        """

    def on_micro_step_begin(self, state: TrainerState, micro_batch: Dict[str, Any], **kwargs: Any) -> None:
        """Hook invoked at the start of each gradient-accumulation micro step.

        Args:
            state: Current Trainer state.
            micro_batch: Input batch for this micro step.
            **kwargs: Additional callback-specific values.
        """

    def on_micro_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the end of each gradient-accumulation micro step.

        Args:
            state: Current Trainer state.
            **kwargs: Additional callback-specific values.
        """

    def on_epoch_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the start of each epoch.

        Args:
            state: Current Trainer state.
            **kwargs: Additional callback-specific values.
        """

    def on_epoch_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the end of each epoch.

        Args:
            state: Current Trainer state.
            **kwargs: Additional callback-specific values.
        """

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the start of training.

        Args:
            state: Current Trainer state.
            **kwargs: Additional callback-specific values.
        """

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Hook invoked at the end of training.

        Args:
            state: Current Trainer state.
            **kwargs: Additional callback-specific values.
        """

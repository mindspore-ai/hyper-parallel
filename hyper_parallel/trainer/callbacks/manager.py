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
"""Callback manager for trainer event dispatch."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

from hyper_parallel.trainer.callbacks.base import (
    BaseCallback,
    CallbackHookNames,
    TrainerCallbackContext,
    TrainerControl,
)

if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer


class CallbackManager:
    """Register callbacks and dispatch trainer events in priority order."""

    def __init__(
            self,
            trainer: Optional["BaseTrainer"] = None,
            callbacks: Optional[Iterable[BaseCallback]] = None,
    ) -> None:
        """Initialize with optional trainer and callbacks."""
        self.trainer = trainer
        self.control = TrainerControl()
        self._callbacks: list[BaseCallback] = []
        for callback in callbacks or ():
            self.register(callback)

    @property
    def callbacks(self) -> tuple[BaseCallback, ...]:
        """Registered callbacks in dispatch order."""
        return tuple(self._callbacks)

    def set_trainer(self, trainer: "BaseTrainer") -> None:
        """Attach the manager to a trainer."""
        self.trainer = trainer

    def register(self, callback: BaseCallback) -> None:
        """Register one callback."""
        self._callbacks.append(callback)
        self._callbacks.sort(key=lambda item: item.priority)

    def unregister(self, callback: BaseCallback) -> None:
        """Remove a previously registered callback."""
        self._callbacks.remove(callback)

    def has_listeners(self, event: CallbackHookNames) -> bool:
        """Return whether any registered callback overrides event."""
        hook_name = event.value
        base_hook = getattr(BaseCallback, hook_name)
        return any(
            getattr(type(callback), hook_name, None) is not base_hook
            for callback in self._callbacks
        )

    def dispatch(
            self,
            event: CallbackHookNames,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> TrainerControl:
        """Dispatch event to callbacks and aggregate control signals."""
        if self.trainer is None:
            raise RuntimeError("CallbackManager is not attached to a trainer.")
        if not self.has_listeners(event):
            return self.control

        hook_name = event.value
        if context.control is None:
            context.control = self.control

        base_hook = getattr(BaseCallback, hook_name)
        for callback in self._callbacks:
            if getattr(type(callback), hook_name, None) is base_hook:
                continue
            callback_hook = getattr(callback, hook_name)
            result = callback_hook(context, **payload)
            self.control.merge(result)
        return self.control

    def aggregate_control(self) -> TrainerControl:
        """Return the currently aggregated control object."""
        return self.control

    def reset_step_control(self) -> None:
        """Reset per-step control flags before a new step."""
        self.control.reset_step_flags()

    def reset_epoch_control(self) -> None:
        """Reset per-epoch control flags before a new epoch."""
        self.control.reset_epoch_flags()

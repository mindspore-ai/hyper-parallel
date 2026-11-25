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
"""Adapt Base, Text and VLM Trainer lifecycles to the recovery session."""

from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
from typing import Any, Callable

from hyper_parallel.high_availability.recovery_session import RecoverySession
from hyper_parallel.high_availability.recovery_state import TrainState
from hyper_parallel.high_availability.validation import validate_recovery_config
from hyper_parallel.trainer.runtime.memory_profiler import memory_profiler


def recovery_constructor(initialize: Callable) -> Callable:
    """Bootstrap the Processor before the Trainer constructs meshes or models.

    Args:
        initialize: Original Trainer constructor accepting its training configuration.

    Returns:
        Constructor wrapper that registers groups during native initialization.
    """
    @wraps(initialize)
    def _construct(trainer: Any, config: Any) -> None:
        availability = vars(config).get("high_availability")
        recovery_config = availability.worker_recovery if availability is not None else None
        if recovery_config is None or not recovery_config.enable:
            initialize(trainer, config)
            vars(trainer).get("base", trainer).recovery = None
            return
        validate_recovery_config(config)
        session = RecoverySession(recovery_config)
        try:
            session.initialize()
            with session.groups:
                initialize(trainer, config)
            base = vars(trainer).get("base", trainer)
            base.recovery = session
        except BaseException:
            session.close()
            raise
    return _construct


def recovery_train_loop(train: Callable) -> Callable:
    """Run the original Trainer entry point inside the recovery exception loop.

    Args:
        train: Original Trainer train method, including its native epoch/step loop.

    Returns:
        Entry wrapper that preserves the original method's result.
    """
    @wraps(train)
    def _run(trainer: Any) -> Any:
        base = vars(trainer).get("base", trainer)
        session = vars(base).get("recovery")
        if session is None:
            return train(trainer)
        try:
            return _train_recoverable(trainer, base, session, train)
        finally:
            session.close()
    return _run


def recovery_train_step(train_step: Callable) -> Callable:
    """Honor recovery commands before invoking the native Trainer step.

    Args:
        train_step: Original Trainer step implementation.

    Returns:
        Step wrapper preserving the original return value and counter updates.
    """
    @wraps(train_step)
    def _run_step(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        base = vars(trainer).get("base", trainer)
        session = vars(base).get("recovery")
        if session is not None:
            session.pause()
        return train_step(trainer, *args, **kwargs)
    return _run_step


def recovery_epoch_initialized(initialize: Callable) -> Callable:
    """Finalize pending RNG after a native epoch initialization stage returns.

    Args:
        initialize: Native stage completing callbacks and iterator construction.

    Returns:
        Wrapper preserving the stage result and applying pending RNG once.
    """
    @wraps(initialize)
    def _initialize(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        result = initialize(trainer, *args, **kwargs)
        base = vars(trainer).get("base", trainer)
        session = vars(base).get("recovery")
        if session is not None:
            session.train_state.restore_pending_rng()
        return result
    return _initialize


def recovery_train_initialized(initialize: Callable) -> Callable:
    """Align initial progress after the native train-begin callbacks return.

    Args:
        initialize: Native train-begin dispatcher, including initial checkpoint loading.

    Returns:
        Wrapper preserving the callback result and synchronizing initial progress.
    """
    @wraps(initialize)
    def _initialize(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        result = initialize(trainer, *args, **kwargs)
        base = vars(trainer).get("base", trainer)
        session = vars(base).get("recovery")
        if session is not None:
            session.initialize_training_progress(base.state.global_step)
        return result
    return _initialize


def recovery_optimizer_update(step: Callable) -> Callable:
    """Keep both Trainer step conventions inside one committed transaction.

    Args:
        step: Original optimizer and scheduler update method.

    Returns:
        Update wrapper that captures the committed step without changing native counters.
    """
    @wraps(step)
    def _step(base: Any) -> Any:
        session = vars(base).get("recovery")
        if session is not None:
            committing_step = session.step + 1
            transaction = session.update(committing_step)
            vars(base)["_recovery_committing_step"] = committing_step
        else:
            transaction = nullcontext()
        try:
            with transaction:
                return step(base)
        finally:
            if session is not None:
                vars(base).pop("_recovery_committing_step", None)
    return _step


class _TrainerProgress:
    """Expose the authoritative committed counters without invoking DCP state_dict."""

    def __init__(self, base: Any) -> None:
        """Bind the Trainer whose progress participates in the transaction."""
        self.base = base

    def state_dict(self) -> dict[str, int]:
        """Capture committed counters without collective checkpoint export."""
        state = asdict(self.base.state)
        committing_step = vars(self.base).get("_recovery_committing_step")
        if committing_step is not None:
            state["global_step"] = committing_step
        return state

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore progress before recreating the training iterator."""
        self.base.state.global_step = state["global_step"]
        self.base.state.epoch = state["epoch"]


def _build_train_state(base: Any, device: Any) -> TrainState:
    """Build the recovery state from the Trainer's loader and state providers."""
    replica_providers = {"progress": _TrainerProgress(base)}
    schedulers = base.lr_scheduler if isinstance(base.lr_scheduler, list) else [base.lr_scheduler]
    replica_providers.update({f"scheduler_{index}": scheduler for index, scheduler in enumerate(schedulers)
                              if scheduler is not None})
    profiler_callback = getattr(base, "profiler_callback", None)
    if profiler_callback is not None and profiler_callback.config.enable:
        replica_providers["profiler"] = profiler_callback
    memory_config = getattr(getattr(base, "config", None), "memory", None)
    if memory_config is not None and (memory_config.enable or memory_config.mem_info):
        replica_providers["memory_profiler"] = memory_profiler
    source_loader = getattr(base.train_dataloader, "source_dataloader", base.train_dataloader)
    data_generator = getattr(source_loader, "generator", None)
    generators = {"data_loader": data_generator} if data_generator is not None else {}
    return TrainState(base.train_dataloader, replica_providers, device, generators)


def _train_recoverable(trainer: Any, base: Any, session: RecoverySession, train: Callable) -> Any:
    """Bind state once and reenter the original Trainer implementation."""
    state = _build_train_state(base, session.npu)
    checkpoint_callback = getattr(base, "checkpoint_callback", None)
    if checkpoint_callback is not None:
        session.register_cleanup(checkpoint_callback.cleanup_state)
    session.attach(base.model, base.optimizer, state, base.state.global_step)

    def _body() -> Any:
        return train(trainer)

    return session.run(_body)

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
"""Opt-in recovery around the native Base, Text and VLM Trainer entry points."""

from contextlib import nullcontext
from dataclasses import asdict
from functools import partial, wraps
from typing import Any, Callable

import torch
import torch.distributed as dist

from hyper_parallel.core.fault_tolerance.session import RecoverySession
from hyper_parallel.core.fault_tolerance.state import TrainState
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback

def validate_recovery_config(config: Any) -> None:
    """Reject combinations without an audited state and reboot bootstrap path."""
    accelerator = config.accelerator
    if accelerator.cp_size != 1 or accelerator.pp_size != 1:
        raise ValueError("CP/PP schedule state requires a dedicated recovery adapter")
    if config.training.backend not in ("hccl", "nccl") or config.training.init_device != "meta":
        raise ValueError("worker recovery requires HCCL and meta-device model construction")
    if (config.compile.enabled or config.fsdp_config.enable_offload or config.optimizer.swap.enabled
            or config.activation_swap != "none" or config.activation_checkpoint.mode not in (None, "off")):
        raise ValueError("worker recovery requires eager training without offload, swap or activation checkpointing")
    if not config.fsdp_config.reshard_after_backward:
        raise ValueError("worker recovery requires reshard_after_backward when parameter sharding is enabled")
    if config.dataloader is None or config.dataloader.use_background_prefetcher:
        raise ValueError("worker recovery requires a stateful loader without background prefetch")
    if config.checkpoint.is_async or config.checkpoint.is_peft:
        raise ValueError("worker recovery requires synchronous full checkpoints")
    if config.model_integration.mode != "off" or config.peft is not None or config.magi is not None:
        raise ValueError("worker recovery does not yet cover integration probes, PEFT or external model state")
    if config.profiler.enable or config.memory.enable or config.stress_detect.op_stress_detect.detection_interval > 0:
        raise ValueError("worker recovery does not support active profilers or stress detection")
    if config.training.eval_iters or config.training.eval_steps or config.training.eval_epochs:
        raise ValueError("worker recovery does not yet cover evaluation data state")
    if config.training.low_precision.enabled:
        raise ValueError("quantization and low-precision auxiliary state are not covered by worker recovery")


def recovery_constructor(initialize: Callable) -> Callable:
    """Bootstrap the Processor before the Trainer constructs meshes or models.

    Args:
        initialize: Original Trainer constructor accepting its training configuration.

    Returns:
        Constructor wrapper that registers groups during native initialization.
    """
    @wraps(initialize)
    def _construct(trainer: Any, config: Any) -> None:
        recovery_config = vars(config).get("worker_recovery")
        if recovery_config is None or not recovery_config.enable_worker_reboot:
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
            if getattr(base.train_dataloader, "num_workers", 0) != 0:
                raise ValueError("worker recovery currently requires DataLoader num_workers=0")
        except BaseException:
            session.close()
            raise
    return _construct


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


def recovery_data_iterator(base: Any, factory: Callable[[], Any] = None) -> Any:
    """Build the native iterator before epoch initialization finalizes restored RNG.

    Args:
        base: Trainer owning the DataLoader and optional recovery session.
        factory: Original iterator constructor, including any native wrapper.

    Returns:
        The original iterator type, or None when no DataLoader is configured.
    """
    if factory is None:
        if base.train_dataloader is None:
            return None
        factory = partial(iter, base.train_dataloader)
    session = vars(base).get("recovery")
    return factory() if session is None else session.make_iterator(factory)


def recovery_epoch_initialized(base: Any) -> None:
    """Run one pending restore after native epoch callbacks and iterator construction.

    Args:
        base: Trainer owning the initialized DataLoader and optional recovery session.
    """
    session = vars(base).get("recovery")
    if session is not None and session.train_state.rng_restore_pending:
        session.finish_restore()


def recovery_train_initialized(base: Any) -> None:
    """Publish initial recovery state after native start callbacks complete.

    Args:
        base: Trainer whose callbacks have loaded any normal initial checkpoint.
    """
    session = vars(base).get("recovery")
    if session is not None:
        session.publish_initial_state(base.state.global_step)


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
            base._recovery_committing_step = committing_step
        else:
            transaction = nullcontext()
        try:
            with transaction:
                return step(base)
        finally:
            if session is not None:
                vars(base).pop("_recovery_committing_step", None)
    return _step


def _synchronize_checkpoint_saved_step(base: Any, session: RecoverySession) -> None:
    """Align the callback's last saved checkpoint step with rank zero."""
    callback = next((callback for callback in vars(base).get("_callbacks", [])
                     if isinstance(callback, CheckpointerCallback)), None)
    if callback is None:
        return
    # Checkpoint saves happen after the committed snapshot. A replacement may
    # otherwise enter DCP collectives for a step the surviving ranks already saved.
    saved_step = torch.tensor([callback._last_saved_step],  # pylint: disable=protected-access
                              dtype=torch.int64, device=torch.device("npu", session.local_rank))
    dist.broadcast(saved_step, src=0)
    callback._last_saved_step = saved_step.item()  # pylint: disable=protected-access


def _train_recoverable(trainer: Any, base: Any, session: RecoverySession, train: Callable) -> Any:
    """Bind state once and reenter the original Trainer implementation."""
    replica_providers = {"progress": _TrainerProgress(base)}
    schedulers = base.lr_scheduler if isinstance(base.lr_scheduler, list) else [base.lr_scheduler]
    replica_providers.update({f"scheduler_{index}": scheduler for index, scheduler in enumerate(schedulers)
                              if scheduler is not None})
    data_generator = getattr(base.train_dataloader, "generator", None)
    generators = {"data_loader": data_generator} if data_generator is not None else {}
    state = TrainState(base.train_dataloader, replica_providers, session.npu, generators)
    session.attach(base.model, base.optimizer, state, base.state.global_step)

    def _body(active: RecoverySession) -> Any:
        _synchronize_checkpoint_saved_step(base, active)
        return train(trainer)

    return session.run(_body)

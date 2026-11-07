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
"""Opt-in recovery lifecycle shared by BaseTrainer and TextTrainer."""

from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
import logging
from typing import Any, Callable

import torch
import torch.distributed as dist

from hyper_parallel.core.fault_tolerance.session import RecoverySession
from hyper_parallel.core.fault_tolerance.state import TrainState
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.runtime.memory_profiler import memory_profiler

logger = logging.getLogger(__name__)


def record_step_metrics(base: Any, loss: float, loss_dict: dict[str, Any], grad_norm: Any) -> None:
    """Include existing step metrics in the same owner-addressed commit as model state."""
    session = vars(base).get("recovery")
    if session is not None:
        base._recovery_step_metrics = {"step": session.step + 1, "loss": loss,
                                       "loss_dict": dict(loss_dict), "grad_norm": float(grad_norm)}

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
    """Bootstrap the Processor before either Trainer constructs meshes or models."""
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
        return asdict(self.base.state)

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore progress before recreating the training iterator."""
        self.base.state.global_step = state["global_step"]
        self.base.state.epoch = state["epoch"]


class _TrainerLogState:
    """Keep committed metrics locally and identify donor-sourced repair logs."""

    def __init__(self, base: Any) -> None:
        """Bind the producer of the next step's logging payload."""
        self.base = base

    def state_dict(self) -> dict[str, Any]:
        """Capture metrics already computed by the training step."""
        metrics = getattr(self.base, "_recovery_step_metrics", {})
        if not metrics:
            return {}
        return {**metrics, "source_rank": getattr(self.base.recovery, "rank", None)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Expose the restored step for logging without replaying optimizer updates."""
        self.base._recovery_step_metrics = state
        self.base._restored_step_metrics = state


def recovery_train_loop(train: Callable) -> Callable:
    """Replace only the loop lifecycle when the user opts into worker recovery."""
    @wraps(train)
    def _run(trainer: Any) -> Any:
        base = vars(trainer).get("base", trainer)
        session = vars(base).get("recovery")
        if session is None:
            return train(trainer)
        try:
            return _train_recoverable(trainer, base, session)
        except BaseException:
            base.abort_profilers()
            raise
        finally:
            session.close()
    return _run


def recovery_optimizer_update(step: Callable) -> Callable:
    """Keep both Trainer step conventions inside one committed transaction."""
    @wraps(step)
    def _step(base: Any) -> Any:
        session = vars(base).get("recovery")
        transaction = session.update(session.step + 1) if session is not None else nullcontext()
        with transaction:
            result = step(base)
            if session is not None:
                base.state.global_step = session.step + 1
        return result
    return _step


def _synchronize_checkpoint_ledger(base: Any, session: RecoverySession) -> None:
    """Align duplicate-save suppression with the surviving checkpoint coordinator."""
    callbacks = [callback for callback in vars(base).get("_callbacks", [])
                 if isinstance(callback, CheckpointerCallback)]
    if not callbacks:
        return
    # Snapshots precede on_step_end saves. Restoring a snapshot's ledger could
    # forget a completed save and make only the replacement enter DCP collectives.
    ledger = torch.tensor([callback._last_saved_step for callback in callbacks],  # pylint: disable=protected-access
                          dtype=torch.int64, device=torch.device("npu", session.local_rank))
    dist.broadcast(ledger, src=0)
    for callback, saved_step in zip(callbacks, ledger.cpu().tolist()):
        callback._last_saved_step = saved_step  # pylint: disable=protected-access


def _train_recoverable(trainer: Any, base: Any, session: RecoverySession) -> None:
    """Load once, reenter only the training body, and tear down once."""
    if not session.reboot:
        trainer.on_train_begin()
    providers = {"progress": _TrainerProgress(base), "logs": _TrainerLogState(base)}
    schedulers = base.lr_scheduler if isinstance(base.lr_scheduler, list) else [base.lr_scheduler]
    providers.update({f"scheduler_{index}": scheduler for index, scheduler in enumerate(schedulers)
                      if scheduler is not None})
    data_generator = getattr(base.train_dataloader, "generator", None)
    generators = {"data_loader": data_generator} if data_generator is not None else {}
    state = TrainState(base.train_dataloader, providers, session.npu, generators,
                       replica_providers=tuple(providers))
    session.attach(base.model, base.optimizer, state, base.state.global_step)
    initialize_reboot_callbacks = session.reboot

    def _body(active: RecoverySession) -> int:
        nonlocal initialize_reboot_callbacks
        _synchronize_checkpoint_ledger(base, active)
        restored_metrics = vars(base).pop("_restored_step_metrics", None)
        if restored_metrics:
            logger.info("Restored committed training metrics: %s", restored_metrics)
        if initialize_reboot_callbacks:
            # Built-in non-checkpoint on_train_begin hooks are local. The
            # replacement must not reload the initial disk checkpoint.
            for callback in base._callbacks:  # pylint: disable=protected-access
                if not isinstance(callback, CheckpointerCallback):
                    callback.on_train_begin(base.state)
            initialize_reboot_callbacks = False
        while base.state.global_step < base.train_iters and base.state.epoch < base.train_epochs:
            epoch = base.state.epoch
            start_step = base.state.global_step - epoch * base.train_steps
            train_steps = min(base.train_steps, base.train_iters - epoch * base.train_steps)
            trainer.on_epoch_begin()
            # The consumed cursor was restored by rollback before SDK reentry.
            data_iterator = active.make_iterator()
            base.data_iterator = data_iterator
            for _ in range(start_step, train_steps):
                active.pause()
                try:
                    trainer.train_step(data_iterator)
                except StopIteration:
                    break
            trainer.on_epoch_end()
            base.state.epoch = epoch + 1
            if hasattr(base.train_dataloader, "set_epoch"):
                base.train_dataloader.set_epoch(base.state.epoch)
        return base.state.global_step

    session.run(_body)
    memory_profiler.stop()
    trainer.on_train_end()
    base.destroy_distributed()

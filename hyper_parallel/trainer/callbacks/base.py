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
"""Callback base types and built-in trainer callbacks."""
from __future__ import annotations

from abc import ABC
import gc
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

from hyper_parallel import get_platform
from hyper_parallel.trainer.callbacks.common import (
    _as_float,
    _format_console_value,
    _should_trigger,
    _sub_cfg,
)

if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer, TrainerState

platform = get_platform()
logger = logging.getLogger(__name__)


class CallbackHookNames(str, Enum):
    """Supported trainer callback hook names."""

    ON_INIT_END = "on_init_end"
    ON_TRAIN_BEGIN = "on_train_begin"
    ON_RESUME = "on_resume"
    ON_TRAIN_END = "on_train_end"
    ON_EPOCH_BEGIN = "on_epoch_begin"
    ON_EPOCH_END = "on_epoch_end"
    ON_STEP_BEGIN = "on_step_begin"
    ON_BEFORE_OPTIMIZER_STEP = "on_before_optimizer_step"
    ON_STEP_END = "on_step_end"
    ON_LOG = "on_log"
    ON_EVALUATE_BEGIN = "on_evaluate_begin"
    ON_EVALUATE_END = "on_evaluate_end"
    ON_PREDICT = "on_predict"
    ON_SAVE = "on_save"
    ON_EXCEPTION = "on_exception"
    ON_MICRO_BATCH_END = "on_micro_batch_end"
    ON_FORWARD_END = "on_forward_end"
    ON_AFTER_GRAD_SYNC = "on_after_grad_sync"
    ON_PREDICTION_STEP = "on_prediction_step"
    ON_PROFILER_STEP = "on_profiler_step"
    ON_MEMORY_SNAPSHOT = "on_memory_snapshot"
    ON_COMMUNICATION_STEP = "on_communication_step"


@dataclass
class TrainerControl:
    """Control signals that callbacks may request."""

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_skip_step: bool = False
    should_skip_optimizer_step: bool = False
    should_log: bool = False
    should_evaluate: bool = False
    should_save: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    _CONTROL_FIELDS: ClassVar[tuple[str, ...]] = (
        "should_training_stop",
        "should_epoch_stop",
        "should_skip_step",
        "should_skip_optimizer_step",
        "should_log",
        "should_evaluate",
        "should_save",
    )

    def merge(self, other: Optional["TrainerControl"]) -> "TrainerControl":
        """Merge another control object into this one."""
        if other is None:
            return self
        for name in self._CONTROL_FIELDS:
            setattr(self, name, bool(getattr(self, name)) or bool(getattr(other, name)))
        self.metadata.update(other.metadata)
        return self

    def has_signal(self) -> bool:
        """Return whether any signal or metadata has been requested."""
        return any(bool(getattr(self, name)) for name in self._CONTROL_FIELDS) or bool(self.metadata)

    def reset(self) -> None:
        """Reset all control signals and metadata."""
        for name in self._CONTROL_FIELDS:
            setattr(self, name, False)
        self.metadata.clear()

    def reset_step_flags(self) -> None:
        """Reset per-step signals before a new training step starts."""
        self.should_skip_step = False
        self.should_skip_optimizer_step = False
        self.should_log = False
        self.should_evaluate = False
        self.should_save = False
        self.metadata.clear()

    def reset_epoch_flags(self) -> None:
        """Reset per-epoch signals before a new epoch starts."""
        self.should_epoch_stop = False
        self.reset_step_flags()

    def copy(self) -> "TrainerControl":
        """Return a detached copy."""
        return type(self).from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Serialize control signals into lightweight Python values."""
        values = {name: getattr(self, name) for name in self._CONTROL_FIELDS}
        values["metadata"] = dict(self.metadata)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainerControl":
        """Build a control object from serialized values."""
        data = {name: bool(values.get(name, False)) for name in cls._CONTROL_FIELDS}
        data["metadata"] = dict(values.get("metadata") or {})
        return cls(**data)


@dataclass
class TrainerCallbackContext:
    """Lightweight trainer view passed to callbacks for one dispatch."""

    state: Optional["TrainerState"] = None
    control: Optional[TrainerControl] = None
    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_world_rank0: bool = True
    is_local_rank0: bool = True
    logs: dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        """Backward-compatible alias for global_rank."""
        return self.global_rank

    @classmethod
    def from_trainer(
            cls,
            trainer: "BaseTrainer",
            control: Optional[TrainerControl] = None,
    ) -> "TrainerCallbackContext":
        """Create callback context from stable trainer attributes."""
        return cls(
            state=trainer.state,
            control=control,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            world_size=trainer.world_size,
            is_world_rank0=trainer.is_world_rank0,
            is_local_rank0=trainer.is_local_rank0,
            logs=dict(trainer._callback_logs),
        )


class BaseCallback(ABC):
    """Base class for trainer callbacks."""
    # pylint: disable=unused-argument

    priority: int = 0

    def on_init_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after the trainer has finished all build steps."""
        return None

    def on_train_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called at the beginning of training."""
        return None

    def on_resume(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after state has been restored from a checkpoint."""
        return None

    def on_train_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after the training loop exits."""
        return None

    def on_epoch_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called at the start of each epoch."""
        return None

    def on_epoch_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called at the end of each epoch."""
        return None

    def on_step_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called before one training step begins."""
        return None

    def on_before_optimizer_step(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after gradients are ready and before optimizer.step."""
        return None

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after one training step finishes."""
        return None

    def on_log(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called when a structured log record is ready."""
        return None

    def on_evaluate_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called when evaluation begins."""
        return None

    def on_evaluate_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called when evaluation metrics are ready."""
        return None

    def on_predict(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called when prediction results are ready."""
        return None

    def on_save(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after a checkpoint save attempt completes."""
        return None

    def on_exception(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called before a trainer exception is re-raised."""
        return None

    def on_micro_batch_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after one micro-batch forward/backward pass."""
        return None

    def on_forward_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called immediately after model forward returns outputs."""
        return None

    def on_after_grad_sync(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after gradient synchronization has completed."""
        return None

    def on_prediction_step(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called after one eval or predict batch."""
        return None

    def on_profiler_step(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called at a profiler sampling point."""
        return None

    def on_memory_snapshot(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called when a memory snapshot point is reached."""
        return None

    def on_communication_step(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Called around communication profiling or debug points."""
        return None


class LoggingCallback(BaseCallback):
    """Log training metrics: loss, grad_norm, lr, and throughput."""

    def __init__(
            self,
            log_steps: int = 10,
            report_global_loss: bool = False,
            report_throughput: bool = True,
            model_flops_per_token: Optional[int] = None,
            peak_tflops: Optional[float] = None,
    ) -> None:
        """Initialize LoggingCallback with metric-reporting configuration."""
        self.log_steps = int(log_steps)
        self.report_global_loss = bool(report_global_loss)
        self.report_throughput = bool(report_throughput)
        self.model_flops_per_token = model_flops_per_token
        self.peak_tflops = peak_tflops
        self._step_start_time = 0.0

    def on_step_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Record the start time of the current training step."""
        self._step_start_time = time.time()

    def _append_throughput_metrics(
            self,
            console_metrics: dict,
            payload: Mapping,
            payload_metrics: Mapping,
            elapsed: float,
            world_size: int,
    ) -> Optional[float]:
        """Add tokens/sec, TFLOPS and MFU to console_metrics; return tokens/sec."""
        if not self.report_throughput:
            return None
        tokens = _as_float(payload.get("tokens", payload_metrics.get("tokens")))
        if tokens is None:
            return None
        tokens_per_sec = tokens / elapsed
        console_metrics["tokens_per_sec"] = f"{tokens_per_sec:,.0f}"
        if self.model_flops_per_token and self.peak_tflops:
            observed_tflops = tokens_per_sec * self.model_flops_per_token / 1e12
            mfu = observed_tflops / (self.peak_tflops * max(world_size, 1))
            console_metrics["tflops"] = f"{observed_tflops:.1f}"
            console_metrics["mfu"] = f"{mfu * 100:.1f}%"
        return tokens_per_sec

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Log metrics at the configured interval and emit a should_log control signal."""
        state = context.state
        if state is None or not _should_trigger(state.global_step, self.log_steps):
            return None

        payload_metrics = payload.get("metrics")
        payload_metrics = payload_metrics if isinstance(payload_metrics, Mapping) else {}
        loss = payload.get("loss", payload_metrics.get("loss"))
        grad_norm = payload.get("grad_norm", payload_metrics.get("grad_norm"))
        lr = _as_float(payload.get("lr", getattr(state, "lr", None))) or 0.0
        elapsed = max(time.time() - self._step_start_time, 1e-9)

        console_metrics = {
            "step": state.global_step,
            "loss": f"{loss:.8f}" if loss is not None else "N/A",
            "grad_norm": f"{grad_norm:.8f}" if grad_norm is not None else "N/A",
            "lr": f"{lr:.2e}",
            "step_time": f"{elapsed:.2f}s",
        }

        tokens_per_sec = self._append_throughput_metrics(
            console_metrics, payload, payload_metrics, elapsed, context.world_size
        )

        extra_metrics = {
            key: value
            for key, value in payload_metrics.items()
            if key not in {"loss", "grad_norm", "tokens", "samples"}
        }
        console_metrics.update(
            {key: _format_console_value(value) for key, value in extra_metrics.items()}
        )
        logger.info_rank0(" | ".join(f"{key}={value}" for key, value in console_metrics.items()))

        record = {
            "step": state.global_step,
            "loss": loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "step_time": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }
        record.update(extra_metrics)
        add_log = getattr(state, "add_log", None)
        if callable(add_log):
            add_log(record)
        else:
            state.log_history.append(record)
        context.logs.update(record)
        return TrainerControl(should_log=True, metadata={"log_record": record})


class EvalCallback(BaseCallback):
    """Evaluation callback stub."""

    def __init__(self, eval_steps: int = 0, eval_dataset: Optional[str] = None) -> None:
        """Initialize EvalCallback with evaluation frequency and dataset name."""
        self.eval_steps = int(eval_steps or 0)
        self.eval_dataset = eval_dataset

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Request evaluation at the configured step interval."""
        state = context.state
        if state is None or not _should_trigger(state.global_step, self.eval_steps):
            return None
        if context.is_world_rank0:
            logger.warning("EvalCallback: evaluation not implemented (step=%d)", state.global_step)
        return TrainerControl(should_evaluate=True, metadata={"eval_name": self.eval_dataset})


class ProfilerCallback(BaseCallback):
    """Training profiler callback stub."""

    def __init__(self, enabled: bool = False) -> None:
        """Initialize ProfilerCallback with enabled flag."""
        self.enabled = bool(enabled)
        self._warned = False

    def on_init_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Warn once if profiler is enabled but not implemented."""
        if self.enabled and context.is_world_rank0 and not self._warned:
            logger.warning(
                "ProfilerCallback: enabled=True but the implementation is a stub; "
                "torch profiler is not started."
            )
            self._warned = True


class WandbCallback(BaseCallback):
    """Weights & Biases logging callback stub."""

    def __init__(self, enabled: bool = False) -> None:
        """Initialize WandbCallback with enabled flag."""
        self.enabled = bool(enabled)
        self._warned = False

    def on_init_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Warn once if W&B is enabled but not implemented."""
        if self.enabled and context.is_world_rank0 and not self._warned:
            logger.warning(
                "WandbCallback: enabled=True but the implementation is a stub; "
                "nothing is sent to W&B."
            )
            self._warned = True


class ProgressCallback(BaseCallback):
    """tqdm progress bar callback."""

    def __init__(self) -> None:
        """Initialize ProgressCallback."""
        self._pbar = None

    def on_train_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Create the tqdm progress bar at the start of training."""
        state = context.state
        if state is None or not context.is_world_rank0:
            return

        try:
            from tqdm import tqdm  # pylint: disable=C0415  # optional dep
            self._pbar = tqdm(
                total=state.max_steps,
                initial=state.global_step,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
            )
        except ImportError:
            logger.warning("ProgressCallback: 'tqdm' not installed; progress bar disabled")

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Advance the progress bar by one step and update displayed metrics."""
        if self._pbar is None:
            return
        metrics = payload.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        loss = _as_float(payload.get("loss", metrics.get("loss")))
        grad_norm = _as_float(payload.get("grad_norm", metrics.get("grad_norm")))
        postfix = {}
        if loss is not None:
            postfix["loss"] = f"{loss:.4f}"
        if grad_norm is not None:
            postfix["gnorm"] = f"{grad_norm:.4f}"
        self._pbar.set_postfix(postfix)
        self._pbar.update(1)

    def on_train_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Close the progress bar at the end of training."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


class MoEMonitorCallback(BaseCallback):
    """Mixture-of-Experts load-balancing monitor."""

    priority = -10

    def __init__(self, enabled: bool = False, lr: float = 1e-3, num_recomputations: int = 1) -> None:
        """Initialize MoEMonitorCallback with load-balancing parameters."""
        self.enabled = bool(enabled)
        self.lr = lr
        self.num_recomputations = int(num_recomputations)
        self._impl = None

    @property
    def last_mean_aux_loss(self) -> Optional[float]:
        """Mean aux_loss across MoE layers from the last step."""
        if self._impl is not None:
            return self._impl.last_mean_aux_loss
        return None

    def on_init_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Initialize the core MoE monitor implementation if enabled."""
        if not self.enabled:
            return None
        model = payload.get("model")
        if model is None:
            return None
        from hyper_parallel.core.moe_utils import (  # pylint: disable=C0415
            MoEMonitorCallback as _CoreMoEMonitorCallback,
        )
        self._impl = _CoreMoEMonitorCallback(
            model=model,
            lr=self.lr,
            dp_group=payload.get("dp_group"),
            tp_group=payload.get("tp_group"),
            cp_group=payload.get("cp_group"),
            num_recomputations=self.num_recomputations,
        )
        return None

    def on_train_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Log that MoE monitoring is active at the start of training."""
        if self.enabled and context.is_world_rank0:
            logger.info("MoEMonitorCallback: MoE expert-load monitoring enabled")

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Delegate to core MoE monitor and inject aux_loss into metrics."""
        if self._impl is not None:
            self._impl.on_step_end()
            aux_loss = self.last_mean_aux_loss
            metrics = payload.get("metrics")
            if aux_loss is not None and isinstance(metrics, dict):
                metrics["aux_loss"] = aux_loss


class GradientHealthCallback(BaseCallback):
    """Detect NaN or Inf grad_norm."""

    def __init__(self, enabled: bool = False) -> None:
        """Initialize GradientHealthCallback with enabled flag."""
        self.enabled = bool(enabled)

    def on_before_optimizer_step(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Raise RuntimeError on rank 0 if grad_norm is NaN or Inf."""
        state = context.state
        grad_norm = _as_float(payload.get("grad_norm_after_clip", payload.get("grad_norm")))
        if state is None or not self.enabled or grad_norm is None:
            return None
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            logger.error(
                "GradientHealthCallback: grad_norm=%s at step %d (NaN/Inf).",
                grad_norm,
                state.global_step,
            )
            if context.is_world_rank0:
                raise RuntimeError(
                    f"Non-finite grad_norm={grad_norm} at step {state.global_step}. "
                    "Disable cfg.train.debug.check_nan_inf to skip this guard."
                )
        return None


class GCCallback(BaseCallback):
    """Explicit garbage-collection scheduler."""

    def __init__(self, gc_steps: int = 0) -> None:
        """Initialize GCCallback with the explicit GC interval in steps."""
        self.gc_steps = int(gc_steps or 0)
        if self.gc_steps > 0:
            gc.disable()
            logger.info("GCCallback: Python gc.collect every %d steps", self.gc_steps)

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Run gc.collect() at the configured step interval."""
        state = context.state
        if state is None or not _should_trigger(state.global_step, self.gc_steps):
            return
        gc.collect()


class TensorBoardCallback(BaseCallback):
    """TensorBoard scalar writer stub."""

    def __init__(self, enabled: bool = False) -> None:
        """Initialize TensorBoardCallback with enabled flag."""
        self.enabled = bool(enabled)
        self._warned = False

    def on_init_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Warn once if TensorBoard is enabled but not implemented."""
        if self.enabled and context.is_world_rank0 and not self._warned:
            logger.warning(
                "TensorBoardCallback: enabled=True but the implementation is a stub; "
                "nothing is written to TensorBoard."
            )
            self._warned = True


class MemoryMonitorCallback(BaseCallback):
    """Peak and current device memory monitor."""

    def __init__(
            self,
            enabled: bool = False,
            log_steps: int = 1,
            reset_peak_each_step: bool = False,
    ) -> None:
        """Initialize MemoryMonitorCallback with logging and peak-reset options."""
        self.enabled = bool(enabled)
        self.log_steps = int(log_steps or 1)
        self.reset_peak_each_step = bool(reset_peak_each_step)

    def on_step_begin(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Reset peak memory stats before the step when configured."""
        if not self.enabled or not self.reset_peak_each_step:
            return None
        reset_fn = getattr(self._device_handle(), "reset_peak_memory_stats", None)
        if callable(reset_fn):
            reset_fn()
        return None

    def on_step_end(
            self,
            context: TrainerCallbackContext,
            **payload: Any,
    ) -> Optional[TrainerControl]:
        """Record device memory metrics at the configured step interval."""
        state = context.state
        if state is None or not self.enabled or not _should_trigger(state.global_step, self.log_steps):
            return None
        metrics = self._memory_metrics()
        if not metrics:
            return None
        payload_metrics = payload.get("metrics")
        if isinstance(payload_metrics, dict):
            payload_metrics.update(metrics)
        context.logs.update(metrics)
        logger.info_rank0(
            "memory_allocated_gb=%s | memory_peak_allocated_gb=%s",
            _format_console_value(metrics.get("memory_allocated_gb")),
            _format_console_value(metrics.get("memory_peak_allocated_gb")),
        )
        return None

    @staticmethod
    def _device_handle() -> Any:
        device_type = platform.device_type()
        try:
            return platform.get_device_handle(device_type)
        except TypeError:
            return platform.get_device_handle()

    def _memory_metrics(self) -> dict[str, float]:
        handle = self._device_handle()
        allocated_fn = getattr(handle, "memory_allocated", None)
        peak_fn = getattr(handle, "max_memory_allocated", None)
        if not callable(allocated_fn) or not callable(peak_fn):
            return {}
        denom = 1024 ** 3
        return {
            "memory_allocated_gb": float(allocated_fn()) / denom,
            "memory_peak_allocated_gb": float(peak_fn()) / denom,
        }


def build_default_callbacks(args: Any) -> list[BaseCallback]:
    """Build the default callback set from trainer config."""
    log_cfg = _sub_cfg(args, "logging")
    eval_cfg = _sub_cfg(args, "eval")
    profile_cfg = _sub_cfg(args, "profile") or _sub_cfg(args, "profiler")
    wandb_cfg = _sub_cfg(args, "wandb")
    moe_cfg = _sub_cfg(args, "moe_monitor")
    debug_cfg = _sub_cfg(args, "debug")
    tb_cfg = _sub_cfg(args, "tensorboard")
    memory_cfg = _sub_cfg(args, "memory_monitor")

    return [
        LoggingCallback(
            log_steps=getattr(log_cfg, "log_steps", 10),
            report_global_loss=getattr(log_cfg, "report_global_loss", False),
            report_throughput=getattr(log_cfg, "report_throughput", True),
            model_flops_per_token=getattr(log_cfg, "model_flops_per_token", None),
            peak_tflops=getattr(log_cfg, "peak_tflops", None),
        ),
        EvalCallback(
            eval_steps=getattr(eval_cfg, "eval_steps", 0),
            eval_dataset=getattr(eval_cfg, "eval_dataset", None),
        ),
        ProfilerCallback(enabled=getattr(profile_cfg, "enabled", False)),
        WandbCallback(enabled=getattr(wandb_cfg, "enabled", False)),
        ProgressCallback(),
        MoEMonitorCallback(
            enabled=getattr(moe_cfg, "enabled", False),
            lr=getattr(moe_cfg, "lr", 1e-3),
            num_recomputations=getattr(moe_cfg, "num_recomputations", 1),
        ),
        GradientHealthCallback(enabled=getattr(debug_cfg, "check_nan_inf", False)),
        GCCallback(gc_steps=getattr(debug_cfg, "gc_steps", 0)),
        TensorBoardCallback(enabled=getattr(tb_cfg, "enabled", False)),
        MemoryMonitorCallback(
            enabled=getattr(memory_cfg, "enabled", False),
            log_steps=getattr(memory_cfg, "log_steps", 1),
            reset_peak_each_step=getattr(memory_cfg, "reset_peak_each_step", False),
        ),
    ]

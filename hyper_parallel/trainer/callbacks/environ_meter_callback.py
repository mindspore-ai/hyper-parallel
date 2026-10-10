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
"""Training and environment metric collection callback."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from hyper_parallel.trainer.runtime.distributed import get_world_size_safe
from hyper_parallel.trainer.runtime.distributed import all_reduce
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.trainer.runtime.device import get_device_type, get_torch_device
from hyper_parallel.trainer.runtime.flops import TransformerFlopsEstimator
from hyper_parallel.trainer.runtime.logging import create_logger

from .base import Callback, TrainerState


logger = create_logger(__name__)


class EnvironMeterCallback(Callback):
    """Collect structured training, throughput, and memory metrics.

    The callback is the single producer of ``trainer.step_train_metrics`` and
    ``trainer.step_env_metrics``. Presentation and remote logging callbacks
    consume those dictionaries without recalculating or reducing metrics.
    """

    def __init__(self, trainer: Any) -> None:
        """Initialize per-step and cumulative counters.

        Args:
            trainer: Trainer that owns the callback lifecycle.
        """
        super().__init__(trainer)
        self._step_start_time = 0.0
        self._local_step_tokens: Any = None
        self._local_step_samples = 0
        self._local_input_tokens = 0
        self._local_flops: Any = 0.0
        self._flops_valid: Any = True
        self._flops_observed = False
        self._flops_warning_shown = False
        target = getattr(getattr(trainer, "config", None), "flops_estimator", None)
        geometry = {"model_config": getattr(trainer, "model_config", None), "model": getattr(trainer, "model", None)}
        self._flops_estimator = (
            target.build(**geometry) if target is not None else TransformerFlopsEstimator(**geometry)
        )
        if not callable(self._flops_estimator):
            raise TypeError("flops_estimator must build a callable(batch, *, cp_size)")
        self._loss_metrics: dict[str, Any] = {}
        self._loss_metric_steps = 0
        self._consumed_tokens = 0
        self._consumed_samples = 0
        self.trainer.step_train_metrics = {}
        self.trainer.step_env_metrics = {}

    @staticmethod
    def _scalar(value: Any, name: str) -> float:
        """Convert a scalar or scalar tensor-like value to ``float``.

        Args:
            value: Scalar value to convert.
            name: Metric name used in validation errors.

        Returns:
            Converted floating-point value.

        Raises:
            ValueError: If the value cannot be converted to a scalar float.
        """
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric {name!r} must be scalar, but got {value!r}") from exc

    @staticmethod
    def _tensor_numel(value: Any) -> int | None:
        """Return ``value.numel()`` when it exposes a tensor-like interface."""
        numel = getattr(value, "numel", None)
        if not callable(numel):
            return None
        return int(numel())

    @classmethod
    def _batch_tokens(cls, batch: Mapping[str, Any]) -> Any:
        """Count text tokens without synchronizing a device scalar to the host."""
        token_count = batch.get("token_count")
        if token_count is not None:
            return token_count

        labels = batch.get("labels")
        if labels is not None and callable(getattr(labels, "sum", None)):
            return (labels != IGNORE_INDEX).sum()

        attention_mask = batch.get("attention_mask")
        attention_mask_shape = getattr(attention_mask, "shape", ())
        if (
            len(attention_mask_shape) <= 2
            and attention_mask is not None
            and callable(getattr(attention_mask, "sum", None))
        ):
            return attention_mask.sum()

        input_ids = batch.get("input_ids")
        input_numel = cls._tensor_numel(input_ids)
        if input_numel is not None:
            return input_numel
        return 0

    @staticmethod
    def _batch_samples(batch: Mapping[str, Any]) -> int:
        """Count logical samples in one micro-batch."""
        value = batch.get("input_ids")
        if value is None:
            value = batch.get("labels")
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            return 0
        if len(shape) == 1:
            return 1
        return int(shape[0])

    @staticmethod
    def _batch_mapping(value: Any) -> Mapping[str, Any] | None:
        """Return metric inputs for a mapping or prepared runtime batch."""
        if isinstance(value, Mapping):
            return value
        loss_count_inputs = getattr(value, "loss_count_inputs", None)
        if not callable(loss_count_inputs):
            return None
        metric_inputs = loss_count_inputs()
        return metric_inputs if isinstance(metric_inputs, Mapping) else None

    @classmethod
    def _micro_batches(cls, value: Any) -> list[Mapping[str, Any]]:
        """Normalize callback input into a list of mapping micro-batches."""
        if value is None:
            return []
        batch = cls._batch_mapping(value)
        if batch is not None:
            return [batch]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            batches = []
            for item in value:
                batch = cls._batch_mapping(item)
                if batch is not None:
                    batches.append(batch)
            return batches
        return []

    def _metric_group(self) -> Any:
        """Return the DP+CP process group used by loss normalization."""
        dp_cp_mesh = getattr(self.trainer.mesh, "dp_cp_mesh", None)
        if dp_cp_mesh is None:
            return None
        return dp_cp_mesh.get_group()

    def _reduce(self, value: Any, op: str) -> float:
        """Reduce one scalar metric, with a single-process no-op fallback."""
        if get_world_size_safe() <= 1:
            return float(value)
        group = self._metric_group()
        mesh = self.trainer.mesh
        if group is None and (getattr(mesh, "tp_size", 1) > 1 or getattr(mesh, "pp_size", 1) > 1):
            if getattr(mesh, "dp_size", 1) * getattr(mesh, "cp_size", 1) == 1:
                return float(value)
            raise ValueError("DP+CP metrics require an explicit group when TP or PP is enabled")
        reduced = all_reduce(value, op=op, group=group)
        return float(reduced)

    def _accumulate_batches(self, value: Any) -> None:
        """Accumulate batch metrics without retaining input tensor references."""
        for batch in self._micro_batches(value):
            token_count = self._batch_tokens(batch)
            if callable(getattr(token_count, "detach", None)):
                token_count = token_count.detach()
            if self._local_step_tokens is None:
                if callable(getattr(token_count, "clone", None)):
                    token_count = token_count.clone()
                self._local_step_tokens = token_count
            else:
                self._local_step_tokens = self._local_step_tokens + token_count
            self._local_step_samples += self._batch_samples(batch)
            shape = getattr(batch.get("input_ids"), "shape", ())
            if len(shape) == 2:
                batch_size, local_length = shape
                self._local_input_tokens += batch_size * local_length
            self._accumulate_flops(batch)

    def _accumulate_flops(self, batch: Mapping[str, Any]) -> None:
        """Collect complete model work from the selected estimator without host sync."""
        self._flops_observed = True
        cp_size = int(getattr(self.trainer.mesh, "cp_size", 1))
        flops = self._flops_estimator(batch, cp_size=cp_size)
        if flops is None or getattr(self.trainer.mesh, "pp_size", 1) > 1:
            self._local_flops = None
            if not self._flops_warning_shown and getattr(self.trainer, "global_rank", 0) == 0:
                logger.warning("Model TFLOP/s unavailable: configure flops_estimator for complete model work; "
                               "loss, token/s and sample/s logging remains available. PP FLOPs are not supported.")
                self._flops_warning_shown = True
        elif self._local_flops is not None:
            if self._tensor_numel(flops) not in (None, 1):
                raise ValueError("flops_estimator must return a scalar or None")
            if callable(getattr(flops, "detach", None)):
                flops = flops.detach().clone()
            valid = flops.isfinite() & (flops >= 0) if callable(getattr(flops, "isfinite", None)) else (
                math.isfinite(flops) and flops >= 0
            )
            self._flops_valid = self._flops_valid & valid
            self._local_flops = self._local_flops + flops / cp_size

    def _flops_throughput(self, seconds: float, world_size: int) -> dict[str, float]:
        """Make availability collective so unequal modality batches cannot hang."""
        available = self._flops_observed and self._local_flops is not None
        local_flops = self._scalar(self._local_flops, "model_flops") if available else 0.0
        valid = bool(self._scalar(self._flops_valid, "flops_valid")) and math.isfinite(local_flops)
        status = self._reduce(int(available) if valid else -1, op="min")
        if status < 0:
            raise ValueError("flops_estimator must return finite nonnegative FLOPs")
        if status == 0 or seconds <= 0:
            return {}
        return {"performance/throughput_tflops_per_device": self._reduce(local_flops, op="sum") / (
            seconds * world_size * 1e12
        )}

    def record_loss_metrics(self, model_output: Any) -> None:
        """Collect detached diagnostic means without adding them to backward loss.

        Args:
            model_output: Output with an optional ``loss_metrics`` scalar mapping.
                Keys must be stable across micro-batches and DP/CP ranks; values
                must represent full TP-replicated local means. Step logging takes
                the arithmetic micro-batch and DP/CP mean, matching MF's tracker
                convention, independently of token-weighted training objectives.
                Standard ``aux_loss`` and ``indexer_loss`` fields are also read.
        """
        def _read(name: str) -> Any:
            value = getattr(model_output, name, None)
            return model_output.get(name) if value is None and isinstance(model_output, Mapping) else value

        metrics = dict(_read("loss_metrics") or {})
        for name in ("aux_loss", "indexer_loss"):
            value = _read(name)
            if value is not None:
                metrics.setdefault(name, value)
        if self._loss_metric_steps and metrics.keys() != self._loss_metrics.keys():
            raise ValueError("loss_metrics keys must remain stable within an optimizer step")
        for name, value in metrics.items():
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError("loss_metrics keys must be nonempty names without '/' separators")
            if self._tensor_numel(value) not in (None, 1):
                raise ValueError(f"loss_metrics[{name!r}] must be scalar")
            if callable(getattr(value, "detach", None)):
                value = value.detach().clone()
            self._loss_metrics[name] = self._loss_metrics.get(name, 0) + value
        self._loss_metric_steps += 1

    def _global_samples(self) -> int:
        """Reduce samples across DP+CP while removing CP replicas."""
        cp_size = int(getattr(self.trainer.mesh, "cp_size", 1))
        if cp_size < 1:
            raise ValueError(f"mesh.cp_size must be positive, but got {cp_size}")
        reduced_samples = self._reduce(self._local_step_samples, op="sum")
        global_samples = reduced_samples / cp_size
        if not global_samples.is_integer():
            raise ValueError(
                "Reduced sample count must be divisible by cp_size, "
                f"but got reduced_samples={reduced_samples} and cp_size={cp_size}"
            )
        return int(global_samples)

    def _current_lr(self) -> float:
        """Return the maximum learning rate across scheduler or optimizer groups."""
        schedulers = self.trainer.lr_scheduler
        if schedulers is not None:
            scheduler_list = schedulers if isinstance(schedulers, list) else [schedulers]
            learning_rates = []
            for scheduler in scheduler_list:
                for learning_rate in scheduler.get_last_lr():
                    learning_rates.append(float(learning_rate))
            if learning_rates:
                return max(learning_rates)

        optimizers = self.trainer.optimizer
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
        learning_rates = []
        for optimizer in optimizer_list:
            if optimizer is None:
                continue
            for param_group in optimizer.param_groups:
                learning_rates.append(float(param_group["lr"]))
        return max(learning_rates, default=0.0)

    def _memory_metrics(self) -> dict[str, float]:
        """Collect maximum accelerator memory metrics, if available."""
        if get_device_type() == "cpu":
            return {}
        device = get_torch_device()
        allocated = self._reduce(device.max_memory_allocated(), op="max")
        reserved = self._reduce(device.max_memory_reserved(), op="max")
        gibibyte = 1024 ** 3
        return {
            "memory/device_max_allocated_gb": allocated / gibibyte,
            "memory/device_max_reserved_gb": reserved / gibibyte,
        }

    def state_dict(self) -> dict[str, int]:
        """Return cumulative metric state for future checkpoint integration."""
        return {
            "consumed_tokens": self._consumed_tokens,
            "consumed_samples": self._consumed_samples,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore cumulative metric state.

        Args:
            state_dict: Mapping produced by :meth:`state_dict`.

        Raises:
            ValueError: If required counters are missing or negative.
        """
        try:
            consumed_tokens = int(state_dict["consumed_tokens"])
            consumed_samples = int(state_dict["consumed_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "EnvironMeterCallback state must contain integer consumed_tokens and consumed_samples"
            ) from exc
        if consumed_tokens < 0 or consumed_samples < 0:
            raise ValueError("EnvironMeterCallback cumulative counters must be non-negative")
        self._consumed_tokens = consumed_tokens
        self._consumed_samples = consumed_samples

    def on_step_begin(
        self,
        state: TrainerState,
        micro_batches: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start timing and count local input tokens and samples.

        Args:
            state: Current training progress.
            micro_batches: Batches available before the step starts.
            **kwargs: Unused callback context.
        """
        del state, kwargs
        self._local_step_tokens = None
        self._local_step_samples = 0
        self._local_input_tokens = 0
        self._local_flops = 0.0
        self._flops_valid = True
        self._flops_observed = False
        self._loss_metrics = {}
        self._loss_metric_steps = 0
        if get_device_type() != "cpu":
            get_torch_device().synchronize()
        self._step_start_time = time.perf_counter()
        self._accumulate_batches(micro_batches)

    def on_micro_step_begin(
        self,
        state: TrainerState,
        micro_batch: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Accumulate metrics for one prepared micro-batch.

        Args:
            state: Current training progress.
            micro_batch: Prepared inputs and lightweight metric metadata.
            **kwargs: Unused callback context.
        """
        del state, kwargs
        self._accumulate_batches(micro_batch)

    def on_step_end(
        self,
        state: TrainerState,
        loss: float,
        loss_dict: dict[str, float] | None,
        grad_norm: float,
        **kwargs: Any,
    ) -> None:
        """Reduce and publish metrics for one completed optimizer step.

        Args:
            state: Current training progress.
            loss: Aggregated loss for the optimizer step.
            loss_dict: Named loss values for the optimizer step.
            grad_norm: Gradient norm measured before the optimizer update.
            **kwargs: Unused callback context.
        """
        del state, kwargs
        if get_device_type() != "cpu":
            get_torch_device().synchronize()
        step_time = max(time.perf_counter() - self._step_start_time, 0.0)
        world_size = get_world_size_safe()
        global_step_time = float(all_reduce(step_time, op="max", group=None)) if world_size > 1 else step_time
        local_step_tokens = 0 if self._local_step_tokens is None else self._local_step_tokens
        global_tokens = int(self._reduce(local_step_tokens, op="sum"))
        global_samples = self._global_samples()
        self._local_step_tokens = None
        self._local_step_samples = 0
        self._consumed_tokens += global_tokens
        self._consumed_samples += global_samples

        train_metrics = {
            "training/total_loss": self._reduce(self._scalar(loss, "total_loss"), op="mean"),
            "training/grad_norm": self._reduce(self._scalar(grad_norm, "grad_norm"), op="mean"),
            "training/lr": self._current_lr(),
        }
        for name, value in sorted((loss_dict or {}).items()):
            metric_name = name if name.startswith("training/") else f"training/{name}"
            train_metrics[metric_name] = self._reduce(self._scalar(value, name), op="mean")
        for name, value in sorted(self._loss_metrics.items()):
            metric_name = f"training/{name}"
            if metric_name in train_metrics:
                raise ValueError(f"Diagnostic metric {name!r} collides with a training objective")
            train_metrics[metric_name] = self._reduce(
                self._scalar(value, name) / self._loss_metric_steps, op="mean",
            )
        self._loss_metrics = {}
        self._loss_metric_steps = 0

        tokens_per_second = global_tokens / global_step_time if global_step_time > 0 else 0.0
        input_tokens = self._reduce(self._local_input_tokens, op="sum")
        throughput = {
            "performance/samples_per_second": global_samples / global_step_time if global_step_time > 0 else 0.0,
            "performance/input_tokens_per_second": input_tokens / global_step_time if global_step_time > 0 else 0.0,
        }
        throughput.update(self._flops_throughput(global_step_time, world_size))
        env_metrics = {
            **train_metrics,
            "performance/step_time": global_step_time,
            "performance/tokens_per_second": tokens_per_second,
            **throughput,
            "data/step_input_tokens": input_tokens,
            "data/step_tokens": float(global_tokens),
            "data/consumed_tokens": float(self._consumed_tokens),
            "data/step_samples": float(global_samples),
            "data/consumed_samples": float(self._consumed_samples),
            **self._memory_metrics(),
        }
        self.trainer.step_train_metrics = train_metrics
        self.trainer.step_env_metrics = env_metrics

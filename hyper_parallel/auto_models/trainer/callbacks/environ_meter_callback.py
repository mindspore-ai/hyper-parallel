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

import time
from collections.abc import Mapping, Sequence
from typing import Any

from hyper_parallel.auto_models.components.distributed.init_utils import get_world_size_safe
from hyper_parallel.auto_models.components.loss.dist_utils import all_reduce
from hyper_parallel.auto_models.components.utils.constants import IGNORE_INDEX
from hyper_parallel.auto_models.components.utils.device import get_device_type, get_torch_device

from .base import Callback, TrainerState


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
        self._local_step_tokens = 0
        self._local_step_samples = 0
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
    def _batch_tokens(cls, batch: Mapping[str, Any]) -> int:
        """Count text tokens in one micro-batch without mutating it."""
        labels = batch.get("labels")
        if labels is not None and callable(getattr(labels, "sum", None)):
            return int((labels != IGNORE_INDEX).sum().item())

        attention_mask = batch.get("attention_mask")
        attention_mask_shape = getattr(attention_mask, "shape", ())
        if (
            len(attention_mask_shape) <= 2
            and attention_mask is not None
            and callable(getattr(attention_mask, "sum", None))
        ):
            return int(attention_mask.sum().item())

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

    def _reduce(self, value: float | int, op: str) -> float:
        """Reduce one scalar metric, with a single-process no-op fallback."""
        if get_world_size_safe() <= 1:
            return float(value)
        reduced = all_reduce(value, op=op, group=self._metric_group())
        return float(reduced)

    def _current_lr(self) -> float:
        """Return the maximum learning rate across scheduler or optimizer groups."""
        schedulers = self.trainer.lr_scheduler
        if schedulers is not None:
            scheduler_list = schedulers if isinstance(schedulers, list) else [schedulers]
            learning_rates = [
                float(learning_rate)
                for scheduler in scheduler_list
                for learning_rate in scheduler.get_last_lr()
            ]
            if learning_rates:
                return max(learning_rates)

        optimizers = self.trainer.optimizer
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]
        learning_rates = [
            float(param_group["lr"])
            for optimizer in optimizer_list
            if optimizer is not None
            for param_group in optimizer.param_groups
        ]
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
        """Start timing and count local input tokens and samples."""
        del state, kwargs
        batches = self._micro_batches(micro_batches)
        self._local_step_tokens = sum(self._batch_tokens(batch) for batch in batches)
        self._local_step_samples = sum(self._batch_samples(batch) for batch in batches)
        self._step_start_time = time.perf_counter()

    def on_step_end(
        self,
        state: TrainerState,
        loss: float,
        loss_dict: dict[str, float] | None,
        grad_norm: float,
        **kwargs: Any,
    ) -> None:
        """Reduce and publish metrics for one completed optimizer step."""
        del state, kwargs
        step_time = max(time.perf_counter() - self._step_start_time, 0.0)
        global_step_time = self._reduce(step_time, op="max")
        global_tokens = int(self._reduce(self._local_step_tokens, op="sum"))
        global_samples = int(self._reduce(self._local_step_samples, op="sum"))
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

        tokens_per_second = global_tokens / global_step_time if global_step_time > 0 else 0.0
        env_metrics = {
            **train_metrics,
            "performance/step_time": global_step_time,
            "performance/tokens_per_second": tokens_per_second,
            "data/step_tokens": float(global_tokens),
            "data/consumed_tokens": float(self._consumed_tokens),
            "data/step_samples": float(global_samples),
            "data/consumed_samples": float(self._consumed_samples),
            **self._memory_metrics(),
        }
        self.trainer.step_train_metrics = train_metrics
        self.trainer.step_env_metrics = env_metrics

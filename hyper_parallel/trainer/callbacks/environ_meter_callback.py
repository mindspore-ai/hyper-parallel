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

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.trainer.runtime.distributed import get_world_size_safe
from hyper_parallel.trainer.runtime.distributed import all_reduce
from hyper_parallel.trainer.runtime.flops import estimate_step_flops
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.trainer.runtime.device import get_device_type, get_torch_device
from hyper_parallel.trainer.runtime.model_flops import (
    DeltaRuleLayer,
    MatmulParamCount,
    ModelFlopsProfile,
    attention_layers_from_config,
    count_matmul_params,
)

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class EnvironMeterCallback(Callback):
    """Collect structured training, throughput, and memory metrics.

    The callback is the single producer of ``trainer.step_train_metrics`` and
    ``trainer.step_env_metrics``. Presentation and remote logging callbacks
    consume those dictionaries without recalculating or reducing metrics.
    """

    def __init__(self, trainer: Any, *, microbatch_mean_metrics: Sequence[str] = ()) -> None:
        """Initialize per-step and cumulative counters.

        Args:
            trainer: Trainer that owns the callback lifecycle.
            microbatch_mean_metrics: Raw metric names to average over observed
                micro-batches before the existing DP+CP mean reduction. Other
                metrics retain the step values supplied by the Trainer.
        """
        super().__init__(trainer)
        self._step_start_time = 0.0
        self._local_step_tokens: Any = None
        self._local_step_input_tokens: Any = None
        self._local_step_seq_len_sq: Any = None
        self._local_step_samples = 0
        self._microbatch_mean_metrics = tuple(microbatch_mean_metrics)
        self._local_mean_metrics: dict[str, tuple[Any, int]] = {}
        self._consumed_tokens = 0
        self._consumed_samples = 0
        self._flops_profile: ModelFlopsProfile | None = None
        self._peak_flops_per_device: float | None = None
        self._recompute_all_layers = False
        self._flops_per_token_logged = False
        self._hfu_extra_flops_per_step = 0.0
        self._target_specs: list[tuple[str, str, float, bool]] = []
        self._target_stats: dict[str, dict[str, float]] = {}
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

    @classmethod
    def _batch_input_geometry(cls, batch: Mapping[str, Any]) -> tuple[Any, Any] | None:
        """Return model-input tokens and the sum of squared sequence lengths.

        The loss token count drops masked prompt tokens, but the model still
        runs attention and the FFN over them, so FLOPs need their own count:
        the tokens that reach attention plus ``sum(S ** 2)`` over the real
        attention boundaries.

        Every returned quantity is local to this rank, because
        :meth:`on_step_end` reduces them over DP+CP itself.

        Args:
            batch: One micro-batch mapping.

        Returns:
            ``(input_tokens, sum_seq_len_sq)`` for the sequences that reach
            attention, or ``None`` when the batch exposes no input geometry.
        """
        cu_seq_lens = batch.get("cu_seq_lens")
        if cu_seq_lens is not None and callable(getattr(cu_seq_lens, "diff", None)):
            local_tokens = cls._tensor_numel(batch.get("input_ids"))
            lengths = cu_seq_lens.diff()
            covered = int(lengths.sum())
            # Packed boundaries are only usable when they describe the tokens
            # this rank actually holds. Under context parallelism the packed
            # boundaries may still span the global sequence, and then the local
            # ``input_ids`` fallbacks below are the honest answer.
            if covered > 0 and (local_tokens is None or covered == local_tokens):
                return lengths.sum(), (lengths * lengths).sum()

        attention_mask = batch.get("attention_mask")
        if (
            attention_mask is not None
            and len(getattr(attention_mask, "shape", ())) == 2
            and callable(getattr(attention_mask, "sum", None))
        ):
            lengths = attention_mask.sum(dim=-1)
            return lengths.sum(), (lengths * lengths).sum()

        shape = getattr(batch.get("input_ids"), "shape", None)
        if shape is None or len(shape) < 2:
            return None
        rows = int(shape[0])
        seq_length = int(shape[-1])
        if rows <= 0 or seq_length <= 0:
            return None
        return rows * seq_length, rows * seq_length * seq_length

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
        reduced = all_reduce(value, op=op, group=self._metric_group())
        return float(reduced)

    @staticmethod
    def _accumulate_scalar(current: Any, value: Any) -> Any:
        """Add one batch scalar to a running per-step total.

        Args:
            current: Running total, or ``None`` before the first batch.
            value: Batch value to add; may be a device scalar.

        Returns:
            Updated running total that never aliases the batch tensor.
        """
        detached = value.detach() if callable(getattr(value, "detach", None)) else value
        if current is None:
            return detached.clone() if callable(getattr(detached, "clone", None)) else detached
        return current + detached

    def _accumulate_batches(self, value: Any) -> None:
        """Accumulate batch metrics without retaining input tensor references."""
        for batch in self._micro_batches(value):
            self._local_step_tokens = self._accumulate_scalar(
                self._local_step_tokens, self._batch_tokens(batch)
            )
            geometry = self._batch_input_geometry(batch)
            if geometry is not None:
                input_tokens, seq_len_sq = geometry
                self._local_step_input_tokens = self._accumulate_scalar(
                    self._local_step_input_tokens, input_tokens
                )
                self._local_step_seq_len_sq = self._accumulate_scalar(
                    self._local_step_seq_len_sq, seq_len_sq
                )
            self._local_step_samples += self._batch_samples(batch)

    def _cp_size(self) -> int:
        """Return the validated context-parallel size of the trainer mesh."""
        cp_size = int(getattr(self.trainer.mesh, "cp_size", 1))
        if cp_size < 1:
            raise ValueError(f"mesh.cp_size must be positive, but got {cp_size}")
        return cp_size

    def _global_samples(self) -> int:
        """Reduce samples across DP+CP while removing CP replicas."""
        cp_size = self._cp_size()
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

    def _log_rank_zero(self, level: int, message: str, *args: Any) -> None:
        """Log once per job from global rank zero."""
        if getattr(self.trainer, "global_rank", 0) == 0:
            logger.log(level, message, *args)

    def _mfu_config(self) -> Any:
        """Return ``training.mfu`` when the trainer carries a configuration."""
        training = getattr(getattr(self.trainer, "config", None), "training", None)
        return getattr(training, "mfu", None)

    def _gather_pipeline_counts(self, count: MatmulParamCount) -> MatmulParamCount:
        """Sum the disjoint parameter counts of pipeline stages.

        Every rank of a stage reports the same global DTensor sizes, so the
        world-wide sum is divided by the number of ranks per stage. Exact
        integers travel as objects because the counts exceed float32 precision.
        """
        pp_size = int(getattr(getattr(self.trainer, "mesh", None), "pp_size", 1) or 1)
        if pp_size <= 1 or not dist.is_initialized():
            return count
        world_size = dist.get_world_size()
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, count)
        ranks_per_stage = world_size // pp_size
        return MatmulParamCount(
            matmul_params=sum((item.matmul_params for item in gathered), start=0) / ranks_per_stage,
            output_head_params=sum((item.output_head_params for item in gathered), start=0) / ranks_per_stage,
            vision_params=sum((item.vision_params for item in gathered), start=0) / ranks_per_stage,
            unsupported=tuple(sorted({reason for item in gathered for reason in item.unsupported})),
        )

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Prepare PaLM model-FLOPs reporting when ``training.mfu.enabled`` is set.

        Args:
            state: Current training progress.
            **kwargs: Unused callback context.

        Raises:
            ValueError: If the configured peak throughput is not positive.
        """
        del state, kwargs
        mfu_config = self._mfu_config()
        if mfu_config is None or not mfu_config.enabled:
            return
        peak_tflops = mfu_config.peak_tflops_per_device
        if peak_tflops is not None and peak_tflops <= 0:
            raise ValueError(f"training.mfu.peak_tflops_per_device must be positive, but got {peak_tflops}")
        self._validate_mfu_targets(mfu_config)

        model_parts = getattr(self.trainer, "model_parts", None) or [self.trainer.model]
        model_config = getattr(self.trainer, "model_config", None)
        count = self._gather_pipeline_counts(count_matmul_params(model_parts, model_config))
        attention_layers, attention_reason = attention_layers_from_config(model_config)
        unsupported = list(count.unsupported) + ([attention_reason] if attention_reason else [])
        if unsupported:
            self._log_rank_zero(logging.WARNING, "PaLM MFU reporting is disabled: %s.", "; ".join(unsupported))
            return

        self._flops_profile = ModelFlopsProfile(
            matmul_params=float(count.matmul_params),
            output_head_params=float(count.output_head_params),
            attention_layers=attention_layers,
        )
        self._peak_flops_per_device = None if peak_tflops is None else float(peak_tflops) * 1e12
        checkpoint_mode = getattr(getattr(self.trainer.config, "activation_checkpoint", None), "mode", None)
        self._recompute_all_layers = checkpoint_mode == "full"
        extra_tflops = mfu_config.hfu_extra_tflops_per_step
        self._hfu_extra_flops_per_step = 0.0 if extra_tflops is None else float(extra_tflops) * 1e12
        self._target_specs = self._build_target_specs(mfu_config)
        if self._recompute_all_layers and extra_tflops is not None:
            hfu_note = f"adds the full-recompute forward and {extra_tflops:g} declared extra TFLOPs/step"
        elif self._recompute_all_layers:
            hfu_note = "adds the full-recompute forward"
        elif extra_tflops is not None:
            hfu_note = f"adds {extra_tflops:g} declared extra TFLOPs/step"
        else:
            hfu_note = "is not reported"
        linear_layers = sum(isinstance(layer, DeltaRuleLayer) for layer in attention_layers)
        self._log_rank_zero(
            logging.INFO,
            "PaLM MFU reporting: N=%.6e active matmul parameters per token (output head %.6e), "
            "%d softmax-attention and %d delta-rule layers%s, peak %s TFLOPS per device; HFU %s.",
            self._flops_profile.matmul_params,
            self._flops_profile.output_head_params,
            len(attention_layers) - linear_layers,
            linear_layers,
            f", decoder only ({float(count.vision_params):.6e} vision-encoder parameters excluded)"
            if count.vision_params else "",
            "not set (MFU not reported)" if peak_tflops is None else f"{peak_tflops:g}",
            hfu_note,
        )
        for label, _, threshold, higher in self._target_specs:
            self._log_rank_zero(logging.INFO, "MFU target registered: %s %s %.6g.",
                                label, ">=" if higher else "<=", threshold)

    def _validate_mfu_targets(self, mfu_config: Any) -> None:
        """Reject impossible target configurations before training starts."""
        for name in ("hfu_extra_tflops_per_step", "target_mfu_percent", "target_hfu_percent",
                     "target_tokens_per_s_per_device", "target_step_time_s"):
            value = getattr(mfu_config, name)
            if value is not None and value <= 0:
                raise ValueError(f"training.mfu.{name} must be positive, but got {value}")
        has_peak = mfu_config.peak_tflops_per_device is not None
        if not has_peak and (mfu_config.target_mfu_percent is not None
                             or mfu_config.target_hfu_percent is not None):
            raise ValueError("training.mfu utilization targets require peak_tflops_per_device")
        if mfu_config.target_hfu_percent is not None and mfu_config.hfu_extra_tflops_per_step is None:
            checkpoint_mode = getattr(getattr(self.trainer.config, "activation_checkpoint", None), "mode", None)
            if checkpoint_mode != "full":
                raise ValueError("training.mfu.target_hfu_percent requires full activation recompute "
                                 "or hfu_extra_tflops_per_step; HFU would never be measured")

    @staticmethod
    def _build_target_specs(mfu_config: Any) -> list[tuple[str, str, float, bool]]:
        """Collect the configured targets as (label, metric key, threshold, higher-is-better)."""
        candidates = (
            ("MFU %", "performance/mfu_percent", mfu_config.target_mfu_percent, True),
            ("HFU %", "performance/hfu_percent", mfu_config.target_hfu_percent, True),
            ("tokens/s/device", "performance/tokens_per_s_per_device",
             mfu_config.target_tokens_per_s_per_device, True),
            ("step time s", "step_time", mfu_config.target_step_time_s, False),
        )
        return [(label, key, float(value), higher)
                for label, key, value, higher in candidates if value is not None]

    def _track_targets(self, metrics: dict[str, float], step_time: float) -> None:
        """Judge each configured target against this logged step."""
        for label, key, threshold, higher in self._target_specs:
            value = step_time if key == "step_time" else metrics.get(key)
            if value is None:
                continue
            stats = self._target_stats.setdefault(
                label, {"sum": 0.0, "steps": 0, "met": 0, "threshold": threshold, "higher": higher})
            stats["sum"] += value
            stats["steps"] += 1
            if (value >= threshold) if higher else (value <= threshold):
                stats["met"] += 1

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Log one verdict line per configured target: mean, met count, met/missed."""
        del state, kwargs
        for label, stats in self._target_stats.items():
            mean = stats["sum"] / stats["steps"]
            higher = bool(stats["higher"])
            met_by_mean = (mean >= stats["threshold"]) if higher else (mean <= stats["threshold"])
            self._log_rank_zero(
                logging.INFO,
                "MFU target summary — %s: required %s %.6g, mean %.6g over %d logged steps "
                "(met in %d), verdict %s.",
                label, ">=" if higher else "<=", stats["threshold"], mean,
                int(stats["steps"]), int(stats["met"]), "MET" if met_by_mean else "MISSED",
            )

    def _flops_metrics(self, model_tokens: float, sequence_squares: float, step_time: float) -> dict[str, float]:
        """Return achieved model TFLOPS per device and the PaLM utilizations.

        The input geometry is shared with the TFLOP/s meter. The
        token-weighted sequence length preserves full-attention work over
        variable lengths and equals the sequence length on fixed batches.
        """
        if self._flops_profile is None or sequence_squares <= 0 or model_tokens <= 0 or step_time <= 0:
            return {}
        seq_len = sequence_squares / model_tokens
        tokens_per_second = model_tokens / step_time
        flops_per_token = self._flops_profile.flops_per_token(seq_len)
        if not self._flops_per_token_logged:
            self._log_rank_zero(
                logging.INFO, "PaLM MFU: sequence length %.6g, model FLOPs per token %.6e.", seq_len, flops_per_token
            )
            self._flops_per_token_logged = True
        world_size = max(get_world_size_safe(), 1)
        model_flops_per_device = flops_per_token * tokens_per_second / world_size
        metrics = {
            "performance/model_tflops_per_device": model_flops_per_device / 1e12,
            "performance/tokens_per_s_per_device": tokens_per_second / world_size,
        }
        if self._peak_flops_per_device is not None:
            metrics["performance/mfu_percent"] = model_flops_per_device / self._peak_flops_per_device * 100.0
            if self._recompute_all_layers or self._hfu_extra_flops_per_step:
                recompute_flops = (self._flops_profile.recompute_flops_per_token(seq_len)
                                   if self._recompute_all_layers else 0.0)
                hardware_flops_per_device = (
                    (flops_per_token + recompute_flops) * tokens_per_second / world_size
                    + self._hfu_extra_flops_per_step / step_time
                )
                metrics["performance/hfu_percent"] = hardware_flops_per_device / self._peak_flops_per_device * 100.0
        self._track_targets(metrics, step_time)
        return metrics

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
        self._local_step_input_tokens = None
        self._local_step_seq_len_sq = None
        self._local_step_samples = 0
        self._step_start_time = time.perf_counter()
        self._local_mean_metrics.clear()
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

    def on_micro_step_end(self, state: TrainerState, loss_dict: dict[str, Any], **kwargs: Any) -> None:
        """Accumulate explicitly registered micro-batch means without host reads."""
        del state, kwargs
        for name in self._microbatch_mean_metrics:
            value = loss_dict.get(name)
            if value is not None:
                total, count = self._local_mean_metrics.get(name, (None, 0))
                self._local_mean_metrics[name] = (self._accumulate_scalar(total, value), count + 1)

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
        step_time = max(time.perf_counter() - self._step_start_time, 0.0)
        global_step_time = self._reduce(step_time, op="max")
        local_step_tokens = 0 if self._local_step_tokens is None else self._local_step_tokens
        global_tokens = int(self._reduce(local_step_tokens, op="sum"))
        global_samples = self._global_samples()
        cp_size = self._cp_size()
        local_input_tokens = (
            0 if self._local_step_input_tokens is None else self._local_step_input_tokens
        )
        local_seq_len_sq = (
            0 if self._local_step_seq_len_sq is None else self._local_step_seq_len_sq
        )
        # Context parallelism slices a sequence along its length, so each rank
        # only sees ``(S / cp) ** 2`` while the real attention cost is ``S ** 2``.
        # Summing over DP+CP therefore needs one factor of ``cp_size`` to
        # restore the per-sequence square. Input tokens need no correction
        # because the slices add back up to ``S``.
        global_input_tokens = self._reduce(local_input_tokens, op="sum")
        global_seq_len_sq = self._reduce(local_seq_len_sq, op="sum") * cp_size
        self._local_step_tokens = None
        self._local_step_input_tokens = None
        self._local_step_seq_len_sq = None
        self._local_step_samples = 0
        self._consumed_tokens += global_tokens
        self._consumed_samples += global_samples

        train_metrics = {
            "training/total_loss": self._reduce(self._scalar(loss, "total_loss"), op="mean"),
            "training/grad_norm": self._reduce(self._scalar(grad_norm, "grad_norm"), op="mean"),
            "training/lr": self._current_lr(),
        }
        for name, value in sorted((loss_dict or {}).items()):
            if name in self._local_mean_metrics:
                total, count = self._local_mean_metrics[name]
                value = total / count
            metric_name = name if name.startswith("training/") else f"training/{name}"
            train_metrics[metric_name] = self._reduce(self._scalar(value, name), op="mean")
        self._local_mean_metrics.clear()

        tokens_per_second = global_tokens / global_step_time if global_step_time > 0 else 0.0
        env_metrics = {
            **train_metrics,
            "performance/step_time": global_step_time,
            "performance/tokens_per_second": tokens_per_second,
            "data/step_tokens": float(global_tokens),
            "data/consumed_tokens": float(self._consumed_tokens),
            "data/step_samples": float(global_samples),
            "data/consumed_samples": float(self._consumed_samples),
            **self._flops_metrics(global_input_tokens, global_seq_len_sq, global_step_time),
            **self._memory_metrics(),
        }

        training_config = getattr(getattr(self.trainer, "config", None), "training", None)
        model_config = getattr(self.trainer, "model_config", None)
        if (
            getattr(training_config, "log_tflops", False)
            and model_config is not None
            and global_input_tokens > 0
            and global_step_time > 0
        ):
            step_flops = estimate_step_flops(
                model_config, global_input_tokens, global_seq_len_sq
            )
            if step_flops is not None:
                env_metrics["performance/tflops_per_device"] = (
                    step_flops / (global_step_time * get_world_size_safe()) / 1e12
                )

        self.trainer.step_train_metrics = train_metrics
        self.trainer.step_env_metrics = env_metrics

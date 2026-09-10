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
"""Standalone step-throughput / TFLOPS / MFU probe callback.

``EnvironMeterCallback`` already publishes ``performance/tokens_per_second``
(and ``performance/tflops`` / ``performance/mfu`` when the FLOPs geometry is
resolvable) into ``trainer.step_env_metrics`` for the shared log line.  This
callback is the explicit, separately-registerable probe for users who want a
dedicated metric record — register it via
``trainer.add_callback(ThroughputMFUCallback(trainer))`` or set
``HP_THROUGHPUT_MFU=1`` in the environment (the trainer auto-registers it
then).  It reads the metrics EnvironMeterCallback produced earlier in the
dispatch order and only measures timing itself when those are absent.

``flops_per_token`` is never passed in; it is derived from the model and
training configuration via
:func:`hyper_parallel.models.flops.resolve_flops_per_token`:

1. a model family's own ``hp_flops_per_token`` wins when present (the model
   knows its exact geometry);
2. otherwise the generic 6N estimator reads the HF-style model config
   (``trainer.model_config``) and the sequence length observed from the
   first training batch, falling back to
   ``config.max_position_embeddings``.

Resolution is lazy (first ``on_step_end``) so it works regardless of
callback-vs-model init ordering.  When the geometry is insufficient,
TFLOPS/MFU are skipped and only tokens/s is reported.

``peak_tflops`` comes from the constructor or ``config.training.peak_tflops``;
without it MFU is skipped (the callback never guesses hardware peaks).  MFU
follows the 6N convention: recompute from activation checkpointing is NOT
counted as useful FLOPs.
"""

import logging
import time
from typing import Any, Optional

from hyper_parallel.models.flops import batch_seq_len, resolve_flops_per_token
from hyper_parallel.trainer.runtime.distributed import get_world_size_safe

from .base import Callback, TrainerState


logger = logging.getLogger(__name__)


class ThroughputMFUCallback(Callback):
    """Measure step throughput (tokens/s) and derive TFLOPS / MFU."""

    def __init__(
        self,
        trainer: Any,
        *,
        peak_tflops: Optional[float] = None,
        log_steps: Optional[int] = None,
    ) -> None:
        """Initialize the probe; FLOPs geometry is resolved lazily.

        Args:
            trainer: Trainer that owns the callback lifecycle.
            peak_tflops: Per-device peak dense TFLOPS for MFU normalization;
                defaults to ``config.training.peak_tflops``.
            log_steps: Emission cadence; defaults to the training config's
                ``logging_steps``.
        """
        super().__init__(trainer)
        training_cfg = trainer.config.training
        self._log_steps = log_steps or training_cfg.logging_steps
        self._peak_tflops = peak_tflops or training_cfg.peak_tflops
        self._global_batch_size = int(getattr(training_cfg, "global_batch_size", 1) or 1)
        self._flops_per_token: Optional[float] = None  # resolved lazily
        self._seq_len: Optional[int] = None  # captured from the first batch
        self._step_start_time = 0.0
        self._warned_missing = False
        self.last_record: dict[str, Any] = {}

    def on_step_begin(
        self,
        state: TrainerState,
        micro_batches: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record the step start timestamp and capture the sequence length.

        Args:
            state: Current trainer state (unused).
            micro_batches: Micro-batches of the step; the first batch
                carrying ``input_ids`` provides the sequence length.
        """
        del state, kwargs
        self._step_start_time = time.perf_counter()
        if self._seq_len is None:
            self._seq_len = batch_seq_len(micro_batches)

    def _resolve_seq_len(self) -> Optional[int]:
        """Return the observed batch sequence length or a config fallback."""
        if self._seq_len is not None:
            return self._seq_len
        model_config = getattr(self.trainer, "model_config", None)
        max_positions = getattr(model_config, "max_position_embeddings", None)
        return int(max_positions) if max_positions else None

    def _resolve_flops_per_token(self) -> Optional[float]:
        """Resolve FLOPs/token lazily from the model and its config geometry."""
        if self._flops_per_token is not None:
            return self._flops_per_token
        value = resolve_flops_per_token(
            getattr(self.trainer, "model", None),
            model_config=getattr(self.trainer, "model_config", None),
            seq_len=self._resolve_seq_len(),
        )
        if value:
            self._flops_per_token = value
        return self._flops_per_token

    def _resolve_throughput(self, state: TrainerState) -> tuple[float, float]:
        """Return ``(tokens_per_sec, step_time)``, preferring shared metrics."""
        del state
        env_metrics = getattr(self.trainer, "step_env_metrics", None) or {}
        tokens_per_sec = env_metrics.get("performance/tokens_per_second")
        step_time = env_metrics.get("performance/step_time")
        if tokens_per_sec is not None and step_time is not None:
            return float(tokens_per_sec), float(step_time)
        elapsed = max(time.perf_counter() - self._step_start_time, 1e-9)
        # Upper-bound estimate until real counts are available.
        tokens = self._global_batch_size * (self._resolve_seq_len() or 1)
        return tokens / elapsed, elapsed

    def on_step_end(
        self,
        state: TrainerState,
        loss: float = None,
        loss_dict: dict = None,
        grad_norm: float = None,
        **kwargs: Any,
    ) -> None:
        """Emit the ``throughput_mfu`` line on global rank zero.

        Args:
            state: Current trainer state; ``global_step`` drives cadence.
            loss: Reduced total loss (unused by this probe).
            loss_dict: Per-component losses (unused by this probe).
            grad_norm: Global gradient norm (unused by this probe).
        """
        del loss, loss_dict, grad_norm, kwargs
        if (
            self._log_steps <= 0
            or getattr(self.trainer, "global_rank", 0) != 0
            or state.global_step % self._log_steps != 0
        ):
            return

        tokens_per_sec, step_time = self._resolve_throughput(state)
        record = {
            "step": state.global_step,
            "step_time": step_time,
            "tokens_per_sec": tokens_per_sec,
        }
        fields = [f"step={state.global_step}", f"tokens_per_sec={tokens_per_sec:,.0f}"]

        flops_per_token = self._resolve_flops_per_token()
        if flops_per_token:
            # Observed TFLOPS = tokens/sec x flops/token / 1e12.
            observed = tokens_per_sec * flops_per_token / 1e12
            record["tflops"] = observed
            fields.append(f"tflops={observed:.1f}")
            if self._peak_tflops:
                world = max(get_world_size_safe(), 1)
                mfu = observed / (self._peak_tflops * world)
                record["mfu"] = mfu
                fields.append(f"mfu={mfu * 100:.1f}%")
            elif not self._warned_missing:
                self._warned_missing = True
                logger.info(
                    "ThroughputMFU: peak_tflops unset; reporting TFLOPS only "
                    "(set config.training.peak_tflops for MFU)"
                )
        elif not self._warned_missing:
            self._warned_missing = True
            logger.info(
                "ThroughputMFU: flops_per_token unresolvable from the model "
                "config; reporting tokens/s only"
            )

        self.last_record = record
        logger.info("throughput_mfu | %s", " | ".join(fields))

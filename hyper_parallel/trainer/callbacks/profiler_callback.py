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
"""MindFormers-derived training profiler backed by PyTorch."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.training import ProfilingConfig
from hyper_parallel.trainer.runtime.device import get_device_type
from hyper_parallel.trainer.state import TrainerState


if TYPE_CHECKING:
    from hyper_parallel.trainer.base import BaseTrainer


logger = logging.getLogger(__name__)


class ProfilerCallback(Callback):
    """Collect a bounded CPU, CUDA, or Ascend NPU profile.

    The profiler schedule, rank selection, output layout, metadata, and NPU
    options follow MindFormers ``ProfileMonitor`` semantics. PyTorch profiler
    activities replace the corresponding MindSpore profiler backend.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        """Build the profiler selected by ``TrainerConfig.profiler``.

        Args:
            trainer: Trainer that owns the callback lifecycle.
        """
        super().__init__(trainer)
        self.config: ProfilingConfig = trainer.config.profiler
        self.profiler: Any = None
        self.output_path: Optional[Path] = None
        self._profiler_module: Any = None
        self._torch_npu: Any = None
        self._device_type: Optional[str] = None
        self._is_profiler_started = False
        self._metadata_recorded = False
        self._training_step = 0
        self._mstx_range_id: Optional[int] = None

        if not self.config.enabled:
            return

        self._validate_types()
        self.start_step, self.stop_step = self._normalize_steps()
        self.start_on_init = self._normalize_start_on_init()
        self.level = self._normalize_level()

        rank = trainer.global_rank
        if not self._is_profile_required(rank):
            return

        self._device_type, self._profiler_module = self._resolve_profiler_module()
        if self.config.mstx and self._device_type != "npu":
            raise ValueError("profiler.mstx is supported only when training on Ascend NPU")

        output_root = Path(self.config.output_path or "./output").expanduser()
        self.output_path = output_root / "profile" / f"rank_{rank}"
        logger.info("Profile save path: %s", self.output_path)

        schedule_config = self._get_schedule()
        profile_kwargs = {
            "activities": self._get_activities(),
            "profile_memory": self.config.memory,
            "with_stack": self.config.with_stack,
            "schedule": schedule_config,
            "on_trace_ready": self._profiler_module.tensorboard_trace_handler(str(self.output_path)),
        }
        if self._device_type == "npu":
            profile_kwargs["experimental_config"] = self._get_npu_experimental_config()
        self.profiler = self._profiler_module.profile(**profile_kwargs)

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Start collection before the first training step when requested.

        Args:
            state: Current trainer state.
            **kwargs: Additional callback context supplied by the trainer.
        """
        del state, kwargs
        if self.profiler is None or not self.start_on_init:
            return
        self._start()
        self._record_metadata()

    def on_step_begin(
        self,
        state: TrainerState,
        micro_batches: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Start the profiler and align its schedule with the first training step.

        Args:
            state: Current trainer state.
            micro_batches: Optional micro-batches prepared for the current step.
            **kwargs: Additional callback context supplied by the trainer.
        """
        del micro_batches, kwargs
        if self.profiler is None:
            return

        self._training_step += 1
        if not self._is_profiler_started:
            self._start()
        if self._training_step == self.start_step:
            self._record_metadata()

        if self.config.mstx:
            mstx = self._torch_npu.npu.mstx
            step_num = state.global_step + 1
            self._mstx_range_id = mstx.range_start(
                f"step {step_num}",
                self._torch_npu.npu.current_stream(),
            )

    def on_step_end(
        self,
        state: TrainerState,
        loss: Optional[float] = None,
        loss_dict: Optional[dict[str, float]] = None,
        grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Advance the profiler schedule after one complete optimizer step.

        Args:
            state: Current trainer state.
            loss: Optional scalar loss for the completed step.
            loss_dict: Optional named losses for the completed step.
            grad_norm: Optional gradient norm for the completed step.
            **kwargs: Additional callback context supplied by the trainer.
        """
        del state, loss, loss_dict, grad_norm, kwargs
        if self.profiler is None or not self._is_profiler_started:
            return

        self._close_mstx_range()
        self.profiler.step()
        if self._training_step == self.stop_step:
            logger.info("End of profiling. Analyze the trace under %s", self.output_path)
            self._stop()

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Flush a partial profile when training ends before the stop step.

        Args:
            state: Final trainer state.
            **kwargs: Additional callback context supplied by the trainer.
        """
        del state, kwargs
        self._stop()

    def _validate_types(self) -> None:
        """Validate profiler fields that dataclass construction cannot constrain."""
        boolean_fields = (
            "enabled",
            "start_on_init",
            "memory",
            "pipeline_stage_leaders",
            "with_stack",
            "data_simplification",
            "mstx",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self.config, name), bool):
                raise TypeError(f"profiler.{name} must be a bool")
        for name in ("start_step", "stop_step"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"profiler.{name} must be an int")
        if self.config.output_path is not None and not isinstance(self.config.output_path, str):
            raise TypeError("profiler.output_path must be a string or null")
        ranks = self.config.rank_ids
        if ranks is not None:
            if not isinstance(ranks, list):
                raise TypeError("profiler.rank_ids must be a list or null")
            if any(isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self.trainer.world_size
                   for rank in ranks):
                raise ValueError(f"profiler.rank_ids must contain only integers in [0, {self.trainer.world_size})")

    def _normalize_steps(self) -> tuple[int, int]:
        """Normalize the inclusive profiling window to valid positive steps."""
        start_step = self.config.start_step
        stop_step = self.config.stop_step
        if start_step < 1:
            logger.warning("profiler.start_step must be greater than 0; reset it to 1")
            start_step = 1
        if stop_step < 1:
            logger.warning("profiler.stop_step must be greater than 0; reset it to 10")
            stop_step = 10
        if start_step > stop_step:
            logger.warning(
                "profiler.stop_step must be greater than or equal to profiler.start_step; reset both to 1 and 10"
            )
            start_step, stop_step = 1, 10
        return start_step, stop_step

    def _normalize_start_on_init(self) -> bool:
        if self.start_step != 1 and self.config.start_on_init:
            logger.warning(
                "profiler.start_step and profiler.start_on_init cannot take effect simultaneously; "
                "reset profiler.start_on_init to false"
            )
            return False
        return self.config.start_on_init

    def _normalize_level(self) -> int:
        level = self.config.level
        if level is None:
            return 0
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("profiler.level must be an int or null")
        if level not in (0, 1, 2):
            logger.warning("Invalid profiler.level %s; reset it to 0", level)
            return 0
        return level

    def _get_schedule(self) -> Any:
        if self.start_on_init:
            active_steps = self.stop_step
            skip_first = 1
        else:
            active_steps = self.stop_step - self.start_step + 1
            skip_first = self.start_step
        return self._profiler_module.schedule(
            wait=0,
            warmup=0,
            active=active_steps,
            repeat=1,
            skip_first=skip_first,
        )

    def _is_profile_required(self, rank: int) -> bool:
        """Return whether the current rank is selected for profiling."""
        ranks = self.config.rank_ids or []
        pipeline_ranks: list[int] = []
        if self.config.pipeline_stage_leaders:
            pipeline_stages = max(1, self.trainer.config.accelerator.pp_size)
            if self.trainer.world_size % pipeline_stages != 0:
                raise ValueError("device count must be divisible by pipeline stage count")
            devices_per_stage = self.trainer.world_size // pipeline_stages
            pipeline_ranks = [stage * devices_per_stage for stage in range(pipeline_stages)]

        if not ranks and not pipeline_ranks:
            return True
        return rank in ranks or rank in pipeline_ranks

    def _resolve_profiler_module(self) -> tuple[str, Any]:
        """Resolve the profiler backend for the active device type."""
        device_type = get_device_type()
        if device_type == "npu":
            # torch-npu is optional and must not become a trainer import-time dependency.
            import torch_npu  # pylint: disable=import-outside-toplevel

            self._torch_npu = torch_npu
            return device_type, torch_npu.profiler
        if device_type in ("cpu", "cuda"):
            return device_type, torch.profiler
        raise RuntimeError(f"profiler does not support device type {device_type!r}")

    def _get_activities(self) -> list[Any]:
        activities = [self._profiler_module.ProfilerActivity.CPU]
        if self._device_type == "cuda":
            activities.append(self._profiler_module.ProfilerActivity.CUDA)
        elif self._device_type == "npu":
            activities.append(self._profiler_module.ProfilerActivity.NPU)
        return activities

    def _get_npu_experimental_config(self) -> Any:
        """Build the torch-npu experimental profiler configuration."""
        profiler_level = getattr(self._profiler_module.ProfilerLevel, f"Level{self.level}")
        options = {
            "profiler_level": profiler_level,
            "data_simplification": self.config.data_simplification,
            "mstx": self.config.mstx,
        }
        export_type = getattr(getattr(self._profiler_module, "ExportType", None), "Text", None)
        if export_type is not None:
            options["export_type"] = [export_type]
        # torch-npu exposes this version-dependent profiler entry point only under its private name.
        experimental_config = getattr(self._profiler_module, "_ExperimentalConfig")
        return experimental_config(**options)

    def _start(self) -> None:
        self.profiler.start()
        self.profiler.step()
        self._is_profiler_started = True

    def _record_metadata(self) -> None:
        """Record distributed topology metadata once per profiling session."""
        if self._metadata_recorded:
            return
        parallel = self.trainer.config.accelerator
        metadata = {
            "tensor_model_parallel_size": parallel.tp_size,
            "pipeline_model_parallel_size": parallel.pp_size,
            "data_parallel_size": getattr(self.trainer.mesh, "dp_size", 1),
            "expert_model_parallel_size": parallel.ep_size,
            "sequence_parallel": parallel.sequence_parallel,
            "parallel_mode": "distributed" if self.trainer.world_size > 1 else "stand_alone",
            "world_size": self.trainer.world_size,
        }
        try:
            self.profiler.add_metadata_json("distributed_args", json.dumps(metadata))
            self._metadata_recorded = True
        except AttributeError as error:
            logger.warning("Profiler failed to record distributed args: %s", error)

    def _close_mstx_range(self) -> None:
        if self._mstx_range_id is None:
            return
        self._torch_npu.npu.mstx.range_end(self._mstx_range_id)
        self._mstx_range_id = None

    def _stop(self) -> None:
        self._close_mstx_range()
        profiler = self.profiler
        self.profiler = None
        if profiler is not None and self._is_profiler_started:
            profiler.stop()
        self._is_profiler_started = False


__all__ = ["ProfilerCallback"]

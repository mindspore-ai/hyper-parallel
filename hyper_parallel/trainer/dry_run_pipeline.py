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
"""Pipeline contracts and P2P mocks used by the Torch LLM Dry-run."""
# This module is part of the explicitly Torch-only LLM dry-run implementation.
# pylint: disable=forbidden-backend-import

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from hyper_parallel.core.pipeline_parallel.scheduler import (
    PipelineScheduleRuntime,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.core.pipeline_parallel.stage import PipelineStage


_PIPELINE_SCHEDULES = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
}


def normalize_pipeline_schedule(schedule_name: Optional[str], pp_vpp: int = 1) -> str:
    """Validate and normalize a Dry-run pipeline schedule.

    Args:
        schedule_name: User-selected schedule, or ``None`` for ``1f1b``.
        pp_vpp: Number of virtual stages owned by each pipeline rank.

    Returns:
        Normalized schedule name. VPP resolves to ``interleaved_1f1b``.

    Raises:
        ValueError: If the schedule or VPP degree is invalid.
    """
    if not isinstance(pp_vpp, int) or isinstance(pp_vpp, bool) or pp_vpp < 1:
        raise ValueError(f"pp_vpp must be a positive integer, got {pp_vpp!r}")
    normalized = (
        "1f1b"
        if schedule_name is None
        else str(schedule_name).lower().replace("-", "_")
    )
    if pp_vpp > 1:
        if normalized not in ("1f1b", "interleaved_1f1b"):
            raise ValueError(
                f"pp_vpp > 1 requires pp_schedule='1f1b', got {schedule_name!r}"
            )
        return "interleaved_1f1b"
    if normalized not in _PIPELINE_SCHEDULES:
        raise ValueError(
            f"Unknown pp_schedule {schedule_name!r}; choose from {sorted(_PIPELINE_SCHEDULES)}"
        )
    return normalized


def build_pipeline_schedule(
        stages: Any,
        micro_batch_num: int,
        *,
        schedule_name: Optional[str] = None,
        pp_vpp: int = 1,
        **schedule_kwargs: Any,
) -> PipelineScheduleRuntime:
    """Build the configured physical or virtual Dry-run pipeline schedule.

    Args:
        stages: One stage or the local list of virtual stages.
        micro_batch_num: Number of micro-batches executed by the schedule.
        schedule_name: ``gpipe`` or ``1f1b``. ``None`` defaults to ``1f1b``.
        pp_vpp: Number of virtual stages per pipeline rank.
        **schedule_kwargs: Batch-dimension and supported scheduler arguments.

    Returns:
        A configured pipeline schedule runtime.

    Raises:
        ValueError: If ``micro_batch_num`` is not a positive integer.
    """
    if (
            not isinstance(micro_batch_num, int)
            or isinstance(micro_batch_num, bool)
            or micro_batch_num < 1
    ):
        raise ValueError(
            f"micro_batch_num must be a positive integer, got {micro_batch_num!r}"
        )
    normalized = normalize_pipeline_schedule(schedule_name, pp_vpp)
    schedule_class = (
        ScheduleInterleaved1F1B
        if normalized == "interleaved_1f1b"
        else _PIPELINE_SCHEDULES[normalized]
    )
    return schedule_class(stages, micro_batch_num, **schedule_kwargs)


@dataclass(frozen=True)
class DryRunBoundaryLeaf:
    """Describe one ordered tensor crossing a static pipeline boundary.

    ``wire_kind`` records whether the stage returns a local Tensor or keeps a
    DTensor wrapper. Placements and local shape always come from the planner.
    """

    global_shape: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    anchor_fqn: Optional[str] = None
    tensor_name: str = "output"
    wire_kind: str = "tensor"

    def __post_init__(self) -> None:
        """Validate the static boundary description."""
        if not self.global_shape or any(size <= 0 for size in self.global_shape):
            raise ValueError(
                "DryRunBoundaryLeaf.global_shape must contain positive dimensions, "
                f"got {self.global_shape}"
            )
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(
                f"DryRunBoundaryLeaf.dtype must be torch.dtype, got {self.dtype!r}"
            )
        if not self.tensor_name:
            raise ValueError("DryRunBoundaryLeaf.tensor_name must be non-empty")
        if self.wire_kind not in ("tensor", "dtensor"):
            raise ValueError(
                "DryRunBoundaryLeaf.wire_kind must be 'tensor' or 'dtensor', "
                f"got {self.wire_kind!r}"
            )


@dataclass(frozen=True)
class DryRunPipelineChunk:
    """Describe one model-specific pipeline chunk returned by an example builder.

    Args:
        module: Stage-local module following the dry-run pipeline forward contract.
        layer_start: First global decoder-layer index owned by the stage.
        layer_end: Exclusive global decoder-layer index owned by the stage.
        hidden_size: Hidden width crossing pipeline boundaries.
        stage_index: Global physical or virtual pipeline-stage index.
        input_boundary: Ordered tensor leaves received from the previous stage.
        output_boundary: Ordered tensor leaves sent to the next stage.
        fsdp_units: Ordered groups of modules manually wrapped as FSDP units.
    """

    module: nn.Module
    layer_start: int
    layer_end: int
    hidden_size: int
    stage_index: int = -1
    input_boundary: tuple[DryRunBoundaryLeaf, ...] = ()
    output_boundary: tuple[DryRunBoundaryLeaf, ...] = ()
    fsdp_units: tuple[tuple[nn.Module, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate the model-builder result at the framework boundary."""
        if not isinstance(self.module, nn.Module):
            raise ValueError("DryRunPipelineChunk.module must be a torch.nn.Module")
        if self.layer_start < 0 or self.layer_end < self.layer_start:
            raise ValueError(
                "DryRunPipelineChunk layer range must be ordered and non-negative, "
                f"got [{self.layer_start}, {self.layer_end})"
            )
        if self.hidden_size <= 0:
            raise ValueError("DryRunPipelineChunk.hidden_size must be positive")
        if self.stage_index < -1:
            raise ValueError("DryRunPipelineChunk.stage_index must be non-negative or -1")
        for unit in self.fsdp_units:
            if not unit or not all(isinstance(module, nn.Module) for module in unit):
                raise ValueError(
                    "DryRunPipelineChunk.fsdp_units must contain non-empty module tuples"
                )


class _NoOpWork:
    """Minimal asynchronous-work stand-in used by the scheduler wait paths."""

    @staticmethod
    def wait() -> None:
        """Complete immediately without moving a payload."""


class DryRunPipelineStage(PipelineStage):
    """Run normal pipeline cache transitions while mocking cross-rank transport."""

    def __init__(
            self,
            submodule: nn.Module,
            *,
            stage_index: int,
            stage_num: int,
            device: torch.device,
            input_metadata: list[list[Any]],
            output_metadata: list[list[Any]],
            mesh: Any,
            group: Any = None,
    ) -> None:
        """Initialize one physical or virtual dry-run pipeline stage.

        Args:
            submodule: Stage-local model chunk.
            stage_index: Global pipeline-stage index owned by this stage.
            stage_num: Total number of physical and virtual pipeline stages.
            device: FakeTensor simulation device.
            input_metadata: Ordered metadata used to allocate forward recv buffers.
            output_metadata: Ordered metadata used to validate forward send tensors.
            mesh: Existing one-dimensional PP mesh used for peer-rank mapping.
            group: Optional explicit PP process group.
        """
        super().__init__(
            submodule,
            stage_index=stage_index,
            stage_num=stage_num,
            device=device,
            group=group,
            mesh=mesh,
        )
        self._input_metadata = input_metadata
        self._output_metadata = output_metadata
        self._meta_cache = []

    def _communicate_meta(self, global_rank: int, meta_send: Any = None) -> Any:
        """Validate sender metadata or supply receiver metadata without communication."""
        del global_rank
        if meta_send is None:
            return [self._input_metadata]
        expected = self._output_metadata
        if len(meta_send) != len(expected):
            raise ValueError(
                "Dry-run PP boundary output count mismatch: expected "
                f"{len(expected)}, got {len(meta_send)}"
            )
        for leaf_index, (actual, expected_leaf) in enumerate(zip(meta_send, expected)):
            if len(actual) != len(expected_leaf):
                raise ValueError(
                    f"Dry-run PP boundary leaf {leaf_index} metadata kind mismatch: "
                    f"expected {len(expected_leaf)} fields, got {len(actual)}"
                )
            if tuple(actual[0]) != tuple(expected_leaf[0]):
                raise ValueError(
                    f"Dry-run PP boundary leaf {leaf_index} shape mismatch: "
                    f"expected {tuple(expected_leaf[0])}, got {tuple(actual[0])}"
                )
            if actual[1] != expected_leaf[1]:
                raise ValueError(
                    f"Dry-run PP boundary leaf {leaf_index} dtype mismatch: "
                    f"expected {expected_leaf[1]}, got {actual[1]}"
                )
            if len(actual) == 4:
                actual_placements = tuple(actual[2].alias_placements)
                expected_placements = tuple(expected_leaf[2].alias_placements)
                if actual_placements != expected_placements:
                    raise ValueError(
                        f"Dry-run PP boundary leaf {leaf_index} layout mismatch: "
                        f"expected {expected_placements}, got {actual_placements}"
                    )
            if bool(actual[-1]) != bool(expected_leaf[-1]):
                raise ValueError(
                    f"Dry-run PP boundary leaf {leaf_index} requires_grad mismatch: "
                    f"expected {bool(expected_leaf[-1])}, got {bool(actual[-1])}"
                )
        return None

    @staticmethod
    def _mock_works(specs: list[tuple[str, Any, int]]) -> list[_NoOpWork]:
        """Return one completed work object for each inherited communication spec."""
        return [_NoOpWork() for _ in specs]

    def exec_fwd_recv_ops(self, micro_index: int) -> list[_NoOpWork]:
        """Allocate and register the inherited forward receive buffers."""
        return self._mock_works(self.fwd_recv_specs(micro_index))

    def exec_fwd_send_ops(self, micro_index: int) -> list[_NoOpWork]:
        """Apply inherited forward-send cache transitions without sending."""
        return self._mock_works(self.fwd_send_specs(micro_index))

    def exec_bwd_recv_ops(self, micro_index: int) -> list[_NoOpWork]:
        """Expose inherited gradient receive buffers without receiving."""
        return self._mock_works(self.bwd_recv_specs(micro_index))

    def exec_bwd_send_ops(self, micro_index: int) -> list[_NoOpWork]:
        """Apply inherited backward-send cache transitions without sending."""
        return self._mock_works(self.bwd_send_specs(micro_index))

    def forward_one_chunk(
            self,
            micro_index: int,
            args: Any = None,
            kwargs: Any = None,
    ) -> Any:
        """Select stage-local labels before delegating the forward cache logic."""
        set_micro_index = getattr(self.submodule, "set_micro_index", None)
        if set_micro_index is not None:
            set_micro_index(micro_index)
        return super().forward_one_chunk(micro_index, args, kwargs)

    def set_micro_labels(self, labels: list[torch.Tensor]) -> None:
        """Install last-stage labels split in scheduler micro-batch order."""
        setter = getattr(self.submodule, "set_micro_labels", None)
        if setter is None:
            raise ValueError("The last dry-run PP stage must implement set_micro_labels")
        setter(labels)

    def clear_all_states(self) -> None:
        """Drop all per-run caches and receive-buffer references."""
        self.clear_states()
        self.clear_cache()
        self.last_stage_outputs = None


__all__ = [
    "DryRunBoundaryLeaf",
    "DryRunPipelineChunk",
    "DryRunPipelineStage",
    "build_pipeline_schedule",
    "normalize_pipeline_schedule",
]

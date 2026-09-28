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
"""Plan and decode TaskDesc-based MegaKernel per-core cycle records."""

from __future__ import annotations

__all__ = []

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import struct
from threading import Lock
from typing import Any

import torch_npu

from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID,
    INVALID_PROFILE_DESC_ID,
    INVALID_PROFILE_OWNER_ID,
    NUM_WORKERS_CUBE,
    NUM_WORKERS_VECTOR,
    RuntimeConfigC,
    TaskDescC,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config


CUBE_SLOT_COUNT = NUM_WORKERS_CUBE
VECTOR_SLOT_COUNT = NUM_WORKERS_VECTOR
CORE_SLOT_COUNT = CUBE_SLOT_COUNT + VECTOR_SLOT_COUNT
RECORD_CAPACITY_ALIGNMENT = 16
MAX_RECORDS_PER_CORE = 256
CORE_HEADER = struct.Struct("<QIIIIII32x")
PROFILE_RECORD = struct.Struct("<QQIIII")
MAX_CORE_STRIDE_BYTES = CORE_HEADER.size + MAX_RECORDS_PER_CORE * PROFILE_RECORD.size
MAX_PROFILE_BUFFER_BYTES = CORE_SLOT_COUNT * MAX_CORE_STRIDE_BYTES
INVALID_OWNER_ID = 0xFFFFFFFF
TASK_TYPE_DESC_BASE = 0x20000
GRAPH_STAGE_DESC_BASE = 0x30000
_PROFILE_METADATA_ATTRIBUTE = "_mega_kernel_profile_metadata"
_SYSTEM_COUNTER_FREQUENCY_MHZ = {
    "ascend910b": 50.0,
    "ascend910c": 50.0,
    "ascend910_93": 50.0,
}

DEFAULT_STAGE_NAMES = {
    0x10000: "WaitDependency",
    0x10006: "TerminateTask",
    0x10007: "TriggerEvent",
}
TASK_TYPE_NAMES = {
    0: "Terminate",
    10: "BeginTaskGraph",
    101: "AddCustom",
    102: "SwiGLU",
    103: "MatMul",
    104: "GroupedMatMul",
    105: "ShmemPutMemSignal",
    106: "SwiGLUGrad",
    107: "MhcPost",
    108: "MhcNormCast",
    109: "MhcProjection",
    110: "MhcMapping",
    111: "RmsNorm",
    112: "MhcInputMix",
    113: "RmsNormGrad",
    114: "MhcGradPrevA",
    115: "MhcGradMapping",
    116: "MhcGradPhiRms",
    117: "MhcGradPrevXAndPost",
    118: "MhcPostGrad",
    119: "MhcGradPrevAAndMapping",
}
CORE_TYPE_NAMES = {
    1: "AIC",
    2: "AIV",
}


@dataclass(frozen=True)
class _ProfileMetadata:
    """Host-only display metadata selected by a concrete MegaKernel builder."""

    kernel_name: str
    owner_label: str
    stage_names: Mapping[int, str]
    task_stage_names: Mapping[int, str]


@dataclass(frozen=True)
class _ProfileSpec:
    """Kernel-level profiling semantics applied to a completed compute graph."""

    kernel_name: str
    owner_label: str
    owner_resolver: Callable[[OperatorNode, TaskDescC, Any], int | None] | None = None


@dataclass(frozen=True)
class _ProfileLayout:
    """Host-computed per-worker record requirements and bounded buffer layout."""

    aic_required_records: int
    aiv_required_records: int
    aic_record_capacity: int
    aiv_record_capacity: int

    @property
    def buffer_size(self) -> int:
        """Return the ordinary Device buffer size for per-core cycle records."""
        return _profile_buffer_bytes_for_capacities(
            self.aic_record_capacity,
            self.aiv_record_capacity,
        )


@dataclass(frozen=True)
class _CycleTraceConfig:
    """Resolved hardware and display metadata used while decoding one buffer."""

    rank: int
    device_id: int | None
    cycle_frequency_mhz: float
    detailed_task_names: bool
    kernel_name: str
    owner_label: str
    stage_names: Mapping[int, str]
    soc_name: str
    task_stage_names: Mapping[int, str] | None = None


def _get_soc_name(device_id: int | None) -> str:
    """Return and validate the current Ascend SoC name through Torch NPU."""
    get_device_name = torch_npu.npu.get_device_name
    try:
        soc_name = get_device_name(device_id)
    except TypeError:
        # Older torch_npu releases only expose the current-device form.
        soc_name = get_device_name()
    if not isinstance(soc_name, str) or not soc_name.strip():
        raise RuntimeError(f"Torch NPU returned an invalid Ascend SoC name: {soc_name!r}")
    return soc_name.strip()


def _resolve_cycle_frequency_mhz(soc_name: str) -> float:
    """Resolve the GetSystemCycle frequency for a backend-reported SoC name."""
    normalized_name = "".join(character for character in soc_name.lower() if character.isalnum() or character == "_")
    if not normalized_name.startswith("ascend"):
        normalized_name = f"ascend{normalized_name}"
    for soc_prefix, frequency_mhz in _SYSTEM_COUNTER_FREQUENCY_MHZ.items():
        if normalized_name.startswith(soc_prefix):
            return frequency_mhz
    supported_soc_names = ", ".join(sorted(_SYSTEM_COUNTER_FREQUENCY_MHZ))
    raise RuntimeError(
        f"Unsupported Ascend SoC for MegaKernel profiling: {soc_name!r}; "
        f"supported SoC prefixes: {supported_soc_names}"
    )


def _validate_runtime_config(runtime_config: RuntimeConfigC) -> None:
    if not isinstance(runtime_config, RuntimeConfigC):
        raise TypeError(f"runtime_config must be RuntimeConfigC, got {type(runtime_config).__name__}")


def _profile_buffer_bytes_for_capacities(aic_record_capacity: int, aiv_record_capacity: int) -> int:
    aic_stride = CORE_HEADER.size + aic_record_capacity * PROFILE_RECORD.size
    aiv_stride = CORE_HEADER.size + aiv_record_capacity * PROFILE_RECORD.size
    return CUBE_SLOT_COUNT * aic_stride + VECTOR_SLOT_COUNT * aiv_stride


def _round_up_record_capacity(required_records: int) -> int:
    """Round a per-worker requirement up to 16 records and enforce the 256-record limit."""
    aligned_capacity = (
        (required_records + RECORD_CAPACITY_ALIGNMENT - 1) // RECORD_CAPACITY_ALIGNMENT * RECORD_CAPACITY_ALIGNMENT
    )
    return min(aligned_capacity, MAX_RECORDS_PER_CORE)


def _records_for_task(task_desc: TaskDescC) -> int:
    """Mirror the wait, compute, and trigger records emitted by ExecuteTaskProfiled()."""
    record_count = 1
    if task_desc.dependent_event != EVENT_INVALID_ID:
        record_count += 1
    if task_desc.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
        record_count += 1
    return record_count


def _max_worker_record_count(
    runtime_config: RuntimeConfigC,
    task_indices: Any,
    scheduled_task_count: int,
    active_worker_count: int,
) -> int:
    """Return the largest record count assigned to one active worker."""
    if scheduled_task_count < 0 or scheduled_task_count > len(task_indices):
        raise ValueError(f"scheduled task count is outside RuntimeConfig capacity: {scheduled_task_count}")
    if scheduled_task_count == 0:
        return 0
    if active_worker_count <= 0:
        raise ValueError("RuntimeConfig.num_workers must identify at least one AIC/AIV worker")

    worker_record_counts = [0] * active_worker_count
    for schedule_index in range(scheduled_task_count):
        task_id = int(task_indices[schedule_index])
        if not 0 <= task_id < len(runtime_config.all_tasks):
            raise ValueError(f"scheduled task ID is outside RuntimeConfig capacity: {task_id}")
        worker_id = schedule_index % active_worker_count
        worker_record_counts[worker_id] += _records_for_task(runtime_config.all_tasks[task_id])
    return max(worker_record_counts)


def _calculate_profile_layout(runtime_config: RuntimeConfigC) -> _ProfileLayout:
    """Reproduce Device task distribution and size AIC/AIV slots from the busiest worker."""
    _validate_runtime_config(runtime_config)
    num_workers = int(runtime_config.num_workers)
    if num_workers < 0 or num_workers % 2 != 0 or num_workers > VECTOR_SLOT_COUNT:
        raise ValueError(
            f"RuntimeConfig.num_workers must be an even value in [0, {VECTOR_SLOT_COUNT}], got {num_workers}"
        )
    active_worker_count = num_workers // 2
    aic_required_records = _max_worker_record_count(
        runtime_config,
        runtime_config.cube_task_indices,
        int(runtime_config.task_index_num[0]),
        active_worker_count,
    )
    aiv_required_records = _max_worker_record_count(
        runtime_config,
        runtime_config.vector_task_indices,
        int(runtime_config.task_index_num[1]),
        active_worker_count,
    )
    return _ProfileLayout(
        aic_required_records=aic_required_records,
        aiv_required_records=aiv_required_records,
        aic_record_capacity=_round_up_record_capacity(aic_required_records),
        aiv_record_capacity=_round_up_record_capacity(aiv_required_records),
    )


def _configure_profile_layout(runtime_config: RuntimeConfigC) -> _ProfileLayout:
    """Calculate and serialize the bounded AIC/AIV record capacities."""
    layout = _calculate_profile_layout(runtime_config)
    runtime_config.aic_profile_record_capacity = layout.aic_record_capacity
    runtime_config.aiv_profile_record_capacity = layout.aiv_record_capacity
    return layout


def _resolve_stage_names(stage_names: Mapping[int, str] | None) -> dict[int, str]:
    """Merge validated kernel-specific names with the common runtime stages."""
    resolved_stage_names = dict(DEFAULT_STAGE_NAMES)
    if stage_names is None:
        return resolved_stage_names
    for desc_id, stage_name in stage_names.items():
        if isinstance(desc_id, bool) or not isinstance(desc_id, int):
            raise TypeError(f"stage_names keys must be integers, got {desc_id!r}")
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError(f"stage_names values must be non-empty strings, got {stage_name!r}")
        resolved_stage_names[desc_id] = stage_name
    return resolved_stage_names


def _set_mega_kernel_profile_metadata(
    runtime_config: RuntimeConfigC,
    *,
    kernel_name: str,
    owner_label: str,
    stage_names: Mapping[int, str],
    task_stage_names: Mapping[int, str] | None = None,
) -> None:
    """Attach concrete-kernel display metadata without changing serialized RuntimeConfig."""
    _validate_runtime_config(runtime_config)
    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError(f"kernel_name must be a non-empty string, got {kernel_name!r}")
    if not isinstance(owner_label, str) or not owner_label.strip():
        raise ValueError(f"owner_label must be a non-empty string, got {owner_label!r}")
    metadata = _ProfileMetadata(
        kernel_name=kernel_name.strip(),
        owner_label=owner_label.strip(),
        stage_names=_resolve_stage_names(stage_names),
        task_stage_names=dict(task_stage_names or {}),
    )
    setattr(runtime_config, _PROFILE_METADATA_ATTRIBUTE, metadata)


def _operator_profile_name(operator: OperatorNode) -> str:
    """Return an explicit diagnostic name or derive one from the graph node name."""
    diagnostic_name = operator.diagnostic_name
    if diagnostic_name is not None:
        if not isinstance(diagnostic_name, str) or not diagnostic_name.strip():
            raise ValueError(
                f"OperatorNode.diagnostic_name must be a non-empty string when set: {diagnostic_name!r}"
            )
        return diagnostic_name.strip()
    if not isinstance(operator.name, str) or not operator.name.strip():
        raise ValueError(f"OperatorNode.name must be a non-empty string, got {operator.name!r}")
    stage_name = "".join(part[:1].upper() + part[1:] for part in operator.name.strip().split("_") if part)
    if not stage_name:
        raise ValueError(f"OperatorNode.name must contain a profiling display character: {operator.name!r}")
    return stage_name


def _scheduled_task_ids(runtime_config: RuntimeConfigC) -> set[int]:
    """Collect queue task IDs, including tasks outside RuntimeConfig.task_num."""
    task_ids = set()
    task_schedules = (
        (runtime_config.cube_task_indices, int(runtime_config.task_index_num[0])),
        (runtime_config.vector_task_indices, int(runtime_config.task_index_num[1])),
        (runtime_config.mix_task_indices, int(runtime_config.task_index_num[2])),
    )
    for task_indices, scheduled_task_count in task_schedules:
        if not 0 <= scheduled_task_count <= len(task_indices):
            raise ValueError(f"scheduled task count is outside RuntimeConfig capacity: {scheduled_task_count}")
        for schedule_index in range(scheduled_task_count):
            task_id = int(task_indices[schedule_index])
            if not 0 <= task_id < len(runtime_config.all_tasks):
                raise ValueError(f"scheduled task ID is outside RuntimeConfig capacity: {task_id}")
            task_ids.add(task_id)
    return task_ids


def _graph_profile_stages(
    graph: ComputeGraph,
) -> tuple[dict[int, str], dict[int, tuple[OperatorNode, int, str]], int]:
    """Map graph task ranges to generated stage IDs and display names."""
    stage_names = {}
    task_operators = {}
    first_task_id = 0
    for stage_index, operator in enumerate(graph.topological_sort()):
        task_num = operator.task_num
        if isinstance(task_num, bool) or not isinstance(task_num, int) or task_num < 0:
            raise ValueError(f"operator {operator.name!r} has an invalid task_num: {task_num!r}")
        last_task_id = first_task_id + task_num
        stage_name = _operator_profile_name(operator)
        desc_id = GRAPH_STAGE_DESC_BASE + stage_index
        stage_names[desc_id] = stage_name
        for task_id in range(first_task_id, last_task_id):
            task_operators[task_id] = (operator, desc_id, stage_name)
        first_task_id = last_task_id
    return stage_names, task_operators, first_task_id


def _resolve_profile_owner_id(
    spec: _ProfileSpec,
    operator: OperatorNode,
    task_desc: TaskDescC,
    context: Any,
) -> int:
    """Resolve and validate an optional business owner for one task."""
    if spec.owner_resolver is None:
        return INVALID_PROFILE_OWNER_ID
    owner_id = spec.owner_resolver(operator, task_desc, context)
    if owner_id is None:
        return INVALID_PROFILE_OWNER_ID
    if isinstance(owner_id, bool) or not isinstance(owner_id, int):
        raise TypeError(f"profile owner_id must be an integer or None, got {owner_id!r}")
    if not 0 <= owner_id < INVALID_PROFILE_OWNER_ID:
        raise ValueError(f"profile owner_id is outside uint32 range: {owner_id}")
    return owner_id


def _resolve_profile_task(
    task_id: int,
    task_desc: TaskDescC,
    task_operators: Mapping[int, tuple[OperatorNode, int, str]],
    spec: _ProfileSpec,
    context: Any,
) -> tuple[int, int, int, str | None]:
    """Resolve serialized profile IDs and the Host display name for one task."""
    operator_entry = task_operators.get(task_id)
    if operator_entry is not None:
        operator, profile_desc_id, stage_name = operator_entry
        profile_owner_id = _resolve_profile_owner_id(spec, operator, task_desc, context)
        return task_id, profile_desc_id, profile_owner_id, stage_name
    if task_desc.task_type == TaskType.TASK_TERMINATE:
        profile_desc_id = 0x10006
        return task_id, profile_desc_id, INVALID_PROFILE_OWNER_ID, DEFAULT_STAGE_NAMES[profile_desc_id]
    return task_id, INVALID_PROFILE_DESC_ID, INVALID_PROFILE_OWNER_ID, None


def _apply_mega_kernel_profile_graph(
    runtime_config: RuntimeConfigC,
    graph: ComputeGraph,
    spec: _ProfileSpec,
    *,
    context: Any = None,
) -> None:
    """Derive per-task profiling metadata from the graph that created RuntimeConfig."""
    _validate_runtime_config(runtime_config)
    if not isinstance(graph, ComputeGraph):
        raise TypeError(f"graph must be ComputeGraph, got {type(graph).__name__}")
    if not isinstance(spec, _ProfileSpec):
        raise TypeError(f"spec must be _ProfileSpec, got {type(spec).__name__}")

    task_count = int(runtime_config.task_num)
    if not 0 <= task_count <= len(runtime_config.all_tasks):
        raise ValueError(f"profile task count is outside RuntimeConfig capacity: {task_count}")
    task_ids = set(range(task_count))
    task_ids.update(_scheduled_task_ids(runtime_config))
    stage_names, task_operators, graph_task_count = _graph_profile_stages(graph)
    if graph_task_count > task_count:
        raise ValueError(
            f"graph describes {graph_task_count} tasks but RuntimeConfig.task_num is only {task_count}"
        )

    resolved_tasks = [
        _resolve_profile_task(task_id, runtime_config.all_tasks[task_id], task_operators, spec, context)
        for task_id in sorted(task_ids)
    ]

    _set_mega_kernel_profile_metadata(
        runtime_config,
        kernel_name=spec.kernel_name,
        owner_label=spec.owner_label,
        stage_names=stage_names,
        task_stage_names={
            task_id: stage_name for task_id, _, _, stage_name in resolved_tasks if stage_name is not None
        },
    )
    for task_id, profile_desc_id, profile_owner_id, _ in resolved_tasks:
        task_desc = runtime_config.all_tasks[task_id]
        task_desc.profile_desc_id = profile_desc_id
        task_desc.profile_owner_id = profile_owner_id


def _get_mega_kernel_profile_metadata(
    runtime_config: RuntimeConfigC,
) -> _ProfileMetadata:
    metadata = getattr(runtime_config, _PROFILE_METADATA_ATTRIBUTE, None)
    if metadata is not None:
        return metadata
    return _ProfileMetadata(
        kernel_name="MegaKernel",
        owner_label="Owner",
        stage_names=_resolve_stage_names(None),
        task_stage_names={},
    )


class _PreparedMegaKernelRuntime:
    """Own disabled/profiled RuntimeConfig variants for one MegaKernel direction."""

    def __init__(
        self,
        runtime_config: RuntimeConfigC,
        *,
        tensor_factory: Callable[[bytes], Any],
        profile_tensor_factory: Callable[[Any], Any],
        rank: int,
        device_id: int,
    ) -> None:
        """Prepare immutable metadata and the default disabled Device tensor."""
        _validate_runtime_config(runtime_config)
        if not callable(tensor_factory):
            raise TypeError("tensor_factory must be callable")
        if not callable(profile_tensor_factory):
            raise TypeError("profile_tensor_factory must be callable")
        self._profile_tensor_factory = profile_tensor_factory
        self._rank = rank
        self._device_id = device_id
        self._metadata = _get_mega_kernel_profile_metadata(runtime_config)
        self._layout = _configure_profile_layout(runtime_config)
        runtime_config.cycle_profiling_enabled = 0
        self._normal_tensor = tensor_factory(serialize_runtime_config(runtime_config))
        self._profile_tensor = None
        self._profile_tensor_lock = Lock()

    @property
    def normal_tensor(self) -> Any:
        """Return the disabled RuntimeConfig tensor used by the fast path."""
        return self._normal_tensor

    @property
    def profile_tensor(self) -> Any:
        """Lazily materialize the enabled RuntimeConfig tensor."""
        if self._profile_tensor is not None:
            return self._profile_tensor
        with self._profile_tensor_lock:
            if self._profile_tensor is None:
                self._profile_tensor = self._profile_tensor_factory(self._normal_tensor)
        return self._profile_tensor

    @property
    def buffer_size(self) -> int:
        """Return the exact ordinary Device buffer size required while profiling."""
        return self._layout.buffer_size

    @property
    def rank(self) -> int:
        """Return the distributed rank that owns this runtime."""
        return self._rank

    @property
    def device_id(self) -> int:
        """Return the local device index that executes this runtime."""
        return self._device_id

    @property
    def kernel_name(self) -> str:
        """Return the concrete kernel display name."""
        return self._metadata.kernel_name

    def parse(self, buffer: Any, *, detailed_task_names: bool) -> dict[str, Any]:
        """Decode one completed invocation buffer.

        Args:
            buffer: Host-visible bytes copied from the profiling buffer.
            detailed_task_names: Whether event names include owner and task details.
        """
        soc_name = _get_soc_name(self._device_id)
        return _parse_cycle_buffer(
            buffer,
            _CycleTraceConfig(
                rank=self._rank,
                device_id=self._device_id,
                cycle_frequency_mhz=_resolve_cycle_frequency_mhz(soc_name),
                detailed_task_names=detailed_task_names,
                kernel_name=self._metadata.kernel_name,
                owner_label=self._metadata.owner_label,
                stage_names=self._metadata.stage_names,
                soc_name=soc_name,
                task_stage_names=self._metadata.task_stage_names,
            ),
            aic_record_capacity=self._layout.aic_record_capacity,
            aiv_record_capacity=self._layout.aiv_record_capacity,
        )


def _prepare_mega_kernel_runtime_config(
    runtime_config: RuntimeConfigC,
    *,
    tensor_factory: Callable[[bytes], Any],
    profile_tensor_factory: Callable[[Any], Any],
    rank: int,
    device_id: int,
) -> _PreparedMegaKernelRuntime:
    """Prepare fast/profiled RuntimeConfig variants without exposing ABI details."""
    return _PreparedMegaKernelRuntime(
        runtime_config,
        tensor_factory=tensor_factory,
        profile_tensor_factory=profile_tensor_factory,
        rank=rank,
        device_id=device_id,
    )


def _as_bytes(buffer: Any) -> bytes:
    """Convert an array-like profile buffer to validated immutable bytes."""
    if hasattr(buffer, "tobytes"):
        raw_buffer = buffer.tobytes()
    else:
        raw_buffer = bytes(buffer)
    minimum_buffer_bytes = CORE_SLOT_COUNT * CORE_HEADER.size
    if len(raw_buffer) < minimum_buffer_bytes:
        raise ValueError(
            f"MegaKernel profile buffer is too small: expected at least {minimum_buffer_bytes} bytes, "
            f"got {len(raw_buffer)}"
        )
    return raw_buffer


def _validate_record_capacity(record_capacity: int, core_name: str) -> None:
    if record_capacity > MAX_RECORDS_PER_CORE:
        raise ValueError(
            f"{core_name} profile record capacity exceeds the {MAX_RECORDS_PER_CORE}-record limit: "
            f"{record_capacity}"
        )
    if record_capacity % RECORD_CAPACITY_ALIGNMENT != 0:
        raise ValueError(
            f"{core_name} profile record capacity must be a multiple of {RECORD_CAPACITY_ALIGNMENT}: "
            f"{record_capacity}"
        )


def _profile_layout_from_buffer(raw_buffer: bytes) -> tuple[int, int, int]:
    """Read AIC/AIV capacities from the first initialized header of each slot region."""
    aic_header = CORE_HEADER.unpack_from(raw_buffer, 0)
    aic_record_capacity = aic_header[5]
    _validate_record_capacity(aic_record_capacity, "AIC")
    aic_stride = CORE_HEADER.size + aic_record_capacity * PROFILE_RECORD.size
    aiv_region_offset = CUBE_SLOT_COUNT * aic_stride
    if len(raw_buffer) < aiv_region_offset + CORE_HEADER.size:
        raise ValueError(
            f"MegaKernel profile buffer is too small for the AIV header: "
            f"expected at least {aiv_region_offset + CORE_HEADER.size} bytes, got {len(raw_buffer)}"
        )
    aiv_header = CORE_HEADER.unpack_from(raw_buffer, aiv_region_offset)
    aiv_record_capacity = aiv_header[5]
    _validate_record_capacity(aiv_record_capacity, "AIV")
    required_buffer_bytes = _profile_buffer_bytes_for_capacities(
        aic_record_capacity,
        aiv_record_capacity,
    )
    if len(raw_buffer) < required_buffer_bytes:
        raise ValueError(
            f"MegaKernel profile buffer is too small for its declared capacities: "
            f"expected at least {required_buffer_bytes} bytes, got {len(raw_buffer)}"
        )
    return aic_record_capacity, aiv_record_capacity, required_buffer_bytes


def _profile_slot_layouts(aic_record_capacity: int, aiv_record_capacity: int):
    aic_stride = CORE_HEADER.size + aic_record_capacity * PROFILE_RECORD.size
    aiv_stride = CORE_HEADER.size + aiv_record_capacity * PROFILE_RECORD.size
    for block_id in range(CUBE_SLOT_COUNT):
        yield 1, block_id, block_id * aic_stride, aic_record_capacity
    aiv_region_offset = CUBE_SLOT_COUNT * aic_stride
    for block_id in range(VECTOR_SLOT_COUNT):
        yield 2, block_id, aiv_region_offset + block_id * aiv_stride, aiv_record_capacity


def _thread_id(core_type: int, block_id: int) -> int:
    return core_type * 1000 + block_id


def _event_name(
    desc_id: int,
    task_id: int,
    stage_task_index: int,
    owner_id: int,
    detailed_task_names: bool,
    owner_label: str,
    stage_names: Mapping[int, str],
    task_stage_names: Mapping[int, str],
) -> str:
    """Build a stage-only or detailed task name while retaining raw IDs in event args."""
    if desc_id in stage_names:
        stage_name = stage_names[desc_id]
    elif desc_id >= TASK_TYPE_DESC_BASE:
        task_type = desc_id - TASK_TYPE_DESC_BASE
        stage_name = f"TaskType_{TASK_TYPE_NAMES.get(task_type, task_type)}"
    else:
        stage_name = f"DescId_{desc_id}"
    task_stage_name = task_stage_names.get(task_id)
    if desc_id in (0x10000, 0x10007) and task_stage_name is not None:
        stage_name = f"{task_stage_name}_{stage_name}"
    if not detailed_task_names:
        return stage_name
    task_name = f"{stage_name}_task{stage_task_index + 1}"
    if owner_id == INVALID_OWNER_ID:
        return task_name
    return f"{owner_label}{owner_id}_{task_name}"


def _resolve_buffer_layout(
    raw_buffer: bytes,
    aic_record_capacity: int | None,
    aiv_record_capacity: int | None,
) -> tuple[int, int, int]:
    """Resolve and validate the two per-core capacities and total buffer size."""
    if aic_record_capacity is None and aiv_record_capacity is None:
        return _profile_layout_from_buffer(raw_buffer)
    if aic_record_capacity is None or aiv_record_capacity is None:
        raise ValueError("AIC and AIV profile capacities must be provided together")
    _validate_record_capacity(aic_record_capacity, "AIC")
    _validate_record_capacity(aiv_record_capacity, "AIV")
    required_buffer_bytes = _profile_buffer_bytes_for_capacities(
        aic_record_capacity,
        aiv_record_capacity,
    )
    if len(raw_buffer) < required_buffer_bytes:
        raise ValueError(
            "MegaKernel profile buffer is smaller than its Host-computed layout: "
            f"expected at least {required_buffer_bytes} bytes, got {len(raw_buffer)}"
        )
    return aic_record_capacity, aiv_record_capacity, required_buffer_bytes


def _validate_profile_slot_header(
    slot: int,
    core_type: int,
    block_id: int,
    expected_core_type: int,
    expected_block_id: int,
    header_capacity: int,
    record_capacity: int,
    record_count: int,
    reserved: int,
) -> None:
    """Validate the identity, layout, and count fields of one Device slot."""
    if core_type != expected_core_type or block_id != expected_block_id:
        raise ValueError(
            f"Profile slot identity mismatch: slot={slot}, expected=({expected_core_type}, {expected_block_id}), "
            f"got=({core_type}, {block_id})"
        )
    if header_capacity != record_capacity:
        raise ValueError(
            f"Profile slot capacity mismatch: slot={slot}, expected={record_capacity}, got={header_capacity}"
        )
    if reserved != 0:
        raise ValueError(f"Profile core header reserved field must be zero: slot={slot}, got={reserved}")
    if record_count > record_capacity:
        raise ValueError(
            f"Profile record count exceeds capacity: slot={slot}, count={record_count}, "
            f"capacity={record_capacity}"
        )


def _decode_slot_records(
    raw_buffer: bytes,
    slot: int,
    slot_offset: int,
    record_count: int,
    core_type: int,
    block_id: int,
    entry_cycle: int,
) -> list[dict[str, int]]:
    """Decode all cycle records stored after one validated core header."""
    records = []
    record_offset = slot_offset + CORE_HEADER.size
    for record_index in range(record_count):
        record = PROFILE_RECORD.unpack_from(raw_buffer, record_offset + record_index * PROFILE_RECORD.size)
        if record[1] < record[0]:
            raise ValueError(
                f"Profile cycle interval is negative: slot={slot}, record={record_index}, "
                f"start={record[0]}, end={record[1]}"
            )
        records.append(
            {
                "start_cycle": record[0],
                "end_cycle": record[1],
                "desc_id": record[2],
                "task_id": record[3],
                "stage_task_index": record[4],
                "owner_id": record[5],
                "core_type": core_type,
                "block_id": block_id,
                "entry_cycle": entry_cycle,
            }
        )
    return records


def _decode_profile_slot(
    raw_buffer: bytes,
    slot: int,
    layout: tuple[int, int, int, int],
) -> tuple[list[dict[str, int]], int | None, int]:
    """Decode one initialized slot and return records, entry cycle, and drops."""
    expected_core_type, expected_block_id, slot_offset, record_capacity = layout
    header = CORE_HEADER.unpack_from(raw_buffer, slot_offset)
    if not any(header):
        return [], None, 0
    entry_cycle, record_count, dropped_count, core_type, block_id, header_capacity, reserved = header
    _validate_profile_slot_header(
        slot,
        core_type,
        block_id,
        expected_core_type,
        expected_block_id,
        header_capacity,
        record_capacity,
        record_count,
        reserved,
    )
    if record_count == 0:
        return [], None, dropped_count
    records = _decode_slot_records(
        raw_buffer,
        slot,
        slot_offset,
        record_count,
        core_type,
        block_id,
        entry_cycle,
    )
    return records, entry_cycle or None, dropped_count


def _decode_cycle_records(
    raw_buffer: bytes,
    aic_record_capacity: int,
    aiv_record_capacity: int,
) -> tuple[list[dict[str, int]], list[int], int]:
    """Decode all initialized AIC/AIV slots and aggregate drop counters."""
    raw_records = []
    active_entry_cycles = []
    dropped_records = 0
    for slot, layout in enumerate(_profile_slot_layouts(aic_record_capacity, aiv_record_capacity)):
        records, entry_cycle, dropped_count = _decode_profile_slot(raw_buffer, slot, layout)
        dropped_records += dropped_count
        if entry_cycle is not None:
            active_entry_cycles.append(entry_cycle)
        raw_records.extend(records)
    return raw_records, active_entry_cycles, dropped_records


def _thread_metadata_events(
    rank: int,
    device_id: int,
    kernel_name: str,
    raw_records: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Build process and thread metadata for all cores that emitted records."""
    trace_events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": rank,
            "tid": 0,
            "args": {"name": f"{kernel_name} rank {rank} device {device_id}"},
        }
    ]
    used_threads = sorted({(record["core_type"], record["block_id"]) for record in raw_records})
    for sort_index, (core_type, block_id) in enumerate(used_threads):
        thread_id = _thread_id(core_type, block_id)
        trace_events.extend(
            [
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": rank,
                    "tid": thread_id,
                    "args": {"name": f"{CORE_TYPE_NAMES[core_type]}/{block_id}"},
                },
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": rank,
                    "tid": thread_id,
                    "args": {"sort_index": sort_index},
                },
            ]
        )
    return trace_events


def _duration_trace_event(
    record: Mapping[str, int],
    *,
    rank: int,
    device_id: int,
    anchor_cycle: int,
    cycle_frequency_mhz: float,
    detailed_task_names: bool,
    owner_label: str,
    stage_names: Mapping[int, str],
    task_stage_names: Mapping[int, str],
) -> dict[str, Any]:
    """Convert one raw cycle record into a Chrome Trace duration event."""
    desc_id = record["desc_id"]
    task_id = record["task_id"]
    args = {
        "rank": rank,
        "device_id": device_id,
        "core_type": CORE_TYPE_NAMES[record["core_type"]],
        "block_id": record["block_id"],
        "task_id": task_id,
        "desc_id": desc_id,
        "stage_task_index": record["stage_task_index"],
        "task_number": record["stage_task_index"] + 1,
        "start_cycle": record["start_cycle"],
        "end_cycle": record["end_cycle"],
        "core_entry_cycle": record["entry_cycle"],
    }
    if task_id in task_stage_names:
        args["task_stage"] = task_stage_names[task_id]
    if desc_id >= TASK_TYPE_DESC_BASE:
        args["task_type"] = desc_id - TASK_TYPE_DESC_BASE
    if record["owner_id"] != INVALID_OWNER_ID:
        args["owner_id"] = record["owner_id"]
    return {
        "name": _event_name(
            desc_id,
            task_id,
            record["stage_task_index"],
            record["owner_id"],
            detailed_task_names,
            owner_label,
            stage_names,
            task_stage_names,
        ),
        "cat": "MegaKernelInternal",
        "ph": "X",
        "pid": rank,
        "tid": _thread_id(record["core_type"], record["block_id"]),
        "ts": (record["start_cycle"] - anchor_cycle) / cycle_frequency_mhz,
        "dur": (record["end_cycle"] - record["start_cycle"]) / cycle_frequency_mhz,
        "args": args,
    }


def _parse_cycle_buffer(
    buffer: Any,
    config: _CycleTraceConfig,
    *,
    aic_record_capacity: int | None = None,
    aiv_record_capacity: int | None = None,
) -> dict[str, Any]:
    """Decode a device cycle buffer with already resolved hardware and display metadata."""
    if config.cycle_frequency_mhz <= 0:
        raise ValueError(f"cycle_frequency_mhz must be positive, got {config.cycle_frequency_mhz}")
    resolved_stage_names = _resolve_stage_names(config.stage_names)
    resolved_task_stage_names = dict(config.task_stage_names or {})
    raw_buffer = _as_bytes(buffer)
    aic_record_capacity, aiv_record_capacity, required_buffer_bytes = _resolve_buffer_layout(
        raw_buffer,
        aic_record_capacity,
        aiv_record_capacity,
    )
    raw_records, active_entry_cycles, dropped_records = _decode_cycle_records(
        raw_buffer,
        aic_record_capacity,
        aiv_record_capacity,
    )
    if not raw_records:
        raise ValueError(
            "MegaKernel cycle profile buffer contains no records; verify that the current profiler schedule is active "
            "and that the profiled operator completed"
        )
    anchor_cycle = (
        min(active_entry_cycles) if active_entry_cycles else min(record["start_cycle"] for record in raw_records)
    )
    resolved_device_id = config.rank if config.device_id is None else config.device_id
    trace_events = _thread_metadata_events(config.rank, resolved_device_id, config.kernel_name, raw_records)
    sorted_records = sorted(
        raw_records,
        key=lambda item: (item["start_cycle"], item["core_type"], item["block_id"]),
    )
    for record in sorted_records:
        trace_events.append(_duration_trace_event(
            record,
            rank=config.rank,
            device_id=resolved_device_id,
            anchor_cycle=anchor_cycle,
            cycle_frequency_mhz=config.cycle_frequency_mhz,
            detailed_task_names=config.detailed_task_names,
            owner_label=config.owner_label,
            stage_names=resolved_stage_names,
            task_stage_names=resolved_task_stage_names,
        ))
    return {
        "traceEvents": trace_events,
        "megaKernelCycleTrace": {
            "schemaVersion": 1,
            "rank": config.rank,
            "deviceId": resolved_device_id,
            "socName": config.soc_name,
            "cycleFrequencyMHz": config.cycle_frequency_mhz,
            "anchorCycle": anchor_cycle,
            "recordCapacityPerCore": {
                "AIC": aic_record_capacity,
                "AIV": aiv_record_capacity,
            },
            "maxRecordCapacityPerCore": MAX_RECORDS_PER_CORE,
            "recordCount": len(raw_records),
            "droppedRecordCount": dropped_records,
            "profileBufferBytes": required_buffer_bytes,
            "detailedTaskNames": config.detailed_task_names,
            "kernelName": config.kernel_name,
            "ownerLabel": config.owner_label,
            "warnings": ([] if dropped_records == 0 else [f"Device dropped {dropped_records} cycle trace records"]),
        },
    }

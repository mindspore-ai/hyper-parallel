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
    MAX_TASK_NUM,
    NUM_WORKERS_CUBE,
    NUM_WORKERS_VECTOR,
    RuntimeConfigC,
    TaskDescC,
    TaskType,
)


__all__ = []


EVENT_COUNTER_BYTES = 4096
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
class _ProfileStage:
    """Declarative Host rule for one concrete MegaKernel stage."""

    desc_id: int
    name: str
    task_type: int | None = None
    tiling_data_position: int | None = None
    input_position: int | None = None
    matcher: Callable[[TaskDescC, Any], bool] | None = None
    owner_resolver: Callable[[TaskDescC, Any], int] | None = None


@dataclass(frozen=True)
class _ProfileSpec:
    """Concrete MegaKernel profiling semantics applied after RuntimeConfig construction."""

    kernel_name: str
    owner_label: str
    stages: tuple[_ProfileStage, ...]


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
    if scheduled_task_count < 0 or scheduled_task_count > len(task_indices):
        raise ValueError(f"scheduled task count is outside RuntimeConfig capacity: {scheduled_task_count}")
    if scheduled_task_count == 0:
        return 0
    if active_worker_count <= 0:
        raise ValueError("RuntimeConfig.num_workers must identify at least one AIC/AIV worker")

    worker_record_counts = [0] * active_worker_count
    for schedule_index in range(scheduled_task_count):
        task_id = int(task_indices[schedule_index])
        if not 0 <= task_id < MAX_TASK_NUM:
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


def _profile_stage_matches(stage: _ProfileStage, task_desc: TaskDescC, context: Any) -> bool:
    """Match the common TaskDesc selectors before an optional kernel-specific predicate."""
    if stage.task_type is not None and task_desc.task_type != stage.task_type:
        return False
    if stage.tiling_data_position is not None and task_desc.tiling_data_position != stage.tiling_data_position:
        return False
    if stage.input_position is not None and task_desc.inputs[0].input_position != stage.input_position:
        return False
    return stage.matcher is None or stage.matcher(task_desc, context)


def _apply_mega_kernel_profile_spec(
    runtime_config: RuntimeConfigC,
    spec: _ProfileSpec,
    *,
    context: Any = None,
) -> None:
    """Resolve per-task profile IDs and attach display metadata from one Host specification."""
    _validate_runtime_config(runtime_config)
    if not isinstance(spec, _ProfileSpec):
        raise TypeError(f"spec must be _ProfileSpec, got {type(spec).__name__}")

    stage_names = {}
    for stage in spec.stages:
        if isinstance(stage.desc_id, bool) or not isinstance(stage.desc_id, int):
            raise TypeError(f"profile desc_id must be an integer, got {stage.desc_id!r}")
        if not 0 <= stage.desc_id < INVALID_PROFILE_DESC_ID:
            raise ValueError(f"profile desc_id is outside the valid uint32 range: {stage.desc_id}")
        if stage.desc_id in stage_names:
            raise ValueError(f"duplicate profile desc_id in spec: {stage.desc_id:#x}")
        if (
            stage.task_type is None
            and stage.tiling_data_position is None
            and stage.input_position is None
            and stage.matcher is None
        ):
            raise ValueError(f"profile stage {stage.name!r} has no TaskDesc selector")
        stage_names[stage.desc_id] = stage.name

    task_count = int(runtime_config.task_num)
    if not 0 <= task_count <= MAX_TASK_NUM:
        raise ValueError(f"profile task count is outside RuntimeConfig capacity: {task_count}")
    task_ids = set(range(task_count))
    task_schedules = (
        (runtime_config.cube_task_indices, int(runtime_config.task_index_num[0])),
        (runtime_config.vector_task_indices, int(runtime_config.task_index_num[1])),
        (runtime_config.mix_task_indices, int(runtime_config.task_index_num[2])),
    )
    for task_indices, scheduled_task_count in task_schedules:
        if not 0 <= scheduled_task_count <= len(task_indices):
            raise ValueError("scheduled task count is outside RuntimeConfig capacity: " f"{scheduled_task_count}")
        for schedule_index in range(scheduled_task_count):
            task_id = int(task_indices[schedule_index])
            if not 0 <= task_id < MAX_TASK_NUM:
                raise ValueError(f"scheduled task ID is outside RuntimeConfig capacity: {task_id}")
            task_ids.add(task_id)

    resolved_tasks = []
    for task_id in sorted(task_ids):
        task_desc = runtime_config.all_tasks[task_id]
        matched_stage = None
        for stage in spec.stages:
            if not _profile_stage_matches(stage, task_desc, context):
                continue
            if matched_stage is not None:
                raise ValueError(
                    f"task {task_id} matches multiple profile stages: " f"{matched_stage.name!r} and {stage.name!r}"
                )
            matched_stage = stage

        profile_desc_id = INVALID_PROFILE_DESC_ID
        profile_owner_id = INVALID_PROFILE_OWNER_ID
        if matched_stage is not None:
            profile_desc_id = matched_stage.desc_id
            if matched_stage.owner_resolver is not None:
                profile_owner_id = matched_stage.owner_resolver(task_desc, context)
                if isinstance(profile_owner_id, bool) or not isinstance(profile_owner_id, int):
                    raise TypeError(f"profile owner_id must be an integer, got {profile_owner_id!r}")
                if not 0 <= profile_owner_id < INVALID_PROFILE_OWNER_ID:
                    raise ValueError(f"profile owner_id is outside uint32 range: {profile_owner_id}")
        stage_name = None if matched_stage is None else matched_stage.name
        resolved_tasks.append((task_id, profile_desc_id, profile_owner_id, stage_name))

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
        self._normal_tensor = tensor_factory(bytes(runtime_config))
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
        """Decode one completed invocation buffer."""
        soc_name = _get_soc_name(self._device_id)
        return _parse_cycle_buffer(
            buffer=buffer,
            rank=self._rank,
            device_id=self._device_id,
            cycle_frequency_mhz=_resolve_cycle_frequency_mhz(soc_name),
            detailed_task_names=detailed_task_names,
            kernel_name=self._metadata.kernel_name,
            owner_label=self._metadata.owner_label,
            stage_names=self._metadata.stage_names,
            soc_name=soc_name,
            task_stage_names=self._metadata.task_stage_names,
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


def _parse_cycle_buffer(
    buffer: Any,
    rank: int,
    device_id: int,
    cycle_frequency_mhz: float,
    detailed_task_names: bool,
    kernel_name: str,
    owner_label: str,
    stage_names: Mapping[int, str],
    soc_name: str,
    task_stage_names: Mapping[int, str] | None = None,
    aic_record_capacity: int | None = None,
    aiv_record_capacity: int | None = None,
) -> dict[str, Any]:
    """Decode a device cycle buffer with already resolved hardware and display metadata."""
    if cycle_frequency_mhz <= 0:
        raise ValueError(f"cycle_frequency_mhz must be positive, got {cycle_frequency_mhz}")
    resolved_stage_names = _resolve_stage_names(stage_names)
    resolved_task_stage_names = dict(task_stage_names or {})
    raw_buffer = _as_bytes(buffer)
    if aic_record_capacity is None and aiv_record_capacity is None:
        aic_record_capacity, aiv_record_capacity, required_buffer_bytes = _profile_layout_from_buffer(raw_buffer)
    elif aic_record_capacity is None or aiv_record_capacity is None:
        raise ValueError("AIC and AIV profile capacities must be provided together")
    else:
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
    resolved_device_id = rank if device_id is None else device_id
    raw_records = []
    active_entry_cycles = []
    dropped_records = 0

    for slot, (
        expected_core_type,
        expected_block_id,
        slot_offset,
        record_capacity,
    ) in enumerate(_profile_slot_layouts(aic_record_capacity, aiv_record_capacity)):
        header = CORE_HEADER.unpack_from(raw_buffer, slot_offset)
        (
            entry_cycle,
            record_count,
            dropped_count,
            core_type,
            block_id,
            header_capacity,
            reserved,
        ) = header
        if not any(header):
            continue
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
        dropped_records += dropped_count
        if record_count == 0:
            continue
        if entry_cycle:
            active_entry_cycles.append(entry_cycle)
        record_offset = slot_offset + CORE_HEADER.size
        for record_index in range(record_count):
            values = PROFILE_RECORD.unpack_from(raw_buffer, record_offset + record_index * PROFILE_RECORD.size)
            start_cycle, end_cycle, desc_id, task_id, stage_task_index, owner_id = values
            if end_cycle < start_cycle:
                raise ValueError(
                    f"Profile cycle interval is negative: slot={slot}, record={record_index}, "
                    f"start={start_cycle}, end={end_cycle}"
                )
            raw_records.append(
                {
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "desc_id": desc_id,
                    "task_id": task_id,
                    "stage_task_index": stage_task_index,
                    "owner_id": owner_id,
                    "core_type": core_type,
                    "block_id": block_id,
                    "entry_cycle": entry_cycle,
                }
            )

    if not raw_records:
        raise ValueError(
            "MegaKernel cycle profile buffer contains no records; verify that the current profiler schedule is active "
            "and that the profiled operator completed"
        )

    anchor_cycle = (
        min(active_entry_cycles) if active_entry_cycles else min(record["start_cycle"] for record in raw_records)
    )
    used_threads = sorted({(record["core_type"], record["block_id"]) for record in raw_records})
    trace_events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": rank,
            "tid": 0,
            "args": {"name": f"{kernel_name} rank {rank} device {resolved_device_id}"},
        }
    ]
    for sort_index, (core_type, block_id) in enumerate(used_threads):
        core_name = CORE_TYPE_NAMES[core_type]
        thread_id = _thread_id(core_type, block_id)
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": rank,
                "tid": thread_id,
                "args": {"name": f"{core_name}/{block_id}"},
            }
        )
        trace_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": rank,
                "tid": thread_id,
                "args": {"sort_index": sort_index},
            }
        )

    for record in sorted(
        raw_records,
        key=lambda item: (item["start_cycle"], item["core_type"], item["block_id"]),
    ):
        desc_id = record["desc_id"]
        task_id = record["task_id"]
        trace_events.append(
            {
                "name": _event_name(
                    desc_id,
                    task_id,
                    record["stage_task_index"],
                    record["owner_id"],
                    detailed_task_names,
                    owner_label,
                    resolved_stage_names,
                    resolved_task_stage_names,
                ),
                "cat": "MegaKernelInternal",
                "ph": "X",
                "pid": rank,
                "tid": _thread_id(record["core_type"], record["block_id"]),
                "ts": (record["start_cycle"] - anchor_cycle) / cycle_frequency_mhz,
                "dur": (record["end_cycle"] - record["start_cycle"]) / cycle_frequency_mhz,
                "args": {
                    "rank": rank,
                    "device_id": resolved_device_id,
                    "core_type": CORE_TYPE_NAMES[record["core_type"]],
                    "block_id": record["block_id"],
                    "task_id": task_id,
                    "desc_id": desc_id,
                    **(
                        {"task_stage": resolved_task_stage_names[task_id]}
                        if task_id in resolved_task_stage_names
                        else {}
                    ),
                    **({"task_type": desc_id - TASK_TYPE_DESC_BASE} if desc_id >= TASK_TYPE_DESC_BASE else {}),
                    "stage_task_index": record["stage_task_index"],
                    "task_number": record["stage_task_index"] + 1,
                    **({} if record["owner_id"] == INVALID_OWNER_ID else {"owner_id": record["owner_id"]}),
                    "start_cycle": record["start_cycle"],
                    "end_cycle": record["end_cycle"],
                    "core_entry_cycle": record["entry_cycle"],
                },
            }
        )

    return {
        "traceEvents": trace_events,
        "megaKernelCycleTrace": {
            "schemaVersion": 1,
            "rank": rank,
            "deviceId": resolved_device_id,
            "socName": soc_name,
            "cycleFrequencyMHz": cycle_frequency_mhz,
            "anchorCycle": anchor_cycle,
            "recordCapacityPerCore": {
                "AIC": aic_record_capacity,
                "AIV": aiv_record_capacity,
            },
            "maxRecordCapacityPerCore": MAX_RECORDS_PER_CORE,
            "recordCount": len(raw_records),
            "droppedRecordCount": dropped_records,
            "profileBufferBytes": required_buffer_bytes,
            "detailedTaskNames": detailed_task_names,
            "kernelName": kernel_name,
            "ownerLabel": owner_label,
            "warnings": ([] if dropped_records == 0 else [f"Device dropped {dropped_records} cycle trace records"]),
        },
    }

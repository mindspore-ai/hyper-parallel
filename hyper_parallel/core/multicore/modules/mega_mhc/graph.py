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
"""Pure AIC/AIV token-tile task graph for shifted Single-Pass HyperMegaMhc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID,
    MIN_EVENT_CAPACITY,
    RuntimeConfigC,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph,
    OperatorNode,
    OpType,
    SplitSpec,
    TensorSpec,
)
from hyper_parallel.core.multicore.tasks.task_base import FillConfig


DEFAULT_TOKEN_TILE = 32
_STAGE1_BS_FACTOR = 32
POST_DONE_EVENT = 0
NORM_EVENT_BASE = 1


@dataclass(frozen=True)
class MegaMhcEventLayout:
    """Contiguous event ranges for one shape-bound pipeline."""

    post_done: int
    norm_base: int
    input_mix_base: int
    projection_base: int
    final: int
    event_count: int


def build_event_layout(tile_count: int) -> MegaMhcEventLayout:
    """Return event IDs for a six-stage, per-token-tile pipeline.

    Args:
        tile_count: Number of logical token tiles.
    """
    if tile_count <= 0:
        raise ValueError(f"tile_count must be positive, got {tile_count}.")
    input_mix_base = NORM_EVENT_BASE + tile_count
    projection_base = input_mix_base + tile_count
    final = projection_base + tile_count
    event_count = final + 1
    if event_count > MIN_EVENT_CAPACITY:
        raise ValueError(
            f"HyperMegaMhc needs {event_count} events for {tile_count} token tiles, "
            f"but RuntimeConfig supports only {MIN_EVENT_CAPACITY}."
        )
    return MegaMhcEventLayout(
        post_done=POST_DONE_EVENT,
        norm_base=NORM_EVENT_BASE,
        input_mix_base=input_mix_base,
        projection_base=projection_base,
        final=final,
        event_count=event_count,
    )


def resolve_token_tile(token_count: int, requested_tile: int, num_cube_cores: int) -> int:
    """Choose an aligned tile that fits event and native Stage1 workspace limits.

    Args:
        token_count: Total flattened token count.
        requested_tile: Preferred token count per task.
        num_cube_cores: Physical AIC core count.
    """
    if token_count <= 0 or requested_tile <= 0 or num_cube_cores <= 0:
        raise ValueError(
            "token_count, requested_tile, and num_cube_cores must all be positive, "
            f"got {(token_count, requested_tile, num_cube_cores)}."
        )
    max_tiles = (MIN_EVENT_CAPACITY - 2) // 3
    minimum_tile = (token_count + max_tiles - 1) // max_tiles
    minimum_tile = (minimum_tile + _STAGE1_BS_FACTOR - 1) // _STAGE1_BS_FACTOR * _STAGE1_BS_FACTOR
    token_tile = max(requested_tile, minimum_tile)
    token_tile = (token_tile + _STAGE1_BS_FACTOR - 1) // _STAGE1_BS_FACTOR * _STAGE1_BS_FACTOR
    stage1_workspace_rows = 4 * num_cube_cores * _STAGE1_BS_FACTOR
    if token_tile > stage1_workspace_rows:
        raise ValueError(
            f"token tile {token_tile} exceeds the native Stage1 X-cast workspace capacity "
            f"of {stage1_workspace_rows} rows."
        )
    return token_tile


def _append_task(
    cfg: RuntimeConfigC,
    tsv: TaskSplitValue,
    *,
    task_type: TaskType,
    core_type: TaskAiCoreType,
    dependent_event: int,
    trigger_event: int,
    task_index: int,
    task_num: int,
    split_value: int,
) -> None:
    """Append one task to the matching static AIC or AIV schedule."""
    task = TaskDescC()
    task.task_type = task_type
    task.task_aicore_type = core_type
    task.dependent_event = dependent_event
    task.trigger_event = trigger_event
    task.task_index = task_index
    task.task_split_num = task_num
    task.task_split_value = split_value
    task_id = tsv.pre_task_num
    cfg.all_tasks[task_id] = task
    if core_type == TaskAiCoreType.TASK_AICORE_CUBE:
        cfg.cube_task_indices[tsv.pre_cube_task_num] = task_id
        cfg.task_index_num[0] += 1
        tsv.pre_cube_task_num += 1
    elif core_type == TaskAiCoreType.TASK_AICORE_VECTOR:
        cfg.vector_task_indices[tsv.pre_vector_task_num] = task_id
        cfg.task_index_num[1] += 1
        tsv.pre_vector_task_num += 1
    else:
        raise ValueError(f"HyperMegaMhc task must be pure AIC or AIV, got {core_type!r}.")
    tsv.pre_task_num += 1


_AIV_PIPELINE_TYPES = (
    TaskType.TASK_MHC_NORM_CAST,
    TaskType.TASK_MHC_INPUT_MIX,
    TaskType.TASK_RMS_NORM,
    TaskType.TASK_MHC_MAPPING,
)


def _reorder_vector_task_waves(
    cfg: RuntimeConfigC,
    tsv: TaskSplitValue,
    tile_count: int,
    wave_size: int,
) -> None:
    """Fill Projection gaps with NormCast/InputMix/RMSNorm waves before Mapping."""
    if wave_size <= 0:
        raise ValueError(f"wave_size must be positive, got {wave_size}.")
    post_task_ids = []
    pipeline_task_ids = {task_type: [] for task_type in _AIV_PIPELINE_TYPES}
    for schedule_index in range(tsv.pre_vector_task_num):
        task_id = cfg.vector_task_indices[schedule_index]
        task_type = TaskType(cfg.all_tasks[task_id].task_type)
        if task_type == TaskType.TASK_MHC_POST:
            post_task_ids.append(task_id)
        elif task_type in pipeline_task_ids:
            pipeline_task_ids[task_type].append(task_id)
        else:
            raise ValueError(f"Unexpected AIV task type in HyperMegaMhc schedule: {task_type!r}.")

    if len(post_task_ids) != tile_count:
        raise ValueError(f"Expected {tile_count} MhcPost tasks, got {len(post_task_ids)}.")
    for task_type, task_ids in pipeline_task_ids.items():
        if len(task_ids) != tile_count:
            raise ValueError(f"Expected {tile_count} {task_type.name} tasks, got {len(task_ids)}.")

    reordered_task_ids = list(post_task_ids)
    for wave_start in range(0, tile_count, wave_size):
        wave_end = min(wave_start + wave_size, tile_count)
        for task_type in _AIV_PIPELINE_TYPES[:3]:
            reordered_task_ids.extend(pipeline_task_ids[task_type][wave_start:wave_end])
    reordered_task_ids.extend(pipeline_task_ids[TaskType.TASK_MHC_MAPPING])
    for schedule_index, task_id in enumerate(reordered_task_ids):
        cfg.vector_task_indices[schedule_index] = task_id


@dataclass(frozen=True)
class _PostFillConfig(FillConfig):
    """Emit token-local residual update tasks on AIV."""

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append the MhcPost task slice to the runtime configuration.

        Args:
            cfg: Runtime configuration receiving tasks.
            op: Scheduled operator node.
            tsv: Mutable task split counters.
        """
        layout = build_event_layout(op.task_num)
        cfg.all_event_num_triggers[layout.post_done] = op.task_num
        for task_index in range(op.task_num):
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_POST,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=EVENT_INVALID_ID,
                trigger_event=layout.post_done,
                task_index=task_index,
                task_num=op.task_num,
                split_value=op.split_value,
            )


@dataclass(frozen=True)
class _NormCastFillConfig(FillConfig):
    """Emit AIV normalization/cast tasks with safe X-cast ring-slot reuse."""

    ring_slots: int

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append NormCast tasks and ring-buffer reuse dependencies.

        Args:
            cfg: Runtime configuration receiving tasks.
            op: Scheduled operator node.
            tsv: Mutable task split counters.
        """
        layout = build_event_layout(op.task_num)
        for task_index in range(op.task_num):
            dependent_event = layout.post_done
            if task_index >= self.ring_slots:
                dependent_event = layout.projection_base + task_index - self.ring_slots
            trigger_event = layout.norm_base + task_index
            cfg.all_event_num_triggers[trigger_event] = 1
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_NORM_CAST,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=dependent_event,
                trigger_event=trigger_event,
                task_index=task_index,
                task_num=op.task_num,
                split_value=op.split_value,
            )


@dataclass(frozen=True)
class _ProjectionFillConfig(FillConfig):
    """Emit projection tasks on AIC."""

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append token-tiled AIC projection tasks.

        Args:
            cfg: Runtime configuration receiving tasks.
            op: Scheduled operator node.
            tsv: Mutable task split counters.
        """
        layout = build_event_layout(op.task_num)
        for task_index in range(op.task_num):
            trigger_event = layout.projection_base + task_index
            cfg.all_event_num_triggers[trigger_event] = 1
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_PROJECTION,
                core_type=TaskAiCoreType.TASK_AICORE_CUBE,
                dependent_event=layout.norm_base + task_index,
                trigger_event=trigger_event,
                task_index=task_index,
                task_num=op.task_num,
                split_value=op.split_value,
            )


@dataclass(frozen=True)
class _MappingFillConfig(FillConfig):
    """Emit per-tile mapping/Sinkhorn tasks after the corresponding Projection."""

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append mapping and Sinkhorn tasks after their projection tiles.

        Args:
            cfg: Runtime configuration receiving tasks.
            op: Scheduled operator node.
            tsv: Mutable task split counters.
        """
        layout = build_event_layout(op.task_num)
        cfg.all_event_num_triggers[layout.final] = 2 * op.task_num
        for task_index in range(op.task_num):
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_MAPPING,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=layout.projection_base + task_index,
                trigger_event=layout.final,
                task_index=task_index,
                task_num=op.task_num,
                split_value=op.split_value,
            )


@dataclass(frozen=True)
class _InputMixFillConfig(FillConfig):
    """Emit input-mix tasks after NormCast to fill the Projection AIV gap."""

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append shifted input-mix tasks after the matching NormCast tiles.

        Args:
            cfg: Runtime configuration receiving tasks.
            op: Scheduled operator node.
            tsv: Mutable task split counters.
        """
        layout = build_event_layout(op.task_num)
        for task_index in range(op.task_num):
            trigger_event = layout.input_mix_base + task_index
            cfg.all_event_num_triggers[trigger_event] = 1
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_INPUT_MIX,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=layout.norm_base + task_index,
                trigger_event=trigger_event,
                task_index=task_index,
                task_num=op.task_num,
                split_value=op.split_value,
            )


@dataclass(frozen=True)
class _RmsNormFillConfig(FillConfig):
    """Emit shifted input RMSNorm tasks on AIV."""

    wave_size: int

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append shifted RMSNorm tasks and arrange the AIV task waves.

        Args:
            cfg: Runtime configuration receiving tasks.
            op: Scheduled operator node.
            tsv: Mutable task split counters.
        """
        layout = build_event_layout(op.task_num)
        for task_index in range(op.task_num):
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_RMS_NORM,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=layout.input_mix_base + task_index,
                trigger_event=layout.final,
                task_index=task_index,
                task_num=op.task_num,
                split_value=op.split_value,
            )
        _reorder_vector_task_waves(cfg, tsv, op.task_num, self.wave_size)


def _assemble_forward_graph(
    topology: TaskSplitValue,
    post: OperatorNode,
    norm_cast: OperatorNode,
    projection: OperatorNode,
    input_mix: OperatorNode,
    rms_norm: OperatorNode,
    mapping: OperatorNode,
) -> ComputeGraph:
    """Connect the forward stages and resolve their task splits."""
    graph = ComputeGraph()
    for operator in (post, norm_cast, projection, input_mix, rms_norm, mapping):
        graph.add_op(operator)
    graph.add_edge(post, norm_cast).add_edge(norm_cast, projection).add_edge(projection, mapping)
    graph.add_edge(norm_cast, input_mix).add_edge(input_mix, rms_norm)
    graph.propagate_splits(topology)
    return graph


def _make_post_and_norm_cast(
    residual: TensorSpec,
    updated_residual: TensorSpec,
    normalized: TensorSpec,
    token_tile: int,
    task_num_fn: Callable[[TaskSplitValue], int],
    ring_slots: int,
) -> tuple[OperatorNode, OperatorNode]:
    """Create the two opening AIV stages."""
    post = OperatorNode(
        name="mhc_post",
        op_type=OpType.MHC_POST,
        inputs=[residual],
        outputs=[updated_residual],
        param_positions=[1, 12],
        split_value=token_tile,
        split_spec=SplitSpec(None, task_num_fn, [0]),
        tiling_position=18,
        fill_config=_PostFillConfig(),
        diagnostic_name="MhcPost",
    )
    norm_cast = OperatorNode(
        name="mhc_norm_cast",
        op_type=OpType.MHC_NORM_CAST,
        inputs=[updated_residual],
        outputs=[normalized],
        param_positions=[12, 17],
        split_value=token_tile,
        split_spec=SplitSpec(None, task_num_fn, [0]),
        tiling_position=18,
        fill_config=_NormCastFillConfig(ring_slots),
        diagnostic_name="MhcNormCast",
    )
    return post, norm_cast


def build_mega_mhc_graph(
    token_count: int,
    hidden_size: int,
    num_cube_cores: int,
    token_tile: int = DEFAULT_TOKEN_TILE,
) -> tuple[ComputeGraph, TaskSplitValue]:
    """Build the overlapped Projection and shifted InputMix/RMSNorm token pipeline.

    Args:
        token_count: Total flattened token count.
        hidden_size: Hidden dimension.
        num_cube_cores: Physical AIC core count.
        token_tile: Preferred token count per task.
    """
    if token_count <= 0 or hidden_size <= 0 or num_cube_cores <= 0:
        raise ValueError(
            "token_count, hidden_size, and num_cube_cores must all be positive, "
            f"got {(token_count, hidden_size, num_cube_cores)}."
        )
    token_tile = resolve_token_tile(token_count, token_tile, num_cube_cores)
    tile_count = (token_count + token_tile - 1) // token_tile
    build_event_layout(tile_count)
    stage1_workspace_rows = 4 * num_cube_cores * _STAGE1_BS_FACTOR
    ring_slots = min(tile_count, stage1_workspace_rows // token_tile)

    topology = TaskSplitValue(tp=1, ep=1, seq_size=token_count, all_expert_num=1, top_k=1)
    residual = TensorSpec("residual", [token_count, 4, hidden_size], 1)
    updated_residual = TensorSpec("updated_residual", [token_count, 4, hidden_size], 12)
    normalized = TensorSpec("normalized", [token_tile, 4, hidden_size], 17, dtype_size=4)
    projection = TensorSpec("projection", [token_count, 24], 17, dtype_size=4)
    mappings = TensorSpec("mappings", [token_count, 24], 13, dtype_size=4)
    mixed_input = TensorSpec("mixed_input", [token_count, hidden_size], 17)
    block_input = TensorSpec("block_input", [token_count, hidden_size], 16)

    def task_num_fn(_tsv: TaskSplitValue) -> int:
        """Return the token tile count shared by all forward stages.

        Args:
            _tsv: Unused topology value required by the scheduler callback.
        """
        return tile_count

    post, norm_cast = _make_post_and_norm_cast(
        residual,
        updated_residual,
        normalized,
        token_tile,
        task_num_fn,
        ring_slots,
    )
    projection_op = OperatorNode(
        name="mhc_projection",
        op_type=OpType.MHC_PROJECTION,
        inputs=[normalized],
        outputs=[projection],
        param_positions=[17],
        split_value=token_tile,
        split_spec=SplitSpec(None, task_num_fn, [0]),
        tiling_position=18,
        fill_config=_ProjectionFillConfig(),
        diagnostic_name="MhcProjection",
    )
    input_mix = OperatorNode(
        name="mhc_input_mix",
        op_type=OpType.MHC_INPUT_MIX,
        inputs=[updated_residual],
        outputs=[mixed_input],
        param_positions=[12, 2, 17],
        split_value=token_tile,
        split_spec=SplitSpec(None, task_num_fn, [0]),
        tiling_position=18,
        fill_config=_InputMixFillConfig(),
        diagnostic_name="MhcInputMix",
    )
    rms_norm = OperatorNode(
        name="mhc_rms_norm",
        op_type=OpType.RMS_NORM,
        inputs=[mixed_input],
        outputs=[block_input],
        param_positions=[16],
        split_value=token_tile,
        split_spec=SplitSpec(None, task_num_fn, [0]),
        tiling_position=18,
        fill_config=_RmsNormFillConfig(2 * num_cube_cores),
        diagnostic_name="ShiftedRmsNorm",
    )
    mapping = OperatorNode(
        name="mhc_mapping",
        op_type=OpType.MHC_MAPPING,
        inputs=[projection, updated_residual],
        outputs=[mappings],
        param_positions=[17, 12, 13],
        split_value=token_tile,
        split_spec=SplitSpec(None, task_num_fn, [0]),
        tiling_position=18,
        fill_config=_MappingFillConfig(),
        diagnostic_name="MhcMapping/Sinkhorn",
    )
    graph = _assemble_forward_graph(
        topology, post, norm_cast, projection_op, input_mix, rms_norm, mapping
    )
    return graph, topology

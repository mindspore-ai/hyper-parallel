# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Token-tiled fused graph for shifted Single-Pass HyperMegaMhcGrad."""

from __future__ import annotations

from dataclasses import dataclass

from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID,
    FAST_DEPENDENCY_POLL_INTERVAL_US,
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


DEFAULT_GRAD_TOKEN_TILE = 32
NATIVE_GRAD_VECTOR_TILE = 32
NATIVE_GRAD_CUBE_TILE = 64
INIT_DONE_EVENT = 0


@dataclass(frozen=True)
class MegaMhcGradTiling:
    """Outer scheduler tiling decoupled from the native 32/64-row micro-tiles."""

    token_count: int
    token_tile: int
    cube_tile: int
    tile_count: int
    macro_count: int

    def macro_for_tile(self, task_index: int) -> int:
        """Return the AIC macro-tile that consumes one AIV tile.

        Args:
            task_index: AIV tile index.

        Returns:
            Corresponding AIC macro-tile index.
        """
        return task_index * self.token_tile // self.cube_tile

    def vector_tiles_in_macro(self, macro_index: int) -> int:
        """Return the number of AIV tiles contributing to one AIC macro-tile.

        Args:
            macro_index: AIC macro-tile index.

        Returns:
            Number of contributing AIV tiles.
        """
        return sum(
            self.macro_for_tile(task_index) == macro_index
            for task_index in range(self.tile_count)
        )


@dataclass(frozen=True)
class MegaMhcGradEventLayout:
    """Contiguous event ranges for the fused backward pipeline."""

    init_done: int
    rms_ready_base: int
    macro_ready_base: int
    phi_rms_base: int
    final: int
    event_count: int


def build_event_layout(tile_count: int, macro_count: int) -> MegaMhcGradEventLayout:
    """Return event IDs for configurable AIV tiles and AIC macro-tiles.

    Args:
        tile_count: Number of AIV token tiles.
        macro_count: Number of AIC macro-tiles.

    Returns:
        Contiguous event ranges for the backward pipeline.
    """
    if tile_count <= 0 or macro_count <= 0:
        raise ValueError(
            f"tile_count and macro_count must be positive, got {(tile_count, macro_count)}."
        )
    rms_ready_base = INIT_DONE_EVENT + 1
    macro_ready_base = rms_ready_base + tile_count
    phi_rms_base = macro_ready_base + macro_count
    final = phi_rms_base + macro_count
    event_count = final + 1
    if event_count > MIN_EVENT_CAPACITY:
        raise ValueError(
            f"HyperMegaMhcGrad needs {event_count} events for {tile_count} AIV tiles "
            f"and {macro_count} AIC tiles, but RuntimeConfig supports only {MIN_EVENT_CAPACITY}."
        )
    return MegaMhcGradEventLayout(
        init_done=INIT_DONE_EVENT,
        rms_ready_base=rms_ready_base,
        macro_ready_base=macro_ready_base,
        phi_rms_base=phi_rms_base,
        final=final,
        event_count=event_count,
    )


def resolve_grad_token_tile(token_count: int, requested_tile: int) -> int:
    """Choose a 32-row-aligned outer tile that fits the event-counter table.

    Args:
        token_count: Total number of flattened tokens.
        requested_tile: Requested AIV token tile.

    Returns:
        Resolved token tile that fits the runtime event capacity.
    """
    if token_count <= 0 or requested_tile <= 0:
        raise ValueError(
            "token_count and requested_tile must both be positive, "
            f"got {(token_count, requested_tile)}."
        )
    token_tile = (
        (requested_tile + NATIVE_GRAD_VECTOR_TILE - 1)
        // NATIVE_GRAD_VECTOR_TILE
        * NATIVE_GRAD_VECTOR_TILE
    )
    while True:
        tiling = make_grad_tiling(token_count, token_tile)
        if 2 + tiling.tile_count + 2 * tiling.macro_count <= MIN_EVENT_CAPACITY:
            return token_tile
        token_tile += NATIVE_GRAD_VECTOR_TILE


def make_grad_tiling(token_count: int, token_tile: int) -> MegaMhcGradTiling:
    """Construct outer AIV/AIC tiling for one resolved token tile.

    Args:
        token_count: Total number of flattened tokens.
        token_tile: Resolved AIV token tile.

    Returns:
        Derived AIV and AIC tiling metadata.
    """
    cube_tile = max(token_tile, NATIVE_GRAD_CUBE_TILE)
    tile_count = (token_count + token_tile - 1) // token_tile
    macro_count = (token_count + cube_tile - 1) // cube_tile
    return MegaMhcGradTiling(token_count, token_tile, cube_tile, tile_count, macro_count)


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
    dependency_poll_interval_us: int = 0,
) -> None:
    """Append one pure AIC or AIV task to RuntimeConfig."""
    task = TaskDescC()
    task.task_type = task_type
    task.task_aicore_type = core_type
    task.dependent_event = dependent_event
    task.trigger_event = trigger_event
    task.task_index = task_index
    task.task_split_num = task_num
    task.task_split_value = split_value
    # Reuse the current runtime ABI's third task-specific value for the
    # dependency polling interval required by the 4-task backward pipeline.
    task.extra_value_2 = dependency_poll_interval_us
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
        raise ValueError(f"HyperMegaMhcGrad task must be pure AIC or AIV, got {core_type!r}.")
    tsv.pre_task_num += 1


@dataclass(frozen=True)
class _InitFillConfig(FillConfig):
    """Zero atomic-reduction outputs before any producer is released."""

    tiling: MegaMhcGradTiling

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append the reduction-output initialization task.

        Args:
            cfg: Runtime configuration receiving the task.
            op: Initialization graph node.
            tsv: Mutable task counters.
        """
        layout = build_event_layout(self.tiling.tile_count, self.tiling.macro_count)
        cfg.all_event_num_triggers[layout.init_done] = op.task_num
        for task_index in range(op.task_num):
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_GRAD_PREV_A,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=EVENT_INVALID_ID,
                trigger_event=layout.init_done,
                task_index=task_index,
                task_num=op.task_num,
                split_value=0,
            )


@dataclass(frozen=True)
class _RmsNormGradFillConfig(FillConfig):
    """Emit one standalone RMSNormGrad task for each AIV token tile."""

    tiling: MegaMhcGradTiling

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append one RMSNorm gradient task per AIV token tile.

        Args:
            cfg: Runtime configuration receiving the tasks.
            op: RMSNorm gradient graph node.
            tsv: Mutable task counters.
        """
        layout = build_event_layout(self.tiling.tile_count, self.tiling.macro_count)
        for task_index in range(op.task_num):
            ready_event = layout.rms_ready_base + task_index
            cfg.all_event_num_triggers[ready_event] = 1
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_RMS_NORM_GRAD,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=layout.init_done,
                trigger_event=ready_event,
                task_index=task_index,
                task_num=op.task_num,
                split_value=self.tiling.token_tile,
                dependency_poll_interval_us=FAST_DEPENDENCY_POLL_INTERVAL_US,
            )


@dataclass(frozen=True)
class _PrepareFillConfig(FillConfig):
    """Emit fused shifted previous-A and mapping tasks."""

    tiling: MegaMhcGradTiling

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append fused previous-A and mapping tasks.

        Args:
            cfg: Runtime configuration receiving the tasks.
            op: Fused preparation graph node.
            tsv: Mutable task counters.
        """
        layout = build_event_layout(self.tiling.tile_count, self.tiling.macro_count)
        for macro_index in range(self.tiling.macro_count):
            producer_count = self.tiling.vector_tiles_in_macro(macro_index)
            cfg.all_event_num_triggers[layout.macro_ready_base + macro_index] = producer_count
        for task_index in range(op.task_num):
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_GRAD_PREV_A_AND_MAPPING,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=layout.rms_ready_base + task_index,
                trigger_event=layout.macro_ready_base + self.tiling.macro_for_tile(task_index),
                task_index=task_index,
                task_num=op.task_num,
                split_value=self.tiling.token_tile,
                dependency_poll_interval_us=FAST_DEPENDENCY_POLL_INTERVAL_US,
            )


@dataclass(frozen=True)
class _PhiRmsFillConfig(FillConfig):
    """Emit AIC ProcessMatmul1/2 tasks over configurable macro-tiles."""

    tiling: MegaMhcGradTiling

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append projection-gradient tasks for all AIC macro-tiles.

        Args:
            cfg: Runtime configuration receiving the tasks.
            op: Projection-gradient graph node.
            tsv: Mutable task counters.
        """
        layout = build_event_layout(self.tiling.tile_count, self.tiling.macro_count)
        for macro_index in range(op.task_num):
            cfg.all_event_num_triggers[layout.phi_rms_base + macro_index] = 1
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_GRAD_PHI_RMS,
                core_type=TaskAiCoreType.TASK_AICORE_CUBE,
                dependent_event=layout.macro_ready_base + macro_index,
                trigger_event=layout.phi_rms_base + macro_index,
                task_index=macro_index,
                task_num=op.task_num,
                split_value=self.tiling.cube_tile,
            )


@dataclass(frozen=True)
class _PreviousXFillConfig(FillConfig):
    """Emit fused ComputeGradX1 and MhcPost backward tasks."""

    tiling: MegaMhcGradTiling

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append fused previous-X and MhcPost gradient tasks.

        Args:
            cfg: Runtime configuration receiving the tasks.
            op: Fused final-stage graph node.
            tsv: Mutable task counters.
        """
        layout = build_event_layout(self.tiling.tile_count, self.tiling.macro_count)
        cfg.all_event_num_triggers[layout.final] = op.task_num
        for task_index in range(op.task_num):
            _append_task(
                cfg,
                tsv,
                task_type=TaskType.TASK_MHC_GRAD_PREV_X_AND_POST,
                core_type=TaskAiCoreType.TASK_AICORE_VECTOR,
                dependent_event=(layout.phi_rms_base + self.tiling.macro_for_tile(task_index)),
                trigger_event=layout.final,
                task_index=task_index,
                task_num=op.task_num,
                split_value=self.tiling.token_tile,
            )


def order_vector_tasks_by_stage(config: RuntimeConfigC) -> None:
    """Queue all tiles of one AIV stage before advancing to the next stage.

    Args:
        config: Mutable runtime configuration whose AIV queue is reordered.
    """
    vector_count = int(config.task_index_num[1])
    task_ids = [int(config.vector_task_indices[index]) for index in range(vector_count)]
    init_tasks = []
    stage_types = (
        TaskType.TASK_RMS_NORM_GRAD,
        TaskType.TASK_MHC_GRAD_PREV_A_AND_MAPPING,
        TaskType.TASK_MHC_GRAD_PREV_X_AND_POST,
    )
    by_type: dict[TaskType, dict[int, int]] = {task_type: {} for task_type in stage_types}
    for task_id in task_ids:
        task = config.all_tasks[task_id]
        task_type = TaskType(task.task_type)
        if task_type == TaskType.TASK_MHC_GRAD_PREV_A and task.task_split_value == 0:
            init_tasks.append(task_id)
        else:
            by_type[task_type][int(task.task_index)] = task_id

    reordered = list(init_tasks)
    for task_type in stage_types:
        reordered.extend(by_type[task_type][index] for index in sorted(by_type[task_type]))
    if len(reordered) != vector_count or set(reordered) != set(task_ids):
        raise RuntimeError("HyperMegaMhcGrad stage ordering lost or duplicated an AIV task.")
    for index, task_id in enumerate(reordered):
        config.vector_task_indices[index] = task_id


def build_mega_mhc_grad_graph(
    token_count: int,
    hidden_size: int,
    token_tile: int = DEFAULT_GRAD_TOKEN_TILE,
    *,
    num_vector_cores: int,
) -> tuple[ComputeGraph, TaskSplitValue]:
    """Build the RMS -> PrevAAndMapping -> PhiRms -> PrevXAndPost token pipeline.

    Args:
        token_count: Total number of flattened tokens.
        hidden_size: Hidden dimension of every token.
        token_tile: Requested AIV token tile.
        num_vector_cores: Physical AIV count reported for the target device.

    Returns:
        Compute graph and its single-rank topology description.
    """
    if token_count <= 0 or hidden_size <= 0 or num_vector_cores <= 0:
        raise ValueError(
            "token_count, hidden_size, and num_vector_cores must be positive, "
            f"got {(token_count, hidden_size, num_vector_cores)}."
        )
    token_tile = resolve_grad_token_tile(token_count, token_tile)
    tiling = make_grad_tiling(token_count, token_tile)
    build_event_layout(tiling.tile_count, tiling.macro_count)
    topology = TaskSplitValue(tp=1, ep=1, seq_size=token_count, all_expert_num=1, top_k=1)
    data = TensorSpec("grad_data", [token_count, hidden_size], 0)

    def make_op(
        name: str,
        task_num: int,
        split_value: int,
        fill_config: FillConfig,
        diagnostic_name: str,
    ) -> OperatorNode:
        """Construct one backward pipeline operator node.

        Args:
            name: Internal graph-node name.
            task_num: Number of logical tasks.
            split_value: Tokens processed by each task.
            fill_config: Runtime task materializer.
            diagnostic_name: Profiler-visible stage name.

        Returns:
            Configured backward operator node.
        """
        return OperatorNode(
            name=name,
            op_type=OpType.MHC_PIPELINE,
            inputs=[data],
            outputs=[],
            param_positions=[],
            split_value=split_value,
            split_spec=SplitSpec(None, lambda _tsv, count=task_num: count, []),
            tiling_position=34,
            fill_config=fill_config,
            diagnostic_name=diagnostic_name,
        )

    init = make_op(
        "grad_output_init",
        num_vector_cores,
        0,
        _InitFillConfig(tiling),
        "GradOutputInit",
    )
    rms_norm_grad = make_op(
        "rms_norm_grad",
        tiling.tile_count,
        token_tile,
        _RmsNormGradFillConfig(tiling),
        "RmsNormGrad",
    )
    prev_a_and_mapping = make_op(
        "mhc_grad_prev_a_and_mapping",
        tiling.tile_count,
        token_tile,
        _PrepareFillConfig(tiling),
        "MhcGradPrevAAndMapping",
    )
    phi_rms = make_op(
        "mhc_grad_phi_rms",
        tiling.macro_count,
        tiling.cube_tile,
        _PhiRmsFillConfig(tiling),
        "MhcGradPhiRms",
    )
    prev_x_and_post = make_op(
        "mhc_grad_prev_x_and_post",
        tiling.tile_count,
        token_tile,
        _PreviousXFillConfig(tiling),
        "MhcGradPrevXAndPost",
    )

    graph = ComputeGraph()
    for operator in (init, rms_norm_grad, prev_a_and_mapping, phi_rms, prev_x_and_post):
        graph.add_op(operator)
    graph.add_edge(init, rms_norm_grad).add_edge(rms_norm_grad, prev_a_and_mapping)
    graph.add_edge(prev_a_and_mapping, phi_rms)
    graph.add_edge(phi_rms, prev_x_and_post)
    graph.propagate_splits(topology)
    return graph, topology

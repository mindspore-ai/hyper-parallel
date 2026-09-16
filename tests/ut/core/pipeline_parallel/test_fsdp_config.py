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
"""Unit tests for pipeline FSDP configuration."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hyper_parallel.core.pipeline_parallel import scheduler as scheduler_module
from hyper_parallel.core.pipeline_parallel import stage as stage_module
from tests.common.mark_utils import arg_mark


class _FakeHSDPModule:
    """Minimal HSDP module used to exercise schedule configuration."""

    def __init__(self) -> None:
        """Create a fake module with mocked HSDP configuration setters."""
        self.set_reshard_after_forward = MagicMock()
        self.set_reshard_after_backward = MagicMock()
        self.set_requires_gradient_sync = MagicMock()
        self.set_is_last_backward = MagicMock()
        self.hsdp_scheduler = SimpleNamespace(
            hsdp_state=SimpleNamespace(
                post_backward=MagicMock(),
                reduce_params=MagicMock(),
                launch_pipeline_reduce_grad=MagicMock(),
            ),
            launch_reduce_grad_for_pipeline=MagicMock(),
            flush_reduce_grad_for_pipeline=MagicMock(),
            wait_for_pending_reductions=MagicMock(),
        )


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_forward_configured_once_for_multiple_microbatches() -> None:
    """
    Feature: Pipeline FSDP configuration.
    Description: Verify forward reshard is disabled once during schedule setup.
    Expectation: Multiple micro-batches trigger only one forward-reshard setter call.
    """
    micro_batch_num = 4
    fsdp_module = _FakeHSDPModule()
    stage = SimpleNamespace(stage_index=0, submodule=fsdp_module)
    schedule = object.__new__(scheduler_module.ScheduleGPipe)
    schedule.stages = [stage]
    schedule._stage_to_rank_index = {0: 0}
    schedule.micro_batch_num = micro_batch_num
    schedule.exec_order = {
        0: [
            scheduler_module.MetaStep(micro_index, scheduler_module.MetaStepType.FWD, 0)
            for micro_index in range(micro_batch_num)
        ]
    }

    with patch.object(scheduler_module, "HSDPModule", _FakeHSDPModule):
        schedule._inject_local_fsdp_actions()

    fsdp_module.set_reshard_after_forward.assert_called_once_with(False)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_backward_configured_once_per_run_for_multiple_microbatches() -> None:
    """
    Feature: Pipeline FSDP configuration.
    Description: Verify backward HSDP flags are configured once before each schedule run.
    Expectation: Each setter is called once per run, independent of micro-batch count.
    """
    micro_batch_num = 4
    run_count = 2
    fsdp_module = _FakeHSDPModule()
    stage = SimpleNamespace(stage_index=0, submodule=fsdp_module, has_backward=True)
    schedule = object.__new__(scheduler_module.ScheduleGPipe)
    schedule.stages = [stage]
    schedule.real_stage_num = 1
    schedule.micro_batch_num = micro_batch_num
    schedule.exec_order = {
        0: [
            scheduler_module.MetaStep(micro_index, scheduler_module.MetaStepType.BWD, 0)
            for micro_index in range(micro_batch_num)
        ]
    }
    schedule._custom_fn_map = {}
    schedule._exec_step = MagicMock()
    schedule.sync_shared_parameters_grad = MagicMock()
    arg_mbs = [[] for _ in range(micro_batch_num)]
    kwarg_mbs = [{} for _ in range(micro_batch_num)]

    with patch.object(scheduler_module, "HSDPModule", _FakeHSDPModule):
        for _ in range(run_count):
            schedule.run_microbatches(arg_mbs, kwarg_mbs, [])

    expected_micro_calls = micro_batch_num * run_count
    assert schedule._exec_step.call_count == expected_micro_calls, (
        f"Expected all micro-batches to execute: expected={expected_micro_calls}, "
        f"got={schedule._exec_step.call_count}"
    )
    assert fsdp_module.set_reshard_after_backward.call_count == run_count, (
        f"Expected one backward-reshard configuration per run: expected={run_count}, "
        f"got={fsdp_module.set_reshard_after_backward.call_count}"
    )
    assert fsdp_module.set_requires_gradient_sync.call_count == run_count, (
        f"Expected one gradient-sync configuration per run: expected={run_count}, "
        f"got={fsdp_module.set_requires_gradient_sync.call_count}"
    )
    fsdp_module.set_reshard_after_backward.assert_called_with(False)
    fsdp_module.set_requires_gradient_sync.assert_called_with(False)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_reduce_grad_waits_after_the_following_chunks_backward() -> None:
    """All reduction waits are emitted after the local schedule.

    Backward for stage 8 launches the reduction first; stage 0's backward
    launches next. Each stage's trailing all-reduce is released once its own
    launches are behind us, and the single terminal wait drains both rank-wide
    queues after the final backward.
    """
    last_micro = 3
    actions = [
        None,
        scheduler_module.MetaStep(last_micro, scheduler_module.MetaStepType.BWD, 8),
        scheduler_module.MetaStep(last_micro, scheduler_module.MetaStepType.BWD, 0),
        scheduler_module.MetaStep(None, scheduler_module.MetaStepType.BATCH_SEND_RECV, None),
    ]

    result = scheduler_module.add_fsdp_reduce_grad(
        actions,
        managed_stage_indices={0, 8},
        micro_batch_num=last_micro + 1,
    )

    assert [
        None if step is None else step.type
        for step in result
    ] == [
        None,
        scheduler_module.MetaStepType.BWD,
        scheduler_module.MetaStepType.FSDP_REDUCE_GRAD,
        scheduler_module.MetaStepType.BWD,
        scheduler_module.MetaStepType.FSDP_REDUCE_GRAD,
        scheduler_module.MetaStepType.BATCH_SEND_RECV,
        scheduler_module.MetaStepType.FSDP_FLUSH_REDUCE_GRAD,
        scheduler_module.MetaStepType.FSDP_FLUSH_REDUCE_GRAD,
        scheduler_module.MetaStepType.FSDP_WAIT_REDUCE_GRAD,
    ]
    assert [
        None if step is None else step.stage_index
        for step in result
    ] == [None, 8, 8, 0, 0, None, 8, 0, 0]


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_reduce_grad_launches_after_overlap_b_f_on_caller() -> None:
    """A worker backward should finish before its HSDP collectives launch."""
    last_micro = 3
    overlap_step = scheduler_module.MetaStep(
        None,
        scheduler_module.MetaStepType.OVERLAP_B_F,
        None,
        sub_steps=(
            scheduler_module.MetaStep(
                last_micro,
                scheduler_module.MetaStepType.BWD,
                8,
            ),
            scheduler_module.MetaStep(
                last_micro,
                scheduler_module.MetaStepType.FWD,
                0,
            ),
        ),
    )
    actions = [
        overlap_step,
        scheduler_module.MetaStep(last_micro, scheduler_module.MetaStepType.BWD, 0),
    ]

    result = scheduler_module.add_fsdp_reduce_grad(
        actions,
        managed_stage_indices={0, 8},
        micro_batch_num=last_micro + 1,
    )

    assert result == [
        overlap_step,
        scheduler_module.MetaStep(
            None,
            scheduler_module.MetaStepType.FSDP_REDUCE_GRAD,
            8,
        ),
        actions[1],
        scheduler_module.MetaStep(
            None,
            scheduler_module.MetaStepType.FSDP_REDUCE_GRAD,
            0,
        ),
        scheduler_module.MetaStep(
            None,
            scheduler_module.MetaStepType.FSDP_FLUSH_REDUCE_GRAD,
            8,
        ),
        scheduler_module.MetaStep(
            None,
            scheduler_module.MetaStepType.FSDP_FLUSH_REDUCE_GRAD,
            0,
        ),
        scheduler_module.MetaStep(
            None,
            scheduler_module.MetaStepType.FSDP_WAIT_REDUCE_GRAD,
            0,
        ),
    ]


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_control_handlers_trigger_then_drain_on_caller_thread() -> None:
    """FSDP control actions should trigger the reduction and drain it once."""
    stage = MagicMock()
    schedule = SimpleNamespace(stages=[SimpleNamespace(submodule=MagicMock())])

    scheduler_module._exec_fsdp_reduce_grad(schedule, stage)
    scheduler_module._exec_fsdp_flush_reduce_grad(schedule, stage)
    scheduler_module._exec_fsdp_wait_reduce_grad(schedule, stage)

    assert [call[0] for call in stage.mock_calls] == [
        "launch_reduce_grad",
        "flush_reduce_grad",
        "wait_reduce_grad",
    ], "one trigger per chunk; the single drain owns every all-reduce wait"


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_terminal_wait_compensates_stages_whose_backward_never_fired() -> None:
    """Every locally managed stage is re-triggered at the schedule end.

    A stage that never reached its final backward never issued its
    reduce-scatter; re-triggering the stages that did fire is a no-op because
    they have no gradient left. The terminal wait must reach all of them through
    the schedule, exactly as the root backward hook compensates its module.
    """
    first = _FakeHSDPModule()
    second = _FakeHSDPModule()
    ignored = SimpleNamespace(submodule=object())
    stage = MagicMock()
    managed = [
        SimpleNamespace(submodule=first, launch_reduce_grad=MagicMock(), flush_reduce_grad=MagicMock()),
        ignored,
        SimpleNamespace(submodule=second, launch_reduce_grad=MagicMock(), flush_reduce_grad=MagicMock()),
    ]
    schedule = SimpleNamespace(stages=managed)

    with patch.object(scheduler_module, "HSDPModule", _FakeHSDPModule):
        scheduler_module._exec_fsdp_wait_reduce_grad(schedule, stage)

    for managed_stage in (managed[0], managed[2]):
        managed_stage.launch_reduce_grad.assert_called_once_with()
        # The compensating launch is the stage's last one, so its queued group
        # must be released before the drain waits on it.
        managed_stage.flush_reduce_grad.assert_called_once_with()
    assert not hasattr(ignored, "launch_reduce_grad"), "non-HSDP stages are skipped"
    stage.wait_reduce_grad.assert_called_once_with()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_stage_triggers_reduction_and_delegates_terminal_wait() -> None:
    """PipelineStage should trigger the unit reduction and delegate the terminal wait."""
    root = _FakeHSDPModule()
    stage = object.__new__(stage_module.PipelineStage)
    stage.submodule = root

    with patch.object(stage_module, "HSDPModule", _FakeHSDPModule), patch.object(
        stage_module.platform,
        "get_cells_and_names",
        return_value=[("", root)],
    ):
        stage.launch_reduce_grad()
        stage.flush_reduce_grad()
        stage.wait_reduce_grad()

    root.set_is_last_backward.assert_called_once_with(True)
    root.set_reshard_after_backward.assert_called_once_with(True)
    root.set_requires_gradient_sync.assert_called_once_with(True)
    root.hsdp_scheduler.launch_reduce_grad_for_pipeline.assert_called_once_with()
    root.hsdp_scheduler.flush_reduce_grad_for_pipeline.assert_called_once_with()
    root.hsdp_scheduler.hsdp_state.post_backward.assert_not_called()
    root.hsdp_scheduler.wait_for_pending_reductions.assert_called_once_with()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_stage_visits_every_nested_hsdp_unit_but_waits_only_at_the_root() -> None:
    """Per-unit stages must fan out over nested HSDP units; the rank-wide wait stays rooted."""
    root = _FakeHSDPModule()
    nested = _FakeHSDPModule()
    stage = object.__new__(stage_module.PipelineStage)
    stage.submodule = root

    with patch.object(stage_module, "HSDPModule", _FakeHSDPModule), patch.object(
        stage_module.platform,
        "get_cells_and_names",
        return_value=[("", root), ("layers.0", nested)],
    ):
        stage.launch_reduce_grad()
        stage.flush_reduce_grad()
        stage.wait_reduce_grad()

    for module in (root, nested):
        module.hsdp_scheduler.launch_reduce_grad_for_pipeline.assert_called_once_with()
        module.hsdp_scheduler.flush_reduce_grad_for_pipeline.assert_called_once_with()
    # The terminal drain owns the shared queues, so it runs once per stage.
    root.hsdp_scheduler.wait_for_pending_reductions.assert_called_once_with()
    nested.hsdp_scheduler.wait_for_pending_reductions.assert_not_called()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_stage_skips_pipeline_stages_for_a_non_hsdp_submodule() -> None:
    """A stage without a fully_shard root must not attempt any gradient reduction."""
    stage = object.__new__(stage_module.PipelineStage)
    stage.submodule = SimpleNamespace()

    with patch.object(stage_module, "HSDPModule", _FakeHSDPModule):
        stage.launch_reduce_grad()
        stage.flush_reduce_grad()
        stage.wait_reduce_grad()

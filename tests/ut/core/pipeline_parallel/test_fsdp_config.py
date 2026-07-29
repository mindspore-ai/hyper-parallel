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
            ),
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
def test_fsdp_reduce_grad_wraps_final_backwards_with_one_terminal_wait() -> None:
    """Each chunk should launch reductions after backward and drain them once."""
    last_micro = 3
    actions = [
        None,
        scheduler_module.MetaStep(last_micro, scheduler_module.MetaStepType.BWD, 8),
        scheduler_module.MetaStep(last_micro, scheduler_module.MetaStepType.BWD, 0),
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
        scheduler_module.MetaStepType.FSDP_WAIT_REDUCE_GRAD,
    ]
    assert [
        None if step is None else step.stage_index
        for step in result
    ] == [None, 8, 8, 0, 0, 0]


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
def test_fsdp_control_handlers_launch_and_wait_on_caller_thread() -> None:
    """FSDP control actions should launch and drain backend work."""
    stage = SimpleNamespace(
        launch_reduce_grad=MagicMock(),
        wait_reduce_grad=MagicMock(),
    )

    scheduler_module._exec_fsdp_reduce_grad(stage)
    scheduler_module._exec_fsdp_wait_reduce_grad(stage)

    stage.launch_reduce_grad.assert_called_once_with()
    stage.wait_reduce_grad.assert_called_once_with()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_fsdp_stage_launches_deferred_reduction_and_delegates_terminal_wait() -> None:
    """PipelineStage should launch reductions and delegate the terminal wait."""
    root = _FakeHSDPModule()
    stage = object.__new__(stage_module.PipelineStage)
    stage.submodule = root

    with patch.object(stage_module, "HSDPModule", _FakeHSDPModule), patch.object(
        stage_module.platform,
        "get_cells_and_names",
        return_value=[("", root)],
    ):
        stage.launch_reduce_grad()
        stage.wait_reduce_grad()

    root.set_is_last_backward.assert_called_once_with(True)
    root.set_reshard_after_backward.assert_called_once_with(True)
    root.set_requires_gradient_sync.assert_called_once_with(True)
    root.hsdp_scheduler.hsdp_state.post_backward.assert_called_once_with()
    root.hsdp_scheduler.hsdp_state.reduce_params.assert_called_once_with()
    root.hsdp_scheduler.wait_for_pending_reductions.assert_called_once_with()

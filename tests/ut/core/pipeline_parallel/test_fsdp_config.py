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
def test_fsdp_reduce_grad_finalizes_sharded_accumulation_once_per_state() -> None:
    """
    Feature: Pipeline sharded gradient accumulation finalization.
    Description: Drain the tail RS, finalize local shards, and deduplicate shared HSDP states.
    Expectation: Legacy states use post_backward while opt-in states use one final replicate reduction.
    """
    feature_state = SimpleNamespace(
        sharded_accumulated_grad=True,
        reshard_after_backward=True,
        post_backward=MagicMock(),
        reduce_params=MagicMock(),
        flush_sharded_accumulation_after_backward=MagicMock(),
        finalize_sharded_accumulated_grads=MagicMock(),
        shard=MagicMock(),
    )
    legacy_state = SimpleNamespace(
        sharded_accumulated_grad=False,
        reshard_after_backward=True,
        post_backward=MagicMock(),
        reduce_params=MagicMock(),
        flush_sharded_accumulation_after_backward=MagicMock(),
        finalize_sharded_accumulated_grads=MagicMock(),
        shard=MagicMock(),
    )
    root = _FakeHSDPModule()
    root.hsdp_scheduler = SimpleNamespace(
        hsdp_state=feature_state,
        _root_backward_hook=MagicMock(),
    )
    legacy = _FakeHSDPModule()
    legacy.hsdp_scheduler = SimpleNamespace(hsdp_state=legacy_state)
    stage = object.__new__(stage_module.PipelineStage)
    stage.submodule = root

    module_tree = [("", root), ("legacy", legacy), ("legacy_alias", legacy)]
    with patch.object(stage_module, "HSDPModule", _FakeHSDPModule), patch.object(
        stage_module.platform,
        "get_cells_and_names",
        return_value=module_tree,
    ):
        stage.flush_sharded_accumulation_after_backward()
        stage.execute_reduce_grad()

    root.set_reshard_after_backward.assert_called_once_with(True)
    root.set_requires_gradient_sync.assert_called_once_with(True)
    feature_state.post_backward.assert_not_called()
    feature_state.reduce_params.assert_not_called()
    feature_state.flush_sharded_accumulation_after_backward.assert_called_once_with()
    legacy_state.post_backward.assert_called_once_with()
    legacy_state.reduce_params.assert_called_once_with()
    legacy_state.flush_sharded_accumulation_after_backward.assert_called_once_with()
    root.hsdp_scheduler._root_backward_hook.assert_called_once_with(force_reduce=True)
    feature_state.finalize_sharded_accumulated_grads.assert_called_once_with()
    feature_state.shard.assert_called_once_with()
    legacy_state.finalize_sharded_accumulated_grads.assert_not_called()
    legacy_state.shard.assert_not_called()

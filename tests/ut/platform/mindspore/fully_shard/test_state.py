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
"""Unit tests for MindSpore fully_shard state and communication scheduling."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mindspore")
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms

from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo, HSDPMeshInfo
from hyper_parallel.platform.mindspore.fully_shard.param_group import AllReduceParamGroup
from hyper_parallel.platform.mindspore.fully_shard.state import (
    MindSporeHSDPStateV2,
    _to_dtype_if_needed,
)
from tests.ut.platform.mindspore.fully_shard.conftest import MindSporeFullyShardUnitTest


def _fake_param(*, shard_size=2, replicate_size=1, grad=True, source_shard_info=None):
    """Build the facts consumed by state communication methods."""
    hsdp_param = MagicMock()
    mesh_cls = HSDPMeshInfo if replicate_size > 1 and shard_size > 1 else (
        DDPMeshInfo if shard_size == 1 else FSDPMeshInfo
    )
    mesh_info = object.__new__(mesh_cls)
    mesh_info.replicate_process_group = "dp" if replicate_size > 1 else None
    mesh_info.replicate_mesh_size = replicate_size
    mesh_info.shard_process_group = "fsdp" if shard_size > 1 else None
    mesh_info.shard_mesh_size = shard_size
    hsdp_param.mesh_info = mesh_info
    hsdp_param.shard_world_size = shard_size
    hsdp_param.replicate_world_size = replicate_size
    hsdp_param.source_shard_info = source_shard_info
    hsdp_param.orig_dtype = ms.float32
    hsdp_param.reduce_dtype = ms.float32
    hsdp_param.reduce_comm_dtype.return_value = ms.float32
    hsdp_param.sharded_size = (2,)
    hsdp_param.sharded_param = SimpleNamespace(
        requires_grad=True,
        grad=None,
        device="CPU:0",
    )
    hsdp_param._unsharded_param = SimpleNamespace(
        grad=ms.Tensor([1.0, 2.0]) if grad else None
    )
    hsdp_param.unsharded_param = hsdp_param._unsharded_param
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.unsharded_accumulated_grad_data = None
    hsdp_param.unsharded_param_buffers = [MagicMock()] if grad else []
    hsdp_param.reduce_partial_output = None
    hsdp_param.reduce_scatter_comm_ctx = SimpleNamespace(
        reduce_scatter_output=None,
        reduce_scatter_handle=None,
    )
    hsdp_param.all_reduce_comm_ctx = SimpleNamespace(
        all_reduce_output=None,
        all_reduce_handle=None,
    )
    return hsdp_param


def _new_state(hsdp_params=None, *, comm_fusion=False):
    """Create an uninitialized state with direct fields set for CPU UT."""
    state = object.__new__(MindSporeHSDPStateV2)
    state.modules = ()
    state.hsdp_params = list(hsdp_params or [])
    state.raw_ignored_params = set()
    state.raw_replicate_params = set()
    state.comm_fusion_policy = SimpleNamespace(
        enable_comm_fusion=comm_fusion,
        comm_fusion_zero_copy=False,
    )
    state.scheduler_ctx = HSDPSchedulerContext()
    state.device = "cpu"
    state.mp_policy = MagicMock()
    state.offload_policy = None
    state.reduce_grads = True
    state.reshard_after_backward = True
    state.requires_all_reduce = True
    state.reduce_op_type = "avg"
    state.param_group = None
    state._reset_sharded_params = False
    state.is_shard = False
    return state


class TestToDtypeIfNeeded(MindSporeFullyShardUnitTest):
    """Test the MindSpore dtype no-op and cast helper."""

    def test_noop_and_cast(self):
        """Same/None dtypes preserve identity while a different dtype casts."""
        tensor = ms.Tensor([1.0], ms.float32)
        self.assertIs(_to_dtype_if_needed(tensor, None), tensor)
        self.assertIs(_to_dtype_if_needed(tensor, ms.float32), tensor)
        self.assertEqual(_to_dtype_if_needed(tensor, ms.float16).dtype, ms.float16)


class TestParameterInitialization(MindSporeFullyShardUnitTest):
    """Test single-list parameter ownership and parameter-specific mesh routes."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.HSDPParamGroup")
    def test_param_group_uses_all_managed_params_and_disables_zero_copy(self, mock_group):
        """MindSpore comm fusion should retain the safe-copy parameter group path."""
        param = _fake_param()
        state = _new_state([param], comm_fusion=True)

        state._init_param_group()

        mock_group.assert_called_once_with(
            [param],
            "cpu",
            state.mp_policy,
            False,
            comm_ctx=state.scheduler_ctx.param_group_comm_ctx,
        )

    def test_build_param_mesh_info_routes_replicate_params(self):
        """Replicate params should use full DP while other params use FSDP/HSDP."""
        state = _new_state()
        sharded_param = object()
        replicate_param = object()
        state.raw_replicate_params = {replicate_param}
        mesh = MagicMock()
        mesh.ndim = 1
        mesh.mesh_shape = (4,)
        mesh.get_group.return_value = MagicMock()
        state.mesh = mesh
        with patch(
            "hyper_parallel.core.fully_shard.utils.get_group_local_rank",
            return_value=0,
        ):
            sharded_info = state._build_param_mesh_info(sharded_param)
            replicate_info = state._build_param_mesh_info(replicate_param)
        self.assertIsInstance(sharded_info, FSDPMeshInfo)
        self.assertIsInstance(replicate_info, DDPMeshInfo)

    def test_build_param_mesh_info_flattens_2d_replicate_route(self):
        """A replicated parameter on HSDP should all-reduce over the flattened DP mesh."""
        state = _new_state()
        param = object()
        state.raw_replicate_params = {param}
        flattened_mesh = MagicMock(mesh_shape=(8,))
        flattened_mesh.get_group.return_value = MagicMock()
        mesh = MagicMock(ndim=2)
        mesh.flatten.return_value = flattened_mesh
        state.mesh = mesh
        with patch(
            "hyper_parallel.core.fully_shard.utils.get_group_local_rank",
            return_value=0,
        ):
            mesh_info = state._build_param_mesh_info(param)
        mesh.flatten.assert_called_once_with()
        self.assertIs(mesh_info.mesh, flattened_mesh)

    def test_state_defaults_to_avg_with_source_shard_metadata(self):
        """Source-layout metadata must not implicitly change the default reduction."""
        source_param = _fake_param(source_shard_info=object())

        def init_hsdp_params(state):
            state.hsdp_params.append(source_param)

        with patch.object(MindSporeHSDPStateV2, "_move_states_to_device"), \
                patch.object(MindSporeHSDPStateV2, "_init_hsdp_params", init_hsdp_params):
            state = MindSporeHSDPStateV2(
                ms.nn.Dense(1, 1),
                MagicMock(),
                None,
                SimpleNamespace(enable_comm_fusion=False),
                MagicMock(),
                None,
                set(),
                set(),
                MagicMock(),
                HSDPSchedulerContext(),
                "cpu",
            )

        self.assertEqual(state.reduce_op_type, "avg")


class TestBackwardCommunication(MindSporeFullyShardUnitTest):
    """Test per-parameter and fused backward communication pipelines."""

    def test_no_sync_accumulates_and_reshards_without_communication(self):
        """Disabled synchronization should retain full grads and perform cleanup."""
        param = _fake_param()
        state = _new_state([param])
        state.reduce_grads = False
        state.shard = MagicMock()

        state.post_backward()

        param.accumulate_unsharded_grad_if_needed.assert_called_once_with()
        param.to_accumulated_grad_if_needed.assert_called_once_with()
        state.shard.assert_called_once_with()
        param.reduce_scatter_grad.assert_not_called()

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.AllReduceParamGroup")
    def test_issue_reduce_scatter_groups_hsdp_and_fsdp(self, mock_group_cls):
        """Only HSDP parameters should enter a fused follow-up all-reduce group."""
        fsdp_param = _fake_param(shard_size=2, replicate_size=1)
        hsdp_param = _fake_param(shard_size=2, replicate_size=2)
        state = _new_state([fsdp_param, hsdp_param])
        all_reduce_group = MagicMock()
        all_reduce_group.get_param_buffer_view.return_value = MagicMock()
        mock_group_cls.return_value = all_reduce_group

        state._issue_reduce_scatter_for_current_module()

        fsdp_param.reduce_scatter_grad.assert_called_once_with(reduce_op="avg")
        self.assertEqual(state.scheduler_ctx.pre_reduce_scatter_params, [fsdp_param])
        mock_group_cls.assert_called_once_with(
            replicate_group="dp",
            hsdp_params=[hsdp_param],
            reduce_op="avg",
        )
        self.assertEqual(state.scheduler_ctx.pre_all_reduce_groups, [all_reduce_group])

    def test_wait_fsdp_reduce_scatter_retains_final_output_for_root(self):
        """Final synchronized RS output should remain in the parameter context."""
        param = _fake_param()
        reduced_grad = ms.Tensor([1.0, 2.0])
        param.reduce_scatter_output.return_value = reduced_grad
        state = _new_state([param])
        state.scheduler_ctx.pre_reduce_scatter_params.append(param)

        state._wait_prev_reduce_scatter_without_all_reduce()

        self.assertIs(param.reduce_scatter_comm_ctx.reduce_scatter_output, reduced_grad)
        param.clear_unsharded_source_grad.assert_called_once_with()

    def test_no_all_reduce_micro_step_parks_partial_output(self):
        """A no-sync HSDP micro-step should keep RS output in reduce dtype."""
        param = _fake_param()
        reduced_grad = ms.Tensor([1.0, 2.0])
        param.reduce_scatter_output.return_value = reduced_grad
        state = _new_state([param])
        state.requires_all_reduce = False
        state.scheduler_ctx.pre_reduce_scatter_params.append(param)

        state._wait_prev_reduce_scatter_without_all_reduce()

        self.assertIs(param.reduce_partial_output, reduced_grad)
        param.clear_reduce_scatter_output.assert_called_once_with()

    def test_fused_all_reduce_group_is_waited_and_split(self):
        """Tree-local pending groups should expose outputs before root applies grads."""
        group = MagicMock(spec=AllReduceParamGroup)
        group.hsdp_params = []
        state = _new_state()
        state.scheduler_ctx.pending_all_reduce_groups.append(group)

        state.wait_and_split_all_reduce_work_groups()

        group.wait_and_split_grads.assert_called_once_with()
        self.assertEqual(state.scheduler_ctx.pending_all_reduce_groups, [])

    def test_comm_fusion_drains_previous_stages_and_launches_current(self):
        """Fused communication should pipeline AR wait, RS wait, then current RS."""
        state = _new_state(comm_fusion=True)
        state.param_group = MagicMock()
        all_reduce_group = MagicMock()
        previous_group = MagicMock()
        state.scheduler_ctx.param_group_comm_ctx.all_reduce_param_group = all_reduce_group
        state.scheduler_ctx.param_group_comm_ctx.pre_param_group = previous_group

        state.post_backward_for_comm_fusion()

        all_reduce_group.wait_all_reduce_and_save_grad.assert_called_once_with()
        previous_group.wait_reduce_scatter_and_issue_all_reduce.assert_called_once_with()
        state.param_group.foreach_reducescatter.assert_called_once_with(
            reduce_scatter_reduce_op="avg",
        )


class TestStateConfiguration(MindSporeFullyShardUnitTest):
    """Test iteration cleanup and reduction configuration."""

    def test_reset_clears_only_tree_local_communication(self):
        """Reset should release this tree's contexts without clearing optimizer grads."""
        param = _fake_param()
        param.sharded_param.grad = "optimizer-grad"
        state = _new_state([param])
        state.scheduler_ctx.pre_reduce_scatter_params.append(param)
        state.scheduler_ctx.pending_all_reduce_groups.append(MagicMock())

        state.reset_iter_state()

        self.assertEqual(state.scheduler_ctx.pre_reduce_scatter_params, [])
        self.assertEqual(state.scheduler_ctx.pending_all_reduce_groups, [])
        self.assertEqual(param.sharded_param.grad, "optimizer-grad")
        param.clear_reduce_scatter_output.assert_called_once_with()
        param.clear_all_reduce_output.assert_called_once_with()

    def test_reduce_op_setter_accepts_mint_strings(self):
        """Only mint-supported SUM and AVG strings should be accepted."""
        state = _new_state()
        state.set_reduce_op_type("SUM")
        self.assertEqual(state.reduce_op_type, "sum")
        state.set_reduce_op_type("avg")
        self.assertEqual(state.reduce_op_type, "avg")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            state.set_reduce_op_type("mean")


if __name__ == "__main__":
    unittest.main()

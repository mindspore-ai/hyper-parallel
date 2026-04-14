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
"""Unit tests for MindSpore fully_shard state bookkeeping."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode, GroupInfo
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2


def _make_state():
    """Create a lightweight state object without invoking distributed setup."""
    state = object.__new__(MindSporeHSDPStateV2)
    state.hsdp_params = []
    state.replicate_params = []
    state.sharded_hsdp_params = []
    state.mp_policy = SimpleNamespace(param_dtype=None, reduce_dtype=None)
    state.offload_policy = None
    state.device = "npu"
    state.config = SimpleNamespace(
        mesh="dummy-mesh",
        replicate_params=None,
        ignored_params=None,
        shard_placement_fn="shard-fn",
    )
    state.mesh_info = SimpleNamespace(mesh="mesh-info")
    state.modules = []
    state.reduce_grads = True
    state.reshard_after_backward = False
    state.requires_all_reduce = True
    state.reduce_op_type = "sum"
    state._reduce_dtype = None
    state._orig_dtype = None
    state._need_div = False
    state._ignored_allreduce_works = []
    state.shard = MagicMock()
    state._apply_reduced_grad = MagicMock()
    return state


class FakeParam:
    """Hashable parameter stub that behaves like an uninitialized MindSpore param."""

    __slots__ = ("name", "_hsdp_param_initialized")

    def __init__(self, name):
        self.name = name
        self._hsdp_param_initialized = False

    __hash__ = object.__hash__
    __eq__ = object.__eq__


class TestStateParamBookkeeping(unittest.TestCase):
    """Test parameter-mode and group bookkeeping without distributed init."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.state._get_param_module_infos")
    @patch("hyper_parallel.platform.mindspore.fully_shard.state.infer_fully_shard_param_mode")
    @patch("hyper_parallel.platform.mindspore.fully_shard.state.MindSporeHSDPParamV2")
    def test_init_hsdp_params_routes_replicate_params_through_shared_mesh_info(
        self,
        mock_param_ctor,
        mock_infer_mode,
        mock_get_module_infos,
    ):
        """Replicate params should reuse the unified mesh_info and disable FSDP sharding."""
        state = _make_state()
        rep_param = FakeParam("rep_param")
        shard_param = FakeParam("shard_param")
        module = MagicMock()
        module.parameters_and_names.return_value = [("weight", rep_param), ("bias", shard_param)]
        state.modules = [module]
        state.config.replicate_params = {rep_param}

        mock_get_module_infos.return_value = [MagicMock(name="rep_info"), MagicMock(name="shard_info")]
        mock_infer_mode.side_effect = [
            FullyShardParamMode.DTENSOR_COMPAT,
            FullyShardParamMode.LOCAL_PARAM,
        ]

        def _make_hsdp_param(param, module_info, mesh_info, **kwargs):
            del module_info
            obj = MagicMock()
            obj.is_sharded = kwargs["enable_fsdp_shard"]
            obj.param = param
            obj.mesh_info = mesh_info
            obj.kwargs = kwargs
            return obj

        mock_param_ctor.side_effect = _make_hsdp_param

        state._init_hsdp_params()

        mock_infer_mode.assert_any_call(state.config.mesh, [rep_param])
        mock_infer_mode.assert_any_call(state.config.mesh, [shard_param])
        self.assertEqual(mock_param_ctor.call_count, 2)
        self.assertEqual(state.replicate_params[0].mesh_info, state.mesh_info)
        self.assertEqual(state.hsdp_params[0].mesh_info, state.mesh_info)
        self.assertEqual(state.replicate_params[0].kwargs["param_mode"], FullyShardParamMode.DTENSOR_COMPAT)
        self.assertFalse(state.replicate_params[0].kwargs["enable_fsdp_shard"])
        self.assertEqual(state.replicate_params[0].kwargs["shard_placement_fn"], "shard-fn")
        self.assertTrue(state.hsdp_params[0].kwargs["enable_fsdp_shard"])
        self.assertEqual(state.hsdp_params[0].kwargs["param_mode"], FullyShardParamMode.LOCAL_PARAM)

    @patch("hyper_parallel.platform.mindspore.fully_shard.state._get_param_module_infos")
    @patch("hyper_parallel.platform.mindspore.fully_shard.state.infer_fully_shard_param_mode")
    @patch("hyper_parallel.platform.mindspore.fully_shard.state.MindSporeHSDPParamV2")
    def test_init_hsdp_params_skips_ignored_params(
        self,
        mock_param_ctor,
        mock_infer_mode,
        mock_get_module_infos,
    ):
        """Ignored params should be filtered before module-info lookup and param construction."""
        state = _make_state()
        keep_param = FakeParam("keep")
        ignored_param = FakeParam("ignored")
        module = MagicMock()
        module.parameters_and_names.return_value = [("weight", keep_param), ("bias", ignored_param)]
        state.modules = [module]
        state.config.ignored_params = {ignored_param}
        mock_get_module_infos.return_value = [MagicMock(name="keep_info")]
        mock_infer_mode.return_value = FullyShardParamMode.LOCAL_PARAM
        mock_param_ctor.return_value = MagicMock(is_sharded=True)

        state._init_hsdp_params()

        mock_get_module_infos.assert_called_once_with([keep_param], tuple(state.modules))
        mock_param_ctor.assert_called_once()
        self.assertEqual(mock_param_ctor.call_args.args[0], keep_param)
        self.assertEqual(len(state.hsdp_params), 1)

    def test_iter_managed_params_combines_sharded_and_replicate_params(self):
        """Managed params should include both sharded and replicate groups."""
        state = _make_state()
        hsdp_param = MagicMock()
        replicate_param = MagicMock()
        state.hsdp_params = [hsdp_param]
        state.replicate_params = [replicate_param]

        self.assertEqual(state._iter_managed_params(), [hsdp_param, replicate_param])

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.dist.all_reduce")
    def test_allreduce_replicate_params_uses_layout_group_info(self, mock_all_reduce):
        """Replicate params should reduce over the layout-driven unsharded group instead of flattening the root mesh."""
        state = _make_state()
        replicate_param = MagicMock()
        replicate_param.unsharded_accumulated_grad = None
        replicate_param.unsharded_accumulated_grad_data = None
        replicate_param.unsharded_param = SimpleNamespace(grad="grad")
        replicate_param.unsharded_grad_data = "normalized-grad"
        replicate_param.unsharded_group_info = GroupInfo("group", "layout-group", 4)
        state.replicate_params = [replicate_param]

        state._allreduce_replicate_params(async_op=True)

        mock_all_reduce.assert_called_once_with(
            "normalized-grad",
            group="layout-group",
            op="sum",
            async_op=True,
        )
        self.assertEqual(state._ignored_allreduce_works, [(replicate_param, "normalized-grad", 4)])

    def test_post_backward_uses_sync_reduction_on_layout_driven_sizes(self):
        """post_backward should use layout-driven sizes and waitable sync reductions before applying grads."""
        state = _make_state()
        hsdp_param = MagicMock()
        hsdp_param.accumulate_unsharded_grad_if_needed = MagicMock()
        hsdp_param.sharded_param = SimpleNamespace(requires_grad=True)
        hsdp_param.unsharded_param = SimpleNamespace(grad="local-grad")
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param.unsharded_accumulated_grad_data = None
        hsdp_param.unsharded_grad_data = "local-grad"
        hsdp_param.shard_size = 2
        hsdp_param.dp_size = 2
        hsdp_param.shard_world_size = 8
        hsdp_param.replicate_world_size = 8
        hsdp_param.reduce_scatter_grad = MagicMock(return_value=("sharded-grad", None))
        hsdp_param.all_reduce_grad = MagicMock(return_value=("reduced-grad", None))
        state.hsdp_params = [hsdp_param]

        state.post_backward()

        hsdp_param.reduce_scatter_grad.assert_called_once_with(
            async_op=False,
            dtype=None,
            reduce_op="sum",
        )
        hsdp_param.all_reduce_grad.assert_called_once_with(
            grad="sharded-grad",
            async_op=False,
            reduce_op="sum",
        )
        state._apply_reduced_grad.assert_called_once_with(hsdp_param, "reduced-grad")

if __name__ == "__main__":
    unittest.main()

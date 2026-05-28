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

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms
from mindspore import ops

from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode, GroupInfo
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.platform.mindspore.fully_shard import state as state_mod
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
        comm_fusion=False,
        comm_fusion_zero_copy=False,
    )
    state.comm_fusion = False
    state.mesh_info = SimpleNamespace(mesh="mesh-info")
    state.modules = []
    state.reduce_grads = True
    state.reshard_after_backward = False
    state.requires_all_reduce = True
    state.is_shard = True
    state.reduce_op_type = ops.ReduceOp.SUM
    state._reduce_dtype = None
    state._orig_dtype = None
    state._need_div = False
    state._ignored_allreduce_works = []
    state._reset_sharded_params = False
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

    def tearDown(self):
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []

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

    def test_prefetch_forwards_unshard_replicate_flag(self):
        """prefetch should forward the replicate-policy bit to unshard."""
        state = _make_state()
        state.unshard = MagicMock()

        state.prefetch(unshard_replicate=False)

        state.unshard.assert_called_once_with(async_op=True, unshard_replicate=False)

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.dist.all_reduce")
    def test_allreduce_replicate_params_uses_layout_group_info(self, mock_all_reduce):
        """Replicate params should reduce over the layout-driven unsharded group instead of flattening the root mesh."""
        state = _make_state()
        replicate_param = MagicMock()
        grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        replicate_param.unsharded_accumulated_grad = None
        replicate_param.unsharded_accumulated_grad_data = None
        replicate_param.unsharded_param = SimpleNamespace(grad=grad)
        # ``_allreduce_replicate_params`` calls ``.contiguous()`` on the grad
        # right before ``dist.all_reduce`` to satisfy Ascend HCCL. Use a Mock
        # whose ``.contiguous()`` returns itself so the existing identity
        # assertions on the AllReduce input still hold.
        normalized_grad = MagicMock(name="normalized_grad")
        normalized_grad.contiguous.return_value = normalized_grad
        replicate_param.unsharded_grad_data = normalized_grad
        replicate_param.unsharded_group_info = GroupInfo("group", "layout-group", 4)
        state.replicate_params = [replicate_param]

        state._allreduce_replicate_params(async_op=True)

        mock_all_reduce.assert_called_once_with(
            normalized_grad,
            group="layout-group",
            op=ops.ReduceOp.SUM,
            async_op=True,
        )
        normalized_grad.contiguous.assert_called_once_with()
        self.assertEqual(state._ignored_allreduce_works, [(replicate_param, normalized_grad, 4)])

    def test_post_backward_uses_sync_reduction_on_layout_driven_sizes(self):
        """post_backward should use layout-driven sizes and waitable sync reductions before applying grads."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        state = _make_state()
        sharded_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        local_grad = ms.Tensor(np.arange(4, dtype=np.float32))
        reduce_scatter_out = MagicMock(return_value=sharded_grad)
        all_reduce_grad = MagicMock(return_value=("reduced-grad", None))
        unsharded = SimpleNamespace(grad=local_grad)

        def _noop_accumulate():
            return None

        hsdp_param = SimpleNamespace(
            accumulate_unsharded_grad_if_needed=_noop_accumulate,
            sharded_param=SimpleNamespace(requires_grad=True),
            _unsharded_param=unsharded,
            unsharded_param=unsharded,
            unsharded_accumulated_grad=None,
            unsharded_accumulated_grad_data=None,
            unsharded_grad_data=local_grad,
            shard_size=2,
            dp_size=2,
            shard_world_size=8,
            replicate_world_size=8,
            reduce_scatter_grad=MagicMock(return_value=("sharded-grad", None)),
            reduce_scatter_output=reduce_scatter_out,
            clear_reduce_scatter_output=MagicMock(),
            all_reduce_grad=all_reduce_grad,
        )
        state.hsdp_params = [hsdp_param]

        state.post_backward()

        hsdp_param.reduce_scatter_grad.assert_called_once_with(
            async_op=True,
            dtype=None,
            reduce_op=ops.ReduceOp.SUM,
        )
        all_reduce_grad.assert_called_once_with(
            grad=sharded_grad,
            dtype=None,
            async_op=True,
            reduce_op=ops.ReduceOp.SUM,
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.dist.all_reduce")
    def test_post_backward_reduces_direct_dtensor_compat_sharded_grad(self, mock_all_reduce):
        """Pure-TP DTENSOR_COMPAT params should reduce grads stored on sharded_param.grad."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        state = _make_state()
        mock_all_reduce.return_value = "work"
        grad = ms.Tensor(np.ones((2,), dtype=np.float32))

        hsdp_param = SimpleNamespace(
            accumulate_unsharded_grad_if_needed=lambda: None,
            param_mode=FullyShardParamMode.DTENSOR_COMPAT,
            enable_fsdp_shard=True,
            is_sharded=False,
            shard_size=1,
            dp_size=4,
            sharded_param=SimpleNamespace(requires_grad=True, grad=grad),
            unsharded_group_info=GroupInfo("group", "layout-group", 4),
        )
        state.hsdp_params = [hsdp_param]

        state.post_backward()

        mock_all_reduce.assert_called_once_with(
            grad,
            group="layout-group",
            op=ops.ReduceOp.SUM,
            async_op=True,
        )
        self.assertEqual(
            MindSporeHSDPStateV2.pre_direct_all_reduce_grads,
            [("work", grad, grad, 4, state._need_div)],
        )
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []

    def test_reduce_params_drains_direct_dtensor_compat_all_reduce(self):
        """The direct compat queue should wait async work and copy cast buffers back."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        state = _make_state()
        handle = MagicMock()
        reduced_grad = MagicMock()
        target_grad = MagicMock()
        reduced_grad.dtype = "float32"
        target_grad.dtype = "float32"
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = [
            (handle, reduced_grad, target_grad, 4, state._need_div)
        ]

        state.reduce_params()

        handle.wait.assert_called_once_with()
        target_grad.data.copy_.assert_called_once_with(reduced_grad)
        self.assertEqual(MindSporeHSDPStateV2.pre_direct_all_reduce_grads, [])

    def test_pending_grad_helpers_and_stream_synchronization(self):
        """Static helpers should normalize pending grads and skip unnecessary stream sync."""
        local_grad = MagicMock(name="local_grad")
        dtensor_grad = SimpleNamespace(to_local=MagicMock(return_value=local_grad))
        accumulated = MagicMock(name="accumulated")
        unsharded_grad = MagicMock(name="unsharded_grad")
        param = SimpleNamespace(
            unsharded_accumulated_grad=accumulated,
            unsharded_accumulated_grad_data=accumulated,
            unsharded_grad_data=unsharded_grad,
            unsharded_param=SimpleNamespace(grad=unsharded_grad),
            sharded_param=SimpleNamespace(grad=dtensor_grad),
        )

        self.assertIs(MindSporeHSDPStateV2._get_pending_unsharded_grad(param), accumulated)
        self.assertTrue(MindSporeHSDPStateV2._has_pending_unsharded_grad(param))
        self.assertIs(MindSporeHSDPStateV2._get_local_sharded_grad(param), local_grad)
        dtensor_grad.to_local.assert_called_once_with()

        no_unsharded = SimpleNamespace(
            unsharded_accumulated_grad=None,
            unsharded_param=None,
            sharded_param=SimpleNamespace(grad=None),
        )
        self.assertFalse(MindSporeHSDPStateV2._has_pending_unsharded_grad(no_unsharded))
        self.assertIsNone(MindSporeHSDPStateV2._get_local_sharded_grad(no_unsharded))

        with patch.object(state_mod.ms.runtime, "current_stream") as current_stream:
            MindSporeHSDPStateV2._synchronize_current_stream_if_needed(False)
            current_stream.assert_not_called()
            stream = MagicMock()
            current_stream.return_value = stream
            MindSporeHSDPStateV2._synchronize_current_stream_if_needed(True)
            stream.synchronize.assert_called_once_with()

    def test_comm_fusion_unsupported_reason_and_param_group_init(self):
        """comm_fusion should validate parameter layout before constructing a param group."""
        param = SimpleNamespace(
            enable_fsdp_shard=True,
            param_mode=FullyShardParamMode.LOCAL_PARAM,
            _sharded_local_tensor=MagicMock(name="local_shard"),
            shard_world_size=2,
            _param_fqn="p0",
        )

        self.assertIn(
            "non-sharded",
            MindSporeHSDPStateV2._comm_fusion_unsupported_reason(
                SimpleNamespace(enable_fsdp_shard=False)
            ),
        )
        self.assertIn(
            "param_mode",
            MindSporeHSDPStateV2._comm_fusion_unsupported_reason(
                SimpleNamespace(enable_fsdp_shard=True, param_mode=FullyShardParamMode.DTENSOR_COMPAT)
            ),
        )
        self.assertIn(
            "missing local shard",
            MindSporeHSDPStateV2._comm_fusion_unsupported_reason(
                SimpleNamespace(enable_fsdp_shard=True, param_mode=FullyShardParamMode.LOCAL_PARAM)
            ),
        )
        with patch.object(state_mod, "build_rs_plan", side_effect=ValueError("bad")):
            self.assertIn("cannot build", MindSporeHSDPStateV2._comm_fusion_unsupported_reason(param))
        with patch.object(state_mod, "build_rs_plan", return_value=object()):
            self.assertIsNone(MindSporeHSDPStateV2._comm_fusion_unsupported_reason(param))

        state = _make_state()
        state.config.comm_fusion = True
        state.hsdp_params = [param]
        with patch.object(MindSporeHSDPStateV2, "_comm_fusion_unsupported_reason", return_value=None):
            with patch.object(state_mod, "HSDPParamGroup", return_value="param-group") as group_ctor:
                state._init_param_group()
        group_ctor.assert_called_once_with(
            state.hsdp_params,
            state.mesh_info,
            state.device,
            state.mp_policy,
            state.config.comm_fusion_zero_copy,
        )
        self.assertEqual(state.param_group, "param-group")

        with patch.object(MindSporeHSDPStateV2, "_comm_fusion_unsupported_reason", return_value="bad layout"):
            with self.assertRaisesRegex(NotImplementedError, "bad layout"):
                state._init_param_group()

    def test_zero_grad_divide_and_move_states_to_device(self):
        """Simple state mutators should touch only managed params and non-meta tensors."""
        state = _make_state()
        hsdp_param = SimpleNamespace(zero_grad=MagicMock())
        replicate_param = SimpleNamespace(zero_grad=MagicMock())
        state.hsdp_params = [hsdp_param]
        state.replicate_params = [replicate_param]
        state.zero_grad()
        hsdp_param.zero_grad.assert_called_once_with()
        replicate_param.zero_grad.assert_called_once_with()

        tensor = MagicMock()
        MindSporeHSDPStateV2._div_if_needed(tensor, 4, False)
        tensor.div_.assert_not_called()
        MindSporeHSDPStateV2._div_if_needed(tensor, 1, True)
        tensor.div_.assert_not_called()
        MindSporeHSDPStateV2._div_if_needed(tensor, 4, True)
        tensor.div_.assert_called_once_with(4)

        move_param = SimpleNamespace(
            _hsdp_param_initialized=False,
            device="cpu:0",
            to=MagicMock(return_value="moved-param"),
        )
        keep_param = SimpleNamespace(
            _hsdp_param_initialized=True,
            device="cpu:0",
            to=MagicMock(),
        )
        move_buffer = SimpleNamespace(device="cpu:0", to=MagicMock(return_value="moved-buffer"))
        meta_buffer = SimpleNamespace(device="meta", to=MagicMock())
        module = SimpleNamespace(
            get_parameters=MagicMock(return_value=[move_param, keep_param]),
            buffers=MagicMock(return_value=[move_buffer, meta_buffer]),
        )
        state.modules = [module]
        state.device = "npu:0"
        with patch.object(state_mod, "normalize_runtime_device", side_effect=lambda device: device):
            state._move_states_to_device()

        self.assertEqual(move_param.data, "moved-param")
        keep_param.to.assert_not_called()
        self.assertEqual(move_buffer.data, "moved-buffer")
        meta_buffer.to.assert_not_called()

    def test_lazy_init_dtype_and_validation_paths(self):
        """lazy_init should reset sharded params once and enforce dtype/device invariants."""
        param = SimpleNamespace(
            is_sharded=True,
            reset_sharded_param=MagicMock(),
            init_dtype_attrs=MagicMock(),
            sharded_param=SimpleNamespace(requires_grad=True, device="cpu:0"),
            orig_dtype="float32",
            reduce_dtype="float16",
            _param_fqn="p0",
        )
        state = _make_state()
        state.hsdp_params = [param]
        state.offload_policy = CPUOffloadPolicy()

        state.lazy_init()
        state.lazy_init()

        param.reset_sharded_param.assert_called_once_with()
        param.init_dtype_attrs.assert_called()
        self.assertEqual(state._orig_dtype, "float32")
        self.assertEqual(state._reduce_dtype, "float16")

        mismatch = SimpleNamespace(
            init_dtype_attrs=MagicMock(),
            sharded_param=SimpleNamespace(requires_grad=True, device="cpu:0"),
            orig_dtype="float16",
            reduce_dtype="float16",
            _param_fqn="p1",
        )
        state.hsdp_params = [param, mismatch]
        with self.assertRaises(AssertionError):
            state._init_mp_dtypes()

        meta_state = _make_state()
        meta_state.hsdp_params = [
            SimpleNamespace(sharded_param=SimpleNamespace(device="meta"), _param_fqn="meta_param")
        ]
        with self.assertRaisesRegex(RuntimeError, "meta device"):
            meta_state._validate_no_meta_params()

        npu_state = _make_state()
        npu_state.offload_policy = CPUOffloadPolicy()
        npu_state.hsdp_params = [
            SimpleNamespace(sharded_param=SimpleNamespace(device="npu:0"), _param_fqn="npu_param")
        ]
        with self.assertRaisesRegex(RuntimeError, "CPU offloading"):
            npu_state._validate_cpu_offload_params()

    def test_finish_ignored_allreduce_waits_divides_and_applies_grad(self):
        """Ignored replicate reductions should materialize grads and clear pending work."""
        state = _make_state()
        state._need_div = True
        state._orig_dtype = "float32"
        reduced_grad = ms.Tensor(np.full((2,), 8.0, dtype=np.float32))
        param = SimpleNamespace(
            all_reduce_handle=MagicMock(),
            apply_reduced_grad=MagicMock(return_value=True),
        )
        state._ignored_allreduce_works = [(param, reduced_grad, 4)]

        with patch.object(MindSporeHSDPStateV2, "_synchronize_current_stream_if_needed") as sync:
            state._finish_ignored_allreduce()

        param.all_reduce_handle.wait.assert_called_once_with()
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, "float32")
        sync.assert_called_once_with(True)
        self.assertEqual(state._ignored_allreduce_works, [])

    def test_reduce_params_drains_reduce_scatter_and_all_reduce_queues(self):
        """Queued sharded reductions should be divided, applied, and synchronized."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        state = _make_state()
        rs_grad = ms.Tensor(np.full((2,), 4.0, dtype=np.float32))
        ar_grad = ms.Tensor(np.full((2,), 6.0, dtype=np.float32))
        rs_param = SimpleNamespace(
            shard_world_size=4,
            reduce_scatter_output=MagicMock(return_value=rs_grad),
            clear_reduce_scatter_output=MagicMock(),
            apply_reduced_grad=MagicMock(return_value=False),
        )
        ar_param = SimpleNamespace(
            replicate_world_size=2,
            all_reduce_output=MagicMock(return_value=ar_grad),
            clear_all_reduce_output=MagicMock(),
            apply_reduced_grad=MagicMock(return_value=True),
        )
        HSDPState.pre_reduce_scatter_params.append((rs_param, "orig", True))
        HSDPState.pre_all_reduce_params.append((ar_param, "orig", True))

        with patch.object(MindSporeHSDPStateV2, "_synchronize_current_stream_if_needed") as sync:
            state.reduce_params()

        rs_param.clear_reduce_scatter_output.assert_called_once_with()
        rs_param.apply_reduced_grad.assert_called_once_with(rs_grad, "orig")
        ar_param.clear_all_reduce_output.assert_called_once_with()
        ar_param.apply_reduced_grad.assert_called_once_with(ar_grad, "orig")
        sync.assert_called_once_with(True)

    def test_post_backward_without_reduce_and_local_apply_branch(self):
        """post_backward should support no-reduce and shard_size=1 local apply paths."""
        grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        param = SimpleNamespace(
            accumulate_unsharded_grad_if_needed=MagicMock(),
            to_accumulated_grad_if_needed=MagicMock(),
            sharded_param=SimpleNamespace(requires_grad=True),
            _unsharded_param=SimpleNamespace(grad=grad),
            unsharded_param=SimpleNamespace(grad=grad),
            unsharded_accumulated_grad=None,
            unsharded_accumulated_grad_data=None,
            unsharded_grad_data=grad,
            shard_size=1,
            dp_size=1,
            apply_reduced_grad=MagicMock(return_value=False),
        )
        state = _make_state()
        state.hsdp_params = [param]
        state.reduce_grads = False
        state.reshard_after_backward = True
        state.post_backward()
        state.shard.assert_called_once_with()
        param.to_accumulated_grad_if_needed.assert_called_once_with()

        state = _make_state()
        state.hsdp_params = [param]
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        state.post_backward()
        param.apply_reduced_grad.assert_called_once_with(grad, None)

    def test_post_backward_for_comm_fusion_and_reduce_op_setter(self):
        """Fused post-backward should drain staged groups and setter should validate ops."""
        state = _make_state()
        state.param_group = SimpleNamespace(foreach_reduce=MagicMock())
        state.reduce_params = MagicMock()
        state._allreduce_replicate_params = MagicMock()
        all_reduce_group = SimpleNamespace(wait_all_reduce_and_apply_grad=MagicMock())
        pre_group = SimpleNamespace(apply_fusion_reduced_grad=MagicMock(), wait_reduce_scatter_and_issue_all_reduce=MagicMock())
        comm_ctx = SimpleNamespace(all_reduce_param_group=all_reduce_group, pre_param_group=pre_group)

        with patch.object(state_mod, "get_comm_ctx", return_value=comm_ctx):
            state.post_backward_for_comm_fusion()

        state.reduce_params.assert_called_once_with()
        all_reduce_group.wait_all_reduce_and_apply_grad.assert_called_once_with()
        pre_group.wait_reduce_scatter_and_issue_all_reduce.assert_called_once_with()
        state.param_group.foreach_reduce.assert_called_once_with(
            reduce_scatter_reduce_op=ops.ReduceOp.SUM,
            needs_avg_div=False,
        )
        state._allreduce_replicate_params.assert_called_once_with()
        self.assertIsNone(comm_ctx.all_reduce_param_group)
        self.assertIsNone(comm_ctx.pre_param_group)

        state.set_requires_grad_sync(False)
        self.assertFalse(state.reduce_grads)
        state.set_reduce_op_type("avg")
        self.assertTrue(state._need_div)
        self.assertEqual(state.reduce_op_type, ops.ReduceOp.SUM)
        with self.assertRaises(ValueError):
            state.set_reduce_op_type("mean")

if __name__ == "__main__":
    unittest.main()

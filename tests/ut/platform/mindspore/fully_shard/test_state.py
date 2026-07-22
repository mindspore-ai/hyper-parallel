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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard import state as state_mod
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2
from tests.ut.platform.mindspore.fully_shard.conftest import (
    MindSporeFullyShardUnitTest,
    UT_MS_DEVICE,
    UT_RUNTIME_DEVICE,
)


def _make_state():
    """Create a lightweight state object without invoking distributed setup."""
    state = object.__new__(MindSporeHSDPStateV2)
    state.hsdp_params = []
    state.replicate_params = []
    state.sharded_hsdp_params = []
    state.mp_policy = SimpleNamespace(
        param_dtype=None,
        reduce_dtype=None,
        apply_grad_on_fp32_main_grad=False,
    )
    state.offload_policy = None
    state.device = UT_RUNTIME_DEVICE
    state.config = SimpleNamespace(
        mesh="dummy-mesh",
        replicate_params=None,
        ignored_params=None,
        shard_placement_fn="shard-fn",
        comm_fusion=False,
        comm_fusion_zero_copy=False,
        sharded_accumulated_grad=False,
    )
    state.comm_fusion = False
    state.sharded_accumulated_grad = False
    state._pending_sharded_grad_all_reduce_groups = []
    state._param_group_grad_ready_expected = frozenset()
    state._param_group_grad_ready_params = set()
    state._param_group_grad_ready_reduced = False
    state.mesh_info = SimpleNamespace(mesh="mesh-info")
    state.platform = SimpleNamespace(profiler_record=lambda _: nullcontext())
    state.module_name = "unit_test"
    state.modules = []
    state.reduce_grads = True
    state.reshard_after_backward = False
    state.requires_all_reduce = True
    state.param_group = None
    state.is_shard = True
    state.reduce_op_type = ops.ReduceOp.SUM
    MindSporeHSDPStateV2.pre_all_reduce_groups.clear()
    MindSporeHSDPStateV2.pending_all_reduce_groups.clear()
    MindSporeHSDPStateV2.pending_sharded_grad_param_groups.clear()
    state._reset_sharded_params = False
    state.shard = MagicMock()
    state.unshard = MagicMock()
    state._apply_reduced_grad = MagicMock()
    return state


def _make_grad_ready_hsdp_param(native_grad=None):
    """Create a lightweight trainable HSDP parameter for grad-ready tests."""
    unsharded_param = SimpleNamespace(grad=native_grad)
    return SimpleNamespace(
        sharded_param=SimpleNamespace(requires_grad=True),
        _unsharded_param=unsharded_param,
        unsharded_param=unsharded_param,
        unsharded_accumulated_grad=None,
        reduce_dtype=None,
        _to_local_unsharded_grad=lambda grad: grad,
        _register_internal_backward_hook=MagicMock(),
        clear_released_unsharded_grad=MagicMock(),
        to_accumulated_grad_if_needed=MagicMock(),
        accumulate_unsharded_grad_if_needed=MagicMock(),
        zero_grad=MagicMock(),
    )


class FakeParam:
    """Hashable parameter stub that behaves like an uninitialized MindSpore param."""

    __slots__ = ("name", "_hsdp_param_initialized")

    def __init__(self, name):
        self.name = name
        self._hsdp_param_initialized = False

    __hash__ = object.__hash__
    __eq__ = object.__eq__


class TestStateParamBookkeeping(MindSporeFullyShardUnitTest):
    """Test parameter-mode and group bookkeeping without distributed init."""

    def tearDown(self):
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        MindSporeHSDPStateV2.pending_sharded_grad_param_groups.clear()

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

    def test_post_backward_without_reduce_supports_sharded_accumulation(self):
        """The opt-in callback should defer staging until backward has returned."""
        state = _make_state()
        hsdp_param = SimpleNamespace(to_accumulated_grad_if_needed=MagicMock())
        state.hsdp_params = [hsdp_param]

        state._post_backward_without_reduce()

        hsdp_param.to_accumulated_grad_if_needed.assert_called_once_with()
        state.shard.assert_not_called()

        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reshard_after_backward = True
        state._reduce_pending_grads = MagicMock()
        hsdp_param = SimpleNamespace(to_accumulated_grad_if_needed=MagicMock())
        state.hsdp_params = [hsdp_param]

        state._post_backward_without_reduce()

        state._reduce_pending_grads.assert_not_called()
        hsdp_param.to_accumulated_grad_if_needed.assert_not_called()
        state.shard.assert_not_called()

    def test_flush_sharded_accumulation_handles_late_leaf_grad(self):
        """Post-backward flush should clear owned grads and reduce a late leaf grad once."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        unsharded = SimpleNamespace(grad=grad)
        hsdp_param = SimpleNamespace(
            unsharded_accumulated_grad=None,
            _unsharded_param=unsharded,
            unsharded_param=unsharded,
            clear_released_unsharded_grad=MagicMock(),
        )
        state.hsdp_params = [hsdp_param]
        hsdp_param.to_accumulated_grad_if_needed = MagicMock()
        state._reduce_pending_grads = MagicMock()

        state.flush_sharded_accumulation_after_backward()

        hsdp_param.to_accumulated_grad_if_needed.assert_called_once_with()
        state._reduce_pending_grads.assert_called_once_with()
        self.assertEqual(hsdp_param.clear_released_unsharded_grad.call_count, 2)

        hsdp_param.unsharded_param.grad = None
        hsdp_param.clear_released_unsharded_grad.reset_mock()
        state._reduce_pending_grads.reset_mock()

        state.flush_sharded_accumulation_after_backward()

        state._reduce_pending_grads.assert_not_called()
        hsdp_param.clear_released_unsharded_grad.assert_called_once_with()

    def test_register_param_group_grad_ready_hooks_tracks_all_trainable_params(self):
        """Grad-ready registration should include sharded and replicated trainable params."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.comm_fusion = True
        state.param_group = MagicMock()
        sharded_param = _make_grad_ready_hsdp_param()
        replicated_param = _make_grad_ready_hsdp_param()
        frozen_param = _make_grad_ready_hsdp_param()
        frozen_param.sharded_param.requires_grad = False
        state.hsdp_params = [sharded_param, frozen_param]
        state.replicate_params = [replicated_param]

        state._register_param_group_grad_ready_hooks()

        self.assertEqual(
            state._param_group_grad_ready_expected,
            frozenset((id(sharded_param), id(replicated_param))),
        )
        sharded_param._register_internal_backward_hook.assert_called_once()
        replicated_param._register_internal_backward_hook.assert_called_once()
        frozen_param._register_internal_backward_hook.assert_not_called()

    def test_register_param_group_grad_ready_hooks_requires_sharded_accumulation(self):
        """Grad-ready registration should remain disabled without the feature switch."""
        state = _make_state()
        state.comm_fusion = True
        state.param_group = MagicMock()
        hsdp_param = _make_grad_ready_hsdp_param()
        state.hsdp_params = [hsdp_param]

        state._register_param_group_grad_ready_hooks()

        self.assertFalse(state._param_group_grad_ready_expected)
        hsdp_param._register_internal_backward_hook.assert_not_called()

    def test_param_group_grad_ready_hook_reduces_once_after_last_param(self):
        """The last ready parameter should launch exactly one fused reduction."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        first_param = _make_grad_ready_hsdp_param()
        second_param = _make_grad_ready_hsdp_param()
        state.hsdp_params = [first_param, second_param]
        state._param_group_grad_ready_expected = frozenset(
            (id(first_param), id(second_param))
        )
        state._reduce_pending_grads = MagicMock()
        first_grad = ms.Tensor(np.full((2,), 1.0, dtype=np.float32))
        second_grad = ms.Tensor(np.full((2,), 2.0, dtype=np.float32))

        returned_grad = state._param_group_grad_ready_hook(first_param, first_grad)

        self.assertIs(returned_grad, first_grad)
        state._reduce_pending_grads.assert_not_called()

        state._param_group_grad_ready_hook(second_param, second_grad)

        state._reduce_pending_grads.assert_called_once_with()
        self.assertTrue(state._param_group_grad_ready_reduced)
        np.testing.assert_allclose(
            first_param.unsharded_accumulated_grad.asnumpy(),
            np.full((2,), 1.0, dtype=np.float32),
        )
        np.testing.assert_allclose(
            second_param.unsharded_accumulated_grad.asnumpy(),
            np.full((2,), 2.0, dtype=np.float32),
        )
        with self.assertRaisesRegex(RuntimeError, "already launched reduce-scatter"):
            state._param_group_grad_ready_hook(first_param, first_grad)

    def test_param_group_grad_ready_hook_stages_in_accumulation_dtype(self):
        """Ready hooks should materialize stable full grads before communication."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        hsdp_param = _make_grad_ready_hsdp_param()
        hsdp_param.reduce_dtype = ms.float32
        state.hsdp_params = [hsdp_param]
        state._param_group_grad_ready_expected = frozenset((id(hsdp_param),))
        state._reduce_pending_grads = MagicMock()
        grad = ms.Tensor(np.ones((2,), dtype=np.float32), dtype=ms.bfloat16)

        state._param_group_grad_ready_hook(hsdp_param, grad)

        self.assertEqual(hsdp_param.unsharded_accumulated_grad.dtype, ms.float32)
        state._reduce_pending_grads.assert_called_once_with()

    def test_param_group_grad_ready_hook_rejects_duplicate_before_group_ready(self):
        """A duplicate parameter hook must fail before it can corrupt staged gradients."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        first_param = _make_grad_ready_hsdp_param()
        second_param = _make_grad_ready_hsdp_param()
        state.hsdp_params = [first_param, second_param]
        state._param_group_grad_ready_expected = frozenset(
            (id(first_param), id(second_param))
        )
        grad = ms.Tensor(np.ones((2,), dtype=np.float32))

        state._param_group_grad_ready_hook(first_param, grad)

        with self.assertRaisesRegex(RuntimeError, "ran more than once"):
            state._param_group_grad_ready_hook(first_param, grad)

    def test_flush_grad_ready_group_discards_duplicate_native_leaf_grads(self):
        """Flush should not stage native leaf grads already captured by ready hooks."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        native_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        hsdp_param = _make_grad_ready_hsdp_param(native_grad)
        hsdp_param.unsharded_accumulated_grad = native_grad
        state.hsdp_params = [hsdp_param]
        state._param_group_grad_ready_expected = frozenset((id(hsdp_param),))
        state._param_group_grad_ready_params.add(id(hsdp_param))
        state._param_group_grad_ready_reduced = True
        state._reduce_pending_grads = MagicMock()

        state.flush_sharded_accumulation_after_backward()

        self.assertIsNone(hsdp_param.unsharded_param.grad)
        hsdp_param.to_accumulated_grad_if_needed.assert_not_called()
        state._reduce_pending_grads.assert_not_called()
        self.assertFalse(state._param_group_grad_ready_params)
        self.assertFalse(state._param_group_grad_ready_reduced)

    def test_flush_grad_ready_group_falls_back_for_missing_hook(self):
        """A partially ready group should stage missing leaf grads and reduce once after backward."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        first_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        second_grad = ms.Tensor(np.full((2,), 2.0, dtype=np.float32))
        first_param = _make_grad_ready_hsdp_param(first_grad)
        first_param.unsharded_accumulated_grad = first_grad
        second_param = _make_grad_ready_hsdp_param(second_grad)
        state.hsdp_params = [first_param, second_param]
        state._param_group_grad_ready_expected = frozenset(
            (id(first_param), id(second_param))
        )
        state._param_group_grad_ready_params.add(id(first_param))
        state._reduce_pending_grads = MagicMock()

        state.flush_sharded_accumulation_after_backward()

        self.assertIsNone(first_param.unsharded_param.grad)
        first_param.to_accumulated_grad_if_needed.assert_not_called()
        second_param.to_accumulated_grad_if_needed.assert_called_once_with()
        state._reduce_pending_grads.assert_called_once_with()

    def test_fused_sharded_accumulation_keeps_one_pending_rs(self):
        """Feature fused RS should retain one pending work item across pipeline actions."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        state.comm_fusion = True
        state.reduce_params = MagicMock()
        state._queue_replicate_params_allreduce = MagicMock()
        first_group = MagicMock()
        second_group = MagicMock()
        third_group = MagicMock()
        MindSporeHSDPStateV2.pending_sharded_grad_param_groups.extend(
            [first_group, second_group]
        )
        state.param_group = third_group
        third_group.foreach_reduce.return_value = "reduce-output"
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=None,
            pre_param_group=None,
        )

        with patch.object(state_mod, "get_comm_ctx", return_value=comm_ctx):
            state.post_backward_for_comm_fusion()

        first_group.wait_deferred_reduce_scatter_and_apply_grad.assert_called_once_with()
        second_group.wait_deferred_reduce_scatter_and_apply_grad.assert_called_once_with()
        third_group.foreach_reduce.assert_called_once_with(
            reduce_scatter_reduce_op=ops.ReduceOp.SUM,
            defer_all_reduce=True,
        )
        self.assertEqual(
            MindSporeHSDPStateV2.pending_sharded_grad_param_groups,
            [third_group],
        )

    def test_fused_sharded_accumulation_drains_same_group_before_reuse(self):
        """A state should wait its previous RS only when that state is reused."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        state.comm_fusion = True
        state.reduce_params = MagicMock()
        state._queue_replicate_params_allreduce = MagicMock()
        older_group = MagicMock()
        state.param_group = MagicMock()
        state.param_group.foreach_reduce.return_value = "next-output"
        MindSporeHSDPStateV2.pending_sharded_grad_param_groups.extend(
            [older_group, state.param_group]
        )
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=None,
            pre_param_group=None,
        )

        with patch.object(state_mod, "get_comm_ctx", return_value=comm_ctx):
            state.post_backward_for_comm_fusion()

        older_group.wait_deferred_reduce_scatter_and_apply_grad.assert_called_once_with()
        self.assertEqual(
            state.param_group.wait_deferred_reduce_scatter_and_apply_grad.call_count,
            1,
        )
        self.assertEqual(
            MindSporeHSDPStateV2.pending_sharded_grad_param_groups,
            [state.param_group],
        )

    def test_sharded_accumulation_zero_grad_drains_comm_and_full_grads(self):
        """zero_grad should finish feature communication before clearing every grad form."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.wait_sharded_accumulated_grad_reduce_scatters = MagicMock()
        state.wait_sharded_accumulated_grad_all_reduces = MagicMock()
        unsharded_param = SimpleNamespace(grad=MagicMock())
        hsdp_param = SimpleNamespace(
            _unsharded_param=unsharded_param,
            unsharded_param=unsharded_param,
            unsharded_accumulated_grad=MagicMock(),
            clear_released_unsharded_grad=MagicMock(),
            zero_grad=MagicMock(),
        )
        state.hsdp_params = [hsdp_param]

        state.zero_grad()

        state.wait_sharded_accumulated_grad_reduce_scatters.assert_called_once_with()
        state.wait_sharded_accumulated_grad_all_reduces.assert_called_once_with()
        hsdp_param.clear_released_unsharded_grad.assert_called_once_with()
        self.assertIsNone(hsdp_param.unsharded_accumulated_grad)
        self.assertIsNone(hsdp_param.unsharded_param.grad)
        hsdp_param.zero_grad.assert_called_once_with()

    def test_pending_rs_wait_failure_keeps_fifo_ownership(self):
        """A failed wait must leave the group reachable so its handle is not destroyed."""
        pending_group = MagicMock()
        pending_group.wait_deferred_reduce_scatter_and_apply_grad.side_effect = RuntimeError(
            "wait failed"
        )
        MindSporeHSDPStateV2.pending_sharded_grad_param_groups.append(pending_group)

        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            MindSporeHSDPStateV2._drain_pending_sharded_grad_param_groups()

        self.assertEqual(
            MindSporeHSDPStateV2.pending_sharded_grad_param_groups,
            [pending_group],
        )

    def test_pending_final_all_reduce_wait_failure_preserves_unfinished_groups(self):
        """A failed final AR wait should retain only the failed and later groups."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        completed_group = MagicMock()
        failed_group = MagicMock()
        failed_group.wait_and_apply_grads.side_effect = RuntimeError("wait failed")
        later_group = MagicMock()
        state._pending_sharded_grad_all_reduce_groups = [
            completed_group,
            failed_group,
            later_group,
        ]

        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            state.wait_sharded_accumulated_grad_all_reduces()

        completed_group.wait_and_apply_grads.assert_called_once_with()
        failed_group.wait_and_apply_grads.assert_called_once_with()
        later_group.wait_and_apply_grads.assert_not_called()
        self.assertEqual(
            state._pending_sharded_grad_all_reduce_groups,
            [failed_group, later_group],
        )

    def test_sharded_accumulation_issues_pure_reduce_scatter_and_releases_full_grad(self):
        """No-sync HSDP should skip replicate AR and transfer the full grad to async RS."""
        HSDPState.pre_reduce_scatter_params.clear()
        MindSporeHSDPStateV2.pre_all_reduce_groups.clear()
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_grads = False
        grad = ms.Tensor(np.ones((4,), dtype=np.float32))
        unsharded = SimpleNamespace(grad=grad)
        hsdp_param = SimpleNamespace(
            _unsharded_param=unsharded,
            unsharded_param=unsharded,
            unsharded_accumulated_grad=None,
            sharded_param=SimpleNamespace(requires_grad=True),
            shard_size=2,
            dp_size=2,
            param_mode=FullyShardParamMode.LOCAL_PARAM,
            enable_fsdp_shard=True,
            is_sharded=True,
            reduce_dtype=ms.float32,
            orig_dtype=ms.float32,
            reduce_scatter_grad=MagicMock(return_value=("shard", "handle")),
        )
        state.hsdp_params = [hsdp_param]

        state._issue_reduce_scatter_for_current_module()

        hsdp_param.reduce_scatter_grad.assert_called_once_with(
            async_op=True,
            dtype=ms.float32,
            reduce_op=ops.ReduceOp.SUM,
            release_unsharded_grad=True,
        )
        self.assertEqual(
            HSDPState.pre_reduce_scatter_params,
            [(hsdp_param, ms.float32, True)],
        )
        self.assertEqual(MindSporeHSDPStateV2.pre_all_reduce_groups, [])

        HSDPState.pre_reduce_scatter_params.clear()
        hsdp_param.reduce_scatter_grad.reset_mock()

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.AllReduceParamGroup")
    def test_finalize_sharded_accumulation_fuses_local_shards(self, mock_group_ctor):
        """The final pipeline action should fuse local shards that share a replicate group."""
        state = _make_state()
        state.sharded_accumulated_grad = True
        state.reduce_op_type = ops.ReduceOp.AVG
        mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        state.mp_policy = mp_policy

        def _make_param(name, values):
            return SimpleNamespace(
                param_mode=FullyShardParamMode.LOCAL_PARAM,
                mp_policy=mp_policy,
                sharded_param=SimpleNamespace(
                    main_grad=SimpleNamespace(
                        _local_tensor=ms.Tensor(np.array(values, dtype=np.float32))
                    ),
                    grad=None,
                ),
                dp_size=2,
                unsharded_group_info=GroupInfo("replicate", "replicate-group", 2),
                accumulated_allreduced_grad=False,
                reduce_dtype=ms.float32,
                orig_dtype=ms.float32,
                _param_fqn=name,
            )

        first_param = _make_param("first", [2.0, 4.0])
        second_param = _make_param("second", [6.0])
        state.hsdp_params = [first_param, second_param]
        group = MagicMock()

        def _finish_group():
            first_param.accumulated_allreduced_grad = True
            second_param.accumulated_allreduced_grad = True
            return False

        group.wait_and_apply_grads.side_effect = _finish_group
        mock_group_ctor.return_value = group
        pending_rs_group = MagicMock()
        MindSporeHSDPStateV2.pending_sharded_grad_param_groups.append(
            pending_rs_group
        )

        state.launch_sharded_accumulated_grad_all_reduces()

        pending_rs_group.wait_deferred_reduce_scatter_and_apply_grad.assert_called_once_with()
        self.assertEqual(mock_group_ctor.call_count, 1)
        self.assertEqual(
            mock_group_ctor.call_args.kwargs["hsdp_params"],
            [first_param, second_param],
        )
        group.pack_existing_grads_to_buffer.assert_called_once_with()
        group.issue_async_allreduce.assert_called_once_with()

        state.wait_sharded_accumulated_grad_all_reduces()
        state.finalize_sharded_accumulated_grads()

        group.wait_and_apply_grads.assert_called_once_with()
        self.assertEqual(mock_group_ctor.call_count, 1)

    def test_prefetch_forwards_unshard_replicate_flag(self):
        """prefetch should forward the replicate-policy bit to unshard."""
        state = _make_state()
        state.unshard = MagicMock()

        state.prefetch(unshard_replicate=False)

        state.unshard.assert_called_once_with(async_op=True, unshard_replicate=False)

    def test_queue_replicate_params_allreduce_queues_compat_path(self):
        """Replicate params should use the compat all-reduce queue instead of a deferred finish list."""
        state = _make_state()
        replicate_param = SimpleNamespace(
            _unsharded_param=SimpleNamespace(grad=ms.Tensor(np.ones((2,), dtype=np.float32))),
            unsharded_param=SimpleNamespace(grad=ms.Tensor(np.ones((2,), dtype=np.float32))),
            unsharded_accumulated_grad=None,
            sharded_param=SimpleNamespace(requires_grad=True),
            dp_size=2,
        )
        state.replicate_params = [replicate_param]
        state._queue_compat_all_reduce = MagicMock()

        state._queue_replicate_params_allreduce()

        state._queue_compat_all_reduce.assert_called_once_with(replicate_param)

    def test_queue_replicate_params_allreduce_applies_local_grad_when_dp_size_is_one(self):
        """Single-replica replicate_params should materialize local grads without all-reduce."""
        state = _make_state()
        grad = ms.Tensor(np.full((2,), 3.0, dtype=np.float32))
        replicate_param = SimpleNamespace(
            _unsharded_param=SimpleNamespace(grad=grad),
            unsharded_param=SimpleNamespace(grad=grad),
            unsharded_accumulated_grad=None,
            unsharded_grad_data=grad,
            sharded_param=SimpleNamespace(requires_grad=True),
            shard_size=1,
            dp_size=1,
            gradient_scaling_factor=None,
            orig_dtype="float32",
            apply_reduced_grad=MagicMock(return_value=False),
        )
        state.replicate_params = [replicate_param]
        state._queue_compat_all_reduce = MagicMock()

        state._queue_replicate_params_allreduce()

        state._queue_compat_all_reduce.assert_not_called()
        replicate_param.apply_reduced_grad.assert_called_once_with(grad, replicate_param.orig_dtype)

    def test_queue_replicate_params_allreduce_applies_local_grad_when_all_reduce_disabled(self):
        """requires_all_reduce=False should still materialize replicate_params grads locally."""
        state = _make_state()
        state.requires_all_reduce = False
        grad = ms.Tensor(np.full((2,), 4.0, dtype=np.float32))
        replicate_param = SimpleNamespace(
            _unsharded_param=SimpleNamespace(grad=grad),
            unsharded_param=SimpleNamespace(grad=grad),
            unsharded_accumulated_grad=None,
            unsharded_grad_data=grad,
            sharded_param=SimpleNamespace(requires_grad=True),
            shard_size=1,
            dp_size=8,
            gradient_scaling_factor=None,
            orig_dtype="float32",
            apply_reduced_grad=MagicMock(return_value=False),
        )
        state.replicate_params = [replicate_param]
        state._queue_compat_all_reduce = MagicMock()

        state._queue_replicate_params_allreduce()

        state._queue_compat_all_reduce.assert_not_called()
        replicate_param.apply_reduced_grad.assert_called_once_with(grad, replicate_param.orig_dtype)

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

        rep_group = "layout-replicate-group"
        hsdp_param = SimpleNamespace(
            accumulate_unsharded_grad_if_needed=_noop_accumulate,
            param_mode=FullyShardParamMode.LOCAL_PARAM,
            enable_fsdp_shard=True,
            is_sharded=True,
            sharded_param=SimpleNamespace(requires_grad=True),
            sharded_size=(2,),
            _unsharded_param=unsharded,
            unsharded_param=unsharded,
            unsharded_accumulated_grad=None,
            unsharded_accumulated_grad_data=None,
            unsharded_grad_data=local_grad,
            unsharded_group_info=GroupInfo("group", rep_group, 8),
            orig_dtype="float32",
            reduce_dtype=None,
            shard_size=2,
            dp_size=2,
            shard_world_size=8,
            replicate_world_size=8,
            reduce_scatter_grad=MagicMock(return_value=("sharded-grad", None)),
            reduce_scatter_output=reduce_scatter_out,
            clear_reduce_scatter_output=MagicMock(),
            all_reduce_grad=all_reduce_grad,
            accumulated_allreduced_grad=True,
        )
        state.hsdp_params = [hsdp_param]
        MindSporeHSDPStateV2.pre_all_reduce_groups.clear()

        state.post_backward()

        hsdp_param.reduce_scatter_grad.assert_called_once()
        call_kw = hsdp_param.reduce_scatter_grad.call_args.kwargs
        self.assertTrue(call_kw.get("async_op"))
        self.assertEqual(call_kw.get("reduce_op"), ops.ReduceOp.SUM)
        self.assertIn("output_buffer", call_kw)
        all_reduce_grad.assert_not_called()
        self.assertEqual(len(MindSporeHSDPStateV2.pre_all_reduce_groups), 1)

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
            gradient_scaling_factor=None,
            orig_dtype="float32",
            reduce_dtype=None,
            mp_policy=MixedPrecisionPolicy(),
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
            [(hsdp_param, "work", grad, grad, 4, False)],
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
        hsdp_param = SimpleNamespace(mp_policy=MixedPrecisionPolicy())
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = [
            (hsdp_param, handle, reduced_grad, target_grad, 4, False)
        ]

        state.reduce_params()

        handle.wait.assert_called_once_with()
        target_grad.data.copy_.assert_called_once_with(reduced_grad)
        self.assertEqual(MindSporeHSDPStateV2.pre_direct_all_reduce_grads, [])

    def test_reduce_params_drains_direct_dtensor_compat_all_reduce_to_main_grad(self):
        """The direct compat queue should route fp32-main-grad policy through apply_reduced_grad."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        state = _make_state()
        handle = MagicMock()
        reduced_grad = MagicMock()
        target_grad = MagicMock()
        hsdp_param = SimpleNamespace(
            mp_policy=MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True),
            orig_dtype="float32",
            apply_reduced_grad=MagicMock(return_value=False),
        )
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = [
            (hsdp_param, handle, reduced_grad, target_grad, 4, False)
        ]

        state.reduce_params()

        handle.wait.assert_called_once_with()
        hsdp_param.apply_reduced_grad.assert_called_once_with(reduced_grad, hsdp_param.orig_dtype)
        target_grad.data.copy_.assert_not_called()
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
        state.comm_fusion = True
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

    def test_zero_grad_and_move_states_to_device(self):
        """Simple state mutators should touch only managed params and non-meta tensors."""
        state = _make_state()
        state.comm_fusion = True
        hsdp_param = SimpleNamespace(zero_grad=MagicMock())
        replicate_param = SimpleNamespace(zero_grad=MagicMock())
        state.hsdp_params = [hsdp_param]
        state.replicate_params = [replicate_param]
        state.zero_grad()
        hsdp_param.zero_grad.assert_called_once_with()
        replicate_param.zero_grad.assert_called_once_with()

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
        state.device = UT_MS_DEVICE
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
            param_dtype=None,
            reduce_dtype="float16",
            _param_fqn="p0",
        )
        state = _make_state()
        state.comm_fusion = True
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
            param_dtype=None,
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

    def test_reduce_params_drains_single_rank_all_reduce_output(self):
        """rank_size<=1 all-reduce outputs should still drain through reduce_params."""
        HSDPState.pre_all_reduce_params.clear()
        state = _make_state()
        reduced_grad = ms.Tensor(np.full((2,), 5.0, dtype=np.float32))
        param = SimpleNamespace(
            all_reduce_output=MagicMock(return_value=reduced_grad),
            clear_all_reduce_output=MagicMock(),
            apply_reduced_grad=MagicMock(return_value=False),
        )
        HSDPState.pre_all_reduce_params.append((param, ms.float32))

        state.reduce_params()

        param.all_reduce_output.assert_called_once_with()
        param.clear_all_reduce_output.assert_called_once_with()
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, ms.float32)

    def test_reduce_params_drains_replicate_compat_queue(self):
        """Queued replicate all-reduces should be drained through reduce_params."""
        HSDPState.pre_all_reduce_params.clear()
        state = _make_state()
        reduced_grad = ms.Tensor(np.full((2,), 8.0, dtype=np.float32))
        param = SimpleNamespace(
            all_reduce_output=MagicMock(return_value=reduced_grad),
            clear_all_reduce_output=MagicMock(),
            apply_reduced_grad=MagicMock(return_value=True),
        )
        HSDPState.pre_all_reduce_params.append((param, "float32"))

        with patch.object(MindSporeHSDPStateV2, "_synchronize_current_stream_if_needed") as sync:
            state.reduce_params()

        param.all_reduce_output.assert_called_once_with()
        param.clear_all_reduce_output.assert_called_once_with()
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, "float32")
        sync.assert_called_once_with(True)
        self.assertEqual(HSDPState.pre_all_reduce_params, [])

    @patch.object(MindSporeHSDPStateV2, "_queue_replicate_params_allreduce")
    def test_post_backward_queues_replicate_params_allreduce(self, mock_queue_replicate):
        """post_backward should queue replicate reductions through the Torch-aligned path."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        state = _make_state()
        state.hsdp_params = []

        state.post_backward()

        mock_queue_replicate.assert_called_once_with()

    def test_post_backward_does_not_double_queue_replicate_params(self):
        """replicate_params must be queued only via _queue_replicate_params_allreduce."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        state = _make_state()
        replicate_param = SimpleNamespace(
            enable_fsdp_shard=False,
            _unsharded_param=SimpleNamespace(grad=ms.Tensor(np.ones((2,), dtype=np.float32))),
            unsharded_param=SimpleNamespace(grad=ms.Tensor(np.ones((2,), dtype=np.float32))),
            unsharded_accumulated_grad=None,
            sharded_param=SimpleNamespace(requires_grad=True),
            shard_size=1,
            dp_size=2,
            accumulate_unsharded_grad_if_needed=MagicMock(),
        )
        state.replicate_params = [replicate_param]
        state.hsdp_params = []

        with patch.object(state, "reduce_params"), \
             patch.object(state, "_needs_overlap_post_backward_steps", return_value=False), \
             patch.object(state, "_queue_compat_all_reduce") as mock_compat:
            state.post_backward()

        mock_compat.assert_called_once_with(replicate_param)

    @patch.object(MindSporeHSDPStateV2, "_run_overlap_post_backward_steps")
    def test_post_backward_skips_overlap_steps_when_pipeline_idle(self, mock_run_overlap):
        """Pure TP/compat layers should skip the 4-step overlap scaffold when idle."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_all_reduce_groups.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        state = _make_state()
        state.hsdp_params = []

        state.post_backward()

        mock_run_overlap.assert_not_called()

    @patch.object(MindSporeHSDPStateV2, "_run_overlap_post_backward_steps")
    def test_post_backward_runs_overlap_steps_when_pipeline_pending(self, mock_run_overlap):
        """Pending overlap state from a previous layer must still run the 4-step pipeline."""
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads = []
        state = _make_state()
        state.hsdp_params = []
        MindSporeHSDPStateV2.pre_all_reduce_groups = [MagicMock()]

        state.post_backward()

        mock_run_overlap.assert_called_once_with()

    def test_needs_overlap_post_backward_steps_for_current_rs_params(self):
        """A sharded HSDP parameter with pending unsharded grad should require overlap steps."""
        state = _make_state()
        hsdp_param = SimpleNamespace(
            _unsharded_param=SimpleNamespace(grad=ms.Tensor(np.ones((2,), dtype=np.float32))),
            unsharded_param=SimpleNamespace(grad=ms.Tensor(np.ones((2,), dtype=np.float32))),
            unsharded_accumulated_grad=None,
            sharded_param=SimpleNamespace(requires_grad=True),
            shard_size=4,
            param_mode=FullyShardParamMode.LOCAL_PARAM,
            enable_fsdp_shard=True,
            is_sharded=True,
            dp_size=2,
        )
        state.hsdp_params = [hsdp_param]
        MindSporeHSDPStateV2.pre_all_reduce_groups.clear()
        HSDPState.pre_reduce_scatter_params.clear()

        self.assertTrue(state._needs_overlap_post_backward_steps())

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
        HSDPState.pre_reduce_scatter_params.append((rs_param, "orig"))
        HSDPState.pre_all_reduce_params.append((ar_param, "orig"))

        with patch.object(MindSporeHSDPStateV2, "_synchronize_current_stream_if_needed") as sync:
            state.reduce_scattered_params()
            state.reduce_params()

        rs_param.clear_reduce_scatter_output.assert_called_once_with()
        rs_param.apply_reduced_grad.assert_called_once_with(rs_grad, "orig")
        ar_param.clear_all_reduce_output.assert_called_once_with()
        ar_param.apply_reduced_grad.assert_called_once_with(ar_grad, "orig")
        sync.assert_has_calls([call(False), call(True)])

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
            gradient_scaling_factor=None,
            apply_reduced_grad=MagicMock(return_value=False),
            orig_dtype="float32",
            accumulated_allreduced_grad=True,
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
        MindSporeHSDPStateV2.pre_all_reduce_groups.clear()
        MindSporeHSDPStateV2.pending_all_reduce_groups.clear()
        state.post_backward()
        param.apply_reduced_grad.assert_called_once_with(grad, param.orig_dtype)

    def test_post_backward_for_comm_fusion_and_reduce_op_setter(self):
        """Fused post-backward should drain staged groups and setter should validate ops."""
        state = _make_state()
        state.param_group = SimpleNamespace(foreach_reduce=MagicMock())
        state.reduce_params = MagicMock()
        state._queue_replicate_params_allreduce = MagicMock()
        all_reduce_group = SimpleNamespace(wait_all_reduce_and_apply_grad=MagicMock())
        pre_group = SimpleNamespace(
            apply_fusion_reduced_grad=MagicMock(),
            wait_reduce_scatter_and_issue_all_reduce=MagicMock(),
        )
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=all_reduce_group,
            pre_param_group=pre_group,
        )

        with patch.object(state_mod, "get_comm_ctx", return_value=comm_ctx):
            state.post_backward_for_comm_fusion()

        state.reduce_params.assert_called_once_with()
        all_reduce_group.wait_all_reduce_and_apply_grad.assert_called_once_with()
        pre_group.wait_reduce_scatter_and_issue_all_reduce.assert_called_once_with()
        state.param_group.foreach_reduce.assert_called_once_with(
            reduce_scatter_reduce_op=ops.ReduceOp.SUM,
        )
        state._queue_replicate_params_allreduce.assert_called_once_with()
        self.assertIsNone(comm_ctx.all_reduce_param_group)
        self.assertIsNone(comm_ctx.pre_param_group)

        state.set_requires_grad_sync(False)
        self.assertFalse(state.reduce_grads)
        state.set_reduce_op_type("avg")
        self.assertEqual(state.reduce_op_type, ops.ReduceOp.AVG)
        with self.assertRaises(ValueError):
            state.set_reduce_op_type("mean")

    def test_resolve_default_reduce_op_uses_avg_for_local_params(self):
        """LOCAL_PARAM-only states should default to native AVG gradient reduction."""
        state = _make_state()
        state.hsdp_params = [SimpleNamespace(param_mode=FullyShardParamMode.LOCAL_PARAM)]
        self.assertEqual(state._resolve_default_reduce_op(), ops.ReduceOp.AVG)

    def test_resolve_default_reduce_op_uses_sum_for_dtensor_params(self):
        """DTensor-backed states should default to SUM gradient reduction."""
        state = _make_state()
        state.hsdp_params = [
            SimpleNamespace(param_mode=FullyShardParamMode.LOCAL_PARAM),
            SimpleNamespace(param_mode=FullyShardParamMode.DTENSOR_UNIFIED),
        ]
        self.assertEqual(state._resolve_default_reduce_op(), ops.ReduceOp.SUM)

        state.hsdp_params = [SimpleNamespace(param_mode=FullyShardParamMode.DTENSOR_COMPAT)]
        self.assertEqual(state._resolve_default_reduce_op(), ops.ReduceOp.SUM)

    def test_resolve_reduce_op_honors_user_override(self):
        """Explicit set_reduce_op_type should override the state default."""
        state = _make_state()
        state.reduce_op_type = ops.ReduceOp.AVG
        state.set_reduce_op_type("sum")
        self.assertEqual(state._resolve_reduce_op(), ops.ReduceOp.SUM)

if __name__ == "__main__":
    unittest.main()

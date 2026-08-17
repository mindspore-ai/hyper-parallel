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
"""Unit tests for MindSpore fully_shard scheduler hooks and root finalization."""

# pylint: disable=protected-access

import os
import unittest
from contextlib import nullcontext
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

from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext, HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState
from hyper_parallel.platform.mindspore.fully_shard import scheduler as scheduler_mod
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2
from tests.ut.platform.mindspore.fully_shard.conftest import (
    MindSporeFullyShardUnitTest,
    UT_RUNTIME_DEVICE,
)


def _make_scheduler():
    """Create a lightweight scheduler without invoking the full constructor."""
    scheduler = object.__new__(MindSporeHSDPSchedulerV2)
    scheduler.modules = []
    scheduler.platform = "platform"
    scheduler.device = UT_RUNTIME_DEVICE
    scheduler.scheduler_ctx = HSDPSchedulerContext()
    scheduler.mesh = SimpleNamespace(ndim=1)
    scheduler.shard_placement_fn = None
    scheduler.comm_fusion_policy = MagicMock()
    scheduler.mp_policy = MagicMock()
    scheduler.offload_policy = None
    scheduler.ignored_params = set()
    scheduler.replicate_params = set()
    scheduler._get_managed_params = MagicMock(return_value=[])
    scheduler.hsdp_state = MagicMock()
    scheduler.scheduler_state = FSDPSchedulerState.PRE_FORWARD
    scheduler._is_root = True
    scheduler.cell = "cell"
    scheduler._fsdp_group_post_pending = None
    return scheduler


class TestMindSporeScheduler(MindSporeFullyShardUnitTest):
    """Test scheduler state creation, hook wrapping, and root drain ordering."""

    def test_zero_grad_register_hooks_and_platform_validation(self):
        """Small delegating methods should use existing state/platform extension points."""
        scheduler = _make_scheduler()
        scheduler._register_forward_backward_hooks = MagicMock()

        MindSporeHSDPSchedulerV2.zero_grad(scheduler)
        MindSporeHSDPSchedulerV2._register_hooks(scheduler)

        scheduler.hsdp_state.zero_grad.assert_called_once_with()
        scheduler._register_forward_backward_hooks.assert_called_once_with()

        with patch.object(scheduler_mod, "get_platform", return_value=object()):
            with self.assertRaisesRegex(ValueError, "expect MindSporePlatform"):
                MindSporeHSDPSchedulerV2._init_platform(scheduler)

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.MindSporeHSDPStateV2")
    def test_new_cell_state_passes_explicit_mesh_to_state(self, mock_state_ctor):
        """State owns parameter-level FSDP/DDP mesh selection from one explicit DP mesh."""
        scheduler = _make_scheduler()
        MindSporeHSDPSchedulerV2._new_cell_state(scheduler)
        mock_state_ctor.assert_called_once_with(
            scheduler.modules,
            scheduler.mesh,
            scheduler.shard_placement_fn,
            scheduler.comm_fusion_policy,
            scheduler.mp_policy,
            scheduler.offload_policy,
            scheduler.ignored_params,
            scheduler.replicate_params,
            scheduler.platform,
            scheduler.scheduler_ctx,
            scheduler.device,
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.PostBackwardFunction.apply")
    def test_register_post_backward_hook_wraps_only_grad_tensors(self, mock_apply):
        """PostBackwardFunction should only see grad-requiring tensors, matching the torch backend."""
        scheduler = _make_scheduler()
        grad_tensor = ms.Tensor([1.0], ms.float32)
        grad_tensor.requires_grad = True
        wrapped_tensor = ms.Tensor([2.0], ms.float32)
        mock_apply.return_value = (wrapped_tensor,)

        args, kwargs = MindSporeHSDPSchedulerV2._register_post_backward_hook(
            scheduler,
            args=(grad_tensor, "arg-b"),
            kwargs={"kw": "arg-k"},
        )

        mock_apply.assert_called_once_with(scheduler, grad_tensor)
        self.assertEqual(args[0].asnumpy().tolist(), [2.0])
        self.assertEqual(args[1], "arg-b")
        self.assertEqual(kwargs, {"kw": "arg-k"})

    def test_register_post_backward_hook_returns_input_when_grad_disabled_or_absent(self):
        """Backward hook registration should be skipped without grad tracking or grad tensors."""
        scheduler = _make_scheduler()

        with patch.object(scheduler_mod._pynative_executor, "enable_grad", return_value=False):
            args, kwargs = MindSporeHSDPSchedulerV2._register_post_backward_hook(
                scheduler,
                args=("arg",),
                kwargs={"kw": "v"},
            )
        self.assertEqual(args, ("arg",))
        self.assertEqual(kwargs, {"kw": "v"})

        with patch.object(scheduler_mod._pynative_executor, "enable_grad", return_value=True):
            args, kwargs = MindSporeHSDPSchedulerV2._register_post_backward_hook(
                scheduler,
                args=("arg",),
                kwargs={"kw": "v"},
            )
        self.assertEqual(args, ("arg",))
        self.assertEqual(kwargs, {"kw": "v"})

    def test_forward_and_backward_hooks_cover_state_branches(self):
        """Forward/backward hooks should respect recompute and scheduler-state guards."""
        scheduler = _make_scheduler()
        scheduler._hsdp_forward_pre_hook = MagicMock(return_value=(("pre",), {"kw": "pre"}))
        scheduler._register_post_backward_hook = MagicMock(return_value=(("wrapped",), {"kw": "wrapped"}))

        self.assertEqual(
            MindSporeHSDPSchedulerV2._forward_pre_hook(scheduler, "cell", ("arg",), {"kw": "v"}),
            (("wrapped",), {"kw": "wrapped"}),
        )
        scheduler._hsdp_forward_pre_hook.assert_called_once_with("cell", ("arg",), {"kw": "v"})

        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        self.assertIsNone(MindSporeHSDPSchedulerV2._forward_hook(scheduler, "cell", (), "out"))

        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler._register_backward_pre_hook = MagicMock(return_value="registered")
        scheduler._restore_forward_prefetch_after_recompute = MagicMock(return_value=True)
        scheduler._hsdp_forward_hook = MagicMock(return_value="hooked")
        scheduler.scheduler_ctx.root_bp_state = True
        self.assertIsNone(MindSporeHSDPSchedulerV2._forward_hook(scheduler, "cell", (), "out"))
        scheduler._restore_forward_prefetch_after_recompute.assert_called_once_with()
        scheduler._hsdp_forward_hook.assert_not_called()

        scheduler.scheduler_ctx.root_bp_state = False
        self.assertEqual(MindSporeHSDPSchedulerV2._forward_hook(scheduler, "cell", (), "out"), "hooked")

        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        with patch.object(scheduler_mod._pynative_executor, "queue_backward_final_callback") as queue_callback:
            self.assertEqual(MindSporeHSDPSchedulerV2._backward_pre_hook(scheduler, "grad"), "grad")
        queue_callback.assert_not_called()

        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler._hsdp_backward_pre_hook = MagicMock()
        with patch.object(scheduler_mod._pynative_executor, "queue_backward_final_callback"):
            MindSporeHSDPSchedulerV2._backward_pre_hook(scheduler, "grad")
        scheduler._hsdp_backward_pre_hook.assert_called_once_with(scheduler.cell, None)
        self.assertTrue(scheduler.scheduler_ctx.root_bp_state)

    def test_root_backward_and_backward_hook_drain_comm_context(self):
        """Root backward should finish staged fused groups and state reductions once."""
        scheduler = _make_scheduler()
        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler._is_root = True
        scheduler._hsdp_backward_hook = MagicMock()
        scheduler.scheduler_ctx.all_hsdp_schedulers = [scheduler]
        all_reduce_group = SimpleNamespace(wait_all_reduce_and_save_grad=MagicMock())
        pre_group = SimpleNamespace(wait_reduce_scatter_and_issue_all_reduce=MagicMock())
        comm_ctx = SimpleNamespace(all_reduce_param_group=all_reduce_group, pre_param_group=pre_group)
        scheduler.scheduler_ctx.param_group_comm_ctx = comm_ctx
        scheduler.hsdp_state._wait_prev_reduce_scatter.return_value = []
        scheduler.hsdp_state.hsdp_params = []

        MindSporeHSDPSchedulerV2._root_backward_hook(scheduler)

        scheduler._hsdp_backward_hook.assert_called_once_with(scheduler.cell, None, None)
        all_reduce_group.wait_all_reduce_and_save_grad.assert_called_once_with()
        pre_group.wait_reduce_scatter_and_issue_all_reduce.assert_called_once_with()
        scheduler.hsdp_state._wait_prev_reduce_scatter.assert_called_once_with()
        scheduler.hsdp_state._wait_prev_reduce_scatter_without_all_reduce.assert_called_once_with()
        scheduler.hsdp_state.wait_and_split_all_reduce_work_groups.assert_called_once_with()
        self.assertFalse(scheduler.scheduler_ctx.root_bp_state)

        scheduler.scheduler_state = FSDPSchedulerState.BACKWARD
        scheduler._hsdp_backward_hook.reset_mock()
        MindSporeHSDPSchedulerV2._backward_hook(scheduler)
        scheduler._hsdp_backward_hook.assert_not_called()

    def test_gradient_accumulation_keeps_post_backward_hook(self):
        """Ordinary no-sync accumulation must retain FSDP post-backward cleanup."""
        scheduler = _make_scheduler()
        scheduler.hsdp_state.reduce_grads = False
        scheduler.hsdp_state.reshard_after_backward = False
        args = (ms.Tensor([1.0]),)
        args[0].requires_grad = True
        kwargs = {}

        with patch.object(scheduler_mod.PostBackwardFunction, "apply", return_value=args) as apply:
            result = MindSporeHSDPSchedulerV2._register_post_backward_hook(scheduler, args, kwargs)

        self.assertEqual(result, (args, kwargs))
        apply.assert_called_once()

    def test_register_forward_backward_hooks_for_single_and_grouped_modules(self):
        """Hook registration should use grouped hooks only when the grouped marker exists."""
        scheduler = _make_scheduler()
        module = SimpleNamespace(register_forward_pre_hook=MagicMock(), register_forward_hook=MagicMock())
        scheduler.modules = [module]
        scheduler._forward_pre_hook = MagicMock(name="forward_pre_hook")
        scheduler._forward_hook = MagicMock(name="forward_hook")

        # Avoid _DisableMsDispatchMode touching a polluted MindSpore runtime after
        # param_group tests that create real ms.Tensor buffers.
        with patch.object(scheduler_mod, "_DisableMsDispatchMode", side_effect=nullcontext):
            scheduler._fsdp_group_post_pending = None
            MindSporeHSDPSchedulerV2._register_forward_backward_hooks(scheduler)
            pre_hook = module.register_forward_pre_hook.call_args.args[0]
            post_hook = module.register_forward_hook.call_args.args[0]
            self.assertTrue(callable(pre_hook))
            self.assertTrue(callable(post_hook))
            self.assertEqual(module.register_forward_pre_hook.call_args.kwargs, {"with_kwargs": True})
            pre_hook("cell", "args", "kwargs")
            post_hook("cell", "args", "output")
            scheduler._forward_pre_hook.assert_called_once_with("cell", "args", "kwargs")
            scheduler._forward_hook.assert_called_once_with("cell", "args", "output")

            grouped_module = SimpleNamespace(
                register_forward_pre_hook=MagicMock(), register_forward_hook=MagicMock()
            )
            scheduler.modules = [grouped_module]
            scheduler._fsdp_group_post_pending = set()
            scheduler._grouped_forward_pre_hook = MagicMock(name="grouped_forward_pre_hook")
            grouped_forward_post_hook = MagicMock(name="grouped_forward_post_hook")
            scheduler._make_grouped_forward_post_hook = MagicMock(return_value=grouped_forward_post_hook)
            MindSporeHSDPSchedulerV2._register_forward_backward_hooks(scheduler)
            grouped_pre_hook = grouped_module.register_forward_pre_hook.call_args.args[0]
            self.assertTrue(callable(grouped_pre_hook))
            self.assertEqual(grouped_module.register_forward_pre_hook.call_args.kwargs, {"with_kwargs": True})
            grouped_pre_hook("cell", "args", "kwargs")
            scheduler._grouped_forward_pre_hook.assert_called_once_with("cell", "args", "kwargs")
            grouped_post_hook = grouped_module.register_forward_hook.call_args.args[0]
            grouped_post_hook("cell", "args", "output")
            scheduler._make_grouped_forward_post_hook.assert_called_once_with(grouped_module)
            grouped_forward_post_hook.assert_called_once_with("cell", "args", "output")


class TestCoreScheduler(unittest.TestCase):
    """Test core scheduler state transitions without constructing distributed state."""

    def _make_core_scheduler(self):
        """Create a core scheduler instance with mocked state and hooks."""
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.reshard_after_forward = True
        scheduler.hsdp_state = MagicMock()
        scheduler.scheduler_ctx = HSDPSchedulerContext()
        scheduler.forward_prefetch_cells = []
        scheduler.backward_prefetch_cells = []
        scheduler._backup_forward_fetch = None
        scheduler._fsdp_group_post_pending = None
        scheduler.scheduler_state = None
        scheduler._forward_pre_hook = MagicMock(return_value=("args", {"kw": "v"}))
        scheduler._forward_hook = MagicMock(return_value="outputs")
        scheduler.modules = []
        return scheduler

    def test_reset_iter_state_only_clears_current_module_tree(self):
        """Reset should clear scheduler, queue, and comm state without touching another tree."""
        scheduler = self._make_core_scheduler()
        other_scheduler = self._make_core_scheduler()
        current_ctx = scheduler.scheduler_ctx
        other_ctx = other_scheduler.scheduler_ctx
        current_ctx.root_bp_state = True
        other_ctx.root_bp_state = True
        current_ctx.pre_reduce_scatter_params.append("current-rs")
        current_ctx.pre_all_reduce_params.append("current-ar")
        current_ctx.pre_direct_all_reduce_grads.append("current-direct-ar")
        current_ctx.pre_all_reduce_groups.append("current-pre-group")
        current_ctx.pending_all_reduce_groups.append("current-pending-group")
        current_ctx.param_group_comm_ctx.pre_param_group = "current-pre-param-group"
        current_ctx.param_group_comm_ctx.all_reduce_param_group = "current-ar-param-group"
        current_ctx.param_group_comm_ctx.comm_handle = "current-rs-handle"
        other_ctx.pre_reduce_scatter_params.append("other-rs")
        other_ctx.pre_all_reduce_params.append("other-ar")
        other_ctx.pre_direct_all_reduce_grads.append("other-direct-ar")
        other_ctx.pre_all_reduce_groups.append("other-pre-group")
        other_ctx.pending_all_reduce_groups.append("other-pending-group")
        other_ctx.param_group_comm_ctx.pre_param_group = "other-pre-param-group"
        other_ctx.param_group_comm_ctx.all_reduce_param_group = "other-ar-param-group"
        other_ctx.param_group_comm_ctx.comm_handle = "other-rs-handle"

        scheduler.reset_iter_state()

        self.assertFalse(current_ctx.root_bp_state)
        self.assertEqual(current_ctx.pre_reduce_scatter_params, [])
        self.assertEqual(current_ctx.pre_all_reduce_params, [])
        self.assertEqual(current_ctx.pre_direct_all_reduce_grads, [])
        self.assertEqual(current_ctx.pre_all_reduce_groups, [])
        self.assertEqual(current_ctx.pending_all_reduce_groups, [])
        self.assertIsNone(current_ctx.param_group_comm_ctx.pre_param_group)
        self.assertIsNone(current_ctx.param_group_comm_ctx.all_reduce_param_group)
        self.assertIsNone(current_ctx.param_group_comm_ctx.comm_handle)
        self.assertTrue(other_ctx.root_bp_state)
        self.assertEqual(other_ctx.pre_reduce_scatter_params, ["other-rs"])
        self.assertEqual(other_ctx.pre_all_reduce_params, ["other-ar"])
        self.assertEqual(other_ctx.pre_direct_all_reduce_grads, ["other-direct-ar"])
        self.assertEqual(other_ctx.pre_all_reduce_groups, ["other-pre-group"])
        self.assertEqual(other_ctx.pending_all_reduce_groups, ["other-pending-group"])
        self.assertEqual(other_ctx.param_group_comm_ctx.pre_param_group, "other-pre-param-group")
        self.assertEqual(other_ctx.param_group_comm_ctx.all_reduce_param_group, "other-ar-param-group")
        self.assertEqual(other_ctx.param_group_comm_ctx.comm_handle, "other-rs-handle")

    def test_setters_update_config_and_state(self):
        """Scheduler setters should validate bool inputs and update owned state."""
        scheduler = self._make_core_scheduler()

        scheduler.set_reshard_after_forward(False)
        scheduler.set_reshard_after_backward(True)
        scheduler.set_requires_all_reduce(False)
        scheduler.set_requires_grad_sync(True)

        self.assertFalse(scheduler.reshard_after_forward)
        scheduler.hsdp_state.set_requires_grad_sync.assert_called_once_with(True)
        self.assertTrue(scheduler.hsdp_state.reshard_after_backward)
        scheduler.hsdp_state.set_requires_all_reduce.assert_called_once_with(False)
        for method, value in [
            (scheduler.set_reshard_after_forward, 1),
            (scheduler.set_reshard_after_backward, 1),
            (scheduler.set_requires_all_reduce, 1),
            (scheduler.set_requires_grad_sync, 1),
        ]:
            with self.subTest(method=method.__name__), self.assertRaises(ValueError):
                method(value)

    def test_prefetch_setters_and_recompute_restore(self):
        """Forward prefetch should be temporarily disabled and later restored."""
        scheduler = self._make_core_scheduler()
        forward_cells = [object()]
        backward_cells = [object()]

        scheduler.set_forward_prefetch_cells(forward_cells)
        scheduler.set_backward_prefetch_cells(backward_cells)
        scheduler._disable_forward_prefetch_for_recompute()

        self.assertEqual(scheduler.forward_prefetch_cells, [])
        self.assertEqual(scheduler.backward_prefetch_cells, backward_cells)
        self.assertTrue(scheduler._restore_forward_prefetch_after_recompute())
        self.assertEqual(scheduler.forward_prefetch_cells, forward_cells)
        self.assertFalse(scheduler._restore_forward_prefetch_after_recompute())

    def test_grouped_forward_pre_hook_runs_once_per_group(self):
        """Grouped pre-hook should run once until the pending set drains."""
        scheduler = self._make_core_scheduler()
        module_a = object()
        module_b = object()
        scheduler.modules = [module_a, module_b]
        scheduler._fsdp_group_post_pending = set()

        first = scheduler._grouped_forward_pre_hook(module_a, ("x",), {"k": "v"})
        second = scheduler._grouped_forward_pre_hook(module_b, ("y",), {})

        self.assertEqual(first, ("args", {"kw": "v"}))
        self.assertEqual(second, (("y",), {}))
        scheduler._forward_pre_hook.assert_called_once_with(module_a, ("x",), {"k": "v"})
        self.assertEqual(scheduler._fsdp_group_post_pending, {module_a, module_b})

    def test_grouped_forward_post_hook_runs_on_last_module(self):
        """Grouped post-hook should defer work until the final module completes."""
        scheduler = self._make_core_scheduler()
        module_a = object()
        module_b = object()
        scheduler._fsdp_group_post_pending = {module_a, module_b}

        hook_a = scheduler._make_grouped_forward_post_hook(module_a)
        hook_b = scheduler._make_grouped_forward_post_hook(module_b)

        self.assertEqual(hook_a(module_a, (), "partial"), "partial")
        self.assertEqual(hook_b(module_b, (), "final"), "outputs")
        scheduler._forward_hook.assert_called_once_with(module_b, (), "final")

    def test_forward_pre_hook_returns_early_during_pre_backward(self):
        """Pre-forward should be a no-op while the scheduler is in pre-backward state."""
        scheduler = self._make_core_scheduler()
        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD

        self.assertEqual(
            scheduler._hsdp_forward_pre_hook("cell", ("arg",), {"kw": "v"}),
            (("arg",), {"kw": "v"}),
        )
        scheduler.hsdp_state.unshard.assert_not_called()


if __name__ == "__main__":
    unittest.main()

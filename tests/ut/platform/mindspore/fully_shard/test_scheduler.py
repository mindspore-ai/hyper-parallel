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
"""Unit tests for MindSpore fully_shard scheduler compatibility behavior."""

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

from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState
from hyper_parallel.platform.mindspore.fully_shard import scheduler as scheduler_mod
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2


def _make_scheduler():
    """Create a lightweight scheduler without invoking the full constructor."""
    scheduler = object.__new__(MindSporeHSDPSchedulerV2)
    scheduler.modules = []
    scheduler.platform = "platform"
    scheduler.device = "npu"
    scheduler.config = SimpleNamespace(mesh=None)
    scheduler.mesh = None
    scheduler._get_managed_params = MagicMock(return_value=[])
    scheduler.hsdp_state = MagicMock()
    scheduler.scheduler_state = FSDPSchedulerState.PRE_FORWARD
    scheduler.cell = "cell"
    scheduler._fsdp_group_post_pending = None
    return scheduler


class FakeMesh:
    """Minimal mesh stub exposing only the hash used by compatibility mode."""

    def __init__(self, mesh_hash):
        self._mesh_hash = mesh_hash

    def to_hash(self):
        return self._mesh_hash


class TestMindSporeScheduler(unittest.TestCase):
    """Test scheduler compatibility-mode mesh resolution and hook wrapping."""

    def tearDown(self):
        HSDPSchedulerV2.root_bp_state = False

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
    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.DDPMeshInfo")
    def test_new_cell_state_uses_compat_mesh_for_mesh_none(self, mock_ddp_mesh_info, mock_state_ctor):
        """mesh=None should reuse the shared DTensor mesh carried by managed parameters."""
        scheduler = _make_scheduler()
        compat_mesh = FakeMesh("mesh-hash")
        scheduler._get_managed_params.return_value = ["p0", "p1"]
        mock_ddp_mesh_info.return_value = "compat-mesh-info"

        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.scheduler.get_dtensor_managed_mesh",
            side_effect=[compat_mesh, compat_mesh],
        ):
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)

        mock_ddp_mesh_info.assert_called_once_with(mesh=compat_mesh, replicate_mesh_dim=0)
        mock_state_ctor.assert_called_once_with(
            scheduler.modules,
            "compat-mesh-info",
            scheduler.config,
            scheduler.platform,
            scheduler.device,
        )
        self.assertEqual(scheduler.mesh_info, "compat-mesh-info")

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.MindSporeHSDPStateV2")
    def test_new_cell_state_uses_explicit_1d_2d_mesh_and_rejects_others(self, mock_state_ctor):
        """Explicit meshes should map to FSDP/HSDP mesh info based on dimensionality."""
        scheduler = _make_scheduler()
        scheduler.mesh = SimpleNamespace(ndim=1)
        with patch.object(scheduler_mod, "FSDPMeshInfo", return_value="fsdp-info") as fsdp_info:
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)
        fsdp_info.assert_called_once_with(mesh=scheduler.mesh, shard_mesh_dim=0)
        mock_state_ctor.assert_called_with(
            scheduler.modules,
            "fsdp-info",
            scheduler.config,
            scheduler.platform,
            scheduler.device,
        )

        scheduler.mesh = SimpleNamespace(ndim=2)
        with patch.object(scheduler_mod, "HSDPMeshInfo", return_value="hsdp-info") as hsdp_info:
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)
        hsdp_info.assert_called_once_with(mesh=scheduler.mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
        self.assertEqual(scheduler.mesh_info, "hsdp-info")

        scheduler.mesh = SimpleNamespace(ndim=3)
        with self.assertRaisesRegex(ValueError, "only supports"):
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)

    def test_new_cell_state_rejects_mesh_none_without_dtensor_mesh(self):
        """Compatibility mode needs at least one DTensor-managed mesh."""
        scheduler = _make_scheduler()
        scheduler._get_managed_params.return_value = ["param"]
        with patch.object(scheduler_mod, "get_dtensor_managed_mesh", return_value=None):
            with self.assertRaisesRegex(ValueError, "without a DTensor"):
                MindSporeHSDPSchedulerV2._new_cell_state(scheduler)

    def test_new_cell_state_rejects_mixed_compat_meshes(self):
        """mesh=None compatibility mode should reject DTensor params with different meshes."""
        scheduler = _make_scheduler()
        mesh_a = FakeMesh("mesh-a")
        mesh_b = FakeMesh("mesh-b")
        scheduler._get_managed_params.return_value = ["p0", "p1"]

        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.scheduler.get_dtensor_managed_mesh",
            side_effect=[mesh_a, mesh_b],
        ), self.assertRaisesRegex(ValueError, "share the same mesh"):
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)

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
        HSDPSchedulerV2.root_bp_state = True
        self.assertIsNone(MindSporeHSDPSchedulerV2._forward_hook(scheduler, "cell", (), "out"))
        scheduler._restore_forward_prefetch_after_recompute.assert_called_once_with()
        scheduler._hsdp_forward_hook.assert_not_called()

        HSDPSchedulerV2.root_bp_state = False
        self.assertEqual(MindSporeHSDPSchedulerV2._forward_hook(scheduler, "cell", (), "out"), "hooked")

        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        with patch.object(scheduler_mod._pynative_executor, "queue_backward_final_callback") as queue_callback:
            self.assertEqual(MindSporeHSDPSchedulerV2._backward_pre_hook(scheduler, "grad"), "grad")
        queue_callback.assert_called_once_with(scheduler._root_backward_hook)

        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler._hsdp_backward_pre_hook = MagicMock()
        with patch.object(scheduler_mod._pynative_executor, "queue_backward_final_callback"):
            MindSporeHSDPSchedulerV2._backward_pre_hook(scheduler, "grad")
        scheduler._hsdp_backward_pre_hook.assert_called_once_with(scheduler.cell, None)
        self.assertTrue(HSDPSchedulerV2.root_bp_state)

    def test_root_backward_and_backward_hook_drain_comm_context(self):
        """Root backward should finish staged fused groups and state reductions once."""
        scheduler = _make_scheduler()
        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler._is_root = True
        scheduler._hsdp_backward_hook = MagicMock()
        all_reduce_group = SimpleNamespace(wait_all_reduce_and_apply_grad=MagicMock())
        pre_group = SimpleNamespace(apply_fusion_reduced_grad=MagicMock())
        comm_ctx = SimpleNamespace(all_reduce_param_group=all_reduce_group, pre_param_group=pre_group)

        with patch.object(scheduler_mod, "get_comm_ctx", return_value=comm_ctx):
            MindSporeHSDPSchedulerV2._root_backward_hook(scheduler)

        scheduler._hsdp_backward_hook.assert_called_once_with(scheduler.cell, None, None)
        all_reduce_group.wait_all_reduce_and_apply_grad.assert_called_once_with()
        pre_group.apply_fusion_reduced_grad.assert_called_once_with()
        scheduler.hsdp_state.reduce_params.assert_called_once_with()
        scheduler.hsdp_state._finish_ignored_allreduce.assert_called_once_with()
        self.assertFalse(HSDPSchedulerV2.root_bp_state)

        scheduler.scheduler_state = FSDPSchedulerState.BACKWARD
        scheduler._hsdp_backward_hook.reset_mock()
        MindSporeHSDPSchedulerV2._backward_hook(scheduler)
        scheduler._hsdp_backward_hook.assert_not_called()

    def test_register_forward_backward_hooks_for_single_and_grouped_modules(self):
        """Hook registration should use grouped hooks only when the grouped marker exists."""
        scheduler = _make_scheduler()
        module = SimpleNamespace(register_forward_pre_hook=MagicMock(), register_forward_hook=MagicMock())
        scheduler.modules = [module]
        scheduler._forward_pre_hook = MagicMock(name="forward_pre_hook")
        scheduler._forward_hook = MagicMock(name="forward_hook")

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

        grouped_module = SimpleNamespace(register_forward_pre_hook=MagicMock(), register_forward_hook=MagicMock())
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
        scheduler.config = SimpleNamespace(reshard_after_forward=True)
        scheduler.hsdp_state = MagicMock()
        scheduler.forward_prefetch_cells = []
        scheduler.backward_prefetch_cells = []
        scheduler._backup_forward_fetch = None
        scheduler._fsdp_group_post_pending = None
        scheduler._forward_pre_hook = MagicMock(return_value=("args", {"kw": "v"}))
        scheduler._forward_hook = MagicMock(return_value="outputs")
        scheduler.modules = []
        return scheduler

    def test_setters_update_config_and_state(self):
        """Scheduler setters should validate bool inputs and update owned state."""
        scheduler = self._make_core_scheduler()

        scheduler.set_reshard_after_forward(False)
        scheduler.set_reshard_after_backward(True)
        scheduler.set_requires_all_reduce(False)
        scheduler.set_requires_grad_sync(True)

        self.assertFalse(scheduler.reshard_after_forward)
        self.assertFalse(scheduler.config.reshard_after_forward)
        scheduler.hsdp_state.set_requires_grad_sync.assert_called_once_with(True)
        self.assertTrue(scheduler.hsdp_state.reshard_after_backward)
        self.assertFalse(scheduler.hsdp_state.requires_all_reduce)
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

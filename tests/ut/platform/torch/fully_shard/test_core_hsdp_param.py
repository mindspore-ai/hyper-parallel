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
"""Unit tests for core HSDP parameter layout helpers."""
import os
import unittest
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard import hsdp_param as hsdp_param_mod
from hyper_parallel.core.fully_shard import hsdp_scheduler as hsdp_scheduler_mod
from hyper_parallel.core.fully_shard.hsdp_param import (
    HSDPParamV2,
    _build_group_info_from_process_group,
    _build_group_info_from_rank_list,
)
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext, HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState, FullyShardParamMode, ShardedState
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo


class FakeMesh:
    """Small mesh double exposing the attributes used by HSDPParamV2."""

    def __init__(self, ndim=2, names=None, shape=None, group=None):
        self.ndim = ndim
        self.mesh_dim_names = names
        self.mesh_shape = shape or tuple(2 for _ in range(ndim))
        self._group = group

    def get_group(self, axis_name):
        del axis_name
        return self._group


def _new_param():
    """Create an uninitialized HSDPParamV2 with the common fields set."""
    param = object.__new__(HSDPParamV2)
    param.mesh_info = SimpleNamespace(mesh=FakeMesh())
    param.param_mode = FullyShardParamMode.LOCAL_PARAM
    param._orig_param_is_dtensor = False
    param.uses_param_shard = False
    param._spmd_shard_mesh_dim = None
    param._spmd_replicate_mesh_dim = None
    return param


def _new_mesh_info(cls, **attrs):
    """Create dataclass-like mesh-info instances without invoking validation."""
    mesh_info = object.__new__(cls)
    for key, value in attrs.items():
        setattr(mesh_info, key, value)
    return mesh_info


def _apply_to_tensors(fn, value):
    """Apply ``fn`` recursively to tensors while preserving container shape."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_apply_to_tensors(fn, item) for item in value)
    if isinstance(value, dict):
        return {key: _apply_to_tensors(fn, item) for key, item in value.items()}
    return value


def _cast_fp_tensor(dtype, value):
    """Cast a tensor value to the requested dtype."""
    return value.to(dtype)


class TestGroupInfoBuilders(unittest.TestCase):
    """Cover group-info construction without initializing distributed runtime."""

    def setUp(self):
        hsdp_param_mod._GROUP_INFO_CACHE.clear()

    def test_build_group_info_from_rank_list_invalid_single_rank(self):
        """Single-rank rank-list groups should be marked invalid."""
        info = _build_group_info_from_rank_list("dp", [0])

        self.assertEqual(info.group_name, "dp_invalid")
        self.assertIsNone(info.group)
        self.assertEqual(info.rank_size, 1)

    def test_build_group_info_from_rank_list_sorts_and_caches_group(self):
        """Rank-list groups should be sorted once and reused from the cache."""
        created_group = object()
        with patch.object(hsdp_param_mod.platform, "create_group", return_value=created_group) as create_group:
            first = _build_group_info_from_rank_list("dp", [3, 1, 2])
            second = _build_group_info_from_rank_list("dp", [2, 3, 1])

        create_group.assert_called_once_with([1, 2, 3])
        self.assertIs(first.group, created_group)
        self.assertIs(second.group, created_group)
        self.assertEqual(first.group_name, "(1, 2, 3)")
        self.assertEqual(first.rank_size, 3)

    def test_build_group_info_from_rank_list_handles_group_creation_error(self):
        """Group creation errors should leave group info without a process group."""
        with patch.object(hsdp_param_mod.platform, "create_group", side_effect=RuntimeError("not initialized")):
            info = _build_group_info_from_rank_list("dp", [0, 1])

        self.assertIsNone(info.group)
        self.assertEqual(info.rank_size, 2)

    def test_build_group_info_from_process_group_invalid_or_named(self):
        """Process-group info should handle invalid groups and resolved names."""
        invalid = _build_group_info_from_process_group("dp", None, 4)
        self.assertEqual(invalid.group_name, "dp_invalid")
        self.assertEqual(invalid.rank_size, 1)

        group = object()
        valid = _build_group_info_from_process_group("dp", group, 4, resolved_group_name="resolved")
        self.assertEqual(valid.group_name, "resolved")
        self.assertIs(valid.group, group)
        self.assertEqual(valid.rank_size, 4)


class TestAbstractMethods(unittest.TestCase):
    """The base class should stay abstract and raise clear NotImplementedError."""

    def test_abstract_methods_raise_not_implemented(self):
        """Abstract HSDPParamV2 methods and properties should raise NotImplementedError."""
        param = _new_param()
        method_calls = [
            partial(HSDPParamV2, None, None, None, None, None, None, None, None),
            partial(param._init_sharded_param, None, None),
            partial(param.init_dtype_attrs, None),
            partial(param.init_all_gather_outputs, [], [], 1, "cpu"),
            param.init_unsharded_param,
            param.to_sharded,
            param.to_unsharded,
            partial(param.to_sharded_dtensor, None),
            param.to_accumulated_grad_if_needed,
            param.accumulate_unsharded_grad_if_needed,
            param.alloc_all_gather_outputs,
            param.free_unsharded_param,
            param._get_unsharded_param_data,
            param.unshard,
            param.wait_for_unshard,
            param.shard,
            param.reduce_scatter_grad,
            param.all_reduce_grad,
            partial(getattr, param, "all_gather_inputs"),
            partial(getattr, param, "unsharded_param"),
            partial(getattr, param, "unsharded_grad_data"),
            partial(getattr, param, "unsharded_accumulated_grad_data"),
            partial(getattr, param, "_sharded_local_tensor"),
        ]

        for call in method_calls:
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError):
                    call()


class TestLayoutHelpers(unittest.TestCase):
    """Cover local, DTensor-compat, and data-parallel placement decisions."""

    def test_resolve_process_group_name_returns_requested_name(self):
        """Resolved process group names should use the requested name."""
        self.assertEqual(_new_param()._resolve_process_group_name("dp", object()), "dp")

    def test_get_base_spmd_placements_for_local_param(self):
        """Local params should start with replicate placements over the DP mesh."""
        param = _new_param()
        param.mesh_info = SimpleNamespace(mesh=FakeMesh(ndim=3))

        placements = param._get_base_spmd_placements()

        self.assertEqual(len(placements), 3)
        self.assertTrue(all(placement.is_replicate() for placement in placements))
        self.assertIs(param._spmd_mesh, param.mesh_info.mesh)

    def test_get_base_spmd_placements_for_compat_dtensor(self):
        """DTensor-compat params should keep original mesh and placements."""
        param = _new_param()
        orig_mesh = FakeMesh(ndim=2)
        orig_placements = (Shard(1), Replicate())
        param.param_mode = FullyShardParamMode.DTENSOR_COMPAT
        param._orig_param_is_dtensor = True
        param._orig_dtensor_mesh = orig_mesh
        param._orig_dtensor_placements = orig_placements

        placements = param._get_base_spmd_placements()

        self.assertEqual(placements, orig_placements)
        self.assertIs(param._spmd_mesh, orig_mesh)

    def test_get_base_spmd_placements_for_unified_dtensor_concatenates_mesh(self):
        """Unified DTensor params should concatenate DP and original meshes."""
        param = _new_param()
        dp_mesh = FakeMesh(ndim=2)
        orig_mesh = FakeMesh(ndim=1)
        combined_mesh = FakeMesh(ndim=3)
        param.mesh_info = SimpleNamespace(mesh=dp_mesh)
        param.param_mode = FullyShardParamMode.DTENSOR_UNIFIED
        param._orig_param_is_dtensor = True
        param._orig_dtensor_mesh = orig_mesh
        param._orig_dtensor_placements = (Shard(0),)

        with patch.object(hsdp_param_mod.DeviceMesh, "concatenate", return_value=combined_mesh) as concatenate:
            placements = param._get_base_spmd_placements()

        concatenate.assert_called_once_with([dp_mesh, orig_mesh])
        self.assertTrue(placements[0].is_replicate())
        self.assertTrue(placements[1].is_replicate())
        self.assertTrue(placements[2].is_shard())
        self.assertIs(param._spmd_mesh, combined_mesh)

    def test_apply_data_parallel_placements_validates_placement_count(self):
        """Data-parallel placement application should validate placement count."""
        param = _new_param()
        param._spmd_mesh = FakeMesh(ndim=2)

        with self.assertRaises(AssertionError):
            param._apply_data_parallel_placements([Replicate()], Shard(0))

    def test_apply_data_parallel_placements_uses_ddp_replicate_axis(self):
        """DDP params should write replicate placement on the replicate axis."""
        param = _new_param()
        param.mesh_info = _new_mesh_info(DDPMeshInfo)
        param._spmd_mesh = FakeMesh(ndim=2)
        param._spmd_replicate_mesh_dim = 1

        placements = param._apply_data_parallel_placements([Shard(0), Shard(1)], Shard(0))

        self.assertTrue(placements[0].is_shard())
        self.assertTrue(placements[1].is_replicate())

    def test_apply_data_parallel_placements_uses_fsdp_shard_axis(self):
        """FSDP params should write the data-parallel shard placement on shard axis."""
        param = _new_param()
        param.mesh_info = _new_mesh_info(FSDPMeshInfo)
        param._spmd_mesh = FakeMesh(ndim=2)
        param.uses_param_shard = True
        param._spmd_shard_mesh_dim = 0
        param._get_data_parallel_shard_placement = MagicMock(return_value=Shard(1))

        placements = param._apply_data_parallel_placements([Replicate(), Replicate()], Shard(0))

        param._get_data_parallel_shard_placement.assert_called_once()
        self.assertTrue(placements[0].is_shard())
        self.assertEqual(placements[0].dim, 1)

    def test_init_group_infos_sets_invalid_shard_group_for_unsharded_param(self):
        """Unsharded params should keep an invalid shard group and real unshard group."""
        param = _new_param()
        param._spmd_placements = (Replicate(),)
        param._spmd_mesh = FakeMesh(ndim=1)

        with patch.object(hsdp_param_mod, "get_rank_list_for_axes", return_value=[0, 1]):
            with patch.object(hsdp_param_mod.platform, "create_group", return_value="dp_group"):
                param._init_group_infos()

        self.assertEqual(param.sharded_group_info.rank_size, 1)
        self.assertEqual(param.unsharded_group_info.rank_size, 2)
        self.assertEqual(param.rank_size, 2)

    def test_init_group_infos_uses_fsdp_shard_process_group(self):
        """FSDP group init should use explicit shard process group metadata."""
        param = _new_param()
        param.uses_param_shard = True
        param.is_sharded = True
        param.mesh_info = _new_mesh_info(
            FSDPMeshInfo,
            shard_process_group="shard_group",
            shard_mesh_size=4,
        )
        param._resolve_process_group_name = MagicMock(return_value="resolved_shard")
        param._build_layout_driven_group_info = MagicMock(return_value=SimpleNamespace(rank_size=3))

        param._init_group_infos()

        self.assertEqual(param.sharded_group_info.group_name, "resolved_shard")
        self.assertEqual(param.sharded_group_info.group, "shard_group")
        self.assertEqual(param.shard_size, 4)
        self.assertEqual(param.dp_size, 3)
        self.assertEqual(param.rank_size, 12)

    def test_build_layout_driven_group_info_without_replicate_axis_is_invalid(self):
        """Layouts without replicate axes should produce an invalid unsharded group."""
        param = _new_param()
        param._spmd_mesh = FakeMesh(ndim=1)
        param._spmd_placements = (Shard(0),)

        info = param._build_layout_driven_group_info()

        self.assertEqual(info.group_name, "fully_shard_unsharded_group_invalid")
        self.assertEqual(info.rank_size, 1)

    def test_build_layout_driven_group_info_uses_named_single_axis_group(self):
        """Named single-axis replica layouts should resolve a mesh group directly."""
        param = _new_param()
        mesh_group = object()
        param._spmd_mesh = FakeMesh(ndim=2, names=("dp", "fsdp"), shape=(2, 4), group=mesh_group)
        param._spmd_placements = (Replicate(), Shard(0))
        param._resolve_process_group_name = MagicMock(return_value="named_dp")

        info = param._build_layout_driven_group_info()

        self.assertEqual(info.group_name, "named_dp")
        self.assertIs(info.group, mesh_group)
        self.assertEqual(info.rank_size, 2)

    def test_build_layout_driven_group_info_uses_named_split_group(self):
        """Named multi-axis replica layouts should use split-group construction."""
        param = _new_param()
        param._spmd_mesh = FakeMesh(ndim=3, names=("dp", "tp", "fsdp"), shape=(2, 3, 4), group=None)
        param._spmd_placements = (Replicate(), Replicate(), Shard(0))
        param._resolve_process_group_name = MagicMock(return_value="split_dp_tp")

        with patch.object(hsdp_param_mod, "get_split_rank_lists_for_axes", return_value=[[0, 1, 2]]) as split_lists:
            with patch.object(hsdp_param_mod.platform, "split_group", return_value="split_group") as split_group:
                info = param._build_layout_driven_group_info()

        split_lists.assert_called_once_with(param._spmd_mesh, [0, 1])
        split_group.assert_called_once_with(split_ranks=[[0, 1, 2]])
        self.assertEqual(info.group_name, "split_dp_tp")
        self.assertEqual(info.rank_size, 6)

    def test_build_layout_driven_group_info_falls_back_to_rank_list(self):
        """Layout-driven group construction should fall back to rank lists on split errors."""
        param = _new_param()
        param._spmd_mesh = FakeMesh(ndim=2, names=("dp", "tp"), shape=(2, 2), group=None)
        param._spmd_placements = (Replicate(), Replicate())

        with patch.object(hsdp_param_mod, "get_split_rank_lists_for_axes", side_effect=RuntimeError("bad mesh")):
            with patch.object(hsdp_param_mod, "get_rank_list_for_axes", return_value=[0, 2]) as rank_list:
                with patch.object(hsdp_param_mod.platform, "create_group", return_value="rank_group"):
                    info = param._build_layout_driven_group_info()

        rank_list.assert_called_once_with(param._spmd_mesh, [0, 1])
        self.assertEqual(info.group, "rank_group")
        self.assertEqual(info.rank_size, 2)


class TestCoreState(unittest.TestCase):
    """Cover shared HSDPState transitions with simple parameter doubles."""

    def _state(self, *, is_shard):
        """Create a core HSDPState double with matching shard flags."""
        state = object.__new__(HSDPState)
        state.is_shard = is_shard
        state.is_replicate_shard = is_shard
        state.sharded_hsdp_params = [SimpleNamespace(to_sharded=MagicMock(), unshard=MagicMock(), wait_for_unshard=MagicMock())]
        state.replicate_params = [
            SimpleNamespace(
                to_sharded=MagicMock(),
                unshard=MagicMock(),
                wait_for_unshard=MagicMock(),
                sharded_state=ShardedState.UNSHARDED if not is_shard else ShardedState.SHARDED,
            )
        ]
        state.hsdp_params = [SimpleNamespace(name="hsdp")]
        state.config = SimpleNamespace(comm_fusion=False)
        state.param_group = None
        return state

    def test_abstract_state_hooks_raise(self):
        """Abstract state hooks should raise NotImplementedError."""
        state = self._state(is_shard=True)

        with self.assertRaises(NotImplementedError):
            state._init_hsdp_params()
        with self.assertRaises(NotImplementedError):
            state._move_states_to_device()

    def test_shard_returns_early_or_shards_managed_params(self):
        """Shard should no-op when already sharded and shard all managed params otherwise."""
        sharded = self._state(is_shard=True)
        self.assertIsNone(sharded.shard())
        sharded.sharded_hsdp_params[0].to_sharded.assert_not_called()

        unsharded = self._state(is_shard=False)
        unsharded.shard()
        unsharded.sharded_hsdp_params[0].to_sharded.assert_called_once_with()
        unsharded.replicate_params[0].to_sharded.assert_called_once_with()
        self.assertTrue(unsharded.is_shard)

        unsharded = self._state(is_shard=False)
        unsharded.shard(shard_replicate=False)
        unsharded.replicate_params[0].to_sharded.assert_not_called()

    def test_unshard_prefetch_and_wait_cover_param_group_and_param_paths(self):
        """Unshard, prefetch, and wait should cover param-group and per-param paths."""
        state = self._state(is_shard=False)
        self.assertIsNone(state.unshard())
        state.sharded_hsdp_params[0].unshard.assert_not_called()

        state = self._state(is_shard=True)
        state.wait_for_unshard = MagicMock()
        state.unshard(async_op=False, unshard_replicate=True)
        state.replicate_params[0].unshard.assert_called_once_with(False)
        state.sharded_hsdp_params[0].unshard.assert_called_once_with(False)
        state.wait_for_unshard.assert_called_once_with(True)

        state = self._state(is_shard=True)
        state.config.comm_fusion = True
        state.param_group = SimpleNamespace(unshard=MagicMock(), wait_for_unshard=MagicMock())
        state.prefetch(unshard_replicate=False)
        state.replicate_params[0].unshard.assert_not_called()
        state.param_group.unshard.assert_called_once_with(True)

        state.wait_for_unshard(wait_for_replicate=False)
        state.replicate_params[0].wait_for_unshard.assert_not_called()
        state.param_group.wait_for_unshard.assert_called_once_with()
        self.assertFalse(state.is_shard)

    def test_wait_for_unshard_early_and_iter_managed_params(self):
        """Wait-for-unshard should no-op when unsharded and iterate managed params."""
        state = self._state(is_shard=False)
        self.assertIsNone(state.wait_for_unshard())
        self.assertEqual(state._iter_managed_params(), [*state.hsdp_params, *state.replicate_params])


class TestCoreScheduler(unittest.TestCase):
    """Cover shared scheduler hooks that platform schedulers delegate to."""

    def _scheduler(self):
        """Create an uninitialized scheduler with mocked platform and state."""
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.modules = ["cell"]
        scheduler.cell = "cell"
        scheduler.config = SimpleNamespace(reshard_after_forward=True)
        scheduler.reshard_after_forward = True
        scheduler.scheduler_state = None
        scheduler.scheduler_ctx = HSDPSchedulerContext()
        scheduler._is_root = False
        scheduler._fsdp_group_post_pending = None
        scheduler.forward_prefetch_cells = []
        scheduler.backward_prefetch_cells = []
        scheduler._backup_forward_fetch = None
        scheduler.mp_policy = SimpleNamespace(
            cast_forward_inputs=True,
            param_dtype=torch.float16,
            output_dtype=torch.float32,
        )
        scheduler.platform = SimpleNamespace(
            cast_fp_tensor=MagicMock(side_effect=_cast_fp_tensor),
            apply_to_tensors=MagicMock(side_effect=_apply_to_tensors),
            profiler_record=MagicMock(return_value=nullcontext()),
        )
        scheduler.hsdp_state = SimpleNamespace(
            module_name="root",
            unshard=MagicMock(),
            shard=MagicMock(),
            post_backward=MagicMock(),
            prefetch=MagicMock(),
            lazy_init=MagicMock(),
        )
        return scheduler

    def tearDown(self):
        HSDPSchedulerV2.root_bp_state = False

    def test_scheduler_context_and_abstract_methods(self):
        """Scheduler context defaults and platform hooks should stay abstract."""
        context = HSDPSchedulerContext()
        self.assertTrue(context.is_last_backward)
        self.assertIsNone(context.root_module)

        scheduler = self._scheduler()
        for method in (
            scheduler._init_platform,
            scheduler._new_cell_state,
            scheduler._register_hooks,
            scheduler._register_forward_backward_hooks,
        ):
            with self.subTest(method=method.__name__), self.assertRaises(NotImplementedError):
                method()

    def test_get_managed_params_delegates_to_shared_helper(self):
        """Managed param discovery should delegate to the shared helper."""
        scheduler = self._scheduler()
        scheduler.ignored_params = {"ignored"}
        with patch.object(hsdp_scheduler_mod, "get_managed_modules_parameters", return_value=["param"]) as helper:
            self.assertEqual(scheduler._get_managed_params(), ["param"])
        helper.assert_called_once_with(scheduler.modules, {"ignored"})

    def test_forward_pre_hook_initializes_root_casts_inputs_and_prefetches(self):
        """Pre-forward hook should initialize state, cast inputs, and prefetch."""
        scheduler = self._scheduler()
        prefetch_state = SimpleNamespace(module_name="next", prefetch=MagicMock())
        prefetch_cell = SimpleNamespace(hsdp_scheduler=SimpleNamespace(hsdp_state=prefetch_state))
        scheduler.forward_prefetch_cells = [prefetch_cell]
        scheduler._init_params_fqn = MagicMock()
        scheduler._lazy_init_all_states = MagicMock()
        input_arg = torch.ones(2, dtype=torch.float32)
        input_kwarg = torch.ones(1, dtype=torch.float32)

        with patch.object(hsdp_scheduler_mod.platform, "get_cells_and_names", return_value=[]):
            args, kwargs = scheduler._hsdp_forward_pre_hook("cell", (input_arg,), {"kw": input_kwarg})

        self.assertEqual(args[0].dtype, torch.float16)
        self.assertEqual(kwargs["kw"].dtype, torch.float16)
        torch.testing.assert_close(args[0].float(), input_arg)
        torch.testing.assert_close(kwargs["kw"].float(), input_kwarg)
        self.assertEqual(scheduler.scheduler_state, FSDPSchedulerState.PRE_FORWARD)
        scheduler._init_params_fqn.assert_called_once_with()
        scheduler._lazy_init_all_states.assert_called_once_with()
        prefetch_state.prefetch.assert_called_once_with()
        scheduler.hsdp_state.unshard.assert_called_once_with()

    def test_forward_pre_hook_returns_early_for_pre_backward_and_disables_recompute_prefetch(self):
        """Pre-forward should skip during pre-backward and disable recompute prefetch."""
        scheduler = self._scheduler()
        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        self.assertEqual(
            scheduler._hsdp_forward_pre_hook("cell", ("x",), {"kw": "y"}),
            (("x",), {"kw": "y"}),
        )

        scheduler = self._scheduler()
        HSDPSchedulerV2.root_bp_state = True
        scheduler._disable_forward_prefetch_for_recompute = MagicMock()
        scheduler._init_params_fqn = MagicMock()
        scheduler._lazy_init_all_states = MagicMock()
        with patch.object(hsdp_scheduler_mod.platform, "get_cells_and_names", return_value=[]):
            scheduler._hsdp_forward_pre_hook("cell", ("x",), {})
        scheduler._disable_forward_prefetch_for_recompute.assert_called_once_with()

    def test_lazy_init_and_param_fqn_assignments_use_root_module(self):
        """Lazy init and FQN assignment should use the scheduler root module."""
        scheduler = self._scheduler()
        scheduler._is_root = True
        scheduler.scheduler_ctx.root_module = "root"
        sharded_param = object()
        hsdp_param = SimpleNamespace(sharded_param=sharded_param)
        hsdp_state = SimpleNamespace(
            lazy_init=MagicMock(),
            _iter_managed_params=MagicMock(return_value=[hsdp_param]),
        )

        with patch.object(hsdp_scheduler_mod.platform, "get_cells_and_names", return_value=[("root", "module")]):
            with patch.object(hsdp_scheduler_mod, "get_hsdp_state", return_value=hsdp_state):
                scheduler._lazy_init_all_states()
                with patch.object(
                    hsdp_scheduler_mod.platform,
                    "parameters_dict",
                    return_value=[("weight", sharded_param), ("alias", sharded_param)],
                ):
                    scheduler._init_params_fqn()

        hsdp_state.lazy_init.assert_called_once_with()
        self.assertEqual(hsdp_param._param_fqn, "weight")

    def test_forward_backward_hooks_and_grouped_skip_defaults(self):
        """Forward/backward hooks should update state and grouped skips should no-op."""
        scheduler = self._scheduler()
        scheduler._init_params_fqn = MagicMock()
        scheduler._lazy_init_all_states = MagicMock()
        with patch.object(hsdp_scheduler_mod.platform, "get_cells_and_names", return_value=[]):
            scheduler._hsdp_forward_pre_hook("cell", (), {})

        output = scheduler._hsdp_forward_hook("cell", (), torch.ones(2, dtype=torch.float16))
        self.assertEqual(output.dtype, torch.float32)
        scheduler.hsdp_state.shard.assert_called_once_with(shard_replicate=False)

        prefetch_state = SimpleNamespace(module_name="prev", prefetch=MagicMock())
        scheduler.backward_prefetch_cells = [SimpleNamespace(hsdp_scheduler=SimpleNamespace(hsdp_state=prefetch_state))]
        scheduler._hsdp_backward_pre_hook("cell", None)
        prefetch_state.prefetch.assert_called_once_with(unshard_replicate=False)
        scheduler.hsdp_state.unshard.assert_called_with(unshard_replicate=False)

        scheduler._fsdp_group_post_pending = {object()}
        scheduler._hsdp_backward_hook("cell", None, None)
        scheduler.hsdp_state.post_backward.assert_called_once_with()
        self.assertEqual(scheduler._fsdp_group_post_pending, set())

        self.assertEqual(scheduler._grouped_forward_pre_hook_skip("cell", ("a",), {"k": "v"}), (("a",), {"k": "v"}))
        self.assertEqual(scheduler._grouped_forward_post_hook_skip("out"), "out")


if __name__ == "__main__":
    unittest.main()

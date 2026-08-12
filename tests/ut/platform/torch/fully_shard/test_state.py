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
"""Unit tests for fully_shard state helpers (no NPU required).

Covers _to_dtype_if_needed from hyper_parallel.platform.torch.fully_shard.state:
dtype no-op vs cast, and invalid input handling. All tests run on CPU.
"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, DDPMeshInfo, HSDPMeshInfo
from hyper_parallel.core.fully_shard.api import HSDPModule, _extend_module_with_hsdp_interface
from hyper_parallel.platform.torch.fully_shard import scheduler as scheduler_mod
from hyper_parallel.platform.torch.fully_shard import state as state_mod
from hyper_parallel.platform.torch.fully_shard.param_group import AllReduceParamGroup
from hyper_parallel.platform.torch.fully_shard.scheduler import TorchHSDPSchedulerV2
from hyper_parallel.platform.torch.fully_shard.state import TorchHSDPStateV2, _to_dtype_if_needed


class _FakeGroup:
    """Small process group double with only size() implemented."""

    def __init__(self, size=2):
        self._size = size

    def size(self):
        """Return fake process group size."""
        return self._size


class _FakeHSDPParam:
    """Small HSDP parameter double for state-machine branch coverage."""

    def __init__(
        self,
        *,
        requires_grad=True,
        is_sharded=True,
        shard_size=2,
        dp_size=2,
        grad=None,
        unsharded_grad=None,
        accumulated_grad=None,
        device=None,
    ):
        self.is_sharded = is_sharded
        self.shard_size = shard_size
        self.shard_world_size = shard_size
        self.dp_size = dp_size
        self.replicate_world_size = dp_size
        self.is_replicate_param = shard_size == 1
        mesh_info_type = DDPMeshInfo if self.is_replicate_param else HSDPMeshInfo
        self.mesh_info = object.__new__(mesh_info_type)
        self.mesh_info.replicate_process_group = _FakeGroup(dp_size)
        self._param_fqn = "module.weight"
        self.sharded_size = torch.Size((2,))
        self.padded_sharded_param_size = self.sharded_size
        self.sharded_param = SimpleNamespace(
            requires_grad=requires_grad,
            grad=grad,
            device=device or torch.device("cpu"),
            _local_tensor=torch.ones(2, device=device or torch.device("cpu")),
        )
        self._unsharded_param = SimpleNamespace(grad=unsharded_grad)
        self.unsharded_accumulated_grad = accumulated_grad
        self.unsharded_accumulated_grad_data = accumulated_grad
        self.unsharded_grad_data = unsharded_grad
        self.unsharded_group_info = GroupInfo("dp", _FakeGroup(dp_size), dp_size)
        self.orig_dtype = torch.float32
        self.reduce_dtype = torch.float32
        self.gradient_scaling_factor = None
        self.init_dtype_attrs = MagicMock()
        self.reset_sharded_param = MagicMock()
        self.accumulate_unsharded_grad_if_needed = MagicMock()
        self.to_accumulated_grad_if_needed = MagicMock()
        self.to_sharded = MagicMock()
        self.reduce_scatter_grad = MagicMock()
        self.all_reduce_grad = MagicMock()
        self.reduce_scatter_output = MagicMock(return_value=torch.ones(2))
        self.all_reduce_output = MagicMock(return_value=torch.ones(2))
        self.clear_reduce_scatter_output = MagicMock(side_effect=self._clear_reduce_scatter_output)
        self.clear_all_reduce_output = MagicMock(side_effect=self._clear_all_reduce_output)
        self.apply_reduced_grad = MagicMock(return_value=False)
        self.all_reduce_tp_replicate_grad_inplace = MagicMock()
        self.allgather_comm_ctx = SimpleNamespace(
            allgather_output=None,
            allgather_handle=None,
        )
        self.reduce_scatter_comm_ctx = SimpleNamespace(
            reduce_scatter_output=None,
            reduce_scatter_handle=None,
        )
        self.all_reduce_comm_ctx = SimpleNamespace(
            all_reduce_output=None,
            all_reduce_handle=None,
        )
        self.reduce_partial_output = None
        self.unsharded_param_buffers = (
            [torch.empty(0)]
            if unsharded_grad is not None or accumulated_grad is not None
            else []
        )
        self._grad = None

    def reduce_comm_dtype(self, grad=None) -> torch.dtype:
        """Return the effective reduction dtype for this parameter double."""
        if self.reduce_dtype is not None:
            return self.reduce_dtype
        if grad is not None:
            return grad.dtype
        if self.unsharded_accumulated_grad is not None:
            return self.unsharded_accumulated_grad_data.dtype
        return self.unsharded_grad_data.dtype

    def _clear_reduce_scatter_output(self) -> None:
        """Clear the fake reduce-scatter output like the production parameter."""
        self.reduce_scatter_comm_ctx.reduce_scatter_output = None
        self._grad = None

    def _clear_all_reduce_output(self) -> None:
        """Clear the fake all-reduce output like the production parameter."""
        self.all_reduce_comm_ctx.all_reduce_output = None

    @property
    def unsharded_param(self):
        return self._unsharded_param


def _new_state(hsdp_params=None, replicate_params=None, *, comm_fusion=False, offload_policy=None):
    """Create an uninitialized TorchHSDPStateV2 with direct state fields set."""
    state = object.__new__(TorchHSDPStateV2)
    state.modules = ()
    state.hsdp_params = list(hsdp_params or [])
    state.replicate_params = list(replicate_params or [])
    state.config = SimpleNamespace(comm_fusion=comm_fusion, comm_fusion_zero_copy=False)
    state.comm_fusion_policy = SimpleNamespace(
        enable_comm_fusion=comm_fusion,
        comm_fusion_zero_copy=False,
    )
    state.platform = SimpleNamespace()
    state.device = torch.device("cpu")
    state.mp_policy = None
    state.offload_policy = offload_policy
    state.reduce_grads = True
    state.reshard_after_backward = True
    state.requires_all_reduce = True
    state._user_reduce_op_type = None
    state.reduce_op_type = torch.distributed.ReduceOp.AVG
    state.comm_fusion = comm_fusion
    state.param_group = None
    state._reset_sharded_params = False
    state.is_shard = False
    return state


def _new_root_scheduler(state):
    """Create a root scheduler double that finalizes one state."""
    scheduler = object.__new__(TorchHSDPSchedulerV2)
    scheduler.hsdp_state = state
    scheduler.scheduler_ctx = SimpleNamespace(all_hsdp_schedulers=[scheduler])
    return scheduler


class TestToDtypeIfNeeded(unittest.TestCase):
    """Unit tests for _to_dtype_if_needed (tensor dtype cast or no-op)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_to_dtype_if_needed_parameterized(self):
        """Parameterized test: same dtype/no-op and None dtype (no cast).

        description: Call _to_dtype_if_needed with same dtype or None.
        expectation: Returns the same tensor object; dtype unchanged for None target.
        feature: fully_shard state _to_dtype_if_needed.
        """
        test_cases = [
            (torch.float32, torch.float32, "same dtype no-op"),
            (torch.float32, None, "None dtype no-op"),
        ]
        for tensor_dtype, target_dtype, desc in test_cases:
            with self.subTest(tensor_dtype=tensor_dtype, target_dtype=target_dtype, desc=desc):
                t = torch.randn(2, 2, dtype=tensor_dtype, device=self.device)
                result = _to_dtype_if_needed(t, target_dtype)
                self.assertIs(result, t)
                if target_dtype is not None:
                    self.assertEqual(result.dtype, target_dtype)
                else:
                    self.assertEqual(result.dtype, tensor_dtype)

    def test_to_dtype_converts_local_tensor(self):
        """Cast path: _to_dtype_if_needed converts local tensor when target dtype differs.

        description: Call _to_dtype_if_needed with float16/bfloat16 targets.
        expectation: New tensor with requested dtype (not same object as input).
        feature: fully_shard state _to_dtype_if_needed.
        """
        test_cases = [
            (torch.float32, torch.float16, "cast to float16"),
            (torch.float32, torch.bfloat16, "cast to bfloat16"),
        ]
        for tensor_dtype, target_dtype, desc in test_cases:
            with self.subTest(tensor_dtype=tensor_dtype, target_dtype=target_dtype, desc=desc):
                t = torch.randn(2, 2, dtype=tensor_dtype, device=self.device)
                result = _to_dtype_if_needed(t, target_dtype)
                self.assertIsNot(result, t)
                self.assertEqual(result.dtype, target_dtype)

    def test_invalid_tensor_raises(self):
        """_to_dtype_if_needed raises when first arg is not a tensor.

        description: Call _to_dtype_if_needed(None, torch.float32).
        expectation: AttributeError (None has no .dtype).
        feature: fully_shard state _to_dtype_if_needed input validation.
        """
        # Act & Assert (None has no .dtype attribute)
        with self.assertRaises(AttributeError):
            _to_dtype_if_needed(None, torch.float32)

class TestUnifiedParamTransitionState(unittest.TestCase):
    """Minimal state-machine coverage for unified managed parameters."""

    def test_prefetch_unshards_all_managed_params(self):
        """Prefetch should launch unshard for every parameter in hsdp_params."""
        state = object.__new__(HSDPState)
        state.is_shard = True
        state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=False)
        state.param_group = None
        state.hsdp_params = [MagicMock(), MagicMock()]

        state.prefetch()

        for hsdp_param in state.hsdp_params:
            hsdp_param.unshard.assert_called_once_with(True)

    def test_reshard_and_unshard_use_one_state_for_all_managed_params(self):
        """All managed parameters should transition through the same state."""
        state = object.__new__(HSDPState)
        state.is_shard = False
        state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=False)
        state.param_group = None
        state.hsdp_params = [MagicMock(), MagicMock()]

        state.shard()
        state.unshard()

        for hsdp_param in state.hsdp_params:
            hsdp_param.to_sharded.assert_called_once_with()
            hsdp_param.unshard.assert_called_once_with(False)
            hsdp_param.wait_for_unshard.assert_called_once_with()
        self.assertFalse(state.is_shard)

    def test_comm_fusion_without_param_group_falls_back_to_per_param_path(self):
        """States without a fused group should not dereference param_group under comm_fusion."""
        state = object.__new__(HSDPState)
        state.is_shard = True
        state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=True)
        state.param_group = None
        hsdp_param = MagicMock()
        state.hsdp_params = [hsdp_param]

        state.unshard()

        hsdp_param.unshard.assert_called_once_with(False)
        hsdp_param.wait_for_unshard.assert_called_once_with()

    def test_comm_fusion_transitions_all_params_through_param_group(self):
        """The parameter group owns sharded and replicate-only parameter transitions."""
        state = object.__new__(HSDPState)
        state.is_shard = True
        state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=True)
        state.param_group = MagicMock()
        fused_param = MagicMock()
        replicate_param = MagicMock()
        state.hsdp_params = [fused_param, replicate_param]

        state.unshard()

        state.param_group.unshard.assert_called_once_with(False)
        state.param_group.wait_for_unshard.assert_called_once_with()
        fused_param.unshard.assert_not_called()
        fused_param.wait_for_unshard.assert_not_called()
        replicate_param.unshard.assert_not_called()
        replicate_param.wait_for_unshard.assert_not_called()
        self.assertFalse(state.is_shard)

    def test_comm_fusion_scaling_is_owned_by_param_group(self):
        """The parameter group applies scaling for every parameter it owns."""
        state = object.__new__(HSDPState)
        state.param_group = SimpleNamespace(gradient_scaling_factor=None)
        replicate_param = SimpleNamespace(gradient_scaling_factor=None)
        state.hsdp_params = [replicate_param]

        state.set_gradient_scaling_factor(2.0)

        self.assertEqual(state.param_group.gradient_scaling_factor, 2.0)
        self.assertIsNone(replicate_param.gradient_scaling_factor)


class TestTorchHSDPStateV2(unittest.TestCase):
    """Unit tests for TorchHSDPStateV2 branch helpers."""

    def tearDown(self):
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        TorchHSDPStateV2.pre_all_reduce_groups.clear()
        TorchHSDPStateV2.all_reduce_work_groups.clear()

    def test_init_param_group_skips_when_disabled_and_constructs_when_enabled(self):
        """Param-group init should include sharded and replicate-only parameters."""
        disabled = _new_state([_FakeHSDPParam()], comm_fusion=False)
        TorchHSDPStateV2._init_param_group(disabled)
        self.assertIsNone(disabled.param_group)

        sharded = _FakeHSDPParam()
        replicated = _FakeHSDPParam(shard_size=1)
        supported = _new_state([sharded, replicated], comm_fusion=True)
        with patch.object(state_mod, "HSDPParamGroup", return_value="group") as param_group:
            TorchHSDPStateV2._init_param_group(supported)

        param_group.assert_called_once()
        self.assertEqual(param_group.call_args.args[0], [sharded, replicated])
        self.assertEqual(supported.param_group, "group")

    def test_set_requires_all_reduce_propagates_to_param_group(self):
        """The state switch should update both state and active parameter group."""
        state = _new_state(comm_fusion=True)
        state.param_group = SimpleNamespace(requires_all_reduce=True)

        state.set_requires_all_reduce(False)

        self.assertFalse(state.requires_all_reduce)
        self.assertFalse(state.param_group.requires_all_reduce)

    def test_init_mp_dtypes_initializes_each_param_independently(self):
        """Mixed-precision dtype init should not require state-wide uniform dtypes."""
        first = _FakeHSDPParam()
        second = _FakeHSDPParam()
        state = _new_state([first, second])

        TorchHSDPStateV2._init_mp_dtypes(state)

        first.init_dtype_attrs.assert_called_once_with(None)
        second.init_dtype_attrs.assert_called_once_with(None)
        self.assertFalse(hasattr(state, "_orig_dtype"))
        self.assertFalse(hasattr(state, "_reduce_dtype"))

        second.orig_dtype = torch.float16
        second.reduce_dtype = torch.float16
        TorchHSDPStateV2._init_mp_dtypes(state)

    def test_validation_helpers_cover_meta_and_cpu_offload(self):
        """Validation helpers should reject meta params and invalid CPU offload params."""
        meta_param = _FakeHSDPParam(device=torch.device("meta"))
        state = _new_state([meta_param])
        with self.assertRaisesRegex(RuntimeError, "meta device"):
            TorchHSDPStateV2._validate_no_meta_params(state)

        cpu_offload_state = _new_state([_FakeHSDPParam()], offload_policy=CPUOffloadPolicy())
        TorchHSDPStateV2._validate_cpu_offload_params(cpu_offload_state)

        non_cpu_param = _FakeHSDPParam()
        non_cpu_param.sharded_param.device = torch.device("meta")
        non_cpu_state = _new_state([non_cpu_param], offload_policy=CPUOffloadPolicy())
        with self.assertRaisesRegex(RuntimeError, "CPU offloading"):
            TorchHSDPStateV2._validate_cpu_offload_params(non_cpu_state)

    def test_lazy_init_resets_sharded_params_once_and_initializes_dtypes(self):
        """Lazy init should reset sharded params once and initialize dtype state."""
        param = _FakeHSDPParam()
        state = _new_state([param])
        state.is_shard = True

        TorchHSDPStateV2.lazy_init(state)
        TorchHSDPStateV2.lazy_init(state)

        param.reset_sharded_param.assert_called_once()
        self.assertTrue(state._reset_sharded_params)

    def test_no_sync_reduce_scatter_accumulates_fp32_partial_without_applying(self):
        """No-sync micro-steps should accumulate RS outputs without touching parameter grads."""
        sharded_grad = torch.tensor([100.0, 100.0], dtype=torch.bfloat16)
        param = _FakeHSDPParam(dp_size=2, grad=sharded_grad.clone())
        param.reduce_scatter_output.side_effect = [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 4.0], dtype=torch.float32),
        ]
        state = _new_state([param])
        state.requires_all_reduce = False

        for _ in range(2):
            HSDPState.pre_reduce_scatter_params.append(param)
            state._wait_prev_reduce_scatter_without_all_reduce()

        torch.testing.assert_close(
            param.reduce_partial_output,
            torch.tensor([4.0, 6.0], dtype=torch.float32),
        )
        self.assertEqual(param.reduce_partial_output.dtype, torch.float32)
        torch.testing.assert_close(param.sharded_param.grad, sharded_grad)
        param.apply_reduced_grad.assert_not_called()

    def test_final_size_one_mesh_merges_partial_and_applies_once_at_root(self):
        """Both (1,) and (1, 1) meshes should finalize through the same local path."""
        for mesh_shape in ((1,), (1, 1)):
            with self.subTest(mesh_shape=mesh_shape):
                current_output = torch.tensor([3.0, 4.0], dtype=torch.float32)
                param = _FakeHSDPParam(dp_size=1)
                param.reduce_partial_output = torch.tensor([1.0, 2.0], dtype=torch.float32)
                param.reduce_scatter_output.return_value = current_output
                param.reduce_scatter_comm_ctx.reduce_scatter_output = current_output
                state = _new_state([param])

                HSDPState.pre_reduce_scatter_params.append(param)
                state._wait_prev_reduce_scatter_without_all_reduce()

                self.assertIsNone(param.reduce_partial_output)
                torch.testing.assert_close(current_output, torch.tensor([4.0, 6.0]))
                param.apply_reduced_grad.assert_not_called()

                scheduler = _new_root_scheduler(state)
                scheduler._finalize_per_param_reductions()
                param.apply_reduced_grad.assert_not_called()
                scheduler.launch_tp_replicate_reduce_and_apply()

                param.all_reduce_tp_replicate_grad_inplace.assert_called_once_with(
                    current_output,
                    torch.distributed.ReduceOp.AVG,
                )
                param.apply_reduced_grad.assert_called_once_with(current_output)

    def test_all_reduce_group_consumes_partial_output_in_reduce_dtype(self):
        """The final HSDP AR buffer should consume partial RS output, not parameter grad."""
        param = _FakeHSDPParam(dp_size=2)
        param.reduce_partial_output = torch.tensor([1.0, 2.0], dtype=torch.float32)
        param.sharded_param.grad = torch.tensor([100.0, 100.0], dtype=torch.bfloat16)
        group = AllReduceParamGroup(
            replicate_group=param.mesh_info.replicate_process_group,
            hsdp_params=[param],
            reduce_op=torch.distributed.ReduceOp.AVG,
        )
        group.allocate_fused_buffer(torch.device("cpu"))
        group.get_param_buffer_view(0).copy_(torch.tensor([3.0, 4.0]))

        group.accumulate_reduce_partial_outputs()

        torch.testing.assert_close(group.get_param_buffer_view(0), torch.tensor([4.0, 6.0]))
        self.assertIsNone(param.reduce_partial_output)
        torch.testing.assert_close(
            param.sharded_param.grad,
            torch.tensor([100.0, 100.0], dtype=torch.bfloat16),
        )

    def test_root_finalization_applies_reduced_grad_only_once(self):
        """Repeated root finalization should not accumulate the same reduced gradient twice."""
        current_output = torch.tensor([3.0, 4.0])
        param = _FakeHSDPParam(dp_size=1, grad=torch.tensor([10.0, 10.0]))
        param.reduce_partial_output = torch.tensor([1.0, 2.0])
        param.reduce_scatter_output.return_value = current_output
        param.reduce_scatter_comm_ctx.reduce_scatter_output = current_output

        def apply_reduced_grad(grad):
            param.sharded_param.grad.add_(grad)
            return False

        param.apply_reduced_grad.side_effect = apply_reduced_grad
        state = _new_state([param])

        HSDPState.pre_reduce_scatter_params.append(param)
        state._wait_prev_reduce_scatter_without_all_reduce()
        scheduler = _new_root_scheduler(state)
        scheduler.launch_tp_replicate_reduce_and_apply()
        scheduler.launch_tp_replicate_reduce_and_apply()

        torch.testing.assert_close(param.sharded_param.grad, torch.tensor([14.0, 16.0]))
        param.apply_reduced_grad.assert_called_once_with(current_output)

    def test_post_backward_no_reduce_accumulates_and_reshards(self):
        """Post-backward without reduction should accumulate grads and reshard."""
        param = _FakeHSDPParam()
        state = _new_state([param])
        state.reduce_grads = False
        state.shard = MagicMock()

        TorchHSDPStateV2.post_backward(state)

        param.accumulate_unsharded_grad_if_needed.assert_called_once()
        param.to_accumulated_grad_if_needed.assert_called_once()
        state.shard.assert_called_once()

    def test_post_backward_groups_sharded_and_replicated_params_by_process_group(self):
        """Post-backward should route sharded and replicated params through the unified pipeline."""
        sharded = _FakeHSDPParam(unsharded_grad=torch.ones(2), shard_size=2)
        replicated = _FakeHSDPParam(unsharded_grad=torch.ones(2), shard_size=1)
        state = _new_state([sharded, replicated])
        state.shard = MagicMock()

        TorchHSDPStateV2.post_backward(state)

        sharded.reduce_scatter_grad.assert_called_once()
        replicated.reduce_scatter_grad.assert_called_once()
        self.assertEqual(len(TorchHSDPStateV2.pre_all_reduce_groups), 2)
        state.shard.assert_called_once()

    def test_post_backward_for_comm_fusion_drains_context_and_launches_param_group(self):
        """Comm-fusion post-backward should drain prior groups before launching this group."""
        replicate = _FakeHSDPParam(unsharded_grad=torch.ones(2), shard_size=1)
        param_group = SimpleNamespace(foreach_reduce=MagicMock())
        state = _new_state([replicate], comm_fusion=True)
        state.param_group = param_group
        previous_all_reduce = SimpleNamespace(wait_all_reduce_and_save_grad=MagicMock())
        previous_reduce_scatter = SimpleNamespace(wait_reduce_scatter_and_issue_all_reduce=MagicMock())
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=previous_all_reduce,
            pre_param_group=previous_reduce_scatter,
        )

        with patch.object(state_mod, "get_comm_ctx", return_value=comm_ctx):
            TorchHSDPStateV2.post_backward_for_comm_fusion(state)

        previous_all_reduce.wait_all_reduce_and_save_grad.assert_called_once()
        previous_reduce_scatter.wait_reduce_scatter_and_issue_all_reduce.assert_called_once()
        param_group.foreach_reduce.assert_called_once_with(
            reduce_scatter_reduce_op=torch.distributed.ReduceOp.AVG
        )
        replicate.reduce_scatter_grad.assert_not_called()
        replicate.all_reduce_grad.assert_not_called()
        self.assertIsNone(comm_ctx.all_reduce_param_group)
        self.assertIsNone(comm_ctx.pre_param_group)

    def test_root_comm_fusion_drain_finalizes_last_replicate_param(self):
        """Root backward should apply the replicate result saved by the parameter group."""
        reduced_grad = torch.ones(2)
        param = _FakeHSDPParam(shard_size=1)
        param.all_reduce_comm_ctx.all_reduce_output = reduced_grad
        state = _new_state([param], comm_fusion=True)
        scheduler = _new_root_scheduler(state)
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=None,
            pre_param_group=None,
        )

        with patch.object(scheduler_mod, "get_comm_ctx", return_value=comm_ctx):
            scheduler._finalize_comm_fusion_reductions()
        param.apply_reduced_grad.assert_not_called()
        scheduler.launch_tp_replicate_reduce_and_apply()

        param.all_reduce_tp_replicate_grad_inplace.assert_called_once_with(
            reduced_grad,
            torch.distributed.ReduceOp.AVG,
        )
        param.apply_reduced_grad.assert_called_once_with(reduced_grad)

    def test_per_param_root_uses_fresh_reduced_grad_instead_of_existing_grad(self):
        """The per-parameter root path must not TP-reduce an optimizer gradient again."""
        existing_grad = torch.full((2,), 10.0)
        reduced_grad = torch.tensor([1.0, 2.0])
        param = _FakeHSDPParam(grad=existing_grad)
        param.reduce_scatter_comm_ctx.reduce_scatter_output = reduced_grad
        state = _new_state([param], comm_fusion=False)
        state._wait_prev_reduce_scatter = MagicMock(return_value=[])
        state._wait_prev_reduce_scatter_without_all_reduce = MagicMock()
        state._issue_prev_fused_all_reduce = MagicMock()
        scheduler = _new_root_scheduler(state)

        with patch.object(
            TorchHSDPStateV2,
            "wait_and_split_all_reduce_work_groups",
        ):
            scheduler._finalize_per_param_reductions()
        param.apply_reduced_grad.assert_not_called()
        scheduler.launch_tp_replicate_reduce_and_apply()

        param.all_reduce_tp_replicate_grad_inplace.assert_called_once_with(
            reduced_grad,
            torch.distributed.ReduceOp.AVG,
        )
        param.apply_reduced_grad.assert_called_once_with(reduced_grad)
        torch.testing.assert_close(param.sharded_param.grad, existing_grad)

    def test_comm_fusion_drain_does_not_consume_non_fusion_output(self):
        """Fusion queue draining must leave per-parameter RS outputs for their own wait path."""
        reduced_grad = torch.tensor([1.0, 2.0])
        param = _FakeHSDPParam()
        param._grad = torch.tensor([3.0, 4.0])
        param.reduce_scatter_comm_ctx.reduce_scatter_output = reduced_grad
        param.reduce_scatter_comm_ctx.reduce_scatter_handle = MagicMock()
        state = _new_state([param], comm_fusion=False)
        scheduler = _new_root_scheduler(state)
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=None,
            pre_param_group=None,
        )

        with patch.object(scheduler_mod, "get_comm_ctx", return_value=comm_ctx):
            scheduler._finalize_comm_fusion_reductions()

        param.apply_reduced_grad.assert_not_called()
        param.clear_reduce_scatter_output.assert_not_called()
        self.assertIs(param.reduce_scatter_comm_ctx.reduce_scatter_output, reduced_grad)
        self.assertIsNotNone(param.reduce_scatter_comm_ctx.reduce_scatter_handle)
        self.assertIsNotNone(param._grad)

    def test_root_finalization_drains_and_applies_mixed_fusion_modes(self):
        """A fused root should drain both DP paths and apply outputs from non-fused children."""
        fusion_output = torch.tensor([1.0, 2.0])
        per_param_output = torch.tensor([3.0, 4.0])
        fusion_param = _FakeHSDPParam()
        fusion_param.all_reduce_comm_ctx.all_reduce_output = fusion_output
        per_param = _FakeHSDPParam()
        per_param.reduce_scatter_comm_ctx.reduce_scatter_output = per_param_output
        fusion_state = _new_state([fusion_param], comm_fusion=True)
        per_param_state = _new_state([per_param], comm_fusion=False)
        fusion_state._wait_prev_reduce_scatter = MagicMock(return_value=[])
        fusion_state._wait_prev_reduce_scatter_without_all_reduce = MagicMock()
        fusion_state._issue_prev_fused_all_reduce = MagicMock()
        root_scheduler = _new_root_scheduler(fusion_state)
        child_scheduler = _new_root_scheduler(per_param_state)
        scheduler_ctx = SimpleNamespace(
            all_hsdp_schedulers=[root_scheduler, child_scheduler],
        )
        root_scheduler.scheduler_ctx = scheduler_ctx
        child_scheduler.scheduler_ctx = scheduler_ctx
        previous_all_reduce = SimpleNamespace(wait_all_reduce_and_save_grad=MagicMock())
        previous_reduce_scatter = SimpleNamespace(wait_reduce_scatter_and_issue_all_reduce=MagicMock())
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=previous_all_reduce,
            pre_param_group=previous_reduce_scatter,
        )

        with patch.object(scheduler_mod, "get_comm_ctx", return_value=comm_ctx):
            root_scheduler._finalize_comm_fusion_reductions()
        root_scheduler._finalize_per_param_reductions()
        root_scheduler.launch_tp_replicate_reduce_and_apply()

        previous_all_reduce.wait_all_reduce_and_save_grad.assert_called_once_with()
        previous_reduce_scatter.wait_reduce_scatter_and_issue_all_reduce.assert_called_once_with()
        fusion_state._wait_prev_reduce_scatter.assert_called_once_with()
        fusion_state._wait_prev_reduce_scatter_without_all_reduce.assert_called_once_with()
        fusion_state._issue_prev_fused_all_reduce.assert_called_once_with([])
        fusion_param.apply_reduced_grad.assert_called_once_with(fusion_output)
        per_param.apply_reduced_grad.assert_called_once_with(per_param_output)

    def test_state_reset_releases_iteration_state_and_preserves_optimizer_grads(self):
        """State reset should release communication storage but preserve optimizer gradients."""
        grad = torch.tensor([3.0, 4.0])
        main_grad = torch.tensor([5.0, 6.0])
        param = _FakeHSDPParam(
            grad=grad,
            unsharded_grad=torch.ones(2),
            accumulated_grad=torch.ones(2),
        )
        param.sharded_param.main_grad = main_grad
        param.unsharded_param_buffers = [torch.ones(2)]
        param.allgather_comm_ctx.allgather_output = torch.ones(2)
        param.allgather_comm_ctx.allgather_handle = MagicMock()
        param.reduce_scatter_comm_ctx.reduce_scatter_output = torch.ones(2)
        param.reduce_scatter_comm_ctx.reduce_scatter_handle = MagicMock()
        param.all_reduce_comm_ctx.all_reduce_output = torch.ones(2)
        param.all_reduce_comm_ctx.all_reduce_handle = MagicMock()
        param.reduce_partial_output = torch.ones(2)
        param._grad = torch.ones(2)
        state = _new_state([param], comm_fusion=True)
        state.param_group = MagicMock()

        pre_group = SimpleNamespace()
        work_group = SimpleNamespace()
        HSDPState.pre_reduce_scatter_params.append(param)
        HSDPState.pre_all_reduce_params.append(param)
        TorchHSDPStateV2.pre_all_reduce_groups.append(pre_group)
        TorchHSDPStateV2.all_reduce_work_groups.append(work_group)

        state.reset_iter_state()

        state.param_group.reset_iter_state.assert_called_once_with()
        self.assertEqual(HSDPState.pre_reduce_scatter_params, [])
        self.assertEqual(HSDPState.pre_all_reduce_params, [])
        self.assertEqual(TorchHSDPStateV2.pre_all_reduce_groups, [])
        self.assertEqual(TorchHSDPStateV2.all_reduce_work_groups, [])
        self.assertIsNone(param.allgather_comm_ctx.allgather_output)
        self.assertIsNone(param.allgather_comm_ctx.allgather_handle)
        self.assertIsNone(param.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIsNone(param.reduce_scatter_comm_ctx.reduce_scatter_handle)
        self.assertIsNone(param.all_reduce_comm_ctx.all_reduce_output)
        self.assertIsNone(param.all_reduce_comm_ctx.all_reduce_handle)
        self.assertIsNone(param.reduce_partial_output)
        self.assertIsNone(param.unsharded_accumulated_grad)
        self.assertIsNone(param.unsharded_param.grad)
        self.assertIs(param.sharded_param.grad, grad)
        self.assertIs(param.sharded_param.main_grad, main_grad)

    def test_state_reset_clears_shared_queues_without_touching_another_state(self):
        """State reset should clear shared queues and only release its own parameter contexts."""
        current_param = _FakeHSDPParam(unsharded_grad=torch.ones(2))
        other_param = _FakeHSDPParam(unsharded_grad=torch.ones(2))
        current_param.reduce_scatter_comm_ctx.reduce_scatter_output = torch.ones(2)
        other_reduce_scatter_output = torch.full((2,), 2.0)
        other_all_reduce_output = torch.full((2,), 3.0)
        other_param.reduce_scatter_comm_ctx.reduce_scatter_output = other_reduce_scatter_output
        other_param.all_reduce_comm_ctx.all_reduce_output = other_all_reduce_output
        other_param.reduce_partial_output = torch.full((2,), 4.0)
        current_state = _new_state([current_param])
        other_state = _new_state([other_param])
        other_state.param_group = MagicMock()

        HSDPState.pre_reduce_scatter_params.extend([current_param, other_param])
        HSDPState.pre_all_reduce_params.extend([current_param, other_param])

        current_state.reset_iter_state()

        self.assertEqual(HSDPState.pre_reduce_scatter_params, [])
        self.assertEqual(HSDPState.pre_all_reduce_params, [])
        self.assertIsNone(current_param.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIs(other_param.reduce_scatter_comm_ctx.reduce_scatter_output, other_reduce_scatter_output)
        self.assertIs(other_param.all_reduce_comm_ctx.all_reduce_output, other_all_reduce_output)
        torch.testing.assert_close(other_param.reduce_partial_output, torch.full((2,), 4.0))
        other_state.param_group.reset_iter_state.assert_not_called()

    def test_scheduler_reset_clears_pipeline_and_recompute_state(self):
        """Scheduler reset should clear hooks and restore prefetch configuration."""
        scheduler = object.__new__(TorchHSDPSchedulerV2)
        scheduler.scheduler_state = object()
        scheduler._fsdp_group_post_pending = {object()}
        scheduler._backup_forward_fetch = ["prefetch"]
        scheduler.forward_prefetch_cells = []
        scheduler.hsdp_state = MagicMock()
        comm_ctx = SimpleNamespace(
            pre_param_group=object(),
            all_reduce_param_group=object(),
        )
        HSDPSchedulerV2.root_bp_state = True

        with patch.object(scheduler_mod, "get_comm_ctx", return_value=comm_ctx):
            scheduler.reset_iter_state()

        self.assertFalse(HSDPSchedulerV2.root_bp_state)
        self.assertIsNone(scheduler.scheduler_state)
        self.assertEqual(scheduler._fsdp_group_post_pending, set())
        self.assertEqual(scheduler.forward_prefetch_cells, ["prefetch"])
        self.assertIsNone(scheduler._backup_forward_fetch)
        self.assertIsNone(comm_ctx.pre_param_group)
        self.assertIsNone(comm_ctx.all_reduce_param_group)
        scheduler.hsdp_state.reset_iter_state.assert_called_once_with()

    def test_module_reset_iter_state_respects_recursive_scope(self):
        """The public reset API should support current-module and recursive scopes."""
        child = torch.nn.Linear(2, 2)
        model = torch.nn.Sequential(child)
        _extend_module_with_hsdp_interface(child)
        _extend_module_with_hsdp_interface(model)
        self.assertIsInstance(model, HSDPModule)
        self.assertIsInstance(child, HSDPModule)
        model.hsdp_scheduler = MagicMock()
        child.hsdp_scheduler = MagicMock()

        model.reset_iter_state(recursive=False)

        model.hsdp_scheduler.reset_iter_state.assert_called_once_with()
        child.hsdp_scheduler.reset_iter_state.assert_not_called()

        model.hsdp_scheduler.reset_iter_state.reset_mock()
        model.reset_iter_state(recursive=True)

        model.hsdp_scheduler.reset_iter_state.assert_called_once_with()
        child.hsdp_scheduler.reset_iter_state.assert_called_once_with()

    def test_reduce_op_setter_accepts_known_values_and_rejects_unknown(self):
        """Reduce-op setter should accept known reductions and reject unknown names."""
        state = _new_state()

        TorchHSDPStateV2.set_requires_grad_sync(state, False)
        self.assertFalse(state.reduce_grads)
        TorchHSDPStateV2.set_reduce_op_type(state, "sum")
        self.assertEqual(state.reduce_op_type, torch.distributed.ReduceOp.SUM)

        with self.assertRaises(ValueError):
            TorchHSDPStateV2.set_reduce_op_type(state, "mean")


if __name__ == "__main__":
    unittest.main()

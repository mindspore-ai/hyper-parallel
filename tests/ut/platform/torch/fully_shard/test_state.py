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
from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode, GroupInfo, ShardedState
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.platform.torch.fully_shard import state as state_mod
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
        param_mode=FullyShardParamMode.LOCAL_PARAM,
        enable_fsdp_shard=True,
        is_sharded=True,
        shard_size=2,
        dp_size=2,
        grad=None,
        unsharded_grad=None,
        accumulated_grad=None,
        device=None,
    ):
        self.param_mode = param_mode
        self.enable_fsdp_shard = enable_fsdp_shard
        self.is_sharded = is_sharded
        self.shard_size = shard_size
        self.shard_world_size = shard_size
        self.dp_size = dp_size
        self._param_fqn = "module.weight"
        self._sharded_local_tensor = torch.ones(2, device=device or torch.device("cpu"))
        self.sharded_size = torch.Size((2,))
        self.sharded_param = SimpleNamespace(
            requires_grad=requires_grad,
            grad=grad,
            device=device or torch.device("cpu"),
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
        self.clear_reduce_scatter_output = MagicMock()
        self.clear_all_reduce_output = MagicMock()
        self.apply_reduced_grad = MagicMock(return_value=False)

    @property
    def unsharded_param(self):
        return self._unsharded_param


class _GradWithToLocal:
    """Gradient double that mimics DTensor.to_local()."""

    def __init__(self, local_grad):
        self.local_grad = local_grad

    def to_local(self):
        return self.local_grad


def _new_state(hsdp_params=None, replicate_params=None, *, comm_fusion=False, offload_policy=None):
    """Create an uninitialized TorchHSDPStateV2 with direct state fields set."""
    state = object.__new__(TorchHSDPStateV2)
    state.modules = ()
    state.hsdp_params = list(hsdp_params or [])
    state.sharded_hsdp_params = [param for param in state.hsdp_params if param.is_sharded]
    state.replicate_params = list(replicate_params or [])
    state.config = SimpleNamespace(comm_fusion=comm_fusion, comm_fusion_zero_copy=False)
    state.mesh_info = SimpleNamespace()
    state.platform = SimpleNamespace()
    state.device = torch.device("cpu")
    state.mp_policy = None
    state.offload_policy = offload_policy
    state.reduce_grads = True
    state.reshard_after_backward = True
    state.requires_all_reduce = True
    state._orig_dtype = torch.float32
    state._reduce_dtype = torch.float32
    state._user_reduce_op_type = None
    state.reduce_op_type = torch.distributed.ReduceOp.AVG
    state.comm_fusion = comm_fusion
    state.param_group = None
    state._reset_sharded_params = False
    state.is_shard = False
    return state


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

class TestReplicateParamTransitionState(unittest.TestCase):
    """Minimal state-machine coverage for replicate_params transitions."""

    def test_backward_prefetch_skips_replicate_params(self):
        """backward prefetch must not unshard already-materialized replicate params."""
        state = object.__new__(HSDPState)
        state.is_shard = True
        state.is_replicate_shard = False
        state.config = SimpleNamespace(comm_fusion=False)
        state.replicate_params = [MagicMock()]
        state.sharded_hsdp_params = [MagicMock()]

        state.prefetch(unshard_replicate=False)

        state.replicate_params[0].unshard.assert_not_called()
        state.sharded_hsdp_params[0].unshard.assert_called_once_with(True)

    def test_default_unshard_skips_already_unsharded_replicate_param_after_forward_reshard(self):
        """Default unshard should be idempotent for already-materialized replicate params."""
        state = object.__new__(HSDPState)
        state.is_shard = False
        state.is_replicate_shard = False
        state.config = SimpleNamespace(comm_fusion=False)
        replicate_param = MagicMock()
        replicate_param.uses_param_shard = False
        replicate_param.sharded_state = ShardedState.UNSHARDED
        sharded_param = MagicMock()
        state.replicate_params = [replicate_param]
        state.sharded_hsdp_params = [sharded_param]

        state.shard(shard_replicate=False)
        state.unshard()

        replicate_param.unshard.assert_not_called()
        replicate_param.wait_for_unshard.assert_not_called()
        sharded_param.unshard.assert_called_once_with(False)
        sharded_param.wait_for_unshard.assert_called_once_with()

    def test_default_unshard_rejects_stale_replicate_state_after_forward_reshard(self):
        """A stale replicate state should still fail instead of being silently skipped."""
        state = object.__new__(HSDPState)
        state.is_shard = False
        state.is_replicate_shard = False
        state.config = SimpleNamespace(comm_fusion=False)
        replicate_param = MagicMock()
        replicate_param._param_fqn = "weight"
        replicate_param.sharded_state = ShardedState.SHARDED
        sharded_param = MagicMock()
        state.replicate_params = [replicate_param]
        state.sharded_hsdp_params = [sharded_param]

        state.shard(shard_replicate=False)
        with self.assertRaisesRegex(AssertionError, "Expected replicate parameter weight"):
            state.unshard()

        replicate_param.unshard.assert_not_called()
        replicate_param.wait_for_unshard.assert_not_called()

    def test_comm_fusion_without_param_group_falls_back_to_per_param_path(self):
        """States without a fused group should not dereference param_group under comm_fusion."""
        state = object.__new__(HSDPState)
        state.is_shard = True
        state.is_replicate_shard = True
        state.config = SimpleNamespace(comm_fusion=True)
        state.param_group = None
        state.replicate_params = []
        sharded_param = MagicMock()
        state.sharded_hsdp_params = [sharded_param]

        state.unshard()

        sharded_param.unshard.assert_called_once_with(False)
        sharded_param.wait_for_unshard.assert_called_once_with()


class TestTorchHSDPStateV2(unittest.TestCase):
    """Unit tests for TorchHSDPStateV2 branch helpers."""

    def tearDown(self):
        HSDPState.pre_reduce_scatter_params.clear()
        HSDPState.pre_all_reduce_params.clear()
        TorchHSDPStateV2.pre_direct_all_reduce_grads.clear()
        TorchHSDPStateV2.pre_all_reduce_groups.clear()
        TorchHSDPStateV2.pending_all_reduce_groups.clear()

    def test_pending_grad_helpers_cover_accumulated_unsharded_and_local_grad(self):
        """Pending-grad helpers should prefer accumulated, unsharded, and local grads."""
        local_grad = torch.ones(2)
        accumulated = torch.full((2,), 2.0)
        unsharded = torch.full((2,), 3.0)
        param = _FakeHSDPParam(
            grad=_GradWithToLocal(local_grad),
            unsharded_grad=unsharded,
            accumulated_grad=accumulated,
        )

        self.assertIs(TorchHSDPStateV2._get_pending_unsharded_grad(param), accumulated)
        self.assertTrue(TorchHSDPStateV2._has_pending_unsharded_grad(param))
        self.assertIs(TorchHSDPStateV2._get_local_sharded_grad(param), local_grad)

        no_grad_param = _FakeHSDPParam(unsharded_grad=None)
        no_grad_param._unsharded_param = None
        self.assertFalse(TorchHSDPStateV2._has_pending_unsharded_grad(no_grad_param))
        self.assertIsNone(TorchHSDPStateV2._get_local_sharded_grad(_FakeHSDPParam(grad=None)))

    def test_comm_fusion_unsupported_reason_validation_paths(self):
        """Comm-fusion validation should report the first unsupported parameter reason."""
        not_sharded = _FakeHSDPParam(enable_fsdp_shard=False)
        self.assertIn("non-sharded", TorchHSDPStateV2._comm_fusion_unsupported_reason(not_sharded))

        compat = _FakeHSDPParam(param_mode=FullyShardParamMode.DTENSOR_COMPAT)
        self.assertIn("param_mode", TorchHSDPStateV2._comm_fusion_unsupported_reason(compat))

        missing_local = _FakeHSDPParam()
        missing_local._sharded_local_tensor = None
        self.assertIn("missing local shard", TorchHSDPStateV2._comm_fusion_unsupported_reason(missing_local))

        bad_plan = _FakeHSDPParam()
        with patch.object(state_mod, "build_rs_plan", side_effect=ValueError("bad shape")):
            self.assertIn("cannot build", TorchHSDPStateV2._comm_fusion_unsupported_reason(bad_plan))

        supported = _FakeHSDPParam()
        with patch.object(state_mod, "build_rs_plan", return_value=object()):
            self.assertIsNone(TorchHSDPStateV2._comm_fusion_unsupported_reason(supported))

    def test_init_param_group_skips_when_disabled_and_constructs_when_enabled(self):
        """Param-group init should skip disabled fusion and build enabled fusion groups."""
        disabled = _new_state([_FakeHSDPParam()], comm_fusion=False)
        TorchHSDPStateV2._init_param_group(disabled)
        self.assertIsNone(disabled.param_group)

        supported = _new_state([_FakeHSDPParam()], comm_fusion=True)
        with patch.object(TorchHSDPStateV2, "_comm_fusion_unsupported_reason", return_value=None):
            with patch.object(state_mod, "HSDPParamGroup", return_value="group") as param_group:
                TorchHSDPStateV2._init_param_group(supported)

        param_group.assert_called_once()
        self.assertEqual(supported.param_group, "group")

    def test_init_param_group_reports_first_unsupported_parameter(self):
        """Param-group init should report the first unsupported fusion reason."""
        param = _FakeHSDPParam()
        state = _new_state([param], comm_fusion=True)

        with patch.object(TorchHSDPStateV2, "_comm_fusion_unsupported_reason", return_value="unsupported layout"):
            with self.assertRaisesRegex(NotImplementedError, "unsupported layout"):
                TorchHSDPStateV2._init_param_group(state)

    def test_init_mp_dtypes_accepts_uniform_and_rejects_mismatch(self):
        """Mixed-precision dtype init should accept uniform dtypes and reject mismatch."""
        first = _FakeHSDPParam()
        second = _FakeHSDPParam()
        state = _new_state([first, second])

        TorchHSDPStateV2._init_mp_dtypes(state)

        first.init_dtype_attrs.assert_called_once_with(None)
        second.init_dtype_attrs.assert_called_once_with(None)
        self.assertEqual(state._orig_dtype, torch.float32)
        self.assertEqual(state._reduce_dtype, torch.float32)

        second.orig_dtype = torch.float16
        with self.assertRaises(AssertionError):
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

    def test_queue_standard_reduce_scatter_and_optional_all_reduce(self):
        """Standard queueing should reduce-scatter and optionally all-reduce."""
        param = _FakeHSDPParam()
        state = _new_state([param])

        TorchHSDPStateV2._queue_reduce_scatter_then_all_reduce(state, param, torch.distributed.ReduceOp.AVG)

        param.reduce_scatter_grad.assert_called_once_with(
            dtype=torch.float32,
            reduce_op=torch.distributed.ReduceOp.AVG,
        )
        param.all_reduce_grad.assert_called_once_with(
            grad=param.reduce_scatter_output.return_value,
            dtype=torch.float32,
            reduce_op=torch.distributed.ReduceOp.AVG,
        )
        self.assertEqual(HSDPState.pre_all_reduce_params, [(param, torch.float32)])
        self.assertFalse(HSDPState.pre_reduce_scatter_params)

        no_all_reduce = _FakeHSDPParam(dp_size=1)
        TorchHSDPStateV2._queue_reduce_scatter_then_all_reduce(
            state, no_all_reduce, torch.distributed.ReduceOp.AVG
        )
        no_all_reduce.all_reduce_grad.assert_not_called()
        self.assertEqual(HSDPState.pre_reduce_scatter_params, [(no_all_reduce, torch.float32)])

    def test_queue_compat_and_direct_all_reduce_paths(self):
        """Compat params should use queued or direct all-reduce paths."""
        param = _FakeHSDPParam(unsharded_grad=torch.ones(2))
        state = _new_state([param])

        TorchHSDPStateV2._queue_compat_all_reduce(state, param, torch.distributed.ReduceOp.SUM)

        # Pure all-reduce path passes grad=None so all_reduce_grad fetches the
        # unsharded grad itself and owns the scaling.
        param.all_reduce_grad.assert_called_once_with(
            dtype=torch.float32,
            reduce_op=torch.distributed.ReduceOp.SUM,
        )
        self.assertEqual(HSDPState.pre_all_reduce_params, [(param, torch.float32)])

        grad = torch.ones(2, dtype=torch.float32)
        direct_param = _FakeHSDPParam(
            param_mode=FullyShardParamMode.DTENSOR_COMPAT,
            is_sharded=False,
            shard_size=1,
            grad=grad,
        )
        state._reduce_dtype = torch.float16
        self.assertTrue(TorchHSDPStateV2._can_direct_all_reduce_compat_grad(state, direct_param))
        handle = SimpleNamespace(wait=MagicMock())
        with patch.object(torch.distributed, "all_reduce", return_value=handle) as all_reduce:
            TorchHSDPStateV2._queue_direct_compat_all_reduce(
                state, direct_param, torch.distributed.ReduceOp.SUM
            )

        all_reduce.assert_called_once()
        queued_handle, reduced_grad, target_grad = TorchHSDPStateV2.pre_direct_all_reduce_grads[0]
        self.assertIs(queued_handle, handle)
        self.assertIs(target_grad, grad)
        self.assertEqual(reduced_grad.dtype, torch.float16)
        torch.testing.assert_close(reduced_grad.float(), grad)

    def test_reduce_params_drains_all_queues_and_copies_direct_grad(self):
        """Reduce params should drain queued collectives and copy direct reduced grads."""
        state = _new_state()
        rs_param = _FakeHSDPParam()
        ar_param = _FakeHSDPParam()
        HSDPState.pre_reduce_scatter_params.append((rs_param, torch.float32))
        HSDPState.pre_all_reduce_params.append((ar_param, torch.float32))
        target_grad = torch.zeros(2)
        reduced_grad = torch.ones(2, dtype=torch.float16)
        handle = SimpleNamespace(wait=MagicMock())
        TorchHSDPStateV2.pre_direct_all_reduce_grads.append((handle, reduced_grad, target_grad))

        TorchHSDPStateV2.reduce_scattered_params(state)
        TorchHSDPStateV2.reduce_params(state)

        rs_param.clear_reduce_scatter_output.assert_called_once()
        rs_param.apply_reduced_grad.assert_called_once()
        ar_param.clear_all_reduce_output.assert_called_once()
        ar_param.apply_reduced_grad.assert_called_once()
        handle.wait.assert_called_once()
        self.assertTrue(torch.equal(target_grad, torch.ones(2)))

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

    def test_post_backward_reduces_sharded_unsharded_and_direct_compat_params(self):
        """Post-backward should route sharded, replicated, and direct compat grads."""
        sharded = _FakeHSDPParam(unsharded_grad=torch.ones(2), shard_size=2)
        replicated = _FakeHSDPParam(unsharded_grad=torch.ones(2), shard_size=1)
        direct = _FakeHSDPParam(
            param_mode=FullyShardParamMode.DTENSOR_COMPAT,
            is_sharded=False,
            shard_size=1,
            grad=torch.ones(2),
        )
        direct._unsharded_param = None
        state = _new_state([sharded, replicated, direct])
        state.shard = MagicMock()

        with patch.object(torch.distributed, "all_reduce", return_value=None):
            TorchHSDPStateV2.post_backward(state)

        sharded.reduce_scatter_grad.assert_called_once()
        replicated.reduce_scatter_grad.assert_called_once()
        self.assertEqual(len(TorchHSDPStateV2.pre_all_reduce_groups), 2)
        self.assertEqual(len(TorchHSDPStateV2.pre_direct_all_reduce_grads), 1)
        state.shard.assert_called_once()

    def test_post_backward_for_comm_fusion_drains_comm_context_and_queues_replicates(self):
        """Comm-fusion post-backward should drain prior groups and queue replicas."""
        replicate = _FakeHSDPParam(unsharded_grad=torch.ones(2), shard_size=1)
        param_group = SimpleNamespace(foreach_reduce=MagicMock())
        state = _new_state([], [replicate], comm_fusion=True)
        state.param_group = param_group
        state.reduce_params = MagicMock()
        previous_all_reduce = SimpleNamespace(wait_all_reduce_and_apply_grad=MagicMock())
        previous_reduce_scatter = SimpleNamespace(wait_reduce_scatter_and_issue_all_reduce=MagicMock())
        comm_ctx = SimpleNamespace(
            all_reduce_param_group=previous_all_reduce,
            pre_param_group=previous_reduce_scatter,
        )

        with patch.object(state_mod, "get_comm_ctx", return_value=comm_ctx):
            TorchHSDPStateV2.post_backward_for_comm_fusion(state)

        state.reduce_params.assert_called_once()
        previous_all_reduce.wait_all_reduce_and_apply_grad.assert_called_once()
        previous_reduce_scatter.wait_reduce_scatter_and_issue_all_reduce.assert_called_once()
        param_group.foreach_reduce.assert_called_once_with(
            reduce_scatter_reduce_op=torch.distributed.ReduceOp.AVG
        )
        replicate.all_reduce_grad.assert_called_once()
        self.assertIsNone(comm_ctx.all_reduce_param_group)
        self.assertIsNone(comm_ctx.pre_param_group)

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

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
"""Unit tests for MindSpore fully_shard mixed precision.

Covered scenarios:
  1. Behavior of the _to_dtype_if_needed helper
  2. dtype initialization and optimization logic in init_dtype_attrs
  3. Inference of _orig_dtype / _reduce_dtype in _init_mp_dtypes
     under different dtype combinations
  4. lazy_init refreshes _orig_dtype after parameter dtype changes
     such as calling net.to(bfloat16) after fully_shard
  5. dtype restoration and gradient assignment logic in
     _apply_reduced_grad
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

# pylint: disable=protected-access,wrong-import-position

# Skip entire module if mindspore is not installed (avoids import failure)
pytest.importorskip("mindspore")

# Force mindspore platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState, ShardedState
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2
from hyper_parallel.platform.mindspore.fully_shard.state import (
    MindSporeHSDPStateV2,
    _to_dtype_if_needed,
)
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2


# ---------------------------------------------------------------------------
# Helper: bypass __init__ and construct a state instance with manually injected attributes
# ---------------------------------------------------------------------------

def _make_state(mp_policy, hsdp_params):
    """Create a lightweight MindSporeHSDPStateV2 instance without hardware initialization."""
    state = object.__new__(MindSporeHSDPStateV2)
    state.mp_policy = mp_policy
    state.offload_policy = None
    state.hsdp_params = hsdp_params
    state.replicate_params = []
    state._ignored_allreduce_works = []
    state._reset_sharded_params = True   # Skip the reset_sharded_param branch
    return state


def _make_init_dtype_attrs(mock_param, orig_dtype_value):
    """Create a bound init_dtype_attrs side effect for a mock hsdp param."""

    def _init_dtype_attrs(policy):
        mock_param.orig_dtype = orig_dtype_value
        inferred_reduce_dtype = policy.reduce_dtype
        if inferred_reduce_dtype == policy.param_dtype:
            inferred_reduce_dtype = None
        mock_param.reduce_dtype = inferred_reduce_dtype

    return _init_dtype_attrs


def _make_mock_hsdp_param(dtype, requires_grad=True):
    """Create a minimal hsdp_param mock for init_dtype_attrs calls."""
    mock_param = MagicMock()
    mock_param.sharded_param.dtype = dtype
    mock_param.sharded_param.requires_grad = requires_grad
    # init_dtype_attrs assigns orig_dtype / param_dtype / reduce_dtype directly
    # MagicMock allows arbitrary attribute assignment without extra setup
    return mock_param


# ---------------------------------------------------------------------------
# 1. _to_dtype_if_needed
# ---------------------------------------------------------------------------

class TestToDtypeIfNeeded(unittest.TestCase):
    """Test the _to_dtype_if_needed helper."""

    def test_no_cast_when_dtype_is_none(self):
        """
        Feature: _to_dtype_if_needed
        Description: Do not cast when dtype=None; return the original tensor directly
        Expectation: tensor.to is not called, and the return value is the input tensor
        """
        tensor = MagicMock()
        result = _to_dtype_if_needed(tensor, None)
        tensor.to.assert_not_called()
        self.assertIs(result, tensor)

    def test_no_cast_when_same_dtype(self):
        """
        Feature: _to_dtype_if_needed
        Description: Do not cast when the target dtype matches tensor.dtype
        Expectation: tensor.to is not called, and the return value is the input tensor
        """
        tensor = MagicMock()
        tensor.dtype = ms.float32
        result = _to_dtype_if_needed(tensor, ms.float32)
        tensor.to.assert_not_called()
        self.assertIs(result, tensor)

    def test_cast_when_different_dtype(self):
        """
        Feature: _to_dtype_if_needed
        Description: Perform a cast when the target dtype differs from tensor.dtype
        Expectation: tensor.to(target_dtype) is called once and returns the cast result
        """
        tensor = MagicMock()
        tensor.dtype = ms.float32
        casted = MagicMock()
        tensor.to.return_value = casted
        result = _to_dtype_if_needed(tensor, ms.float16)
        tensor.to.assert_called_once_with(ms.float16)
        self.assertIs(result, casted)


# ---------------------------------------------------------------------------
# 2. init_dtype_attrs (parameter-level dtype initialization logic)
# ---------------------------------------------------------------------------

class TestInitDtypeAttrs(unittest.TestCase):
    """Test dtype inference in MindSporeHSDPParamV2.init_dtype_attrs."""

    def _call_init_dtype_attrs(self, orig_dtype, param_dtype, reduce_dtype):
        """Run init_dtype_attrs on a mock param object and return it."""
        mock_self = MagicMock()
        mock_self.sharded_param = MagicMock()
        mock_self.sharded_param.dtype = orig_dtype
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        MindSporeHSDPParamV2.init_dtype_attrs(mock_self, policy)
        return mock_self

    def test_param_dtype_same_as_orig_set_to_none(self):
        """
        Feature: init_dtype_attrs
        Description: When param_dtype matches the parameter's original dtype, it should be set to None (no cast needed)
        Expectation: self.param_dtype is None
        """
        obj = self._call_init_dtype_attrs(ms.float32, ms.float32, ms.float32)
        self.assertIsNone(obj.param_dtype)

    def test_reduce_dtype_same_as_param_dtype_set_to_none(self):
        """
        Feature: init_dtype_attrs
        Description: When reduce_dtype matches param_dtype, it should be set to None to avoid redundant casts
        Expectation: self.reduce_dtype is None
        """
        obj = self._call_init_dtype_attrs(ms.float32, ms.float16, ms.float16)
        self.assertIsNone(obj.reduce_dtype)

    def test_different_dtypes_kept(self):
        """
        Feature: init_dtype_attrs
        Description: Keep both values when param_dtype != orig_dtype and reduce_dtype != param_dtype
        Expectation: self.param_dtype == float16, self.reduce_dtype == float32
        """
        obj = self._call_init_dtype_attrs(ms.float32, ms.float16, ms.float32)
        self.assertEqual(obj.param_dtype, ms.float16)
        self.assertEqual(obj.reduce_dtype, ms.float32)

    def test_orig_dtype_always_set(self):
        """
        Feature: init_dtype_attrs
        Description: orig_dtype always records the parameter's original dtype
        Expectation: self.orig_dtype == the parameter's initial dtype
        """
        obj = self._call_init_dtype_attrs(ms.bfloat16, ms.float16, ms.float32)
        self.assertEqual(obj.orig_dtype, ms.bfloat16)


# ---------------------------------------------------------------------------
# 3. _init_mp_dtypes (state-level dtype inference)
# ---------------------------------------------------------------------------

class TestInitMpDtypes(unittest.TestCase):
    """Test _orig_dtype / _reduce_dtype inference in MindSporeHSDPStateV2._init_mp_dtypes."""

    def _run(self, mp_policy, params_config):
        """
        params_config: list of (orig_dtype, reduce_dtype, requires_grad)
        """
        mock_params = []
        for param_config in params_config:
            orig_dtype, _, requires_grad = param_config
            mock_param = _make_mock_hsdp_param(orig_dtype, requires_grad)
            # init_dtype_attrs runs for real here, so dtype values must be bound
            mock_param.init_dtype_attrs.side_effect = _make_init_dtype_attrs(
                mock_param, orig_dtype
            )
            mock_params.append(mock_param)

        state = _make_state(mp_policy, mock_params)
        state._init_mp_dtypes()
        return state

    def test_basic_fp16_param_dtype(self):
        """
        Feature: _init_mp_dtypes
        Description: param_dtype=float16 and the parameter's original dtype=float32
        Expectation: _orig_dtype=float32 and _reduce_dtype=None (reduce matches param)
        """
        policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float16)
        state = self._run(policy, [(ms.float32, None, True)])
        self.assertEqual(state._orig_dtype, ms.float32)
        self.assertIsNone(state._reduce_dtype)

    def test_separate_reduce_dtype(self):
        """
        Feature: _init_mp_dtypes
        Description: param_dtype=float16 and reduce_dtype=float32 (different from param_dtype)
        Expectation: _orig_dtype=float32, _reduce_dtype=float32
        """
        policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float32)
        state = self._run(policy, [(ms.float32, ms.float32, True)])
        self.assertEqual(state._orig_dtype, ms.float32)
        self.assertEqual(state._reduce_dtype, ms.float32)

    def test_no_trainable_params_returns_none(self):
        """
        Feature: _init_mp_dtypes
        Description: When there are no parameters with requires_grad=True, both _orig_dtype and _reduce_dtype are None
        Expectation: _orig_dtype is None and _reduce_dtype is None
        """
        policy = MixedPrecisionPolicy(param_dtype=ms.float16)
        state = self._run(policy, [(ms.float32, None, False)])
        self.assertIsNone(state._orig_dtype)
        self.assertIsNone(state._reduce_dtype)

    def test_non_uniform_orig_dtype_raises(self):
        """
        Feature: _init_mp_dtypes
        Description: Raise AssertionError when multiple parameters have inconsistent orig_dtype values
        Expectation: Raise AssertionError with a message containing 'uniform original parameter dtype'
        """
        policy = MixedPrecisionPolicy(param_dtype=ms.float16)
        with self.assertRaises(AssertionError) as ctx:
            self._run(policy, [
                (ms.float32, None, True),
                (ms.float16, None, True),
            ])
        self.assertIn("uniform original parameter dtype", str(ctx.exception))


# ---------------------------------------------------------------------------
# 4. lazy_init refreshes _orig_dtype after parameter dtype changes
# ---------------------------------------------------------------------------

class TestLazyInitDtypeRefresh(unittest.TestCase):
    """
    Verify that lazy_init correctly refreshes _orig_dtype when net.to(bfloat16)
    is called after fully_shard wrapping and before the next forward pass.
    """

    def _make_state_with_dtype(self, dtype):
        """Create a state whose hsdp_param.sharded_param.dtype can change dynamically."""
        mock_param = MagicMock()
        mock_param.sharded_param.requires_grad = True
        mock_param.sharded_param.device = "Ascend:0"

        # init_dtype_attrs reads sharded_param.dtype directly and assigns it to orig_dtype
        def init_dtype_attrs(policy):
            mock_param.orig_dtype = mock_param.sharded_param.dtype
            rd = policy.reduce_dtype
            pd = policy.param_dtype
            if rd == pd:
                rd = None
            mock_param.reduce_dtype = rd
        mock_param.init_dtype_attrs.side_effect = init_dtype_attrs

        policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float16)
        state = _make_state(policy, [mock_param])
        state._reset_sharded_params = False  # Trigger the reset_sharded_param branch
        state.offload_policy = None

        # Initial dtype
        mock_param.sharded_param.dtype = dtype
        return state, mock_param

    def test_orig_dtype_refreshed_after_dtype_change(self):
        """
        Feature: lazy_init dtype refresh
        Description: Simulate calling net.to(bfloat16) after fully_shard, then trigger lazy_init
        Expectation: _orig_dtype is updated from float32 to bfloat16
        """
        state, mock_param = self._make_state_with_dtype(ms.float32)

        with patch.object(state, '_validate_no_meta_params'), \
             patch.object(state, '_validate_cpu_offload_params'), \
             patch.object(mock_param, 'is_sharded', new=False):
            # First lazy_init call (parameter is float32)
            state.lazy_init()
            self.assertEqual(state._orig_dtype, ms.float32)

            # Simulate net.to(bfloat16)
            mock_param.sharded_param.dtype = ms.bfloat16

            # Second lazy_init call (parameter has changed to bfloat16)
            state._reset_sharded_params = True  # Do not trigger reset again
            state.lazy_init()
            self.assertEqual(state._orig_dtype, ms.bfloat16)


# ---------------------------------------------------------------------------
# 5. _apply_reduced_grad dtype restoration and gradient assignment
# ---------------------------------------------------------------------------

class TestApplyReducedGrad(unittest.TestCase):
    """Test dtype conversion and gradient assignment behavior in _apply_reduced_grad."""

    def _make_state_for_apply(self, orig_dtype):
        """Create a state for _apply_reduced_grad tests."""
        policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float32)
        state = _make_state(policy, [])
        state._orig_dtype = orig_dtype
        return state

    def test_grad_cast_to_orig_dtype(self):
        """
        Feature: _apply_reduced_grad
        Description: Cast to _orig_dtype when reduced_grad.dtype differs from _orig_dtype
        Expectation: _to_dtype_if_needed is called with _orig_dtype
        """
        state = self._make_state_for_apply(ms.float32)

        # Mock hsdp_param
        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (8, 8)
        hsdp_param.offload_to_cpu = False
        hsdp_param.sharded_param.grad = None
        hsdp_param.unsharded_accumulated_grad_data = None
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        hsdp_param.to_sharded_dtensor.return_value = sharded_grad_dtensor

        # Mock reduced_grad
        reduced_grad = MagicMock()
        viewed_grad = MagicMock()
        viewed_grad.dtype = ms.float16  # Different from _orig_dtype
        casted_grad = MagicMock()
        casted_grad.dtype = ms.float32
        viewed_grad.to.return_value = casted_grad
        reduced_grad.view.return_value = viewed_grad

        with patch(
            'hyper_parallel.platform.mindspore.fully_shard.state._to_dtype_if_needed',
            side_effect=lambda t, d: t.to(d) if (d is not None and t.dtype != d) else t
        ):
            state._apply_reduced_grad(hsdp_param, reduced_grad)

        # Verify that view is called
        reduced_grad.view.assert_called_once_with((8, 8))
        # Verify the cast to orig_dtype
        viewed_grad.to.assert_called_once_with(ms.float32)
        # Verify that sharded_param.grad receives the result first,
        # then becomes the optimizer-visible DTensor grad
        hsdp_param.to_sharded_dtensor.assert_called_once_with(casted_grad)
        self.assertIs(hsdp_param.sharded_param.grad, sharded_grad_dtensor)
        self.assertIsNone(hsdp_param.unsharded_param.grad)

    def test_grad_accumulated_when_existing(self):
        """
        Feature: _apply_reduced_grad
        Description: When sharded_param.grad already exists,
        the new gradient should be accumulated instead of overwritten
        Expectation: hsdp_param.sharded_param.grad._local_tensor += reduced_grad is executed
        """
        state = self._make_state_for_apply(ms.float32)

        existing_grad = MagicMock(spec=DTensor)
        local_tensor = MagicMock()
        existing_grad._local_tensor = local_tensor
        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (4,)
        hsdp_param.offload_to_cpu = False
        hsdp_param.sharded_param.grad = existing_grad
        hsdp_param.unsharded_accumulated_grad_data = None

        reduced_grad = MagicMock()
        viewed_grad = MagicMock()
        viewed_grad.dtype = ms.float32  # Matches _orig_dtype, so no cast is needed
        reduced_grad.view.return_value = viewed_grad

        with patch(
            'hyper_parallel.platform.mindspore.fully_shard.state._to_dtype_if_needed',
            return_value=viewed_grad
        ):
            state._apply_reduced_grad(hsdp_param, reduced_grad)

        # Verify that the gradient is accumulated into the underlying local tensor
        local_tensor.__iadd__.assert_called_once_with(viewed_grad)

    def test_unsharded_grad_cleared_after_apply(self):
        """
        Feature: _apply_reduced_grad
        Description: After gradient assignment,
        the temporary unsharded grad should be cleared
        Expectation: the reduced DTensor grad is attached to sharded_param.grad
        and the transient full-grad reference is cleared.
        """
        state = self._make_state_for_apply(ms.float32)

        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (4,)
        hsdp_param.offload_to_cpu = False
        hsdp_param.sharded_param.grad = None
        hsdp_param.unsharded_accumulated_grad_data = None
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        hsdp_param.to_sharded_dtensor.return_value = sharded_grad_dtensor

        reduced_grad = MagicMock()
        casted_grad = MagicMock()
        reduced_grad.view.return_value = casted_grad

        with patch(
            'hyper_parallel.platform.mindspore.fully_shard.state._to_dtype_if_needed',
            return_value=casted_grad
        ):
            state._apply_reduced_grad(hsdp_param, reduced_grad)

        hsdp_param.to_sharded_dtensor.assert_called_once_with(casted_grad)
        self.assertIs(hsdp_param.sharded_param.grad, sharded_grad_dtensor)
        self.assertIsNone(hsdp_param.unsharded_param.grad)


class TestReplicateParamGradHandling(unittest.TestCase):
    """Test gradient handling logic related to replicate_params."""

    def _make_state_for_replicate(self, orig_dtype):
        """Create a state for replicate_params branch tests."""
        policy = MixedPrecisionPolicy(param_dtype=ms.float16, reduce_dtype=ms.float32)
        state = _make_state(policy, [])
        state._orig_dtype = orig_dtype
        state._need_div = True
        return state

    def test_zero_grad_clears_replicate_params(self):
        """
        Feature: zero_grad
        Description: zero_grad should clear both hsdp_params and replicate_params
        Expectation: zero_grad is called once for both parameter groups
        """
        state = self._make_state_for_replicate(ms.float32)
        hsdp_param = MagicMock()
        replicate_param = MagicMock()
        state.hsdp_params = [hsdp_param]
        state.replicate_params = [replicate_param]

        state.zero_grad()

        hsdp_param.zero_grad.assert_called_once_with()
        replicate_param.zero_grad.assert_called_once_with()

    def test_finish_ignored_allreduce_materializes_sharded_grad(self):
        """
        Feature: _finish_ignored_allreduce
        Description: After all-reduce finishes for replicate_params,
        the reduced local grad should be materialized on sharded_param.grad
        Expectation: the DTensor grad is assigned and the transient full grad is cleared
        """
        state = self._make_state_for_replicate(ms.float32)

        reduced_grad = MagicMock()
        reduced_grad.dtype = ms.float32
        reduced_grad.to.return_value = reduced_grad
        flat_mesh = MagicMock()
        flat_mesh.rank_list = [0, 1]

        local_grad = MagicMock()
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        sharded_grad_dtensor.to_local.return_value = local_grad

        param = MagicMock()
        param.all_reduce_handle = MagicMock()
        param.offload_to_cpu = False
        param.sharded_param.grad = None
        param.to_sharded_dtensor.return_value = sharded_grad_dtensor
        param.unsharded_accumulated_grad_data = None

        state._ignored_allreduce_works = [(param, reduced_grad, flat_mesh)]

        with patch(
            'hyper_parallel.platform.mindspore.fully_shard.state._to_dtype_if_needed',
            return_value=reduced_grad
        ):
            state._finish_ignored_allreduce()

        param.all_reduce_handle.wait.assert_called_once_with()
        reduced_grad.view.assert_not_called()
        param.to_sharded_dtensor.assert_called_once_with(reduced_grad)
        self.assertIs(param.sharded_param.grad, sharded_grad_dtensor)
        self.assertIsNone(param.unsharded_param.grad)
        self.assertEqual(state._ignored_allreduce_works, [])


class TestParameterRebinding(unittest.TestCase):
    """Test MindSpore fully_shard parameter rebinding behavior."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.Parameter")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.as_strided")
    def test_init_unsharded_param_uses_parameter_empty_then_data_assign(
        self, mock_as_strided, mock_parameter
    ):
        """
        Feature: init_unsharded_param
        Description: Build _unsharded_param from Parameter([]) and then assign .data
        Expectation: Parameter is constructed with [] and its data points to the as_strided tensor view
        """
        param = object.__new__(MindSporeHSDPParamV2)
        param.all_gather_outputs = [MagicMock(name="all_gather_output")]
        param._orig_size = (2, 3)
        param._contiguous_orig_stride = (3, 1)
        param.sharded_param = MagicMock()
        param.sharded_param.name = "weight"
        param.sharded_param.requires_grad = True

        unsharded_tensor = MagicMock(name="unsharded_tensor")
        mock_as_strided.return_value = unsharded_tensor
        unsharded_param = MagicMock(name="unsharded_param")
        mock_parameter.return_value = unsharded_param

        param.init_unsharded_param()

        mock_parameter.assert_called_once_with(
            [],
            name="weight",
            requires_grad=True,
        )
        mock_as_strided.assert_called_once_with(
            param.all_gather_outputs[0],
            (2, 3),
            (3, 1),
            storage_offset=0,
        )
        self.assertIs(param._unsharded_param, unsharded_param)
        self.assertIs(unsharded_param.data, unsharded_tensor)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed")
    def test_to_unsharded_rebinds_module_param(self, mock_set_requires_grad):
        """
        Feature: to_unsharded
        Description: Switch module references to the unsharded parameter object
        Expectation: requires_grad is synced, _setattr_on_modules gets _unsharded_param, and state updates
        """
        param = object.__new__(MindSporeHSDPParamV2)
        param.sharded_param = MagicMock(name="sharded_param")
        param._unsharded_param = MagicMock(name="unsharded_param")
        param._setattr_on_modules = MagicMock()

        param.to_unsharded()

        mock_set_requires_grad.assert_called_once_with(param.sharded_param, param._unsharded_param)
        param._setattr_on_modules.assert_called_once_with(param._unsharded_param)
        self.assertEqual(param.sharded_state, ShardedState.UNSHARDED)

    def test_to_sharded_rebinds_module_param(self):
        """
        Feature: to_sharded
        Description: Switch module references back to the sharded parameter object
        Expectation: _setattr_on_modules gets sharded_param, unsharded storage is freed, and state updates
        """
        param = object.__new__(MindSporeHSDPParamV2)
        param.sharded_param = MagicMock(name="sharded_param")
        param._setattr_on_modules = MagicMock()
        param.free_unsharded_param = MagicMock()

        param.to_sharded()

        param._setattr_on_modules.assert_called_once_with(param.sharded_param)
        param.free_unsharded_param.assert_called_once_with()
        self.assertEqual(param.sharded_state, ShardedState.SHARDED)


class TestPrefetchStateMachine(unittest.TestCase):
    """Test the shared per-parameter unshard/prefetch state machine."""

    def _make_param(self):
        """Create a lightweight fully_shard param object with mocked hooks."""
        param = object.__new__(MindSporeHSDPParamV2)
        param.sharded_state = ShardedState.SHARDED
        param.prefetch_handle = None
        param._assert_in_states = MagicMock()
        param._get_unsharded_param_data = MagicMock()
        param.init_unsharded_param = MagicMock()
        param.to_unsharded = MagicMock()
        param.to_sharded = MagicMock()
        return param

    def test_async_unshard_uses_prefetch_handle_as_pending_state(self):
        """
        Feature: per-parameter prefetch
        Description: async unshard should record the backend handle as the pending state
        Expectation: the handle is stored and used by wait_for_unshard
        """
        param = self._make_param()
        mock_handle = MagicMock()
        param._get_unsharded_param_data.return_value = (MagicMock(name="output"), mock_handle)

        param.unshard(async_op=True)

        param._get_unsharded_param_data.assert_called_once_with(async_op=True)
        self.assertIs(param.prefetch_handle, mock_handle)

    def test_sync_unshard_is_noop_when_pending_prefetch_exists(self):
        """
        Feature: per-parameter prefetch
        Description: sync unshard should not reissue communication or eagerly consume a pending prefetch
        Expectation: pending handle remains until wait_for_unshard() explicitly finishes the transition
        """
        param = self._make_param()
        mock_handle = MagicMock()
        param._get_unsharded_param_data.return_value = (MagicMock(name="output"), mock_handle)

        param.unshard(async_op=True)
        param.unshard(async_op=False)

        param._get_unsharded_param_data.assert_called_once_with(async_op=True)
        mock_handle.wait.assert_not_called()
        param.init_unsharded_param.assert_not_called()
        param.to_unsharded.assert_not_called()
        self.assertIs(param.prefetch_handle, mock_handle)

    def test_sync_unshard_records_handle_and_wait_for_unshard_finishes_transition(self):
        """
        Feature: per-parameter prefetch
        Description: sync unshard should match torch semantics by issuing communication and deferring the transition
        Expectation: the handle is recorded first, then wait_for_unshard completes the transition
        """
        param = self._make_param()
        mock_handle = MagicMock()
        param._get_unsharded_param_data.return_value = (MagicMock(name="output"), mock_handle)

        param.unshard(async_op=False)
        self.assertIs(param.prefetch_handle, mock_handle)

        param.wait_for_unshard()

        param._get_unsharded_param_data.assert_called_once_with(async_op=False)
        mock_handle.wait.assert_called_once_with()
        param.init_unsharded_param.assert_called_once_with()
        param.to_unsharded.assert_called_once_with()
        self.assertIsNone(param.prefetch_handle)


class TestSchedulerBackwardCompatFlow(unittest.TestCase):
    """Test MindSpore scheduler behavior added by the backward-compat refactor."""

    def tearDown(self):
        HSDPSchedulerV2.root_bp_state = False

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler._pynative_executor")
    def test_backward_pre_hook_queues_final_callback_before_pre_backward(self, mock_executor):
        """
        Feature: backward pre hook
        Description: Queue the final backward callback and then trigger the HSDP pre-backward hook
        Expectation: queue_backward_final_callback is called and the incoming grad is returned unchanged
        """
        scheduler = object.__new__(MindSporeHSDPSchedulerV2)
        scheduler.scheduler_state = FSDPSchedulerState.PRE_FORWARD
        scheduler.cell = MagicMock()
        scheduler._hsdp_backward_pre_hook = MagicMock()
        scheduler._root_backward_hook = MagicMock()
        grad = MagicMock()

        result = scheduler._backward_pre_hook(grad)

        mock_executor.queue_backward_final_callback.assert_called_once_with(scheduler._root_backward_hook)
        scheduler._hsdp_backward_pre_hook.assert_called_once_with(scheduler.cell, None)
        self.assertTrue(HSDPSchedulerV2.root_bp_state)
        self.assertIs(result, grad)

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler._pynative_executor")
    def test_backward_pre_hook_is_noop_when_already_in_pre_backward(self, mock_executor):
        """
        Feature: backward pre hook
        Description: Avoid re-entering pre-backward when already in PRE_BACKWARD state
        Expectation: only the final callback is queued and the HSDP pre-backward hook is skipped
        """
        scheduler = object.__new__(MindSporeHSDPSchedulerV2)
        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        scheduler.cell = MagicMock()
        scheduler._hsdp_backward_pre_hook = MagicMock()
        scheduler._root_backward_hook = MagicMock()
        grad = MagicMock()

        result = scheduler._backward_pre_hook(grad)

        mock_executor.queue_backward_final_callback.assert_called_once_with(scheduler._root_backward_hook)
        scheduler._hsdp_backward_pre_hook.assert_not_called()
        self.assertIs(result, grad)

    def test_forward_pre_hook_disables_prefetch_during_recompute(self):
        """
        Feature: recompute forward guard
        Description: During activation recompute, forward pre hook should suppress prefetch issuance
        Expectation: forward_prefetch_cells is cleared before entering the shared pre-forward path
        """
        scheduler = object.__new__(MindSporeHSDPSchedulerV2)
        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler.forward_prefetch_cells = [MagicMock(name="next_cell")]
        scheduler._backup_forward_fetch = None
        scheduler._hsdp_forward_pre_hook = MagicMock(return_value=(("arg",), {"k": "v"}))
        scheduler._register_post_backward_hook = MagicMock(return_value=("wrapped_args", "wrapped_kwargs"))

        HSDPSchedulerV2.root_bp_state = True

        result = scheduler._forward_pre_hook(MagicMock(), ("arg",), {"k": "v"})

        self.assertEqual(scheduler.forward_prefetch_cells, [])
        self.assertEqual(len(scheduler._backup_forward_fetch), 1)
        scheduler._hsdp_forward_pre_hook.assert_called_once()
        scheduler._register_post_backward_hook.assert_called_once_with(("arg",), {"k": "v"})
        self.assertEqual(result, ("wrapped_args", "wrapped_kwargs"))

    def test_forward_hook_restores_prefetch_after_recompute(self):
        """
        Feature: recompute forward guard
        Description: During activation recompute, forward hook should restore prefetched
                     targets and skip post-forward logic
        Expectation: forward prefetch list is restored after registering backward hooks
        """
        scheduler = object.__new__(MindSporeHSDPSchedulerV2)
        scheduler.scheduler_state = FSDPSchedulerState.PRE_FORWARD
        scheduler.forward_prefetch_cells = []
        restored_prefetch = [MagicMock(name="next_cell")]
        scheduler._backup_forward_fetch = restored_prefetch.copy()
        scheduler._register_backward_pre_hook = MagicMock()
        scheduler._hsdp_forward_hook = MagicMock()
        outputs = MagicMock(name="outputs")

        HSDPSchedulerV2.root_bp_state = True

        result = scheduler._forward_hook(MagicMock(), MagicMock(), outputs)

        scheduler._register_backward_pre_hook.assert_called_once_with(outputs)
        scheduler._hsdp_forward_hook.assert_not_called()
        self.assertIsNone(result)
        self.assertEqual(scheduler.forward_prefetch_cells, restored_prefetch)
        self.assertIsNone(scheduler._backup_forward_fetch)

    def test_root_backward_hook_resets_root_backward_state(self):
        """
        Feature: recompute forward guard
        Description: Final root backward hook should clear the shared recompute state
        Expectation: root_bp_state is reset after the outermost post-backward finishes
        """
        scheduler = object.__new__(MindSporeHSDPSchedulerV2)
        scheduler.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        scheduler._backward_hook = MagicMock()

        HSDPSchedulerV2.root_bp_state = True

        scheduler._root_backward_hook()

        scheduler._backward_hook.assert_called_once_with()
        self.assertFalse(HSDPSchedulerV2.root_bp_state)

    def test_root_backward_hook_keeps_root_state_for_non_root_callback(self):
        """
        Feature: recompute forward guard
        Description: Non-root final callbacks should not clear the shared recompute state
        Expectation: root_bp_state remains set until the outermost callback runs
        """
        scheduler = object.__new__(MindSporeHSDPSchedulerV2)
        scheduler.scheduler_state = FSDPSchedulerState.BACKWARD
        scheduler._backward_hook = MagicMock()

        HSDPSchedulerV2.root_bp_state = True

        scheduler._root_backward_hook()

        scheduler._backward_hook.assert_called_once_with()
        self.assertTrue(HSDPSchedulerV2.root_bp_state)


if __name__ == "__main__":
    unittest.main()

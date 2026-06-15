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
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState, ShardedState
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2
from hyper_parallel.platform.mindspore.fully_shard.state import (
    MindSporeHSDPStateV2,
    _to_dtype_if_needed,
)
from hyper_parallel.platform.mindspore.fully_shard.param import (
    MindSporeHSDPParamV2,
    _pack_for_reduce_scatter,
)


def _new_hsdp_param_v2() -> MindSporeHSDPParamV2:
    """Bare :class:`MindSporeHSDPParamV2` with ``all_gather_outputs`` initialized."""
    obj = object.__new__(MindSporeHSDPParamV2)
    obj.all_gather_outputs = []
    obj.enable_fsdp_shard = True
    return obj


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
    MindSporeHSDPStateV2._ignored_allreduce_works = []
    state._reset_sharded_params = True   # Skip the reset_sharded_param branch
    state.is_shard = True                # Match HSDPState.__init__ default
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
# 5. apply_reduced_grad dtype restoration and gradient assignment
# ---------------------------------------------------------------------------

class TestApplyReducedGrad(unittest.TestCase):
    """Test dtype conversion and gradient assignment behavior in apply_reduced_grad."""

    def test_grad_cast_to_orig_dtype(self):
        """
        Feature: apply_reduced_grad
        Description: Cast to _orig_dtype when reduced_grad.dtype differs from _orig_dtype
        Expectation: _to_dtype_if_needed is called with _orig_dtype
        """
        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (8, 8)
        hsdp_param.offload_to_cpu = False
        hsdp_param.sharded_param.grad = None
        hsdp_param.unsharded_accumulated_grad_data = None
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        hsdp_param.to_sharded_dtensor.return_value = sharded_grad_dtensor

        reduced_grad = MagicMock()
        viewed_grad = MagicMock()
        viewed_grad.dtype = ms.float16
        casted_grad = MagicMock()
        casted_grad.dtype = ms.float32
        viewed_grad.to.return_value = casted_grad
        reduced_grad.view.return_value = viewed_grad

        MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float32)

        reduced_grad.view.assert_called_once_with((8, 8))
        viewed_grad.to.assert_called_once_with(ms.float32)
        hsdp_param.to_sharded_dtensor.assert_called_once_with(casted_grad)
        self.assertIs(hsdp_param.sharded_param.grad, sharded_grad_dtensor)
        self.assertIsNone(hsdp_param.unsharded_param.grad)

    def test_grad_accumulated_when_existing(self):
        """
        Feature: apply_reduced_grad
        Description: When sharded_param.grad already exists,
        the new gradient should be accumulated instead of overwritten
        Expectation: hsdp_param.sharded_param.grad._local_tensor += reduced_grad is executed
        """
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
        viewed_grad.dtype = ms.float32
        reduced_grad.view.return_value = viewed_grad

        MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float32)

        local_tensor.__iadd__.assert_called_once_with(viewed_grad)

    def test_unsharded_grad_cleared_after_apply(self):
        """
        Feature: apply_reduced_grad
        Description: After gradient assignment,
        the temporary unsharded grad should be cleared
        Expectation: the reduced DTensor grad is attached to sharded_param.grad
        and the transient full-grad reference is cleared.
        """
        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (4,)
        hsdp_param.offload_to_cpu = False
        hsdp_param.sharded_param.grad = None
        hsdp_param.unsharded_accumulated_grad_data = None
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        hsdp_param.to_sharded_dtensor.return_value = sharded_grad_dtensor

        reduced_grad = MagicMock()
        casted_grad = MagicMock()
        casted_grad.dtype = ms.float32
        reduced_grad.view.return_value = casted_grad

        MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float32)

        hsdp_param.to_sharded_dtensor.assert_called_once_with(casted_grad)
        self.assertIs(hsdp_param.sharded_param.grad, sharded_grad_dtensor)
        self.assertIsNone(hsdp_param.unsharded_param.grad)

    def test_unsharded_accumulated_grad_cleared_after_apply(self):
        """
        Feature: apply_reduced_grad
        Description: After gradient assignment, the no-sync accumulated grad cache should be cleared
        Expectation: unsharded_accumulated_grad is reset to None after materializing sharded_param.grad
        """
        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (4,)
        hsdp_param.offload_to_cpu = False
        hsdp_param.sharded_param.grad = None
        hsdp_param.unsharded_accumulated_grad = MagicMock(name="accumulated_grad")
        hsdp_param.unsharded_param.grad = MagicMock(name="live_unsharded_grad")
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        hsdp_param.to_sharded_dtensor.return_value = sharded_grad_dtensor

        reduced_grad = MagicMock()
        casted_grad = MagicMock()
        casted_grad.dtype = ms.float32
        reduced_grad.view.return_value = casted_grad

        MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float32)

        self.assertIs(hsdp_param.sharded_param.grad, sharded_grad_dtensor)
        self.assertIsNone(hsdp_param.unsharded_accumulated_grad)
        self.assertIsNotNone(hsdp_param.unsharded_param.grad)

    def test_cpu_offload_requests_synchronize(self):
        """
        Feature: apply_reduced_grad
        Description: Return True when CPU offload uses a non-blocking device-to-host copy
        Expectation: callers can synchronize the current stream after gradient materialization
        """
        hsdp_param = MagicMock()
        hsdp_param.sharded_size = (4,)
        hsdp_param.offload_to_cpu = True
        hsdp_param.pin_memory = True
        hsdp_param.sharded_param.grad = None
        hsdp_param.unsharded_accumulated_grad_data = None
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        hsdp_param.to_sharded_dtensor.return_value = sharded_grad_dtensor

        reduced_grad = MagicMock()
        viewed_grad = MagicMock()
        viewed_grad.dtype = ms.float32
        cpu_grad = MagicMock()
        viewed_grad.to.return_value = cpu_grad
        reduced_grad.view.return_value = viewed_grad

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(
            hsdp_param, reduced_grad, ms.float32
        )

        viewed_grad.to.assert_called_once_with("cpu", non_blocking=True)
        self.assertTrue(need_synchronize)


class TestNoSyncAccumulatedGrad(unittest.TestCase):
    """Test no-sync gradient accumulation bookkeeping matches Torch semantics."""

    def test_to_accumulated_grad_keeps_grad_when_reduce_dtype_is_none(self):
        """
        Feature: to_accumulated_grad_if_needed
        Description: no-sync should stash the unsharded grad even when reduce_dtype is None
        Expectation: unsharded_accumulated_grad stores the moved grad and clears unsharded_param.grad
        """
        hsdp_param = _new_hsdp_param_v2()
        grad = MagicMock(name="micro_grad")
        hsdp_param._unsharded_param = MagicMock()
        hsdp_param._unsharded_param.grad = grad
        hsdp_param.reduce_dtype = None
        hsdp_param.unsharded_accumulated_grad = None

        MindSporeHSDPParamV2.to_accumulated_grad_if_needed(hsdp_param)

        self.assertIsNone(hsdp_param._unsharded_param.grad)
        self.assertIs(hsdp_param.unsharded_accumulated_grad, grad)

    def test_to_accumulated_grad_accumulates_instead_of_overwriting(self):
        """
        Feature: to_accumulated_grad_if_needed
        Description: repeated no-sync micro steps should accumulate onto the existing cached grad
        Expectation: existing accumulated grad receives an in-place add from the new micro grad
        """
        hsdp_param = _new_hsdp_param_v2()
        incoming_grad = MagicMock(name="incoming_grad")
        accumulated_grad = MagicMock(name="accumulated_grad")
        hsdp_param._unsharded_param = MagicMock()
        hsdp_param._unsharded_param.grad = incoming_grad
        hsdp_param.reduce_dtype = None
        hsdp_param.unsharded_accumulated_grad = accumulated_grad

        MindSporeHSDPParamV2.to_accumulated_grad_if_needed(hsdp_param)

        accumulated_grad.__iadd__.assert_called_once_with(incoming_grad)
        self.assertIsNone(hsdp_param._unsharded_param.grad)


@unittest.skip("TestReplicateParamGradHandling temporarily skipped.")
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
        local_grad = MagicMock()
        sharded_grad_dtensor = MagicMock(spec=DTensor)
        sharded_grad_dtensor.to_local.return_value = local_grad

        param = MagicMock()
        param.all_reduce_handle = MagicMock()
        param.offload_to_cpu = False
        param.sharded_size = (4,)
        param.sharded_param.grad = None
        param.to_sharded_dtensor.return_value = sharded_grad_dtensor
        param.unsharded_accumulated_grad_data = None

        MindSporeHSDPStateV2._ignored_allreduce_works = [(param, reduced_grad, 2, ms.float32, True)]

        state._finish_ignored_allreduce()

        param.all_reduce_handle.wait.assert_called_once_with()
        reduced_grad.div_.assert_called_once_with(2)
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, ms.float32)
        self.assertEqual(MindSporeHSDPStateV2._ignored_allreduce_works, [])


class TestParameterRebinding(unittest.TestCase):
    """Test MindSpore fully_shard parameter rebinding behavior."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.Parameter")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_delays_requires_grad_until_after_data_assign(
        self, mock_get_unsharded_param, mock_parameter
    ):
        """
        Feature: init_unsharded_param
        Description: Build _unsharded_param from Parameter([]), bind shared storage, then restore gradients
        Expectation: Parameter is constructed frozen first so shape metadata is recorded after the real data lands
        """
        param = _new_hsdp_param_v2()
        param._orig_param_is_dtensor = False
        param._unsharded_param = None
        param.sharded_param = MagicMock()
        param.sharded_param.name = "weight"
        param.sharded_param.requires_grad = True

        unsharded_tensor = MagicMock(name="unsharded_tensor")
        mock_get_unsharded_param.return_value = unsharded_tensor
        unsharded_param = MagicMock(name="unsharded_param")
        unsharded_param.requires_grad = False
        mock_parameter.return_value = unsharded_param

        param.init_unsharded_param()

        mock_parameter.assert_called_once_with(
            [],
            name="weight",
            requires_grad=False,
        )
        mock_get_unsharded_param.assert_called_once_with()
        self.assertIs(param._unsharded_param, unsharded_param)
        self.assertIs(unsharded_param.data, unsharded_tensor)
        self.assertIs(unsharded_param.requires_grad, True)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed")
    def test_to_unsharded_rebinds_module_param(self, mock_set_requires_grad):
        """
        Feature: to_unsharded
        Description: Switch module references to the unsharded parameter object
        Expectation: requires_grad is synced, _setattr_on_modules gets _unsharded_param, and state updates
        """
        param = _new_hsdp_param_v2()
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
        param = _new_hsdp_param_v2()
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
        param = _new_hsdp_param_v2()
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

class TestAsyncReduceStateMachine(unittest.TestCase):
    """Test per-parameter async reduce/all-reduce pending state."""

    def _make_param(self):
        """Create a lightweight param mock for async reduce API tests."""
        param = MagicMock()
        param.unsharded_accumulated_grad = None
        param._reduce_scatter_output = None
        param.reduce_scatter_handle = None
        param._all_reduce_output = None
        param.all_reduce_handle = None
        param._assert_in_states = MagicMock()
        param.unsharded_param = MagicMock()
        param.is_sharded = True
        param.shard_world_size = 2
        param.shard_size = 2
        param.replicate_world_size = 2
        param.dp_size = 2
        param.sharded_group_info = MagicMock()
        param.sharded_group_info.group = MagicMock()
        param.sharded_group_info.rank_size = 2
        param.unsharded_group_info = MagicMock()
        param.unsharded_group_info.group = MagicMock()
        param.unsharded_group_info.rank_size = 2
        param._to_local_unsharded_grad.side_effect = lambda grad: grad
        return param

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.pack_for_reduce_scatter")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.build_rs_plan")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.ms.mint.empty")
    def test_reduce_scatter_grad_records_output_and_handle(
        self, mock_empty, mock_reduce_scatter, mock_build_plan, mock_pack
    ):
        """
        Feature: async reduce-scatter
        Description: Launch reduce-scatter and cache output/handle on the param wrapper
        Expectation: output and handle are returned and stored for deferred wait
        """
        param = self._make_param()
        grad = MagicMock()
        grad.dtype = ms.float32
        grad.device = "Ascend:0"
        grad_flat = MagicMock()
        grad.view.return_value = grad_flat
        grad_flat.numel.return_value = 8
        grad.to.return_value = grad
        param.unsharded_grad_data = grad
        param.hsdp_placement = MagicMock()
        param.hsdp_placement.dim = 0
        output = MagicMock(name="reduce_scatter_output")
        handle = MagicMock(name="reduce_scatter_handle")
        plan = MagicMock()
        mock_build_plan.return_value = plan
        mock_pack.return_value = grad_flat
        mock_empty.return_value = output
        mock_reduce_scatter.return_value = handle
        fake_mesh_info = type("FakeFSDPMeshInfo", (), {})
        param.mesh_info = fake_mesh_info()
        param.mesh_info.shard_process_group = MagicMock()

        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.param.FSDPMeshInfo",
            new=fake_mesh_info,
        ):
            reduced_grad, returned_handle = MindSporeHSDPParamV2.reduce_scatter_grad(
                param, async_op=True
            )

        self.assertIs(reduced_grad, output)
        self.assertIs(returned_handle, handle)
        self.assertIs(param._reduce_scatter_output, output)
        self.assertIs(param.reduce_scatter_handle, handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.pack_for_reduce_scatter")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.build_rs_plan")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.ms.mint.empty")
    def test_reduce_scatter_grad_uses_packed_gradient_layout(
        self, mock_empty, mock_reduce_scatter, mock_build_plan, mock_pack
    ):
        """
        Feature: reduce-scatter packing
        Description: Pack the unsharded gradient before passing it to reduce_scatter_tensor
        Expectation: the packed tensor is used as the communication input
        """
        param = self._make_param()
        grad = MagicMock()
        grad.dtype = ms.float32
        grad.device = "Ascend:0"
        packed_grad = MagicMock()
        packed_grad.reshape.return_value = packed_grad
        packed_grad.numel.return_value = 8
        # ``reduce_scatter_grad`` calls ``.contiguous()`` on the packed flat
        # tensor right before ``dist.reduce_scatter_tensor`` to satisfy Ascend
        # HCCL's contig requirement; route the mock back to ``packed_grad`` so
        # the identity assertion below still passes.
        packed_grad.contiguous.return_value = packed_grad
        grad.to.return_value = grad
        param.unsharded_grad_data = grad
        param.hsdp_placement = MagicMock()
        param.hsdp_placement.dim = 1
        output = MagicMock()
        handle = MagicMock()
        plan = MagicMock()
        mock_build_plan.return_value = plan
        mock_pack.return_value = packed_grad
        mock_empty.return_value = output
        mock_reduce_scatter.return_value = handle
        MindSporeHSDPParamV2.reduce_scatter_grad(param, async_op=True)

        mock_build_plan.assert_called_once_with(param, grad, 2)
        mock_pack.assert_called_once_with(grad, plan)
        self.assertIs(mock_reduce_scatter.call_args.args[1], packed_grad)

    def test_reduce_scatter_output_waits_and_clears_handle(self):
        """
        Feature: async reduce-scatter
        Description: Consume a pending reduce-scatter result through reduce_scatter_output
        Expectation: the handle is waited once and then cleared
        """
        param = MagicMock()
        handle = MagicMock()
        output = MagicMock()
        param.reduce_scatter_handle = handle
        param._reduce_scatter_output = output

        result = MindSporeHSDPParamV2.reduce_scatter_output(param)

        handle.wait.assert_called_once_with()
        self.assertIs(result, output)
        self.assertIsNone(param.reduce_scatter_handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_records_output_and_handle(self, mock_all_reduce):
        """
        Feature: async all-reduce
        Description: Launch all-reduce and cache output/handle on the param wrapper
        Expectation: output and handle are returned and stored for deferred wait
        """
        param = self._make_param()
        grad = MagicMock()
        # ``all_reduce_grad`` calls ``.contiguous()`` right before
        # ``dist.all_reduce`` to satisfy Ascend HCCL; route the mock back to
        # ``grad`` so the identity assertions below still pass.
        grad.contiguous.return_value = grad
        handle = MagicMock()
        mock_all_reduce.return_value = handle
        reduced_grad, returned_handle = MindSporeHSDPParamV2.all_reduce_grad(
            param, grad=grad, async_op=True
        )

        self.assertIs(reduced_grad, grad)
        self.assertIs(returned_handle, handle)
        self.assertIs(param._all_reduce_output, grad)
        self.assertIs(param.all_reduce_handle, handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_casts_to_requested_dtype(self, mock_all_reduce):
        """
        Feature: async all-reduce
        Description: Cast the all-reduce gradient to the requested reduce dtype
        Expectation: the casted tensor is cached and returned
        """
        param = self._make_param()
        grad = MagicMock()
        grad.dtype = ms.float16
        cast_grad = MagicMock()
        # The dtype-cast result is what reaches the dist op; ``all_reduce_grad``
        # then calls ``.contiguous()`` on it for Ascend HCCL. Route the mock
        # back to ``cast_grad`` so the identity assertions still pass.
        cast_grad.contiguous.return_value = cast_grad
        grad.to.return_value = cast_grad
        handle = MagicMock()
        mock_all_reduce.return_value = handle
        reduced_grad, _ = MindSporeHSDPParamV2.all_reduce_grad(
            param, grad=grad, dtype=ms.float32, async_op=True
        )

        grad.to.assert_called_once_with(ms.float32)
        self.assertIs(reduced_grad, cast_grad)
        self.assertIs(param._all_reduce_output, cast_grad)

    def test_all_reduce_output_waits_and_clears_handle(self):
        """
        Feature: async all-reduce
        Description: Consume a pending all-reduce result through all_reduce_output
        Expectation: the handle is waited once and then cleared
        """
        param = MagicMock()
        handle = MagicMock()
        output = MagicMock()
        param.all_reduce_handle = handle
        param._all_reduce_output = output

        result = MindSporeHSDPParamV2.all_reduce_output(param)

        handle.wait.assert_called_once_with()
        self.assertIs(result, output)
        self.assertIsNone(param.all_reduce_handle)


class TestAsyncReduceDrain(unittest.TestCase):
    """Test state-level draining of pending sharded reductions."""

    def tearDown(self):
        HSDPState.pre_reduce_scatter_params = []
        HSDPState.pre_all_reduce_params = []

    def test_reduce_scattered_params_drains_reduce_scatter_queue(self):
        """
        Feature: async reduce drain
        Description: Drain a pending reduce-scatter entry from the shared HSDPState queue
        Expectation: reduced grad is waited, cleared, and applied with dtype context
        """
        state = object.__new__(MindSporeHSDPStateV2)
        state.gradient_scaling_factor = None

        param = MagicMock()
        reduced_grad = MagicMock()
        param.reduce_scatter_output.return_value = reduced_grad
        param.apply_reduced_grad.return_value = False
        HSDPState.pre_reduce_scatter_params.append((param, ms.float32))

        state.reduce_scattered_params()

        param.reduce_scatter_output.assert_called_once_with()
        param.clear_reduce_scatter_output.assert_called_once_with()
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, ms.float32)
        self.assertFalse(param.accumulated_allreduced_grad)
        self.assertEqual(HSDPState.pre_reduce_scatter_params, [])

    def test_reduce_params_drains_all_reduce_queue(self):
        """
        Feature: async reduce drain
        Description: Drain a pending all-reduce entry from the shared HSDPState queue
        Expectation: reduced grad is waited, cleared, and applied with dtype context
        """
        state = object.__new__(MindSporeHSDPStateV2)
        state.gradient_scaling_factor = None

        param = MagicMock()
        reduced_grad = MagicMock()
        param.all_reduce_output.return_value = reduced_grad
        param.apply_reduced_grad.return_value = False
        HSDPState.pre_all_reduce_params.append((param, ms.float32))

        state.reduce_params()

        param.all_reduce_output.assert_called_once_with()
        param.clear_all_reduce_output.assert_called_once_with()
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, ms.float32)
        self.assertEqual(HSDPState.pre_all_reduce_params, [])

    def test_reduce_scattered_params_drains_pending_created_by_other_state_instance(self):
        """
        Feature: async reduce drain
        Description: Drain pending reduce-scatter work created by another fully_shard state instance
        Expectation: the shared base-state queue allows cross-state pending work consumption
        """
        state_b = object.__new__(MindSporeHSDPStateV2)
        state_b.gradient_scaling_factor = None

        param = MagicMock()
        reduced_grad = MagicMock()
        param.reduce_scatter_output.return_value = reduced_grad
        param.apply_reduced_grad.return_value = False
        HSDPState.pre_reduce_scatter_params.append((param, ms.float32))

        state_b.reduce_scattered_params()

        param.reduce_scatter_output.assert_called_once_with()
        param.apply_reduced_grad.assert_called_once_with(reduced_grad, ms.float32)
        self.assertEqual(HSDPState.pre_reduce_scatter_params, [])

    def test_reduce_params_rejects_legacy_pending_tuple(self):
        """
        Feature: async reduce drain
        Description: Reject legacy pending all-reduce metadata that still carries need_div
        Expectation: reduce_params raises ValueError during 3-tuple unpack
        """
        state = object.__new__(MindSporeHSDPStateV2)
        param = MagicMock()
        HSDPState.pre_all_reduce_params.append((param, ms.float32, True))

        with self.assertRaisesRegex(ValueError, r"too many values to unpack"):
            state.reduce_params()

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.ms.runtime.current_stream")
    def test_reduce_scattered_params_synchronizes_after_cpu_offload(self, mock_current_stream):
        """
        Feature: async reduce drain
        Description: Synchronize the current stream after apply_reduced_grad requests it
        Expectation: non-blocking CPU offload is made visible before later consumers use the grad
        """
        state = object.__new__(MindSporeHSDPStateV2)
        state.gradient_scaling_factor = None

        stream = MagicMock()
        mock_current_stream.return_value = stream

        param = MagicMock()
        reduced_grad = MagicMock()
        param.reduce_scatter_output.return_value = reduced_grad
        param.apply_reduced_grad.return_value = True
        HSDPState.pre_reduce_scatter_params.append((param, ms.float32))

        state.reduce_scattered_params()

        stream.synchronize.assert_called_once_with()

    @patch("hyper_parallel.platform.mindspore.fully_shard.state.ms.runtime.current_stream")
    def test_reduce_params_synchronizes_once_for_multiple_offloaded_grads(self, mock_current_stream):
        """
        Feature: async reduce drain
        Description: Aggregate synchronize requests across all queued all-reduce gradients
        Expectation: the current stream is synchronized once after the whole drain, matching torch
        """
        state = object.__new__(MindSporeHSDPStateV2)
        state.gradient_scaling_factor = None

        stream = MagicMock()
        mock_current_stream.return_value = stream

        param_a = MagicMock()
        reduced_grad_a = MagicMock()
        param_a.all_reduce_output.return_value = reduced_grad_a
        param_a.apply_reduced_grad.return_value = True

        param_b = MagicMock()
        reduced_grad_b = MagicMock()
        param_b.all_reduce_output.return_value = reduced_grad_b
        param_b.apply_reduced_grad.return_value = True

        HSDPState.pre_all_reduce_params.append((param_a, ms.float32))
        HSDPState.pre_all_reduce_params.append((param_b, ms.float32))

        state.reduce_params()

        stream.synchronize.assert_called_once_with()


class TestAsyncReducePostBackwardHelpers(unittest.TestCase):
    """Test state-level post-backward helper alignment with torch behavior."""

    def tearDown(self):
        HSDPState.pre_reduce_scatter_params = []
        HSDPState.pre_all_reduce_params = []

    def test_has_pending_unsharded_grad_accepts_accumulated_grad(self):
        """
        Feature: async reduce post-backward
        Description: Detect pending accumulated unsharded grad even when .grad is empty
        Expectation: the parameter is considered ready for gradient reduction
        """
        state = object.__new__(MindSporeHSDPStateV2)
        param = MagicMock()
        param.sharded_param.requires_grad = True
        param.unsharded_accumulated_grad = MagicMock()
        param.unsharded_param.grad = None

        self.assertTrue(state._has_pending_unsharded_grad(param))

    def test_queue_compat_all_reduce_passes_reduce_dtype(self):
        """
        Feature: async reduce post-backward
        Description: Launch compat all-reduce with the state's reduce dtype
        Expectation: all_reduce_grad receives dtype so local-only shard paths still cast consistently
        """
        state = object.__new__(MindSporeHSDPStateV2)
        state._reduce_dtype = ms.float16
        state._orig_dtype = ms.float32
        state.reduce_op_type = "reduce_op"
        state.requires_all_reduce = True
        pending_grad = MagicMock()
        state._get_pending_unsharded_grad = MagicMock(return_value=pending_grad)
        param = MagicMock()
        param.dp_size = 2

        state._queue_compat_all_reduce(param)

        # Pure all-reduce path passes grad=None so all_reduce_grad fetches the
        # unsharded grad itself and owns the scaling (mirrors the torch side).
        param.all_reduce_grad.assert_called_once_with(
            dtype=ms.float16,
            async_op=True,
            reduce_op="reduce_op",
        )
        self.assertEqual(HSDPState.pre_all_reduce_params, [(param, ms.float32)])


class TestReduceScatterPackHelpers(unittest.TestCase):
    """Test MindSpore reduce-scatter packing helper alignment with torch semantics."""

    def test_pack_for_reduce_scatter_keeps_dim0_layout(self):
        """
        Feature: reduce-scatter packing
        Description: Keep dim0-sharded gradients in their original row-major layout
        Expectation: the helper returns the original tensor object
        """
        tensor = MagicMock()
        result = _pack_for_reduce_scatter(tensor, shard_dim=0, world_size=2)
        self.assertIs(result, tensor)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.ms.mint.cat")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.ms.mint.chunk")
    def test_pack_for_reduce_scatter_reorders_non_dim0_shards(self, mock_chunk, mock_cat):
        """
        Feature: reduce-scatter packing
        Description: Reorder non-dim0 shards into rank-major reduce-scatter layout
        Expectation: chunk on shard dim, concatenate on dim 0, and return contiguous output
        """
        tensor = MagicMock()
        chunk_0 = MagicMock()
        chunk_1 = MagicMock()
        packed = MagicMock()
        contiguous = MagicMock()
        mock_chunk.return_value = (chunk_0, chunk_1)
        mock_cat.return_value = packed
        packed.contiguous.return_value = contiguous

        result = _pack_for_reduce_scatter(tensor, shard_dim=1, world_size=2)

        mock_chunk.assert_called_once_with(tensor, 2, dim=1)
        mock_cat.assert_called_once_with((chunk_0, chunk_1), dim=0)
        packed.contiguous.assert_called_once_with()
        self.assertIs(result, contiguous)


@unittest.skip(
    "MindSpore scheduler backward-compat hook flow is sensitive to pynative/executor state; "
    "covered under distributed ST/msrun.",
)
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
        scheduler.scheduler_state = FSDPSchedulerState.PRE_FORWARD
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
        scheduler.hsdp_state = MagicMock()

        HSDPSchedulerV2.root_bp_state = True

        scheduler._root_backward_hook()

        scheduler._backward_hook.assert_called_once_with()
        scheduler.hsdp_state.reduce_params.assert_called_once_with()
        scheduler.hsdp_state._finish_ignored_allreduce.assert_called_once_with()
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
        scheduler.hsdp_state = MagicMock()

        HSDPSchedulerV2.root_bp_state = True

        scheduler._root_backward_hook()

        scheduler._backward_hook.assert_called_once_with()
        scheduler.hsdp_state.reduce_params.assert_not_called()
        scheduler.hsdp_state._finish_ignored_allreduce.assert_not_called()
        self.assertTrue(HSDPSchedulerV2.root_bp_state)


if __name__ == "__main__":
    unittest.main()

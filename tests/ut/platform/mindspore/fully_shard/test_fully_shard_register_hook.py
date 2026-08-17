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
"""Unit tests for _register_post_backward_hook in MindSporeHSDPSchedulerV2."""
# pylint: disable=W0212
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

# pylint: disable=C0413
import mindspore as ms
import mindspore.nn as msnn
import numpy as np

from mindspore.common.api import _no_grad
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState
from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2


def _make_scheduler_stub() -> MindSporeHSDPSchedulerV2:
    """Create a minimal MindSporeHSDPSchedulerV2 stub that can call _register_post_backward_hook."""
    scheduler = object.__new__(MindSporeHSDPSchedulerV2)
    scheduler.scheduler_ctx = HSDPSchedulerContext()
    return scheduler


def _call_register_post_backward_hook(scheduler, args, kwargs):
    """Call the tested protected hook helper in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return scheduler._register_post_backward_hook(args, kwargs)


class TestRegisterPostBackwardHook(unittest.TestCase):
    """Unit tests for MindSporeHSDPSchedulerV2._register_post_backward_hook."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
        self.scheduler = _make_scheduler_stub()

    def test_all_no_grad_returns_early(self):
        """When no tensor requires grad, args/kwargs are returned unchanged.

        description: Pass only requires_grad=False tensors.
        expectation: Returns early with original args and kwargs unchanged.
        feature: _register_post_backward_hook early return.
        """
        t1 = ms.Tensor(np.random.randn(2).astype(np.float32))
        t1.requires_grad = False
        t2 = ms.Tensor(np.random.randn(3).astype(np.float32))
        t2.requires_grad = False

        args = (t1, t2)
        kwargs = {}

        out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object (early return), "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object (early return), "
            "got different object"
        )

    def test_no_grad_context_returns_early(self):
        """Under _no_grad context, args/kwargs are returned unchanged.

        description: Wrap the call in _no_grad() context with a requires_grad=True tensor.
        expectation: Returns early with original args and kwargs since grad is disabled.
        feature: _register_post_backward_hook _no_grad context path.
        """
        t1 = ms.Tensor(np.random.randn(2).astype(np.float32))
        t1.requires_grad = True
        args = (t1,)
        kwargs = {}

        with _no_grad():
            out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object under _no_grad, "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object under _no_grad, "
            "got different object"
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.PostBackwardFunction.apply")
    def test_multiple_grad_tensors_preserve_identity(self, mock_apply):
        """Multiple requires_grad=True tensors are passed through correctly.

        description: Pass multiple grad and no-grad tensors in both args and kwargs.
            PostBackwardFunction.apply is mocked so the test stays CPU-only and
            does not invoke MindSpore's pynative autograd / device init.
        expectation: All requires_grad flags are preserved; tensor data is unchanged.
        feature: _register_post_backward_hook multi-tensor correctness.
        """
        g1 = ms.Tensor(np.random.randn(2, 3).astype(np.float32))
        g1.requires_grad = True
        g2 = ms.Tensor(np.random.randn(4).astype(np.float32))
        g2.requires_grad = True
        ng1 = ms.Tensor(np.random.randn(5).astype(np.float32))
        ng1.requires_grad = False
        ng2 = ms.Tensor(np.random.randn(1, 2).astype(np.float32))
        ng2.requires_grad = False

        # Mock apply to echo back the grad-requiring tensors it receives; the
        # scheduler writes them back into their original flattened positions
        # while no-grad tensors are passed through untouched.
        mock_apply.side_effect = lambda _scheduler, *flat: tuple(flat)

        args = (g1, ng1)
        kwargs = {"a": g2, "b": ng2}

        out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        mock_apply.assert_called_once_with(self.scheduler, g1, g2)
        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for args[0], "
            f"got {out_args[0].requires_grad}"
        )
        assert np.array_equal(out_args[0].asnumpy(), g1.asnumpy()), (
            f"Value mismatch for args[0]: "
            f"expected {g1.asnumpy()}, got {out_args[0].asnumpy()}"
        )
        assert np.array_equal(out_args[1].asnumpy(), ng1.asnumpy()), (
            f"Value mismatch for args[1]: "
            f"expected {ng1.asnumpy()}, got {out_args[1].asnumpy()}"
        )
        assert out_kwargs["a"].requires_grad is True, (
            f"Expected requires_grad=True for kwargs['a'], "
            f"got {out_kwargs['a'].requires_grad}"
        )
        assert np.array_equal(out_kwargs["a"].asnumpy(), g2.asnumpy()), (
            f"Value mismatch for kwargs['a']: "
            f"expected {g2.asnumpy()}, got {out_kwargs['a'].asnumpy()}"
        )
        assert np.array_equal(out_kwargs["b"].asnumpy(), ng2.asnumpy()), (
            f"Value mismatch for kwargs['b']: "
            f"expected {ng2.asnumpy()}, got {out_kwargs['b'].asnumpy()}"
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.PostBackwardFunction.apply")
    def test_integer_tensor_not_affected(self, mock_apply):
        """Integer tensors (non-floating) are not affected by requires_grad.

        description: Pass an integer tensor alongside floating-point tensors.
            PostBackwardFunction.apply is mocked so the test stays CPU-only and
            does not invoke MindSpore's pynative autograd / device init.
        expectation: Integer tensor is passed through unchanged; float tensors
            preserve their requires_grad flags.
        feature: _register_post_backward_hook integer tensor handling.
        """
        grad_float = ms.Tensor(np.random.randn(3).astype(np.float32))
        grad_float.requires_grad = True
        int_tensor = ms.Tensor(np.array([1, 2, 3]), dtype=ms.int64)
        int_tensor.requires_grad = False
        no_grad_float = ms.Tensor(np.random.randn(3).astype(np.float32))
        no_grad_float.requires_grad = False

        mock_apply.side_effect = lambda _scheduler, *flat: tuple(flat)

        args = (grad_float, int_tensor, no_grad_float)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        mock_apply.assert_called_once_with(self.scheduler, grad_float)
        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for float grad tensor, "
            f"got {out_args[0].requires_grad}"
        )
        assert np.array_equal(out_args[0].asnumpy(), grad_float.asnumpy()), (
            f"Value mismatch for grad float: "
            f"expected {grad_float.asnumpy()}, got {out_args[0].asnumpy()}"
        )
        assert np.array_equal(out_args[1].asnumpy(), int_tensor.asnumpy()), (
            f"Value mismatch for int tensor: "
            f"expected {int_tensor.asnumpy()}, got {out_args[1].asnumpy()}"
        )
        assert np.array_equal(out_args[2].asnumpy(), no_grad_float.asnumpy()), (
            f"Value mismatch for no-grad float: "
            f"expected {no_grad_float.asnumpy()}, got {out_args[2].asnumpy()}"
        )


@unittest.skip(
    "Prefetch/recompute guard needs full scheduler + platform context; validate under msrun ST.",
)
class TestRecomputeForwardPrefetchGuard(unittest.TestCase):
    """Unit tests for forward-prefetch suppression during activation recompute."""

    def test_forward_pre_hook_disables_prefetch_during_recompute(self):
        """_hsdp_forward_pre_hook clears prefetch targets when root_bp_state is True.

        Description:
            The prefetch-clearing logic lives in _hsdp_forward_pre_hook
            (_disable_forward_prefetch_for_recompute), so _hsdp_forward_pre_hook
            must NOT be mocked — its internal dependencies are mocked instead.
        Expectation: forward_prefetch_cells is empty and _backup_forward_fetch
            holds the original list; _register_post_backward_hook is called with
            the args/kwargs returned by _hsdp_forward_pre_hook.
        """
        scheduler = _make_scheduler_stub()
        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler.cell = MagicMock(name="cell")
        scheduler.scheduler_ctx.root_module = MagicMock(name="root_module")
        original_prefetch = [MagicMock(name="next_module")]
        scheduler.forward_prefetch_cells = list(original_prefetch)
        scheduler._backup_forward_fetch = None
        scheduler._is_root = False
        # module_name non-empty so get_cells_and_names lookup is skipped
        scheduler.hsdp_state = MagicMock(module_name="mod")
        # Mock internal dependencies of _hsdp_forward_pre_hook
        mock_mp_policy = MagicMock()
        mock_mp_policy.cast_forward_inputs = False
        scheduler.mp_policy = mock_mp_policy
        scheduler.platform = MagicMock()
        scheduler.platform.profiler_record.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )
        scheduler._init_params_fqn = MagicMock()
        scheduler._lazy_init_all_states = MagicMock()
        scheduler._register_post_backward_hook = MagicMock(return_value=("wrapped_args", "wrapped_kwargs"))

        scheduler.scheduler_ctx.root_bp_state = True

        result = scheduler._forward_pre_hook(MagicMock(), ("arg",), {"k": "v"})

        self.assertEqual(scheduler.forward_prefetch_cells, [])
        self.assertEqual(scheduler._backup_forward_fetch, original_prefetch)
        scheduler._register_post_backward_hook.assert_called_once_with(("arg",), {"k": "v"})
        self.assertEqual(result, ("wrapped_args", "wrapped_kwargs"))

    def test_forward_hook_restores_prefetch_after_recompute(self):
        """forward hook restores prefetch targets and skips post-forward logic during recompute."""
        scheduler = _make_scheduler_stub()
        scheduler.scheduler_state = FSDPSchedulerState.PRE_FORWARD
        scheduler.forward_prefetch_cells = []
        restored_prefetch = [MagicMock(name="next_module")]
        scheduler._backup_forward_fetch = restored_prefetch.copy()
        outputs = MagicMock(name="outputs")
        scheduler._register_backward_pre_hook = MagicMock()
        scheduler._hsdp_forward_hook = MagicMock()

        scheduler.scheduler_ctx.root_bp_state = True

        result = scheduler._forward_hook(MagicMock(), MagicMock(), outputs)

        scheduler._register_backward_pre_hook.assert_called_once_with(outputs)
        scheduler._hsdp_forward_hook.assert_not_called()
        self.assertIsNone(result)
        self.assertEqual(scheduler.forward_prefetch_cells, restored_prefetch)
        self.assertIsNone(scheduler._backup_forward_fetch)

    def test_backward_pre_hook_prefetches_without_replicate_params(self):
        """backward prefetch should skip replicate_params when reshard_after_forward is enabled."""
        scheduler = _make_scheduler_stub()
        scheduler.scheduler_state = FSDPSchedulerState.FORWARD
        scheduler.cell = MagicMock(name="cell")
        scheduler.reshard_after_forward = True
        scheduler.hsdp_state = MagicMock(module_name="mod")
        scheduler.hsdp_state.unshard = MagicMock()
        prefetch_state = MagicMock(module_name="next_mod")
        prefetch_state.prefetch = MagicMock()
        prefetch_cell = MagicMock()
        prefetch_cell.hsdp_scheduler.hsdp_state = prefetch_state
        scheduler.backward_prefetch_cells = [prefetch_cell]
        scheduler.platform = MagicMock()
        scheduler.platform.profiler_record.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )

        scheduler._hsdp_backward_pre_hook(scheduler.cell, None)

        scheduler.hsdp_state.unshard.assert_called_once_with(unshard_replicate=False)
        prefetch_state.prefetch.assert_called_once_with(unshard_replicate=False)

    def test_forward_pre_hook_with_param_fqn_init(self):
        """_init_params_fqn assigns correct FQNs for a multi-layer nested model.

        Description:
            Build a two-level nested MindSpore Cell (root → layer1 → sub).
            Each submodule that has local parameters contributes mock hsdp_params
            whose sharded_param tensors are the real ms.Parameter objects.
            After calling _init_params_fqn the _param_fqn attribute on each
            wrapper must equal the FQN returned by parameters_and_names().
        Expectation: Success — every hsdp_param._param_fqn matches the FQN
            produced by model.parameters_and_names().
        """
        class _Sub(msnn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(ms.Tensor(np.random.randn(4, 4).astype(np.float32)))

            def construct(self, x):
                return x

        class _Layer(msnn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(ms.Tensor(np.random.randn(4, 4).astype(np.float32)))
                self.sub = _Sub()

            def construct(self, x):
                return x

        class _Root(msnn.Cell):
            def __init__(self):
                super().__init__()
                self.bias = ms.Parameter(ms.Tensor(np.random.randn(4).astype(np.float32)))
                self.layer1 = _Layer()

            def construct(self, x):
                return x

        model = _Root()

        # Ground-truth FQNs from parameters_and_names() before scheduler setup.
        expected_fqns = {param: name for name, param in model.parameters_and_names()}

        class _FakeHSDPParam:
            def __init__(self, real_param):
                self.sharded_param = real_param
                self._param_fqn = None

        class _FakeHSDPState:
            def __init__(self, hsdp_params):
                self.hsdp_params = hsdp_params

        submodule_hsdp_params = {}
        for _, cell in model.cells_and_names():
            local_params = [p for p in cell._params.values() if p is not None]
            if local_params:
                submodule_hsdp_params[cell] = [_FakeHSDPParam(p) for p in local_params]

        scheduler = _make_scheduler_stub()
        scheduler._is_root = True
        scheduler.scheduler_ctx.root_module = model
        scheduler.scheduler_ctx.all_hsdp_schedulers = [
            MagicMock(hsdp_state=_FakeHSDPState(hsdp_params))
            for hsdp_params in submodule_hsdp_params.values()
        ]

        # pylint: disable=protected-access
        scheduler._init_params_fqn()

        for cell, hsdp_params in submodule_hsdp_params.items():
            for hp in hsdp_params:
                expected = expected_fqns.get(hp.sharded_param)
                assert hp._param_fqn == expected, (
                    f"_param_fqn mismatch: expected={expected!r}, got={hp._param_fqn!r}"
                )


if __name__ == "__main__":
    unittest.main()

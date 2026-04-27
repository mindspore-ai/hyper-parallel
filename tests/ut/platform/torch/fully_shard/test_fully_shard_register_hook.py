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
"""Unit tests for _register_post_backward_hook in TorchHSDPSchedulerV2.

Verifies that when forward inputs contain a mix of requires_grad=True and
requires_grad=False tensors, the requires_grad attribute is preserved correctly
after passing through PostBackwardFunction.apply.
"""
# pylint: disable=W0212
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
from torch import nn
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext, HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState
from hyper_parallel.platform.torch.fully_shard.scheduler import TorchHSDPSchedulerV2


def _make_scheduler_stub() -> TorchHSDPSchedulerV2:
    """Create a minimal TorchHSDPSchedulerV2 stub that can call _register_post_backward_hook."""
    scheduler = object.__new__(TorchHSDPSchedulerV2)
    scheduler.scheduler_ctx = HSDPSchedulerContext()
    return scheduler


def _call_register_post_backward_hook(scheduler, args, kwargs):
    """Call the tested protected hook helper in one place for lint cleanliness."""
    # pylint: disable=protected-access
    return scheduler._register_post_backward_hook(args, kwargs)


@unittest.skip("TestRegisterPostBackwardHook temporarily skipped.")
class TestRegisterPostBackwardHook(unittest.TestCase):
    """Unit tests for TorchHSDPSchedulerV2._register_post_backward_hook."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.scheduler = _make_scheduler_stub()

    def test_mixed_requires_grad_preserves_grad_flag(self):
        """Tensors with requires_grad=False stay False after the hook.

        description: Pass args containing both requires_grad=True and
            requires_grad=False floating-point tensors.
        expectation: The requires_grad flag of each tensor in the output
            matches the flag of the corresponding input tensor.
        feature: _register_post_backward_hook mixed requires_grad handling.
        """
        grad_tensor = torch.randn(3, 4, requires_grad=True)
        no_grad_tensor = torch.randn(3, 4, requires_grad=False)

        args = (grad_tensor, no_grad_tensor)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for first arg, "
            f"got {out_args[0].requires_grad}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for second arg, "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[0].data, grad_tensor.data), (
            f"Value mismatch for grad tensor: "
            f"expected {grad_tensor.data}, got {out_args[0].data}"
        )
        assert torch.equal(out_args[1].data, no_grad_tensor.data), (
            f"Value mismatch for no-grad tensor: "
            f"expected {no_grad_tensor.data}, got {out_args[1].data}"
        )

    def test_mixed_requires_grad_in_kwargs(self):
        """Tensors with requires_grad=False in kwargs stay False after the hook.

        description: Pass kwargs containing both requires_grad=True and
            requires_grad=False floating-point tensors.
        expectation: The requires_grad flag is preserved for kwargs tensors.
        feature: _register_post_backward_hook kwargs handling.
        """
        args = ()
        kwargs = {
            "x": torch.randn(2, 2, requires_grad=True),
            "y": torch.randn(2, 2, requires_grad=False),
        }

        _, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_kwargs["x"].requires_grad is True, (
            f"Expected requires_grad=True for kwarg 'x', "
            f"got {out_kwargs['x'].requires_grad}"
        )
        assert out_kwargs["y"].requires_grad is False, (
            f"Expected requires_grad=False for kwarg 'y', "
            f"got {out_kwargs['y'].requires_grad}"
        )
        assert torch.equal(out_kwargs["x"].data, kwargs["x"].data), (
            f"Value mismatch for kwarg 'x': "
            f"expected {kwargs['x'].data}, got {out_kwargs['x'].data}"
        )
        assert torch.equal(out_kwargs["y"].data, kwargs["y"].data), (
            f"Value mismatch for kwarg 'y': "
            f"expected {kwargs['y'].data}, got {out_kwargs['y'].data}"
        )

    def test_mixed_grad_with_non_tensor(self):
        """Non-tensor objects are passed through unchanged.

        description: Pass args containing a grad tensor, a no-grad tensor,
            an integer, and None.
        expectation: Tensors preserve requires_grad; non-tensors are unchanged.
        feature: _register_post_backward_hook non-tensor passthrough.
        """
        grad_tensor = torch.randn(2, requires_grad=True)
        no_grad_tensor = torch.randn(2, requires_grad=False)

        args = (grad_tensor, no_grad_tensor, 42, None)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for arg[0], "
            f"got {out_args[0].requires_grad}"
        )
        assert torch.equal(out_args[0].data, grad_tensor.data), (
            f"Value mismatch for grad tensor: "
            f"expected {grad_tensor.data}, got {out_args[0].data}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for arg[1], "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[1].data, no_grad_tensor.data), (
            f"Value mismatch for no-grad tensor: "
            f"expected {no_grad_tensor.data}, got {out_args[1].data}"
        )
        assert out_args[2] == 42, (
            f"Expected integer 42, got {out_args[2]}"
        )
        assert out_args[3] is None, (
            f"Expected None, got {out_args[3]}"
        )

    def test_all_no_grad_returns_early(self):
        """When no tensor requires grad, args/kwargs are returned unchanged.

        description: Pass only requires_grad=False tensors.
        expectation: Returns early with original args and kwargs unchanged.
        feature: _register_post_backward_hook early return.
        """
        t1 = torch.randn(2, requires_grad=False)
        t2 = torch.randn(3, requires_grad=False)

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

    def test_grad_disabled_returns_early(self):
        """When grad is disabled, args/kwargs are returned unchanged.

        description: Disable grad globally and pass tensors.
        expectation: Returns early with original args and kwargs unchanged.
        feature: _register_post_backward_hook grad disabled path.
        """
        t1 = torch.randn(2, requires_grad=True)
        args = (t1,)
        kwargs = {}

        with torch.no_grad():
            out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args is args, (
            "Expected same args object under no_grad, "
            "got different object"
        )
        assert out_kwargs is kwargs, (
            "Expected same kwargs object under no_grad, "
            "got different object"
        )

    def test_multiple_grad_tensors_preserve_identity(self):
        """Multiple requires_grad=True tensors are passed through correctly.

        description: Pass multiple grad and no-grad tensors in both args and kwargs.
        expectation: All requires_grad flags are preserved; tensor data is unchanged.
        feature: _register_post_backward_hook multi-tensor correctness.
        """
        g1 = torch.randn(2, 3, requires_grad=True)
        g2 = torch.randn(4, requires_grad=True)
        ng1 = torch.randn(5, requires_grad=False)
        ng2 = torch.randn(1, 2, requires_grad=False)

        args = (g1, ng1)
        kwargs = {"a": g2, "b": ng2}

        out_args, out_kwargs = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for args[0], "
            f"got {out_args[0].requires_grad}"
        )
        assert torch.equal(out_args[0].data, g1.data), (
            f"Value mismatch for args[0]: "
            f"expected {g1.data}, got {out_args[0].data}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for args[1], "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[1].data, ng1.data), (
            f"Value mismatch for args[1]: "
            f"expected {ng1.data}, got {out_args[1].data}"
        )
        assert out_kwargs["a"].requires_grad is True, (
            f"Expected requires_grad=True for kwargs['a'], "
            f"got {out_kwargs['a'].requires_grad}"
        )
        assert torch.equal(out_kwargs["a"].data, g2.data), (
            f"Value mismatch for kwargs['a']: "
            f"expected {g2.data}, got {out_kwargs['a'].data}"
        )
        assert out_kwargs["b"].requires_grad is False, (
            f"Expected requires_grad=False for kwargs['b'], "
            f"got {out_kwargs['b'].requires_grad}"
        )
        assert torch.equal(out_kwargs["b"].data, ng2.data), (
            f"Value mismatch for kwargs['b']: "
            f"expected {ng2.data}, got {out_kwargs['b'].data}"
        )

    def test_integer_tensor_not_affected(self):
        """Integer tensors (non-floating) are not affected by requires_grad.

        description: Pass an integer tensor alongside floating-point tensors.
        expectation: Integer tensor is passed through unchanged; float tensors
            preserve their requires_grad flags.
        feature: _register_post_backward_hook integer tensor handling.
        """
        grad_float = torch.randn(3, requires_grad=True)
        int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        no_grad_float = torch.randn(3, requires_grad=False)

        args = (grad_float, int_tensor, no_grad_float)
        kwargs = {}

        out_args, _ = _call_register_post_backward_hook(self.scheduler, args, kwargs)

        assert out_args[0].requires_grad is True, (
            f"Expected requires_grad=True for float grad tensor, "
            f"got {out_args[0].requires_grad}"
        )
        assert torch.equal(out_args[0].data, grad_float.data), (
            f"Value mismatch for grad float: "
            f"expected {grad_float.data}, got {out_args[0].data}"
        )
        assert out_args[1].requires_grad is False, (
            f"Expected requires_grad=False for int tensor, "
            f"got {out_args[1].requires_grad}"
        )
        assert torch.equal(out_args[1], int_tensor), (
            f"Value mismatch for int tensor: "
            f"expected {int_tensor}, got {out_args[1]}"
        )
        assert out_args[2].requires_grad is False, (
            f"Expected requires_grad=False for no-grad float tensor, "
            f"got {out_args[2].requires_grad}"
        )
        assert torch.equal(out_args[2].data, no_grad_float.data), (
            f"Value mismatch for no-grad float: "
            f"expected {no_grad_float.data}, got {out_args[2].data}"
        )


class TestRecomputeForwardPrefetchGuard(unittest.TestCase):
    """Unit tests for forward-prefetch suppression during activation recompute."""

    def tearDown(self):
        HSDPSchedulerV2.root_bp_state = False

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

        HSDPSchedulerV2.root_bp_state = True

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

        HSDPSchedulerV2.root_bp_state = True

        result = scheduler._forward_hook(MagicMock(), MagicMock(), outputs)

        scheduler._register_backward_pre_hook.assert_called_once_with(outputs)
        scheduler._hsdp_forward_hook.assert_not_called()
        self.assertIsNone(result)
        self.assertEqual(scheduler.forward_prefetch_cells, restored_prefetch)
        self.assertIsNone(scheduler._backup_forward_fetch)

    def test_forward_pre_hook_with_param_fqn_init(self):
        """_init_params_fqn assigns correct FQNs for a multi-layer nested model.

        Description:
            Build a two-level nested model (root → layer1 → sub).  Each
            submodule that has local parameters contributes mock hsdp_params
            whose sharded_param tensors are the real nn.Parameter objects.
            After calling _init_params_fqn the _param_fqn attribute on each
            wrapper must equal the FQN returned by named_parameters().
        Expectation: Success — every hsdp_param._param_fqn matches the FQN
            produced by model.named_parameters().
        """
        # Two-level nested model: root has its own param, layer1 is a Linear,
        # layer1.sub is a nested Linear — all registered via nn.Module.
        class _Sub(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))
                self.sub = _Sub()

        class _Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(4))
                self.layer1 = _Layer()

        model = _Root()

        # Ground-truth FQNs from named_parameters() before scheduler setup.
        expected_fqns = {param: name for name, param in model.named_parameters()}

        # Use a plain object instead of MagicMock so that attribute writes on
        # _param_fqn and sharded_param behave like normal Python attributes.
        # MagicMock intercepts single-underscore names and _iter_managed_params
        # .return_value would silently produce a new Mock instead of our list.
        class _FakeHSDPParam:
            def __init__(self, real_param):
                self.sharded_param = real_param
                self._param_fqn = None

        class _FakeHSDPState:
            def __init__(self, hsdp_params):
                self._hsdp_params = hsdp_params

            def _iter_managed_params(self):
                return self._hsdp_params

        submodule_hsdp_params = {}
        for module in model.modules():
            local_params = [p for p in module._parameters.values() if p is not None]
            if local_params:
                submodule_hsdp_params[module] = [_FakeHSDPParam(p) for p in local_params]

        def _fake_get_hsdp_state(module):
            if module not in submodule_hsdp_params:
                return None
            return _FakeHSDPState(submodule_hsdp_params[module])

        scheduler = _make_scheduler_stub()
        scheduler._is_root = True
        scheduler.scheduler_ctx.root_module = model

        with patch(
            "hyper_parallel.core.fully_shard.hsdp_scheduler.get_hsdp_state",
            side_effect=_fake_get_hsdp_state,
        ):
            # pylint: disable=protected-access
            scheduler._init_params_fqn()
        # Every hsdp_param must have received the correct FQN.
        all_fqns = [
            (hp._param_fqn, expected_fqns.get(hp.sharded_param))
            for hsdp_params in submodule_hsdp_params.values()
            for hp in hsdp_params
        ]
        print("\n[_param_fqn check]")
        for got, expected in all_fqns:
            print(f"  got={got!r:40s}  expected={expected!r}")
        for module, hsdp_params in submodule_hsdp_params.items():
            for hp in hsdp_params:
                expected = expected_fqns.get(hp.sharded_param)
                assert hp._param_fqn == expected, (
                    f"_param_fqn mismatch: expected={expected!r}, got={hp._param_fqn!r}"
                )


if __name__ == "__main__":
    unittest.main()

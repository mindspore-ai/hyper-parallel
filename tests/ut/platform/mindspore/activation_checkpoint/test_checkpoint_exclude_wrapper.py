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
"""Tests for the MindSpore checkpoint exclusion wrapper."""
import contextlib
import importlib
from typing import Iterator, Tuple
import unittest

import numpy as np
import pytest
from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    ensure_mindspore_platform_default,
)

ms = pytest.importorskip("mindspore")
ParameterTuple = ms.ParameterTuple
Tensor = ms.Tensor
nn = ms.nn
ops = ms.ops
_pynative_executor = importlib.import_module("mindspore.graph.api")._pynative_executor

ensure_mindspore_platform_default()

activation_checkpoint = importlib.import_module("hyper_parallel.core.activation_checkpoint")
checkpoint_wrapper = activation_checkpoint.checkpoint_wrapper
checkpoint_exclude_wrapper = activation_checkpoint.checkpoint_exclude_wrapper
is_recomputing = activation_checkpoint.is_recomputing
get_platform = importlib.import_module("hyper_parallel.platform").get_platform
forward_and_gradfn = importlib.import_module(
    "hyper_parallel.platform.mindspore.pipeline_parallel.backward"
).forward_and_gradfn
checkpoint_exclude_wrapper_module = importlib.import_module(
    "hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_exclude_wrapper"
)
CheckpointExcludeWrapper = checkpoint_exclude_wrapper_module.CheckpointExcludeWrapper


class _FunctionBlock(nn.Cell):
    """Checkpointed block with a plain function excluded from recomputation."""

    def __init__(self, calls: dict) -> None:
        """Initialize the wrapped middle function and shared call counter."""
        super().__init__()
        self.calls = calls

        def middle(x: Tensor, y: Tensor) -> Tensor:
            """Multiply inputs and count real executions."""
            self.calls["middle"] += 1
            return x * y

        self.middle = checkpoint_exclude_wrapper(middle)

    def construct(self, x: Tensor, y: Tensor) -> Tensor:
        """Keep the excluded function between two recomputed operations."""
        head = x * y
        middle = self.middle(head, y)
        tail = middle * y
        return tail.sum()


class _StateBlock(nn.Cell):
    """Record the unified recompute state around an excluded region."""

    def __init__(self, states: list, calls: dict) -> None:
        """Initialize the recorder and wrapped middle function."""
        super().__init__()
        self.states = states

        def middle(x: Tensor) -> Tensor:
            """Record one real execution of the excluded region."""
            calls["middle"] += 1
            return x * x

        self.middle = checkpoint_exclude_wrapper(middle)

    def construct(self, x: Tensor) -> Tensor:
        """Record the current phase before entering the wrapped region."""
        self.states.append(is_recomputing())
        output = self.middle(x * 2)
        return (output * x).sum()


class _PrecisionMiddle(nn.Cell):
    """Parameterized nonlinear region used by the precision test."""

    def __init__(self, calls: dict) -> None:
        """Initialize deterministic parameters and the shared call counter."""
        super().__init__()
        self.calls = calls
        self.dense = nn.Dense(
            5,
            5,
            has_bias=True,
            weight_init=Tensor(np.linspace(-0.2, 0.2, 25, dtype=np.float32).reshape(5, 5)),
            bias_init=Tensor(np.linspace(-0.1, 0.1, 5, dtype=np.float32)),
        )

    def construct(self, x: Tensor) -> Tensor:
        """Run a nonlinear transformation and record real executions."""
        self.calls["middle"] += 1
        output = self.dense(x)
        return ops.tanh(output) + output * output * 0.01


class _PrecisionBlock(nn.Cell):
    """Small deterministic network with an optionally excluded middle Cell."""

    def __init__(self, calls: dict, skip_middle: bool) -> None:
        """Initialize deterministic layers and optionally wrap the middle Cell."""
        super().__init__()
        self.head = nn.Dense(
            4,
            5,
            has_bias=True,
            weight_init=Tensor(np.linspace(-0.3, 0.3, 20, dtype=np.float32).reshape(5, 4)),
            bias_init=Tensor(np.linspace(-0.05, 0.05, 5, dtype=np.float32)),
        )
        middle = _PrecisionMiddle(calls)
        self.middle = checkpoint_exclude_wrapper(middle) if skip_middle else middle
        self.tail = nn.Dense(
            5,
            3,
            has_bias=True,
            weight_init=Tensor(np.linspace(-0.25, 0.25, 15, dtype=np.float32).reshape(3, 5)),
            bias_init=Tensor(np.linspace(-0.02, 0.02, 3, dtype=np.float32)),
        )

    def construct(self, x: Tensor) -> Tensor:
        """Return a scalar loss for input and parameter gradient comparison."""
        output = self.head(x)
        output = self.middle(output)
        output = self.tail(output)
        return (output * output).mean()


def _run_precision_sequence(net):
    """Run two different inputs through one network and collect all gradients."""
    weights = ParameterTuple(net.trainable_params())
    value_and_grad = ms.value_and_grad(net, grad_position=0, weights=weights)
    base_input = np.linspace(-1.5, 1.25, 24, dtype=np.float32).reshape(6, 4)
    results = []
    for scale in (1.0, 1.7):
        value, (input_grad, param_grads) = value_and_grad(Tensor(base_input * scale))
        results.append((value, input_grad, tuple(param_grads)))
    return results


def _run_split_backward(net, x: Tensor, prefire_recompute: bool):
    """Run the PP recompute-session sequence and return value, dx, and dw."""
    platform = get_platform()
    weights = tuple(net.trainable_params())
    with platform.recompute_handle_collector_ctx() as handles:
        value, grad_fn = forward_and_gradfn(net, x, weights=weights, grad_position=0)

    session_id = ("checkpoint_exclude_dxdw", id(net))
    if prefire_recompute:
        if not handles:
            raise AssertionError("PP dx/dw validation requires at least one recompute handle")
        with platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
            for handle in handles:
                platform.recompute_handle(handle, session_id)

    session_context = (
        platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=True)
        if handles else contextlib.nullcontext()
    )
    with session_context:
        input_grad = grad_fn.compute_input_grad()

    session_context = (
        platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=False)
        if handles else contextlib.nullcontext()
    )
    with session_context:
        weight_grads = grad_fn.compute_weight_grad()
    if handles:
        platform.clear_recompute_session(session_id)
    return value, input_grad, weight_grads


class TestCheckpointExcludeWrapper(unittest.TestCase):
    """Validate MindSpore function and Cell checkpoint exclusion wrappers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run saved tensor hooks in supported PyNative mode."""
        ms.set_context(mode=ms.PYNATIVE_MODE)

    def setUp(self) -> None:
        """Clear PyNative autodiff state left by earlier same-process tests."""
        ensure_mindspore_platform_default()
        _pynative_executor.clear_res()

    def tearDown(self) -> None:
        """Keep this test's PyNative autodiff state away from later tests."""
        _pynative_executor.clear_res()

    def test_saved_tensor_hooks_detach_only_when_packing(self):
        """The pack hook detaches while the unpack hook restores its result unchanged."""
        tensor = Tensor(np.arange(4, dtype=np.float32))

        packed = checkpoint_exclude_wrapper_module._pack_saved_tensor(tensor)

        self.assertIsNot(packed, tensor)
        np.testing.assert_array_equal(packed.asnumpy(), tensor.asnumpy())
        self.assertIs(checkpoint_exclude_wrapper_module._unpack_saved_tensor(packed), packed)

    def test_function_wrapper_skips_middle_recompute(self):
        """A wrapped function should run in forward and reuse its output in recompute."""
        calls = {"middle": 0}
        block = _FunctionBlock(calls)
        net = checkpoint_wrapper(block)
        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))

        value, grads = ms.value_and_grad(net, grad_position=(0, 1))(x, y)

        self.assertIsInstance(block.middle, CheckpointExcludeWrapper)
        self.assertEqual(calls["middle"], 1)
        np.testing.assert_allclose(value.asnumpy(), np.array(80.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grads[0].asnumpy(), np.full((4,), 8, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            grads[1].asnumpy(),
            np.array([12, 24, 36, 48], np.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_unified_state_reports_forward_and_recompute(self):
        """The shared state API should distinguish checkpoint execution phases."""
        states = []
        calls = {"middle": 0}
        net = checkpoint_wrapper(_StateBlock(states, calls))
        x = Tensor(np.arange(1, 5, dtype=np.float32))

        _ = ms.value_and_grad(net, grad_position=0)(x)

        self.assertEqual(states, [False, True])
        self.assertEqual(calls["middle"], 1)
        self.assertFalse(is_recomputing())

    def test_wrapper_is_transparent_outside_checkpoint(self):
        """Outside checkpoint execution the wrapper should call through normally."""
        calls = {"middle": 0}

        def multiply(x: Tensor, y: Tensor) -> Tensor:
            """Multiply inputs and record direct executions."""
            calls["middle"] += 1
            return x * y

        wrapped = checkpoint_exclude_wrapper(multiply)
        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))

        output = wrapped(x, y)

        self.assertEqual(calls["middle"], 1)
        np.testing.assert_allclose(output.asnumpy(), (x * y).asnumpy(), atol=1e-6, rtol=1e-6)

    def test_same_checkpoint_wrapper_keeps_invocation_caches_isolated(self):
        """Separate checkpoint invocations should reuse their matching outputs."""
        calls = {"middle": 0}
        net = checkpoint_wrapper(_FunctionBlock(calls))

        def forward(x: Tensor, y: Tensor) -> Tensor:
            """Invoke one checkpoint wrapper twice with different inputs."""
            return net(x, y) + net(x * 1.7, y)

        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))

        value, grads = ms.value_and_grad(forward, grad_position=(0, 1))(x, y)

        self.assertEqual(calls["middle"], 2)
        np.testing.assert_allclose(value.asnumpy(), np.array(216.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grads[0].asnumpy(), np.full((4,), 21.6, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            grads[1].asnumpy(),
            np.array([32.4, 64.8, 97.2, 129.6], np.float32),
            atol=1e-5,
            rtol=1e-6,
        )

    def test_checkpoint_exclusion_composes_with_user_context(self):
        """A user context should compose with checkpoint exclusion state."""
        calls = {"middle": 0}
        events = []

        @contextlib.contextmanager
        def record(name: str) -> Iterator[None]:
            """Record entry and exit around one checkpoint phase."""
            events.append(f"enter:{name}")
            try:
                yield
            finally:
                events.append(f"exit:{name}")

        def context_fn() -> Tuple[object, object]:
            """Create user contexts for the original forward and replay."""
            return record("forward"), record("recompute")

        net = checkpoint_wrapper(_FunctionBlock(calls), context_fn=context_fn)
        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))

        value, grads = ms.value_and_grad(net, grad_position=(0, 1))(x, y)

        self.assertEqual(calls["middle"], 1)
        self.assertEqual(events, ["enter:forward", "exit:forward", "enter:recompute", "exit:recompute"])
        np.testing.assert_allclose(value.asnumpy(), np.array(80.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grads[0].asnumpy(), np.full((4,), 8, np.float32), atol=1e-6, rtol=1e-6)

    def test_platform_checkpoint_preserves_default_reentrant_behavior(self):
        """The platform checkpoint API should retain MindSpore's reentrant default."""
        calls = {"middle": 0}
        block = _FunctionBlock(calls)
        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))

        def forward(input_x: Tensor, input_y: Tensor) -> Tensor:
            """Use the low-level platform checkpoint without checkpoint kwargs."""
            return get_platform().checkpoint(block, input_x, input_y)

        value, grads = ms.value_and_grad(forward, grad_position=(0, 1))(x, y)

        self.assertEqual(calls["middle"], 2)
        np.testing.assert_allclose(value.asnumpy(), np.array(80.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grads[0].asnumpy(), np.full((4,), 8, np.float32), atol=1e-6, rtol=1e-6)

    def test_rejects_non_callable_input(self):
        """The public wrapper should reject values that cannot be executed."""
        with self.assertRaisesRegex(ValueError, "Cell or callable"):
            checkpoint_exclude_wrapper(1)

    def test_cell_wrapper_preserves_repeated_step_precision(self):
        """Repeated calls should match non-checkpoint values and all gradients."""
        reference_calls = {"middle": 0}
        skip_calls = {"middle": 0}
        reference_net = _PrecisionBlock(reference_calls, skip_middle=False)
        skip_block = _PrecisionBlock(skip_calls, skip_middle=True)
        skip_net = checkpoint_wrapper(skip_block)

        reference_results = _run_precision_sequence(reference_net)
        skip_results = _run_precision_sequence(skip_net)

        self.assertEqual(reference_calls["middle"], 2)
        self.assertEqual(skip_calls["middle"], 2)
        for step, (reference, actual) in enumerate(zip(reference_results, skip_results)):
            reference_value, reference_input_grad, reference_param_grads = reference
            actual_value, actual_input_grad, actual_param_grads = actual
            np.testing.assert_allclose(
                actual_value.asnumpy(), reference_value.asnumpy(), atol=1e-6, rtol=1e-6,
                err_msg=f"value mismatch at step {step}",
            )
            np.testing.assert_allclose(
                actual_input_grad.asnumpy(), reference_input_grad.asnumpy(), atol=1e-6, rtol=1e-6,
                err_msg=f"input gradient mismatch at step {step}",
            )
            self.assertEqual(len(actual_param_grads), len(reference_param_grads))
            for param_index, (reference_grad, actual_grad) in enumerate(zip(reference_param_grads, actual_param_grads)):
                np.testing.assert_allclose(
                    actual_grad.asnumpy(), reference_grad.asnumpy(), atol=1e-6, rtol=1e-6,
                    err_msg=f"parameter gradient {param_index} mismatch at step {step}",
                )

    def test_prefired_recompute_supports_dxdw_split(self):
        """PP pre-recompute, dx, and dw should share one excluded-region result."""
        reference_calls = {"middle": 0}
        skip_calls = {"middle": 0}
        reference_net = _PrecisionBlock(reference_calls, skip_middle=False)
        skip_net = checkpoint_wrapper(_PrecisionBlock(skip_calls, skip_middle=True))
        input_data = np.linspace(-1.5, 1.25, 24, dtype=np.float32).reshape(6, 4)

        reference = _run_split_backward(reference_net, Tensor(input_data), prefire_recompute=False)
        actual = _run_split_backward(skip_net, Tensor(input_data), prefire_recompute=True)

        self.assertEqual(reference_calls["middle"], 1)
        self.assertEqual(skip_calls["middle"], 1)
        for name, reference_value, actual_value in zip(("value", "dx"), reference[:2], actual[:2]):
            np.testing.assert_allclose(
                actual_value.asnumpy(), reference_value.asnumpy(), atol=1e-6, rtol=1e-6,
                err_msg=f"{name} mismatch for PP dx/dw split",
            )
        self.assertEqual(len(actual[2]), len(reference[2]))
        for index, (reference_grad, actual_grad) in enumerate(zip(reference[2], actual[2])):
            np.testing.assert_allclose(
                actual_grad.asnumpy(), reference_grad.asnumpy(), atol=1e-6, rtol=1e-6,
                err_msg=f"dw {index} mismatch for PP dx/dw split",
            )


if __name__ == "__main__":
    unittest.main()

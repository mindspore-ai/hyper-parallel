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
import gc
import importlib
from typing import Any, Iterator, Tuple
import unittest
import weakref

import numpy as np
import pytest
from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    ensure_mindspore_platform_default,
)

ms = pytest.importorskip("mindspore")
Parameter = ms.Parameter
Tensor = ms.Tensor
nn = ms.nn
ops = ms.ops
_pynative_executor = importlib.import_module("mindspore.graph.api")._pynative_executor

ensure_mindspore_platform_default()
enable_mindspore_backward_compat = importlib.import_module(
    "hyper_parallel.platform.mindspore.autograd_compat"
).enable_mindspore_backward_compat
enable_mindspore_backward_compat()

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


class _SavedInputBlock(nn.Cell):
    """Checkpointed block whose excluded tail saves its exact input."""

    def __init__(self, calls: dict) -> None:
        """Wrap a custom saved-input function and retain the shared counter."""
        super().__init__()
        self.calls = calls

        def excluded(tensor: Tensor) -> Tensor:
            """Save the exact recomputed input without rerunning on replay."""
            self.calls["excluded"] += 1
            return tensor * tensor

        self.excluded = checkpoint_exclude_wrapper(excluded)

    def construct(self, tensor: Tensor) -> Tensor:
        """End the checkpoint body at the excluded region's scalar output."""
        self.calls["block"] += 1
        return self.excluded(tensor * 3).sum()


class _MultiOutputSaveBlock(nn.Cell):
    """Checkpointed block with two SAVE outputs consumed as one argument."""

    def __init__(self, calls: dict) -> None:
        """Initialize adjacent SAVE regions and their execution counters."""
        super().__init__()

        def first(tensor: Tensor) -> tuple:
            """Return two differentiable outputs while saving the input."""
            calls["first"] += 1
            return tensor * tensor, tensor * tensor * tensor

        def second(outputs: tuple) -> Tensor:
            """Consume the complete structured output as one SAVE argument."""
            calls["second"] += 1
            return outputs[0] + outputs[1]

        self.first = checkpoint_exclude_wrapper(first, save_output=False)
        self.second = checkpoint_exclude_wrapper(second)

    def construct(self, tensor: Tensor) -> Tensor:
        """Return a loss depending on both outputs of the first SAVE region."""
        return self.second(self.first(tensor)).sum()


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
    weights = tuple(net.trainable_params())
    base_input = np.linspace(-1.5, 1.25, 24, dtype=np.float32).reshape(6, 4)
    results = []
    for scale in (1.0, 1.7):
        tensor = Tensor(base_input * scale)
        tensor.requires_grad = True
        value = net(tensor)
        value.backward()
        input_grad = tensor.grad
        param_grads = tuple(parameter.grad for parameter in weights)
        results.append((value, input_grad, tuple(param_grads)))
        for parameter in weights:
            parameter.grad = None
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
        _pynative_executor.set_grad_flag(True)

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

    def test_saved_tensor_hook_uses_tensor_user_data_handle(self):
        """Tensor user data should carry the deferred handle into the pack hook."""
        tensor = Tensor(np.arange(4, dtype=np.float32))
        recomputed = Tensor(np.arange(4, dtype=np.float32) * 2)
        handle = checkpoint_exclude_wrapper_module._RecomputedInputHandle()
        key = checkpoint_exclude_wrapper_module._RECOMPUTE_INPUT_HANDLE_KEY
        tensor._set_user_data(key, handle)

        packed = checkpoint_exclude_wrapper_module._pack_saved_tensor(tensor)

        self.assertIs(packed, handle)
        self.assertTrue(handle.used)
        with self.assertRaisesRegex(RuntimeError, "before recomputation"):
            checkpoint_exclude_wrapper_module._unpack_saved_tensor(packed)
        handle.materialize(recomputed.data)
        unpacked = checkpoint_exclude_wrapper_module._unpack_saved_tensor(packed)
        np.testing.assert_array_equal(unpacked.asnumpy(), recomputed.asnumpy())

    def test_recompute_boundary_saves_zero_element_trigger(self):
        """The replay boundary should not retain storage from its tensor output."""
        packed_shapes = []
        unpacked_shapes = []

        def pack_hook(tensor: Tensor) -> Tensor:
            """Record the tensor packed by the boundary."""
            packed_shapes.append(tuple(tensor.shape))
            return tensor

        def unpack_hook(tensor: Tensor) -> Tensor:
            """Record the tensor unpacked by the boundary."""
            unpacked_shapes.append(tuple(tensor.shape))
            return tensor

        def apply_boundary(tensor: Tensor) -> Tensor:
            """Apply the boundary in a differentiable function."""
            trigger = checkpoint_exclude_wrapper_module._get_recompute_trigger()
            return checkpoint_exclude_wrapper_module._RecomputeBoundary.apply(tensor, trigger).sum()

        tensor = Tensor(np.arange(4, dtype=np.float32))
        tensor.requires_grad = True
        with ms.saved_tensors_hooks(pack_hook, unpack_hook):
            value = apply_boundary(tensor)
            value.backward()

        self.assertEqual(packed_shapes, [(0,)])
        self.assertEqual(unpacked_shapes, [(0,)])
        np.testing.assert_array_equal(tensor.grad.asnumpy(), np.ones(4, dtype=np.float32))

    def test_replay_placeholder_is_lazily_reused(self):
        """Replay should reuse one zero-element placeholder."""
        get_placeholder = checkpoint_exclude_wrapper_module._get_replay_placeholder
        get_placeholder.cache_clear()

        self.assertEqual(get_placeholder.cache_info().currsize, 0)
        placeholder = get_placeholder()
        self.assertIs(placeholder, get_placeholder())
        self.assertEqual(tuple(placeholder.shape), (0,))
        self.assertEqual(get_placeholder.cache_info().currsize, 1)

    def test_parameter_input_is_not_marked(self):
        """Parameter inputs should keep the existing saved-tensor behavior."""
        parameter = Parameter(Tensor(np.arange(4, dtype=np.float32)), name="exclude_parameter")

        bindings, previous_handles = checkpoint_exclude_wrapper_module._mark_recompute_inputs(
            object(), (parameter,), {}
        )

        self.assertEqual(bindings, [])
        self.assertEqual(previous_handles, [])

    def test_input_marking_rolls_back_when_nested_traversal_fails(self):
        """A traversal failure should not leave handles on inputs already visited."""
        class _BrokenDict(dict):
            """Raise while exposing nested items."""

            def items(self) -> Any:
                """Fail before returning nested items."""
                raise RuntimeError("nested traversal failed")

        tensor = Tensor(np.arange(4, dtype=np.float32))
        key = checkpoint_exclude_wrapper_module._RECOMPUTE_INPUT_HANDLE_KEY

        with self.assertRaisesRegex(RuntimeError, "nested traversal failed"):
            checkpoint_exclude_wrapper_module._mark_recompute_inputs(
                object(), (tensor, _BrokenDict()), {}
            )

        self.assertIsNone(tensor._get_user_data(key))

    def test_nested_input_paths_materialize_matching_replay_tensors(self):
        """Nested args and kwargs should bind handles without retaining forward inputs."""
        first = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        second = Tensor(np.array([3.0, 4.0], dtype=np.float32))
        args = ({"items": [first]}, first)
        kwargs = {"second": second}

        bindings, previous_handles = checkpoint_exclude_wrapper_module._mark_recompute_inputs(
            object(), args, kwargs
        )
        checkpoint_exclude_wrapper_module._restore_recompute_inputs(previous_handles)
        for binding in bindings:
            binding.handle.mark_used()

        replay_first = Tensor(np.array([5.0, 6.0], dtype=np.float32))
        replay_second = Tensor(np.array([7.0, 8.0], dtype=np.float32))
        entry = checkpoint_exclude_wrapper_module._ExcludeCacheEntry(
            output=None,
            input_bindings=bindings,
        )
        checkpoint_exclude_wrapper_module._materialize_recompute_inputs(
            entry,
            ({"items": [replay_first]}, replay_first),
            {"second": replay_second},
        )

        self.assertEqual(len(bindings), 2)
        self.assertEqual(bindings[0].path, (("arg", 0), ("key", "items"), ("index", 0)))
        self.assertEqual(bindings[1].path, (("kwarg", "second"),))
        np.testing.assert_array_equal(
            bindings[0].handle.get_recomputed_tensor().asnumpy(), replay_first.asnumpy()
        )
        np.testing.assert_array_equal(
            bindings[1].handle.get_recomputed_tensor().asnumpy(), replay_second.asnumpy()
        )

    def test_input_traversal_does_not_require_cycle_collection(self):
        """Input traversal should release tensor leaves without waiting for cyclic GC."""
        def traverse_tensor() -> weakref.ReferenceType:
            """Collect a tensor input and return a non-owning reference to it."""
            tensor = Tensor(np.arange(4, dtype=np.float32))
            leaves = checkpoint_exclude_wrapper_module._collect_tensor_inputs(([tensor],), {})
            self.assertEqual(len(leaves), 1)
            return weakref.ref(tensor)

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            tensor_ref = traverse_tensor()
            self.assertIsNone(tensor_ref())
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()

    def test_recomputed_input_materializes_before_excluded_backward(self):
        """A terminal excluded region should recover its saved input from replay."""
        calls = {"block": 0, "excluded": 0}
        net = checkpoint_wrapper(_SavedInputBlock(calls))
        tensor = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        tensor.requires_grad = True

        value = net(tensor)
        value.backward()

        np.testing.assert_allclose(value.asnumpy(), np.array(45.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            tensor.grad.asnumpy(),
            np.array([18.0, 36.0], np.float32),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(calls, {"block": 2, "excluded": 1})

    def test_save_output_source_is_not_marked_for_recomputation(self):
        """Inputs produced by SAVE regions should remain owned by their downstream pack."""
        invocation_id = object()
        tensor = Tensor(np.arange(4, dtype=np.float32))
        tensor.requires_grad = True
        output = checkpoint_exclude_wrapper_module._finalize_save_outputs(tensor, True, invocation_id)

        bindings, previous_handles = checkpoint_exclude_wrapper_module._mark_recompute_inputs(
            invocation_id, (output,), {}
        )

        self.assertEqual(bindings, [])
        self.assertEqual(previous_handles, [])

        bindings, previous_handles = checkpoint_exclude_wrapper_module._mark_recompute_inputs(
            object(), (output,), {}
        )
        checkpoint_exclude_wrapper_module._restore_recompute_inputs(previous_handles)
        self.assertEqual(len(bindings), 1)

    def test_elided_output_cache_does_not_retain_tensor_storage(self):
        """An elided SAVE output cache should not own the forward tensor."""
        tensor = Tensor(np.arange(4, dtype=np.float32))
        tensor_ref = weakref.ref(tensor)

        entry = checkpoint_exclude_wrapper_module._ExcludeCacheEntry(
            output=None,
            input_bindings=[],
        )
        del tensor

        self.assertIsNone(tensor_ref())
        self.assertIsNone(entry.output)
        placeholder = checkpoint_exclude_wrapper_module._get_replay_placeholder()
        self.assertEqual(tuple(placeholder.shape), (0,))
        self.assertEqual(placeholder.dtype, ms.float32)
        self.assertFalse(placeholder._requires_grad)  # pylint: disable=protected-access

    def test_elided_multi_output_replay_preserves_boundary_count(self):
        """Replay should create one placeholder boundary for each forward Tensor output."""
        calls = {"first": 0, "second": 0}
        net = checkpoint_wrapper(_MultiOutputSaveBlock(calls))
        tensor = Tensor(np.array([1.0, 2.0], dtype=np.float32))
        tensor.requires_grad = True

        value = net(tensor)
        value.backward()

        np.testing.assert_allclose(value.asnumpy(), np.array(14.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            tensor.grad.asnumpy(),
            np.array([5.0, 16.0], np.float32),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(calls, {"first": 1, "second": 1})

    def test_function_wrapper_skips_middle_recompute(self):
        """A wrapped function should run in forward and reuse its output in recompute."""
        calls = {"middle": 0}
        block = _FunctionBlock(calls)
        net = checkpoint_wrapper(block)
        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))
        x.requires_grad = True
        y.requires_grad = True

        value = net(x, y)
        value.backward()
        grads = (x.grad, y.grad)

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
        x.requires_grad = True

        value = net(x)
        value.backward()

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
        x.requires_grad = True
        y.requires_grad = True

        value = forward(x, y)
        value.backward()
        grads = (x.grad, y.grad)

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
        x.requires_grad = True
        y.requires_grad = True

        value = net(x, y)
        value.backward()
        grads = (x.grad, y.grad)

        self.assertEqual(calls["middle"], 1)
        self.assertEqual(events, ["enter:forward", "exit:forward", "enter:recompute", "exit:recompute"])
        np.testing.assert_allclose(value.asnumpy(), np.array(80.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grads[0].asnumpy(), np.full((4,), 8, np.float32), atol=1e-6, rtol=1e-6)

    def test_platform_checkpoint_supports_non_reentrant_compat_backward(self):
        """The low-level checkpoint API should support compatibility backward."""
        calls = {"middle": 0}
        block = _FunctionBlock(calls)
        x = Tensor(np.arange(1, 5, dtype=np.float32))
        y = Tensor(np.full((4,), 2, dtype=np.float32))
        x.requires_grad = True
        y.requires_grad = True

        def forward(input_x: Tensor, input_y: Tensor) -> Tensor:
            """Use the supported non-reentrant low-level checkpoint path."""
            return get_platform().checkpoint(block, input_x, input_y, use_reentrant=False)

        value = forward(x, y)
        value.backward()
        grads = (x.grad, y.grad)

        self.assertEqual(calls["middle"], 2)
        np.testing.assert_allclose(value.asnumpy(), np.array(80.0, np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grads[0].asnumpy(), np.full((4,), 8, np.float32), atol=1e-6, rtol=1e-6)

    def test_rejects_non_callable_input(self):
        """The public wrapper should reject values that cannot be executed."""
        with self.assertRaisesRegex(ValueError, "Cell or callable"):
            checkpoint_exclude_wrapper(1)

    def test_rejects_non_boolean_save_output(self):
        """The save_output option should reject ambiguous truthy values."""
        with self.assertRaisesRegex(ValueError, "save_output must be a bool"):
            checkpoint_exclude_wrapper(lambda tensor: tensor, save_output=1)

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

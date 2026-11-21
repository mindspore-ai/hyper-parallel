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
"""Unit tests for the eager non-reentrant checkpoint implementation."""
# The backend selector must be set before importing HyperParallel modules.
# Tests also inspect private helpers to cover backend-independent control flow.
# pylint: disable=wrong-import-position,protected-access
import contextlib
import importlib
import inspect
import unittest
from typing import Callable
from unittest.mock import MagicMock, call, patch, sentinel

import torch


from hyper_parallel.core.activation_memory.checkpoint import (
    CheckpointError,
    checkpoint,
    clear_recompute_session,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
)

_checkpoint = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint")


class _TensorWithDeviceType(torch.Tensor):
    """CPU-backed tensor subclass exposing a synthetic device for helper tests."""

    __slots__ = ("device_type", "device_index")

    @staticmethod
    def __new__(cls, device_type, device_index=0):
        instance = torch.Tensor._make_subclass(cls, torch.empty(0), False)
        instance.device_type = device_type
        instance.device_index = device_index
        return instance

    @property
    def device(self) -> torch.device:
        """Return the synthetic device without allocating hardware memory."""
        return torch.device(self.device_type)

    def get_device(self) -> int:
        """Return the synthetic device index."""
        return self.device_index


def _make_tail_recording_function(
    function: Callable[..., torch.Tensor], tail_calls: list[None]
) -> Callable[..., torch.Tensor]:
    """Wrap a test function and record each completed invocation."""
    def wrapped(*args: torch.Tensor) -> torch.Tensor:
        """Run the wrapped function and append one tail-call marker."""
        result = function(*args)
        tail_calls.append(None)
        return result

    return wrapped


class _CheckpointTestCase(unittest.TestCase):
    """Base class that isolates checkpoint module global state."""

    def setUp(self) -> None:
        """Save RNG state and start each test without registered sessions."""
        self.rng_state = torch.get_rng_state()
        with _checkpoint._SESSION_FRAMES_LOCK:
            _checkpoint._SESSION_FRAMES.clear()

    def tearDown(self) -> None:
        """Restore RNG state and release any retained recomputation data."""
        torch.set_rng_state(self.rng_state)
        with _checkpoint._SESSION_FRAMES_LOCK:
            session_ids = list(_checkpoint._SESSION_FRAMES)
        for session_id in session_ids:
            clear_recompute_session(session_id)


class TestCheckpointHelpers(_CheckpointTestCase):
    """Unit tests for checkpoint state and device helper functions."""

    def test_noop_context_fn_returns_two_contexts(self):
        """The default context factory should provide usable no-op contexts."""
        forward_context, recompute_context = _checkpoint._noop_context_fn()

        with forward_context, recompute_context:
            value = "inside"

        self.assertEqual(value, "inside")

    def test_default_metadata_fn_describes_tensor(self):
        """Default determinism metadata should include shape, dtype, and device."""
        tensor = torch.ones(2, 3, dtype=torch.float64)

        metadata = _checkpoint._default_metadata_fn(tensor)

        self.assertEqual(
            metadata,
            {"shape": torch.Size([2, 3]), "dtype": torch.float64, "device": torch.device("cpu")},
        )

    def test_infer_device_type_uses_default_for_cpu_inputs(self):
        """CPU-only nested inputs should use Torch's configured default device type."""
        cases = (
            ("xla", {"tensor": torch.ones(2), "nested": [torch.zeros(1)]}),
            ("npu", torch.ones(1)),
        )
        for expected, inputs in cases:
            with self.subTest(device_type=expected), patch.object(
                _checkpoint.DefaultDeviceType, "get_device_type", return_value=expected
            ):
                self.assertEqual(_checkpoint._infer_device_type(inputs), expected)

    def test_infer_device_type_warns_and_prefers_cuda(self):
        """Mixed non-CPU device types should warn and prefer CUDA."""
        cuda_tensor = _TensorWithDeviceType("cuda")
        xla_tensor = _TensorWithDeviceType("xla")

        with self.assertWarnsRegex(UserWarning, "multiple non-CPU device types"):
            device_type = _checkpoint._infer_device_type({"xla": xla_tensor}, cuda_tensor)

        self.assertEqual(device_type, "cuda")

    def test_get_device_module_handles_meta_and_regular_devices(self):
        """Device lookup should special-case meta and expose regular Torch modules."""
        self.assertEqual(_checkpoint._get_device_module("meta"), torch.device("meta"))
        self.assertIs(_checkpoint._get_device_module("cpu"), torch.cpu)

    def test_get_and_set_device_states_use_device_contexts(self):
        """RNG helpers should capture and restore each matching input device."""
        tensor = _TensorWithDeviceType("cuda", device_index=3)
        device_module = MagicMock()
        device_module.device.side_effect = contextlib.nullcontext
        device_module.get_rng_state.return_value = sentinel.rng_state

        with patch.object(_checkpoint, "_get_device_module", return_value=device_module):
            devices, states = _checkpoint._get_device_states("cuda", {"tensor": tensor}, torch.ones(1))
            _checkpoint._set_device_states("cuda", devices, states)

        self.assertEqual(devices, [3])
        self.assertEqual(states, [sentinel.rng_state])
        self.assertEqual(device_module.device.call_args_list, [call(3), call(3)])
        device_module.get_rng_state.assert_called_once_with()
        device_module.set_rng_state.assert_called_once_with(sentinel.rng_state)

    def test_disabled_autocast_does_not_initialize_accelerator(self):
        """Inactive accelerator autocast must not probe a device backend."""
        disabled = {"enabled": False, "dtype": torch.float16, "cache_enabled": True}

        with patch.object(_checkpoint.torch.amp, "autocast") as autocast:
            context = _checkpoint._autocast_context("npu", disabled)

        self.assertIsInstance(context, contextlib.nullcontext)
        autocast.assert_not_called()

    def test_enabled_autocast_preserves_device_options(self):
        """Active accelerator autocast should pass through the captured options."""
        enabled = {"enabled": True, "dtype": torch.float16, "cache_enabled": False}
        expected = contextlib.nullcontext()

        with patch.object(_checkpoint.torch.amp, "autocast", return_value=expected) as autocast:
            context = _checkpoint._autocast_context("cuda", enabled)

        self.assertIs(context, expected)
        autocast.assert_called_once_with(device_type="cuda", **enabled)

    def test_meta_device_states_are_empty_and_restore_is_noop(self):
        """Meta tensors should not participate in RNG state capture or restore."""
        devices, states = _checkpoint._get_device_states("meta", torch.empty(2, device="meta"))

        self.assertEqual(devices, [])
        self.assertEqual(states, [])
        self.assertIsNone(_checkpoint._set_device_states("meta", [0], [sentinel.rng_state]))

    def test_noop_save_inputs_reconstructs_mixed_arguments(self):
        """The internal input saver should retain tensors and non-tensor values."""
        dummy = torch.empty(0, requires_grad=True)
        tensor = torch.tensor([1.0], requires_grad=True)

        saved = _checkpoint._NoopSaveInputs.apply(dummy, {"scale": 2}, tensor, "tag")
        input_context = saved.grad_fn
        restored = input_context.get_args(input_context.saved_tensors)

        self.assertEqual(restored[0], {"scale": 2})
        self.assertIs(restored[1], tensor)
        self.assertEqual(restored[2], "tag")

    def test_noop_save_inputs_rejects_direct_backward(self):
        """Direct backward through the internal input saver should fail clearly."""
        dummy = torch.empty(0, requires_grad=True)
        saved = _checkpoint._NoopSaveInputs.apply(dummy, {})

        with self.assertRaisesRegex(CheckpointError, "must not be backwarded directly"):
            saved.sum().backward()

    def test_internal_assert_raises_checkpoint_error(self):
        """Internal invariant failures should use the checkpoint-specific error."""
        with self.assertRaisesRegex(CheckpointError, "broken invariant"):
            _checkpoint._internal_assert(False, "broken invariant")


class TestCheckpointValidation(_CheckpointTestCase):
    """Unit tests for checkpoint argument validation and compile delegation."""

    def test_checkpoint_rejects_invalid_boolean_options(self):
        """Checkpoint boolean controls should reject truthy non-boolean values."""
        cases = [
            ({"use_reentrant": True}, "only supports use_reentrant=False"),
            ({"early_stop": 1}, "early_stop must be bool"),
            ({"preserve_rng_state": 1}, "preserve_rng_state must be bool"),
        ]

        for options, message in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    checkpoint(lambda value: value, torch.ones(1), **options)

    def test_checkpoint_rejects_non_callable_context_fn(self):
        """The context factory must be callable."""
        with self.assertRaisesRegex(ValueError, "context_fn must be callable"):
            checkpoint(lambda value: value, torch.ones(1), context_fn=None)

    def test_checkpoint_rejects_invalid_context_result(self):
        """The context factory must return exactly two context managers."""
        context_factories = [contextlib.nullcontext, lambda: (contextlib.nullcontext(),)]

        for context_fn in context_factories:
            with self.subTest(context_fn=context_fn):
                with self.assertRaisesRegex(ValueError, "context_fn must return"):
                    checkpoint(lambda value: value, torch.ones(1), context_fn=context_fn)

    def test_checkpoint_rejects_invalid_determinism_mode(self):
        """Only the documented determinism modes should be accepted."""
        with self.assertRaisesRegex(ValueError, "determinism_check must be one of"):
            checkpoint(lambda value: value, torch.ones(1), determinism_check="invalid")

    def test_eager_checkpoint_rejects_debug_mode(self):
        """Eager execution should reject unsupported debug tracing."""
        with patch.object(_checkpoint, "_is_compiling", return_value=False):
            with self.assertRaisesRegex(ValueError, "debug=True is not supported"):
                checkpoint(lambda value: value, torch.ones(1), debug=True)

    def test_compile_checkpoint_delegates_all_options(self):
        """Compile execution should delegate to the native checkpoint adapter."""
        function = MagicMock()
        context_fn = MagicMock()
        tensor = torch.ones(2)

        with patch.object(_checkpoint, "_is_compiling", return_value=True), patch.object(
            _checkpoint,
            "_native_checkpoint",
            return_value=sentinel.result,
        ) as native_checkpoint:
            result = checkpoint(
                function,
                tensor,
                context_fn=context_fn,
                preserve_rng_state=False,
                determinism_check="none",
                debug=True,
                early_stop=False,
                scale=2,
            )

        self.assertIs(result, sentinel.result)
        native_checkpoint.assert_called_once_with(
            function,
            tensor,
            context_fn=context_fn,
            preserve_rng_state=False,
            determinism_check="none",
            debug=True,
            early_stop=False,
            scale=2,
        )

    def test_native_checkpoint_uses_non_reentrant_api_and_early_stop_context(self):
        """The native adapter should set early-stop state and force non-reentrant mode."""
        function = MagicMock()
        context_fn = MagicMock()
        early_stop_context = MagicMock()

        with patch.object(
            _checkpoint,
            "set_checkpoint_early_stop",
            return_value=early_stop_context,
        ) as set_early_stop, patch.object(
            _checkpoint,
            "torch_checkpoint",
            return_value=sentinel.result,
        ) as torch_checkpoint:
            result = _checkpoint._native_checkpoint(
                function,
                sentinel.argument,
                context_fn=context_fn,
                preserve_rng_state=False,
                determinism_check="none",
                debug=True,
                early_stop=False,
                keyword=sentinel.keyword,
            )

        self.assertIs(result, sentinel.result)
        set_early_stop.assert_called_once_with(False)
        torch_checkpoint.assert_called_once_with(
            function,
            sentinel.argument,
            use_reentrant=False,
            context_fn=context_fn,
            preserve_rng_state=False,
            determinism_check="none",
            debug=True,
            keyword=sentinel.keyword,
        )
        early_stop_context.__enter__.assert_called_once_with()
        early_stop_context.__exit__.assert_called_once()

    def test_checkpoint_closes_generator_when_forward_raises(self):
        """A failing checkpointed function should close its setup generator."""
        generator_closed = []

        def _setup_generator(*args, **kwargs):
            del args, kwargs
            try:
                yield
            finally:
                generator_closed.append(True)

        def _failing_function():
            raise RuntimeError("forward failed")

        with patch.object(_checkpoint, "_checkpoint_without_reentrant_generator", _setup_generator):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                checkpoint(_failing_function)

        self.assertEqual(generator_closed, [True])

    def test_checkpoint_rejects_generator_with_multiple_yields(self):
        """The checkpoint setup generator must complete after the forward call."""
        def _invalid_generator(*args, **kwargs):
            del args, kwargs
            yield
            yield

        with patch.object(_checkpoint, "_checkpoint_without_reentrant_generator", _invalid_generator):
            with self.assertRaisesRegex(CheckpointError, "yielded more than once"):
                checkpoint(lambda: sentinel.result)


class TestEagerCheckpoint(_CheckpointTestCase):
    """Exercise eager checkpoint control flow with real CPU autograd."""

    def test_function_keyword_arguments_are_forwarded(self) -> None:
        """Non-checkpoint control kwargs should be passed to the wrapped function."""
        input_tensor = torch.randn(4, requires_grad=True)

        output = checkpoint(lambda x, scale: x.sin() * scale, input_tensor, scale=3.0)
        output.sum().backward()

        self.assertTrue(torch.allclose(output, input_tensor.sin() * 3.0))

    def test_early_stop_controls_tail_execution(self) -> None:
        """early_stop should stop before side effects after the last saved tensor."""
        for early_stop, expected_tail_calls in ((True, 1), (False, 2)):
            with self.subTest(early_stop=early_stop):
                tail_calls = []

                function = _make_tail_recording_function(
                    lambda input_tensor: input_tensor.sin().cos(), tail_calls
                )

                input_tensor = torch.randn(4, requires_grad=True)
                checkpoint(function, input_tensor, early_stop=early_stop).sum().backward()

                self.assertEqual(len(tail_calls), expected_tail_calls)

    def test_native_early_stop_context_does_not_override_keyword(self) -> None:
        """The eager implementation should use only Hyper's explicit keyword."""
        tail_calls = []

        def function(input_tensor):
            output = input_tensor.sin().cos()
            tail_calls.append(None)
            return output

        input_tensor = torch.randn(4, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            checkpoint(function, input_tensor, early_stop=True).sum().backward()

        self.assertEqual(len(tail_calls), 1)

    def test_forward_and_recompute_contexts_are_used(self) -> None:
        """The two contexts should surround their corresponding executions."""
        events = []

        @contextlib.contextmanager
        def record(name):
            events.append(f"enter:{name}")
            try:
                yield
            finally:
                events.append(f"exit:{name}")

        def context_fn():
            return record("forward"), record("recompute")

        input_tensor = torch.randn(4, requires_grad=True)
        checkpoint(lambda x: x.sin(), input_tensor, context_fn=context_fn).sum().backward()

        self.assertEqual(
            events,
            ["enter:forward", "exit:forward", "enter:recompute", "exit:recompute"],
        )

    def test_no_grad_execution_does_not_create_handle(self) -> None:
        """A checkpoint outside autograd should behave as a direct function call."""
        with recompute_handle_collector_ctx() as handles:
            with torch.no_grad():
                output = checkpoint(lambda x: x.sin(), torch.ones(4))

        self.assertTrue(torch.equal(output, torch.ones(4).sin()))
        self.assertEqual(handles, [])

    def test_failed_forward_does_not_leave_handle(self) -> None:
        """A failed checkpoint invocation should be removed from its collector."""
        def function(input_tensor):
            del input_tensor
            raise RuntimeError("forward failed")

        with recompute_handle_collector_ctx() as handles:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                checkpoint(function, torch.ones(1, requires_grad=True))

        self.assertEqual(handles, [])


class TestRecomputeSessions(_CheckpointTestCase):
    """Unit tests for recomputation session validation."""

    def test_nested_session_context_is_rejected_and_outer_context_recovers(self):
        """Nested session scopes should fail without leaking the outer activation."""
        with recompute_session_ctx("outer"):
            with self.assertRaisesRegex(CheckpointError, "Nested recompute session contexts"):
                with recompute_session_ctx("inner"):
                    self.fail("The nested context should not be entered.")

        self.assertIsNone(_checkpoint._RECOMPUTE_SESSION.get())

    def test_session_apis_validate_handles_and_session_ids(self):
        """Session APIs should reject invalid handles, null IDs, unhashable IDs, and retention flags."""
        with self.assertRaisesRegex(ValueError, "handle must be produced"):
            recompute_handle(object(), "session")
        session_parameter = inspect.signature(recompute_session_ctx).parameters["session_id"]
        self.assertIs(session_parameter.default, inspect.Parameter.empty)
        with self.assertRaisesRegex(ValueError, "session_id must not be None"):
            clear_recompute_session(None)
        with self.assertRaisesRegex(ValueError, "session_id must not be None"):
            with recompute_session_ctx(None):
                self.fail("A null session should not be entered.")
        with self.assertRaisesRegex(ValueError, "session_id must be hashable"):
            with recompute_session_ctx([]):
                self.fail("An unhashable session should not be entered.")
        with self.assertRaisesRegex(ValueError, "retain_on_unpack must be bool"):
            with recompute_session_ctx("session", retain_on_unpack=1):
                self.fail("An invalid retention policy should not be entered.")


    def test_repeated_sessions_keep_iterations_isolated(self) -> None:
        """Repeated iterations should not reuse frames or tensors from earlier sessions."""
        weight = torch.randn(4, 4, requires_grad=True)
        calls = []

        def function(current_input, current_weight):
            calls.append(None)
            return torch.nn.functional.gelu(current_input @ current_weight)

        for step in range(3):
            input_tensor = torch.randn(4, 4, requires_grad=True)
            reference_input = input_tensor.detach().clone().requires_grad_()
            reference_weight = weight.detach().clone().requires_grad_()
            reference_output = torch.nn.functional.gelu(reference_input @ reference_weight)
            expected_dx, expected_dw = torch.autograd.grad(
                reference_output.sum(),
                (reference_input, reference_weight),
            )
            with recompute_handle_collector_ctx() as handles:
                output = checkpoint(function, input_tensor, weight)

            session_id = ("iteration", step, id(output))
            try:
                recompute_handle(handles[0], session_id)
                with recompute_session_ctx(session_id, retain_on_unpack=True):
                    actual_dx = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
                with recompute_session_ctx(session_id, retain_on_unpack=False):
                    actual_dw = torch.autograd.grad(output.sum(), weight)[0]
            finally:
                clear_recompute_session(session_id)
            clear_recompute_session(session_id)

            self.assertTrue(torch.allclose(actual_dx, expected_dx))
            self.assertTrue(torch.allclose(actual_dw, expected_dw))
            self.assertEqual(len(calls), (step + 1) * 2)

    def test_partial_and_failed_sessions_can_be_cleared_and_reused(self) -> None:
        """Early-stop, partial consumption, and failed prefire should leave reusable frames."""
        for early_stop, expected_tail_calls in ((True, 1), (False, 3)):
            with self.subTest(early_stop=early_stop):
                input_tensor = torch.randn(4, 4, requires_grad=True)
                weight = torch.randn(4, 4, requires_grad=True)
                reference_input = input_tensor.detach().clone().requires_grad_()
                reference_weight = weight.detach().clone().requires_grad_()
                reference_output = torch.nn.functional.gelu(reference_input @ reference_weight)
                expected_dx, expected_dw = torch.autograd.grad(
                    reference_output.sum(),
                    (reference_input, reference_weight),
                )
                tail_calls = []

                function = _make_tail_recording_function(
                    lambda current_input, current_weight: torch.nn.functional.gelu(current_input @ current_weight),
                    tail_calls,
                )

                with recompute_handle_collector_ctx() as handles:
                    output = checkpoint(function, input_tensor, weight, early_stop=early_stop)

                session_id = ("partial", early_stop, id(output))
                try:
                    recompute_handle(handles[0], session_id)
                    with recompute_session_ctx(session_id, retain_on_unpack=True):
                        actual_dx = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
                finally:
                    clear_recompute_session(session_id)

                actual_dw = torch.autograd.grad(output.sum(), weight)[0]
                self.assertEqual(len(tail_calls), expected_tail_calls)
                self.assertTrue(torch.allclose(actual_dx, expected_dx))
                self.assertTrue(torch.allclose(actual_dw, expected_dw))

        input_tensor = torch.randn(4, requires_grad=True)
        should_fail = True
        calls = []

        def failing_function(value):
            calls.append(None)
            result = value.sin().cos()
            if should_fail and len(calls) > 1:
                raise RuntimeError("expected recomputation failure")
            return result

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(failing_function, input_tensor, early_stop=False)

        failed_session_id = ("failed", id(output))
        try:
            with self.assertRaisesRegex(RuntimeError, "expected recomputation failure"):
                recompute_handle(handles[0], failed_session_id)
        finally:
            clear_recompute_session(failed_session_id)

        should_fail = False
        recovered_session_id = ("recovered", id(output))
        try:
            recompute_handle(handles[0], recovered_session_id)
            with recompute_session_ctx(recovered_session_id, retain_on_unpack=False):
                actual_grad = torch.autograd.grad(output.sum(), input_tensor)[0]
        finally:
            clear_recompute_session(recovered_session_id)

        reference_input = input_tensor.detach().clone().requires_grad_()
        expected_grad = torch.autograd.grad(reference_input.sin().cos().sum(), reference_input)[0]
        self.assertEqual(len(calls), 3)
        self.assertTrue(torch.allclose(actual_grad, expected_grad))

    def test_prefired_session_survives_contextvar_loss(self) -> None:
        """A worker without the caller ContextVar should still find prefired tensors."""
        calls = []
        input_tensor = torch.randn(4, 4, requires_grad=True)
        weight = torch.randn(4, 4, requires_grad=True)

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(function, input_tensor, weight)

        session_id = "worker-context-loss"
        try:
            recompute_handle(handles[0], session_id)
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                token = _checkpoint._RECOMPUTE_SESSION.set(None)
                try:
                    torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)
                finally:
                    _checkpoint._RECOMPUTE_SESSION.reset(token)
            with recompute_session_ctx(session_id, retain_on_unpack=False):
                token = _checkpoint._RECOMPUTE_SESSION.set(None)
                try:
                    torch.autograd.grad(output.sum(), weight)
                finally:
                    _checkpoint._RECOMPUTE_SESSION.reset(token)
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(calls), 2)

    def test_scheduled_nested_checkpoint_raises(self) -> None:
        """Scheduled recomputation should reject nested checkpoint regions explicitly."""
        input_tensor = torch.randn(4, requires_grad=True)

        def inner(value):
            return value.sin()

        def outer(value):
            return checkpoint(inner, value).cos()

        with recompute_handle_collector_ctx() as handles:
            checkpoint(outer, input_tensor)

        session_id = "nested-prefire"
        try:
            with self.assertRaisesRegex(CheckpointError, "Nested checkpoint is not supported"):
                recompute_handle(handles[0], session_id)
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(handles), 2)

    def test_clear_session_is_idempotent(self) -> None:
        """A retained session can be cleared repeatedly in cleanup paths."""
        input_tensor = torch.randn(4, requires_grad=True)
        with recompute_handle_collector_ctx() as handles:
            checkpoint(lambda x: x.sin(), input_tensor)

        session_id = "clear-twice"
        recompute_handle(handles[0], session_id)
        clear_recompute_session(session_id)
        clear_recompute_session(session_id)

    def test_unpack_outside_backward_uses_temporary_graph_key(self) -> None:
        """An eager saved-tensor access outside backward should recompute successfully."""
        input_tensor = torch.randn(4, requires_grad=True)
        output = checkpoint(lambda value: value.sin(), input_tensor)

        saved_input = output.grad_fn._saved_self  # pylint: disable=W0212

        self.assertTrue(torch.equal(saved_input, input_tensor))

    def test_incomplete_forward_ignores_extra_save_only_without_early_stop(self) -> None:
        """Only a full recomputation may run ahead of an incomplete original forward."""
        input_tensor = torch.randn(4, requires_grad=True)
        early_stop_frame = _checkpoint._CheckpointFrame(lambda: None, True, None)
        with self.assertRaises(CheckpointError):
            with _checkpoint._create_recomputation_hooks(early_stop_frame, "early-stop"):
                input_tensor.sin()

        full_recompute_frame = _checkpoint._CheckpointFrame(lambda: None, False, None)
        with _checkpoint._create_recomputation_hooks(full_recompute_frame, "full-recompute"):
            input_tensor.sin()
        self.assertTrue(full_recompute_frame.ignore_saved_mismatch)

    def test_recompute_metadata_mismatch_raises(self) -> None:
        """Default determinism checks should detect changed recompute tensor metadata."""
        recomputing = False

        @contextlib.contextmanager
        def recompute_context():
            nonlocal recomputing
            recomputing = True
            try:
                yield
            finally:
                recomputing = False

        def function(input_tensor):
            value = input_tensor.half() if recomputing else input_tensor
            return value.sin()

        input_tensor = torch.randn(4, requires_grad=True)
        output = checkpoint(
            function,
            input_tensor,
            context_fn=lambda: (contextlib.nullcontext(), recompute_context()),
        )

        with self.assertRaises(CheckpointError):
            output.sum().backward()

    def test_none_determinism_check_allows_metadata_change(self) -> None:
        """The none mode should keep count checks while skipping tensor metadata checks."""
        recomputing = False

        @contextlib.contextmanager
        def recompute_context():
            nonlocal recomputing
            recomputing = True
            try:
                yield
            finally:
                recomputing = False

        def function(input_tensor):
            value = input_tensor.half() if recomputing else input_tensor
            return value.sin()

        input_tensor = torch.randn(4, requires_grad=True)
        output = checkpoint(
            function,
            input_tensor,
            context_fn=lambda: (contextlib.nullcontext(), recompute_context()),
            determinism_check="none",
        )

        output.sum().backward()


if __name__ == "__main__":
    unittest.main()

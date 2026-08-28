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
"""NPU system-test cases for Hyper's Torch non-reentrant checkpoint implementation."""
# pylint: disable=missing-public-docstring,missing-public-type-hints,wrong-import-position,protected-access
import contextlib
import importlib
import inspect
import os
import unittest
from typing import Callable, List
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform.torch.activation_checkpoint.checkpoint import (
    CheckpointError,
    checkpoint,
    clear_recompute_session,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
)
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, checkpoint as core_checkpoint
from tests.torch.activation_checkpoint.utils import set_seed

checkpoint_module = importlib.import_module("hyper_parallel.platform.torch.activation_checkpoint.checkpoint")

_NPU_DEVICE = "npu"


def _randn(*size: int, **kwargs) -> torch.Tensor:
    """Create a random tensor explicitly on NPU."""
    return torch.randn(*size, device=_NPU_DEVICE, **kwargs)


def _ones(*size: int, **kwargs) -> torch.Tensor:
    """Create an all-ones tensor explicitly on NPU."""
    return torch.ones(*size, device=_NPU_DEVICE, **kwargs)


def _make_tail_recording_function(
    function: Callable[..., torch.Tensor], tail_calls: List[None]
) -> Callable[..., torch.Tensor]:
    """Wrap a test function and record each completed invocation."""
    def wrapped(*args: torch.Tensor) -> torch.Tensor:
        """Run the wrapped function and append one tail-call marker."""
        result = function(*args)
        tail_calls.append(None)
        return result

    return wrapped


class TestCheckpoint(unittest.TestCase):
    """Validate eager checkpoint semantics and input checks."""

    def test_backward_matches_non_checkpointed_function(self):
        """Checkpoint gradients should match regular autograd for inputs and weights."""
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()
        baseline_weight = weight.detach().clone().requires_grad_()

        baseline = torch.sin(baseline_input * baseline_weight)
        baseline.sum().backward()

        output = checkpoint(lambda x, w: torch.sin(x * w), input_tensor, weight)
        output.sum().backward()

        self.assertTrue(torch.allclose(input_tensor.grad, baseline_input.grad))
        self.assertTrue(torch.allclose(weight.grad, baseline_weight.grad))

    def test_function_keyword_arguments_are_forwarded(self):
        """Non-checkpoint control kwargs should be passed to the wrapped function."""
        input_tensor = _randn(4, requires_grad=True)

        output = checkpoint(lambda x, scale: x.sin() * scale, input_tensor, scale=3.0)
        output.sum().backward()

        self.assertTrue(torch.allclose(output, input_tensor.sin() * 3.0))

    def test_early_stop_controls_tail_execution(self):
        """early_stop should stop before side effects after the last saved tensor."""
        for early_stop, expected_tail_calls in ((True, 1), (False, 2)):
            with self.subTest(early_stop=early_stop):
                tail_calls = []

                function = _make_tail_recording_function(
                    lambda input_tensor: input_tensor.sin().cos(), tail_calls
                )

                input_tensor = _randn(4, requires_grad=True)
                checkpoint(function, input_tensor, early_stop=early_stop).sum().backward()

                self.assertEqual(len(tail_calls), expected_tail_calls)

    def test_native_early_stop_context_does_not_override_keyword(self):
        """The eager implementation should use only Hyper's explicit keyword."""
        tail_calls = []

        def function(input_tensor):
            output = input_tensor.sin().cos()
            tail_calls.append(None)
            return output

        input_tensor = _randn(4, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            checkpoint(function, input_tensor, early_stop=True).sum().backward()

        self.assertEqual(len(tail_calls), 1)

    def test_preserve_rng_state_matches_regular_autograd(self):
        """Random masks should be restored when preserve_rng_state is enabled."""
        baseline_input = _ones(32, requires_grad=True)
        checkpoint_input = baseline_input.detach().clone().requires_grad_()

        set_seed(7)
        baseline_output = torch.nn.functional.dropout(baseline_input, p=0.5, training=True)
        baseline_output.sum().backward()

        set_seed(7)
        output = checkpoint(
            lambda x: torch.nn.functional.dropout(x, p=0.5, training=True),
            checkpoint_input,
            preserve_rng_state=True,
        )
        output.sum().backward()

        self.assertTrue(torch.equal(output, baseline_output))
        self.assertTrue(torch.equal(checkpoint_input.grad, baseline_input.grad))

    def test_forward_and_recompute_contexts_are_used(self):
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

        input_tensor = _randn(4, requires_grad=True)
        checkpoint(lambda x: x.sin(), input_tensor, context_fn=context_fn).sum().backward()

        self.assertEqual(
            events,
            ["enter:forward", "exit:forward", "enter:recompute", "exit:recompute"],
        )

    def test_grad_inside_forward_matches_regular_autograd(self):
        """Checkpoint should support an unpack triggered by grad inside forward."""
        input_tensor = _randn(4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()

        def function(value):
            intermediate = value.sin()
            inner_grad = torch.autograd.grad(intermediate.sum(), value, create_graph=True)[0]
            return intermediate * inner_grad

        baseline_output = function(baseline_input)
        baseline_output.sum().backward()
        output = checkpoint(function, input_tensor)
        output.sum().backward()

        self.assertTrue(torch.allclose(output, baseline_output))
        self.assertTrue(torch.allclose(input_tensor.grad, baseline_input.grad))

    def test_no_grad_execution_does_not_create_handle(self):
        """A checkpoint outside autograd should behave as a direct function call."""
        with recompute_handle_collector_ctx() as handles:
            with torch.no_grad():
                output = checkpoint(lambda x: x.sin(), _ones(4))

        self.assertTrue(torch.equal(output, _ones(4).sin()))
        self.assertEqual(handles, [])

    def test_failed_forward_does_not_leave_handle(self):
        """A failed checkpoint invocation should be removed from its collector."""
        def function(input_tensor):
            del input_tensor
            raise RuntimeError("forward failed")

        with recompute_handle_collector_ctx() as handles:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                checkpoint(function, _ones(1, requires_grad=True))

        self.assertEqual(handles, [])

    def test_invalid_checkpoint_options_raise(self):
        """Unsupported modes and malformed options should fail at the API boundary."""
        input_tensor = _ones(1, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            checkpoint(lambda x: x, input_tensor, use_reentrant=True)
        with self.assertRaisesRegex(ValueError, "early_stop must be bool"):
            checkpoint(lambda x: x, input_tensor, early_stop=1)
        with self.assertRaisesRegex(ValueError, "preserve_rng_state must be bool"):
            checkpoint(lambda x: x, input_tensor, preserve_rng_state=1)
        with self.assertRaisesRegex(ValueError, "determinism_check"):
            checkpoint(lambda x: x, input_tensor, determinism_check="invalid")
        with self.assertRaisesRegex(ValueError, "debug=True"):
            checkpoint(lambda x: x, input_tensor, debug=True)

    def test_compile_uses_native_checkpoint(self):
        """Compile state should fall back to the public native checkpoint API."""
        with patch.object(checkpoint_module, "_is_compiling", return_value=True), patch.object(
            checkpoint_module, "_native_checkpoint", return_value="compiled-result"
        ) as mock_native_checkpoint:
            result = checkpoint(lambda x: x, 1, early_stop=False)

        self.assertEqual(result, "compiled-result")
        self.assertFalse(mock_native_checkpoint.call_args.kwargs["early_stop"])


class TestScheduledRecomputation(unittest.TestCase):
    """Validate early scheduling and dx/dw separated backward behavior."""

    def test_separate_dx_dw_without_session_recomputes_per_graph_task(self):
        """Native GraphTask keys should avoid unpack loss across dx and dw calls."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        output = checkpoint(function, input_tensor, weight)
        torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)
        torch.autograd.grad(output.sum(), weight)

        self.assertEqual(len(calls), 3)

    def test_prefired_session_is_shared_by_dx_and_dw(self):
        """A retained session should reuse one prefired recomputation for dx and dw."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()
        baseline_weight = weight.detach().clone().requires_grad_()

        baseline = torch.nn.functional.gelu(baseline_input @ baseline_weight)
        expected_dx = torch.autograd.grad(baseline.sum(), baseline_input, retain_graph=True)[0]
        expected_dw = torch.autograd.grad(baseline.sum(), baseline_weight)[0]

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(function, input_tensor, weight)

        session_id = ("micro-batch", 0)
        try:
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                recompute_handle(handles[0], session_id)
                actual_dx = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
                actual_dw = torch.autograd.grad(output.sum(), weight)[0]
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(handles), 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.allclose(actual_dx, expected_dx))
        self.assertTrue(torch.allclose(actual_dw, expected_dw))

    def test_one_session_serves_multiple_checkpoint_frames(self):
        """One session should prefire and serve multiple sequential checkpoint frames."""
        input_tensor = _randn(4, 4, requires_grad=True)
        first_weight = _randn(4, 6, requires_grad=True)
        second_weight = _randn(6, 3, requires_grad=True)
        reference_input = input_tensor.detach().clone().requires_grad_()
        reference_first_weight = first_weight.detach().clone().requires_grad_()
        reference_second_weight = second_weight.detach().clone().requires_grad_()
        reference_hidden = torch.nn.functional.gelu(reference_input @ reference_first_weight)
        reference_output = torch.nn.functional.silu(reference_hidden @ reference_second_weight)
        expected_grads = torch.autograd.grad(
            reference_output.sum(),
            (reference_input, reference_first_weight, reference_second_weight),
        )
        calls = [0, 0]

        def first_block(current_input, current_weight):
            calls[0] += 1
            return torch.nn.functional.gelu(current_input @ current_weight)

        def second_block(current_input, current_weight):
            calls[1] += 1
            return torch.nn.functional.silu(current_input @ current_weight)

        with recompute_handle_collector_ctx() as handles:
            hidden = checkpoint(first_block, input_tensor, first_weight)
            output = checkpoint(second_block, hidden, second_weight)

        session_id = ("multiple-frames", id(output))
        try:
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                for handle in handles:
                    recompute_handle(handle, session_id)
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                actual_dx = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
            with recompute_session_ctx(session_id, retain_on_unpack=False):
                actual_dws = torch.autograd.grad(output.sum(), (first_weight, second_weight))
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(handles), 2)
        self.assertEqual(calls, [2, 2])
        self.assertTrue(torch.allclose(actual_dx, expected_grads[0]))
        self.assertTrue(torch.allclose(actual_dws[0], expected_grads[1]))
        self.assertTrue(torch.allclose(actual_dws[1], expected_grads[2]))

    def test_repeated_sessions_keep_iterations_isolated(self):
        """Repeated iterations should not reuse frames or tensors from earlier sessions."""
        weight = _randn(4, 4, requires_grad=True)
        calls = []

        def function(current_input, current_weight):
            calls.append(None)
            return torch.nn.functional.gelu(current_input @ current_weight)

        for step in range(3):
            input_tensor = _randn(4, 4, requires_grad=True)
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

    def test_partial_and_failed_sessions_can_be_cleared_and_reused(self):
        """Early-stop, partial consumption, and failed prefire should leave reusable frames."""
        for early_stop, expected_tail_calls in ((True, 1), (False, 3)):
            with self.subTest(early_stop=early_stop):
                input_tensor = _randn(4, 4, requires_grad=True)
                weight = _randn(4, 4, requires_grad=True)
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

        input_tensor = _randn(4, requires_grad=True)
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

    def test_prefired_session_survives_contextvar_loss(self):
        """A worker without the caller ContextVar should still find prefired tensors."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(function, input_tensor, weight)

        session_id = "worker-context-loss"
        try:
            recompute_handle(handles[0], session_id)
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                token = checkpoint_module._RECOMPUTE_SESSION.set(None)
                try:
                    torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)
                finally:
                    checkpoint_module._RECOMPUTE_SESSION.reset(token)
            with recompute_session_ctx(session_id, retain_on_unpack=False):
                token = checkpoint_module._RECOMPUTE_SESSION.set(None)
                try:
                    torch.autograd.grad(output.sum(), weight)
                finally:
                    checkpoint_module._RECOMPUTE_SESSION.reset(token)
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(calls), 2)

    def test_scheduled_nested_checkpoint_raises(self):
        """Scheduled recomputation should reject nested checkpoint regions explicitly."""
        input_tensor = _randn(4, requires_grad=True)

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

    def test_nested_checkpoint_without_session_still_works(self):
        """Ordinary GraphTask-based recomputation should preserve native nested behavior."""
        input_tensor = _randn(4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()

        def inner(value):
            return value.sin()

        def outer(value):
            return checkpoint(inner, value).cos()

        expected = baseline_input.sin().cos()
        expected_grad = torch.autograd.grad(expected.sum(), baseline_input)[0]
        output = checkpoint(outer, input_tensor)
        actual_grad = torch.autograd.grad(output.sum(), input_tensor)[0]

        self.assertTrue(torch.allclose(output, expected))
        self.assertTrue(torch.allclose(actual_grad, expected_grad))

    def test_prefired_session_consumes_sac_cache_only_once(self):
        """SAC replay should happen during prefire, not again for dx and dw consumers."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        def policy_fn(ctx, op, *args, **kwargs):
            del ctx, op, args, kwargs
            return CheckpointPolicy.MUST_SAVE

        with recompute_handle_collector_ctx() as handles:
            output = core_checkpoint(function, input_tensor, weight, policy_fn=policy_fn)

        session_id = "sac-split-backward"
        try:
            recompute_handle(handles[0], session_id)
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)
                torch.autograd.grad(output.sum(), weight)
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(calls), 2)

    def test_clear_session_is_idempotent(self):
        """A retained session can be cleared repeatedly in cleanup paths."""
        input_tensor = _randn(4, requires_grad=True)
        with recompute_handle_collector_ctx() as handles:
            checkpoint(lambda x: x.sin(), input_tensor)

        session_id = "clear-twice"
        recompute_handle(handles[0], session_id)
        clear_recompute_session(session_id)
        clear_recompute_session(session_id)

    def test_invalid_handle_and_session_options_raise(self):
        """Scheduling APIs should reject handles and session options from outside Hyper."""
        with self.assertRaisesRegex(ValueError, "handle must be produced"):
            recompute_handle(object(), "session")
        with self.assertRaisesRegex(ValueError, "session_id must be hashable"):
            with recompute_session_ctx([], retain_on_unpack=False):
                pass
        with self.assertRaisesRegex(ValueError, "retain_on_unpack must be bool"):
            with recompute_session_ctx("session", retain_on_unpack=1):
                pass

    def test_session_id_is_required_and_must_not_be_none(self):
        """Session contexts should require callers to propagate one explicit stable id."""
        session_parameter = inspect.signature(recompute_session_ctx).parameters["session_id"]
        self.assertIs(session_parameter.default, inspect.Parameter.empty)
        with self.assertRaisesRegex(ValueError, "session_id must not be None"):
            with recompute_session_ctx(None):
                pass

    def test_unpack_outside_backward_uses_temporary_graph_key(self):
        """An eager saved-tensor access outside backward should recompute successfully."""
        input_tensor = _randn(4, requires_grad=True)
        output = checkpoint(lambda value: value.sin(), input_tensor)

        saved_input = output.grad_fn._saved_self  # pylint: disable=W0212

        self.assertTrue(torch.equal(saved_input, input_tensor))

    def test_default_device_type_is_used_without_device_tensor_arguments(self):
        """Device-less arguments should honor Torch's stable checkpoint default."""
        with patch.object(checkpoint_module.DefaultDeviceType, "get_device_type", return_value="npu"):
            device_type = checkpoint_module._infer_device_type(torch.ones(1, device="cpu"))

        self.assertEqual(device_type, "npu")

    def test_incomplete_forward_ignores_extra_save_only_without_early_stop(self):
        """Only a full recomputation may run ahead of an incomplete original forward."""
        input_tensor = _randn(4, requires_grad=True)
        early_stop_frame = checkpoint_module._CheckpointFrame(lambda: None, True, None)
        with self.assertRaises(CheckpointError):
            with checkpoint_module._create_recomputation_hooks(early_stop_frame, "early-stop"):
                input_tensor.sin()

        full_recompute_frame = checkpoint_module._CheckpointFrame(lambda: None, False, None)
        with checkpoint_module._create_recomputation_hooks(full_recompute_frame, "full-recompute"):
            input_tensor.sin()
        self.assertTrue(full_recompute_frame.ignore_saved_mismatch)

    def test_recompute_metadata_mismatch_raises(self):
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

        input_tensor = _randn(4, requires_grad=True)
        output = checkpoint(
            function,
            input_tensor,
            context_fn=lambda: (contextlib.nullcontext(), recompute_context()),
        )

        with self.assertRaises(CheckpointError):
            output.sum().backward()

    def test_none_determinism_check_allows_metadata_change(self):
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

        input_tensor = _randn(4, requires_grad=True)
        output = checkpoint(
            function,
            input_tensor,
            context_fn=lambda: (contextlib.nullcontext(), recompute_context()),
            determinism_check="none",
        )

        output.sum().backward()


def run_checkpoint_cases() -> None:
    """Run all checkpoint cases on NPU."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        (
            loader.loadTestsFromTestCase(TestCheckpoint),
            loader.loadTestsFromTestCase(TestScheduledRecomputation),
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(
            f"Checkpoint NPU suite failed with {len(result.failures)} failures and {len(result.errors)} errors."
        )


if __name__ == "__main__":
    run_checkpoint_cases()

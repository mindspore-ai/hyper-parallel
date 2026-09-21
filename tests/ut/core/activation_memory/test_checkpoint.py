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
import os
import unittest
from unittest.mock import MagicMock, call, patch, sentinel

import torch


from hyper_parallel.core.activation_memory.checkpoint import (
    CheckpointError,
    checkpoint,
    clear_recompute_session,
    recompute_handle,
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
        inputs = {"tensor": torch.ones(2), "nested": [torch.zeros(1)]}

        with patch.object(_checkpoint.DefaultDeviceType, "get_device_type", return_value="xla"):
            device_type = _checkpoint._infer_device_type(inputs)

        self.assertEqual(device_type, "xla")

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
        with self.assertRaisesRegex(ValueError, "session_id must not be None"):
            clear_recompute_session(None)
        with self.assertRaisesRegex(ValueError, "session_id must be hashable"):
            with recompute_session_ctx([]):
                self.fail("An unhashable session should not be entered.")
        with self.assertRaisesRegex(ValueError, "retain_on_unpack must be bool"):
            with recompute_session_ctx("session", retain_on_unpack=1):
                self.fail("An invalid retention policy should not be entered.")

if __name__ == "__main__":
    unittest.main()

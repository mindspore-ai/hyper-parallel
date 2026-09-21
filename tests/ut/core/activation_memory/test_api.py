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
"""Unit tests for ``hyper_parallel.core.activation_memory.api``."""
# The backend selector must be set before importing platform aliases.  The
# local imports and patched platform fixture are intentional test setup.
# pylint: disable=wrong-import-position,import-outside-toplevel,unused-argument
import contextlib
import importlib
import os
import unittest
from unittest.mock import MagicMock, patch, sentinel

import torch


from hyper_parallel.core.activation_memory.api import (
    CheckpointPolicy,
    async_save_on_cpu,
    clear_recompute_session,
    checkpoint,
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
    create_native_selective_checkpoint_contexts,
    create_selective_checkpoint_contexts,
    get_class_activation_wrapper,
    ignore_sac_ops,
    is_compiling,
    noop_context_fn,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
    swap,
    swap_tensor_wrapper,
    swap_wrapper,
)
from hyper_parallel.core.activation_memory.recompute_state import (
    create_recompute_contexts,
    get_recompute_state,
    is_recomputing,
)
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper

_facade = importlib.import_module("hyper_parallel.core.activation_memory.api")
_public_api = importlib.import_module("hyper_parallel.core.activation_memory")
_checkpoint_impl = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint")
_compile_impl = importlib.import_module("hyper_parallel.core.activation_memory.compile_adapter")
_sac_impl = importlib.import_module("hyper_parallel.core.activation_memory.sac")
_wrapper_impl = importlib.import_module("hyper_parallel.core.activation_memory.wrapper")


def _spy_checkpoint(result="result"):
    """Return a context manager that records calls to the checkpoint implementation.

    Args:
        result: Sentinel returned in place of the real checkpoint result.

    Returns:
        A context manager yielding the mocked checkpoint implementation.
    """
    return patch.object(_checkpoint_impl, "checkpoint", return_value=result)


class TestApiExports(unittest.TestCase):
    """Unit tests for the activation-checkpoint API export contract."""

    def test_all_contains_public_api(self):
        """Test ``__all__`` exposes every supported API symbol."""
        self.assertEqual(
            _facade.__all__,
            [
                "CheckpointPolicy",
                "checkpoint",
                "swap",
                "checkpoint_wrapper",
                "checkpoint_exclude_wrapper",
                "swap_wrapper",
                "swap_tensor_wrapper",
                "recompute_handle_collector_ctx",
                "recompute_handle",
                "recompute_session_ctx",
                "clear_recompute_session",
                "ignore_sac_ops",
                "create_selective_checkpoint_contexts",
                "create_native_selective_checkpoint_contexts",
                "async_save_on_cpu",
                "get_class_activation_wrapper",
                "noop_context_fn",
                "is_compiling",
            ],
        )


class TestContextComposition(unittest.TestCase):
    """Unit tests for context-factory composition used by ``checkpoint``."""

    @staticmethod
    def _recording_context(events, name, fail_on_enter=False):
        """Return a context manager that records entry and exit events."""
        @contextlib.contextmanager
        def _context():
            events.append(f"enter:{name}")
            try:
                if fail_on_enter:
                    raise RuntimeError(f"failed:{name}")
                yield
            finally:
                events.append(f"exit:{name}")

        return _context()

    def test_composed_contexts_enter_in_order_and_exit_in_reverse(self):
        """Test forward and recompute contexts preserve stack ordering."""
        events = []

        def _first_factory():
            return (
                self._recording_context(events, "first_forward"),
                self._recording_context(events, "first_recompute"),
            )

        def _second_factory():
            return (
                self._recording_context(events, "second_forward"),
                self._recording_context(events, "second_recompute"),
            )

        composed = _facade._compose_context_fns((_first_factory, _second_factory))  # pylint: disable=W0212
        forward_context, recompute_context = composed()

        with forward_context:
            events.append("forward")
        with recompute_context:
            events.append("recompute")

        self.assertEqual(
            events,
            [
                "enter:first_forward",
                "enter:second_forward",
                "forward",
                "exit:second_forward",
                "exit:first_forward",
                "enter:first_recompute",
                "enter:second_recompute",
                "recompute",
                "exit:second_recompute",
                "exit:first_recompute",
            ],
        )

    def test_failed_context_entry_unwinds_entered_contexts(self):
        """Test a failed nested entry closes both the failing and entered contexts."""
        events = []
        first_factory = MagicMock(
            return_value=(
                self._recording_context(events, "first"),
                contextlib.nullcontext(),
            )
        )
        second_factory = MagicMock(
            return_value=(
                self._recording_context(events, "second", fail_on_enter=True),
                contextlib.nullcontext(),
            )
        )
        composed = _facade._compose_context_fns((first_factory, second_factory))  # pylint: disable=W0212
        forward_context, _ = composed()

        with self.assertRaisesRegex(RuntimeError, "failed:second"):
            with forward_context:
                self.fail("The composed context should fail during entry.")

        self.assertEqual(events, ["enter:first", "enter:second", "exit:second", "exit:first"])


class TestLazyApiDelegates(unittest.TestCase):
    """Unit tests for API functions that lazily delegate to implementation modules."""

    def test_is_compiling_delegates_to_torch_compiler(self):
        """Test compile-state lookup returns the Torch compiler result."""
        with patch.object(torch.compiler, "is_compiling", return_value=sentinel.compiling) as mock_is_compiling:
            result = is_compiling()

        self.assertIs(result, sentinel.compiling)
        mock_is_compiling.assert_called_once_with()

    def test_noop_context_fn_returns_torch_factory(self):
        """Test no-op context lookup returns the Torch-native factory."""
        with patch("torch.utils.checkpoint.noop_context_fn", sentinel.noop_context_fn):
            result = noop_context_fn()

        self.assertIs(result, sentinel.noop_context_fn)

    def test_checkpoint_exclude_wrapper_forwards_arguments(self):
        """Test checkpoint-exclusion wrapper forwards its output-retention flag."""
        with patch.object(
            _wrapper_impl,
            "checkpoint_exclude_wrapper",
            return_value=sentinel.wrapper,
        ) as mock_wrapper:
            result = checkpoint_exclude_wrapper(sentinel.module, save_output=False)

        self.assertIs(result, sentinel.wrapper)
        mock_wrapper.assert_called_once_with(sentinel.module, save_output=False)

    def test_checkpoint_wrapper_forwards_keyword_arguments(self):
        """Test checkpoint wrapper delegates all checkpoint options."""
        with patch.object(_wrapper_impl, "ckpt_wrapper", return_value=sentinel.wrapper) as mock_wrapper:
            result = checkpoint_wrapper(sentinel.module, policy_fn=sentinel.policy, group_swap=True)

        self.assertIs(result, sentinel.wrapper)
        mock_wrapper.assert_called_once_with(sentinel.module, policy_fn=sentinel.policy, group_swap=True)

    def test_swap_wrapper_omits_unset_cpu_pool(self):
        """Test swap wrapper does not pass an unset backend-specific pool."""
        with patch.object(_wrapper_impl, "swap_wrapper", return_value=sentinel.wrapper) as mock_wrapper:
            result = swap_wrapper(sentinel.module, policy_fn=sentinel.policy, group_swap=True)

        self.assertIs(result, sentinel.wrapper)
        mock_wrapper.assert_called_once_with(sentinel.module, policy_fn=sentinel.policy, group_swap=True)

    def test_swap_wrapper_forwards_cpu_pool(self):
        """Test swap wrapper forwards an explicitly configured CPU pool."""
        with patch.object(_wrapper_impl, "swap_wrapper", return_value=sentinel.wrapper) as mock_wrapper:
            result = swap_wrapper(sentinel.module, cpu_pool=sentinel.pool)

        self.assertIs(result, sentinel.wrapper)
        mock_wrapper.assert_called_once_with(
            sentinel.module,
            policy_fn=None,
            group_swap=False,
            cpu_pool=sentinel.pool,
        )

    def test_swap_tensor_wrapper_omits_unset_cpu_pool(self):
        """Test tensor wrapper does not pass an unset backend-specific pool."""
        with patch.object(
            _wrapper_impl,
            "swap_tensor_wrapper",
            return_value=sentinel.wrapped_tensor,
        ) as mock_wrapper:
            result = swap_tensor_wrapper(sentinel.tensor, tag="hidden", group_swap=True)

        self.assertIs(result, sentinel.wrapped_tensor)
        mock_wrapper.assert_called_once_with(sentinel.tensor, tag="hidden", group_swap=True)

    def test_swap_tensor_wrapper_forwards_cpu_pool(self):
        """Test tensor wrapper forwards an explicitly configured CPU pool."""
        with patch.object(
            _wrapper_impl,
            "swap_tensor_wrapper",
            return_value=sentinel.wrapped_tensor,
        ) as mock_wrapper:
            result = swap_tensor_wrapper(sentinel.tensor, cpu_pool=sentinel.pool)

        self.assertIs(result, sentinel.wrapped_tensor)
        mock_wrapper.assert_called_once_with(
            sentinel.tensor,
            tag=None,
            group_swap=False,
            cpu_pool=sentinel.pool,
        )

    def test_get_class_activation_wrapper_returns_backend_class(self):
        """Test activation-wrapper class lookup uses the wrapper implementation."""
        with patch.object(_wrapper_impl, "ActivationWrapper", sentinel.activation_wrapper):
            result = get_class_activation_wrapper()

        self.assertIs(result, sentinel.activation_wrapper)

    def test_ignore_sac_ops_forwards_operations(self):
        """Test ignored selective-checkpoint operations reach the SAC implementation."""
        ignored_ops = [sentinel.first_op, sentinel.second_op]
        with patch.object(_sac_impl, "ignore_sac_ops", return_value=sentinel.unused) as mock_ignore:
            self.assertIsNone(ignore_sac_ops(ignored_ops))

        mock_ignore.assert_called_once_with(ignored_ops)

    def test_create_selective_checkpoint_contexts_forwards_options(self):
        """Test selective-context creation forwards policy and cache/swap options."""
        with patch.object(
            _sac_impl,
            "create_selective_checkpoint_contexts",
            return_value=sentinel.contexts,
        ) as mock_create:
            result = create_selective_checkpoint_contexts(
                sentinel.policy,
                allow_cache_entry_mutation=True,
                group_swap=True,
                cpu_pool=sentinel.pool,
            )

        self.assertIs(result, sentinel.contexts)
        mock_create.assert_called_once_with(
            sentinel.policy,
            allow_cache_entry_mutation=True,
            group_swap=True,
            cpu_pool=sentinel.pool,
        )

    def test_create_native_selective_checkpoint_contexts_forwards_policy(self):
        """Test native selective-context creation forwards its policy."""
        with patch.object(
            _compile_impl,
            "create_native_selective_checkpoint_contexts",
            return_value=sentinel.contexts,
        ) as mock_create:
            result = create_native_selective_checkpoint_contexts(sentinel.policy)

        self.assertIs(result, sentinel.contexts)
        mock_create.assert_called_once_with(sentinel.policy)

    def test_async_save_on_cpu_constructs_context_with_all_options(self):
        """Test async host-save context construction forwards every option."""
        with patch.object(_wrapper_impl, "AsyncSaveOnCpu", return_value=sentinel.context) as mock_context:
            result = async_save_on_cpu(
                policy_fn=sentinel.policy,
                group_swap=True,
                cpu_pool=sentinel.pool,
            )

        self.assertIs(result, sentinel.context)
        mock_context.assert_called_once_with(
            policy_fn=sentinel.policy,
            group_swap=True,
            cpu_pool=sentinel.pool,
        )

    def test_recompute_handle_collector_ctx_returns_implementation_context(self):
        """Test recompute-handle collection delegates to the checkpoint implementation."""
        with patch.object(
            _checkpoint_impl,
            "recompute_handle_collector_ctx",
            return_value=sentinel.context,
        ) as mock_collector:
            result = recompute_handle_collector_ctx()

        self.assertIs(result, sentinel.context)
        mock_collector.assert_called_once_with()

    def test_recompute_handle_forwards_handle_and_session(self):
        """Test recompute dispatch forwards the handle and session ID."""
        with patch.object(_checkpoint_impl, "recompute_handle", return_value=sentinel.result) as mock_recompute:
            result = recompute_handle(sentinel.handle, sentinel.session_id)

        self.assertIs(result, sentinel.result)
        mock_recompute.assert_called_once_with(sentinel.handle, sentinel.session_id)

    def test_recompute_session_ctx_forwards_options(self):
        """Test recompute-session context forwards retention configuration."""
        with patch.object(_checkpoint_impl, "recompute_session_ctx", return_value=sentinel.context) as mock_session:
            result = recompute_session_ctx(sentinel.session_id, retain_on_unpack=True)

        self.assertIs(result, sentinel.context)
        mock_session.assert_called_once_with(session_id=sentinel.session_id, retain_on_unpack=True)

    def test_recompute_session_ctx_rejects_none_session_id(self):
        """Test a missing recompute-session ID is rejected before delegation."""
        with patch.object(_checkpoint_impl, "recompute_session_ctx") as mock_session:
            with self.assertRaisesRegex(ValueError, "session_id must not be None"):
                recompute_session_ctx(None)

        mock_session.assert_not_called()

    def test_clear_recompute_session_forwards_session_id(self):
        """Test recompute-session cleanup delegates and returns its result."""
        with patch.object(
            _checkpoint_impl,
            "clear_recompute_session",
            return_value=sentinel.result,
        ) as mock_clear:
            result = clear_recompute_session(sentinel.session_id)

        self.assertIs(result, sentinel.result)
        mock_clear.assert_called_once_with(sentinel.session_id)


class TestCheckpointPolicy(unittest.TestCase):
    """Unit tests for CheckpointPolicy enum."""

    def test_enum_values(self):
        """Test all enum values are correct."""
        self.assertEqual(CheckpointPolicy.MUST_SAVE.value, 0)
        self.assertEqual(CheckpointPolicy.PREFER_SAVE.value, 1)
        self.assertEqual(CheckpointPolicy.MUST_RECOMPUTE.value, 2)
        self.assertEqual(CheckpointPolicy.PREFER_RECOMPUTE.value, 3)
        self.assertEqual(CheckpointPolicy.MUST_SWAP.value, 4)

    def test_enum_membership(self):
        """Test enum member type checking."""
        self.assertIsInstance(CheckpointPolicy.MUST_SAVE, CheckpointPolicy)
        self.assertIsInstance(CheckpointPolicy.MUST_SWAP, CheckpointPolicy)

    def test_enum_str(self):
        """Test string representation of enum members."""
        self.assertEqual(str(CheckpointPolicy.MUST_SWAP), "CheckpointPolicy.MUST_SWAP")


class TestCheckpointFunction(unittest.TestCase):
    """Unit tests for checkpoint() function."""

    def test_checkpoint_no_swap_no_policy(self):
        """Test checkpoint without swap_inputs and without policy_fn."""
        def _dummy_fn(x):
            return x * 2

        with _spy_checkpoint("checkpoint_result") as spy:
            result = checkpoint(_dummy_fn, 3)

        self.assertEqual(result, "checkpoint_result")
        spy.assert_called_once()

        call_args = spy.call_args.args
        call_kwargs = spy.call_args.kwargs
        self.assertIs(call_args[0], _dummy_fn)
        self.assertEqual(call_args[1], 3)
        self.assertEqual(call_kwargs.get("use_reentrant"), False)
        context_fn = call_kwargs.get("context_fn")
        self.assertTrue(callable(context_fn))
        forward_context, recompute_context = context_fn()
        self.assertFalse(is_recomputing())
        with forward_context:
            self.assertFalse(is_recomputing())
        with recompute_context:
            self.assertTrue(is_recomputing())
        self.assertFalse(is_recomputing())

    def test_checkpoint_with_swap_inputs(self):
        """Test checkpoint with swap_inputs=True enters the async host-offload context."""
        def _dummy_fn(x):
            return x * 2

        with _spy_checkpoint(), patch(
            "hyper_parallel.core.activation_memory.api.async_save_on_cpu",
            return_value=contextlib.nullcontext(),
        ) as mock_async:
            result = checkpoint(_dummy_fn, 3, swap_inputs=True)

        self.assertEqual(result, "result")
        mock_async.assert_called_once_with(group_swap=False)

    def test_checkpoint_with_policy_fn_forwards_selective_options(self):
        """Test policy context creation receives the default selective options."""
        policy = lambda context, op, *a, **kw: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001
        selective_contexts = (contextlib.nullcontext(), contextlib.nullcontext())
        with _spy_checkpoint() as spy, patch.object(
            _facade,
            "create_selective_checkpoint_contexts",
            return_value=selective_contexts,
        ) as mock_create:
            result = checkpoint(sentinel.function, 3, policy_fn=policy)
            forward_context, recompute_context = spy.call_args.kwargs["context_fn"]()

        self.assertEqual(result, "result")
        self.assertIsInstance(forward_context, _facade._StackedCtx)  # pylint: disable=W0212
        self.assertIsInstance(recompute_context, _facade._StackedCtx)  # pylint: disable=W0212
        mock_create.assert_called_once_with(policy, group_swap=False)

    def test_checkpoint_without_policy_does_not_create_selective_context(self):
        """Test group swap alone keeps the default recompute-state context."""
        with _spy_checkpoint() as spy, patch.object(
            _facade,
            "create_selective_checkpoint_contexts",
        ) as mock_create:
            result = checkpoint(sentinel.function, 3, group_swap=True)

        self.assertEqual(result, "result")
        self.assertIs(spy.call_args.kwargs["context_fn"], create_recompute_contexts)
        mock_create.assert_not_called()

    def test_checkpoint_policy_context_forwards_group_swap_and_cpu_pool(self):
        """Test selective context receives group-swap and explicit pool options once."""
        policy = lambda context, op, *a, **kw: CheckpointPolicy.MUST_SWAP  # pylint: disable=C3001
        selective_contexts = (contextlib.nullcontext(), contextlib.nullcontext())
        with _spy_checkpoint() as spy, patch.object(
            _facade,
            "create_selective_checkpoint_contexts",
            return_value=selective_contexts,
        ) as mock_create:
            result = checkpoint(
                sentinel.function,
                3,
                policy_fn=policy,
                group_swap=True,
                cpu_pool=sentinel.pool,
            )
            forward_context, recompute_context = spy.call_args.kwargs["context_fn"]()

        self.assertEqual(result, "result")
        self.assertIsInstance(forward_context, _facade._StackedCtx)  # pylint: disable=W0212
        self.assertIsInstance(recompute_context, _facade._StackedCtx)  # pylint: disable=W0212
        mock_create.assert_called_once_with(
            policy,
            group_swap=True,
            cpu_pool=sentinel.pool,
        )

    def test_checkpoint_with_kwargs(self):
        """Test checkpoint passes kwargs to underlying function."""
        def _dummy_fn(x, scale=1.0):
            return x * scale

        with _spy_checkpoint() as spy:
            result = checkpoint(_dummy_fn, 3, scale=2.0)

        self.assertEqual(result, "result")
        call_args = spy.call_args.args
        call_kwargs = spy.call_args.kwargs
        self.assertEqual(call_args, (_dummy_fn, 3))
        self.assertEqual(call_kwargs["scale"], 2.0)

    def test_checkpoint_forwards_early_stop_keyword(self):
        """Test checkpoint forwards the explicit early_stop control keyword."""
        with _spy_checkpoint() as spy:
            result = checkpoint(lambda value: value, 3, **{"early_stop": False})

        self.assertEqual(result, "result")
        self.assertFalse(spy.call_args.kwargs["early_stop"])

    def test_checkpoint_forces_non_reentrant_implementation(self):
        """Test callers cannot override the non-reentrant implementation mode."""
        with _spy_checkpoint() as spy:
            checkpoint(sentinel.function, use_reentrant=True)

        self.assertFalse(spy.call_args.kwargs["use_reentrant"])

    def test_checkpoint_rejects_non_boolean_early_stop(self):
        """Test checkpoint rejects ambiguous early_stop values."""
        with _spy_checkpoint() as spy:
            with self.assertRaisesRegex(ValueError, "early_stop must be bool"):
                checkpoint(lambda value: value, 3, early_stop=1)
            spy.assert_not_called()

    def test_checkpoint_composes_recompute_state_and_user_contexts(self):
        """Unified recompute state should surround user checkpoint contexts."""
        events = []

        @contextlib.contextmanager
        def _record(name):
            events.append(f"enter:{name}")
            try:
                yield
            finally:
                events.append(f"exit:{name}")

        def _user_context_fn():
            return _record("user_fwd"), _record("user_rec")

        with _spy_checkpoint() as spy:
            checkpoint(lambda value: value, 1, context_fn=_user_context_fn)
        composed_context_fn = spy.call_args.kwargs["context_fn"]
        forward_context, recompute_context = composed_context_fn()

        with forward_context:
            events.append(f"forward:{is_recomputing()}")
        with recompute_context:
            events.append(f"recompute:{is_recomputing()}")

        self.assertEqual(
            events,
            [
                "enter:user_fwd",
                "forward:False",
                "exit:user_fwd",
                "enter:user_rec",
                "recompute:True",
                "exit:user_rec",
            ],
        )

    def test_checkpoint_swap_inputs_forwards_group_swap_and_cpu_pool(self):
        """Test input swapping enters its context with group and pool settings."""
        events = []

        @contextlib.contextmanager
        def _swap_context():
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        with _spy_checkpoint(sentinel.result) as spy, patch.object(
            _facade,
            "async_save_on_cpu",
            return_value=_swap_context(),
        ) as mock_async:
            result = checkpoint(
                sentinel.function,
                swap_inputs=True,
                group_swap=True,
                cpu_pool=sentinel.pool,
            )

        self.assertIs(result, sentinel.result)
        self.assertEqual(events, ["enter", "exit"])
        mock_async.assert_called_once_with(group_swap=True, cpu_pool=sentinel.pool)

    def test_checkpoint_compile_mode_without_policy_omits_context_fn(self):
        """Test compile mode uses plain non-reentrant checkpointing by default."""
        with patch.object(_facade, "is_compiling", return_value=True), _spy_checkpoint() as spy:
            result = checkpoint(sentinel.function, sentinel.argument, keyword=sentinel.keyword)

        self.assertEqual(result, "result")
        self.assertNotIn("context_fn", spy.call_args.kwargs)
        self.assertFalse(spy.call_args.kwargs["use_reentrant"])
        self.assertTrue(spy.call_args.kwargs["early_stop"])
        self.assertIs(spy.call_args.kwargs["keyword"], sentinel.keyword)

    def test_checkpoint_compile_mode_uses_native_selective_context(self):
        """Test compile mode converts a policy into the native context factory."""
        with patch.object(_facade, "is_compiling", return_value=True), patch.object(
            _facade,
            "create_native_selective_checkpoint_contexts",
            return_value=sentinel.contexts,
        ) as mock_create, _spy_checkpoint() as spy:
            result = checkpoint(sentinel.function, policy_fn=sentinel.policy)
            contexts = spy.call_args.kwargs["context_fn"]()

        self.assertEqual(result, "result")
        self.assertIs(contexts, sentinel.contexts)
        mock_create.assert_called_once_with(sentinel.policy)

    def test_checkpoint_compile_mode_rejects_unsupported_options(self):
        """Test compile mode reports each unsupported HyperParallel option."""
        cases = [
            ({"swap_inputs": True}, "swap_inputs"),
            ({"group_swap": True}, "group_swap"),
            ({"cpu_pool": sentinel.pool}, "cpu_pool"),
            ({"context_fn": sentinel.context_factory}, "custom context_fn"),
            ({"use_reentrant": True}, "use_reentrant=True"),
        ]

        with patch.object(_facade, "is_compiling", return_value=True), _spy_checkpoint() as spy:
            for options, expected in cases:
                with self.subTest(option=expected):
                    with self.assertRaises(ValueError) as error:
                        checkpoint(sentinel.function, **options)
                    self.assertIn(expected, str(error.exception))

        spy.assert_not_called()


class TestRecomputeState(unittest.TestCase):
    """Unit tests for invocation-scoped recompute execution state."""

    def test_contexts_share_invocation_and_switch_phase(self):
        """Forward and recompute should expose one invocation with different phases."""
        forward_context, recompute_context = create_recompute_contexts()

        with forward_context:
            forward_state = get_recompute_state()
            self.assertIsNotNone(forward_state)
            self.assertFalse(is_recomputing())
            invocation_id = forward_state.invocation_id

        with recompute_context:
            recompute_state = get_recompute_state()
            self.assertIsNotNone(recompute_state)
            self.assertTrue(is_recomputing())
            self.assertEqual(recompute_state.invocation_id, invocation_id)

        self.assertIsNone(get_recompute_state())

    def test_recompute_exit_clears_unconsumed_resources(self):
        """Early recompute exit should clear resources even when wrappers are not revisited."""
        forward_context, recompute_context = create_recompute_contexts()
        resource = MagicMock()

        with forward_context:
            get_recompute_state().get_resource("saved_output", lambda: resource)
        with recompute_context:
            pass

        resource.clear.assert_called_once_with()


class TestSwapFunction(unittest.TestCase):
    """Unit tests for swap() function."""

    def test_swap_no_policy(self):
        """Test swap passes through function result without policy."""
        def _dummy_fn(x):
            return x * 2

        with patch(
            "hyper_parallel.core.activation_memory.api.async_save_on_cpu",
            return_value=contextlib.nullcontext(),
        ) as mock_async:
            result = swap(_dummy_fn, 3)

        self.assertEqual(result, 6)
        mock_async.assert_called_once_with(policy_fn=None, group_swap=False)

    def test_swap_with_policy_fn(self):
        """Test swap passes policy_fn to async_save_on_cpu."""
        policy = lambda t: CheckpointPolicy.MUST_SAVE  # pylint: disable=C3001

        def _dummy_fn(x):
            return x * 2

        with patch(
            "hyper_parallel.core.activation_memory.api.async_save_on_cpu",
            return_value=contextlib.nullcontext(),
        ) as mock_async:
            result = swap(_dummy_fn, 3, policy_fn=policy)

        self.assertEqual(result, 6)
        mock_async.assert_called_once_with(policy_fn=policy, group_swap=False)

    def test_swap_with_kwargs(self):
        """Test swap forwards kwargs to the function."""
        def _dummy_fn(x, scale=1.0):
            return x * scale

        with patch(
            "hyper_parallel.core.activation_memory.api.async_save_on_cpu",
            return_value=contextlib.nullcontext(),
        ) as mock_async:
            result = swap(_dummy_fn, 3, scale=2.0)

        self.assertEqual(result, 6.0)
        mock_async.assert_called_once_with(policy_fn=None, group_swap=False)

    def test_swap_with_args_and_kwargs(self):
        """Test swap with args and kwargs."""
        def _dummy_fn(a, b, c=1):
            return a + b + c

        with patch(
            "hyper_parallel.core.activation_memory.api.async_save_on_cpu",
            return_value=contextlib.nullcontext(),
        ) as mock_async:
            result = swap(_dummy_fn, 1, 2, c=3)

        self.assertEqual(result, 6)
        mock_async.assert_called_once_with(policy_fn=None, group_swap=False)

    def test_swap_forwards_group_swap_and_cpu_pool(self):
        """Test swap forwards group and pool settings to its host-save context."""
        function = MagicMock(return_value=sentinel.result)
        with patch.object(
            _facade,
            "async_save_on_cpu",
            return_value=contextlib.nullcontext(),
        ) as mock_async:
            result = swap(
                function,
                sentinel.argument,
                policy_fn=sentinel.policy,
                group_swap=True,
                cpu_pool=sentinel.pool,
                keyword=sentinel.keyword,
            )

        self.assertIs(result, sentinel.result)
        mock_async.assert_called_once_with(
            policy_fn=sentinel.policy,
            group_swap=True,
            cpu_pool=sentinel.pool,
        )
        function.assert_called_once_with(sentinel.argument, keyword=sentinel.keyword)

    def test_swap_compile_mode_raises_before_calling_function(self):
        """Test activation swap is rejected without executing work during compile."""
        function = MagicMock()
        with patch.object(_facade, "is_compiling", return_value=True), patch.object(
            _facade,
            "async_save_on_cpu",
        ) as mock_async:
            with self.assertRaisesRegex(ValueError, "not supported in compile mode"):
                swap(function, sentinel.argument)

        function.assert_not_called()
        mock_async.assert_not_called()


class _BaseWrapperModule(torch.nn.Module):
    """Minimal torch module used in checkpoint_wrapper alias tests."""

    def __init__(self, factor: int = 2) -> None:
        """Initialize the wrapped module with a result scale factor."""
        super().__init__()
        self.factor = factor
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer and scale its result."""
        return self.linear(x) * self.factor


class TestCkptWrapper(unittest.TestCase):
    """Unit tests for checkpoint_wrapper() factory function."""

    def test_returns_checkpoint_wrapper(self):
        """Test checkpoint_wrapper returns a CheckpointWrapper instance."""
        mod = _BaseWrapperModule()

        result = checkpoint_wrapper(mod)

        self.assertIsInstance(result, CheckpointWrapper)

    def test_passes_group_swap(self):
        """Test checkpoint_wrapper passes group_swap to CheckpointWrapper."""
        mod = _BaseWrapperModule()

        result = checkpoint_wrapper(mod, group_swap=True)

        self.assertTrue(result.checkpoint_kwargs["group_swap"])

    def test_passes_checkpoint_kwargs(self):
        """Test checkpoint_wrapper passes kwargs to CheckpointWrapper."""
        mod = _BaseWrapperModule()
        policy = lambda x: x  # pylint: disable=C3001

        result = checkpoint_wrapper(mod, policy_fn=policy)

        self.assertIn("policy_fn", result.checkpoint_kwargs)
        self.assertEqual(result.checkpoint_kwargs["policy_fn"], policy)

    def test_with_callable(self):
        """Test checkpoint_wrapper works with callable (lambda)."""
        fn = lambda x: x * 2  # pylint: disable=C3001

        result = checkpoint_wrapper(fn)

        self.assertIsInstance(result, CheckpointWrapper)


class TestModuleLevelAliases(unittest.TestCase):
    """Unit tests for module-level aliases (swap_wrapper, swap_tensor_wrapper, checkpoint_wrapper)."""

    def test_swap_wrapper_is_callable(self):
        """Test swap_wrapper is a callable."""
        self.assertTrue(callable(_public_api.swap_wrapper))

    def test_swap_tensor_wrapper_is_callable(self):
        """Test swap_tensor_wrapper is a callable."""
        self.assertTrue(callable(_public_api.swap_tensor_wrapper))


if __name__ == "__main__":
    unittest.main()

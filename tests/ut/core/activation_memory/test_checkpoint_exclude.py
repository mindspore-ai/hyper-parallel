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
"""Unit tests for checkpoint-exclusion saved-tensor helpers."""
# The Torch backend must be selected before importing HyperParallel modules.
# Tests intentionally inspect private helper state to cover the extracted implementation.
# pylint: disable=wrong-import-position,protected-access
from collections import OrderedDict
import importlib
import os
from typing import NamedTuple
import unittest
from unittest.mock import patch, sentinel

import torch


_checkpoint_exclude = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint_exclude")


class _Pair(NamedTuple):
    """Named tuple used to verify output container preservation."""

    left: object
    right: object


class _CheckpointExcludeTestCase(unittest.TestCase):
    """Base class that isolates cached placeholder and boundary objects."""

    def setUp(self) -> None:
        """Start each test with empty helper caches."""
        _checkpoint_exclude._get_replay_placeholder.cache_clear()
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()

    def tearDown(self) -> None:
        """Release helper cache references after each test."""
        _checkpoint_exclude._get_replay_placeholder.cache_clear()
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()


class TestRecomputedInputHandle(_CheckpointExcludeTestCase):
    """Unit tests for deferred checkpoint-exclusion inputs."""

    def test_handle_tracks_use_and_materialized_tensor(self):
        """A handle should record use and return its replay-produced tensor."""
        handle = _checkpoint_exclude._RecomputedInputHandle()
        tensor = torch.tensor([1.0, 2.0])

        self.assertFalse(handle.used)
        handle.mark_used()
        handle.materialize(tensor)

        self.assertTrue(handle.used)
        self.assertIs(handle.get_recomputed_tensor(), tensor)

    def test_handle_rejects_access_before_materialization(self):
        """An unresolved handle should fail instead of returning an invalid tensor."""
        handle = _checkpoint_exclude._RecomputedInputHandle()

        with self.assertRaisesRegex(RuntimeError, "requested before recomputation"):
            handle.get_recomputed_tensor()


class TestExcludeCache(_CheckpointExcludeTestCase):
    """Unit tests for invocation-local excluded output storage."""

    @staticmethod
    def _entry(output):
        """Create a cache entry without input bindings."""
        return _checkpoint_exclude._ExcludeCacheEntry(output, [])

    def test_cache_pops_entries_in_fifo_order_per_wrapper(self):
        """Each wrapper should replay its saved calls in original call order."""
        cache = _checkpoint_exclude._ExcludeCache()
        first = self._entry(sentinel.first)
        second = self._entry(sentinel.second)
        other = self._entry(sentinel.other)

        cache.save(1, first)
        cache.save(1, second)
        cache.save(2, other)

        self.assertIs(cache.pop(1), first)
        self.assertIs(cache.pop(2), other)
        self.assertIs(cache.pop(1), second)
        self.assertEqual(cache._entries, {})

    def test_cache_pop_rejects_missing_or_exhausted_wrapper(self):
        """Replay should fail clearly when no matching forward output remains."""
        cache = _checkpoint_exclude._ExcludeCache()
        cache.save(1, self._entry(sentinel.output))
        cache.pop(1)

        for wrapper_id in (1, 2):
            with self.subTest(wrapper_id=wrapper_id):
                with self.assertRaisesRegex(RuntimeError, "No cached forward output"):
                    cache.pop(wrapper_id)

    def test_cache_clear_releases_all_wrappers(self):
        """Cleanup should release partially consumed entries for every wrapper."""
        cache = _checkpoint_exclude._ExcludeCache()
        cache.save(1, self._entry(sentinel.first))
        cache.save(2, self._entry(sentinel.second))

        cache.clear()

        self.assertEqual(cache._entries, {})


class TestSavedTensorHooks(_CheckpointExcludeTestCase):
    """Unit tests for exclusion-region saved-tensor packing and unpacking."""

    def test_pack_marked_tensor_returns_handle_and_marks_it_used(self):
        """A marked replay input should be replaced by its deferred handle."""
        tensor = torch.arange(3.0, requires_grad=True)
        handle = _checkpoint_exclude._RecomputedInputHandle()
        setattr(tensor, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR, handle)

        packed = _checkpoint_exclude._pack_saved_tensor(tensor)

        self.assertIs(packed, handle)
        self.assertTrue(handle.used)

    def test_pack_regular_tensor_detaches_only_when_required(self):
        """Regular saved tensors should drop grad history while no-grad tensors remain unchanged."""
        grad_tensor = torch.arange(3.0, requires_grad=True)
        plain_tensor = torch.arange(3.0)

        packed_grad = _checkpoint_exclude._pack_saved_tensor(grad_tensor)
        packed_plain = _checkpoint_exclude._pack_saved_tensor(plain_tensor)

        self.assertIsNot(packed_grad, grad_tensor)
        self.assertFalse(packed_grad.requires_grad)
        self.assertTrue(torch.equal(packed_grad, grad_tensor))
        self.assertIs(packed_plain, plain_tensor)

    def test_unpack_returns_plain_value_or_materialized_handle_value(self):
        """Unpack should pass ordinary values through and resolve deferred handles."""
        handle = _checkpoint_exclude._RecomputedInputHandle()
        tensor = torch.tensor([4.0])
        handle.materialize(tensor)

        self.assertIs(_checkpoint_exclude._unpack_saved_tensor(sentinel.value), sentinel.value)
        self.assertIs(_checkpoint_exclude._unpack_saved_tensor(handle), tensor)

    def test_saved_tensor_context_uses_materialized_replay_input_in_backward(self):
        """The combined hooks should supply replay data to an excluded operation's backward."""
        tensor = torch.tensor([2.0], requires_grad=True)
        handle = _checkpoint_exclude._RecomputedInputHandle()
        setattr(tensor, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR, handle)

        with _checkpoint_exclude._saved_tensors_context():
            output = tensor.square()
        delattr(tensor, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR)
        handle.materialize(torch.tensor([3.0]))
        output.backward()

        self.assertTrue(handle.used)
        torch.testing.assert_close(tensor.grad, torch.tensor([6.0]))


class TestInputBindings(_CheckpointExcludeTestCase):
    """Unit tests for collecting, marking, resolving, and materializing inputs."""

    def test_collect_tensor_inputs_records_nested_positional_and_keyword_paths(self):
        """Tensor leaves should retain deterministic paths through supported containers."""
        first = torch.tensor([1.0])
        second = torch.tensor([2.0])
        third = torch.tensor([3.0])
        args = (first, ["skip", {"second": second}])
        kwargs = {"payload": ({"third": third},), "label": "skip"}

        leaves = _checkpoint_exclude._collect_tensor_inputs(args, kwargs)

        self.assertEqual(
            leaves,
            [
                ((("arg", 0),), first),
                ((("arg", 1), ("index", 1), ("key", "second")), second),
                ((("kwarg", "payload"), ("index", 0), ("key", "third")), third),
            ],
        )

    def test_append_tensor_inputs_accepts_an_explicit_tensor_type(self):
        """The recursive collector should support its injected tensor leaf type."""
        leaves = []
        leaf = sentinel.leaf

        _checkpoint_exclude._append_tensor_inputs(
            {"nested": [leaf]},
            (("root", "value"),),
            leaves,
            type(leaf),
        )

        self.assertEqual(leaves, [((("root", "value"), ("key", "nested"), ("index", 0)), leaf)])

    def test_mark_recompute_inputs_skips_parameters_duplicates_and_saved_outputs(self):
        """Only replay-produced activation inputs should receive deferred handles."""
        invocation_id = object()
        activation = torch.tensor([1.0], requires_grad=True)
        keyword_activation = torch.tensor([2.0], requires_grad=True)
        parameter = torch.nn.Parameter(torch.tensor([3.0]))
        saved_output = torch.tensor([4.0], requires_grad=True)
        setattr(saved_output, _checkpoint_exclude._SAVE_OUTPUT_SOURCE_ATTR, invocation_id)

        bindings, previous_handles = _checkpoint_exclude._mark_recompute_inputs(
            invocation_id,
            (activation, [activation, parameter, saved_output]),
            {"keyword": keyword_activation},
        )

        self.assertEqual(
            [binding.path for binding in bindings],
            [(('arg', 0),), (("kwarg", "keyword"),)],
        )
        self.assertEqual(len(previous_handles), 2)
        self.assertIs(
            getattr(activation, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR),
            bindings[0].handle,
        )
        self.assertIs(
            getattr(keyword_activation, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR),
            bindings[1].handle,
        )
        self.assertFalse(hasattr(parameter, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))
        self.assertFalse(hasattr(saved_output, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))

        _checkpoint_exclude._restore_recompute_inputs(previous_handles)
        self.assertFalse(hasattr(activation, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))
        self.assertFalse(hasattr(keyword_activation, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))

    def test_restore_recompute_inputs_restores_existing_or_missing_attributes(self):
        """Input cleanup should restore prior marker state exactly."""
        with_previous = torch.tensor([1.0])
        without_previous = torch.tensor([2.0])
        previous = object()
        setattr(with_previous, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR, previous)

        _, previous_handles = _checkpoint_exclude._mark_recompute_inputs(
            object(),
            (with_previous, without_previous),
            {},
        )
        _checkpoint_exclude._restore_recompute_inputs(previous_handles)

        self.assertIs(getattr(with_previous, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR), previous)
        self.assertFalse(hasattr(without_previous, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))

    def test_restore_ignores_already_removed_transient_attribute(self):
        """Cleanup should tolerate a missing attribute that did not exist before marking."""
        tensor = torch.tensor([1.0])
        _, previous_handles = _checkpoint_exclude._mark_recompute_inputs(object(), (tensor,), {})
        delattr(tensor, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR)

        self.assertIsNone(_checkpoint_exclude._restore_recompute_inputs(previous_handles))
        self.assertFalse(hasattr(tensor, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))

    def test_mark_recompute_inputs_rolls_back_after_attribute_failure(self):
        """A partial marking failure should restore every input already modified."""
        tensor = torch.tensor([1.0])
        paths_and_values = [
            ((('arg', 0),), tensor),
            ((('arg', 1),), object()),
        ]

        with patch.object(_checkpoint_exclude, "_collect_tensor_inputs", return_value=paths_and_values):
            with self.assertRaises(AttributeError):
                _checkpoint_exclude._mark_recompute_inputs(object(), (tensor,), {})

        self.assertFalse(hasattr(tensor, _checkpoint_exclude._RECOMPUTE_INPUT_HANDLE_ATTR))

    def test_resolve_input_follows_positional_and_keyword_paths(self):
        """Replay path resolution should traverse both root kinds and nested containers."""
        positional = torch.tensor([1.0])
        keyword = torch.tensor([2.0])
        args = ([{"leaf": positional}],)
        kwargs = {"payload": {"items": [keyword]}}

        resolved_positional = _checkpoint_exclude._resolve_input(
            args,
            kwargs,
            (("arg", 0), ("index", 0), ("key", "leaf")),
        )
        resolved_keyword = _checkpoint_exclude._resolve_input(
            args,
            kwargs,
            (("kwarg", "payload"), ("key", "items"), ("index", 0)),
        )

        self.assertIs(resolved_positional, positional)
        self.assertIs(resolved_keyword, keyword)

    def test_materialize_recompute_inputs_resolves_only_used_handles(self):
        """Materialization should bind detached replay tensors only for saved inputs."""
        used_handle = _checkpoint_exclude._RecomputedInputHandle()
        unused_handle = _checkpoint_exclude._RecomputedInputHandle()
        used_handle.mark_used()
        entry = _checkpoint_exclude._ExcludeCacheEntry(
            sentinel.output,
            [
                _checkpoint_exclude._InputBinding((("arg", 0), ("key", "used")), used_handle),
                _checkpoint_exclude._InputBinding((("arg", 0), ("key", "missing")), unused_handle),
            ],
        )
        replay_tensor = torch.tensor([3.0], requires_grad=True)

        _checkpoint_exclude._materialize_recompute_inputs(entry, ({"used": replay_tensor},), {})
        materialized = used_handle.get_recomputed_tensor()

        self.assertIsNot(materialized, replay_tensor)
        self.assertFalse(materialized.requires_grad)
        torch.testing.assert_close(materialized, replay_tensor)
        with self.assertRaisesRegex(RuntimeError, "requested before recomputation"):
            unused_handle.get_recomputed_tensor()

    def test_materialize_recompute_inputs_rejects_non_tensor_replay_value(self):
        """A used activation path must still resolve to a tensor during replay."""
        handle = _checkpoint_exclude._RecomputedInputHandle()
        handle.mark_used()
        entry = _checkpoint_exclude._ExcludeCacheEntry(
            sentinel.output,
            [_checkpoint_exclude._InputBinding((("kwarg", "value"),), handle)],
        )

        with self.assertRaisesRegex(RuntimeError, "did not reproduce a tensor input"):
            _checkpoint_exclude._materialize_recompute_inputs(entry, (), {"value": "not-a-tensor"})

    def test_has_used_input_reports_binding_state(self):
        """Boundary selection should reflect whether any deferred input was saved."""
        unused = _checkpoint_exclude._RecomputedInputHandle()
        used = _checkpoint_exclude._RecomputedInputHandle()
        used.mark_used()

        self.assertFalse(_checkpoint_exclude._has_used_input([]))
        self.assertFalse(
            _checkpoint_exclude._has_used_input([_checkpoint_exclude._InputBinding((('arg', 0),), unused)])
        )
        self.assertTrue(
            _checkpoint_exclude._has_used_input(
                [
                    _checkpoint_exclude._InputBinding((('arg', 0),), unused),
                    _checkpoint_exclude._InputBinding((('arg', 1),), used),
                ]
            )
        )

    def test_saved_output_provenance_prevents_remarking_in_same_invocation(self):
        """Adjacent excluded regions should not mark an output already cached by the same invocation."""
        invocation_id = object()
        tensor = torch.tensor([1.0], requires_grad=True)
        finalized = _checkpoint_exclude._finalize_save_outputs(tensor, False, invocation_id)

        same_bindings, same_previous = _checkpoint_exclude._mark_recompute_inputs(
            invocation_id,
            (finalized,),
            {},
        )
        other_bindings, other_previous = _checkpoint_exclude._mark_recompute_inputs(
            object(),
            (finalized,),
            {},
        )

        self.assertEqual(same_bindings, [])
        self.assertEqual(same_previous, [])
        self.assertEqual(len(other_bindings), 1)
        _checkpoint_exclude._restore_recompute_inputs(other_previous)


class TestReplayOutputs(_CheckpointExcludeTestCase):
    """Unit tests for replay placeholders and SAVE output finalization."""

    def test_replay_placeholder_output_matches_tensor_leaf_count(self):
        """Output elision should preserve the number of tensor leaves seen in forward."""
        empty = _checkpoint_exclude._make_replay_placeholder_output(0)
        single = _checkpoint_exclude._make_replay_placeholder_output(1)
        multiple = _checkpoint_exclude._make_replay_placeholder_output(3)

        self.assertEqual(empty, ())
        self.assertIs(single, _checkpoint_exclude._get_replay_placeholder())
        self.assertEqual(len(multiple), 3)
        self.assertTrue(all(item is single for item in multiple))
        self.assertEqual(single.numel(), 0)
        self.assertEqual(single.device.type, "cpu")

    def test_recompute_trigger_and_boundary_are_cached(self):
        """Zero-element trigger and autograd boundary types should be reused."""
        trigger = _checkpoint_exclude._get_recompute_trigger()
        boundary = _checkpoint_exclude._get_recompute_boundary()

        self.assertIs(trigger, _checkpoint_exclude._get_recompute_trigger())
        self.assertIs(boundary, _checkpoint_exclude._get_recompute_boundary())
        self.assertEqual(trigger.numel(), 0)
        self.assertEqual(trigger.device.type, "cpu")
        self.assertTrue(trigger.requires_grad)

    def test_recompute_boundary_saves_trigger_and_passes_gradient(self):
        """The boundary should expose only its zero-size dependency to outer saved-tensor hooks."""
        packed_shapes = []
        unpacked_shapes = []

        def _pack_hook(tensor):
            packed_shapes.append(tuple(tensor.shape))
            return tensor

        def _unpack_hook(tensor):
            unpacked_shapes.append(tuple(tensor.shape))
            return tensor

        tensor = torch.arange(4.0, requires_grad=True)
        boundary = _checkpoint_exclude._get_recompute_boundary()
        with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
            output = boundary.apply(tensor, _checkpoint_exclude._get_recompute_trigger())
            output.sum().backward()

        self.assertEqual(packed_shapes, [(0,)])
        self.assertEqual(unpacked_shapes, [(0,)])
        torch.testing.assert_close(tensor.grad, torch.ones(4))

    def test_finalize_outputs_preserves_nested_container_types_and_counts_tensors(self):
        """Finalization should recursively preserve list, tuple, named tuple, and mapping structure."""
        invocation_id = object()
        first = torch.tensor([1.0], requires_grad=True)
        second = torch.tensor([2.0])
        output = OrderedDict(
            [
                ("list", [first, "plain"]),
                ("pair", _Pair(first + 1, {"leaf": second})),
                ("tuple", (second, 7)),
            ]
        )
        tensor_leaf_count = [0]

        finalized = _checkpoint_exclude._finalize_save_outputs(
            output,
            False,
            invocation_id,
            tensor_leaf_count,
        )

        self.assertIsInstance(finalized, OrderedDict)
        self.assertIsInstance(finalized["list"], list)
        self.assertIsInstance(finalized["pair"], _Pair)
        self.assertIsInstance(finalized["pair"].right, dict)
        self.assertIsInstance(finalized["tuple"], tuple)
        self.assertIs(finalized["list"][0], first)
        self.assertEqual(finalized["list"][1], "plain")
        self.assertEqual(finalized["tuple"][1], 7)
        self.assertEqual(tensor_leaf_count, [4])
        tensor_leaves = [
            finalized["list"][0],
            finalized["pair"].left,
            finalized["pair"].right["leaf"],
            finalized["tuple"][0],
        ]
        self.assertTrue(
            all(
                getattr(tensor, _checkpoint_exclude._SAVE_OUTPUT_SOURCE_ATTR) is invocation_id
                for tensor in tensor_leaves
            )
        )

    def test_finalize_outputs_adds_boundary_and_provenance_to_tensor(self):
        """A required recompute boundary should preserve values and gradients on a new tensor."""
        invocation_id = object()
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        tensor_leaf_count = [0]

        finalized = _checkpoint_exclude._finalize_save_outputs(
            tensor,
            True,
            invocation_id,
            tensor_leaf_count,
        )
        finalized.sum().backward()

        self.assertIsNot(finalized, tensor)
        torch.testing.assert_close(finalized, tensor.detach())
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
        self.assertIs(getattr(finalized, _checkpoint_exclude._SAVE_OUTPUT_SOURCE_ATTR), invocation_id)
        self.assertEqual(tensor_leaf_count, [1])

    def test_finalize_replay_output_does_not_add_forward_provenance(self):
        """Replay finalization without an invocation ID should leave provenance unset."""
        tensor = torch.tensor([1.0], requires_grad=True)

        finalized = _checkpoint_exclude._finalize_save_outputs(tensor, False, None)

        self.assertIs(finalized, tensor)
        self.assertFalse(hasattr(finalized, _checkpoint_exclude._SAVE_OUTPUT_SOURCE_ATTR))

    def test_finalize_outputs_leaves_non_tensor_value_unchanged(self):
        """Non-container, non-tensor outputs should pass through unchanged."""
        result = _checkpoint_exclude._finalize_save_outputs(
            sentinel.output,
            True,
            sentinel.invocation_id,
            [0],
        )

        self.assertIs(result, sentinel.output)


if __name__ == "__main__":
    unittest.main()

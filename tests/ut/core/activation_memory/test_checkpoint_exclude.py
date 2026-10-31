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
# Tests intentionally inspect private helper state to cover the extracted implementation.
# pylint: disable=wrong-import-position,protected-access
from collections import OrderedDict
import importlib
from typing import NamedTuple
import unittest
from unittest.mock import MagicMock, sentinel

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.core.activation_memory.wrapper import checkpoint_exclude_wrapper


_checkpoint_exclude = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint_exclude")


class _Pair(NamedTuple):
    """Named tuple used to verify output container preservation."""

    left: object
    right: object


class _CountedViewSquare(torch.nn.Module):
    """Square a flattened view and record real executions."""

    def __init__(self) -> None:
        """Initialize the call counter."""
        super().__init__()
        self.calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a scalar whose backward saves a view of the input."""
        self.calls += 1
        return tensor.view(-1).square().sum()


class _ViewSavedInputBlock(torch.nn.Module):
    """Checkpointed block whose SAVE input is saved through a view."""

    def __init__(self) -> None:
        """Initialize the excluded region."""
        super().__init__()
        self.excluded = checkpoint_exclude_wrapper(_CountedViewSquare())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Recompute the head but execute the view-saving SAVE tail only once."""
        return self.excluded(tensor * 3)


class _RngAdvancingSave(torch.nn.Module):
    """SAVE region that advances RNG for a value used by backward."""

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Multiply by a random mask drawn only on the forward pass."""
        return tensor * torch.rand_like(tensor)


class _RngBoundaryBlock(torch.nn.Module):
    """Checkpoint body with RNG work both inside and after a SAVE region."""

    def __init__(self) -> None:
        """Initialize the excluded random region and a scalar scale."""
        super().__init__()
        self.excluded = checkpoint_exclude_wrapper(_RngAdvancingSave())
        self.scale = torch.nn.Parameter(torch.tensor([1.3, 0.7]))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Draw RNG after SAVE, whose replay depends on the skipped draw."""
        saved = self.excluded(tensor * self.scale).sum()
        replay_consumer = (torch.rand_like(tensor) * tensor).sum()
        return saved + replay_consumer


class _CheckpointExcludeTestCase(unittest.TestCase):
    """Base class that isolates cached checkpoint-boundary objects."""

    def setUp(self) -> None:
        """Start each test with empty checkpoint-boundary caches."""
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()

    def tearDown(self) -> None:
        """Release helper cache references after each test."""
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()

    @staticmethod
    def _pack_with_inputs(args, tensor):
        """Pack one tensor while treating *args* as the excluded call's inputs."""
        inputs = _checkpoint_exclude._capture_recompute_inputs(_checkpoint_exclude._ExcludeCache(), args, {})
        bindings = []
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(_checkpoint_exclude._PackState(inputs, bindings))
        try:
            packed = _checkpoint_exclude._pack_saved_tensor(tensor)
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)
        return packed, bindings


class TestRecomputedInputHandle(_CheckpointExcludeTestCase):
    """Unit tests for deferred checkpoint-exclusion aliases."""

    def test_handle_rejects_access_before_materialization(self):
        """An unresolved handle should fail instead of returning an invalid tensor."""
        tensor = torch.arange(4.0, requires_grad=True)
        inputs = _checkpoint_exclude._capture_recompute_inputs(_checkpoint_exclude._ExcludeCache(), (tensor,), {})
        recipe = _checkpoint_exclude._make_view_recipe(inputs[0].metadata, inputs[0].metadata)
        handle = _checkpoint_exclude._RecomputedInputHandle(recipe)

        with self.assertRaisesRegex(RuntimeError, "requested before recomputation"):
            handle.get_recomputed_tensor()
        handle.materialize(tensor)
        self.assertIs(handle.get_recomputed_tensor(), tensor)


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

    def test_cache_clear_releases_entries(self):
        """Cleanup should release cached outputs for every wrapper."""
        cache = _checkpoint_exclude._ExcludeCache()
        cache.save(1, self._entry(sentinel.first))

        cache.clear()

        self.assertEqual(cache._entries, {})


class TestSavedTensorHooks(_CheckpointExcludeTestCase):
    """Unit tests for exclusion-region saved-tensor packing and unpacking."""

    def test_pack_regular_tensor_detaches_and_records_version(self):
        """Every real save should store a detached tensor with its save-time version."""
        grad_tensor = torch.arange(3.0, requires_grad=True)
        plain_tensor = torch.arange(3.0)

        packed_grad = _checkpoint_exclude._pack_saved_tensor(grad_tensor)
        packed_plain = _checkpoint_exclude._pack_saved_tensor(plain_tensor)

        self.assertIsInstance(packed_grad, _checkpoint_exclude._SavedTensorData)
        self.assertIsInstance(packed_plain, _checkpoint_exclude._SavedTensorData)
        self.assertIsNot(packed_grad.tensor, grad_tensor)
        self.assertIsNot(packed_plain.tensor, plain_tensor)
        self.assertFalse(packed_grad.tensor.requires_grad)
        self.assertFalse(packed_plain.tensor.requires_grad)
        self.assertTrue(torch.equal(packed_grad.tensor, grad_tensor))
        self.assertTrue(torch.equal(packed_plain.tensor, plain_tensor))
        self.assertEqual(packed_grad.version, grad_tensor._version)
        self.assertEqual(packed_plain.version, plain_tensor._version)

    def test_unpack_rejects_in_place_modified_real_save(self):
        """The fallback path must reproduce PyTorch's in-place version guard."""
        tensor = torch.arange(3.0, requires_grad=True)
        packed = _checkpoint_exclude._pack_saved_tensor(tensor)

        with torch.no_grad():
            tensor.add_(1)

        with self.assertRaisesRegex(RuntimeError, "modified in-place"):
            _checkpoint_exclude._unpack_saved_tensor(packed)

    def test_pack_exact_input_returns_replay_handle(self):
        """An exact saved input should be replaced by a replayable handle."""
        tensor = torch.arange(4.0, requires_grad=True)

        packed, bindings = self._pack_with_inputs((tensor,), tensor)

        self.assertIsInstance(packed, _checkpoint_exclude._RecomputedInputHandle)
        self.assertTrue(packed.recipe.exact_input)
        self.assertEqual(len(bindings), 1)

    def test_pack_recomputed_input_bypasses_swap_hooks(self):
        """Replayable input aliases should remain handles instead of entering swap."""
        tensor = torch.arange(4.0, requires_grad=True)
        inputs = _checkpoint_exclude._capture_recompute_inputs(
            _checkpoint_exclude._ExcludeCache(),
            (tensor,),
            {},
        )
        bindings = []
        swap_hooks = MagicMock()
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(
            _checkpoint_exclude._PackState(inputs, bindings, swap_hooks=swap_hooks)
        )
        try:
            packed = _checkpoint_exclude._pack_saved_tensor(tensor)
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)

        self.assertIsInstance(packed, _checkpoint_exclude._RecomputedInputHandle)
        swap_hooks.pack_hook.assert_not_called()

    def test_pack_parameter_and_view_bypass_swap_hooks(self):
        """Parameters and their views should remain local version-checked saves."""
        parameter = torch.nn.Parameter(torch.arange(4.0))
        swap_hooks = MagicMock()
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(
            _checkpoint_exclude._PackState((), [], swap_hooks=swap_hooks)
        )
        try:
            packed_parameter = _checkpoint_exclude._pack_saved_tensor(parameter)
            packed_view = _checkpoint_exclude._pack_saved_tensor(parameter[:2])
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)

        self.assertIsNone(packed_parameter.swap_hooks)
        self.assertIsNone(packed_view.swap_hooks)
        swap_hooks.pack_hook.assert_not_called()

    def test_pack_and_unpack_real_save_use_swap_hooks(self):
        """Non-recomputable saves should use the invocation-local swap hooks."""
        tensor = torch.arange(4.0, requires_grad=True)
        detached = tensor.detach()
        swap_hooks = MagicMock()
        swap_hooks.pack_hook.return_value = detached
        swap_hooks.unpack_hook.return_value = detached
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(
            _checkpoint_exclude._PackState((), [], swap_hooks=swap_hooks)
        )
        try:
            packed = _checkpoint_exclude._pack_saved_tensor(tensor)
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)

        unpacked = _checkpoint_exclude._unpack_saved_tensor(packed)

        swap_hooks.pack_hook.assert_called_once_with(tensor)
        swap_hooks.unpack_hook.assert_called_once_with(detached)
        self.assertIs(packed.swap_hooks, swap_hooks)
        self.assertIs(unpacked, detached)

    def test_pack_saved_view_uses_relative_layout_recipe(self):
        """A different tensor object sharing input storage should still be optimized."""
        tensor = torch.arange(12.0, requires_grad=True).reshape(3, 4)

        packed, bindings = self._pack_with_inputs((tensor,), tensor.transpose(0, 1))

        self.assertIsInstance(packed, _checkpoint_exclude._RecomputedInputHandle)
        self.assertFalse(packed.recipe.exact_input)
        self.assertEqual(packed.recipe.shape, (4, 3))
        self.assertEqual(packed.recipe.stride, (1, 4))
        self.assertEqual(len(bindings), 1)

    def test_pack_unsafe_alias_falls_back_to_real_save(self):
        """Dtype reinterpretation and strided bases should stay on the identity path."""
        tensor = torch.arange(16.0, requires_grad=True)
        strided_base = tensor[::2]
        inputs = _checkpoint_exclude._capture_recompute_inputs(
            _checkpoint_exclude._ExcludeCache(),
            (strided_base,),
            {},
        )
        bindings = []
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(_checkpoint_exclude._PackState(inputs, bindings))
        try:
            packed_reinterpret = _checkpoint_exclude._pack_saved_tensor(tensor.view(torch.int32))
            packed_strided = _checkpoint_exclude._pack_saved_tensor(strided_base[1:])
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)

        self.assertIsInstance(packed_reinterpret, _checkpoint_exclude._SavedTensorData)
        self.assertIsInstance(packed_strided, _checkpoint_exclude._SavedTensorData)
        self.assertEqual(bindings, [])

    def test_unpack_returns_plain_value_or_materialized_handle_value(self):
        """Unpack should pass ordinary values through and resolve deferred handles."""
        tensor = torch.arange(4.0, requires_grad=True)
        inputs = _checkpoint_exclude._capture_recompute_inputs(_checkpoint_exclude._ExcludeCache(), (tensor,), {})
        recipe = _checkpoint_exclude._make_view_recipe(inputs[0].metadata, inputs[0].metadata)
        handle = _checkpoint_exclude._RecomputedInputHandle(recipe)
        handle.materialize(tensor)

        self.assertIs(_checkpoint_exclude._unpack_saved_tensor(sentinel.value), sentinel.value)
        self.assertIs(_checkpoint_exclude._unpack_saved_tensor(handle), tensor)

    def test_recomputed_handle_rejects_mutation_after_materialization(self):
        """A replay-materialized saved alias must retain its expected version."""
        tensor = torch.arange(4.0, requires_grad=True)
        inputs = _checkpoint_exclude._capture_recompute_inputs(_checkpoint_exclude._ExcludeCache(), (tensor,), {})
        recipe = _checkpoint_exclude._make_view_recipe(inputs[0].metadata, inputs[0].metadata)
        handle = _checkpoint_exclude._RecomputedInputHandle(recipe)
        handle.materialize(tensor.detach())

        with torch.no_grad():
            tensor.add_(1)

        with self.assertRaisesRegex(RuntimeError, "after replay materialization"):
            handle.get_recomputed_tensor()

    def test_saved_tensor_context_uses_rebuilt_view_in_backward(self):
        """The combined hooks should supply a rebuilt view to excluded backward."""
        tensor = torch.arange(6.0, requires_grad=True)
        saved_view = tensor.view(3, 2)
        inputs = _checkpoint_exclude._capture_recompute_inputs(_checkpoint_exclude._ExcludeCache(), (tensor,), {})
        bindings = []
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(_checkpoint_exclude._PackState(inputs, bindings))
        try:
            with _checkpoint_exclude._saved_tensors_context():
                output = saved_view.square()
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)
        replay = tensor.detach().mul(2)
        _checkpoint_exclude._materialize_recompute_inputs(
            _checkpoint_exclude._ExcludeCacheEntry(None, bindings),
            (replay,),
            {},
        )

        output.sum().backward()

        torch.testing.assert_close(tensor.grad, replay.mul(2))


class TestInputBindings(_CheckpointExcludeTestCase):
    """Unit tests for collecting, capturing, resolving, and materializing inputs."""

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

    def test_capture_skips_parameters_and_duplicates(self):
        """All unique activation inputs should be captured when outputs are cached."""
        cache = _checkpoint_exclude._ExcludeCache()
        activation = torch.arange(4.0, requires_grad=True)
        keyword_activation = torch.arange(4.0, 8.0, requires_grad=True)
        parameter = torch.nn.Parameter(torch.arange(3.0))
        prior_output = torch.arange(8.0, 12.0, requires_grad=True)

        inputs = _checkpoint_exclude._capture_recompute_inputs(
            cache,
            (activation, [activation, parameter, prior_output]),
            {"keyword": keyword_activation},
        )

        self.assertEqual(
            [input_info.path for input_info in inputs],
            [((("arg", 0),)), ((("arg", 1), ("index", 2))), ((("kwarg", "keyword"),))],
        )

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

    def test_materialize_rebuilds_exact_input_and_view(self):
        """Materialization should detach exact inputs and rebuild saved views."""
        source = torch.arange(12.0, requires_grad=True).reshape(3, 4)
        inputs = _checkpoint_exclude._capture_recompute_inputs(_checkpoint_exclude._ExcludeCache(), (source,), {})
        bindings = []
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(_checkpoint_exclude._PackState(inputs, bindings))
        try:
            packed_exact = _checkpoint_exclude._pack_saved_tensor(source)
            packed_view = _checkpoint_exclude._pack_saved_tensor(source.transpose(0, 1))
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)
        replay = source.detach().mul(2)

        _checkpoint_exclude._materialize_recompute_inputs(
            _checkpoint_exclude._ExcludeCacheEntry(None, bindings),
            (replay,),
            {},
        )

        torch.testing.assert_close(packed_exact.get_recomputed_tensor(), replay)
        torch.testing.assert_close(packed_view.get_recomputed_tensor(), replay.transpose(0, 1))
        self.assertFalse(packed_exact.get_recomputed_tensor().requires_grad)

    def test_materialize_recompute_inputs_rejects_non_tensor_replay_value(self):
        """A captured activation path must still resolve to a tensor during replay."""
        source = torch.arange(4.0, requires_grad=True)
        inputs = _checkpoint_exclude._capture_recompute_inputs(
            _checkpoint_exclude._ExcludeCache(),
            (source,),
            {},
        )
        bindings = []
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(_checkpoint_exclude._PackState(inputs, bindings))
        try:
            _checkpoint_exclude._pack_saved_tensor(source)
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)
        entry = _checkpoint_exclude._ExcludeCacheEntry(None, bindings)

        with self.assertRaisesRegex(RuntimeError, "did not reproduce a tensor input"):
            _checkpoint_exclude._materialize_recompute_inputs(entry, ("not-a-tensor",), {})

    def test_materialize_rejects_replay_input_version_change(self):
        """Replay must not rebuild a saved alias from a mutated upstream activation."""
        source = torch.arange(4.0, requires_grad=True)
        inputs = _checkpoint_exclude._capture_recompute_inputs(
            _checkpoint_exclude._ExcludeCache(),
            (source,),
            {},
        )
        bindings = []
        token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(_checkpoint_exclude._PackState(inputs, bindings))
        try:
            _checkpoint_exclude._pack_saved_tensor(source)
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(token)
        entry = _checkpoint_exclude._ExcludeCacheEntry(None, bindings)
        replay = source.detach().clone()
        replay.add_(1)

        with self.assertRaisesRegex(RuntimeError, "layout/version changed"):
            _checkpoint_exclude._materialize_recompute_inputs(entry, (replay,), {})


class TestReplayOutputs(_CheckpointExcludeTestCase):
    """Unit tests for RNG recovery and SAVE output finalization."""

    def test_recompute_boundary_saves_trigger_and_passes_gradient(self):
        """The boundary should expose only its zero-size dependency to outer hooks."""
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

    def test_rng_snapshot_restores_cpu_state_after_skipped_draw(self):
        """The SAVE exit snapshot should reproduce RNG consumption during replay."""
        tensor = torch.ones(8)
        torch.manual_seed(101)
        expected = torch.rand(4)
        torch.manual_seed(101)
        snapshot = _checkpoint_exclude._capture_rng_state((tensor,), {}, tensor, True)
        torch.rand(32)

        _checkpoint_exclude._restore_rng_state(snapshot)

        torch.testing.assert_close(torch.rand(4), expected)

    def test_finalize_outputs_preserves_nested_container_types_and_tensors(self):
        """Finalization should preserve containers and tensor identity without boundaries."""
        first = torch.tensor([1.0], requires_grad=True)
        second = torch.tensor([2.0])
        output = OrderedDict(
            [
                ("list", [first, "plain"]),
                ("pair", _Pair(first + 1, {"leaf": second})),
                ("tuple", (second, 7)),
            ]
        )
        finalized = _checkpoint_exclude._finalize_save_outputs(output, False)

        self.assertIsInstance(finalized, OrderedDict)
        self.assertIsInstance(finalized["list"], list)
        self.assertIsInstance(finalized["pair"], _Pair)
        self.assertIsInstance(finalized["pair"].right, dict)
        self.assertIsInstance(finalized["tuple"], tuple)
        self.assertIs(finalized["list"][0], first)
        self.assertIs(finalized["pair"].right["leaf"], second)
        self.assertEqual(finalized["list"][1], "plain")
        self.assertEqual(finalized["tuple"][1], 7)


class TestCheckpointExcludeIntegration(_CheckpointExcludeTestCase):
    """End-to-end coverage through the checkpoint wrapper."""

    @classmethod
    def setUpClass(cls) -> None:
        """Force CPU checkpoint execution so CPU-only CI does not initialize NPU."""
        previous_device_type = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        cls.addClassCleanup(DefaultDeviceType.set_device_type, previous_device_type)

    def test_recompute_to_save_uses_replayed_input_view(self):
        """A saved view of a replayed activation should be rederived during backward."""
        block = checkpoint_wrapper(_ViewSavedInputBlock())
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)

        value = block(tensor)
        value.backward()

        self.assertEqual(block.excluded.calls, 1)
        torch.testing.assert_close(value, torch.tensor(45.0))
        torch.testing.assert_close(tensor.grad, torch.tensor([18.0, 36.0]))
        self.assertFalse(hasattr(tensor, "_hyper_parallel_recompute_input_handle"))

    def test_excluded_region_restores_rng_state_during_replay(self):
        """RNG advanced by a skipped SAVE body must stay aligned downstream."""
        torch.manual_seed(17)
        baseline = _RngBoundaryBlock()
        torch.manual_seed(23)
        optimized = _RngBoundaryBlock()
        optimized.load_state_dict(baseline.state_dict())
        baseline_input = torch.tensor([1.0, 2.0], requires_grad=True)
        optimized_input = baseline_input.detach().clone().requires_grad_()

        torch.manual_seed(31)
        baseline_value = baseline(baseline_input)
        baseline_grads = torch.autograd.grad(baseline_value, (baseline_input, baseline.scale))
        torch.manual_seed(31)
        optimized_value = checkpoint_wrapper(optimized)(optimized_input)
        optimized_grads = torch.autograd.grad(optimized_value, (optimized_input, optimized.scale))

        torch.testing.assert_close(optimized_value, baseline_value)
        torch.testing.assert_close(optimized_grads[0], baseline_grads[0])
        torch.testing.assert_close(optimized_grads[1], baseline_grads[1])


if __name__ == "__main__":
    unittest.main()

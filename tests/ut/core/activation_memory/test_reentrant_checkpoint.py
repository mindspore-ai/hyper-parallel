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
"""CPU checks for reentrant checkpoints and shared exclusion graphs."""
import gc
import contextlib
import importlib
import unittest
import weakref
from copy import deepcopy
from functools import partial
from unittest.mock import patch

import torch

from hyper_parallel.core.activation_memory import (
    checkpoint, checkpoint_exclude_wrapper, reentrant_checkpoint_scope,
    recompute_handle_collector_ctx, recompute_handle, recompute_session_ctx, clear_recompute_session,
    is_recomputing,
)
from hyper_parallel.core.activation_memory.api import CheckpointPolicy, create_selective_checkpoint_contexts

_RECIPES = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint_exclude")


class TestReentrantCheckpoint(unittest.TestCase):
    """Compare ordinary and split checkpoint gradients with eager autograd."""

    def test_sac_exclusion_preserves_direct_and_nested_checkpoint_gradients(self):
        """Excluded linears must not consume the enclosing SAC cache slots."""
        def policy(_ctx, op, *_args, **_kwargs):
            if op == torch.ops.aten.addmm.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE

        reference = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Sigmoid(), torch.nn.Linear(2, 2), torch.nn.Sigmoid(),
        )
        inputs = torch.ones(2, 2, requires_grad=True)
        expected = reference(inputs)
        expected.sum().backward()
        for kind in ("non_reentrant", "reentrant", "nested"):
            with self.subTest(kind=kind):
                model = deepcopy(reference)
                model.zero_grad(set_to_none=True)
                calls = []
                model[0].register_forward_hook(lambda *_args: calls.append(1))
                model[0] = checkpoint_exclude_wrapper(model[0])
                candidate_inputs = inputs.detach().requires_grad_()

                def nested(value):
                    return checkpoint(model, value, use_reentrant=True)

                actual = checkpoint(
                    nested if kind == "nested" else model,
                    candidate_inputs,
                    use_reentrant=kind == "reentrant",
                    context_fn=partial(create_selective_checkpoint_contexts, policy),
                )
                actual.sum().backward()
                self.assertEqual(len(calls), 2 if kind == "nested" else 1)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(candidate_inputs.grad, inputs.grad)
                for got, want in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(got.grad, want.grad)

    def test_excluded_internal_saves_and_input_recipes_release_after_backward(self) -> None:
        """The local graph retains real internal saves but rebuilds input aliases on replay."""
        source = torch.tensor(2.0, requires_grad=True)
        counts = {"recipes": 0, "internal": 0, "calls": 0}

        def _excluded_body(value: torch.Tensor) -> torch.Tensor:
            """Create both replayable input and independent internal saves."""
            counts["calls"] += 1
            return (value.square() + 1).square()

        excluded = checkpoint_exclude_wrapper(_excluded_body)
        original_pack = _RECIPES._pack_saved_tensor  # pylint: disable=protected-access

        def _record_pack(tensor: torch.Tensor) -> object:
            """Count saved-tensor hook results without retaining activations."""
            packed = original_pack(tensor)
            if isinstance(packed, _RECIPES._RecomputedInputHandle):  # pylint: disable=protected-access
                counts["recipes"] += 1
            elif isinstance(packed, _RECIPES._SavedTensorData):  # pylint: disable=protected-access
                counts["internal"] += 1
            return packed

        with patch.object(_RECIPES, "_pack_saved_tensor", side_effect=_record_pack):
            with recompute_handle_collector_ctx() as handles:
                result = checkpoint(lambda value: excluded(value * 2), source, use_reentrant=True)
            frame = handles[0]
            self.assertGreater(counts["recipes"], 0)
            self.assertGreater(counts["internal"], 0)
            self.assertEqual(counts["calls"], 1)
            result.backward()
        reference = torch.tensor(2.0, requires_grad=True)
        ((reference * 2).square() + 1).square().backward()
        torch.testing.assert_close(result.detach(), ((reference.detach() * 2).square() + 1).square())
        torch.testing.assert_close(source.grad, reference.grad)
        self.assertEqual(counts["calls"], 1)
        self.assertTrue(frame.released)
        self.assertEqual(frame.entries, [])

    def test_scope_provider_is_lazy(self):
        """A plain forward never enumerates scoped stage parameters."""
        calls = []
        param = torch.nn.Parameter(torch.tensor(2.0))

        def _provide():
            calls.append(1)
            return (param,)

        with reentrant_checkpoint_scope(_provide):
            self.assertEqual((torch.tensor(2.0) * param).item(), 4.0)
            self.assertEqual(calls, [])
            output = checkpoint(lambda value: value * param,
                                torch.tensor(3.0, requires_grad=True), use_reentrant=True)
        self.assertEqual(calls, [1])
        self.assertEqual(output.item(), 6.0)

    def test_context_fn_brackets_forward_and_replay(self):
        """Reentrant contexts enter in the correct execution phase."""
        events = []

        @contextlib.contextmanager
        def _phase(name):
            events.append((name, "enter"))
            try:
                yield
            finally:
                events.append((name, "exit"))

        def _contexts():
            return _phase("forward"), _phase("replay")

        source = torch.tensor(2.0, requires_grad=True)
        checkpoint(lambda value: value.square(), source, use_reentrant=True,
                   context_fn=_contexts).backward()
        self.assertEqual(events, [
            ("forward", "enter"), ("forward", "exit"),
            ("replay", "enter"), ("replay", "exit"),
        ])

    def test_recompute_state_visible_in_replay(self):
        """The shared execution-state API distinguishes original forward from replay."""
        observed = []

        def _layer(value):
            observed.append(is_recomputing())
            return value.square()

        checkpoint(_layer, torch.tensor(2.0, requires_grad=True), use_reentrant=True).backward()
        self.assertEqual(observed, [False, True])

    def test_excluded_nested_repeated_inputs_and_unused_output(self):
        """An excluded body executes once and propagates every used input gradient."""
        weight = torch.nn.Parameter(torch.tensor(2.0))
        source = torch.tensor(3.0, requires_grad=True)
        refs = []
        calls = []
        excluded = checkpoint_exclude_wrapper(lambda data, *, extras: (
            (data[0] + data[1] + extras["value"]) * weight,
            data[0] * weight,
        ))

        def _layer(value):
            activation = value * 2
            if not torch.is_grad_enabled():
                refs.append(weakref.ref(activation.untyped_storage()))
            calls.append(torch.is_grad_enabled())
            return excluded((activation, activation), extras={"value": activation})

        output, unused = checkpoint(_layer, source, use_reentrant=True)
        del unused
        gc.collect()
        self.assertIsNone(refs[0]())
        output.backward()
        self.assertEqual(calls, [False, True])
        self.assertEqual(source.grad.item(), 12.0)
        self.assertEqual(weight.grad.item(), 18.0)

    def test_parameter_only_checkpoint(self):
        """A parameter can receive gradients without differentiable activations."""
        param = torch.nn.Parameter(torch.tensor(3.0))
        result = checkpoint(lambda value: value * param, torch.tensor(4.0), use_reentrant=True)
        result.backward()
        self.assertEqual(param.grad.item(), 4.0)

    def test_parameter_only_excluded_region(self):
        """A local body with no differentiable input still accumulates its parameter."""
        param = torch.nn.Parameter(torch.tensor(3.0))
        excluded = checkpoint_exclude_wrapper(lambda value: value * param)
        result = checkpoint(excluded, torch.tensor(4.0), use_reentrant=True)
        result.backward()
        self.assertEqual(param.grad.item(), 4.0)

    def test_prefired_session_retains_graph_until_clear(self):
        """A scheduled session reuses the same replay for two gradient requests."""
        weight = torch.nn.Parameter(torch.tensor(2.0))
        source = torch.tensor(3.0, requires_grad=True)
        calls = []

        def _layer(value):
            calls.append(torch.is_grad_enabled())
            return value * weight

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(_layer, source, use_reentrant=True)
        session_id = object()
        with recompute_session_ctx(session_id, retain_on_unpack=True):
            recompute_handle(handles[0], session_id)
            dx = torch.autograd.grad(output, source, retain_graph=True)[0]
            second_dx = torch.autograd.grad(output, source)[0]
        self.assertEqual((dx.item(), second_dx.item()), (2.0, 2.0))
        self.assertEqual(calls, [False, True])
        clear_recompute_session(session_id)
        self.assertTrue(handles[0].released)

    def test_session_registration_after_context_entry(self):
        """A handle prefired inside an active session inherits retention."""
        source = torch.tensor(2.0, requires_grad=True)
        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(lambda value: value.square(), source, use_reentrant=True)
        session_id = object()
        with recompute_session_ctx(session_id, retain_on_unpack=True):
            recompute_handle(handles[0], session_id)
            self.assertTrue(handles[0].retain_graph)
            self.assertEqual(torch.autograd.grad(output, source, retain_graph=True)[0].item(), 4.0)
        self.assertFalse(handles[0].retain_graph)
        clear_recompute_session(session_id)
        self.assertTrue(handles[0].released)

    def test_transposed_offset_alias_recipe(self):
        """A saved nonzero-offset view is rebuilt without retaining its first storage."""
        class SaveView(torch.autograd.Function):
            """Force saving a transposed view within the excluded graph."""

            @staticmethod
            def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
                """Save input and its transpose for backward."""
                ctx.save_for_backward(value, value.T)
                return value.square()

            @staticmethod
            def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
                """Read both reconstructed aliases."""
                value, transpose = ctx.saved_tensors
                return grad * (value + transpose.T)

        source = torch.arange(16.0).reshape(4, 4).requires_grad_()
        refs = []
        excluded = checkpoint_exclude_wrapper(SaveView.apply)

        def _layer(value):
            activation = (value * 2 + 1).flatten()[4:8].reshape(2, 2)
            if not torch.is_grad_enabled():
                refs.append(weakref.ref(activation.untyped_storage()))
            return excluded(activation)

        output = checkpoint(_layer, source, use_reentrant=True)
        gc.collect()
        self.assertIsNone(refs[0]())
        output.sum().backward()
        expected = torch.zeros_like(source)
        expected.flatten()[4:8] = 4 * (source.detach().flatten()[4:8] * 2 + 1)
        torch.testing.assert_close(source.grad, expected)

    def test_split_exclude_gradient_targets(self):
        """DX avoids parameter accumulation; DW reaches both local and upstream weights."""
        upstream = torch.nn.Parameter(torch.tensor(2.0))
        local = torch.nn.Parameter(torch.tensor(5.0))
        source = torch.tensor(3.0, requires_grad=True)
        excluded = checkpoint_exclude_wrapper(lambda value: value * local)
        with reentrant_checkpoint_scope((upstream, local)) as handles:
            output = checkpoint(lambda value: excluded(value * upstream), source, use_reentrant=True)
        handles[0].set_phase("dx")
        self.assertEqual(torch.autograd.grad(output, source, retain_graph=True)[0].item(), 10.0)
        self.assertIsNone(upstream.grad)
        self.assertIsNone(local.grad)
        handles[0].set_phase("dw")
        torch.autograd.backward(output, inputs=(upstream, local, handles[0].backward_trigger))
        self.assertEqual(upstream.grad.item(), 15.0)
        self.assertEqual(local.grad.item(), 6.0)

    def test_integer_first_input_and_detached_output(self):
        """Nondifferentiable leaves do not hide later activation gradient slots."""
        weight = torch.nn.Parameter(torch.tensor(2.0))
        source = torch.tensor(3.0, requires_grad=True)
        excluded = checkpoint_exclude_wrapper(lambda integer, value: (integer, value * weight, value.detach()))
        result = checkpoint(lambda value: excluded(torch.tensor(7), value * 2), source, use_reentrant=True)
        self.assertFalse(result[0].requires_grad)
        self.assertFalse(result[2].requires_grad)
        result[1].backward()
        self.assertEqual(source.grad.item(), 4.0)
        self.assertEqual(weight.grad.item(), 6.0)

    def test_explicit_parameter_also_captured(self):
        """A parameter in both the scope and arguments contributes only once."""
        param = torch.nn.Parameter(torch.tensor(5.0))
        source = torch.tensor(3.0, requires_grad=True)
        excluded = checkpoint_exclude_wrapper(lambda value, explicit: value * explicit + value * param)
        with reentrant_checkpoint_scope((param, param)) as handles:
            output = checkpoint(excluded, source, param, use_reentrant=True)
        handles[0].set_phase("dx")
        dx = torch.autograd.grad(output, source, retain_graph=True)[0]
        self.assertEqual(dx.item(), 10.0)
        self.assertIsNone(param.grad)
        handles[0].set_phase("dw")
        torch.autograd.backward(output, inputs=(param, handles[0].backward_trigger))
        self.assertEqual(param.grad.item(), 6.0)

    def test_explicit_parameter_hook_fires_only_in_dw(self):
        """A scoped explicit parameter never invokes its accumulation hook in DX."""
        param = torch.nn.Parameter(torch.tensor(3.0))
        source = torch.tensor(2.0, requires_grad=True)
        hits = []
        hook = param.register_hook(lambda grad: hits.append(grad.item()))
        excluded = checkpoint_exclude_wrapper(lambda value, weight: value * weight)
        with reentrant_checkpoint_scope((param,)) as handles:
            output = checkpoint(excluded, source, param, use_reentrant=True)
        handles[0].set_phase("dx")
        self.assertEqual(torch.autograd.grad(output, source, retain_graph=True)[0].item(), 3.0)
        self.assertEqual(hits, [])
        handles[0].set_phase("dw")
        torch.autograd.backward(output, inputs=(param, handles[0].backward_trigger))
        self.assertEqual(hits, [2.0])
        self.assertEqual(param.grad.item(), 2.0)
        hook.remove()

    def test_mixed_explicit_and_captured_parameter_hook(self):
        """A transforming hook sees each explicit and captured DW contribution once."""
        param = torch.nn.Parameter(torch.tensor(5.0))
        source = torch.tensor(3.0, requires_grad=True)
        hits = []

        def _transform(grad):
            hits.append(grad.item())
            return grad * 2

        hook = param.register_hook(_transform)
        excluded = checkpoint_exclude_wrapper(lambda value, explicit: value * explicit + value * param)
        with reentrant_checkpoint_scope((param,)) as handles:
            output = checkpoint(excluded, source, param, use_reentrant=True)
        handles[0].set_phase("dx")
        self.assertEqual(torch.autograd.grad(output, source, retain_graph=True)[0].item(), 10.0)
        self.assertEqual(hits, [])
        handles[0].set_phase("dw")
        torch.autograd.backward(output, inputs=(param, handles[0].backward_trigger))
        self.assertEqual(sorted(hits), [3.0, 3.0])
        self.assertEqual(param.grad.item(), 12.0)
        hook.remove()

    def test_captured_parameter_hook_receives_only_dw_gradient(self):
        """A closure-only parameter has no synthetic None-gradient hook call."""
        param = torch.nn.Parameter(torch.tensor(3.0))
        source = torch.tensor(2.0, requires_grad=True)
        hits = []

        def _transform(grad):
            hits.append(grad.item())
            return grad * 2

        hook = param.register_hook(_transform)
        excluded = checkpoint_exclude_wrapper(lambda value: value * param)
        with reentrant_checkpoint_scope((param,)) as handles:
            output = checkpoint(excluded, source, use_reentrant=True)
        handles[0].set_phase("dx")
        self.assertEqual(torch.autograd.grad(output, source, retain_graph=True)[0].item(), 3.0)
        self.assertEqual(hits, [])
        handles[0].set_phase("dw")
        torch.autograd.backward(output, inputs=(param, handles[0].backward_trigger))
        self.assertEqual(hits, [2.0])
        self.assertEqual(param.grad.item(), 4.0)
        hook.remove()

    def test_repeated_explicit_parameter_input(self):
        """Repeated explicit references share one local proxy contribution."""
        param = torch.nn.Parameter(torch.tensor(5.0))
        excluded = checkpoint_exclude_wrapper(lambda value, first, second: value * (first + second))
        with reentrant_checkpoint_scope((param,)) as handles:
            output = checkpoint(excluded, torch.tensor(3.0, requires_grad=True), param, param,
                                use_reentrant=True)
        handles[0].set_phase("dw")
        torch.autograd.backward(output, inputs=(param, handles[0].backward_trigger))
        self.assertEqual(param.grad.item(), 6.0)

    def test_no_grad_bypass_and_failed_forward_cleanup(self):
        """Inference runs directly; failed forward does not leave a collected handle."""
        with torch.no_grad():
            output = checkpoint(lambda value: value + 1, torch.tensor(2.0), use_reentrant=True)
        self.assertEqual(output.item(), 3.0)
        self.assertIsNone(output.grad_fn)
        with reentrant_checkpoint_scope(()) as handles:
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                checkpoint(lambda value: (_ for _ in ()).throw(RuntimeError("forward failed")),
                           torch.tensor(2.0, requires_grad=True), use_reentrant=True)
        self.assertEqual(handles, [])

    def test_local_graph_released_with_output_still_held(self):
        """A completed frame does not retain its excluded output through the bridge."""
        source = torch.tensor(3.0, requires_grad=True)
        excluded = checkpoint_exclude_wrapper(lambda value: value.square())
        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(lambda value: excluded(value * 2) + 1, source, use_reentrant=True)
        frame = handles[0]
        entry = next(iter(frame.exclude_cache._entries.values()))[0]  # pylint: disable=protected-access
        local_ref = weakref.ref(entry.tape.outputs[0])
        output.backward()
        gc.collect()
        self.assertIsNone(local_ref())
        self.assertIsNone(entry.output)
        self.assertTrue(frame.released)

    def test_nested_exclusions_reject_and_clean_up(self):
        """Nested local graphs fail explicitly and do not leave collector frames."""
        inner = checkpoint_exclude_wrapper(lambda value: value.square())
        outer = checkpoint_exclude_wrapper(lambda value: inner(value) * 2)
        with recompute_handle_collector_ctx() as handles:
            with self.assertRaisesRegex(ValueError, "Nested checkpoint exclusion"):
                checkpoint(outer, torch.tensor(3.0, requires_grad=True),
                           use_reentrant=True)
        self.assertEqual(handles, [])

    def test_nested_reentrant_checkpoints_reject(self):
        """Nested checkpoint calls cannot silently bypass the inner checkpoint."""
        with self.assertRaisesRegex(ValueError, "Nested reentrant checkpoints"):
            checkpoint(lambda value: checkpoint(lambda item: item.square(), value,
                                                use_reentrant=True),
                       torch.tensor(3.0, requires_grad=True), use_reentrant=True)

    def test_rng_advance_across_excluded_body(self):
        """Replay skips the excluded random op without shifting subsequent RNG."""
        torch.manual_seed(123)
        source = torch.tensor(2.0, requires_grad=True)
        excluded = checkpoint_exclude_wrapper(lambda value: value * torch.rand(()))
        observed = []

        def _layer(value):
            first = value * torch.rand(())
            middle = excluded(first)
            last = torch.rand(())
            observed.append(last.item())
            return middle * last

        checkpoint(_layer, source, use_reentrant=True).backward()
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0], observed[1])

    def test_split_sequential_checkpoint_upstream_weight(self):
        """DW must propagate input gradients to earlier parameterized regions."""
        first = torch.nn.Parameter(torch.tensor(2.0))
        second = torch.nn.Parameter(torch.tensor(5.0))
        source = torch.tensor(3.0, requires_grad=True)
        counts = [0, 0]

        def _layer_a(value):
            counts[0] += 1
            return value * first

        def _layer_b(value):
            counts[1] += 1
            return value * second

        with reentrant_checkpoint_scope((first, second)) as handles:
            middle = checkpoint(_layer_a, source, use_reentrant=True)
            output = checkpoint(_layer_b, middle, use_reentrant=True)
        for handle in handles:
            handle.set_phase("dx")
        dx = torch.autograd.grad(output, source, retain_graph=True)[0]
        self.assertEqual(dx.item(), 10.0)
        self.assertIsNone(first.grad)
        self.assertIsNone(second.grad)
        for handle in handles:
            handle.set_phase("dw")
        torch.autograd.backward(output, inputs=(first, second, *(item.backward_trigger for item in handles)))
        self.assertEqual(first.grad.item(), 15.0)
        self.assertEqual(second.grad.item(), 6.0)
        self.assertEqual(counts, [2, 2])


if __name__ == "__main__":
    unittest.main()

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
"""CPU lifecycle and alias tests for reentrant checkpoint exclusion."""

import gc
import threading
import unittest
import weakref

import torch

from hyper_parallel.core.activation_memory import (
    checkpoint, checkpoint_exclude_wrapper, clear_recompute_session, recompute_handle,
    recompute_handle_collector_ctx, recompute_session_ctx,
)


class TestReentrantLifecycle(unittest.TestCase):
    """Exercise retained and abandoned local graphs without device dependencies."""

    def test_transposed_nonzero_offset_alias_releases_original_storage(self) -> None:
        """An excluded matmul saves a strided offset alias without retaining its base."""
        torch.manual_seed(49)
        weight_data = torch.randn(4, 3)
        input_data = torch.randn(4, 6)

        def run(reentrant: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
            """Return reference or checkpoint loss and gradients."""
            weight = torch.nn.Parameter(weight_data.clone())
            source = input_data.clone().requires_grad_()
            storage_refs = []
            excluded = checkpoint_exclude_wrapper(lambda alias: alias @ weight)

            def layer(value: torch.Tensor) -> torch.Tensor:
                """Pass an offset transposed view into the excluded matmul."""
                base = (value * 1.2).contiguous()
                alias = base[:, 1:5].transpose(0, 1)
                if not torch.is_grad_enabled():
                    storage_refs.append(weakref.ref(base.untyped_storage()))
                return excluded(alias).square().sum()

            loss = checkpoint(layer, source, use_reentrant=True) if reentrant else layer(source)
            if reentrant:
                gc.collect()
                self.assertIsNone(storage_refs[0](), "offset alias kept original storage alive")
            loss.backward()
            return loss.detach(), source.grad, weight.grad, storage_refs

        expected = run(False)
        observed = run(True)
        for actual, reference in zip(observed[:3], expected[:3]):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        self.assertIsNone(observed[3][0](), "backward retained the excluded input storage")

    def test_abandoned_forward_releases_cached_graph(self) -> None:
        """Dropping an output without backward leaves no cached local graph alive."""
        source = torch.randn(4, requires_grad=True)
        weight = torch.nn.Parameter(torch.randn(4))
        excluded = checkpoint_exclude_wrapper(lambda value: value.square() * weight)
        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(lambda value: excluded(value * 3).sum(), source, use_reentrant=True)
        frame_ref = weakref.ref(handles[0])
        del output, handles
        gc.collect()
        self.assertIsNone(frame_ref(), "abandoned checkpoint kept its frame alive")

    def test_backward_exception_releases_graph_with_output_held(self) -> None:
        """An error inside the excluded backward clears cached graph references."""
        class FailingBackward(torch.autograd.Function):
            """Raise from the local graph after its forward is cached."""

            @staticmethod
            def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
                """Return a distinct output for exclusion."""
                del ctx
                return value.square()

            @staticmethod
            def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
                """Fail during the local backward traversal."""
                raise RuntimeError("deliberate local backward failure")

        excluded = checkpoint_exclude_wrapper(FailingBackward.apply)
        source = torch.tensor(2.0, requires_grad=True)
        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(lambda value: excluded(value * 2) + 1, source, use_reentrant=True)
        frame = handles[0]
        with self.assertRaisesRegex(RuntimeError, "deliberate local backward failure"):
            output.backward()
        self.assertTrue(frame.released)
        self.assertFalse(frame.entries)
        self.assertIsNone(frame.replay_outputs)

    def test_chained_excludes_and_direct_closure_parameter(self) -> None:
        """Two SAVE regions and a parameter passed explicitly and captured add once."""
        def run(reentrant: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
            """Return reference or checkpoint gradients and SAVE call counts."""
            param = torch.nn.Parameter(torch.tensor(2.0))
            source = torch.tensor(3.0, requires_grad=True)
            calls = [0, 0]

            def first(value: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
                """Use the same parameter through direct and captured edges."""
                calls[0] += 1
                return value * direct + value * param

            def second(value: torch.Tensor) -> torch.Tensor:
                """Save the output of the first excluded region."""
                calls[1] += 1
                return value.square()

            first_save = checkpoint_exclude_wrapper(first)
            second_save = checkpoint_exclude_wrapper(second)

            def layer(value: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
                """Chain both excluded regions in one forward."""
                return second_save(first_save(value * 2, direct)).sum()

            output = checkpoint(layer, source, param, use_reentrant=True) if reentrant else layer(source, param)
            output.backward()
            return output.detach(), source.grad, param.grad, calls

        expected = run(False)
        observed = run(True)
        for actual, reference in zip(observed[:3], expected[:3]):
            torch.testing.assert_close(actual, reference)
        self.assertEqual(observed[3], [1, 1])

    def test_retained_session_replays_once_across_thread(self) -> None:
        """A session activated on another thread retains the graph for two VJPs."""
        source = torch.tensor(3.0, requires_grad=True)
        param = torch.nn.Parameter(torch.tensor(2.0))
        calls = []
        excluded = checkpoint_exclude_wrapper(lambda value: value * param)

        def layer(value: torch.Tensor) -> torch.Tensor:
            """Track original and replay calls through the excluded region."""
            calls.append(torch.is_grad_enabled())
            return excluded(value * 2)

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(layer, source, use_reentrant=True)
        session_id = object()
        errors = []
        grads = []

        def worker() -> None:
            """Request two VJPs while the retained session is active."""
            try:
                with recompute_session_ctx(session_id, retain_on_unpack=True):
                    grads.append(torch.autograd.grad(output, source, retain_graph=True)[0])
                    grads.append(torch.autograd.grad(output, source)[0])
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        try:
            recompute_handle(handles[0], session_id)
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "cross-thread backward did not finish")
            if errors:
                raise errors[0]
            self.assertEqual(len(grads), 2)
            for grad in grads:
                torch.testing.assert_close(grad, torch.tensor(4.0))
            self.assertEqual(calls, [False, True])
        finally:
            clear_recompute_session(session_id)
        self.assertTrue(handles[0].released)


if __name__ == "__main__":
    unittest.main()

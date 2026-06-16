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
"""Unit tests for skip_offload decorator and opaque region behaviour."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession, skip_offload


class _MockSessionContext:
    """Minimal patches so OffloadSession can run on CPU-only hosts.

    Patches ``DeviceTimer``, ``profile_transfer_bandwidth``,
    ``torch.Stream``, and ``torch.accelerator.current_stream``
    — all components that would otherwise require hardware
    accelerator support.
    """

    def __init__(self) -> None:
        self._patches = []

    def start(self) -> None:
        """Apply patches for CPU-only testing."""
        timer_patch = patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.DeviceTimer"
        )
        mock_timer_cls = timer_patch.start()
        mock_timer_cls.return_value.start.return_value = None
        mock_timer_cls.return_value.stop.return_value = 0.0
        self._patches.append(timer_patch)

        bw_patch = patch(
            "hyper_parallel.auto_parallel.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
            return_value=(16.0, 16.0),
        )
        bw_patch.start()
        self._patches.append(bw_patch)

        stream_patch = patch("torch.Stream")
        stream_patch.start()
        self._patches.append(stream_patch)

        acc_stream_patch = patch("torch.accelerator.current_stream")
        mock_stream = acc_stream_patch.start()
        mock_stream.return_value.wait_stream.return_value = None
        mock_stream.return_value.synchronize.return_value = None
        self._patches.append(acc_stream_patch)

    def stop(self) -> None:
        for p in reversed(self._patches):
            p.stop()


class TestSkipOffloadDecorator(unittest.TestCase):
    """skip_offload decorator behaviour."""

    def setUp(self) -> None:
        self._mock_ctx = _MockSessionContext()
        self._mock_ctx.start()

    def tearDown(self) -> None:
        self._mock_ctx.stop()

    def test_non_callable_raises_type_error(self) -> None:
        """skip_offload on a non-callable should raise TypeError."""
        with self.assertRaises(TypeError):
            skip_offload(42)

    def test_outside_session_calls_function_directly(self) -> None:
        """When called outside a session, skip_offload should call the function directly."""
        @skip_offload
        def my_fn(x):
            return x + 1

        result = my_fn(torch.tensor(5.0))
        self.assertEqual(result, torch.tensor(6.0))

    def test_inside_session_uses_opaque_op(self) -> None:
        """When called inside a session, skip_offload should go through execute_opaque_op."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        call_count = 0

        @skip_offload
        def my_fn(x):
            nonlocal call_count
            call_count += 1
            return x.sin()

        with session:
            result = my_fn(torch.tensor(0.5))
        self.assertAlmostEqual(result.item(), 0.479, places=3)
        self.assertEqual(call_count, 1)

    def test_function_name_preserved(self) -> None:
        """skip_offload should preserve the original function name."""
        @skip_offload
        def my_custom_function(x):
            return x + 1

        self.assertEqual(my_custom_function.__name__, "my_custom_function")
        self.assertIn("my_custom_function", my_custom_function.__wrapped__.__name__)

    def test_multiple_skip_offload_in_one_session(self) -> None:
        """Multiple skip_offload-decorated functions should work in one session."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        @skip_offload
        def fn1(x):
            return x.sin()

        @skip_offload
        def fn2(x):
            return x.cos()

        with session:
            t = torch.randn(4)
            a = fn1(t)
            b = fn2(t)
            loss = (a + b).sum()
            loss.backward()

    def test_skip_offload_trace_has_virtual_ops(self) -> None:
        """The warmup trace should contain virtual forward+backward ops for skip_offload regions."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        @skip_offload
        def region(x):
            return x.sin()

        with session:
            t = torch.randn(4, requires_grad=True)
            y = region(t)
            loss = y.sum()
            loss.backward()
            ops = session.executor._ops  # pylint: disable=protected-access
            # Should have virtual ops for the skip_offload region
            fwd_ops = [op for op in ops if "region_fwd" in op.name]
            bwd_ops = [op for op in ops if "region_bwd" in op.name]
            self.assertGreaterEqual(len(fwd_ops), 1)
            self.assertGreaterEqual(len(bwd_ops), 1)

    def test_skip_offload_nested(self) -> None:
        """Nested skip_offload regions should work."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        @skip_offload
        def inner(x):
            return x.sin()

        @skip_offload
        def outer(x):
            return inner(x).cos()

        with session:
            t = torch.randn(4, requires_grad=True)
            y = outer(t)
            loss = y.sum()
            loss.backward()
            # Access _ops while still inside the session (WarmupExecutor)
            ops = session.executor._ops  # pylint: disable=protected-access
            fwd_ops = [op for op in ops if "_fwd" in op.name]
            # outer and inner may be merged or separate depending on nesting
            self.assertGreaterEqual(len(fwd_ops), 1)

    def test_skip_offload_preserves_gradient_flow(self) -> None:
        """Gradients should flow correctly through skip_offload regions."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        @skip_offload
        def swish(x):
            return x * torch.sigmoid(x)

        x = torch.randn(4, requires_grad=True)
        # Warmup
        with session:
            y = swish(x)
            loss = y.sum()
            loss.backward()
        grad_warmup = x.grad.clone()

        # Reference
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = x_ref * torch.sigmoid(x_ref)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        torch.testing.assert_close(grad_warmup, x_ref.grad, rtol=1e-4, atol=1e-6)

    def test_skip_offload_replay_preserves_gradients(self) -> None:
        """Replay should also preserve gradients through skip_offload regions."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        @skip_offload
        def swish(x):
            return x * torch.sigmoid(x)

        # Warmup
        x = torch.randn(4, requires_grad=True)
        with session:
            y = swish(x)
            loss = y.sum()
            loss.backward()

        # Replay
        x2 = torch.randn(4, requires_grad=True)
        with session:
            y2 = swish(x2)
            loss2 = y2.sum()
            loss2.backward()

        # Gradients should be non-None and reasonable
        self.assertIsNotNone(x2.grad)
        self.assertFalse(torch.isnan(x2.grad).any())
        self.assertFalse(torch.isinf(x2.grad).any())

        # Reference: call the same function outside a session
        x_ref = x2.detach().clone().requires_grad_(True)
        y_ref = x_ref * torch.sigmoid(x_ref)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        torch.testing.assert_close(x2.grad, x_ref.grad, rtol=1e-4, atol=1e-6)


class TestSkipOffloadWarmupReplayConsistency(unittest.TestCase):
    """Consistency between warmup and replay for skip_offload regions."""

    def setUp(self) -> None:
        self._mock_ctx = _MockSessionContext()
        self._mock_ctx.start()

    def tearDown(self) -> None:
        self._mock_ctx.stop()

    def test_replay_output_matches_warmup(self) -> None:
        """Replay output should match warmup output for the same input."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        @skip_offload
        def region(x):
            return x.sin().cos()

        x = torch.randn(4)
        # Warmup
        with session:
            y1 = region(x)

        # Replay
        with session:
            y2 = region(x)

        torch.testing.assert_close(y2, y1, rtol=1e-5, atol=1e-7)

    def test_replay_more_ops_in_skip_region_does_not_crash(self) -> None:
        """If the replay has more internal ops than warmup, it should not crash."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        use_many_ops = False

        @skip_offload
        def branch_region(x):
            if use_many_ops:
                for _ in range(10):
                    x = x.sin()
                return x
            return x.sin()

        x = torch.randn(4, requires_grad=True)

        # Warmup: 1 internal op
        with session:
            y = branch_region(x)
            loss = y.sum()
            loss.backward()

        # Replay: 10 internal ops — should not crash
        use_many_ops = True
        x2 = torch.randn(4, requires_grad=True)
        with session:
            y = branch_region(x2)
            loss = y.sum()
            loss.backward()

        # Reference: call the function directly (outside session) with the same flag
        x_ref = x2.detach().clone().requires_grad_(True)
        y_ref = branch_region(x_ref)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # Output and gradients should match the direct computation
        torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(x2.grad, x_ref.grad, rtol=1e-4, atol=1e-6)

    def test_replay_fewer_ops_in_skip_region_does_not_desync(self) -> None:
        """If replay has fewer internal ops, op_idx should still reach the expected end."""
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))

        use_two_ops = True

        @skip_offload
        def branch_region(x):
            if use_two_ops:
                t = x.cos()
                return t.sin()
            return x.sin()

        x = torch.randn(4, requires_grad=True)

        # Warmup: 2 internal ops
        with session:
            y = branch_region(x)
            loss = y.sum()
            loss.backward()

        # Replay: 1 internal op
        use_two_ops = False
        x2 = torch.randn(4, requires_grad=True)
        with session:
            y = branch_region(x2)
            loss = y.sum()
            loss.backward()

        # Reference: call the function directly (outside session) with the same flag
        x_ref = x2.detach().clone().requires_grad_(True)
        y_ref = branch_region(x_ref)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # Output and gradients should match the direct computation
        torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(x2.grad, x_ref.grad, rtol=1e-4, atol=1e-6)

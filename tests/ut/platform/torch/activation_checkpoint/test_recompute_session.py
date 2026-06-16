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
"""Unit tests for PyTorch recompute session (dx/dw split support)."""
import os
import unittest
import uuid

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform.torch.activation_checkpoint.recompute_session import (
    _clear_recompute_session,
    _recompute_session_ctx,
    _recompute_session_handles,
    checkpoint_with_session,
)


class _CountingModule(torch.nn.Module):
    """Simple module that counts how many times it is called."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.ones_(self.linear.weight)
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return self.linear(x).sin()


class _TripleCountingModule(torch.nn.Module):
    """Module with three independent checkpointed sub-modules."""

    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Linear(4, 4, bias=False)
        self.block2 = torch.nn.Linear(4, 4, bias=False)
        self.block3 = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.ones_(self.block1.weight)
        torch.nn.init.ones_(self.block2.weight)
        torch.nn.init.ones_(self.block3.weight)

    def forward(self, x):
        x1 = checkpoint_with_session(self.block1, x)
        x2 = checkpoint_with_session(self.block2, x1)
        x3 = checkpoint_with_session(self.block3, x2)
        return x3.sin().sum()


class TestDxDwSharedRecompute(unittest.TestCase):
    """Test that dx/dw split shares one recompute under the same session."""

    def test_single_recompute_in_dxdw_split(self):
        """Same checkpoint block in two backward passes under one session
        should trigger only one recompute (call_count == 2: forward + recompute)."""
        mod = _CountingModule()
        x = torch.randn(2, 4, requires_grad=True)
        session_id = uuid.uuid4().hex

        # Forward under session to register the handle
        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out = checkpoint_with_session(mod, x)

        # First backward (dx) — triggers recompute
        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out.sum().backward(retain_graph=True)

        self.assertEqual(mod.call_count, 2, "Should be forward(1) + recompute(1) = 2")

        # Second backward (dw) — reuses cached recompute
        with _recompute_session_ctx(session_id, retain_on_unpack=False):
            out.sum().backward()

        # Still 2 — no additional recompute
        self.assertEqual(mod.call_count, 2, "dw should reuse cached recompute, count stays 2")

        _clear_recompute_session(session_id)


class TestRetainAndClearLifecycle(unittest.TestCase):
    """Test retain_on_unpack and clear_recompute_session lifecycle."""

    def test_retain_then_clear(self):
        """After dx retains and dw consumes, clear should remove all session state."""
        mod = _CountingModule()
        x = torch.randn(2, 4, requires_grad=True)
        session_id = uuid.uuid4().hex

        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out = checkpoint_with_session(mod, x)

        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out.sum().backward(retain_graph=True)

        with _recompute_session_ctx(session_id, retain_on_unpack=False):
            out.sum().backward()

        _clear_recompute_session(session_id)

        # After clear, the session data should be gone from the global registry
        self.assertNotIn(session_id, _recompute_session_handles)


class TestNoSessionBackwardCompatibility(unittest.TestCase):
    """Test that checkpoint_with_session falls back to native PyTorch when no session is active."""

    def test_no_session_matches_native_checkpoint(self):
        """Without session, checkpoint_with_session should behave like native checkpoint."""
        mod = _CountingModule()
        x = torch.randn(2, 4, requires_grad=True)

        out = checkpoint_with_session(mod, x)
        out.sum().backward()

        # Native checkpoint behavior: forward(1) + recompute(1) = 2
        self.assertEqual(mod.call_count, 2)

        # Gradient should be computable
        self.assertIsNotNone(x.grad)


class TestExceptionCleanup(unittest.TestCase):
    """Test that explicit clear after exception leaves no residual session state."""

    def test_clear_after_exception_no_residual(self):
        """After an exception during dx, clear should allow a fresh session with the same id."""
        mod = _CountingModule()
        x = torch.randn(2, 4, requires_grad=True)
        session_id = uuid.uuid4().hex

        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out = checkpoint_with_session(mod, x)

        try:
            with _recompute_session_ctx(session_id, retain_on_unpack=True):
                out.sum().backward(retain_graph=True)
                raise RuntimeError("Simulated dx failure")
        except RuntimeError:
            pass
        finally:
            _clear_recompute_session(session_id)

        # Fresh session with same session_id should work cleanly
        mod2 = _CountingModule()
        x2 = torch.randn(2, 4, requires_grad=True)
        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out2 = checkpoint_with_session(mod2, x2)
        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out2.sum().backward(retain_graph=True)
        with _recompute_session_ctx(session_id, retain_on_unpack=False):
            out2.sum().backward()
        _clear_recompute_session(session_id)

        self.assertEqual(mod2.call_count, 2)


class TestIncompleteConsumption(unittest.TestCase):
    """Test that clear cleans up even when not all checkpoint blocks were unpacked."""

    def test_clear_cleans_all_blocks(self):
        """After clear, all three blocks' session state should be gone
        even if only one was unpacked during backward."""
        mod = _TripleCountingModule()
        x = torch.randn(2, 4, requires_grad=True)
        session_id = uuid.uuid4().hex

        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out = mod(x)

        # Only do one backward — not all blocks may be fully consumed
        with _recompute_session_ctx(session_id, retain_on_unpack=True):
            out.backward(retain_graph=True)

        _clear_recompute_session(session_id)

        self.assertNotIn(session_id, _recompute_session_handles)


if __name__ == "__main__":
    unittest.main()

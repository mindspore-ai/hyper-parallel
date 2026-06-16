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
"""Unit tests for ActivationTracker."""

from __future__ import annotations

import unittest

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup.tracker import ActivationTracker
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import PhysicalBuffer


class TestActivationTracker(unittest.TestCase):
    """ActivationTracker: storage identity and lifecycle."""

    def setUp(self) -> None:
        self.tracker = ActivationTracker()

    def test_initial_state(self) -> None:
        self.assertEqual(len(self.tracker._activation_sids), 0)  # pylint: disable=protected-access
        self.assertEqual(len(self.tracker._storage_sizes), 0)

    def test_get_activation_sid_returns_none_for_unknown_tensor(self) -> None:
        t = torch.randn(4, 4)
        sid = self.tracker.get_activation_sid(t)
        self.assertIsNone(sid)

    def test_get_activation_sid_returns_none_for_cpu_tensor(self) -> None:
        """CPU tensors are excluded from activation tracking."""
        t = torch.randn(4, 4)
        sid = self.tracker.get_activation_sid(t)
        self.assertIsNone(sid)

    def test_get_activation_sid_returns_sid_for_shadow_tensor(self) -> None:
        """ShadowTensor should return its storage_id directly."""
        t = torch.randn(2, 3)
        buf = PhysicalBuffer(device=t.device)
        shadow = ShadowTensor(t, buf, storage_id=42)
        sid = self.tracker.get_activation_sid(shadow)
        self.assertEqual(sid, 42)

    def test_register_op_activations_multiple_outputs(self) -> None:
        """Multiple output tensors should each get storage IDs, but CPU tensors are excluded."""
        t1 = torch.randn(2, 2)
        t2 = torch.randn(3, 3)
        self.tracker.register_op_activations([], [t1, t2])
        # Both are CPU → not tracked
        self.assertIsNone(self.tracker.get_activation_sid(t1))
        self.assertIsNone(self.tracker.get_activation_sid(t2))

    def test_register_op_activations_tracks_new_outputs(self) -> None:
        """New output storages should get IDs and be tracked."""
        t = torch.randn(4, 4)
        self.tracker.register_op_activations([], [t])
        # Use _ensure_id since get_activation_sid excludes CPU tensors by design
        sid = self.tracker._ensure_id(t)  # pylint: disable=protected-access
        self.assertIsNotNone(sid)
        self.assertIn(sid, self.tracker._activation_sids)  # pylint: disable=protected-access

    def test_register_op_activations_skips_inputs(self) -> None:
        """Outputs that share storage with inputs should not be newly registered."""
        inp = torch.randn(4, 4)
        out = inp.clone()  # new storage
        self.tracker.register_op_activations([inp], [out])
        inp_sid = self.tracker._ensure_id(inp)  # pylint: disable=protected-access
        out_sid = self.tracker._ensure_id(out)  # pylint: disable=protected-access
        self.assertIsNotNone(inp_sid)
        self.assertIsNotNone(out_sid)
        self.assertNotEqual(inp_sid, out_sid)
        # inp is not an activation (it was an input)
        self.assertNotIn(inp_sid, self.tracker._activation_sids)  # pylint: disable=protected-access
        # out should be an activation (new storage)
        self.assertIn(out_sid, self.tracker._activation_sids)  # pylint: disable=protected-access

    def test_storage_sizes_accumulated(self) -> None:
        """Storage sizes should accumulate across registrations."""
        t1 = torch.randn(8, 8)
        t2 = torch.randn(16, 16)
        self.tracker.register_op_activations([], [t1])
        self.tracker.register_op_activations([], [t2])

        sizes = self.tracker.storage_sizes
        # At least 2 entries
        self.assertGreaterEqual(len(sizes), 2)

    def test_clear_activations_resets_state(self) -> None:
        """clear_activations should clear activation set and storage sizes."""
        t = torch.randn(4, 4)
        self.tracker.register_op_activations([], [t])
        self.tracker.clear_activations()
        self.assertEqual(len(self.tracker._activation_sids), 0)  # pylint: disable=protected-access
        self.assertEqual(len(self.tracker._storage_sizes), 0)

    def test_multiple_ops_with_shared_storage(self) -> None:
        """When the same storage appears in multiple ops, it should keep the same sid."""
        t = torch.randn(4, 4)
        self.tracker.register_op_activations([], [t])
        sid1 = self.tracker._ensure_id(t)  # pylint: disable=protected-access
        # Re-register (same tensor, same storage)
        self.tracker.register_op_activations([], [t])
        sid2 = self.tracker._ensure_id(t)  # pylint: disable=protected-access
        self.assertEqual(sid1, sid2)

    def test_get_activation_sid_handles_meta_tensor(self) -> None:
        """Meta tensors or tensors without untyped_storage should return None."""
        meta = torch.empty(2, 2, device="meta")
        sid = self.tracker.get_activation_sid(meta)
        self.assertIsNone(sid)

    def test_repr(self) -> None:
        r = repr(self.tracker)
        self.assertIn("ActivationTracker", r)

    def test_get_activation_sid_for_quantized_tensor(self) -> None:
        """If untyped_storage() raises, should return None."""
        # Use a mocked tensor that raises on untyped_storage
        class _NoStorageTensor:
            """Fake tensor without storage."""

            device = torch.device("cpu")

            def untyped_storage(self):
                raise RuntimeError("no storage")

        t = _NoStorageTensor()
        self.assertIsNone(self.tracker._ensure_id(t))  # pylint: disable=protected-access

    def test_get_activation_sid_does_not_return_non_activation_sid(self) -> None:
        """Even if storage_id exists in _storage_tracker, it should only return if
        it's in _activation_sids."""
        t = torch.randn(4, 4)
        # Manually add to storage_tracker without registering as activation
        sid = self.tracker._ensure_id(t)  # pylint: disable=protected-access
        self.assertIsNotNone(sid)
        # get_activation_sid should still return None because it's not an activation
        self.assertIsNone(self.tracker.get_activation_sid(t))

    def test_register_op_activations_with_shared_storage(self) -> None:
        """When input and output share storage (e.g., in-place op), output should
        not be registered as a new activation."""
        # We can't easily create two tensors sharing storage, but we can test
        # the logic: if output sid is in input_sids, it's skipped.
        t = torch.randn(4, 4)
        sid = self.tracker._ensure_id(t)  # pylint: disable=protected-access
        self.assertIsNotNone(sid)
        # Now register with the same tensor as both input and output
        self.tracker.register_op_activations([t], [t])
        # t's sid should NOT be in activation_sids
        self.assertNotIn(sid, self.tracker._activation_sids)  # pylint: disable=protected-access

    def test_get_activation_sid_returns_none_for_no_untyped_storage_attr(self) -> None:
        """A tensor-like object without untyped_storage attr should return None."""
        class _NoStorage:
            """Fake tensor with no untyped_storage attribute."""
            device = torch.device("meta")
            type = "fake"

        t = _NoStorage()
        sid = self.tracker.get_activation_sid(t)
        self.assertIsNone(sid)

    def test_get_activation_sid_returns_none_when_untyped_storage_raises(self) -> None:
        """A tensor whose untyped_storage() raises should return None."""
        class _RaisesOnStorage:
            """Fake tensor that raises on untyped_storage()."""
            device = torch.device("meta")

            def untyped_storage(self) -> None:
                raise RuntimeError("no storage")

        t = _RaisesOnStorage()
        sid = self.tracker.get_activation_sid(t)
        self.assertIsNone(sid)

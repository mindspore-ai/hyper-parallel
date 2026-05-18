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
"""Unit tests for :func:`hyper_parallel.core.dtensor.random.manual_seed` and helpers."""
from __future__ import annotations

import os
import unittest
import warnings
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.random import is_rng_supported_mesh, manual_seed
from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER


class TestIsRngSupportedMesh(unittest.TestCase):
    """Tests for :func:`is_rng_supported_mesh`."""

    def test_cpu_mesh_returns_false(self):
        """CPU ``DeviceMesh`` is treated as unsupported for DTensor RNG."""
        mesh = MagicMock()
        mesh.device_type = "cpu"
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self.assertFalse(is_rng_supported_mesh(mesh))
        self.assertTrue(any("cpu" in str(w.message).lower() for w in rec))

    @patch("hyper_parallel.core.dtensor.random.platform.get_device_handle")
    def test_returns_true_when_handle_has_set_rng_state(self, mock_get_handle):
        """When the platform device handle exposes ``set_rng_state``, RNG is supported."""
        mock_get_handle.return_value = MagicMock(spec=["set_rng_state"])
        self.assertTrue(is_rng_supported_mesh(None))

    @patch("hyper_parallel.core.dtensor.random.platform.get_device_handle")
    def test_returns_false_without_set_rng_state(self, mock_get_handle):
        """Missing ``set_rng_state`` on the handle yields ``False`` for a non-CPU mesh."""

        class _HandleWithoutRngState:
            """Minimal device handle without ``set_rng_state``."""

        mock_get_handle.return_value = _HandleWithoutRngState()
        mesh = MagicMock()
        mesh.device_type = "npu"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.assertFalse(is_rng_supported_mesh(mesh))


class TestManualSeed(unittest.TestCase):
    """Tests for :func:`manual_seed` (mocked backend; no real device mesh required)."""

    def tearDown(self) -> None:
        _OP_DISPATCHER._rng_tracker = None

    def test_no_op_on_cpu_mesh_does_not_touch_dispatcher_seed(self):
        """Unsupported mesh: early return; no ``platform.manual_seed``."""
        mesh = MagicMock()
        mesh.device_type = "cpu"
        with patch("hyper_parallel.core.dtensor.random.platform.manual_seed") as mock_ms:
            manual_seed(42, mesh)
        mock_ms.assert_not_called()

    @patch("hyper_parallel.core.dtensor.random.OffsetBasedRNGTracker")
    @patch("hyper_parallel.core.dtensor.random.is_rng_supported_mesh", return_value=True)
    def test_raises_when_current_rank_not_in_mesh(self, mock_is_rng, mock_tracker_cls):
        """``get_coordinate()`` is ``None`` → ``RuntimeError`` (after tracker install)."""
        del mock_is_rng
        mock_tracker_cls.return_value = MagicMock()
        mesh = MagicMock()
        mesh.device_type = "npu"
        mesh.get_coordinate.return_value = None
        with self.assertRaisesRegex(RuntimeError, "manual_seed requires"):
            manual_seed(7, mesh)
        mock_tracker_cls.assert_called_once_with(run_state_sync=False)

    @patch("hyper_parallel.core.dtensor.random.OffsetBasedRNGTracker")
    @patch("hyper_parallel.core.dtensor.random.platform.manual_seed")
    @patch("hyper_parallel.core.dtensor.random.is_rng_supported_mesh", return_value=True)
    def test_calls_platform_manual_seed_with_seed(
        self, mock_is_rng, mock_platform_manual_seed, mock_tracker_cls
    ):
        """Happy path: install tracker once, then ``platform.manual_seed(seed)``."""
        del mock_is_rng
        mock_tracker_cls.return_value = MagicMock()
        mesh = MagicMock()
        mesh.device_type = "npu"
        mesh.get_coordinate.return_value = (0,)
        manual_seed(99_001, mesh)
        mock_platform_manual_seed.assert_called_once_with(99_001)
        self.assertIs(_OP_DISPATCHER._rng_tracker, mock_tracker_cls.return_value)

    @patch("hyper_parallel.core.dtensor.random.OffsetBasedRNGTracker")
    @patch("hyper_parallel.core.dtensor.random.platform.manual_seed")
    @patch("hyper_parallel.core.dtensor.random.is_rng_supported_mesh", return_value=True)
    def test_tracker_constructed_only_once_across_two_calls(
        self, mock_is_rng, mock_platform_manual_seed, mock_tracker_cls
    ):
        """Second ``manual_seed`` reuses existing ``_rng_tracker`` (no second construction)."""
        del mock_is_rng
        mock_tracker_cls.return_value = MagicMock()
        mesh = MagicMock()
        mesh.device_type = "npu"
        mesh.get_coordinate.return_value = (0,)
        manual_seed(1, mesh)
        manual_seed(2, mesh)
        self.assertEqual(mock_tracker_cls.call_count, 1)
        self.assertEqual(mock_platform_manual_seed.call_count, 2)


if __name__ == "__main__":
    unittest.main()

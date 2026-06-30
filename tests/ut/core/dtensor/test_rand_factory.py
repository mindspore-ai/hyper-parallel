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
"""Unit tests for DTensor rand/randn factory helpers."""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor import dtensor as dtensor_mod
from hyper_parallel.core.dtensor.dtensor import rand, randn
from hyper_parallel.core.dtensor.placement_types import Shard


class TestRandFactory(unittest.TestCase):
    """Tests for :func:`rand` and :func:`randn`."""

    def test_rand_uses_rng_tracker_when_supported(self):
        """rand() should initialize inside OffsetBasedRNGTracker when mesh supports RNG."""
        mesh = MagicMock()
        placements = [Shard(0)]
        tracker = MagicMock()
        tracker._distribute_region.return_value.__enter__ = MagicMock(return_value=None)
        tracker._distribute_region.return_value.__exit__ = MagicMock(return_value=False)
        dispatcher = MagicMock()
        dispatcher._rng_tracker = tracker
        layout = MagicMock()
        layout.placements = (Shard(0),)

        with patch.object(dtensor_mod, "compute_local_shape_and_global_offset", return_value=(2, 4)), \
             patch.object(dtensor_mod, "_build_layout", return_value=layout), \
             patch("hyper_parallel.core.dtensor.random.is_rng_supported_mesh", return_value=True), \
             patch("hyper_parallel.core.shard._op_dispatch._OP_DISPATCHER", dispatcher), \
             patch.object(dtensor_mod.platform, "rand", return_value=torch.randn(2, 4)) as mock_rand, \
             patch.object(dtensor_mod, "DTensor") as mock_dtensor:
            rand((4, 4), mesh, placements, dtype=torch.float32)
        mock_rand.assert_called_once_with((2, 4), dtype=torch.float32)
        tracker._distribute_region.assert_called_once()
        mock_dtensor.from_local.assert_called_once()

    def test_randn_falls_back_without_rng_support(self):
        """randn() should call platform.randn directly when RNG mesh is unsupported."""
        mesh = MagicMock()
        placements = [Shard(0)]

        with patch.object(dtensor_mod, "compute_local_shape_and_global_offset", return_value=(2, 4)), \
             patch("hyper_parallel.core.dtensor.random.is_rng_supported_mesh", return_value=False), \
             patch.object(dtensor_mod.platform, "randn", return_value=torch.randn(2, 4)) as mock_randn, \
             patch.object(dtensor_mod, "DTensor") as mock_dtensor:
            randn((4, 4), mesh, placements)
        mock_randn.assert_called_once_with((2, 4))
        mock_dtensor.from_local.assert_called_once()


if __name__ == "__main__":
    unittest.main()

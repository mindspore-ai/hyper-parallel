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
"""Unit tests for :meth:`DTensor.from_local` with ``run_check=True``."""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.dtensor import _from_local_utils


class TestFromLocalRunCheck(unittest.TestCase):
    """Tests for ``DTensor.from_local(..., run_check=True)``."""

    def test_run_check_false_skips_validation(self):
        local = torch.ones(2, 2)
        mesh = MagicMock()
        with patch.object(_from_local_utils, "run_from_local_checks") as mock_checks:
            DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
        mock_checks.assert_not_called()

    def test_run_check_true_invokes_validation(self):
        """run_check=True should invoke run_from_local_checks before wrapping."""
        local = torch.ones(2, 2)
        mesh = MagicMock()
        layout = MagicMock()
        layout.placements = (Shard(0),)
        with patch("hyper_parallel.core.dtensor.dtensor._build_layout", return_value=layout), \
             patch.object(_from_local_utils, "run_from_local_checks") as mock_checks:
            DTensor.from_local(local, mesh, [Shard(0)], run_check=True)
        mock_checks.assert_called_once()
        self.assertIs(mock_checks.call_args.args[0], local)
        self.assertIs(mock_checks.call_args.args[1], mesh)
        self.assertEqual(mock_checks.call_args.args[2], layout.placements)

    def test_check_tensor_meta_raises_on_mismatch(self):
        """Inconsistent metadata across ranks should raise ValueError."""
        local = torch.ones(2, 2)
        gathered = [
            {"dtype": "torch.float32", "requires_grad": False, "shape": (2, 2), "stride": (2, 1)},
            {"dtype": "torch.float32", "requires_grad": False, "shape": (3, 2), "stride": (2, 1)},
        ]

        def fake_all_gather(out_list, obj, group=None):
            del obj
            out_list[:] = gathered

        with patch("hyper_parallel.core.dtensor._from_local_utils.platform.all_gather_object", side_effect=fake_all_gather):
            with self.assertRaises(ValueError):
                _from_local_utils.check_tensor_meta(
                    local,
                    group=MagicMock(),
                    group_size=2,
                    check_shape_stride=True,
                )

    def test_run_from_local_checks_broadcasts_replicate_dims(self):
        local = torch.ones(2, 2)
        mesh = MagicMock()
        placements = (Replicate(), Shard(0))
        with patch.object(_from_local_utils, "_mesh_check_group", return_value=(MagicMock(), 2)), \
             patch.object(_from_local_utils, "check_tensor_meta") as mock_meta, \
             patch.object(_from_local_utils, "mesh_broadcast") as mock_broadcast:
            _from_local_utils.run_from_local_checks(local, mesh, placements)
        mock_meta.assert_called_once()
        mock_broadcast.assert_called_once_with(local, mesh, 0)


if __name__ == "__main__":
    unittest.main()

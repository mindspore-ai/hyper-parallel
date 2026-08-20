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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.api`."""
# pylint: disable=wrong-import-position
import importlib
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.api as api_mod
import hyper_parallel.core.distributed_checkpoint.standard_planner as planner_mod

importlib.reload(planner_mod)
importlib.reload(api_mod)

from hyper_parallel.core.distributed_checkpoint.api import (
    _gather_from_all_ranks,
    _raise_if_stage_failed,
    load,
    save,
)
from hyper_parallel.core.distributed_checkpoint.storage import METADATA_FILE_NAME


class TestApi(unittest.TestCase):
    """Tests for distributed checkpoint save/load API."""

    def setUp(self) -> None:
        """Reset the selected platform and checkpoint planner cache."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        importlib.reload(api_mod)
        planner_mod.StandardSavePlanner._cached_save_result.clear()

    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    def test_save_requires_checkpoint_id_or_writer(self, mock_barrier):
        """
        Feature: save input validation.
        Description: Call save without checkpoint_id or storage_writer.
        Expectation: ValueError is raised.
        """
        with self.assertRaises(ValueError) as ctx:
            save({"w": torch.zeros(1)}, no_dist=True)
        self.assertIn("checkpoint_id", str(ctx.exception))
        mock_barrier.assert_not_called()

    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.get_rank", return_value=0)
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    def test_save_load_roundtrip_no_dist(self, mock_barrier, mock_rank, mock_world_size):
        """
        Feature: save and load round-trip in single-process mode.
        Description: Save nested state dict then load into a fresh dict with no_dist=True.
        Expectation: Tensor and byte leaves match after load.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            weight = torch.nn.Parameter(torch.randn(3, 4))
            step = 7
            save({"weight": weight, "step": step}, checkpoint_id=tmpdir, no_dist=True)
            rank_meta = Path(tmpdir, f"0{METADATA_FILE_NAME}")
            self.assertTrue(rank_meta.exists() or Path(tmpdir, METADATA_FILE_NAME).exists())

            loaded = {"weight": torch.zeros(3, 4), "step": None}
            load(loaded, checkpoint_id=tmpdir, no_dist=True)
            torch.testing.assert_close(loaded["weight"], weight)
            self.assertEqual(loaded["step"], step)
        mock_barrier.assert_not_called()
        mock_rank.assert_not_called()
        mock_world_size.assert_not_called()

    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.get_rank", return_value=0)
    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.barrier")
    def test_save_returns_metadata_with_tensor_entries(self, mock_barrier, mock_rank, mock_world_size):
        """
        Feature: save return value.
        Description: Save a single-parameter checkpoint with no_dist=True.
        Expectation: Returned Metadata lists the weight FQN in state_dict_metadata.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = save({"weight": torch.zeros(2, 2)}, checkpoint_id=tmpdir, no_dist=True)
            self.assertIn("weight", metadata.state_dict_metadata)
        mock_barrier.assert_not_called()
        mock_rank.assert_not_called()
        mock_world_size.assert_not_called()

    def test_gather_from_all_ranks_single_process(self):
        """
        Feature: _gather_from_all_ranks without collectives.
        Description: use_collectives=False with world_size implied as 1.
        Expectation: Returns a one-element list containing the local object.
        """
        result = _gather_from_all_ranks({"plan": 1}, world_size=1, use_collectives=False)
        self.assertEqual(result, [{"plan": 1}])

    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.all_gather_object")
    def test_gather_from_all_ranks_uses_collectives(self, mock_all_gather):
        """
        Feature: _gather_from_all_ranks collective path.
        Description: use_collectives=True with world_size > 1.
        Expectation: platform.all_gather_object is invoked and its output is returned.
        """
        expected = [{"a": 1}, {"a": 2}]

        def _gather_side_effect(out, local_obj):
            if local_obj is None:
                out[:] = [None, None]
            else:
                out[:] = [pickle.dumps(value) for value in expected]

        mock_all_gather.side_effect = _gather_side_effect
        result = _gather_from_all_ranks({"a": 1}, world_size=2, use_collectives=True)
        self.assertEqual(mock_all_gather.call_count, 3)
        self.assertEqual(result, expected)

    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.all_gather_object")
    def test_gather_propagates_payload_serialization_failure(self, mock_all_gather):
        """
        Feature: Collective payload serialization failure propagation.
        Description: The local rank cannot pickle its plan before collective exchange.
        Expectation: Every rank can fail in the status collective before payload exchange.
        """

        def _gather_side_effect(out, local_error):
            out[:] = [local_error, None]

        mock_all_gather.side_effect = _gather_side_effect
        with self.assertRaisesRegex(RuntimeError, "payload serialization"):
            _gather_from_all_ranks(lambda: None, world_size=2, use_collectives=True)
        mock_all_gather.assert_called_once()

    @patch("hyper_parallel.core.distributed_checkpoint.api.platform.all_gather_object")
    def test_stage_failure_from_peer_is_raised_on_healthy_rank(self, mock_all_gather):
        """
        Feature: Distributed checkpoint phase failure propagation.
        Description: A peer reports setup failure while the local rank succeeds.
        Expectation: The healthy rank raises before entering the next checkpoint phase.
        """

        def _gather_side_effect(out, local_error):
            self.assertIsNone(local_error)
            out[:] = ["rank 0 failed", None]

        mock_all_gather.side_effect = _gather_side_effect
        with self.assertRaisesRegex(RuntimeError, "planning setup"):
            _raise_if_stage_failed(None, "planning setup", 2, True)


if __name__ == "__main__":
    unittest.main()

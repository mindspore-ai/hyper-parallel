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
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock, patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.api as api_mod
import hyper_parallel.core.distributed_checkpoint.standard_planner as planner_mod

importlib.reload(planner_mod)
importlib.reload(api_mod)

from hyper_parallel.core.distributed_checkpoint.api import (
    load,
    save,
)
from hyper_parallel.core.distributed_checkpoint.util import all_gather_object
from hyper_parallel.core.distributed_checkpoint.storage import METADATA_FILE_NAME


class TestApi(unittest.TestCase):
    """Tests for distributed checkpoint save/load API."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        importlib.reload(api_mod)
        planner_mod.StandardSavePlanner.cached_save_result.clear()

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
        mock_barrier.assert_called()
        mock_rank.assert_called()
        mock_world_size.assert_called()

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
        mock_barrier.assert_called_once()
        mock_rank.assert_called()
        mock_world_size.assert_called()

    def test_gather_from_all_ranks_single_process(self):
        """
        Feature: all_gather_object without collectives.
        Description: use_collectives=False with world_size implied as 1.
        Expectation: Returns a one-element list containing the local object.
        """
        result = all_gather_object({"plan": 1}, world_size=1, use_collectives=False)
        self.assertEqual(result, [{"plan": 1}])

    @patch("hyper_parallel.core.distributed_checkpoint.util.platform.all_gather_object")
    def test_gather_from_all_ranks_uses_collectives(self, mock_all_gather):
        """
        Feature: all_gather_object collective path.
        Description: use_collectives=True with world_size > 1.
        Expectation: platform.all_gather_object is invoked and its output is returned.
        """
        expected = [{"a": 1}, {"a": 2}]

        def gather_side_effect(out, local_obj):
            del local_obj
            out[0] = expected[0]
            out[1] = expected[1]

        mock_all_gather.side_effect = gather_side_effect
        result = all_gather_object({"a": 1}, world_size=2, use_collectives=True)
        mock_all_gather.assert_called_once()
        self.assertEqual(result, expected)


class TestCreatePersistProcess(unittest.TestCase):
    """Tests for the persist mode ``async_save`` picks for a given set of arguments.

    All three modes end up writing a correct checkpoint, so an end-to-end ST cannot tell them
    apart - a ``use_gloo=True`` that silently fell back to storage communication would still
    pass one. These tests pin which target the child process actually runs, and with which
    communication flag.
    """

    _MASTER_ENV = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}

    def setUp(self) -> None:
        """Rebuild the api module so the recorded targets are the objects it holds."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        importlib.reload(api_mod)

    @staticmethod
    def _create(
        no_dist: bool = False,
        use_collectives: bool = True,
        use_gloo: bool = False,
        world_size: int = 4,
        env: Optional[dict] = None,
    ) -> Any:
        """Build the persist process with a recording ``mp.Process`` and return that recorder."""
        recorder = Mock(name="Process")
        with patch("hyper_parallel.core.distributed_checkpoint.api.mp.Process", recorder), \
                patch("hyper_parallel.core.distributed_checkpoint.api.platform.get_world_size",
                      return_value=world_size), \
                patch("hyper_parallel.core.distributed_checkpoint.api.platform.get_rank", return_value=0), \
                patch.dict(os.environ, env if env is not None else TestCreatePersistProcess._MASTER_ENV,
                           clear=True):
            api_mod._create_persist_process(
                None,
                {},
                Path("ckpt"),
                None,
                None,
                no_dist,
                use_collectives,
                use_gloo,
            )
        return recorder

    def _assert_plain_persist(self, recorder: Any, use_storage_comm: bool) -> None:
        """Assert the child runs the plain persist target with the expected comm flag."""
        kwargs = recorder.call_args.kwargs
        self.assertIs(kwargs["target"], api_mod.execute_async_persist)
        self.assertEqual(kwargs["args"][-1], use_storage_comm)

    def test_without_collectives_the_child_runs_without_comm(self):
        """
        Feature: persist mode for use_collectives=False.
        Description: Ask for an async save that does not coordinate with the other ranks.
        Expectation: The plain persist target runs with storage communication switched off.
        """
        self._assert_plain_persist(self._create(use_collectives=False), use_storage_comm=False)

    def test_with_collectives_the_child_exchanges_results_through_storage(self):
        """
        Feature: persist mode for use_collectives=True without gloo, the default.
        Description: Ask for a coordinated async save on four ranks.
        Expectation: The plain persist target runs with storage communication switched on.
        """
        self._assert_plain_persist(self._create(), use_storage_comm=True)

    def test_use_gloo_runs_the_gloo_target_with_the_rendezvous(self):
        """
        Feature: persist mode for use_gloo=True.
        Description: Ask for a coordinated async save over gloo, with the rendezvous in the env.
        Expectation: The gloo target runs, carrying this rank, the world size and MASTER_ADDR /
            MASTER_PORT, which the child needs to rebuild the process group.
        """
        recorder = self._create(use_gloo=True)

        kwargs = recorder.call_args.kwargs
        self.assertIs(kwargs["target"], api_mod.execute_async_persist_with_gloo)
        self.assertEqual(kwargs["args"][5:], (0, 4, "127.0.0.1", 29500))

    def test_use_gloo_is_ignored_when_no_coordination_is_needed(self):
        """
        Feature: use_gloo without collectives.
        Description: Ask for gloo on a save that does not coordinate at all.
        Expectation: The plain persist target runs and gloo is silently ignored - there is
            nothing to coordinate, so no process group is rebuilt.
        """
        self._assert_plain_persist(
            self._create(use_collectives=False, use_gloo=True), use_storage_comm=False
        )

    def test_a_single_rank_needs_no_coordination(self):
        """
        Feature: persist mode on a world of one.
        Description: Ask for collectives and gloo, but run on a single rank.
        Expectation: The plain persist target runs without storage communication.
        """
        self._assert_plain_persist(
            self._create(use_gloo=True, world_size=1), use_storage_comm=False
        )

    def test_no_dist_needs_no_coordination(self):
        """
        Feature: persist mode for no_dist=True.
        Description: Ask for collectives and gloo, but save in single process mode.
        Expectation: The plain persist target runs without storage communication.
        """
        self._assert_plain_persist(
            self._create(no_dist=True, use_gloo=True), use_storage_comm=False
        )

    def test_gloo_without_master_addr_raises(self):
        """
        Feature: gloo rendezvous validation.
        Description: Request gloo while MASTER_ADDR is missing from the environment.
        Expectation: AssertionError naming MASTER_ADDR, instead of a child that cannot connect.
        """
        with self.assertRaises(AssertionError) as ctx:
            self._create(use_gloo=True, env={"MASTER_PORT": "29500"})
        self.assertIn("MASTER_ADDR", str(ctx.exception))

    def test_gloo_without_master_port_raises(self):
        """
        Feature: gloo rendezvous validation.
        Description: Request gloo while MASTER_PORT is missing from the environment.
        Expectation: AssertionError naming MASTER_PORT, instead of a child that cannot connect.
        """
        with self.assertRaises(AssertionError) as ctx:
            self._create(use_gloo=True, env={"MASTER_ADDR": "127.0.0.1"})
        self.assertIn("MASTER_PORT", str(ctx.exception))


class TestSaveImplStorageComm(unittest.TestCase):
    """Tests for the checkpoint directory the storage based coordination needs."""

    def setUp(self) -> None:
        """Rebuild the api module before every case."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(planner_mod)
        importlib.reload(api_mod)

    def test_storage_comm_without_a_checkpoint_dir_raises(self):
        """
        Feature: _save_impl input validation for storage based coordination.
        Description: Pass a storage_writer that exposes no checkpoint_dir and no checkpoint_id,
            while asking the ranks to exchange their plans through the storage.
        Expectation: ValueError naming checkpoint_id, rather than plan files written into a
            path built from the string "None".
        """
        # A spec without checkpoint_dir: getattr falls back to None, as it would for a
        # writer that never learned where the checkpoint goes.
        writer = Mock(spec=["initialize_writer", "configure_writer", "optimize_local_plan"])

        with self.assertRaises(ValueError) as ctx:
            api_mod._save_impl({}, storage_writer=writer, use_storage_comm=True)

        self.assertIn("checkpoint_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

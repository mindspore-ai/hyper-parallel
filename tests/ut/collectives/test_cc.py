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
"""Unit tests for the PyTorch distributed process-group wrappers."""
from datetime import timedelta
import os
import unittest
from unittest.mock import call, MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.collectives import cc as collectives_cc


@patch("hyper_parallel.collectives.cc.dist")
class TestProcessGroupWrappers(unittest.TestCase):
    """Verify direct delegation to ``torch.distributed``."""

    def setUp(self) -> None:
        collectives_cc._EXISTING_COMM_GROUPS.clear()

    def tearDown(self) -> None:
        collectives_cc._EXISTING_COMM_GROUPS.clear()

    def test_init_process_group_forwards_all_arguments(self, mock_dist: MagicMock) -> None:
        """Initialization forwards every supported argument unchanged."""
        timeout = timedelta(minutes=30)
        store = MagicMock(name="store")
        pg_options = MagicMock(name="pg_options")

        collectives_cc.init_process_group(
            "hccl",
            init_method="env://",
            timeout=timeout,
            world_size=8,
            rank=2,
            store=store,
            pg_options=pg_options,
            device_id=2,
        )

        mock_dist.init_process_group.assert_called_once_with(
            backend="hccl",
            init_method="env://",
            timeout=timeout,
            world_size=8,
            rank=2,
            store=store,
            pg_options=pg_options,
            device_id=2,
        )

    def test_init_process_group_forwards_defaults(self, mock_dist: MagicMock) -> None:
        """Initialization preserves the public wrapper defaults."""
        collectives_cc.init_process_group("gloo")

        mock_dist.init_process_group.assert_called_once_with(
            backend="gloo",
            init_method=None,
            timeout=None,
            world_size=-1,
            rank=-1,
            store=None,
            pg_options=None,
            device_id=None,
        )

    def test_destroy_process_group_forwards_group_and_evicts_cache(self, mock_dist: MagicMock) -> None:
        """Destroying a group removes its cached rank-list entry."""
        group = MagicMock(name="group")
        collectives_cc._EXISTING_COMM_GROUPS["(0, 1)"] = group

        collectives_cc.destroy_process_group(group)

        mock_dist.destroy_process_group.assert_called_once_with(group)
        self.assertEqual(collectives_cc._EXISTING_COMM_GROUPS, {})

    def test_destroy_default_process_group_clears_cache(self, mock_dist: MagicMock) -> None:
        """Destroying the default group clears all locally cached groups."""
        collectives_cc._EXISTING_COMM_GROUPS["(0, 1)"] = MagicMock()

        collectives_cc.destroy_process_group()

        mock_dist.destroy_process_group.assert_called_once_with(None)
        self.assertEqual(collectives_cc._EXISTING_COMM_GROUPS, {})

    def test_get_process_group_ranks_uses_world_for_none(self, mock_dist: MagicMock) -> None:
        """The default rank query resolves to ``dist.group.WORLD``."""
        mock_dist.get_process_group_ranks.return_value = [0, 1]

        result = collectives_cc.get_process_group_ranks()

        self.assertEqual(result, [0, 1])
        mock_dist.get_process_group_ranks.assert_called_once_with(mock_dist.group.WORLD)

    def test_get_process_group_ranks_forwards_explicit_group(self, mock_dist: MagicMock) -> None:
        """An explicit process group is passed directly to PyTorch."""
        group = MagicMock(name="group")
        mock_dist.get_process_group_ranks.return_value = [2, 3]

        result = collectives_cc.get_process_group_ranks(group)

        self.assertEqual(result, [2, 3])
        mock_dist.get_process_group_ranks.assert_called_once_with(group)

    def test_get_backend_forwards_group(self, mock_dist: MagicMock) -> None:
        """Backend lookup delegates directly to PyTorch."""
        group = MagicMock(name="group")
        mock_dist.get_backend.return_value = "nccl"

        result = collectives_cc.get_backend(group)

        self.assertEqual(result, "nccl")
        mock_dist.get_backend.assert_called_once_with(group)

    def test_get_group_local_rank_forwards_group(self, mock_dist: MagicMock) -> None:
        """Group-local rank lookup uses ``dist.get_rank(group)``."""
        group = MagicMock(name="group")
        mock_dist.get_rank.return_value = 1

        result = collectives_cc.get_group_local_rank(group)

        self.assertEqual(result, 1)
        mock_dist.get_rank.assert_called_once_with(group)

    def test_split_group_creates_and_selects_current_group(self, mock_dist: MagicMock) -> None:
        """Group splitting creates every subgroup and returns the rank's subgroup."""
        group0 = MagicMock(name="group0")
        group1 = MagicMock(name="group1")
        mock_dist.get_rank.return_value = 2
        mock_dist.new_group.side_effect = [group0, group1]

        result = collectives_cc.split_group(split_ranks=[[0, 1], [2, 3]])

        self.assertIs(result, group1)
        self.assertEqual(
            mock_dist.new_group.call_args_list,
            [call(ranks=[0, 1]), call(ranks=[2, 3])],
        )

    def test_split_group_reuses_cached_groups(self, mock_dist: MagicMock) -> None:
        """Repeated rank lists reuse cached PyTorch process groups."""
        group = MagicMock(name="group")
        collectives_cc._EXISTING_COMM_GROUPS["(0, 1)"] = group
        mock_dist.get_rank.return_value = 0

        result = collectives_cc.split_group(split_ranks=[[1, 0]])

        self.assertIs(result, group)
        mock_dist.new_group.assert_not_called()

    def test_split_group_rejects_empty_ranks(self, mock_dist: MagicMock) -> None:
        """An empty split specification is invalid."""
        del mock_dist
        with self.assertRaises(ValueError):
            collectives_cc.split_group(split_ranks=[])

    def test_mark_created_groups_populates_cache(self, mock_dist: MagicMock) -> None:
        """Existing PyTorch groups are cached by their sorted global ranks."""
        group0 = MagicMock(name="group0")
        group1 = MagicMock(name="group1")
        mock_dist.get_process_group_ranks.side_effect = [[1, 0], [3, 2]]

        collectives_cc.mark_created_groups([group0, group1])

        self.assertIs(collectives_cc._EXISTING_COMM_GROUPS["(0, 1)"], group0)
        self.assertIs(collectives_cc._EXISTING_COMM_GROUPS["(2, 3)"], group1)


class TestCollectivesPublicExports(unittest.TestCase):
    """Sanity checks for package wiring and public re-exports."""

    def test_cc_module_exposes_all_collective_entry_points(self) -> None:
        """Every process-group helper remains publicly available."""
        expected = (
            "init_process_group",
            "destroy_process_group",
            "get_process_group_ranks",
            "get_backend",
            "split_group",
            "get_group_local_rank",
            "mark_created_groups",
        )
        for name in expected:
            self.assertTrue(hasattr(collectives_cc, name), msg=f"missing {name}")

    def test_hyper_parallel_reexports_collectives_api(self) -> None:
        """Top-level exports continue to point at the collectives wrappers."""
        import hyper_parallel as hp  # pylint: disable=import-outside-toplevel

        for name in (
            "init_process_group",
            "destroy_process_group",
            "get_process_group_ranks",
            "get_backend",
            "split_group",
            "get_group_local_rank",
            "mark_created_groups",
        ):
            self.assertIs(getattr(hp, name), getattr(collectives_cc, name))


if __name__ == "__main__":
    unittest.main()

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
"""Unit tests for ``hyper_parallel.collectives.cc`` process-group API wrappers.

The collectives module is a thin delegation layer over :func:`get_platform()`.
Tests mock ``hyper_parallel.collectives.cc.platform`` and verify argument forwarding
and return-value propagation without initializing a real distributed backend.
"""
from __future__ import annotations

import os
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.collectives import cc as collectives_cc
from hyper_parallel.collectives.cc import (
    destroy_process_group,
    get_backend,
    get_group_local_rank,
    get_process_group_ranks,
    init_process_group,
    mark_created_groups,
    split_group,
)


@patch("hyper_parallel.collectives.cc.platform")
class TestInitProcessGroup(unittest.TestCase):
    """Tests for :func:`init_process_group`."""

    def test_forwards_all_keyword_arguments(self, mock_platform: MagicMock) -> None:
        """
        Feature: init_process_group delegates to the active platform
        Description: call with backend, init_method, timeout, world_size, rank, store, pg_options, device_id
        Expectation: platform.init_process_group receives the same keyword arguments
        """
        timeout = timedelta(minutes=30)
        store = MagicMock(name="store")
        pg_options = MagicMock(name="pg_options")
        device_id = 0

        init_process_group(
            "hccl",
            init_method="env://",
            timeout=timeout,
            world_size=8,
            rank=2,
            store=store,
            pg_options=pg_options,
            device_id=device_id,
        )

        mock_platform.init_process_group.assert_called_once_with(
            backend="hccl",
            init_method="env://",
            timeout=timeout,
            world_size=8,
            rank=2,
            store=store,
            pg_options=pg_options,
            device_id=device_id,
        )

    def test_forwards_defaults_when_only_backend_given(self, mock_platform: MagicMock) -> None:
        """
        Feature: init_process_group default parameters
        Description: call with backend only
        Expectation: platform receives default world_size=-1 and rank=-1
        """
        init_process_group("gloo")

        mock_platform.init_process_group.assert_called_once_with(
            backend="gloo",
            init_method=None,
            timeout=None,
            world_size=-1,
            rank=-1,
            store=None,
            pg_options=None,
            device_id=None,
        )


@patch("hyper_parallel.collectives.cc.platform")
class TestDestroyProcessGroup(unittest.TestCase):
    """Tests for :func:`destroy_process_group`."""

    def test_destroy_default_group(self, mock_platform: MagicMock) -> None:
        """
        Feature: destroy_process_group with implicit default group
        Description: call without a group argument
        Expectation: platform.destroy_process_group is called with group=None
        """
        destroy_process_group()
        mock_platform.destroy_process_group.assert_called_once_with(group=None)

    def test_destroy_explicit_group(self, mock_platform: MagicMock) -> None:
        """
        Feature: destroy_process_group with an explicit group
        Description: pass a process group handle
        Expectation: platform receives the same group object
        """
        group = MagicMock(name="pg")
        destroy_process_group(group)
        mock_platform.destroy_process_group.assert_called_once_with(group=group)


@patch("hyper_parallel.collectives.cc.platform")
class TestGetProcessGroupRanks(unittest.TestCase):
    """Tests for :func:`get_process_group_ranks`."""

    def test_returns_platform_rank_list(self, mock_platform: MagicMock) -> None:
        """
        Feature: get_process_group_ranks return value
        Description: platform returns a sorted rank list for the default group
        Expectation: API returns the same list unchanged
        """
        mock_platform.get_process_group_ranks.return_value = [0, 1, 2, 3]
        ranks = get_process_group_ranks()
        self.assertEqual(ranks, [0, 1, 2, 3])
        mock_platform.get_process_group_ranks.assert_called_once_with(group=None)

    def test_forwards_explicit_group(self, mock_platform: MagicMock) -> None:
        """
        Feature: get_process_group_ranks with explicit group
        Description: pass a subgroup handle
        Expectation: platform is queried with that group
        """
        group = MagicMock(name="sub_pg")
        mock_platform.get_process_group_ranks.return_value = [2, 3]
        ranks = get_process_group_ranks(group)
        self.assertEqual(ranks, [2, 3])
        mock_platform.get_process_group_ranks.assert_called_once_with(group=group)


@patch("hyper_parallel.collectives.cc.platform")
class TestGetBackend(unittest.TestCase):
    """Tests for :func:`get_backend`."""

    def test_returns_platform_backend_name(self, mock_platform: MagicMock) -> None:
        """
        Feature: get_backend return value
        Description: platform reports backend string for default group
        Expectation: API returns the same backend name
        """
        mock_platform.get_backend.return_value = "hccl"
        backend = get_backend()
        self.assertEqual(backend, "hccl")
        mock_platform.get_backend.assert_called_once_with(group=None)

    def test_forwards_explicit_group(self, mock_platform: MagicMock) -> None:
        """
        Feature: get_backend with explicit group
        Description: pass a subgroup handle
        Expectation: platform is queried with that group
        """
        group = MagicMock(name="sub_pg")
        mock_platform.get_backend.return_value = "nccl"
        backend = get_backend(group)
        self.assertEqual(backend, "nccl")
        mock_platform.get_backend.assert_called_once_with(group=group)


@patch("hyper_parallel.collectives.cc.platform")
class TestSplitGroup(unittest.TestCase):
    """Tests for :func:`split_group`."""

    def test_forwards_arguments_and_returns_subgroup(self, mock_platform: MagicMock) -> None:
        """
        Feature: split_group delegation
        Description: split default parent group into rank lists with timeout and metadata
        Expectation: platform.split_group receives all kwargs; return value is propagated
        """
        parent_pg = MagicMock(name="parent_pg")
        split_ranks = [[0, 1], [2, 3]]
        timeout = timedelta(seconds=60)
        pg_options = MagicMock(name="pg_options")
        expected_subgroup = MagicMock(name="sub_pg")
        mock_platform.split_group.return_value = expected_subgroup

        result = split_group(
            parent_pg=parent_pg,
            split_ranks=split_ranks,
            timeout=timeout,
            pg_options=pg_options,
            group_desc="tp",
        )

        self.assertIs(result, expected_subgroup)
        mock_platform.split_group.assert_called_once_with(
            parent_pg=parent_pg,
            split_ranks=split_ranks,
            timeout=timeout,
            pg_options=pg_options,
            group_desc="tp",
        )

    def test_forwards_defaults(self, mock_platform: MagicMock) -> None:
        """
        Feature: split_group default parameters
        Description: call with no arguments
        Expectation: platform receives None defaults for optional parameters
        """
        mock_platform.split_group.return_value = None
        result = split_group()
        self.assertIsNone(result)
        mock_platform.split_group.assert_called_once_with(
            parent_pg=None,
            split_ranks=None,
            timeout=None,
            pg_options=None,
            group_desc=None,
        )


@patch("hyper_parallel.collectives.cc.platform")
class TestGetGroupLocalRank(unittest.TestCase):
    """Tests for :func:`get_group_local_rank`."""

    def test_returns_platform_local_rank(self, mock_platform: MagicMock) -> None:
        """
        Feature: get_group_local_rank return value
        Description: platform reports local rank within default group
        Expectation: API returns the same integer
        """
        mock_platform.get_group_local_rank.return_value = 1
        local_rank = get_group_local_rank()
        self.assertEqual(local_rank, 1)
        mock_platform.get_group_local_rank.assert_called_once_with(group=None)

    def test_forwards_explicit_group(self, mock_platform: MagicMock) -> None:
        """
        Feature: get_group_local_rank with explicit group
        Description: pass a subgroup handle
        Expectation: platform is queried with that group
        """
        group = MagicMock(name="sub_pg")
        mock_platform.get_group_local_rank.return_value = 0
        local_rank = get_group_local_rank(group)
        self.assertEqual(local_rank, 0)
        mock_platform.get_group_local_rank.assert_called_once_with(group=group)


@patch("hyper_parallel.collectives.cc.platform")
class TestMarkCreatedGroups(unittest.TestCase):
    """Tests for :func:`mark_created_groups`."""

    def test_forwards_single_group(self, mock_platform: MagicMock) -> None:
        """
        Feature: mark_created_groups with one process group
        Description: register a single subgroup in the platform cache
        Expectation: platform.mark_created_groups is called with that group
        """
        group = MagicMock(name="pg")
        mock_platform.mark_created_groups.return_value = None
        result = mark_created_groups(group)
        self.assertIsNone(result)
        mock_platform.mark_created_groups.assert_called_once_with(process_group=group)

    def test_forwards_group_list(self, mock_platform: MagicMock) -> None:
        """
        Feature: mark_created_groups with a list of process groups
        Description: register multiple subgroups at once
        Expectation: platform receives the same list object
        """
        groups = [MagicMock(name="pg0"), MagicMock(name="pg1")]
        mark_created_groups(groups)
        mock_platform.mark_created_groups.assert_called_once_with(process_group=groups)


class TestCollectivesPublicExports(unittest.TestCase):
    """Sanity checks for package wiring and public re-exports."""

    def test_cc_module_exposes_all_collective_entry_points(self) -> None:
        """
        Feature: collectives.cc public API surface
        Description: inspect module attributes
        Expectation: all process-group helpers are defined on cc
        """
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
        """
        Feature: hyper_parallel top-level re-exports
        Description: import collectives helpers from hyper_parallel package
        Expectation: each name is callable and defined under collectives.cc
        """
        import hyper_parallel as hp

        for name in (
            "init_process_group",
            "destroy_process_group",
            "get_process_group_ranks",
            "get_backend",
            "split_group",
            "get_group_local_rank",
            "mark_created_groups",
        ):
            exported = getattr(hp, name)
            cc_fn = getattr(collectives_cc, name)
            self.assertTrue(callable(exported))
            self.assertEqual(exported.__name__, cc_fn.__name__)
            self.assertIn("collectives.cc", exported.__module__)


if __name__ == "__main__":
    unittest.main()

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
"""Native Torch pipeline communicator initialization tests."""
import unittest
from unittest import mock
from unittest.mock import MagicMock

from hyper_parallel.core.pipeline_parallel import _p2p


class TestPipelineGroups(unittest.TestCase):
    """Verify communicator creation order and startup synchronization."""

    @mock.patch("torch.distributed.barrier")
    def test_prepare_batch_p2p_group_uses_public_barrier(self, mock_barrier):
        """Prepare batched P2P without relying on private ProcessGroup APIs."""
        _p2p.prepare_batch_p2p_group(mock.sentinel.pp_group)

        mock_barrier.assert_called_once_with(group=mock.sentinel.pp_group)


    def test_create_p2p_multi_stream_groups_creates_local_edges_in_stable_order(self) -> None:
        """
        Feature: PyTorch multi-stream pipeline P2P groups.
        Description: Initialize two interleaved PP rings while the current rank belongs to one.
        Expectation: Every rank creates the global edge set in one order and retains only local groups.
        """
        groups = [MagicMock(name=f"group_{index}") for index in range(8)]

        def _all_gather_pp_rank_lists(output, local_ranks):
            self.assertEqual(local_ranks, [0, 1, 2, 3])
            output[:] = [[0, 1, 2, 3]] * 4 + [[4, 5, 6, 7]] * 4

        with mock.patch.dict(
                "hyper_parallel.core.pipeline_parallel._p2p._P2P_MULTI_STREAM_GROUPS",
                clear=True,
        ), mock.patch(
            "hyper_parallel.core.pipeline_parallel._p2p.dist.get_rank",
            return_value=1,
        ), mock.patch(
            "hyper_parallel.core.pipeline_parallel._p2p.dist.get_world_size",
            return_value=8,
        ), mock.patch(
            "hyper_parallel.core.pipeline_parallel._p2p.dist.all_gather_object",
            side_effect=_all_gather_pp_rank_lists,
        ), mock.patch(
            "hyper_parallel.core.pipeline_parallel._p2p.dist.new_group",
            side_effect=groups,
        ) as new_group:
            local_groups = _p2p.create_p2p_multi_stream_groups(
                [0, 1, 2, 3],
                include_wrap=True,
            )

        expected_calls = [
            mock.call(ranks=[0, 1]),
            mock.call(ranks=[0, 3]),
            mock.call(ranks=[1, 2]),
            mock.call(ranks=[2, 3]),
            mock.call(ranks=[4, 5]),
            mock.call(ranks=[4, 7]),
            mock.call(ranks=[5, 6]),
            mock.call(ranks=[6, 7]),
        ]
        self.assertEqual(new_group.call_args_list, expected_calls)
        self.assertEqual(local_groups, {0: groups[0], 2: groups[2]})

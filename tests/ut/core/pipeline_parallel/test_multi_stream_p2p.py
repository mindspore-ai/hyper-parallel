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
"""Unit tests for multi-stream pipeline P2P groups."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hyper_parallel.core.pipeline_parallel.scheduler as scheduler_module
from hyper_parallel.core.pipeline_parallel.scheduler import ScheduleInterleaved1F1B


def _make_runtime(mode: str = "multi_stream") -> ScheduleInterleaved1F1B:
    """Build a scheduler shell without initializing a distributed backend."""
    runtime = object.__new__(ScheduleInterleaved1F1B)
    runtime._p2p_mode = mode
    runtime._batch_p2p_group = None
    runtime._p2p_multi_stream_groups = {}
    return runtime


def test_init_p2p_multi_stream_groups_uses_mesh_order_and_interleaved_wrap() -> None:
    """
    Feature: Multi-stream pipeline P2P group initialization.
    Description: Build peer-specific groups from the ordered PP mesh for an interleaved schedule.
    Expectation: The platform receives the PP ranks in mesh order with wrap enabled.
    """
    runtime = _make_runtime()
    runtime.stages = [SimpleNamespace(mesh=SimpleNamespace(rank_list=(0, 2, 4, 6)), pp_group="unused")]
    runtime.n_local_stages = 2
    expected_groups = {2: "multi-stream-0-2", 6: "multi-stream-0-6"}

    with patch.object(
            scheduler_module.platform,
            "create_p2p_multi_stream_groups",
            return_value=expected_groups,
    ) as create_multi_stream_groups:
        runtime._init_p2p_multi_stream_groups()

    create_multi_stream_groups.assert_called_once_with([0, 2, 4, 6], include_wrap=True)
    assert runtime._p2p_multi_stream_groups == expected_groups, (
        f"Expected multi-stream groups {expected_groups}, got={runtime._p2p_multi_stream_groups}"
    )


def test_batched_issue_passes_multi_stream_group_to_platform() -> None:
    """
    Feature: Multi-stream pipeline P2P descriptor routing.
    Description: Issue a batched send through the group mapped to its peer.
    Expectation: The P2P descriptor receives the peer's group and returns one batch handle.
    """
    runtime = _make_runtime()
    runtime._p2p_multi_stream_groups = {3: "multi-stream-1-3"}
    tensor = object()
    descriptor = object()
    handle = object()

    with patch.object(scheduler_module.platform, "p2p_op", return_value=descriptor) as p2p_op, \
            patch.object(
                scheduler_module.platform,
                "batch_isend_irecv",
                return_value=handle,
            ) as batch_isend_irecv:
        handles = runtime._batched_issue([("isend", tensor, 3)])

    p2p_op.assert_called_once_with("isend", tensor, 3, group="multi-stream-1-3")
    batch_isend_irecv.assert_called_once_with([descriptor])
    assert handles == [handle], f"Expected one batch handle {[handle]}, got={handles}"


def test_multi_stream_transport_rejects_peer_without_matching_group() -> None:
    """
    Feature: Multi-stream pipeline P2P descriptor validation.
    Description: Request a multi-stream descriptor for a peer absent from the initialized mapping.
    Expectation: The scheduler raises before issuing an operation on the default group.
    """
    runtime = _make_runtime()
    runtime._p2p_multi_stream_groups = {1: "multi-stream-0-1"}

    with pytest.raises(RuntimeError, match="peer global rank 2"):
        runtime._p2p_op("irecv", object(), 2)


def test_batch_transport_keeps_pipeline_group() -> None:
    """
    Feature: Existing batch P2P transport compatibility.
    Description: Build a descriptor while the scheduler is using batch transport.
    Expectation: The descriptor keeps the pipeline group prepared for batched P2P.
    """
    runtime = _make_runtime(mode="batch")
    runtime._batch_p2p_group = "pipeline-group"
    tensor = object()

    with patch.object(scheduler_module.platform, "p2p_op", return_value=object()) as p2p_op:
        runtime._p2p_op("irecv", tensor, 5)

    p2p_op.assert_called_once_with("irecv", tensor, 5, group="pipeline-group")

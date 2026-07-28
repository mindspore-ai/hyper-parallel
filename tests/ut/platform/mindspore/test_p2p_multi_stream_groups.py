# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for MindSpore multi-stream PP group creation."""
# pylint: disable=wrong-import-position

import os
from unittest.mock import call, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import pytest

pytest.importorskip("mindspore")

from tests.common.mark_utils import arg_mark
from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

from hyper_parallel.platform.mindspore.platform import MindSporePlatform


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_create_p2p_multi_stream_groups_initializes_only_local_edges() -> None:
    """
    Feature: MindSpore multi-stream pipeline P2P groups.
    Description: Initialize peer-specific groups for an interior rank in an interleaved PP ring.
    Expectation: Only the two incident groups are created and returned by peer rank.
    """
    with patch.dict(
            "hyper_parallel.platform.mindspore.platform.EXISTING_COMM_GROUPS",
            clear=True,
    ), patch.object(
        MindSporePlatform,
        "get_rank",
        return_value=2,
    ), patch.object(
        MindSporePlatform,
        "_create_group_with_options",
    ) as create_group:
        local_groups = MindSporePlatform.create_p2p_multi_stream_groups(
            [0, 2, 4, 6],
            include_wrap=True,
        )

    assert create_group.call_args_list == [
        call("(0, 2)", [0, 2]),
        call("(2, 4)", [2, 4]),
    ], f"Unexpected MindSpore multi-stream group creation order: {create_group.call_args_list}"
    assert local_groups == {0: "(0, 2)", 4: "(2, 4)"}, (
        f"Expected groups for peers 0 and 4, got={local_groups}"
    )

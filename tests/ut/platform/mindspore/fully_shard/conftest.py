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
"""Shared fixtures for MindSpore fully_shard unit tests."""
import unittest

import pytest

from hyper_parallel.platform.mindspore.fully_shard.param_group import AllGatherMetadataCache
from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    _force_cpu_device_target,
    ensure_mindspore_platform_for_fully_shard,
)

# Hardware-agnostic device tags for MindSpore fully_shard UTs (CPU backend only).
UT_RUNTIME_DEVICE = "cpu"
UT_MS_DEVICE = "CPU"
UT_MS_DEVICE_TAG = "CPU:0"


def reset_mindspore_fully_shard_shared_state() -> None:
    """Pin the CPU backend and clear shared metadata between UT cases."""
    ensure_mindspore_platform_for_fully_shard()
    _force_cpu_device_target()
    AllGatherMetadataCache._cache.clear()


class MindSporeFullyShardUnitTest(unittest.TestCase):
    """Base unittest case: MindSpore platform on CPU, no distributed init."""

    def setUp(self):
        """Reset platform singleton and fused-comm globals before each test."""
        reset_mindspore_fully_shard_shared_state()


@pytest.fixture(autouse=True)
def _reset_mindspore_fully_shard_shared_state():
    """Pytest autouse hook mirroring :class:`MindSporeFullyShardUnitTest.setUp`."""
    reset_mindspore_fully_shard_shared_state()

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
"""Tests for init_empty_weights -> fully_shard -> init weight consistency (MindSpore)."""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# pylint: disable=C0413
import pytest

from hyper_parallel.core.dtensor.init_weights import init_on_device
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

# Run with cwd ``tests/mindspore/st`` (same pattern as ``hsdp/hsdp.py``, ``shard/base_shard.py``).
_TEST_INIT_WEIGHTS = "_test_init_weights.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_init_weights():
    """
    Feature: parallel run case in init_weights (MindSpore)
    Description:
        1. test_init_weights_consistency
        2. test_init_weights_with_randn_like
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(_TEST_INIT_WEIGHTS, "test_init_weights_with_randn_like", 12351, 2, 2),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_init_on_device_include_buffers_true_raises() -> None:
    """MindSpore backend should reject include_buffers=True."""
    with pytest.raises(ValueError, match="does not support include_buffers=True"):
        with init_on_device("meta", include_buffers=True):
            pass


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_init_on_device_invalid_device_raises() -> None:
    """MindSpore backend should reject unsupported external device values."""
    with pytest.raises(ValueError, match='only "npu", "cpu", and "meta" are allowed'):
        with init_on_device("Ascend:0"):
            pass

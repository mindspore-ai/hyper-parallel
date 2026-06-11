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
"""Pytest launcher for MindSpore StridedShard full_tensor coverage."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_WORKER = str(Path(__file__).resolve().parent / "strided_shard_full_tensor.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_strided_shard_full_tensor_mindspore():
    """
    Feature: MindSpore StridedShard DTensor.full_tensor.
    Description: Run 8-card same-dim StridedShard roundtrip scenarios.
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(_WORKER, "test_strided_shard_full_tensor_roundtrip", 12783, 8, 8),
    ])

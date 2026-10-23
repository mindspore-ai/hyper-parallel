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
"""Lightweight launchers for the native ragged EP regression."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_ragged_joint_graph(dtype):
    """
    Feature: Native variable-split EP graph capture.
    Description: Change expert loads, including empty ranks, within one joint graph.
    Expectation: Ten updates and two accumulated microbatches match eager training.
    """
    worker = str(Path(__file__).with_name("_test_ragged_graph_ep.py"))
    torchrun_case(file_name=worker, case_name=f"test_ragged_joint_graph[{dtype}]", num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_ragged_fsdp_joint_graph(dtype):
    """
    Feature: Native EP with graph FSDP.
    Description: Use four dense shards and two expert data-parallel shards.
    Expectation: Every gradient and updated shard matches eager training.
    """
    worker = str(Path(__file__).with_name("_test_ragged_graph_ep.py"))
    torchrun_case(file_name=worker, case_name=f"test_ragged_fsdp_joint_graph[{dtype}]", num_proc=4)

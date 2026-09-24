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
"""test DCP load through the read-and-broadcast pipeline"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_PIPELINE_LOAD = "dcp_pipeline_load.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_pipeline_load_in_flight():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pipeline_more_shards_than_broadcasts_in_flight
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_more_shards_than_broadcasts_in_flight",
                  num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_pipeline_load_interleaved_groups():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pipeline_interleaves_groups_and_private_shards
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_interleaves_groups_and_private_shards",
                  num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_dcp_pipeline_load_non_tensor_state():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pipeline_carries_non_tensor_state
        2.test_dcp_load_pipeline_repeats_cleanly
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_carries_non_tensor_state", num_proc=4),
    ])
    parallel_run([
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_repeats_cleanly", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_pipeline_load_gloo():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_load_pipeline_more_shards_than_broadcasts_in_flight
        2.test_dcp_load_pipeline_interleaves_groups_and_private_shards
        3.test_dcp_load_pipeline_carries_non_tensor_state
        4.test_dcp_load_pipeline_repeats_cleanly
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_more_shards_than_broadcasts_in_flight",
                  num_proc=4),
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_interleaves_groups_and_private_shards",
                  num_proc=4),
    ])
    parallel_run([
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_carries_non_tensor_state", num_proc=4),
        TorchCase(DCP_PIPELINE_LOAD, "test_dcp_load_pipeline_repeats_cleanly", num_proc=4),
    ])

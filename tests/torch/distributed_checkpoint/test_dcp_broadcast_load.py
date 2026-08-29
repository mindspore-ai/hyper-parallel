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
"""test DCP load with broadcast_from_minimum_rank"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_BROADCAST_LOAD = "dcp_broadcast_load.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_broadcast_load_on_demand_groups():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_load_broadcast_from_minimum_rank
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_BROADCAST_LOAD, "test_dcp_load_broadcast_from_minimum_rank", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_broadcast_load_prebuilt_groups():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_load_broadcast_with_prebuilt_groups
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_BROADCAST_LOAD, "test_dcp_load_broadcast_with_prebuilt_groups", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_broadcast_load_plain_tensor():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_load_broadcast_plain_tensor_with_chunk_info
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_BROADCAST_LOAD, "test_dcp_load_broadcast_plain_tensor_with_chunk_info", num_proc=4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_broadcast_load_gloo():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_load_broadcast_from_minimum_rank
        2.test_dcp_load_broadcast_with_prebuilt_groups
        3.test_dcp_load_broadcast_plain_tensor_with_chunk_info
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_BROADCAST_LOAD, "test_dcp_load_broadcast_from_minimum_rank", num_proc=4),
        TorchCase(DCP_BROADCAST_LOAD, "test_dcp_load_broadcast_with_prebuilt_groups", num_proc=4),
    ])
    parallel_run([
        TorchCase(DCP_BROADCAST_LOAD, "test_dcp_load_broadcast_plain_tensor_with_chunk_info", num_proc=4),
    ])

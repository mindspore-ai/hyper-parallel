# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""all testcases in distributed_checkpoint"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, MindSporeCase

RESHARD_HANDLER = "reshard_handler.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_checkpoint_group1():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_reshard_between_layout_and_none_layout
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(RESHARD_HANDLER, "test_reshard_between_layout_and_none_layout", 11000)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_checkpoint_group2():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_wrong_to_rank_id_while_to_layout_not_none
        2.test_wrong_to_rank_id_while_to_layout_is_none
        3.test_shape_not_even_divided_by_mesh_shape
        4.test_reshard_between_fully_shard
        5.test_reshard_between_fully_shard_and_not_shard
        6.test_reshard_between_not_fully_shard
        7.test_from_tensor_map_missing_rank
        8.test_from_tensor_map_has_unexpected_data
    Expectation: Run success.
    """
    parallel_run([
        MindSporeCase(RESHARD_HANDLER, "test_wrong_to_rank_id_while_to_layout_not_none", 11001),
        MindSporeCase(RESHARD_HANDLER, "test_wrong_to_rank_id_while_to_layout_is_none", 11002),
        MindSporeCase(RESHARD_HANDLER, "test_shape_not_even_divided_by_mesh_shape", 11003),
        MindSporeCase(RESHARD_HANDLER, "test_reshard_between_fully_shard", 11004),
        MindSporeCase(RESHARD_HANDLER, "test_reshard_between_fully_shard_and_not_shard", 11005),
        MindSporeCase(RESHARD_HANDLER, "test_reshard_between_not_fully_shard", 11006),
        MindSporeCase(RESHARD_HANDLER, "test_from_tensor_map_missing_rank", 11007),
        MindSporeCase(RESHARD_HANDLER, "test_from_tensor_map_has_unexpected_data", 11008)
    ])

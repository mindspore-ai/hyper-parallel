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
"""parallel_embedding_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

FILE_NAME = "embedding_shard_in_python.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_data_parallel():
    """
    Feature: Embedding operator.
    Description: Test embedding Data Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_data_parallel"
    master_port = 11301
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_column_parallel():
    """
    Feature: Embedding operator.
    Description: Test embedding Column Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_column_parallel"
    master_port = 11302
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_row_parallel():
    """
    Feature: Embedding operator.
    Description: Test embedding Row Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_row_parallel"
    master_port = 11303
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_dp_cp():
    """
    Feature: Embedding operator.
    Description: Test embedding Data Parallel + Column Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_dp_cp"
    master_port = 11304
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_dp_rp():
    """
    Feature: Embedding operator.
    Description: Test embedding Data Parallel + Row Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_dp_rp"
    master_port = 11305
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_sequence_parallel():
    """
    Feature: Embedding operator.
    Description: Test embedding Sequence Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_sequence_parallel"
    master_port = 11306
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_sp_cp():
    """
    Feature: Embedding operator.
    Description: Test embedding Sequence Parallel + Column Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_sp_cp"
    master_port = 11307
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_sp_rp():
    """
    Feature: Embedding operator.
    Description: Test embedding Sequence Parallel + Row Parallel in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_sp_rp"
    master_port = 11308
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_weight_2d_sharding():
    """
    Feature: Embedding operator.
    Description: Test embedding Weight 2D Sharding in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_weight_2d_sharding"
    master_port = 11309
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_weight_row_parallel_only():
    """
    Feature: Embedding operator.
    Description: Test embedding Weight Row Parallel Only in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_weight_row_parallel_only"
    master_port = 11310
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_cp_padding_and_scale():
    """
    Feature: Embedding operator.
    Description: Test embedding Column Parallel with padding_idx & scale_grad_by_freq in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_cp_padding_and_scale"
    master_port = 11311
    msrun_case(glog_v, FILE_NAME, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_rp_padding():
    """
    Feature: Embedding operator.
    Description: Test embedding Row Parallel with padding_idx in python shard.
    Expectation: Run success.
    """
    glog_v = 2
    case_name = "test_distributed_embedding_rp_padding"
    master_port = 11313
    msrun_case(glog_v, FILE_NAME, case_name, master_port)

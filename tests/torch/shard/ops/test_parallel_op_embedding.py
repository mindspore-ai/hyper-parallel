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
"""test distributed embedding op st cases"""

from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


FILE_NAME = "parallel_op_embedding.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_data_parallel():
    '''
    Feature: test parallel op embedding (Data Parallel).
    Description: Verify local embedding equivalence for batch sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_data_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_data_parallel():
    '''
    Feature: test parallel op embedding (Data Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for batch sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_data_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_column_parallel():
    '''
    Feature: test parallel op embedding (Column Parallel).
    Description: Verify local embedding equivalence for weight embed_dim sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_column_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_column_parallel():
    '''
    Feature: test parallel op embedding (Column Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for weight embed_dim sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_column_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_row_parallel():
    '''
    Feature: test parallel op embedding (Row Parallel).
    Description: Verify mask wrapper and partial state resolution for weight vocab sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_row_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_row_parallel():
    '''
    Feature: test parallel op embedding (Row Parallel) (gloo cpu).
    Description: Verify mask wrapper and partial state resolution for weight vocab sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_row_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_dp_cp():
    '''
    Feature: test parallel op embedding (Data Parallel + Column Parallel).
    Description: Verify local embedding equivalence for batch sharding & weight embed_dim sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_dp_cp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_dp_cp():
    '''
    Feature: test parallel op embedding (Data Parallel + Column Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for batch sharding & weight embed_dim sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_dp_cp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_dp_rp():
    '''
    Feature: test parallel op embedding (Data Parallel + Row Parallel).
    Description: Verify local embedding equivalence for batch sharding & weight vocab sharding with Partial reduce.
    Expectation: Run success.
    '''
    case_name = "test_embedding_dp_rp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_dp_rp():
    '''
    Feature: test parallel op embedding (Data Parallel + Row Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for batch sharding & weight vocab sharding with Partial reduce.
    Expectation: Run success.
    '''
    case_name = "test_embedding_dp_rp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_sequence_parallel():
    '''
    Feature: test parallel op embedding (Sequence Parallel).
    Description: Verify local embedding equivalence for sequence length sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_sequence_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_sequence_parallel():
    '''
    Feature: test parallel op embedding (Sequence Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for sequence length sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_sequence_parallel"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_sp_cp():
    '''
    Feature: test parallel op embedding (Sequence Parallel + Column Parallel).
    Description: Verify local embedding equivalence for seq_len sharding & weight embed_dim sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_sp_cp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_sp_cp():
    '''
    Feature: test parallel op embedding (Sequence Parallel + Column Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for seq_len sharding & weight embed_dim sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_sp_cp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_sp_rp():
    '''
    Feature: test parallel op embedding (Sequence Parallel + Row Parallel).
    Description: Verify local embedding equivalence for seq_len sharding & weight vocab sharding with Partial reduce.
    Expectation: Run success.
    '''
    case_name = "test_embedding_sp_rp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_sp_rp():
    '''
    Feature: test parallel op embedding (Sequence Parallel + Row Parallel) (gloo cpu).
    Description: Verify local embedding equivalence for seq_len sharding & weight vocab sharding with Partial reduce.
    Expectation: Run success.
    '''
    case_name = "test_embedding_sp_rp"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_weight_2d_sharding():
    '''
    Feature: test parallel op embedding (Weight 2D Sharding).
    Description: Verify local embedding equivalence for weight vocab sharding and embed_dim sharding simultaneously.
    Expectation: Run success.
    '''
    case_name = "test_embedding_weight_2d_sharding"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_weight_2d_sharding():
    '''
    Feature: test parallel op embedding (Weight 2D Sharding) (gloo cpu).
    Description: Verify local embedding equivalence for weight vocab sharding and embed_dim sharding simultaneously.
    Expectation: Run success.
    '''
    case_name = "test_embedding_weight_2d_sharding"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_weight_row_parallel_only():
    '''
    Feature: test parallel op embedding (Weight Row Parallel Only).
    Description: Verify mask wrapper and partial state resolution for weight vocab sharding
    isolated with replicated input.
    Expectation: Run success.
    '''
    case_name = "test_embedding_weight_row_parallel_only"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_weight_row_parallel_only():
    '''
    Feature: test parallel op embedding (Weight Row Parallel Only) (gloo cpu).
    Description: Verify mask wrapper and partial state resolution for weight vocab sharding
    isolated with replicated input.
    Expectation: Run success.
    '''
    case_name = "test_embedding_weight_row_parallel_only"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_cp_padding_and_scale():
    '''
    Feature: test parallel op embedding with padding_idx & scale_grad_by_freq (Column Parallel).
    Description: Verify native compatibility for padding_idx and scale_grad_by_freq in Column-Sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_cp_padding_and_scale"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_cp_padding_and_scale():
    '''
    Feature: test parallel op embedding with padding_idx & scale_grad_by_freq (Column Parallel) (gloo cpu).
    Description: Verify native compatibility for padding_idx and scale_grad_by_freq in Column-Sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_cp_padding_and_scale"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_cp_max_norm_error():
    '''
    Feature: test parallel op embedding with max_norm (Column Parallel).
    Description: Verify max_norm raises NotImplementedError in Column-Sharding.
    Expectation: Run success.
    '''
    case_name = "test_distributed_embedding_cp_max_norm_error"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_rp_padding():
    '''
    Feature: test parallel op embedding with padding_idx (Row Parallel).
    Description: Verify padding_idx is correctly mapped or ignored in Row-Sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_rp_padding"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_rp_padding():
    '''
    Feature: test parallel op embedding with padding_idx (Row Parallel) (gloo cpu).
    Description: Verify padding_idx is correctly mapped or ignored in Row-Sharding.
    Expectation: Run success.
    '''
    case_name = "test_embedding_rp_padding"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_rp_kwargs_error():
    '''
    Feature: test parallel op embedding with kwargs error (Row Parallel).
    Description: Verify max_norm and scale_grad_by_freq raise NotImplementedError in Row-Sharding.
    Expectation: Run success.
    '''
    case_name = "test_distributed_embedding_rp_kwargs_error"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_embedding_args_positional():
    '''
    Feature: test parallel op embedding (Positional Args).
    Description: Verify positional argument parsing branches (padding_idx, max_norm, scale_grad_by_freq).
    Expectation: Run success.
    '''
    case_name = "test_embedding_args_positional"
    torchrun_case(FILE_NAME, case_name)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_embedding_args_positional():
    '''
    Feature: test parallel op embedding (Positional Args) (gloo cpu).
    Description: Verify positional argument parsing branches (padding_idx, max_norm, scale_grad_by_freq).
    Expectation: Run success.
    '''
    case_name = "test_embedding_args_positional"
    torchrun_case(FILE_NAME, case_name)

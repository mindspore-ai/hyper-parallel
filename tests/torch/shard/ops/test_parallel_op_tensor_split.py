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
"""test base dtensor tensor_split"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_by_sections_unsharded():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split by integer sections.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_by_sections_unsharded"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_by_sections_unsharded():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split by integer sections.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_by_sections_unsharded"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_by_indices_unsharded():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split by tuple of indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_by_indices_unsharded"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_by_indices_unsharded():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split by tuple of indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_by_indices_unsharded"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_default_dim():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split using default dim 0.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_default_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_default_dim():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split using default dim 0.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_default_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_negative_dim():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split using negative dim.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_negative_dim():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split using negative dim.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_negative_dim"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_3d_sections():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split by sections on a 3D tensor.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_3d_sections"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_3d_sections():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split by sections on a 3D tensor.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_3d_sections"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_1d_tensor_indices():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split using a 1D tensor of indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_1d_tensor_indices"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_1d_tensor_indices():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split using a 1D tensor of indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_1d_tensor_indices"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_uneven_sections():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split with uneven sections.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_uneven_sections"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_uneven_sections():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split with uneven sections.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_uneven_sections"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_out_of_bounds_indices():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split with out-of-bounds indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_out_of_bounds_indices"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_out_of_bounds_indices():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split with out-of-bounds indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_out_of_bounds_indices"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_4d_multi_shard():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split on a 4D tensor with multi-sharding.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_4d_multi_shard"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_4d_multi_shard():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split on a 4D tensor with multi-sharding.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_4d_multi_shard"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_list_indices():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split using a list of indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_list_indices"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_list_indices():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split using a list of indices.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_list_indices"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_tensor_split_replicated():
    """
    Feature: test parallel op tensor_split.
    Description: test parallel op tensor_split on a fully replicated tensor.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_replicated"
    torchrun_case(file_name, case_name)

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_tensor_split_replicated():
    """
    Feature: test parallel op tensor_split (gloo cpu).
    Description: test parallel op tensor_split on a fully replicated tensor.
    Expectation: Run success.
    """

    file_name = "parallel_op_tensor_split.py"
    case_name = "test_tensor_split_replicated"
    torchrun_case(file_name, case_name)

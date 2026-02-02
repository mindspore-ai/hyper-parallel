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
"""test base dtensor"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_interleave_layout_inference():
    """
    Feature: test parallel op repeat_interleave.
    Description: test parallel op repeat_interleave.
    Expectation: Run success.
    """
    master_port = 10892
    file_name = "/home/f50056840/hyper-parallel/tests/torch/shard/ops/parallel_op_repeat_interleave.py"
    case_name = "test_distributed_repeat_interleave_layout_inference"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_interleave_with_tensor():
    """
    Feature: test parallel op repeat_interleave.
    Description: test parallel op repeat_interleave.
    Expectation: Run success.
    """
    master_port = 10892
    file_name = "/home/f50056840/hyper-parallel/tests/torch/shard/ops/parallel_op_repeat_interleave.py"
    case_name = "test_distributed_repeat_interleave_with_tensor"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_interleave_dim_none():
    """
    Feature: test parallel op repeat_interleave.
    Description: test parallel op repeat_interleave.
    Expectation: Run success.
    """
    master_port = 10892
    file_name = "/home/f50056840/hyper-parallel/tests/torch/shard/ops/parallel_op_repeat_interleave.py"
    case_name = "test_distributed_repeat_interleave_dim_None"
    torchrun_case(file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_repeat_interleave_sharded_dim_error():
    """
    Feature: test parallel op repeat_interleave.
    Description: test parallel op repeat_interleave.
    Expectation: Run success.
    """
    master_port = 10892
    file_name = "/home/f50056840/hyper-parallel/tests/torch/shard/ops/parallel_op_repeat_interleave.py"
    case_name = "test_distributed_repeat_interleave_sharded_dim_error"
    torchrun_case(file_name, case_name, master_port)

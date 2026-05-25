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
"""test parallel op layernorm"""
from pathlib import Path
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_layernorm.py")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group1():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_data_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_data_parallel", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group1_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_data_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_data_parallel", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group2():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_model_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_model_parallel", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group2_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_model_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_model_parallel", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group3():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_hybrid_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_hybrid_parallel", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group3_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_hybrid_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_hybrid_parallel", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group4():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_all_replicate
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_all_replicated", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_layernorm_group4_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1. test_layernorm_all_replicate
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_layernorm_all_replicated", num_proc=4),
    ])

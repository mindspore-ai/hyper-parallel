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
"""test parallel op squeeze"""
from pathlib import Path
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_squeeze.py")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_squeeze_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_squeeze_basic
        2.test_squeeze_no_args_all_dims
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_squeeze_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_squeeze_no_args_all_dims", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_squeeze_group1_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_squeeze_basic
        2.test_squeeze_no_args_all_dims
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_squeeze_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_squeeze_no_args_all_dims", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_squeeze_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_squeeze_specific_axis_negative
        2.test_squeeze_scalar_like
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_squeeze_specific_axis_negative", num_proc=4),
        TorchCase(IMPL_FILE, "test_squeeze_scalar_like", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_squeeze_group2_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_squeeze_specific_axis_negative
        2.test_squeeze_scalar_like
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_squeeze_specific_axis_negative", num_proc=4),
        TorchCase(IMPL_FILE, "test_squeeze_scalar_like", num_proc=4),
    ])

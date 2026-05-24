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
"""test parallel op expand"""
from pathlib import Path
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

IMPL_FILE = str(Path(__file__).resolve().parent / "_test_parallel_op_expand.py")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group1():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_basic_unsharded
        2.test_expand_scalar_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_basic_unsharded", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_scalar_tensor", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group1_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_basic_unsharded
        2.test_expand_scalar_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_basic_unsharded", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_scalar_tensor", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group2():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_as_basic
        2.test_expand_as_scalar_to_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_as_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_as_scalar_to_tensor", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group2_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_as_basic
        2.test_expand_as_scalar_to_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_as_basic", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_as_scalar_to_tensor", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group3():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_3d
        2.test_expand_prepend_new_dimensions
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_prepend_new_dimensions", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group3_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_3d
        2.test_expand_prepend_new_dimensions
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_3d", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_prepend_new_dimensions", num_proc=4),
    ])

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group4():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_as_3d_preservation
        2.test_expand_as_prepend_dimensions
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_as_3d_preservation", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_as_prepend_dimensions", num_proc=4),
    ])

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_op_expand_group4_gloo():
    """
    Feature: parallel run case in shard
    Description:
        1.test_expand_as_3d_preservation
        2.test_expand_as_prepend_dimensions
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(IMPL_FILE, "test_expand_as_3d_preservation", num_proc=4),
        TorchCase(IMPL_FILE, "test_expand_as_prepend_dimensions", num_proc=4),
    ])

# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test hybrid shard data parallel"""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

HSDP = "hsdp.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_pure_data_parallel():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_pure_data_parallel
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_pure_data_parallel", 12341, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero1_fully_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero1_fully_shard", 12342, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero1_partial_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero1_partial_shard", 12343, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero2_fully_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero2_fully_shard", 12344, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero2_partial_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero2_partial_shard", 12345, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero3_fully_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero3_fully_shard", 12346, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero3_partial_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero3_partial_shard", 12347, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard_with_acc_grad():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero1_fully_shard_with_acc_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero1_fully_shard_with_acc_grad", 12342, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard_with_acc_grad():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero1_partial_shard_with_acc_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero1_partial_shard_with_acc_grad", 12343, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard_with_acc_grad():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero2_fully_shard_with_acc_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero2_fully_shard_with_acc_grad", 12344, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard_with_acc_grad():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero2_partial_shard_with_acc_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero2_partial_shard_with_acc_grad", 12345, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard_with_acc_grad():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero3_fully_shard_with_acc_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero3_fully_shard_with_acc_grad", 12346, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard_with_acc_grad():
    """
    Feature: parallel run case in hsdp
    Description:
        1.test_zero3_partial_shard_with_acc_grad
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(HSDP, "test_zero3_partial_shard_with_acc_grad", 12347, 8)
    ])

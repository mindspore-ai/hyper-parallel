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
"""test hsdp with msrun multi card"""
import os
from tests.common.mark_utils import arg_mark


LOSS_REL_ABSOLUTE_TOL: float = 1e-2
FIRST_STEP_LOSS_REL_ABSOLUTE_TOL: float = 5e-3


def _run_hsdp_case_by_name(case_name: str):
    log = f"log_{case_name}"
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir={log} --join=True --master_port=18181\
            pytest -s hsdp.py::{case_name}"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_pure_dp():
    '''
    Feature: pure data parallel with hsdp api.
    Description: pure data parallel.
    Expectation: Run success
    '''
    case_name = "test_pure_dp"
    _run_hsdp_case_by_name(case_name)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard():
    '''
    Feature: zero1 fully shard data parallel with hsdp api.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero1_fully_shard"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard():
    '''
    Feature: zero1 partial shard data parallel with hsdp api.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero1_partial_shard"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard():
    '''
    Feature: zero2 fully shard data parallel with hsdp api.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero2_fully_shard"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard():
    '''
    Feature: zero2 partial shard data parallel with hsdp api.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero2_partial_shard"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard():
    '''
    Feature: zero3 fully shard data parallel with hsdp api.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_fully_shard"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard():
    '''
    Feature: zero3 partial shard data parallel with hsdp api.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_partial_shard"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_pure_dp_with_acc_grad():
    '''
    Feature: pure data parallel with grad accumulation.
    Description: pure data parallel.
    Expectation: Run success
    '''
    case_name = "test_pure_dp_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard_with_acc_grad():
    '''
    Feature: zero1 fully shard data parallel with grad accumulation.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero1_fully_shard_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard_with_acc_grad():
    '''
    Feature: zero1 partial shard data parallel with grad accumulation.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero1_partial_shard_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard_with_acc_grad():
    '''
    Feature: zero2 fully shard data parallel with grad accumulation.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero2_fully_shard_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard_with_acc_grad():
    '''
    Feature: zero2 partial shard data parallel with grad accumulation.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero2_partial_shard_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard_with_acc_grad():
    '''
    Feature: zero3 fully shard data parallel with grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_fully_shard_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard_with_acc_grad():
    '''
    Feature: zero3 partial shard data parallel with grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_partial_shard_with_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard_with_async_acc_grad():
    '''
    Feature: zero3 partial shard data parallel with async grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_partial_shard_with_async_acc_grad"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_with_comm_fusion():
    '''
    Feature: zero3 partial shard data parallel with comm fusion.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_with_comm_fusion"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_with_comm_fusion_bucket_size():
    '''
    Feature: zero3 with comm fusion bucket size, gradient will be fused to buffer whose size is limited by bucket size.
    Description: zero3 gradient fusion bucket size.
    Expectation: Run success
    '''
    case_name = "test_zero3_with_comm_fusion_bucket_size"
    _run_hsdp_case_by_name(case_name)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_with_comm_fusion_bucket_size0():
    '''
    Feature: zero3 with comm fusion bucket size 0 which mean gradient will not be fused into buffer.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    case_name = "test_zero3_with_comm_fusion_bucket_size0"
    _run_hsdp_case_by_name(case_name)

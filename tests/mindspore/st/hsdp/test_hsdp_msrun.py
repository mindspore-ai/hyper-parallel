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

import inspect
from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case, skip_if_ms_version_lt, skip_if_ms_plugin_not_exist
import pytest


LOSS_REL_ABSOLUTE_TOL: float = 1e-2
FIRST_STEP_LOSS_REL_ABSOLUTE_TOL: float = 5e-3


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_pure_dp(mode):
    '''
    Feature: pure data parallel with hsdp api.
    Description: pure data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero1_fully_shard(mode):
    '''
    Feature: zero1 fully shard data parallel with hsdp api.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero1_partial_shard(mode):
    '''
    Feature: zero1 partial shard data parallel with hsdp api.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero2_fully_shard(mode):
    '''
    Feature: zero2 fully shard data parallel with hsdp api.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero2_partial_shard(mode):
    '''
    Feature: zero2 partial shard data parallel with hsdp api.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_fully_shard(mode):
    '''
    Feature: zero3 fully shard data parallel with hsdp api.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_partial_shard(mode):
    '''
    Feature: zero3 partial shard data parallel with hsdp api.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_pure_dp_with_acc_grad(mode):
    '''
    Feature: pure data parallel with grad accumulation.
    Description: pure data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero1_fully_shard_with_acc_grad(mode):
    '''
    Feature: zero1 fully shard data parallel with grad accumulation.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero1_partial_shard_with_acc_grad(mode):
    '''
    Feature: zero1 partial shard data parallel with grad accumulation.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero2_fully_shard_with_acc_grad(mode):
    '''
    Feature: zero2 fully shard data parallel with grad accumulation.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", ["eager", pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1")])])
def test_zero2_partial_shard_with_acc_grad(mode):
    '''
    Feature: zero2 partial shard data parallel with grad accumulation.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_fully_shard_with_acc_grad(mode):
    '''
    Feature: zero3 fully shard data parallel with grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_partial_shard_with_acc_grad(mode):
    '''
    Feature: zero3 partial shard data parallel with grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_partial_shard_with_async_acc_grad(mode):
    '''
    Feature: zero3 partial shard data parallel with async grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_with_comm_fusion(mode):
    '''
    Feature: zero3 partial shard data parallel with comm fusion.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_with_comm_fusion_bucket_size(mode):
    '''
    Feature: zero3 with comm fusion bucket size, gradient will be fused to buffer whose size is limited by bucket size.
    Description: zero3 gradient fusion bucket size.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("mode", [
    "eager",
    pytest.param("jit_ast", marks=[skip_if_ms_version_lt("2.8.1"), skip_if_ms_plugin_not_exist()])
])
def test_zero3_with_comm_fusion_bucket_size0(mode):
    '''
    Feature: zero3 with comm fusion bucket size 0 which mean gradient will not be fused into buffer.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    glob_v = 2
    file_name = "hsdp.py"
    master_port = 18181
    case_name = f"{inspect.stack()[0].function}[{mode}]"
    msrun_case(glob_v, file_name, case_name, master_port)

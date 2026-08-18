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
"""test DCP async save + load"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_ASYNC_SAVE = "dcp_async_save.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_async_save_load():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_async_save_load
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_ASYNC_SAVE, "test_dcp_async_save_load", 12404, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_async_save_load_with_storage_comm():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_async_save_load_with_storage_comm
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_ASYNC_SAVE, "test_dcp_async_save_load_with_storage_comm", 12405, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_async_save_load_with_gloo_comm():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_async_save_load_with_gloo_comm
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_ASYNC_SAVE, "test_dcp_async_save_load_with_gloo_comm", 12406, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_async_save_twice_reuses_the_plan_cache():
    """
    Feature: parallel run case in distributed_checkpoint
    Description:
        1.test_dcp_async_save_twice_reuses_the_plan_cache
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_ASYNC_SAVE, "test_dcp_async_save_twice_reuses_the_plan_cache", 12407, 4),
    ])

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

from tests.mindspore.tensor_parallel.utils import run_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_flash_attention_score_model_parallel():
    '''
    Feature: sum operator.
    Description: Test cell shard in python.
    Expectation: Run success.
    '''
    file_name = "flash_attention_score_shard_in_python.py"
    case_name = "test_flash_attention_score_model_parallel"
    master_port = 11298
    run_case(file_name, case_name, master_port)

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
"""test checkpoint DCP loads of safetensors checkpoints written outside DCP"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run

DCP_HF_LOAD = "dcp_hf_load.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_hf_load_group1() -> None:
    """
    Feature: parallel run cases for DCP loads of safetensors checkpoints written outside DCP.
    Description:
        1.test_dcp_hf_load_resharded
        2.test_dcp_torch_sharded_load_resharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_HF_LOAD, "test_dcp_hf_load_resharded", 12262, 4),
        TorchCase(DCP_HF_LOAD, "test_dcp_torch_sharded_load_resharded", 12263, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_hf_load_group1_gloo() -> None:
    """
    Feature: parallel run cases for DCP loads of safetensors checkpoints written outside DCP.
    Description:
        1.test_dcp_hf_load_resharded
        2.test_dcp_torch_sharded_load_resharded
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_HF_LOAD, "test_dcp_hf_load_resharded", num_proc=4),
        TorchCase(DCP_HF_LOAD, "test_dcp_torch_sharded_load_resharded", num_proc=4),
    ])

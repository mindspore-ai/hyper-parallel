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
"""test saver loader"""
import os

from hyper_parallel.core.checkpoint.loader import load_checkpoint
from hyper_parallel.core.checkpoint.saver import save_checkpoint
from tests.common.mark_utils import arg_mark
from tests.torch.common_net import SimpleModel


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_save_load_checkpoint():
    """
    Feature: Test checkpoint saver and loader.
    Description: Test using saver to save checkpoint to safetensors file, and then using loader to load checkpoint from
    this safetensors file.
    Expectation: Run success.
    """
    model = SimpleModel().npu()
    original_state_dict = model.state_dict()
    save_data = {
        "model_state_dict": original_state_dict,
    }

    # save checkpoint
    file_path = "tmp1.safetensors"
    save_checkpoint(save_data, file_path)

    # load checkpoint
    load_dict = load_checkpoint(file_path)
    os.remove(file_path)
    assert isinstance(load_dict, dict)

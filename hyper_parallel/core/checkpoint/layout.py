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
"""save load layout"""
import json
from hyper_parallel.platform import get_platform


platform = get_platform()


def get_current_layout(cell):
    param_dict = platform.parameters_dict(cell)
    layout_dict = {}
    for name, param in param_dict.items():
        if name in layout_dict:
            raise RuntimError("param in cell can not have same name")
        if param.layout:
            layout_dict[param.name] = param.layout.to_dict()

    return layout_dict


def save_layout(layout_dict, file_path):
    # todo: check and create file path
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(layout_dict, f, ensure_ascii=False)


def load_layout(file_path):
    # todo check path
    with open(file_path, 'r', encoding='utf-8') as f:
        param_layout_dict = json.load(f)
    return param_layout_dict

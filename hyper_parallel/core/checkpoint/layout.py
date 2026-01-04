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
import os
from pathlib import Path
from typing import Union

from hyper_parallel.platform import get_platform

platform = get_platform()


def get_current_layout(cell):
    """
    Get current layout from cell
    Args:
        cell: instance of Cell in MindSpore or Module in PyTorch

    Returns:
        dict: A dictionary where keys are rank IDs and values are dictionaries
            mapping parameter names to their layout information, including
            data type and full shape.
    """
    layout_dict = {}
    rank_id = str(platform.get_rank())
    layout_dict[rank_id] = {}

    param_dict = platform.parameters_dict(cell)
    for name, param in param_dict.items():
        if name in layout_dict:
            raise RuntimeError("param in cell can not have same name")
        if param.layout:
            layout_dict[rank_id][param.name] = param.layout.to_dict()
            layout_dict[rank_id][param.name]["type"] = str(param.dtype)
            layout_dict[rank_id][param.name]["full_shape"] = param.shape

    return layout_dict


def save_layout(layout_dict: dict, file_path: Union[Path, str]) -> None:
    """
    Save layout to file
    """
    # todo: check and create file path
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(layout_dict, f, ensure_ascii=False)


def load_layout(file_path: Union[Path, str]) -> dict:
    """
    Load layout from file
    """
    # todo check path
    with open(file_path, 'r', encoding='utf-8') as f:
        param_layout_dict = json.load(f)
    return param_layout_dict


def combine_layout(directory: Union[Path, str]) -> dict:
    """
    Combines layout files from the specified directory into a single layout dictionary.

    This function scans the given directory for files with a '.layout' extension,
    loads each layout file, and merges them into one dictionary.

    Args:
        directory (Union[Path, str]): The directory to scan for layout files.

    Returns:
        dict: A dictionary containing the combined layout information keyed by rank ID.

    Raises:
        RuntimeError: If duplicate rank IDs are found across the layout files.

    Note:
        Only processes files with '.layout' extension.
    """
    layout_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.layout'):
            load_dict = load_layout(os.path.join(directory, filename))
            for rank_id, param_dict in load_dict.items():
                if rank_id in layout_dict:
                    raise ValueError("rank_id in files must be unique")
                layout_dict[rank_id] = param_dict

    return layout_dict

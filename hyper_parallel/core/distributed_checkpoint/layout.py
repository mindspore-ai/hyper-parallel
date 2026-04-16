# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Layout I/O utilities for saving, loading and gathering layout information."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Union

from hyper_parallel.platform import get_platform

platform = get_platform()
logger = logging.getLogger(__name__)


def get_current_layout(cell: Any) -> dict:
    """
    Get current layout from cell
    Args:
        cell (Any): Instance of Cell (model/network object).

    Returns:
        dict: A dictionary where keys are rank IDs and values are dictionaries
            mapping parameter names to their layout information, including
            data type and full shape.
    """
    current_rank = str(platform.get_rank())
    layout_dict = {current_rank: {}}

    params_without_layout_attr = []
    param_dict = platform.parameters_dict(cell)
    for name, param in param_dict:
        if name in layout_dict:
            raise RuntimeError("param in cell can not have same name")
        if not hasattr(param, "layout"):
            params_without_layout_attr.append(name)
            continue
        if param.layout:
            layout_info = dict(param.layout.to_dict())
            if "mesh_shape" in layout_info:
                layout_info["device_matrix"] = layout_info.pop("mesh_shape")
            layout_dict[current_rank][param.name] = layout_info
            layout_dict[current_rank][param.name]["type"] = str(param.dtype)
            layout_dict[current_rank][param.name]["full_shape"] = param.shape

    if params_without_layout_attr:
        logger.info(
            "The following parameters have no layout attribute and were skipped: %s",
            params_without_layout_attr,
        )

    return layout_dict


def save_layout(layout_dict: dict, file_path: Union[Path, str]) -> None:
    """
    Save layout to file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(layout_dict, f, ensure_ascii=False)


def load_layout(file_path: Union[Path, str]) -> dict:
    """
    Load layout from file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Layout file not found: {file_path}")
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


def get_global_layout(cell: Any) -> dict:
    """
    Get global layout information from all ranks, and gather them into a dict.

    Args:
        cell (Any): Instance of Cell (model/network object).

    Return:
        dict: A dictionary containing the global layout information keyed by rank ID.
    """
    # global layout
    global_layout_dict = {}

    # prepare empty global_layout_list
    global_layout_list = []
    world_size = platform.get_world_size()
    for _ in range(world_size):
        global_layout_list.append(None)

    # local layout
    local_layout = get_current_layout(cell)

    # all gather object
    platform.all_gather_object(global_layout_list, local_layout)

    # cast list to dict
    for layout_dict in global_layout_list:
        global_layout_dict.update(layout_dict)

    return global_layout_dict

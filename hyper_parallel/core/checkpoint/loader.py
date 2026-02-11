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
"""Load model parameters from checkpoint file."""
from pathlib import Path
from typing import Union

from hyper_parallel.core.checkpoint.util import has_valid_filename
from hyper_parallel.platform import get_platform

platform = get_platform()


def load_checkpoint(file_path: Union[Path, str]) -> dict:
    """
    Load model parameters from checkpoint file.

    Args:
        file_path (Union[Path, str]): Path to the checkpoint file. Path must contain file name with extension.

    Returns:
        dict: A dictionary containing the loaded parameters.
    """
    file_path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    if not has_valid_filename(file_path_obj):
        raise ValueError(f"Loader file_path should contains valid filename but get: {file_path}")

    param_dict = platform.load_checkpoint(str(file_path_obj))
    return param_dict

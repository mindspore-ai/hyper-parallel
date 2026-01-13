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
"""Save checkpoint to safetensors."""
from pathlib import Path
from typing import Union

from hyper_parallel.core.checkpoint.util import check_path, has_valid_filename
from hyper_parallel.platform import get_platform

platform = get_platform()


def save_checkpoint(save_obj, file_path: Union[Path, str]) -> None:
    """
    Save model checkpoint parameters to a file using safetensors format.

    Args:
        save_obj: Net instance containing parameters to save. Can be a Cell or a Module.
        file_path (Union[Path, str]): The path or file to save checkpoint parameters. Path must contain the file name
        with extension.
    """
    file_path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    if not has_valid_filename(file_path_obj):
        raise ValueError(f"Saver file_path should contains valid filename but get: {file_path}")

    # Create parent directory
    check_path(file_path_obj.parent)
    platform.save_checkpoint(save_obj, str(file_path_obj))

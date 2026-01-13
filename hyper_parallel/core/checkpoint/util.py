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
"""Common utility functions."""
from pathlib import Path
from typing import Union


def check_path(path: Union[Path, str]) -> None:
    """
    Check whether path is existing or not.

    Args:
        path (Union[Path, str]): path to check. Can only a file name in current directory, a pure directory, or a file
        name with directory. When path contains a directory, the function will check whether the directory exists, if
        not, the directory will be created.
    """
    path_obj = Path(path) if isinstance(path, str) else path

    if path_obj.exists():
        return

    if path_obj.suffix:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        path_obj.mkdir(parents=True, exist_ok=True)


def has_valid_filename(path: Path) -> bool:
    """
    Check whether path has valid filename. A filename should contain name and suffix, name and suffix must contain
    letters, and then can have numbers and underscores.

    Args:
        path (Path): path to check.

    Return:
        bool: whether path has a valid filename.
    """
    conditions = (
        path.name,
        path.suffix,
        len(path.suffix) > 1,
        path.stem,
        any(c.isalpha() for c in path.stem),
        any(c.isalpha() for c in path.suffix[1:])
    )
    return all(conditions)

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

"""
MindSpore custom graph optimization passes.
This package contains C++ implementations compiled into libhyper_parallel_mindspore.so.
"""
import warnings
import os
from pathlib import Path

# Check for SO file existence (for diagnostics)
MS_SO_NAME = "libhyper_parallel_mindspore.so"
MS_SO_PATH = Path(__file__).parent / MS_SO_NAME

if not MS_SO_PATH.exists() and "READTHEDOCS" not in os.environ:
    warnings.warn(
        f"Mindspore custom pass library not found at {MS_SO_PATH}. "
        "Graph optimizations for MindSpore will be disabled.",
        ImportWarning
    )

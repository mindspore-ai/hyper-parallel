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
"""Bootstrap helpers shared by training entry scripts."""

import os
from importlib import import_module


def maybe_import_torch_npu() -> None:
    """Import ``torch_npu`` early so it can register the HCCL backend."""
    ascend_env_markers = (
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        os.environ.get("ASCEND_VISIBLE_DEVICES"),
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    )
    if not any(ascend_env_markers) and not os.path.exists("/usr/local/Ascend/ascend-toolkit"):
        return
    try:
        import_module("torch_npu")
    except ModuleNotFoundError:
        return

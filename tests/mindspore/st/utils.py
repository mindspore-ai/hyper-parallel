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
"""MindSpore worker-side test utils (imports mindspore — not for launchers).

For spawning ``msrun`` from a launcher / ``parallel_case`` wrapper, import
:func:`tests.common.distributed_launcher.msrun_case` instead. That module
intentionally avoids importing ``mindspore``.
"""
import pytest
import mindspore as ms
from packaging import version


def skip_if_ms_version_lt(min_ms_version):
    """
    Skip if mindspore version less then `min_ms_version`.
    """
    return pytest.mark.skipif(
        version.parse(ms.__version__) < version.parse(min_ms_version),
        reason=f"Requires MindSpore >= {min_ms_version}, but got {ms.__version__}"
    )


def skip_if_ms_version_ge(max_ms_version):
    """
    Skip if mindspore version greater than or equal to `max_ms_version`.
    """
    return pytest.mark.skipif(
        version.parse(ms.__version__) >= version.parse(max_ms_version),
        reason=f"Test broken on MindSpore >= {max_ms_version}, got {ms.__version__}"
    )

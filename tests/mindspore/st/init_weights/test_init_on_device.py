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
"""One-card init_on_device API checks (MindSpore).

Heavy imports are deferred into test bodies so collecting this module alone
does not load MindSpore during unrelated ST launcher collection when pytest
is pointed at sibling launchers only. Running these tests still loads MS.
"""

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_init_on_device_include_buffers_true_raises() -> None:
    """MindSpore backend should reject include_buffers=True."""
    # pylint: disable=C0415
    import pytest
    from hyper_parallel.core.dtensor.init_weights import init_on_device

    with pytest.raises(ValueError, match="does not support include_buffers=True"):
        with init_on_device("meta", include_buffers=True):
            pass


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_init_on_device_invalid_device_raises() -> None:
    """MindSpore backend should reject unsupported external device values."""
    # pylint: disable=C0415
    import pytest
    from hyper_parallel.core.dtensor.init_weights import init_on_device

    with pytest.raises(ValueError, match='only "npu", "cpu", and "meta" are allowed'):
        with init_on_device("Ascend:0"):
            pass

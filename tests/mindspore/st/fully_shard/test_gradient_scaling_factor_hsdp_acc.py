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
"""Launch MindSpore fully_shard set_gradient_scaling_factor HSDP + grad-acc ST."""
import pytest

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

_FILE_NAME = "_test_gradient_scaling_factor_hsdp_acc.py"
_SKIP_REASON = (
    "Temporarily skipped: the case body passes (verified locally), but MindSpore "
    "msrun workers may core dump during process teardown after replicate_params "
    "cases complete -- same known issue as test_fully_shard_replicate_prefetch_"
    "regression.py. The torch counterpart (test_gradient_scaling_factor_hsdp_acc.py) "
    "still gates this feature. Remove the skip once the teardown core dump is fixed."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_gradient_scaling_factor_hsdp_acc():
    """
    Feature: fully_shard set_gradient_scaling_factor under HSDP + gradient
        accumulation + replicate_params (pure all-reduce path), MindSpore.
    Description: Launch a 4-card msrun case on a 2x2 HSDP mesh; verify the scaling
        factor scales every parameter's accumulated gradient exactly once,
        including replicate params that only go through all-reduce.
    Expectation: Run success.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_gradient_scaling_factor_hsdp_grad_accumulation",
        18535,
        worker_num=4,
        local_worker_num=4,
    )

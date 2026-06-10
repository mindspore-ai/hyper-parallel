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
"""Launch MindSpore fully_shard replicate_params prefetch regression ST."""
import pytest

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case

_FILE_NAME = "_test_fully_shard_replicate_prefetch_regression.py"
_SKIP_REASON = (
    "Temporarily skipped: the regression body passes, but MindSpore msrun workers "
    "may core dump during process teardown after replicate_params cases complete."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_replicate_params_backward_prefetch_regression():
    """
    Feature: fully_shard replicate_params with backward prefetch (MindSpore).
    Description: Launch a 4-card msrun case that creates the mixed state introduced by
        ``replicate_params`` and then triggers backward prefetch on the already-materialized child.
    Expectation: Run success without the historical ``Expected sharded_state ... got UNSHARDED``
        assertion during prefetch.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_fully_shard_replicate_params_backward_prefetch_regression",
        18533,
        worker_num=4,
        local_worker_num=4,
    )


@pytest.mark.skip(reason=_SKIP_REASON)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ms_hsdp_replicate_dtensor_state_visible_after_backward():
    """
    Feature: HSDP replicate_params with a TP-sharded DTensor state (MindSpore).
    Description: Launch a 4-card msrun case that updates a max_logits-like
        replicate parameter in forward and reads it after HSDP post-backward reshard.
    Expectation: Callback-style post-step read observes the updated non-zero value.
    """
    msrun_case(
        2,
        _FILE_NAME,
        "test_ms_hsdp_replicate_dtensor_state_visible_after_backward",
        18534,
        worker_num=4,
        local_worker_num=4,
    )

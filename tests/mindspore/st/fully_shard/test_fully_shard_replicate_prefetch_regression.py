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
import os

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_replicate_prefetch_regression.py")
_SKIP_REASON = (
    "Temporarily skipped: the case body passes, but MindSpore msrun workers may "
    "segfault during process teardown after the replicate_params backward prefetch "
    "case completes. Remove the skip once the teardown core dump is fixed."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_ms_fully_shard_replicate_prefetch_regression_suite():
    """
    Feature: fully_shard replicate_params prefetch and TP DTensor state regression (MindSpore).
    Description:
        1. test_ms_fully_shard_replicate_params_backward_prefetch_regression (2 card)
           - replicate_params + backward prefetch mixed state.
        2. test_ms_hsdp_replicate_dtensor_state_visible_after_backward (4 card)
           - TP DTensor replicate state read after backward.
    Expectation: Run success without historical assertion errors.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE,
                      "test_ms_fully_shard_replicate_params_backward_prefetch_regression",
                      worker_num=2, local_worker_num=2),
        MindSporeCase(_TEST_FILE,
                      "test_ms_hsdp_replicate_dtensor_state_visible_after_backward",
                      worker_num=4, local_worker_num=4),
    ])

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
"""Declare the DeepSeek-V4.1 recompute regions used by normal training."""

from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import (
    configure_shared_state,
)


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer is one region and every CSA2 layer replays. The
    model-owned hook then pins each replayed index source's discrete selection as
    an exclusion region, so a replay neither chooses a different key set nor
    publishes its state twice; published cross-layer entries simply stay alive
    until the graph drops the state in backward.
    """
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        on_applied=configure_shared_state,
    )


__all__ = ["build_recompute_policy"]

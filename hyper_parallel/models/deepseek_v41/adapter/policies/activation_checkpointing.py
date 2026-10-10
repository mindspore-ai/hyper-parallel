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
    enable_early_release,
)


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer is one region, and the CSA2 attention modules inside it
    are excluded: attention publishes and consumes the per-forward shared state
    and launches CP collectives, so it must not be replayed. The cost of that
    exclusion is attention's own saved activations, while the rest of the layer
    (mHC, MoE, norms, Engram, and the in-layer mixing ops) is recomputed. The
    model-owned hook then enables releasing published state as soon as its last
    consumer has read it.
    """
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        exclude_patterns=("model.layers.*.self_attn",),
        on_applied=enable_early_release,
    )


__all__ = ["build_recompute_policy"]

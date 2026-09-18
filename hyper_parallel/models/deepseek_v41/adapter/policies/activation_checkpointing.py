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
"""Declare activation-checkpoint-safe DeepSeek-V4.1 regions."""

from hyper_parallel.models.adapter_spec import RecomputePolicy


def build_recompute_policy() -> RecomputePolicy:
    """Return V4.1 checkpoint-safe regions used by normal training."""
    return RecomputePolicy(
        safe_module_patterns=(
            "model.layers.*.input_layernorm",
            "model.layers.*.post_attention_layernorm",
            "model.layers.*.mlp",
        ),
        no_replay_module_patterns=("model.layers.*.self_attn",),
    )


__all__ = ["build_recompute_policy"]

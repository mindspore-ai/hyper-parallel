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
"""Compatibility entry point for pinned images with legacy plugin metadata."""

from rl.roles.rollout.vllm_plugin import (
    HYPER_QWEN3_5_ARCHITECTURE,
    register_hyper_models,
)

__all__ = ["HYPER_QWEN3_5_ARCHITECTURE", "register_hyper_models"]

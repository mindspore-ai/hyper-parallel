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
"""Wan model-family adapter declarations."""

from hyper_parallel.models.wan.adapter.registration import WAN_ADAPTER_SPEC
from hyper_parallel.models.wan.adapter.replacements import replace_wan_attention_backend

__all__ = [
    "WAN_ADAPTER_SPEC",
    "replace_wan_attention_backend",
]

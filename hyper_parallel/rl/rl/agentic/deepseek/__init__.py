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
"""DeepSeek Harness black-box rollout integration."""

from rl.agentic.deepseek.gateway import DeepSeekGateway
from rl.agentic.deepseek.harness import (
    DeepSeekAgentProgram,
    DeepSeekProgramFactory,
    DeepSeekRuntime,
)
from rl.agentic.deepseek.protocol import DeepSeekChatProtocol
from rl.agentic.deepseek.trajectory import build_deepseek_trajectory

__all__ = [
    "DeepSeekAgentProgram",
    "DeepSeekChatProtocol",
    "DeepSeekGateway",
    "DeepSeekProgramFactory",
    "DeepSeekRuntime",
    "build_deepseek_trajectory",
]

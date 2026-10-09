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
"""Optional model scoring service; task reward functions live in examples."""

from rl.reward_model.client import RewardModelClient
from rl.reward_model.scoring import load_reward_function, score_model_batch, scorer_fingerprint

__all__ = ["RewardModelClient", "load_reward_function", "score_model_batch", "scorer_fingerprint"]

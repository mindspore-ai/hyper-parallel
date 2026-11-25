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
"""Synthetic alternating rewards for CP update/publication tests, not GSM8K accuracy."""

from typing import Any

from rl.agentic.core.types import EpisodeContext
from rl.agentic.envs.environment import ENVIRONMENTS
from examples.gsm8k.agent import GSM8KSingleTurnEnvironment


class CPUpdateEnvironment(GSM8KSingleTurnEnvironment):
    """Keep actual generated tokens while assigning explicit nonzero test advantages."""

    def score_reward(self, prompt: Any, **kwargs: Any) -> float:
        """Use the fixture's sample index as the declared reward rule."""
        del prompt, kwargs
        return 1.0 if self.episode.sample_index % 2 == 0 else -1.0


@ENVIRONMENTS.register('cp_update_fixture')
def build_environment(context: EpisodeContext) -> CPUpdateEnvironment:
    """Construct the test task without changing the production GRPO pipeline."""
    return CPUpdateEnvironment(context)

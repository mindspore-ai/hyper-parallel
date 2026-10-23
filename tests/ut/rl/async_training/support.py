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
"""Canonical real trajectory fixture shared by asynchronous contracts."""
import torch

from rl.async_training.protocol import SampleGroup, SamplingTask
from rl.dataset.contracts import Trajectory, Turn


def group(task: SamplingTask, candidates: int = 2) -> SampleGroup:
    """Build complete candidates with distinct rewards and valid sampled tokens."""
    rows = tuple(Trajectory(
        trajectory_id=f"{task.identity}:{index}", prompt_id=str(task.identity), group_id=str(task.identity),
        policy_version=task.policy_version, worker_policy_version=task.policy_version,
        turns=(Turn("user", "question", 0, 1, False), Turn("assistant", "answer", 1, 3, True)),
        token_ids=torch.tensor([1, 2 + index, 3]), attention_mask=torch.ones(3, dtype=torch.bool),
        action_mask=torch.tensor([False, True, True]), rollout_log_probs=torch.tensor([-0.7, -0.8]),
        reward=float(index % 2), reward_components={"task": float(index % 2)}, done=True, truncated=False,
        terminal_reason="completed",
    ) for index in range(candidates))
    return SampleGroup(task, rows, 0.1)

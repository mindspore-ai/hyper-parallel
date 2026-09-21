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
"""Reject invalid token alignment and stale rollout versions before optimization."""

from dataclasses import replace
import unittest

import torch

from rl.dataset.contracts import ExperienceBatch, Trajectory, Turn


class TestTokenContracts(unittest.TestCase):
    """Exercise the contracts shared by internal and external Agent trajectories."""

    def setUp(self) -> None:
        """Build a right-padded CPU trajectory with two trainable action tokens."""
        self.trajectory = Trajectory(
            trajectory_id="t", prompt_id="p", group_id="g", policy_version=2,
            turns=(Turn("user", "question", 0, 1, False), Turn("assistant", "answer", 1, 3, True)),
            token_ids=torch.tensor([10, 11, 12, 0]),
            attention_mask=torch.tensor([True, True, True, False]),
            action_mask=torch.tensor([False, True, True, False]),
            rollout_log_probs=torch.tensor([-0.1, -0.2, 0.0]),
            reward=1.0, reward_components={"answer": 1.0}, done=True,
            truncated=False, terminal_reason="completed", worker_policy_version=2,
        )
        self.batch = ExperienceBatch(
            trajectories=(self.trajectory,), sequences=self.trajectory.token_ids.unsqueeze(0),
            attention_mask=self.trajectory.attention_mask.unsqueeze(0),
            action_mask=self.trajectory.action_mask.unsqueeze(0), rewards=torch.tensor([1.0]),
            old_log_probs=self.trajectory.rollout_log_probs.unsqueeze(0),
            responses=("answer",), generation_seconds=0.0, worker_policy_version=2,
        )

    def test_loss_mask_selects_next_token_actions_only(self) -> None:
        """Prompt and padding positions must never contribute to the policy loss."""
        torch.testing.assert_close(self.batch.loss_action_mask, torch.tensor([[True, True, False]]))

    def test_trajectory_rejects_misalignment_and_stale_versions(self) -> None:
        """Both runner types must honor the same published policy and token contract."""
        cases = (
            ({"policy_version": -1}, "non-negative"),
            ({"worker_policy_version": 1}, "worker policy version"),
            ({"attention_mask": torch.ones(3)}, "attention_mask"),
            ({"action_mask": torch.zeros(3)}, "action_mask"),
            ({"action_mask": torch.tensor([False, True, True, True])}, "padding"),
            ({"rollout_log_probs": torch.zeros(4)}, "next-token"),
        )
        for changes, error in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                replace(self.trajectory, **changes)

    def test_experience_rejects_invalid_sequence_fields(self) -> None:
        """Invalid batches fail before a policy forward or collective starts."""
        cases = (
            ({"sequences": torch.zeros(4)}, "rank two"),
            ({"attention_mask": torch.ones(1, 3)}, "attention_mask"),
            ({"action_mask": torch.zeros(1, 3)}, "action_mask"),
            ({"action_mask": torch.tensor([[False, True, True, True]])}, "padding"),
            ({"action_mask": torch.tensor([[True, True, True, False]])}, "first sequence token"),
            ({"rewards": torch.ones(2)}, "one value per sequence"),
            ({"bootstrap_values": torch.ones(2)}, "bootstrap_values"),
            ({"worker_policy_version": 1}, "every trajectory"),
        )
        for changes, error in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                replace(self.batch, **changes)

    def test_all_training_targets_use_next_token_dimensions(self) -> None:
        """PPO and GRPO targets share the same [batch, sequence - 1] dimensions."""
        for field in ("old_log_probs", "advantages", "returns", "values", "reference_log_probs"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} must align"):
                    replace(self.batch, **{field: torch.zeros(1, 4)})
                accepted = replace(self.batch, **{field: torch.zeros(1, 3)})
                self.assertEqual(getattr(accepted, field).shape, (1, 3))

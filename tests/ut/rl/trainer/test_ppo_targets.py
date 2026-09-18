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
"""PPO target mathematics and complete Actor/Critic checkpoint ownership."""
# White-box regression tests intentionally exercise internal state and lifecycle hooks.
# pylint: disable=protected-access

import pickle
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from rl import checkpoint
from rl.algorithm.advantage import GAEAdvantageEstimator
from rl.dataset.batch_builder import get_bootstrap_values, normalize_advantages

from .test_checkpoint import _manager, _Stateful


class TestPPOTargets(unittest.TestCase):
    """Use hand-computed examples that distinguish terminal and truncated targets."""

    def test_gae_bootstrap_and_observation_gap(self) -> None:
        """Truncation carries value across observation gaps instead of resetting GAE."""
        estimator = GAEAdvantageEstimator(gamma=0.9, gae_lambda=0.8, normalize=False)
        values = torch.tensor([[0.5, 99.0, 1.0]])
        mask = torch.tensor([[True, False, True]])
        result = estimator.estimate(torch.tensor([2.0]), mask, values=values,
                                    bootstrap_values=torch.tensor([3.0]))
        torch.testing.assert_close(result.advantages, torch.tensor([[3.064, 0.0, 3.7]]))
        torch.testing.assert_close(result.returns, torch.tensor([[3.564, 0.0, 4.7]]))
        terminal = estimator.estimate(torch.tensor([2.0]), mask, values=values)
        torch.testing.assert_close(terminal.returns, torch.tensor([[1.62, 0.0, 2.0]]))

    def test_normalization_uses_global_dp_statistics(self) -> None:
        """Unequal local statistics must still produce global batch whitening."""
        local = torch.tensor([[1.0, 2.0, 100.0]])
        mask = torch.tensor([[True, True, False]])
        def all_reduce(stats: Any, group: Any) -> None:
            """Add statistics of the other DP rank's two action tokens."""
            assert group == "dp"
            stats.add_(torch.tensor([7.0, 25.0, 2.0]))
        with patch("rl.dataset.batch_builder.dist.all_reduce", side_effect=all_reduce) as reduce:
            actual = normalize_advantages(local, mask, SimpleNamespace(rank_size=2, group="dp"))
        expected = ((local - 2.5) / (torch.tensor(1.25).sqrt() + 1e-6)) * mask
        torch.testing.assert_close(actual, expected)
        self.assertEqual(reduce.call_count, 1)

    def test_bootstrap_rejects_incomplete_context(self) -> None:
        """Only explicitly complete nonterminal states may bootstrap."""
        trajectory = SimpleNamespace(done=False, truncated=True, metadata={})
        batch = SimpleNamespace(trajectories=(trajectory,), attention_mask=torch.tensor([[1, 1, 0]]))
        values = torch.tensor([[1.0, 2.0, 99.0]])
        with self.assertRaisesRegex(ValueError, "complete bootstrap context"):
            get_bootstrap_values(batch, values)
        trajectory.metadata["bootstrap_context_complete"] = True
        torch.testing.assert_close(get_bootstrap_values(batch, values), torch.tensor([2.0]))
        trajectory.done = True
        torch.testing.assert_close(get_bootstrap_values(batch, values), torch.tensor([0.0]))


class TestPPOCheckpoint(unittest.TestCase):
    """Restore independently mutated Actor/Critic and both optimizer states."""

    def test_both_roles_survive_resume(self) -> None:
        """Persist and restore both role models and their independent optimizer states."""
        with tempfile.TemporaryDirectory() as directory:
            manager, trainer = _manager(Path(directory))
            trainer.critic = SimpleNamespace(critic_model=_Stateful("critic-live"),
                                            optimizer=_Stateful("critic-optim-live"),
                                            lr_scheduler=_Stateful("critic-scheduler-live"))
            state = SimpleNamespace(global_step=1, epoch=2, consumed_samples=4, consumed_tokens=32)
            saved = {}
            def save(data: Any, *, checkpoint_id: Any, **_kwargs: Any) -> None:
                """Capture the distributed and rank-local checkpoint payloads."""
                saved[str(checkpoint_id)] = deepcopy(data)
            def load(data: Any, *, checkpoint_id: Any, **_kwargs: Any) -> None:
                """Restore captured payloads with the checkpoint byte-reader contract."""
                values = deepcopy(saved[str(checkpoint_id)])
                if "runtime" in values:
                    values["runtime"] = pickle.loads(values["runtime"])
                data.update(values)
            with patch.object(checkpoint, "dcp_save", side_effect=save), \
                 patch.object(checkpoint, "dcp_load", side_effect=load), \
                 patch.object(checkpoint.dist, "get_rank", return_value=0), \
                 patch.object(checkpoint.dist, "get_world_size", return_value=1), \
                 patch.object(checkpoint.torch, "get_rng_state", return_value=b"rng"), \
                 patch.object(checkpoint.torch, "set_rng_state"):
                manager._save(state)
                trainer.model.value = "changed"
                trainer.optimizer.value = "changed"
                trainer.critic.critic_model.value = "changed"
                trainer.critic.optimizer.value = "changed"
                trainer.critic.lr_scheduler.value = "changed"
                manager.begin(state)
            self.assertEqual(trainer.model.value, "model-live")
            self.assertEqual(trainer.optimizer.value, "optimizer-live")
            self.assertEqual(trainer.critic.critic_model.value, "critic-live")
            self.assertEqual(trainer.critic.optimizer.value, "critic-optim-live")
            self.assertEqual(trainer.critic.lr_scheduler.value, "critic-scheduler-live")

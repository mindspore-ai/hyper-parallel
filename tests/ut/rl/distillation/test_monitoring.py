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
"""Probability rejection and token-weighted teacher monitoring contracts."""

# Tests inspect package-owned state and stage boundaries without public runtime hooks.
# pylint: disable=protected-access

import unittest
from typing import Any
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rl.trainer.sync import SyncTrainer
from rl.distillation.config import DistillationConfig, parse_distillation_config
from rl.distillation.monitoring import DistillationMetricAccumulator, validate_probability_gate
from tests.ut.rl.distillation.test_distillation import batch, configuration


class TestDistillationMonitoring(unittest.TestCase):
    """Check failures before optimizer updates and mixed-length teacher accounting."""

    def test_gate_masks_padding_and_observations(self) -> None:
        """Invalid scores at non-action positions must not contaminate acceptance."""
        experience = batch()
        current = experience.old_log_probs.clone()
        current[~experience.loss_action_mask.bool()] = float("nan")
        experience = replace(experience, old_log_probs=current)
        metrics = validate_probability_gate(experience, DistillationConfig(), group=None, size=1)
        self.assertEqual(metrics["distillation/pre_update_logprob_diff_max"], 0)

    def test_gate_rejects_bad_alignment_and_nonfinite(self) -> None:
        """Finite means alone cannot conceal an outlier or broad ratio-tail mismatch."""
        for value in (-5.0, float("nan"), float("inf"), 0.1, -1.7):
            with self.subTest(value=value):
                experience = replace(batch(), old_log_probs=torch.full((2, 3), value))
                with self.assertRaisesRegex(RuntimeError, "no optimizer update allowed"):
                    validate_probability_gate(experience, DistillationConfig(), group=None, size=1)

    def test_remote_rank_rejection_propagates(self) -> None:
        """A passing local batch still rejects when another rank reports invalid probabilities."""
        def collective(tensor: torch.Tensor, **kwargs: Any) -> None:
            """Inject a remote rank rejection into the acceptance reduction."""
            del kwargs
            if tensor.dtype == torch.int32:
                tensor.zero_()
        with patch("rl.distillation.monitoring.dist.all_reduce", side_effect=collective) as reduce:
            with self.assertRaisesRegex(RuntimeError, "finite=False"):
                validate_probability_gate(batch(), DistillationConfig(), group=object(), size=2)
        self.assertEqual(reduce.call_count, 3)

    def test_invalid_gate_limits(self) -> None:
        """Thresholds must be explicit finite nonnegative numbers with bounded fractions."""
        for gate in ({"max_logprob_diff": float("nan")}, {"max_ratio_outside_fraction": 1.1},
                     {"enabled": False}, {"max_mean_logprob_diff": True}):
            value = configuration()
            value["probability_gate"] = gate
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                parse_distillation_config(value)

    def test_empty_padding_rank_uses_global_tokens(self) -> None:
        """An empty local rank must participate without rejecting another rank's valid tokens."""
        experience = replace(batch(), action_mask=torch.zeros((2, 4), dtype=torch.bool))

        def collective(tensor: torch.Tensor, **kwargs: Any) -> None:
            """Inject a valid peer contribution for an empty local rank."""
            if tensor.numel() == 3:
                tensor.copy_(torch.tensor([0.03, 3.0, 0.0]))
            elif kwargs.get("op") == torch.distributed.ReduceOp.MAX:
                tensor.fill_(0.02)

        with patch("rl.distillation.monitoring.dist.all_reduce", side_effect=collective):
            result = validate_probability_gate(experience, DistillationConfig(), group=None, size=2)
        self.assertAlmostEqual(result["distillation/pre_update_logprob_diff_mean"], 0.01)

    def test_dual_clip_and_saturation_have_separate_metrics(self) -> None:
        """A negative advantage with an extreme ratio activates dual clipping and numerical protection."""
        config = DistillationConfig(enabled=True)
        current = torch.tensor([[-1.0]])
        accumulator = DistillationMetricAccumulator(current, config)
        accumulator.add(current, torch.tensor([[-22.0]]), torch.tensor([[-4.0]]),
                        torch.ones((1, 1), dtype=torch.bool), ("teacher",))
        result = accumulator.finalize(group=None, size=1)
        self.assertEqual(result["distillation/log_ratio_max"], 21.0)
        self.assertEqual(result["distillation/ratio_outside_fraction"], 1.0)
        self.assertEqual(result["distillation/log_ratio_saturation_fraction"], 1.0)
        self.assertEqual(result["distillation/dual_clip_fraction"], 1.0)
        self.assertEqual(result["distillation/clip_fraction"], 0.0)
        self.assertEqual(result["distillation/loss"], 9.0)

    def test_teacher_token_weighting_and_microbatch_invariance(self) -> None:
        """Equal sample counts can have unequal token contributions; slicing must preserve totals."""
        config = DistillationConfig(enabled=True, teachers=(("a", None), ("b", None)))
        old = torch.full((2, 3), -2.0)
        current = old + torch.tensor([[0.0, 0.3, -0.3], [0.1, 9.0, 9.0]])
        teacher = torch.full_like(old, -1.0)
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        full = DistillationMetricAccumulator(old, config)
        full.add(current, old, teacher, mask, ("a", "b"))
        split = DistillationMetricAccumulator(old, config)
        for index, key in enumerate(("a", "b")):
            split.add(current[index:index+1], old[index:index+1], teacher[index:index+1],
                      mask[index:index+1], (key,))
        result = full.finalize(group=None, size=1)
        self.assertEqual(result, split.finalize(group=None, size=1))
        self.assertEqual(result["distillation/teacher/a/token_fraction"], 0.75)
        self.assertEqual(result["distillation/teacher/b/token_fraction"], 0.25)
        self.assertEqual(result["distillation/ratio_outside_fraction"], 0.5)
        self.assertEqual(result["distillation/clip_fraction"], 0.25)
        self.assertAlmostEqual(result["distillation/loss"], sum(result[f"distillation/teacher/{key}/weighted_loss"]
                                                               for key in ("a", "b")))

    def test_trainer_rejects_before_actor_critic_and_publication(self) -> None:
        """Exercise the real step pipeline and prove rejected scores cannot reach updates."""
        trainer = object.__new__(SyncTrainer)
        trainer.reward_scorer = None
        trainer.residency = SimpleNamespace(move=lambda **kw: None, release=lambda: None, offload_manager=None, metrics={})
        experience = replace(batch(), old_log_probs=torch.full((2, 3), -5.0))
        trainer.state = SimpleNamespace(global_step=0)
        trainer.device = torch.device("cpu")
        trainer._log_steps = 1
        trainer.evaluator = None
        trainer.parallel_dims = SimpleNamespace(dp_size=1)
        trainer._dp_group_info = SimpleNamespace(group=None, rank_size=1)
        trainer.distillation_config = DistillationConfig(enabled=True)
        trainer.teacher_manager = Mock()
        trainer._run_rank_synchronized = lambda name, callback: callback()
        trainer.rollout_manager = SimpleNamespace(generate=Mock(return_value=experience))
        trainer.rollout_engine = Mock()
        trainer.actor = Mock()
        trainer.critic = Mock()
        trainer.teacher_scorer = SimpleNamespace(score=Mock(return_value=experience))
        trainer.targets = SimpleNamespace(prepare=Mock(return_value=(experience, {})))
        trainer.residency.release = Mock()
        trainer._publish_policy = Mock()
        trainer._complete_step = Mock()
        inputs = {"sample_indices": [0], "input_ids": torch.ones((1, 1), dtype=torch.long),
                  "attention_mask": torch.ones((1, 1), dtype=torch.bool)}
        with (
            patch("rl.trainer.sync.dist.get_rank", return_value=0),
            patch("rl.trainer.sync.build_prompt_records", return_value=[]),
            patch("rl.trainer.sync.pad_agent_call_batch_for_dp", return_value=experience),
        ):
            with self.assertRaisesRegex(RuntimeError, "pre-update probability gate rejected"):
                trainer._train_step(inputs)
        trainer.actor.update.assert_not_called()
        trainer.critic.update.assert_not_called()
        trainer._publish_policy.assert_not_called()
        trainer._complete_step.assert_not_called()
        self.assertEqual(trainer.state.global_step, 0)

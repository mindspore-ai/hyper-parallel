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
"""Closed-form probability, gradient and group semantics for frozen correction."""
from dataclasses import asdict, replace
import math
import unittest
from unittest.mock import patch

import torch

from rl.algorithm import build_algorithm
from rl.algorithm.rollout_correction import correction_metrics, prepare_rollout_correction
from rl.config.schema import ColocateAsyncConfig, RolloutCorrectionConfig
from rl.workers.actor import Actor
from tests.ut.rl.colocate_async.support import trainer_fixture


class TestRolloutCorrection(unittest.TestCase):
    """Use real trajectory data and independent analytic loss expectations."""

    def setUp(self) -> None:
        """Construct sampled behavior scores with deliberately unequal action weights."""
        trainer, _, payloads = trainer_fixture()
        self.batch = trainer._build_local_batch(payloads)
        behavior = torch.tensor([[-4., -2.], [-2., -.2]])
        self.batch = replace(self.batch, old_log_probs=behavior, rollout_log_probs=behavior.clone(),
                             advantages=torch.tensor([[1., 1.], [-1., -1.]]))
        self.anchor = torch.tensor([[-2., -3.], [-2., -.2]])
        self.config = RolloutCorrectionConfig(mode="decoupled")

    def test_detached_weights_preserve_behavior_and_freeze_anchor(self):
        corrected, analysis = prepare_rollout_correction(self.batch, self.anchor, self.config, 7)
        expected = torch.tensor([[2., math.exp(-1.)], [1., 1.]])
        torch.testing.assert_close(analysis.weights, expected)
        self.assertFalse(analysis.weights.requires_grad)
        self.assertEqual(corrected.rollout_log_probs.data_ptr(), self.batch.rollout_log_probs.data_ptr())
        torch.testing.assert_close(corrected.old_log_probs, self.anchor)
        self.assertEqual(corrected.metadata["off_policy_anchor_version"], 7)
        self.assertEqual(analysis.rejected_groups, ())

    def test_metrics_only_preserves_loss_inputs(self):
        config = replace(self.config, mode="metrics_only")
        corrected, _ = prepare_rollout_correction(self.batch, self.anchor, config, 7)
        self.assertEqual(corrected.old_log_probs.data_ptr(), self.batch.old_log_probs.data_ptr())
        self.assertIsNone(corrected.rollout_is_weights)

    def test_contract_rejects_malformed_correction_weights(self):
        corrected, _ = prepare_rollout_correction(self.batch, self.anchor, self.config, 1)
        invalid = (corrected.rollout_is_weights.half(),
                   corrected.rollout_is_weights.clone().requires_grad_(),
                   torch.full_like(corrected.rollout_is_weights, float("nan")),
                   -corrected.rollout_is_weights,
                   corrected.rollout_is_weights[:, :1])
        for weights in invalid:
            with self.subTest(dtype=weights.dtype, shape=tuple(weights.shape)), self.assertRaises(ValueError):
                replace(corrected, rollout_is_weights=weights)

    def test_weighted_ppo_gradient_matches_independent_formula(self):
        """Verify weighted ppo gradient matches independent formula."""
        corrected, _ = prepare_rollout_correction(self.batch, self.anchor, self.config, 1)
        algorithm = build_algorithm({"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"})
        current = (self.anchor + .05).requires_grad_()
        output = algorithm.compute_actor_loss(current, corrected.old_log_probs, None,
                                              corrected.advantages, corrected.loss_action_mask,
                                              policy_loss_weights=corrected.rollout_is_weights)
        ratio = (current - self.anchor).exp()
        advantage = self.batch.advantages
        expected = (torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.2))
                    * torch.tensor([[2., math.exp(-1.)], [1., 1.]])).sum()
        torch.testing.assert_close(output.total_loss_sum, expected)
        actual_grad = torch.autograd.grad(output.total_loss_sum, current, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, current)[0])

    def test_equal_policies_reduce_to_original_ppo(self):
        corrected, _ = prepare_rollout_correction(self.batch, self.batch.rollout_log_probs, self.config, 1)
        algorithm = build_algorithm({"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"})
        current = (corrected.old_log_probs + .01).requires_grad_()
        args = (current, corrected.old_log_probs, None, corrected.advantages, corrected.loss_action_mask)
        original = algorithm.compute_actor_loss(*args)
        weighted = algorithm.compute_actor_loss(*args, policy_loss_weights=corrected.rollout_is_weights)
        self.assertTrue(torch.equal(original.total_loss_sum, weighted.total_loss_sum))
        first = torch.autograd.grad(original.total_loss_sum, current, retain_graph=True)[0]
        self.assertTrue(torch.equal(first, torch.autograd.grad(weighted.total_loss_sum, current)[0]))

    def test_anchor_and_weights_stay_fixed_across_epochs_and_optimizer_steps(self):
        """Verify anchor and weights stay fixed across epochs and optimizer steps."""
        model = TinyCategoricalPolicy()
        algorithm = build_algorithm({"name": "grpo", "kl_coef": 0., "loss_aggregation": "token-mean"})
        actor = Actor(model, algorithm, 1, optimizer=torch.optim.SGD(model.parameters(), lr=.05, foreach=False),
                      device=torch.device("cpu"), response_mini_batch_size=1, update_epochs=2, max_grad_norm=0.)
        corrected, _ = prepare_rollout_correction(self.batch, actor.compute_log_probs(self.batch), self.config, 1)
        anchor, weights = corrected.old_log_probs.clone(), corrected.rollout_is_weights.clone()
        with patch.object(algorithm, "compute_actor_loss", wraps=algorithm.compute_actor_loss) as loss:
            update = actor.update(corrected)
        self.assertEqual(update.optimizer_steps, 4)
        self.assertEqual(loss.call_count, 4)
        for index, call in enumerate(loss.call_args_list):
            torch.testing.assert_close(call.kwargs["old_log_probs"], anchor[index % 2:index % 2 + 1])
        torch.testing.assert_close(corrected.old_log_probs, anchor)
        torch.testing.assert_close(corrected.rollout_is_weights, weights)
        self.assertGreater(model.table.abs().sum().item(), 0.)

    def test_correction_does_not_reweight_reference_kl(self):
        corrected, _ = prepare_rollout_correction(self.batch, self.anchor, self.config, 1)
        algorithm = build_algorithm({"name": "grpo", "kl_coef": .1, "loss_aggregation": "token-mean"})
        args = (self.anchor + .1, self.anchor, self.anchor - .2,
                self.batch.advantages, self.batch.loss_action_mask)
        original = algorithm.compute_actor_loss(*args)
        weighted = algorithm.compute_actor_loss(*args, policy_loss_weights=corrected.rollout_is_weights)
        torch.testing.assert_close(original.regularization_loss_sum, weighted.regularization_loss_sum)
        torch.testing.assert_close(weighted.total_loss_sum,
                                   weighted.policy_loss_sum + .1 * weighted.regularization_loss_sum)

    def test_prompt_and_padding_are_excluded_from_diagnostics(self):
        batch = replace(self.batch, trajectories=(), action_mask=torch.tensor([[False, True, False],
                                                                             [False, False, False]]))
        anchor = torch.tensor([[-2., float("nan")], [float("nan"), float("nan")]])
        corrected, analysis = prepare_rollout_correction(batch, anchor, self.config, 1)
        metrics = correction_metrics(analysis, batch.loss_action_mask, self.config)
        self.assertEqual(metrics["off_policy/action_tokens"], 1)
        self.assertEqual(metrics["off_policy/is_ess"], 1)
        self.assertEqual(corrected.rollout_is_weights.count_nonzero().item(), 1)

    def test_ess_and_clipping_use_active_token_weights(self):
        _, analysis = prepare_rollout_correction(self.batch, self.anchor, self.config, 1)
        metrics = correction_metrics(analysis, self.batch.loss_action_mask, self.config)
        weights = torch.tensor([2., math.exp(-1.), 1., 1.])
        self.assertAlmostEqual(metrics["off_policy/is_ess"], (weights.sum().square()/weights.square().sum()).item(),
                               places=5)
        self.assertEqual(metrics["off_policy/is_weight_clip_fraction"], .25)
        self.assertEqual(metrics["off_policy/is_weight_max"], 2.)
        self.assertAlmostEqual(metrics["off_policy/is_weight_p95_approx"], 2.)

    def test_extreme_finite_ratios_do_not_overflow_weights(self):
        batch = replace(self.batch, rollout_log_probs=torch.full((2, 2), -10000.))
        _, analysis = prepare_rollout_correction(batch, self.anchor, self.config, 1)
        self.assertTrue(bool(analysis.weights.isfinite().all()))
        self.assertEqual(analysis.weights.max().item(), 2.)
        metrics = correction_metrics(analysis, batch.loss_action_mask, self.config)
        self.assertEqual(metrics["off_policy/diagnostic_clamp_fraction"], 1.)

    def test_bad_scores_fail_instead_of_becoming_rejected_samples(self):
        for value in (float("nan"), float("inf"), .1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_rollout_correction(self.batch, torch.full((2, 2), value), self.config, 1)

    def test_one_outlier_rejects_its_complete_grpo_group(self):
        config = replace(self.config, rejection="group", max_abs_log_ratio=1.5)
        _, analysis = prepare_rollout_correction(self.batch, self.anchor, config, 1)
        self.assertEqual(analysis.rejected_groups, ("0",))

    def test_invalid_controls_and_sampler_configuration_are_separate(self):
        for values in ({"mode": "unknown"}, {"importance_sampling": "episode"},
                       {"max_importance_weight": True}, {"max_importance_weight": float("inf")},
                       {"max_importance_weight": 1.e100},
                       {"rejection": "group"}, {"max_refill_rounds": -1}, {"unknown": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RolloutCorrectionConfig.from_mapping(values)
        settings = ColocateAsyncConfig.from_mapping({"off_policy": {"mode": "decoupled"}})
        self.assertNotIn("off_policy", asdict(settings.sampling_config(2)))

    def test_rejection_exhaustion_never_updates_or_publishes(self):
        """Verify rejection exhaustion never updates or publishes."""
        trainer, _, _ = trainer_fixture()
        trainer.settings = replace(trainer.settings, off_policy=replace(self.config, rejection="group",
                                    max_abs_log_ratio=.5, max_refill_rounds=0))
        trainer._correct_experience = lambda batch: (batch, {}, ("0",))
        with patch("rl.trainer.colocate_async.coordinator_call",
                   side_effect=lambda _, callback: callback()):
            with self.assertRaisesRegex(RuntimeError, "max_refill_rounds"):
                trainer.train()
        trainer.actor.update.assert_not_called()
        trainer.publication.publish.assert_not_called()
        trainer.rollouter.commit_batch.assert_not_called()
        trainer.rollouter.reject_batch_groups.assert_called_once_with(("0",))
        self.assertEqual(trainer.state.global_step, 0)

    def test_refill_resumes_same_version_then_updates_once(self):
        """Verify refill resumes same version then updates once."""
        trainer, events, _ = trainer_fixture()
        trainer.settings = replace(trainer.settings, off_policy=self.config)
        trainer._correct_experience = MockCorrection()
        with patch("rl.trainer.colocate_async.coordinator_call",
                   side_effect=lambda _, callback: callback()):
            trainer.train()
        self.assertEqual(trainer.state.global_step, 1)
        trainer.actor.update.assert_called_once()
        trainer.publication.publish.assert_called_once_with(2)
        self.assertLess(events.index("partial_resume_done"), events.index("update_start"))
        self.assertEqual(trainer.rollouter.resume.call_args_list[0].args, (1,))
        trainer.rollouter.commit_batch.assert_called_once()


class MockCorrection:
    """Reject one pre-update lease, then allow the next complete batch."""

    def __init__(self) -> None:
        """Start with one pending artificial rejection."""
        self.first = True

    def __call__(self, batch: object) -> tuple:
        """Return the untouched batch and a deterministic rejection decision."""
        rejected = ("0",) if self.first else ()
        self.first = False
        return batch, {}, rejected


class TinyCategoricalPolicy(torch.nn.Module):
    """Small trainable CPU policy with real autoregressive probabilities."""

    def __init__(self) -> None:
        """Start with a uniform eight-token categorical distribution."""
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(8, 8))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                use_cache: bool = False) -> dict[str, torch.Tensor]:
        """Return trainable logits at each recorded input token."""
        del attention_mask, use_cache
        return {"logits": self.table[input_ids]}

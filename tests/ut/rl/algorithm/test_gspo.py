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
"""CPU tests for GSPO math, Reference ownership, and Actor normalization."""
import math
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch
import yaml

from examples import gsm8k
from rl.algorithm import GSPOAlgorithm, GSPOConfig, build_algorithm
from rl.algorithm.loss import GSPOPolicyObjective
from rl.dataset.batch_builder import ExperiencePreparer
from rl.dataset.contracts import ExperienceBatch
from rl.roles.policy.actor import Actor
from rl.utils.monitoring.metrics import ActorMetricAccumulator, ActorMicroBatchMetrics, build_training_metrics


def _algorithm(name: str = "gspo", **kwargs):
    """Build one recipe with its required aggregation configuration."""
    return build_algorithm({
        "name": name,
        "loss_aggregation": "seq-mean-token-mean" if name == "gspo" else "token-mean",
        "kl_coef": 0.0,
        **kwargs,
    })


def _experience() -> ExperienceBatch:
    """Create unequal action lengths, disjoint actions, and an empty trajectory."""
    mask = torch.tensor([
        [True, False, False, False],
        [True, True, True, False],
        [False, True, False, True],
        [False, False, False, False],
    ])
    sequences = torch.zeros((4, 5), dtype=torch.long)
    sequences[:, 0] = torch.arange(4)
    advantages = torch.tensor([1.0, -1.0, 0.5, 0.0]).unsqueeze(-1).expand_as(mask) * mask
    return ExperienceBatch(
        trajectories=(),
        sequences=sequences,
        attention_mask=torch.ones_like(sequences, dtype=torch.bool),
        action_mask=torch.cat((torch.zeros((4, 1), dtype=torch.bool), mask), dim=-1),
        rewards=torch.tensor([1.0, 0.0, 0.5, 0.0]),
        old_log_probs=torch.full((4, 4), -2.0),
        responses=("a", "b", "c", ""),
        generation_seconds=0.0,
        advantages=advantages,
    )


class _TableActor(Actor):
    """Use a small trainable logprob table to exercise real Actor backward."""

    def sequence_log_probs(self, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Select complete padded trajectories without depending on a model backend."""
        del attention_mask
        return self.actor_model.weight[sequences[:, 0]]


def _actor(micro_batch_size: int = 4, dp_size: int = 1, name: str = "gspo", epochs: int = 1):
    """Create an SGD Actor whose gradient and parameter changes have a closed form."""
    model = torch.nn.Embedding(4, 4)
    with torch.no_grad():
        model.weight.fill_(-2.0)
    return _TableActor(
        model, _algorithm(name), micro_batch_size,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1, foreach=False),
        device=torch.device("cpu"), dp_size=dp_size,
        dp_group_info=SimpleNamespace(group="dp-only"),
        response_mini_batch_size=4, max_grad_norm=0.0, update_epochs=epochs,
    )


def _cpu_optimizer_context() -> ExitStack:
    """Mock only hardware orchestration, retaining Torch backward and SGD."""
    stack = ExitStack()
    stack.enter_context(patch("rl.roles.policy.actor.hsdp_sync_stream"))
    stack.enter_context(patch("rl.roles.policy.actor.SkipDTensorDispatch", nullcontext))
    stack.enter_context(patch("rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_))
    return stack


class TestGSPOObjective(unittest.TestCase):
    """Check first-order math against an independently expressed sequence loss."""

    def test_loss_and_gradient_match_sequence_oracle(self):
        """Cover both clipped directions and both non-clipped advantage signs."""
        mask = torch.tensor([
            [True, False, False], [True, True, False], [True, False, True],
            [True, True, True], [True, True, False], [True, False, False],
        ])
        ratios = torch.tensor([1.4, 0.6, 0.6, 1.4, 1.05, 1.0])
        seq_adv = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.7, 0.0])
        old = torch.full(mask.shape, -3.0, requires_grad=True)
        current = (old.detach() + ratios.log().unsqueeze(-1)).requires_grad_()
        advantages = seq_adv.unsqueeze(-1).expand_as(mask).clone().requires_grad_()
        algorithm = _algorithm(clip_ratio_low=0.1, clip_ratio_high=0.2)
        output = algorithm.compute_actor_loss(current, old, None, advantages, mask)
        loss = output.total_loss_sum / 6
        gradient, = torch.autograd.grad(loss, current)

        oracle_current = current.detach().clone().requires_grad_()
        sequence_ratio = torch.exp(
            ((oracle_current - old.detach()) * mask).sum(dim=-1) / mask.sum(dim=-1)
        )
        oracle = -torch.minimum(
            sequence_ratio * seq_adv, sequence_ratio.clamp(0.9, 1.2) * seq_adv,
        ).mean()
        oracle_gradient, = torch.autograd.grad(oracle, oracle_current)
        torch.testing.assert_close(loss, oracle)
        torch.testing.assert_close(gradient, oracle_gradient)
        self.assertEqual(output.clipped_sequence_count.item(), 2)
        self.assertEqual(output.clipped_token_count.item(), 3)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_geometric_mean_and_non_action_values(self):
        """Noncontiguous action ratios 2 and 0.5 yield one sequence ratio of 1."""
        current = torch.tensor([[math.log(2), float("nan"), math.log(0.5)]], requires_grad=True)
        old = torch.tensor([[0.0, float("inf"), 0.0]])
        mask = torch.tensor([[True, False, True]])
        output = _algorithm().compute_actor_loss(current, old, None, torch.ones_like(current), mask)
        torch.testing.assert_close(output.policy_loss_sum, torch.tensor(-1.0))
        output.total_loss_sum.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-0.5, 0.0, -0.5]]))
        self.assertTrue(output.old_policy_kl_sum.isfinite())

    def test_empty_actions_remain_differentiable(self):
        """A fully masked micro-batch has finite zero loss connected to current logprob."""
        current = torch.full((2, 3), float("nan"), requires_grad=True)
        output = _algorithm().compute_actor_loss(
            current, current.detach(), None, torch.full_like(current, float("inf")),
            torch.zeros_like(current, dtype=torch.bool),
        )
        output.total_loss_sum.backward()
        self.assertEqual(output.total_loss_sum.item(), 0)
        self.assertEqual(output.valid_token_count.item(), 0)
        torch.testing.assert_close(current.grad, torch.zeros_like(current))

    def test_log_cap_and_bfloat16(self):
        """Clamping log ratios caps negative-advantage loss and stops saturated gradients."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                current = torch.tensor([[9.0], [11.0]], dtype=dtype, requires_grad=True)
                output = _algorithm().compute_actor_loss(
                    current, torch.zeros_like(current), None, -torch.ones_like(current),
                    torch.ones_like(current, dtype=torch.bool),
                )
                self.assertEqual(output.total_loss_sum.dtype, torch.float32)
                output.total_loss_sum.backward()
                self.assertGreater(current.grad[0].item(), 0)
                self.assertEqual(current.grad[1].item(), 0)
                torch.testing.assert_close(output.total_loss_sum, torch.tensor(math.exp(9) + math.exp(10)))

    def test_sequence_averaging_and_optional_kl(self):
        """Unequal lengths get equal sequence weight, including Reference KL."""
        mask = torch.tensor([[True, False, False], [True, True, True]])
        current = torch.zeros((2, 3), requires_grad=True)
        advantages = torch.tensor([[1.0] * 3, [-1.0] * 3])
        reference = torch.full_like(current, math.log(2), requires_grad=True)
        output = _algorithm(kl_coef=0.5).compute_actor_loss(
            current, current.detach(), reference, advantages, mask,
        )
        self.assertEqual(output.policy_loss_sum.item(), 0)
        torch.testing.assert_close(output.regularization_loss_sum, torch.tensor(2 * (1 - math.log(2))))
        output.total_loss_sum.backward()
        self.assertIsNone(reference.grad)

    def test_objective_requires_aligned_mask(self):
        """Missing masks and mismatched inputs fail at the objective boundary."""
        values = torch.zeros((2, 3))
        objective = GSPOPolicyObjective()
        with self.assertRaisesRegex(ValueError, "action_mask"):
            objective.compute(values, values, values)
        with self.assertRaisesRegex(ValueError, "align"):
            objective.compute(values, values[:, :2], values, action_mask=values.bool())

    def test_targets_and_critic_contract(self):
        """GSPO uses complete reward groups and has no value targets."""
        algorithm = _algorithm()
        targets = algorithm.build_targets(
            torch.tensor([1.0, 3.0, 7.0, 7.0]), torch.ones((4, 3), dtype=torch.bool),
            ("a", "a", "b", "b"),
        )
        self.assertIsNone(targets.returns)
        torch.testing.assert_close(targets.advantages[:, 0], torch.tensor([-1, 1, 0, 0]) / math.sqrt(2))
        with self.assertRaisesRegex(RuntimeError, "Critic"):
            algorithm.compute_critic_loss(None, None, None, None)


class TestReferenceRequirements(unittest.TestCase):
    """All built-in algorithms derive Reference ownership from their own KL coefficient."""

    def test_zero_and_positive_kl_are_instance_local(self):
        """PPO retains its Critic even when no Reference is needed."""
        for name in ("grpo", "ppo", "gspo"):
            with self.subTest(name=name):
                zero, positive = _algorithm(name), _algorithm(name, kl_coef=0.1)
                self.assertFalse(zero.requirements.roles.reference)
                self.assertFalse(zero.requirements.data.reference_log_probs)
                self.assertTrue(positive.requirements.roles.reference)
                self.assertTrue(positive.requirements.data.reference_log_probs)
                self.assertEqual(zero.requirements.roles.critic, name == "ppo")
                self.assertEqual(zero.requirements.data.values, name == "ppo")
                self.assertEqual(zero.requirements.data.returns, name == "ppo")

    def test_zero_kl_ignores_reference_and_preserves_old_policy_diagnostics(self):
        """A supplied NaN reference is ignored rather than multiplied by zero."""
        current = torch.zeros((2, 3), requires_grad=True)
        old = torch.full_like(current, -0.01)
        mask = torch.ones_like(current, dtype=torch.bool)
        for name in ("grpo", "ppo", "gspo"):
            with self.subTest(name=name):
                algorithm = _algorithm(name)
                absent = algorithm.compute_actor_loss(current, old, None, torch.ones_like(current), mask)
                ignored = algorithm.compute_actor_loss(
                    current, old, torch.full_like(current, float("nan")), torch.ones_like(current), mask,
                )
                torch.testing.assert_close(absent.total_loss_sum, ignored.total_loss_sum)
                self.assertEqual(absent.regularization_loss_sum.item(), 0)
                self.assertGreater(absent.old_policy_kl_sum.item(), 0)
                with self.assertRaisesRegex(ValueError, "reference"):
                    _algorithm(name, kl_coef=0.1).compute_actor_loss(
                        current, old, None, torch.ones_like(current), mask,
                    )

    def test_invalid_coefficients_and_gspo_fields(self):
        """Reject invalid ownership coefficients and fields copied from another recipe."""
        for name in ("grpo", "ppo", "gspo"):
            for coefficient in (-1.0, float("nan"), float("inf"), True):
                with self.subTest(name=name, coefficient=coefficient), self.assertRaises(ValueError):
                    _algorithm(name, kl_coef=coefficient)
        for field, value in (
            ("clip_ratio_low", 1.0), ("clip_ratio_low", -0.1), ("clip_ratio_high", float("inf")),
            ("advantage_epsilon", 0), ("clip_ratio_c", 3), ("dual_clip", 3),
            ("loss_aggregation", "token-mean"), ("kl_type", "other"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                _algorithm(**{field: value})
        with self.assertRaises(ValueError):
            GSPOConfig(kl_coef=float("nan"))

    def test_preparer_without_reference(self):
        """Zero-KL target preparation accepts missing Reference for all recipes."""
        experience = _experience()
        for name in ("grpo", "ppo", "gspo"):
            with self.subTest(name=name):
                values = torch.zeros((4, 4)) if name == "ppo" else None
                result = ExperiencePreparer(_algorithm(name)).prepare(experience, values=values)
                self.assertIsNone(result.reference_log_probs)
                self.assertEqual(result.returns is not None, name == "ppo")
                self.assertFalse(result.advantages.requires_grad)

    def test_shipped_recipe_builds_gspo(self):
        """The complete YAML selects GSPO without obsolete dual-clip fields."""
        path = Path(gsm8k.__file__).resolve().parent / "configs/qwen3_4b_gsm8k_gspo.yaml"
        config = yaml.safe_load(path.read_text())
        algorithm = build_algorithm(config["algorithm"])
        self.assertIsInstance(algorithm, GSPOAlgorithm)
        self.assertFalse(algorithm.requirements.roles.reference)
        self.assertGreaterEqual(config["rollout"]["num_return_sequences"], 2)


class TestGSPOActor(unittest.TestCase):
    """Exercise SGD updates and global scaling without real process groups."""

    def test_microbatch_splits_match_closed_form_update(self):
        """The same minibatch yields identical updates even with an empty micro-batch."""
        experience = _experience()
        mask = experience.loss_action_mask
        expected = torch.full((4, 4), -2.0) + (
            0.1 * experience.advantages / mask.sum(dim=-1).clamp_min(1).unsqueeze(-1) / 3
        )
        for size in (1, 2, 3, 4):
            with self.subTest(size=size), _cpu_optimizer_context():
                actor = _actor(size)
                metrics = actor.update(experience)
                torch.testing.assert_close(actor.actor_model.weight, expected)
                self.assertEqual(metrics.valid_tokens, 6)
                self.assertEqual(metrics.valid_sequences, 3)
                self.assertEqual(metrics.optimizer_steps, 1)
                self.assertAlmostEqual(metrics.policy_loss, -1 / 6)

    def test_dp_partition_gradient_matches_complete_batch(self):
        """Uneven action counts across ranks use one global sequence denominator."""
        experience = _experience()
        complete = _actor()
        complete.forward_backward(experience, 0, 4, global_tokens=6, global_sequences=3)
        gradients = []
        for start, end in ((0, 1), (1, 4)):
            actor = _actor(dp_size=2)
            with patch("rl.roles.policy.actor.dist.all_reduce") as reduce:
                actor.forward_backward(experience, start, end, global_tokens=6, global_sequences=3)
                self.assertEqual(reduce.call_args.kwargs["group"], "dp-only")
            gradients.append(actor.actor_model.weight.grad)
        torch.testing.assert_close((gradients[0] + gradients[1]) / 2, complete.actor_model.weight.grad)

    def test_counts_use_dp_group_and_exclude_empty_rows(self):
        """The count collective combines tokens and sequences, excluding TP duplicates."""
        actor = _actor(dp_size=2)

        def add_remote(counts: torch.Tensor, *, group: str) -> None:
            """Supply a differently sized remote partition to the count reduction."""
            self.assertEqual(group, "dp-only")
            torch.testing.assert_close(counts, torch.tensor([6, 3]))
            counts.add_(torch.tensor([9, 2]))

        with patch("rl.roles.policy.actor.dist.all_reduce", side_effect=add_remote):
            counts = actor._global_batch_counts(  # pylint: disable=protected-access
                _experience().loss_action_mask
            )
            self.assertEqual(counts, (15, 5))

    def test_nonfinite_and_empty_global_batches_fail_before_step(self):
        """Fixed invalid targets or current logits cannot reach optimizer.step."""
        experience = _experience()
        actor = _actor()
        for batch in (
            replace(experience, rewards=torch.full((4,), float("nan"))),
            replace(experience, old_log_probs=torch.full((4, 4), float("inf"))),
            replace(experience, action_mask=torch.zeros((4, 5), dtype=torch.bool)),
        ):
            with self.subTest(batch=batch), patch.object(actor.optimizer, "step") as step:
                with self.assertRaises((ValueError, RuntimeError)):
                    actor.update(batch)
                step.assert_not_called()
        with torch.no_grad():
            actor.actor_model.weight[0, 0] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            actor.forward_backward(experience, 0, 4, global_tokens=6, global_sequences=3)

    def test_remote_preflight_failure_propagates(self):
        """A valid rank stops when another rank reports invalid fixed targets."""
        actor = _actor(dp_size=2)

        def remote_failure(errors: list, local_error: Optional[str], *, group: str) -> None:
            """Report a failure from the other simulated data-parallel rank."""
            self.assertIsNone(local_error)
            self.assertEqual(group, "dp-only")
            errors[:] = [None, "non-finite rewards"]

        with patch("rl.roles.policy.actor.dist.all_gather_object", side_effect=remote_failure):
            with self.assertRaisesRegex(ValueError, "non-finite rewards"):
                actor.update(_experience())

    def test_grpo_and_ppo_still_use_token_normalization(self):
        """Reference-free legacy recipes keep the original token denominator."""
        experience = _experience()
        expected = torch.full((4, 4), -2.0) + 0.1 * experience.advantages / 6
        for name in ("grpo", "ppo"):
            with self.subTest(name=name), _cpu_optimizer_context():
                actor = _actor(2, name=name)
                actor.update(experience)
                torch.testing.assert_close(actor.actor_model.weight, expected)

    def test_dp_update_preserves_optimizer_and_sync_schedule(self):
        """Exercise the new count and preflight interfaces with complete DP mocks."""
        actor = _actor(2, dp_size=2, epochs=2)
        count_inputs = []

        def gather(errors: list, local_error: Optional[str], *, group: str) -> None:
            """Both simulated ranks pass fixed-target validation."""
            self.assertEqual(group, "dp-only")
            self.assertIsNone(local_error)
            errors[:] = [None, None]

        def reduce(tensor: torch.Tensor, *, group: str, op: object = None) -> None:
            """Model identical DP partitions without reducing a success flag by sum."""
            self.assertEqual(group, "dp-only")
            if op is None:
                if tensor.numel() == 2:
                    count_inputs.append(tensor.tolist())
                tensor.mul_(2)

        with _cpu_optimizer_context(), \
                patch("rl.roles.policy.actor.dist.all_gather_object", side_effect=gather), \
                patch("rl.roles.policy.actor.dist.all_reduce", side_effect=reduce), \
                patch.object(actor, "_set_gradient_sync") as sync:
            metrics = actor.update(_experience())
        self.assertEqual(count_inputs, [[6, 3], [6, 3]])
        self.assertEqual([call.args[0] for call in sync.call_args_list], [False, True, False, True])
        self.assertEqual(metrics.valid_tokens, 24)
        self.assertEqual(metrics.valid_sequences, 12)
        self.assertEqual(metrics.optimizer_steps, 2)


class TestGSPOMetrics(unittest.TestCase):
    """Keep sequence loss means separate from token-weighted diagnostics."""

    def test_mixed_denominators_and_dp_totals(self):
        """Unequal batches are count-weighted and global counts are not reduced twice."""
        accumulator = ActorMetricAccumulator.create(
            torch.tensor(0.0), dp_group_info=SimpleNamespace(group="dp-only"), dp_size=2,
            loss_aggregation="seq-mean-token-mean",
        )
        accumulator.add_micro_batch(ActorMicroBatchMetrics(
            *[torch.tensor(value) for value in (4, 2, 1, 8, 12, 4)], clipped_sequence_count=torch.tensor(1.0),
        ))
        accumulator.add_optimizer_step(global_tokens=8, global_sequences=3, gradient_norm=2)
        accumulator.add_optimizer_step(global_tokens=12, global_sequences=2, gradient_norm=4)

        def add_remote(totals: torch.Tensor, *, group: str) -> None:
            """Add unequal remote sums without changing the global denominators."""
            self.assertEqual(group, "dp-only")
            totals.add_(torch.tensor([6, 3, 2, 2, 8, 6, 2]))

        with patch("rl.utils.monitoring.metrics.dist.all_reduce", side_effect=add_remote):
            metrics = accumulator.finalize(learning_rate=0.1)
        self.assertEqual(metrics.total_loss, 2)
        self.assertEqual(metrics.policy_loss, 1)
        self.assertAlmostEqual(metrics.kl_loss, 0.6)
        self.assertEqual(metrics.old_policy_kl, 0.5)
        self.assertEqual(metrics.old_current_log_ratio_abs, 1)
        self.assertEqual(metrics.clip_fraction, 0.5)
        self.assertAlmostEqual(metrics.sequence_clip_fraction, 0.6)
        self.assertEqual(metrics.valid_tokens, 20)
        self.assertEqual(metrics.valid_sequences, 5)
        with patch("rl.utils.monitoring.metrics._system_memory_metrics", return_value={}):
            public = build_training_metrics(step=1, actor_update=metrics, rollout_metrics={})
        self.assertEqual(public["train/valid_sequences"], 5)
        self.assertAlmostEqual(public["train/sequence_clip_fraction"], 0.6)

    def test_epochs_count_training_exposures(self):
        """Repeated optimizer passes count sequences and tokens on each pass."""
        actor = _actor(2, epochs=2)
        with _cpu_optimizer_context():
            metrics = actor.update(_experience())
        self.assertEqual(metrics.valid_tokens, 12)
        self.assertEqual(metrics.valid_sequences, 6)
        self.assertEqual(metrics.optimizer_steps, 2)

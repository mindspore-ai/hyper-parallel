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
"""CPU tests for teacher alignment, routing and independent PG objectives."""

# Tests inspect package-owned state and stage boundaries without public runtime hooks.
# pylint: disable=protected-access

from contextlib import nullcontext
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
from typing import Any, Optional

import torch

from rl.algorithm.distillation import PureDistillationAlgorithm, compose_training_algorithm, compute_opd_loss
from rl.algorithm.loss import build_algorithm
from rl.data.experience import ExperiencePreparer, pad_agent_call_batch_for_dp
from rl.data.contracts import ExperienceBatch, Trajectory
from rl.distillation.config import (
    distillation_fingerprint,
    DistillationConfig,
    TeacherConfig,
    parse_distillation_config,
)
from rl.distillation.teacher import TeacherClient, TeacherManager
from rl.trainer.sync import SyncTrainer
from rl.workers.targets import TargetPreparer
from rl.distillation.runtime import TeacherScorer
from rl.workers.actor import Actor
from rl.utils.monitoring.metrics import summarize_training_diagnostics


def configuration(**loss: Any) -> dict[str, Any]:
    """Create a small valid teacher configuration without opening a connection."""
    return {
        "enabled": True,
        "teachers": {"general": {"model_path": "/teacher", "endpoint": "http://localhost:9000", "max_model_len": 32}},
        "loss": loss,
    }


def batch(teacher: Optional[torch.Tensor] = None) -> ExperienceBatch:
    """Create a mixed-length batch with a masked observation and padded tail."""
    return ExperienceBatch(
        trajectories=(),
        sequences=torch.tensor([[1, 2, 3, 4], [1, 3, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        action_mask=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 0]]),
        rewards=torch.tensor([1.0, 0.0]),
        old_log_probs=torch.full((2, 3), -2.0),
        rollout_log_probs=torch.full((2, 3), -2.0),
        advantages=torch.zeros(2, 3),
        responses=("", ""),
        generation_seconds=0.0,
        teacher_log_probs=teacher,
        teacher_keys=None if teacher is None else ("general", "general"),
    )


class TinyPolicy(torch.nn.Module):
    """A real softmax policy for checking optimizer and microbatch behavior."""

    def __init__(self) -> None:
        """Initialize a deterministic learnable probability vector."""
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.1, 0.3, -0.2, 0.4, -0.5]))

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Expose the same logits at each input state."""
        del kwargs
        return {"logits": self.logits.expand(*input_ids.shape, -1)}


class TestDistillation(unittest.TestCase):
    """Verify teacher-dependent gradients and optional execution contracts."""

    def test_disabled_needs_no_teacher(self) -> None:
        """Default and explicitly disabled configurations allocate no teacher."""
        self.assertFalse(parse_distillation_config(None).enabled)
        self.assertFalse(parse_distillation_config({"enabled": False}).enabled)
        algorithm = compose_training_algorithm(
            build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), DistillationConfig()
        )
        self.assertEqual(algorithm.name, "grpo")

    def test_invalid_configuration(self) -> None:
        """Reject ambiguous objectives, routes and numerical settings."""
        for loss in (
            {"mode": "forward_kl_topk"},
            {"use_policy_gradient": False},
            {"clip_ratio": 1.0},
            {"coefficient": float("nan")},
            {"dual_clip": 1.0},
        ):
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                parse_distillation_config(configuration(**loss))
        with self.assertRaises(ValueError):
            parse_distillation_config({"enabled": True})

    def test_teacher_changes_gradient_and_has_no_gradient(self) -> None:
        """Detached rewards still direct the student update in the teacher's direction."""
        config = DistillationConfig(enabled=True)
        mask = torch.tensor([[True]])
        gradients = []
        for teacher_value in (-1.0, -3.0):
            current = torch.tensor([[-2.0]], requires_grad=True)
            teacher = torch.tensor([[teacher_value]], requires_grad=True)
            loss = compute_opd_loss(current, current.detach(), teacher, mask, config)
            loss.backward()
            gradients.append(current.grad.item())
            self.assertIsNone(teacher.grad)
        self.assertLess(gradients[0], 0)
        self.assertGreater(gradients[1], 0)

    def test_equal_teacher_zero_and_masked_nan_ignored(self) -> None:
        """Equal distributions give zero signal and observations cannot affect it."""
        current = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
        loss = compute_opd_loss(
            current, current.detach(), current.detach(), torch.tensor([[True, False]]), DistillationConfig(enabled=True)
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))

    def test_pure_opd_skips_group_and_critic(self) -> None:
        """One response needs neither GRPO normalization nor PPO value targets."""
        config = {
            "algorithm": {"name": "grpo", "loss_aggregation": "token-mean"},
            "distillation": configuration(use_task_rewards=False),
        }
        algorithm = compose_training_algorithm(
            build_algorithm(config["algorithm"]), parse_distillation_config(config["distillation"])
        )
        self.assertEqual(algorithm.name, "opd")
        self.assertFalse(algorithm.requirements.roles.critic)
        self.assertFalse(algorithm.requirements.data.grouped_responses)
        prepared = ExperiencePreparer(algorithm).prepare(batch())
        self.assertTrue(torch.equal(prepared.advantages, torch.zeros(2, 3)))

    def test_teacher_contract_rejects_bad_alignment(self) -> None:
        """Teacher tensors cannot drift from the next-token contract."""
        for teacher in (torch.zeros(2, 4), torch.full((2, 3), float("nan")), torch.zeros(2, 3, requires_grad=True)):
            with self.subTest(shape=teacher.shape), self.assertRaises(ValueError):
                batch(teacher)

    def test_client_echo_alignment(self) -> None:
        """Drop the first no-context probability and ignore the generated extra token."""
        client = TeacherClient(TeacherConfig("/teacher", "http://localhost", "teacher", "/teacher", 8), 1)
        result = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["token_id:1", "token_id:2", "token_id:3", "token_id:4"],
                        "token_logprobs": [None, -1.0, -2.0, -3.0],
                    }
                }
            ]
        }
        with patch.object(client, "_request", return_value=result) as request:
            self.assertEqual(client.score([1, 2, 3]), [-1.0, -2.0])
            self.assertEqual(request.call_args.args[1]["prompt"], [1, 2, 3])
        result["choices"][0]["logprobs"]["tokens"][1] = "token_id:9"
        with patch.object(client, "_request", return_value=result), self.assertRaises(ValueError):
            client.score([1, 2, 3])

    def test_client_rejects_missing_probability_and_overlong_context(self) -> None:
        """Do not silently truncate context or accept incomplete teacher scores."""
        client = TeacherClient(TeacherConfig("/teacher", "http://localhost", "teacher", "/teacher", 3), 1)
        with self.assertRaises(ValueError):
            client.score([1, 2, 3])
        result = {"choices": [{"logprobs": {"tokens": ["token_id:1", "token_id:2"], "token_logprobs": [None, None]}}]}
        with patch.object(client, "_request", return_value=result), self.assertRaises(ValueError):
            client.score([1, 2])

    def test_multi_teacher_order_and_padding(self) -> None:
        """Each real row uses one selected teacher; padding never sends a request."""
        teacher = TeacherConfig("/teacher", "http://localhost", "teacher", "/teacher", 32)
        clients = {}

        def factory(config: TeacherConfig, timeout: float) -> Mock:
            """Return distinguishable clients without network access."""
            del config, timeout
            client = Mock()
            client.score.return_value = [-1.0, -2.0]
            clients[len(clients)] = client
            return client

        manager = TeacherManager(
            DistillationConfig(enabled=True, teachers=(("search", teacher), ("code", teacher))),
            factory,
        )
        manager.clients["code"].score.return_value = [-3.0, -4.0]
        trajectories = tuple(
            SimpleNamespace(prompt_id=prompt, token_ids=torch.tensor([1, 2, 3]), metadata={"dp_padding": padding})
            for prompt, padding in [("a", False), ("b", False), ("pad", True), ("a", False)]
        )
        experience = SimpleNamespace(sequences=torch.zeros(4, 4), trajectories=trajectories)
        prompts = [
            SimpleNamespace(prompt_id="a", metadata={"teacher_key": "code"}),
            SimpleNamespace(prompt_id="b", metadata={"teacher_key": "search"}),
        ]
        scores, keys = manager.score_batch(experience, prompts)
        self.assertEqual(keys, ("code", "search", "", "code"))
        self.assertEqual(scores, [[-3.0, -4.0, 0.0], [-1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [-3.0, -4.0, 0.0]])
        self.assertEqual(manager.clients["code"].score.call_count, 2)
        with self.assertRaises(ValueError):
            manager.resolve({})
        with self.assertRaises(ValueError):
            manager.resolve({"teacher_key": "unknown"})

    def test_single_teacher_ignores_route(self) -> None:
        """A singleton uses the same path as a multi-teacher route to that teacher."""
        config = parse_distillation_config(configuration())
        manager = TeacherManager(config, lambda *args: Mock())
        self.assertEqual(manager.resolve({"teacher_key": "other"}), "general")

    def test_actor_opd_microbatch_equivalence(self) -> None:
        """Accumulating uneven action counts reproduces an unsplit optimizer update."""
        models = []
        for micro in (1, 2):
            model = TinyPolicy()
            actor = Actor(
                model,
                PureDistillationAlgorithm(),
                micro,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False),
                device=torch.device("cpu"),
                response_mini_batch_size=2,
                distillation=DistillationConfig(enabled=True),
                max_grad_norm=100.0,
            )
            experience = batch(torch.full((2, 3), -1.0))
            experience = replace(experience, old_log_probs=actor.compute_log_probs(experience))
            with nullcontext():
                update = actor.update(experience)
            self.assertGreater(update.gradient_norm, 0)
            self.assertNotEqual(update.total_loss, 0)
            self.assertEqual(actor.distillation_metrics()["distillation/supervised_tokens"], 3)
            models.append(torch.detach(model.logits))
        self.assertTrue(torch.allclose(models[0], models[1]))


    def test_opd_diagnostics_use_hccl_supported_dtype(self) -> None:
        """Reduce actual update statistics through an HCCL-compatible collective."""
        model = TinyPolicy()
        actor = Actor(
            model, PureDistillationAlgorithm(), 1,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False),
            device=torch.device("cpu"), response_mini_batch_size=2,
            distillation=DistillationConfig(enabled=True),
        )
        experience = batch(torch.full((2, 3), -1.0))
        experience = replace(experience, old_log_probs=actor.compute_log_probs(experience))
        with nullcontext():
            actor.update(experience)
        local = actor.distillation_metrics()
        actor._dp_size = 2
        actor._dp_group_info = SimpleNamespace(group=object())

        def hccl_sum(tensor: torch.Tensor, **kwargs: Any) -> None:
            """Simulate HCCL while checking every metric uses a supported dtype."""
            self.assertEqual(tensor.dtype, torch.float32, f"HCCL metric dtype: {tensor.dtype}")
            if kwargs.get("op", torch.distributed.ReduceOp.SUM) == torch.distributed.ReduceOp.SUM:
                tensor.mul_(2)

        with patch("rl.distillation.monitoring.dist.all_reduce", side_effect=hccl_sum) as collective:
            reduced = actor.distillation_metrics()
        self.assertEqual(collective.call_count, 3)
        self.assertEqual(reduced["distillation/supervised_tokens"], 6.0, f"Reduced metrics: {reduced}")
        self.assertAlmostEqual(reduced["distillation/loss"], local["distillation/loss"])

    def test_actor_disabled_never_requires_teacher(self) -> None:
        """Existing RL execution works without teacher fields or clients."""
        model = TinyPolicy()
        actor = Actor(
            model,
            build_algorithm({"name": "grpo", "loss_aggregation": "token-mean", "kl_coef": 0.0}),
            1,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False),
            device=torch.device("cpu"),
            response_mini_batch_size=2,
        )
        with nullcontext():
            actor.update(batch())
        self.assertEqual(actor.distillation_metrics(), {})

    def test_teacher_failure_rejects_batch(self) -> None:
        """No partially scored batch is returned when a service fails."""
        config = parse_distillation_config(configuration())
        client = Mock()
        client.score.side_effect = TimeoutError("teacher unavailable")
        manager = TeacherManager(config, lambda *args: client)
        experience = SimpleNamespace(
            sequences=torch.zeros(1, 3),
            trajectories=(SimpleNamespace(prompt_id="a", metadata={}, token_ids=torch.tensor([1, 2, 3])),),
        )
        with self.assertRaises(TimeoutError):
            manager.score_batch(experience, [SimpleNamespace(prompt_id="a", metadata={})])

    def test_rl_objectives_combine_without_replacing_advantages(self) -> None:
        """GRPO/PPO/GSPO keep task targets and receive a separate teacher gradient."""
        for name, aggregation in [("grpo", "token-mean"), ("ppo", "token-mean"), ("gspo", "seq-mean-token-mean")]:
            with self.subTest(algorithm=name):
                gradients = []
                for enabled in (False, True):
                    model = TinyPolicy()
                    algorithm = build_algorithm({"name": name, "loss_aggregation": aggregation, "kl_coef": 0.0})
                    actor = Actor(
                        model,
                        algorithm,
                        2,
                        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False),
                        device=torch.device("cpu"),
                        response_mini_batch_size=2,
                        distillation=DistillationConfig(enabled=True) if enabled else None,
                    )
                    experience = replace(
                        batch(torch.full((2, 3), -1.0)) if enabled else batch(), advantages=torch.full((2, 3), 0.5)
                    )
                    with nullcontext():
                        experience = replace(experience, old_log_probs=actor.compute_log_probs(experience))
                        actor.forward_backward(experience, 0, 2, global_tokens=3, global_sequences=2)
                    gradients.append(model.logits.grad.clone())
                    self.assertTrue(torch.equal(experience.advantages, torch.full((2, 3), 0.5)))
                self.assertFalse(torch.allclose(gradients[0], gradients[1]))

    def test_tp_nonowner_does_not_request_teacher(self) -> None:
        """A TP sibling consumes the owner result without duplicating HTTP requests."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.reward_scorer = None
        trainer.residency = SimpleNamespace(move=lambda **kw: None, release=lambda: None, offload_manager=None, metrics={})
        trainer.parallel_dims = SimpleNamespace(
            tp_size=2, tp_rank=1, device_mesh={"tp": SimpleNamespace(get_group=lambda: "tp")}
        )
        trainer.teacher_manager = Mock()
        trainer._run_rank_synchronized = lambda operation, callback: callback()
        result = ([[-1.0, -2.0, -3.0], [-1.0, 0.0, 0.0]], ("general", "general"))

        def broadcast(values: list[Any], **kwargs: Any) -> None:
            """Supply the owner result to a simulated TP sibling."""
            del kwargs
            values[0] = result

        with (
            patch("rl.distillation.runtime.run_synchronized", side_effect=lambda _, fn: fn()),
            patch("rl.trainer.sync.dist.get_global_rank", return_value=0),
            patch("rl.trainer.sync.dist.broadcast_object_list", side_effect=broadcast),
        ):
            scored = TeacherScorer(teacher_manager=trainer.teacher_manager, parallel_dims=trainer.parallel_dims).score([], batch())
        trainer.teacher_manager.score_batch.assert_not_called()
        self.assertEqual(scored.teacher_keys, result[1])

    def test_teacher_failure_prevents_optimizer_and_publication(self) -> None:
        """The train-step pipeline rejects teacher failure before any student update."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.reward_scorer = None
        trainer.residency = SimpleNamespace(move=lambda **kw: None, release=lambda: None, offload_manager=None, metrics={})
        trainer.state = SimpleNamespace(global_step=0)
        trainer.device = torch.device("cpu")
        trainer.rollout_manager = Mock()
        trainer.rollout_manager.generate.return_value = batch()
        trainer.rollout_engine = Mock()
        trainer.teacher_manager = Mock()
        trainer._dp_group_info = None
        trainer.actor = Mock()
        trainer._publish_policy = Mock()
        trainer._run_rank_synchronized = lambda operation, callback: callback()
        trainer.teacher_scorer = SimpleNamespace(score=Mock(side_effect=TimeoutError("teacher unavailable")))
        inputs = {
            "sample_indices": [0],
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        with (
            patch("rl.trainer.sync.dist.get_rank", return_value=0),
            patch("rl.trainer.sync.build_prompt_records", return_value=[]),
            patch("rl.trainer.sync.pad_agent_call_batch_for_dp", side_effect=lambda rollout, group: rollout),
            self.assertRaises(TimeoutError),
        ):
            trainer._train_step(inputs)
        trainer.actor.update.assert_not_called()
        trainer._publish_policy.assert_not_called()

    def test_multi_routes_to_equal_teachers_match_single_teacher(self) -> None:
        """Teacher count alone must not change scores or student gradients."""
        teacher = TeacherConfig("/teacher", "http://localhost", "teacher", "/teacher", 32)
        client = Mock()
        client.score.return_value = [-1.0, -2.0, -3.0]
        prompts = [
            SimpleNamespace(prompt_id="a", metadata={"teacher_key": "math"}),
            SimpleNamespace(prompt_id="b", metadata={"teacher_key": "code"}),
        ]
        trajectories = tuple(
            SimpleNamespace(prompt_id=record.prompt_id, metadata={}, token_ids=torch.tensor([1, 2, 3, 4]))
            for record in prompts
        )
        source = SimpleNamespace(sequences=torch.zeros(2, 4), trajectories=trajectories)
        gradients = []
        for teachers in ((("general", teacher),), (("math", teacher), ("code", teacher))):
            config = DistillationConfig(enabled=True, teachers=teachers)
            manager = TeacherManager(config, lambda *args: client)
            scores, _ = manager.score_batch(source, prompts)
            current = torch.full((2, 3), -2.0, requires_grad=True)
            compute_opd_loss(
                current, current.detach(), torch.tensor(scores), torch.ones_like(current, dtype=torch.bool), config
            ).backward()
            gradients.append(current.grad)
        self.assertTrue(torch.equal(gradients[0], gradients[1]))

    def test_resume_identity_tracks_teacher_and_objective(self) -> None:
        """Changed teacher artifacts or objectives cannot silently resume the same run."""
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory)
            (teacher / "config.json").write_text('{"model_type":"qwen3"}')
            (teacher / "tokenizer.json").write_text('{"version":"1"}')
            value = configuration()
            value["teachers"]["general"]["model_path"] = directory
            baseline = distillation_fingerprint(value)
            value["teachers"]["general"]["endpoint"] = "http://localhost:9001"
            self.assertEqual(baseline, distillation_fingerprint(value))
            value["loss"]["coefficient"] = 2.0
            self.assertNotEqual(baseline, distillation_fingerprint(value))
            value["loss"]["coefficient"] = 1.0
            (teacher / "config.json").write_text('{"model_type":"qwen3_moe"}')
            self.assertNotEqual(baseline, distillation_fingerprint(value))
        self.assertIsNone(distillation_fingerprint({"enabled": False}))


class TestProbabilityContract(unittest.TestCase):
    """Verify original sampling probabilities survive anchoring and drive gradients."""

    def test_diagnostics_detect_real_rollout_difference(self) -> None:
        """A changed training anchor must not turn the diagnostic into self-comparison."""
        original = batch()
        actor_probs = original.old_log_probs + 0.125
        anchored = replace(original, old_log_probs=actor_probs)
        with patch("rl.utils.monitoring.metrics.dist.get_rank", return_value=0), patch(
            "rl.utils.monitoring.metrics.dist.get_world_size", return_value=1
        ):
            metrics = summarize_training_diagnostics(anchored, actor_probs)
        self.assertGreater(metrics["training/rollout_log_probs_diff_max"], 0.12)
        self.assertTrue(torch.equal(anchored.rollout_log_probs, original.rollout_log_probs))

    def test_rollout_contract_rejects_invalid_action_scores(self) -> None:
        """Reject missing shape, attached gradients and invalid action probabilities."""
        for value in (
            torch.zeros(2, 4), torch.full((2, 3), float("nan")),
            torch.full((2, 3), 0.1), torch.zeros(2, 3, requires_grad=True),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(batch(), rollout_log_probs=value)
        value = batch().rollout_log_probs.clone()
        value[0, 1] = float("nan")
        replace(batch(), rollout_log_probs=value)

    def test_opd_requires_original_rollout_before_optimizer(self) -> None:
        """Never silently fall back to recomputed anchors in enabled OPD."""
        model = TinyPolicy()
        optimizer = Mock()
        actor = Actor(model, PureDistillationAlgorithm(), 1, optimizer=optimizer,
                      device=torch.device("cpu"), response_mini_batch_size=2,
                      distillation=DistillationConfig(enabled=True))
        with self.assertRaisesRegex(ValueError, "original rollout_log_probs"):
            actor.update(replace(batch(torch.full((2, 3), -1.0)), rollout_log_probs=None))
        optimizer.step.assert_not_called()

    def test_task_and_opd_use_sampling_denominator(self) -> None:
        """Both losses use the same behavior probabilities when the train anchor differs."""
        model = TinyPolicy()
        algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean", "kl_coef": 0.0})
        actor = Actor(model, algorithm, 2,
                      optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False),
                      device=torch.device("cpu"), response_mini_batch_size=2,
                      distillation=DistillationConfig(enabled=True))
        experience = replace(batch(torch.full((2, 3), -1.0)), old_log_probs=torch.full((2, 3), -1.5))
        with patch.object(algorithm, "compute_actor_loss", wraps=algorithm.compute_actor_loss) as task, patch(
            "rl.workers.actor.compute_opd_loss", wraps=compute_opd_loss
        ) as opd, nullcontext():
            actor.forward_backward(experience, 0, 2, global_tokens=3)
        torch.testing.assert_close(task.call_args.kwargs["old_log_probs"], experience.rollout_log_probs)
        torch.testing.assert_close(opd.call_args.args[1], experience.rollout_log_probs)

    def test_enumerated_gradient_matches_full_vocabulary_reverse_kl(self) -> None:
        """Exact action enumeration checks gradient, including a distinct behavior policy."""
        teacher_probs = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
        for behavior in ([0.6, 0.3, 0.1], [0.57, 0.32, 0.11]):
            with self.subTest(behavior=behavior):
                logits = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log().requires_grad_()
                current = logits.log_softmax(-1)
                exact = (current.exp() * (current - teacher_probs.log())).sum()
                expected, = torch.autograd.grad(exact, logits, retain_graph=True)
                old = torch.tensor(behavior, dtype=torch.float64)
                sampled = sum(
                    old[index] * compute_opd_loss(
                        current[index].reshape(1, 1), old[index].log().reshape(1, 1),
                        teacher_probs[index].log().reshape(1, 1), torch.ones(1, 1, dtype=torch.bool),
                        DistillationConfig(enabled=True),
                    ) for index in range(3)
                )
                actual, = torch.autograd.grad(sampled, logits)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


class TestTrainerProbabilityContract(unittest.TestCase):
    """Exercise the actual pre-update path that previously erased diagnostics."""

    def test_prepare_retains_sampling_and_recomputed_probabilities(self) -> None:
        """Preparing teacher targets must preserve both probability identities."""
        trainer = SyncTrainer.__new__(SyncTrainer)
        trainer.reward_scorer = None
        trainer.residency = SimpleNamespace(move=lambda **kw: None, release=lambda: None, offload_manager=None, metrics={})
        trainer.algorithm = PureDistillationAlgorithm()
        trainer.teacher_manager = Mock()
        trainer._consistency_profile = "off"
        trainer.actor = Mock()
        source = batch()
        expected = source.old_log_probs + 0.125
        trainer.actor.compute_log_probs.return_value = expected
        trainer._run_rank_synchronized = lambda operation, callback: callback()
        trainer.experience_preparer = ExperiencePreparer(trainer.algorithm)
        with patch("rl.utils.monitoring.metrics.dist.get_rank", return_value=0), patch(
            "rl.utils.monitoring.metrics.dist.get_world_size", return_value=1
        ):
            prepared, diagnostics = TargetPreparer(
                actor=trainer.actor, reference_actor=None, critic=None, algorithm=trainer.algorithm,
                experience_preparer=trainer.experience_preparer, run_synchronized=trainer._run_rank_synchronized,
                has_teacher=True).prepare(source, True, {})
        torch.testing.assert_close(prepared.rollout_log_probs, source.rollout_log_probs)
        torch.testing.assert_close(prepared.old_log_probs, expected)
        self.assertGreater(diagnostics["training/rollout_log_probs_diff_max"], 0.12)


class TestOPDClipping(unittest.TestCase):
    """Check intentional clipped-surrogate behavior separately from the KL oracle."""

    def test_clipped_branches_stop_gradient(self) -> None:
        """PPO upper/lower clipping and dual clipping must stop the ratio gradient."""
        for student, old, teacher in [(-1.0, -1.5, -0.5), (-2.0, -1.0, -3.0), (-0.1, -2.0, -1.0)]:
            with self.subTest(student=student, old=old, teacher=teacher):
                current = torch.tensor([[student]], requires_grad=True)
                loss = compute_opd_loss(current, torch.tensor([[old]]), torch.tensor([[teacher]]),
                                        torch.ones(1, 1, dtype=torch.bool), DistillationConfig(enabled=True))
                loss.backward()
                self.assertEqual(float(current.grad), 0.0)


class TestProbabilityPadding(unittest.TestCase):
    """Cover the two probability fields through uneven DP call alignment."""

    def test_padding_retains_real_probabilities_and_zeroes_dummy_rows(self) -> None:
        """A peer with more calls must not alter this rank's behavior probabilities."""
        source = batch()
        rows = tuple(Trajectory(
            trajectory_id=str(index), prompt_id=str(index), group_id=None, policy_version=0, turns=(),
            token_ids=source.sequences[index], attention_mask=source.attention_mask[index],
            action_mask=source.action_mask[index], rollout_log_probs=source.rollout_log_probs[index],
            reward=float(source.rewards[index]), reward_components={}, done=True, truncated=False,
            terminal_reason="completed",
        ) for index in range(2))
        source = replace(source, trajectories=rows, advantages=None)
        original = source.rollout_log_probs.clone()

        def peer_maximum(summary: torch.Tensor, **kwargs: Any) -> None:
            """Simulate a segmented peer with three calls."""
            del kwargs
            summary.copy_(summary.new_tensor([1, 3]))

        with patch("rl.data.experience.dist.all_reduce", side_effect=peer_maximum):
            padded = pad_agent_call_batch_for_dp(source, SimpleNamespace(rank_size=2, group=object()))
        self.assertEqual(tuple(padded.rollout_log_probs.shape), (3, 3))
        torch.testing.assert_close(padded.rollout_log_probs[:2], original)
        torch.testing.assert_close(padded.old_log_probs[:2], source.old_log_probs)
        self.assertFalse(bool(padded.loss_action_mask[2].any()))
        torch.testing.assert_close(padded.rollout_log_probs[2], torch.zeros(3))
        torch.testing.assert_close(source.rollout_log_probs, original)

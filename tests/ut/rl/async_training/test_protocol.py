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
"""Bounded group ownership, supported recipes and durable checkpoint contracts."""
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

import rl

from rl.algorithm import GRPOAlgorithm, GRPOConfig
from rl.algorithm.loss import build_algorithm
from rl.async_training.fully_async_rollouter import ExperienceBuffer
from rl.async_training.protocol import (
    AsyncCheckpoint, AsyncTrainingConfig, SampleGroup, SamplingTask, validate_async_config,
)
from rl.cli import load_config
from rl.config import validate_config
from rl.dataset.batch_builder import build_experience_batch, ExperiencePreparer
from rl.dataset.episodes import episode_rows
from rl.roles.rollout.base import GenerationSettings
from rl.utils.async_audit import ActorLossAudit, EventTrace
from tests.ut.rl.async_training.support import group


class TestExperienceBuffer(unittest.TestCase):
    """Use actual CPU trajectory payloads rather than mocking data validation."""

    def setUp(self) -> None:
        """Initialize the supported test configuration."""
        self.settings = AsyncTrainingConfig(max_inflight_groups=2, max_buffer_groups=2,
                                            max_group_bytes=65536, max_buffer_bytes=131072)
        self.buffer = ExperienceBuffer(self.settings, 2)

    def submit(self, task: Any) -> None:
        """Encode and deliver one reserved test group."""
        self.buffer.accept(group(task).encode(2, 65536))

    def test_capacity_includes_selected_batch(self):
        """Test capacity includes selected batch."""
        first = self.buffer.reserve(1, 0)
        second = self.buffer.reserve(1, 0)
        self.assertIsNone(self.buffer.reserve(1, 0))
        self.submit(first)
        self.submit(second)
        self.buffer.select(2)
        self.assertIsNone(self.buffer.reserve(1, 0))
        with self.assertRaises(RuntimeError):
            self.buffer.state_dict()
        self.buffer.commit()
        self.assertIsNotNone(self.buffer.reserve(2, 1))

    def test_duplicate_and_unreserved_results_fail(self):
        task = self.buffer.reserve(1, 0)
        self.submit(task)
        with self.assertRaises(ValueError):
            self.submit(task)
        with self.assertRaises(ValueError):
            self.submit(SamplingTask(100, 1, 0))

    def test_version_mismatch_does_not_release_reservation(self):
        task = self.buffer.reserve(1, 0)
        with self.assertRaises(ValueError):
            self.submit(SamplingTask(task.identity, 2, 1))
        self.assertIn(task.identity, self.buffer.reserved)

    def test_staleness_is_whole_group_and_strictly_over_threshold(self):
        task = self.buffer.reserve(1, 0)
        self.submit(task)
        self.assertEqual(self.buffer.evict_stale(4), 0)
        self.assertEqual(self.buffer.evict_stale(5), 1)
        self.assertEqual(self.buffer.dropped_groups, 1)

    def test_snapshot_roundtrip_preserves_accounting(self):
        self.submit(self.buffer.reserve(1, 0))
        state = self.buffer.state_dict()
        restored = ExperienceBuffer(self.settings, 2)
        restored.load_state_dict(state)
        restored.select(1)
        restored.commit()
        self.assertEqual(restored.consumed_groups, 1)
        self.assertEqual(restored.reserve(2, 1).identity, 1)

    def test_corrupt_snapshot_fails(self):
        state = {"next_identity": 4, "ready": [], "consumed_groups": 0, "dropped_groups": 0}
        with self.assertRaises(ValueError):
            self.buffer.load_state_dict(state)

    def test_mixed_versions_keep_sync_default_strict(self):
        """Test mixed versions keep sync default strict."""
        rows = group(SamplingTask(0, 1, 0)).trajectories + group(SamplingTask(1, 2, 1)).trajectories
        settings = GenerationSettings(2, 1., 1., 0, True, 0, 3, collect_log_probs=True)
        with self.assertRaises(ValueError):
            build_experience_batch(rows, 0.1, settings, {})
        batch = build_experience_batch(rows, 0.1, settings, {}, mixed_policy_versions=True)
        self.assertIsNone(batch.worker_policy_version)
        algorithm = build_algorithm({"name": "grpo", "kl_coef": 0, "loss_aggregation": "token-mean"})
        prepared = ExperiencePreparer(algorithm).prepare(batch)
        torch.testing.assert_close(prepared.advantages[:, 0],
                                   torch.tensor([-2**-0.5, 2**-0.5, -2**-0.5, 2**-0.5]))
        torch.testing.assert_close(batch.old_log_probs, batch.rollout_log_probs)

    def test_wire_budget_rejects_oversize(self):
        with self.assertRaises(ValueError):
            group(SamplingTask(0, 1, 0)).encode(2, 16)


def call_group(task: SamplingTask) -> SampleGroup:
    """One candidate has one call, another has three calls crossing publications."""
    rows = []
    for candidate, template in enumerate(group(task).trajectories):
        count = 1 if candidate == 0 else 3
        for index in range(count):
            version = task.policy_version + index
            rows.append(replace(
                template, trajectory_id=f"{task.identity}:{candidate}:call:{index}",
                policy_version=version, worker_policy_version=version,
                metadata={"episode_id": f"{task.identity}:{candidate}", "call_index": index,
                          "call_count": count, "async_training": True,
                          "behavior_segments": ({"start": 0, "end": 2, "policy_version": version,
                                                 "source_update": version - 1},)},
            ))
    return SampleGroup(task, tuple(rows), .1)


class TestMultiturnBatch(unittest.TestCase):
    """Use real wire validation, target construction and buffer ownership."""

    def test_complete_group_counts_episodes_and_preserves_behavior_probabilities(self):
        """Test complete group counts episodes and preserves behavior probabilities."""
        sample = call_group(SamplingTask(0, 1, 0))
        sample.validate(2)
        self.assertEqual(episode_rows(sample.trajectories), [[0], [1, 2, 3]])
        settings = GenerationSettings(8, 1., 1., 0, True, 0, 3, collect_log_probs=True)
        batch = build_experience_batch(sample.trajectories, .1, settings, {}, mixed_policy_versions=True)
        algorithm = build_algorithm({"name": "grpo", "kl_coef": 0, "loss_aggregation": "token-mean"})
        experience = ExperiencePreparer(algorithm).prepare(batch)
        expected = torch.tensor([-2**-.5, 2**-.5, 2**-.5, 2**-.5])
        torch.testing.assert_close(experience.advantages[:, 0], expected)
        torch.testing.assert_close(experience.old_log_probs, batch.rollout_log_probs)

    def test_incomplete_calls_and_duplicate_calls_are_rejected(self):
        sample = call_group(SamplingTask(0, 1, 0))
        for rows in (sample.trajectories[:-1], sample.trajectories + sample.trajectories[-1:]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                replace(sample, trajectories=rows).validate(2)

    def test_sync_episode_still_requires_one_version(self):
        sample = call_group(SamplingTask(0, 1, 0))
        rows = tuple(replace(row, metadata={**row.metadata, "async_training": False})
                     for row in sample.trajectories)
        with self.assertRaisesRegex(ValueError, "policy_version"):
            episode_rows(rows)

    def test_call_versions_must_not_go_backwards(self):
        sample = call_group(SamplingTask(0, 1, 0))
        last = replace(sample.trajectories[-1], policy_version=1, worker_policy_version=1)
        with self.assertRaisesRegex(ValueError, "versions"):
            replace(sample, trajectories=sample.trajectories[:-1] + (last,)).validate(2)

    def test_infrastructure_discard_preserves_checkpoint_conservation(self):
        """Test infrastructure discard preserves checkpoint conservation."""
        settings = AsyncTrainingConfig()
        buffer = ExperienceBuffer(settings, 2)
        task = buffer.reserve(1, 0)
        buffer.discard(task)
        with self.assertRaises(ValueError):
            buffer.discard(task)
        restored = ExperienceBuffer(settings, 2)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(restored.dropped_groups, 1)
        self.assertEqual(restored.reserve(2, 1).identity, 1)


class TestAsyncCheckpoint(unittest.TestCase):
    """A model checkpoint alone must never imply a complete async snapshot."""

    def setUp(self) -> None:
        """Initialize the supported test configuration."""
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.root = Path(self.directory)
        (self.root / "checkpoint_complete.json").write_text("{}")
        self.config = {"train": {"max_steps": 4, "checkpoint": {}}, "data": {"train_path": "a"}}
        self.state = {"learner_update": 2, "source_update": 2, "policy_version": 3,
                      "source": {"cursor": 7}, "buffer": {}}

    def test_commit_requires_model_completion(self):
        (self.root / "checkpoint_complete.json").unlink()
        with self.assertRaises(RuntimeError):
            AsyncCheckpoint.commit(self.root, self.state, self.config)

    def test_auto_resume_selects_latest_complete_and_honors_explicit_path(self):
        """Test auto resume selects latest complete and honors explicit path."""
        self.config["train"]["checkpoint"]["output_dir"] = self.directory
        self.config["train"]["max_steps"] = 20
        self.assertIsNone(AsyncCheckpoint.resolve_path(self.config))
        for step in (2, 10, 11):
            root = self.root / f"step_{step}"
            root.mkdir()
            (root / "checkpoint_complete.json").write_text("{}")
            state = dict(self.state, learner_update=step, source_update=step, policy_version=step + 1)
            AsyncCheckpoint.commit(root, state, self.config)
        AsyncCheckpoint.invalidate(root)
        selected = AsyncCheckpoint.resolve_path(self.config)
        self.assertEqual(Path(selected).name, "step_10")
        self.assertEqual(AsyncCheckpoint.load(selected, self.config)["learner_update"], 10)
        self.config["train"]["checkpoint"]["load_path"] = str(self.root / "step_2")
        self.assertEqual(Path(AsyncCheckpoint.resolve_path(self.config)).name, "step_2")
        (Path(selected) / "async_state.pkl").write_bytes(b"corrupt")
        self.config["train"]["checkpoint"].pop("load_path")
        with self.assertRaises(ValueError):
            AsyncCheckpoint.load(AsyncCheckpoint.resolve_path(self.config), self.config)

    def test_roundtrip_and_longer_resume(self):
        AsyncCheckpoint.commit(self.root, self.state, self.config)
        self.config["train"]["max_steps"] = 10
        self.assertEqual(AsyncCheckpoint.load(self.directory, self.config), self.state)
        self.config["data"]["train_path"] = "other"
        with self.assertRaises(ValueError):
            AsyncCheckpoint.load(self.directory, self.config)

    def test_replaced_dataset_is_rejected_at_the_same_path(self):
        """Test replaced dataset is rejected at the same path."""
        data = self.root / "data.parquet"
        data.write_bytes(b"original source")
        self.config["data"]["train_path"] = str(data)
        AsyncCheckpoint.commit(self.root, self.state, self.config)
        data.write_bytes(b"replaced source")
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            AsyncCheckpoint.load(self.directory, self.config)

    def test_corrupted_payload_is_rejected(self):
        AsyncCheckpoint.commit(self.root, self.state, self.config)
        with (self.root / "async_state.pkl").open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaises(ValueError):
            AsyncCheckpoint.load(self.directory, self.config)

    def test_invalidate_removes_only_commit_marker(self):
        AsyncCheckpoint.commit(self.root, self.state, self.config)
        AsyncCheckpoint.invalidate(self.root)
        self.assertFalse((self.root / "async_complete.json").exists())
        self.assertTrue((self.root / "async_state.pkl").exists())


def recipe() -> dict:
    """Return the smallest supported logical recipe."""
    return {
        "model": {"name": "qwen3"}, "algorithm": {"name": "grpo"},
        "rollout": {"engine": "vllm", "num_return_sequences": 2, "temperature": 1.,
                    "top_p": 1., "top_k": 0, "vllm": {"deployment": "disjoint"}},
        "agentic": {"max_turns": 1, "interaction_mode": "single_turn"},
        "train": {"micro_batch_size": 1, "response_mini_batch_size": 2,
                  "accelerator": {"dp_shard": 1}},
    }


class TestAsyncConfig(unittest.TestCase):
    """Reject unsupported combinations rather than silently overriding user settings."""

    def test_task_recipes_pass_common_validation_without_loading_models(self):
        """Test task recipes pass common validation without loading models."""
        root = Path(rl.__file__).resolve().parents[1] / "examples/async_training/configs"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            }), encoding="utf-8")
            (directory / "train.parquet").touch()
            for filename in ("qwen3_4b_search_r1.yaml", "qwen3_4b_swe_smith.yaml"):
                with self.subTest(recipe=filename):
                    config = load_config(str(root / filename), [])
                    config["model"].update(weights_path=str(directory), tokenizer_path=str(directory))
                    config["data"]["train_path"] = str(directory / "train.parquet")
                    validate_async_config(config)
                    del config["async_training"]
                    with patch.dict("os.environ", {"ASCEND_RT_VISIBLE_DEVICES": "0,1", "WORLD_SIZE": "2"}), patch(
                            "rl.config.torch.accelerator.current_accelerator",
                            return_value=SimpleNamespace(type="npu")):
                        validate_config(config, build_algorithm(config["algorithm"]))

    def test_retry_budget_allows_zero_but_rejects_invalid_types(self):
        self.assertEqual(AsyncTrainingConfig.from_mapping({"max_candidate_retries": 0}).max_candidate_retries, 0)
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AsyncTrainingConfig.from_mapping({"max_candidate_retries": value})

    def test_codex_evaluation_requires_separate_gateway(self):
        config = recipe()
        config["agentic"] = {"runner": "codex", "max_turns": 10, "codex": {}}
        config["evaluation"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "evaluation"):
            validate_async_config(config)

    def test_unsupported_algorithms_and_devices(self):
        for section, key, value in (
            ("algorithm", "name", "ppo"), ("model", "name", "qwen3_moe"),
            ("agentic", "max_turns", 2), ("rollout", "temperature", 0.8),
            ("rollout", "top_p", 0.9), ("rollout", "num_return_sequences", 1),
        ):
            config = recipe()
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_async_config(config)

    def test_capacity_configuration_is_strict(self):
        for settings in ({"max_buffer_groups": True}, {"operation_timeout": float("nan")},
                         {"groups_per_dp_rank": 0}, {"publish_every_updates": -1},
                         {"unknown": 1}, {"max_buffer_bytes": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                AsyncTrainingConfig.from_mapping(settings)


def scalar_loss(current: float, old: float, reference: float, advantage: float,
                coefficient: float = 0.07) -> float:
    """Evaluate the clipped objective using Python scalars, independent of Torch losses."""
    ratio = math.exp(current - old)
    loss = max(-advantage * ratio, -advantage * min(1.2, max(0.8, ratio)))
    if advantage < 0:
        loss = min(loss, -3.0 * advantage)
    delta = reference - current
    return loss + coefficient * min(10.0, max(0.0, math.exp(delta) - delta - 1.0))


class TestScalarBackward(unittest.TestCase):
    """Check observed backward against independent scalar finite differences."""

    def test_scalar_finite_differences_match_observed_backward(self):
        """Test scalar finite differences match observed backward."""
        ratios = [0.6, 0.9, 1.1, 1.6, 0.6, 0.9, 1.4, 4.0, 1.05]
        old = torch.full((1, len(ratios)), -5.0, dtype=torch.float64)
        current = (old + torch.tensor([math.log(value) for value in ratios])).requires_grad_()
        reference = (current.detach() - 0.27).float()
        advantages = torch.tensor([[1., 1., 1., 1., -1., -1., -1., -1., 1.]], dtype=torch.float64)
        mask = torch.tensor([[True] * 8 + [False]])
        algorithm = GRPOAlgorithm(GRPOConfig(kl_coef=0.07))
        with tempfile.TemporaryDirectory() as directory:
            trace = EventTrace(directory, "trainer")
            audit = ActorLossAudit(algorithm, trace)
            experience = SimpleNamespace(
                sequences=torch.ones(1, 10), rollout_log_probs=old, old_log_probs=old,
                advantages=advantages, rewards=torch.ones(1), loss_action_mask=mask,
                trajectories=[SimpleNamespace(group_id="g", trajectory_id="g:0", policy_version=2)],
            )
            audit.begin(
                3, experience, {"response_mini_batch_size": 1, "micro_batch_size": 1, "policy_update_epochs": 1}, 1,
            )
            output = audit.compute_actor_loss(
                current_log_probs=current, old_log_probs=old, reference_log_probs=reference,
                advantages=advantages, action_mask=mask,
            )
            (output.total_loss_sum / int(mask.sum())).backward()
            audit.finish()
            observed = torch.load(Path(directory) / "loss_step_3_rank_0_call_0.pt", weights_only=False)
            expected = []
            epsilon = 1.0e-4
            for index in range(len(ratios)):
                value, anchor, target = current[0, index].item(), old[0, index].item(), reference[0, index].item()
                advantage = advantages[0, index].item()
                derivative = (scalar_loss(value + epsilon, anchor, target, advantage)
                              - scalar_loss(value - epsilon, anchor, target, advantage)) / (2 * epsilon)
                expected.append(derivative / int(mask.sum()) if mask[0, index] else 0.)
            # Production KL intentionally uses FP32 even when policy inputs are FP64.
            torch.testing.assert_close(observed["log_prob_gradient"], torch.tensor([expected], dtype=torch.float64),
                                       atol=2.0e-8, rtol=2.0e-5)
            torch.testing.assert_close(observed["ratio"], torch.tensor([ratios], dtype=torch.float64),
                                       atol=2.0e-7, rtol=2.0e-7)
            self.assertEqual(float(observed["log_prob_gradient"][0, -1]), 0.0)

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
"""Unit tests for the VeRL V1 TransferQueue optional fields."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from omegaconf import OmegaConf
import torch

from tests.ut.integration.verl._contracts import ADAPTERS

optional_fields = ADAPTERS.optional_fields


class BatchMetadata:
    """Minimal TransferQueue metadata for one trajectory."""

    keys = ["sample"]
    partition_id = "train"
    tags = [{"min_global_steps": 2, "max_global_steps": 3}]

    def __len__(self) -> int:
        return len(self.keys)


class TestOptionalFields(unittest.TestCase):
    """Verify the requested data and published metrics for supported configurations."""

    @staticmethod
    def _trainer(enabled: bool) -> SimpleNamespace:
        return SimpleNamespace(
            use_critic=enabled,
            use_reference_policy=enabled,
            _get_n_gpus_for_throughput=Mock(return_value=1),
            _rollout_moe_lb_metrics_accumulator=None,
            kl_ctrl_in_reward=None,
            config=OmegaConf.create({
                "algorithm": {
                    "use_kl_in_reward": enabled,
                    "rollout_correction": {} if enabled else None,
                    "adv_estimator": "grpo",
                    "gamma": 1.0, "lam": 1.0, "kl_penalty": "kl",
                },
                "actor_rollout_ref": {"rollout": {"n": 1}, "model": {
                    "mtp": {"enable": enabled, "enable_rollout": enabled},
                }},
            }),
        )

    def test_advantage_uses_only_fields_with_producers(self) -> None:
        """Verify both full PPO reward inputs and critic-free GRPO field selection."""
        batch = BatchMetadata()
        for enabled in (False, True):
            with self.subTest(enabled=enabled), ExitStack() as stack:
                trainer = self._trainer(enabled)
                tensor = torch.ones((1, 2))
                padded = {
                    "uid": torch.tensor([1]), "rm_scores": tensor, "rollout_log_probs": tensor,
                    "response_mask": tensor, "advantages": tensor, "returns": tensor,
                    "rollout_is_weights": tensor,
                }
                queue = stack.enter_context(patch.object(optional_fields, "tq"))
                queue.kv_batch_get.return_value = MagicMock(
                    __getitem__=Mock(return_value=tensor), to_padded_tensor=Mock(return_value=padded),
                )
                proto = SimpleNamespace(batch=padded, non_tensor_batch={})
                stack.enter_context(patch.object(optional_fields, "DataProto", return_value=proto))
                stack.enter_context(patch.object(
                    optional_fields, "compute_advantage_for_multi_trajectories", return_value=proto,
                ))
                stack.enter_context(patch.object(optional_fields, "response_to_nested", side_effect=lambda x, _: x))
                if enabled:
                    padded["token_level_rewards"] = tensor
                    stack.enter_context(patch.object(optional_fields, "apply_kl_penalty", return_value=(proto, {})))
                    stack.enter_context(patch.object(
                        optional_fields, "compute_rollout_correction_and_add_to_batch", return_value=(proto, {}),
                    ))

                returned = optional_fields._compute_advantage(trainer, batch, {})

                self.assertIs(returned, queue.kv_batch_put.return_value)
                requested = queue.kv_batch_get.call_args.kwargs["select_fields"]
                for name in ("ref_log_prob", "values"):
                    self.assertEqual(name in requested, enabled)
                published = queue.kv_batch_put.call_args.kwargs["fields"]
                self.assertEqual("token_level_rewards" in published, enabled)
                self.assertEqual("rollout_is_weights" in published, enabled)

    def test_metrics_use_only_fields_with_producers(self) -> None:
        """Retain tracked PPO fields and fall back to raw scores for GRPO."""
        batch = BatchMetadata()
        for enabled in (False, True):
            with self.subTest(enabled=enabled), ExitStack() as stack:
                trainer = self._trainer(enabled)
                sequence = torch.nested.nested_tensor([torch.ones(2)], layout=torch.jagged)
                padded = {"rm_scores": torch.ones((1, 2))}
                if enabled:
                    padded["token_level_rewards"] = torch.zeros((1, 2))
                raw = MagicMock(to_padded_tensor=Mock(return_value=padded), pop=Mock(return_value=torch.tensor([3])))
                raw.__getitem__ = Mock(return_value=sequence)
                read_data = stack.enter_context(patch.object(
                    optional_fields, "get_metric_data_with_optional_routed_experts", return_value=raw,
                ))
                queue = stack.enter_context(patch.object(optional_fields, "tq"))
                queue.kv_batch_get.return_value = {"extra_fields": Mock(tolist=Mock(return_value=[{
                    "spec_num_draft_tokens": 3, "spec_num_accepted_tokens": 2, "spec_num_verify_steps": 1,
                }]))}
                proto = Mock()
                proto.select_idxs.return_value = SimpleNamespace(non_tensor_batch={})
                stack.enter_context(patch.object(optional_fields, "DataProto", return_value=proto))
                for name in (
                    "compute_moe_lb_metrics", "compute_data_metrics", "compute_timing_metrics",
                    "compute_throughout_metrics", "compute_variance_proxy_metrics",
                ):
                    stack.enter_context(patch.object(optional_fields, name, return_value={}))
                spec_metrics = stack.enter_context(patch.object(
                    optional_fields, "compute_spec_decode_metrics", return_value={},
                ))
                metrics = {}

                optional_fields._compute_metrics(trainer, batch, metrics, {}, global_steps=5, epoch=1)

                requested = read_data.call_args.kwargs["fields"]
                for name in ("values", "token_level_rewards"):
                    self.assertEqual(name in requested, enabled)
                if enabled:
                    self.assertEqual(spec_metrics.call_args.args[:3], ([3], [2], [1]))
                else:
                    self.assertIs(padded["token_level_rewards"], padded["rm_scores"])
                    self.assertEqual(spec_metrics.call_args.args[:3], (None, None, None))
                self.assertEqual(metrics["training/num_turns/mean"], 3)
                self.assertEqual(metrics["training/off_policy/trajectory_staleness/mean"], 1)

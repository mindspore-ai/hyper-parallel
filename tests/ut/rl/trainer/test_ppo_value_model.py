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
"""Value-model loading, parameter ownership and parallel-plan regression tests."""

import unittest
from unittest.mock import patch
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from rl.roles.policy.critic import attach_value_head, build_value_model
from rl.roles.qwen3_builder import Qwen3ShardingPlanner
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh


class TestQwen3ValueModel(unittest.TestCase):
    """Check real tiny checkpoints without requiring accelerator hardware."""

    def setUp(self) -> None:
        """Use a tiny Qwen3 with explicit GQA dimensions."""
        self.config = Qwen3Config.from_dict({
            "vocab_size": 64, "hidden_size": 24, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "max_position_embeddings": 64, "tie_word_embeddings": True,
            "architectures": ["Qwen3ForCausalLM"],
        })

    def test_value_loader_preserves_backbone_and_trains_head(self) -> None:
        """Load causal weights without a vocab head and update scalar values."""
        torch.manual_seed(12)
        source = Qwen3ForCausalLM(self.config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            with patch("hyper_parallel.models._transformers.auto_model._current_device",
                       return_value=torch.device("cpu")):
                model = build_value_model(directory, torch_dtype="float32", force_hf=True,
                                          local_files_only=True)
            self.assertFalse(model.config.tie_word_embeddings)
            self.assertFalse(hasattr(model, "lm_head"))
            self.assertTrue(self.config.tie_word_embeddings)
            for name, parameter in source.model.named_parameters():
                torch.testing.assert_close(dict(model.model.named_parameters())[name], parameter)
            tokens = torch.tensor([[1, 2, 3, 4]])
            values = model(tokens)["values"]
            self.assertEqual(tuple(values.shape), (1, 4))
            torch.testing.assert_close(values, torch.zeros_like(values))
            (values - 1).square().mean().backward()
            self.assertGreater(model.value_head.weight.grad.norm().item(), 0)
            self.assertIsNone(source.model.embed_tokens.weight.grad)
            # CPU math must not trigger torch-npu's automatic foreach device probe.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False, fused=False)
            optimizer.step()
            self.assertGreater(model(tokens)["values"].abs().sum().item(), 0)
            with patch("hyper_parallel.models._transformers.auto_model._current_device",
                       return_value=torch.device("cpu")):
                restored = build_value_model(directory, torch_dtype="float32", force_hf=True,
                                             local_files_only=True)
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored.value_head.weight, model.value_head.weight)

    def test_tp_plan_covers_scalar_head(self) -> None:
        """A scalar head must remain replicated instead of being vocab-sharded."""
        model = attach_value_head(Qwen3ForCausalLM(self.config))
        spec = ModuleShardingSpec(params={"weight": {"tp": Replicate()}}, region_dispatch=False)
        plan = Qwen3ShardingPlanner(plan_overrides={"value_head": spec}).plan(
            model, FakeDeviceMesh((2,), ("tp",)), tp_size=2,
        )
        self.assertEqual(plan.modules["value_head"].params["weight"], {"tp": Replicate()})

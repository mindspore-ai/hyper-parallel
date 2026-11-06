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
"""Authoritative HF RoPE parity after Qwen3 meta materialization."""

# Tests inspect package-owned state and stage boundaries without public runtime hooks.
# pylint: disable=protected-access

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from rl.workers.engine.materialization import _rope_factory, register_qwen3_materialization
from rl.workers.adapters.qwen3 import adapt_materialization

from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state


class TestQwen3Materialization(unittest.TestCase):
    """Derived buffer restoration must preserve loaded parameters and persistent state."""

    def test_frequency_math_uses_cpu_before_target_device_transfer(self) -> None:
        """Materialization device cannot change canonical HF frequency arithmetic."""
        rotary = Qwen3RotaryEmbedding(Qwen3Config.from_dict({"head_dim": 128}))
        with patch.object(rotary, "compute_default_rope_parameters",
                          wraps=rotary.compute_default_rope_parameters) as calculate:
            factory = _rope_factory(rotary)
            result = factory(MaterializationContext("checkpoint_load", torch.device("meta")))
        self.assertEqual(calculate.call_args.kwargs["device"], torch.device("cpu"))
        self.assertEqual(result.device.type, "meta")
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, rotary.inv_freq.shape)

    def test_meta_checkpoint_and_random_init_preserve_native_rope(self) -> None:
        """Default and scaled frequencies agree with native HF after repeated to_empty cycles."""
        for scaling in (None, {"rope_type": "linear", "factor": 2.0}):
            with self.subTest(scaling=scaling):
                config = Qwen3Config.from_dict({
                    "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
                    "max_position_embeddings": 64, "vocab_size": 32, "rope_scaling": scaling,
                })
                native = Qwen3RotaryEmbedding(config)
                with torch.device("meta"):
                    model = nn.Module()
                    model.rotary = Qwen3RotaryEmbedding(config)
                    model.linear = nn.Linear(16, 16, bias=False)
                    model.register_buffer("persistent_state", torch.empty(1))
                register_qwen3_materialization(model)
                for reason in ("checkpoint_load", "random_init"):
                    model.to_empty(device="cpu")
                    with torch.no_grad():
                        model.linear.weight.fill_(42)
                        model.persistent_state.fill_(17)
                        model.rotary.inv_freq.zero_()
                        model.rotary.original_inv_freq.zero_()
                    weight_identity = id(model.linear.weight)
                    buffer_identity = id(model.rotary.inv_freq)
                    rebuild_materialized_state(model, MaterializationContext(reason, torch.device("cpu")))
                    self.assertEqual(id(model.linear.weight), weight_identity)
                    self.assertEqual(id(model.rotary.inv_freq), buffer_identity)
                    self.assertTrue(torch.equal(model.linear.weight, torch.full((16, 16), 42.0)))
                    self.assertEqual(model.persistent_state.item(), 17)
                    self.assertTrue(torch.equal(model.rotary.inv_freq, native.inv_freq))
                    self.assertTrue(torch.equal(model.rotary.original_inv_freq, native.original_inv_freq))
                    self.assertEqual(model.rotary.inv_freq.dtype, torch.float32)
                    self.assertNotIn("rotary.inv_freq", model.state_dict())
                    hidden = torch.zeros((1, 8, 16), dtype=torch.bfloat16)
                    positions = torch.arange(8)[None, :]
                    actual = model.rotary(hidden, positions)
                    expected = native(hidden, positions)
                    for candidate, reference in zip(actual, expected):
                        self.assertTrue(torch.equal(candidate, reference))

    def test_rl_registration_is_idempotent_and_instance_scoped(self) -> None:
        """RL construction leaves public classes and unrelated model buffers untouched."""
        config = Qwen3Config.from_dict({"head_dim": 4, "tie_word_embeddings": False})
        baseline = nn.Module()
        baseline.rotary = Qwen3RotaryEmbedding(config)
        model = nn.Module()
        model.config = config
        model.rotary = Qwen3RotaryEmbedding(config)
        native_to_empty = nn.Module.to_empty
        adapt_materialization(model)
        adapt_materialization(model)
        self.assertIs(nn.Module.to_empty, native_to_empty)
        self.assertNotIn("_hp_rebuildable_buffer_specs", baseline.rotary.__dict__)
        self.assertEqual(set(model.rotary._hp_rebuildable_buffer_specs), {"inv_freq", "original_inv_freq"})
        model.to_empty(device="cpu")
        rebuild_materialized_state(model, MaterializationContext("checkpoint_load", torch.device("cpu")))
        self.assertTrue(torch.equal(model.rotary.inv_freq, baseline.rotary.inv_freq))

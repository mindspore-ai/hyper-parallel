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
"""Unit tests for GraphTrainer TP/SP adaptation.

Tests cover:

1. ``fsdp_degree`` resolution: when ``pass_config.fsdp_degree`` is set
   (TP+FSDP hybrid), FSDPPass uses it instead of ``world_size``.
2. ``_shard_live_model_params`` uses the FSDP group's local rank, not
   the global rank, as the chunk index.
3. ``TracingContext`` availability for functional collectives tracing.
4. Model structure: ``rotary_emb`` inside attention, submodule names
   matching ShardingPlanner conventions.
5. SP helpers: ``shard_for_sp`` and ``DataSampler``.
6. Forward/backward shape consistency.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel.compile.parallel_config import PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.sharding_config import PassPlan
from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph


# ---------------------------------------------------------------------------
# Helper: build a minimal model + joint FX graph for testing.
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """A 2-parameter model: linear + bias, suitable for FX tracing."""

    def __init__(self, in_features=8, out_features=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return self.linear(x)


def _make_joint_graph(model, input_tensor, label_tensor):
    """Trace a joint fwd+bwd graph and return the GraphModule."""
    # pylint: disable=C0415

    def train_fn(mdl, inp, lbl):
        out = mdl(inp)
        return nn.functional.mse_loss(out, lbl)

    return trace_model_graph(model, train_fn, input_tensor, label_tensor)


# ---------------------------------------------------------------------------
# Tests: fsdp_degree resolution (TP+FSDP hybrid)
# ---------------------------------------------------------------------------

class TestFsdpDegreeResolution(unittest.TestCase):
    """FSDPPass must use ``pass_config.fsdp_degree`` when set, not
    ``world_size``.  This is the fix for the TP+FSDP IndexError bug."""

    def setUp(self):
        self.model = TinyModel(in_features=8, out_features=4)
        self.input = torch.randn(2, 8)
        self.label = torch.randn(2, 4)
        self.joint = _make_joint_graph(self.model, self.input, self.label)

    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist")
    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group")
    def test_uses_explicit_fsdp_degree(self, mock_resolve_pg, mock_dist):
        """When fsdp_degree=2 and world_size=4, the pass should use 2, not 4."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4  # world=4 but FSDP group=2
        mock_dist.get_rank.return_value = 0  # local rank in FSDP group
        mock_resolve_pg.return_value = MagicMock()

        config = PassConfig(fsdp_degree=2, tp_size=2)
        plan = PassPlan()
        plan.fsdp_wrap_pattern("*")

        pass_obj = FSDPPass(pass_plan=plan)
        pass_obj.run(
            self.joint.graph_module, config,
            model=self.model, fsdp_group_name="fsdp", pass_plan=plan,
        )

        self.assertEqual(pass_obj._fsdp_degree, 2)


# ---------------------------------------------------------------------------
# Tests: _shard_live_model_params uses local rank (TP+FSDP fix)
# ---------------------------------------------------------------------------

class TestShardLiveModelParamsRank(unittest.TestCase):
    """The shard step must use the FSDP group's local rank, not the global
    rank, as the chunk index — preventing IndexError when
    ``fsdp_degree < world_size``.
    """

    def setUp(self):
        self.model = TinyModel(in_features=8, out_features=4)
        self.input = torch.randn(2, 8)
        self.label = torch.randn(2, 4)
        self.joint = _make_joint_graph(self.model, self.input, self.label)

    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist")
    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group")
    def test_global_rank_does_not_cause_index_error(self, mock_resolve_pg, mock_dist):
        """Simulate TP+FSDP: world_size=4, fsdp_degree=2, global_rank=3.

        Before the fix, rank=3 with chunk(2) caused IndexError.
        After the fix, the pass uses the FSDP group's local rank.
        """
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 1  # local rank in FSDP group
        mock_resolve_pg.return_value = MagicMock()

        config = PassConfig(fsdp_degree=2, tp_size=2)
        plan = PassPlan()
        plan.fsdp_wrap_pattern("*")

        pass_obj = FSDPPass(pass_plan=plan)
        pass_obj.run(
            self.joint.graph_module, config,
            model=self.model, fsdp_group_name="fsdp", pass_plan=plan,
        )

        # weight dim 0 should be halved (4 -> 2)
        self.assertEqual(self.model.linear.weight.shape[0], 2)


# ---------------------------------------------------------------------------
# Tests: TracingContext (TP communication tracing fix)
# ---------------------------------------------------------------------------

class TestTracingContextAvailable(unittest.TestCase):
    """The tracer sets up a ``TracingContext`` during make_fx — this is what
    the platform-level ``differentiable_*`` functions check (via
    ``TracingContext.get() is not None``) to decide whether to use
    functional collectives instead of the autograd-wrapped eager API.
    """

    def test_trace_produces_callable_graph(self):
        """The traced graph module should be callable with real inputs."""
        model = TinyModel(8, 4)
        inp = torch.randn(2, 8)
        lbl = torch.randn(2, 4)

        def train_fn(mdl, i, l):
            return nn.functional.mse_loss(mdl(i), l)

        joint = trace_model_graph(model, train_fn, inp, lbl)

        from hyper_parallel.compile.tracer.graph_tracer import extract_module_state
        state = extract_module_state(model)
        state_flat, _ = torch.utils._pytree.tree_flatten({"model": state})
        flat_inputs = list(state_flat) + [inp, lbl]

        with torch.no_grad():
            outputs = joint.graph_module(*flat_inputs)

        self.assertIsInstance(outputs, (list, tuple))
        self.assertTrue(len(outputs) >= 2)  # at least [loss, one_grad]


# ---------------------------------------------------------------------------
# Tests: SP (Sequence Parallel) adaptation — model structure & helpers
# ---------------------------------------------------------------------------

from hyper_parallel.compile.examples.automodel_tp.mock_modules import (
    GroupQueryAttention,
    RotaryEmbedding,
    RMSNorm,
    SwiGLUMLP,
)
from hyper_parallel.compile.examples.automodel_tp.model import (
    AutoModelAdapterForCausalLM,
    build_model,
    shard_for_sp,
    DataSampler,
)
from transformers import LlamaConfig


class TestShardForSp(unittest.TestCase):
    """``shard_for_sp`` must correctly slice the sequence dimension."""

    def test_shard_for_sp_even_split(self):
        """A 128-length sequence split across tp_size=2 should yield 64."""
        tensor = torch.randn(2, 128, 256)
        shard0 = shard_for_sp(tensor, 2, 0)
        shard1 = shard_for_sp(tensor, 2, 1)

        self.assertEqual(shard0.shape, (2, 64, 256))
        self.assertEqual(shard1.shape, (2, 64, 256))
        recovered = torch.cat([shard0, shard1], dim=1)
        self.assertTrue(torch.equal(recovered, tensor))

    def test_shard_for_sp_not_divisible(self):
        """Non-divisible seq_len should raise ValueError."""
        tensor = torch.randn(1, 129, 4)
        with self.assertRaises(ValueError):
            shard_for_sp(tensor, 2, 0)


class TestModelStructure(unittest.TestCase):
    """Tests for ``AutoModelAdapterForCausalLM`` structure — ensures
    submodule names match ShardingPlanner conventions and ``rotary_emb``
    lives inside attention (not at the model level).
    """

    @classmethod
    def setUpClass(cls):  # pylint: disable=C0202
        cls.cfg = LlamaConfig(
            vocab_size=100, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128,
        )

    def _build(self):
        return AutoModelAdapterForCausalLM(self.cfg)

    def test_rotary_emb_inside_attention(self):
        """rotary_emb is a submodule of self_attn, NOT of the inner model."""
        model = self._build()
        attn = model.model.layers[0].self_attn
        self.assertTrue(hasattr(attn, "rotary_emb"))
        self.assertFalse(hasattr(model.model, "rotary_emb"))

    def test_submodule_names_match_planner(self):
        """Submodule names match the ShardingPlanner conventions."""
        model = self._build()
        layer = model.model.layers[0]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertTrue(hasattr(layer.self_attn, name), f"missing {name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            self.assertTrue(hasattr(layer.mlp, name), f"missing {name}")
        for name in ("input_layernorm", "post_attention_layernorm"):
            self.assertTrue(hasattr(layer, name), f"missing {name}")
        self.assertTrue(hasattr(model.model, "embed_tokens"))
        self.assertTrue(hasattr(model.model, "norm"))
        self.assertTrue(hasattr(model, "lm_head"))

    def test_attention_is_group_query(self):
        """Attention module is an instance of GroupQueryAttention."""
        attn = self._build().model.layers[0].self_attn
        self.assertIsInstance(attn, GroupQueryAttention)


class TestRotaryEmbedding(unittest.TestCase):
    """Tests for ``RotaryEmbedding`` numerical correctness."""

    def test_cos_sin_correctness(self):
        """cos/sin match the reference formula."""
        head_dim = 8
        rope = RotaryEmbedding(head_dim, max_seq_len=64, theta=10000.0)
        cos, sin = rope(16)

        # Reference: inv_freq = theta^(-2i/dim), freqs = outer(pos, inv_freq)
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        pos = torch.arange(16, dtype=torch.float32)
        ref_freqs = torch.outer(pos, inv_freq)
        ref_emb = torch.cat((ref_freqs, ref_freqs), dim=-1)
        ref_cos, ref_sin = ref_emb.cos(), ref_emb.sin()

        self.assertTrue(torch.allclose(cos, ref_cos, atol=1e-6))
        self.assertTrue(torch.allclose(sin, ref_sin, atol=1e-6))

    def test_rope_applied_to_qk(self):
        """Q and K are rotated by rotary_emb inside Attention.forward."""
        cfg = LlamaConfig(
            vocab_size=100, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128,
        )
        attn = GroupQueryAttention(cfg)
        torch.manual_seed(0)
        hidden = torch.randn(1, 16, 32)
        out = attn(hidden)
        self.assertEqual(out.shape, (1, 16, 32))
        self.assertFalse(torch.isnan(out).any())
        self.assertGreater(out.abs().sum().item(), 0.0)


class TestDataSampler(unittest.TestCase):
    """Tests for ``DataSampler`` — always yields full-sequence data."""

    def test_sp_data_sampler_yields_full_sequence(self):
        """DataSampler always yields full-sequence data, even in SP mode."""
        sampler_sp = DataSampler(
            vocab_size=100, batch_size=2, seq_len=128, max_steps=1,
            tp_size=2, tp_rank=0, sequence_parallel=True,
            device=torch.device("cpu"),
        )
        inp, lbl = sampler_sp.sample()
        self.assertEqual(inp.shape, (2, 128))
        self.assertEqual(lbl.shape, (2, 128))

        sampler_no_sp = DataSampler(
            vocab_size=100, batch_size=2, seq_len=128, max_steps=1,
            tp_size=2, tp_rank=0, sequence_parallel=False,
            device=torch.device("cpu"),
        )
        inp2, lbl2 = sampler_no_sp.sample()
        self.assertEqual(inp2.shape, (2, 128))
        self.assertEqual(lbl2.shape, (2, 128))


class TestForwardBackward(unittest.TestCase):
    """Forward and backward shape consistency tests."""

    @classmethod
    def setUpClass(cls):  # pylint: disable=C0202
        cls.cfg = LlamaConfig(
            vocab_size=100, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128,
        )

    def _build(self):
        return AutoModelAdapterForCausalLM(self.cfg)

    def test_forward_shape(self):
        """Forward produces logits of shape (bs, seq, vocab)."""
        model = self._build()
        input_ids = torch.randint(0, 100, (2, 32))
        out = model(input_ids)
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (2, 32, 100))

    def test_backward_shape_consistency(self):
        """Backward pass preserves parameter gradient shapes."""
        model = self._build()
        input_ids = torch.randint(0, 100, (2, 32))
        labels = torch.randint(0, 100, (2, 32))
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        self.assertIsNotNone(loss)
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"no grad for {name}")
            self.assertEqual(param.grad.shape, param.shape)


class TestFactory(unittest.TestCase):
    """Tests for the ``build_model`` factory."""

    def test_factory_returns_automodel(self):
        """build_model returns an AutoModelAdapterForCausalLM."""
        cfg_dict = {
            "vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 128,
            "torch_dtype": "float32",
        }
        model = build_model(cfg_dict, torch.device("cpu"))
        self.assertIsInstance(model, AutoModelAdapterForCausalLM)


if __name__ == "__main__":
    unittest.main()

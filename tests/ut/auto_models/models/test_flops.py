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
"""Unit tests for ``hyper_parallel.models.flops`` FLOPs-per-token estimation."""

import unittest
from types import SimpleNamespace

from hyper_parallel.models.flops import (
    batch_seq_len,
    estimate_flops_per_token,
    resolve_flops_per_token,
)
from tests.common.mark_utils import arg_mark


# Kimi-K2.6 full geometry (MLA + fine-grained MoE); the 6N expectation is the
# value cross-checked against the previously hand-tuned per-token FLOPs.
_KIMI_GEOMETRY = dict(
    hidden_size=7168,
    num_hidden_layers=61,
    vocab_size=163840,
    num_attention_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    n_routed_experts=384,
    num_experts_per_tok=8,
    n_shared_experts=1,
    first_k_dense_replace=1,
    max_position_embeddings=4096,
)
# 6 * (61 * attn_proj + 60 * (8 routed + 1 shared experts + router)
#      + 1 dense MLP + lm_head) with the geometry above.
_KIMI_EXPECTED_6N = 190_116_397_056.0
# Additional attention score/weight term: 6 * 61 * 64 * (192 + 128) * 512.
_KIMI_EXPECTED_SEQ512 = _KIMI_EXPECTED_6N + 3_837_788_160.0


class TestEstimateFlopsPerToken(unittest.TestCase):
    """Geometry-driven estimation across attention and MoE conventions."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mla_moe_geometry_mapping_and_object(self):
        """MLA + DeepSeek-style MoE fields, accepted as mapping and object."""
        for config in (dict(_KIMI_GEOMETRY), SimpleNamespace(**_KIMI_GEOMETRY)):
            self.assertEqual(estimate_flops_per_token(config), _KIMI_EXPECTED_6N)
            self.assertEqual(
                estimate_flops_per_token(config, seq_len=512),
                _KIMI_EXPECTED_SEQ512,
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dense_gqa_geometry(self):
        """Dense GQA model: projections + dense MLP + lm_head + attention term."""
        config = SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=32,
            vocab_size=128256,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
        )
        # 6 * (32 * 41943040 attn + 32 * 3 * 4096 * 14336 MLP + 4096 * 128256)
        # + 6 * 32 * 32 * 256 * 2048 attention term.
        expected = 48_249_176_064.0
        self.assertEqual(estimate_flops_per_token(config, seq_len=2048), expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_qwen_moe_field_convention(self):
        """Qwen-MoE-style fields (num_experts/num_experts_per_tok, no shared)."""
        config = SimpleNamespace(
            hidden_size=2048,
            num_hidden_layers=48,
            vocab_size=151936,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=6144,
            moe_intermediate_size=768,
            num_experts=128,
            num_experts_per_tok=8,
        )
        self.assertEqual(estimate_flops_per_token(config), 18_249_940_992.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_insufficient_geometry_returns_none(self):
        """Missing layers/heads or MoE top-k yield None instead of a guess."""
        self.assertIsNone(estimate_flops_per_token(SimpleNamespace(hidden_size=128)))
        self.assertIsNone(estimate_flops_per_token(None))
        no_topk = dict(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            n_routed_experts=8,
        )
        self.assertIsNone(estimate_flops_per_token(no_topk))


class TestBatchSeqLen(unittest.TestCase):
    """Sequence-length extraction from micro-batches."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_seq_len_extraction(self):
        """The trailing input_ids dim of the first usable batch wins."""
        batch = {"input_ids": SimpleNamespace(shape=(2, 512))}
        self.assertEqual(batch_seq_len([batch]), 512)
        self.assertEqual(batch_seq_len(batch), 512)
        self.assertIsNone(batch_seq_len([{"labels": None}]))
        self.assertIsNone(batch_seq_len(None))


class TestResolveFlopsPerToken(unittest.TestCase):
    """Resolution policy: model property wins, config estimate is the fallback."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_model_property_wins(self):
        """A model-provided hp_flops_per_token overrides the estimate."""
        model = SimpleNamespace(hp_flops_per_token=1.0e11)
        value = resolve_flops_per_token(model, model_config=dict(_KIMI_GEOMETRY))
        self.assertEqual(value, 1.0e11)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_config_fallback_and_none(self):
        """Without the property, estimate from config; nothing -> None."""
        model = SimpleNamespace(config=SimpleNamespace(**_KIMI_GEOMETRY))
        self.assertEqual(resolve_flops_per_token(model), _KIMI_EXPECTED_6N)
        self.assertEqual(
            resolve_flops_per_token(model, seq_len=512), _KIMI_EXPECTED_SEQ512
        )
        self.assertIsNone(resolve_flops_per_token(None))

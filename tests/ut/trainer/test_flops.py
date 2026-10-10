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
"""Architecture and workload FLOPs tests, independent of model family names."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.trainer.runtime.flops import (
    CompositeFlopsEstimator, FlopsComponent, TransformerFlopsEstimator, transformer_flops,
)


def decoder_config(**overrides: object) -> SimpleNamespace:
    """Describe a small GQA decoder without a model type discriminator."""
    values = {"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "intermediate_size": 32, "vocab_size": 64, "hidden_act": "silu"}
    return SimpleNamespace(**(values | overrides))


def mla_config(**overrides: object) -> SimpleNamespace:
    """Describe MLA geometry without selecting a model family."""
    values = vars(decoder_config()) | {
        "qk_nope_head_dim": 4, "qk_rope_head_dim": 2, "v_head_dim": 4, "q_lora_rank": 8, "kv_lora_rank": 4,
        "moe_intermediate_size": 8, "num_experts_per_tok": 2, "n_shared_experts": 1, "n_routed_experts": 4,
        "first_k_dense_replace": 1,
    }
    return SimpleNamespace(**(values | overrides))


def vision_components(frozen: bool = False) -> list[FlopsComponent]:
    """Describe a patch-token VLM with all four executed components."""
    factor = 1 if frozen else 3
    return [
        FlopsComponent("pixel_values", linear_dimensions=(12, 16), forward_backward_factor=factor),
        FlopsComponent("pixel_values", config_path="vision_config", forward_backward_factor=factor),
        FlopsComponent("pixel_values", linear_dimensions=(16, 16)),
        FlopsComponent("input_ids", config_path="text_config", causal=True, include_logits=True, cp_partitioned=True),
    ]


class TestFlops(unittest.TestCase):
    """Check analytic cases, active expert scaling, composition and device metadata."""

    def test_dense_gqa_and_gelu(self) -> None:
        """Hand-count Q/K/V/O, gated MLP, causal attention and vocabulary GEMMs."""
        config = decoder_config()
        self.assertEqual(transformer_flops(config).estimate(16, 128), 565248)
        mha = transformer_flops(decoder_config(num_key_value_heads=4)).estimate(16, 128)
        self.assertEqual(mha - 565248, 49152)
        gelu = transformer_flops(decoder_config(hidden_act="gelu")).estimate(16, 128)
        self.assertEqual(565248 - gelu, 98304)
        for name in ("llama", "qwen3", "custom_decoder", None):
            with self.subTest(name=name):
                self.assertEqual(transformer_flops(decoder_config(model_type=name)), transformer_flops(config))

    def test_moe_counts_active_experts_and_shared_width(self) -> None:
        """Inactive stored experts do not inflate theoretical model FLOPs."""
        config = decoder_config(num_experts=4, moe_router_topk=2, moe_ffn_hidden_size=8,
                                moe_layer_freq=[0, 1], moe_shared_expert_intermediate_size=8)
        coefficients = transformer_flops(config)
        self.assertEqual(coefficients.estimate(16, 128), 528384)
        config.moe_latent_size = 16
        self.assertEqual(transformer_flops(config).estimate(16, 128) - 528384, 49152)
        config.moe_latent_size = None
        config.num_experts = 64
        self.assertEqual(transformer_flops(config), coefficients)
        config.moe_router_topk = 3
        self.assertEqual(transformer_flops(config).estimate(16, 128) - 528384, 36864)

    def test_hf_and_megatron_geometry_aliases(self) -> None:
        """Both configuration dialects describe identical architecture."""
        generic = {"hidden_size": 16, "num_layers": 2, "num_attention_heads": 4, "num_query_groups": 2,
                   "ffn_hidden_size": 32, "padded_vocab_size": 64, "swiglu": True}
        self.assertEqual(transformer_flops(generic), transformer_flops(decoder_config()))
        generic.update(group_query_attention=False, num_query_groups=1, vocab_size=32)
        self.assertEqual(transformer_flops(generic), transformer_flops(decoder_config(num_key_value_heads=4)))
        self.assertEqual(transformer_flops(mla_config()).estimate(8, 64), 271488)

    def test_mtp_requires_execution_not_checkpoint_metadata(self) -> None:
        """A checkpoint may store MTP geometry that its model does not execute."""
        config = mla_config(num_nextn_predict_layers=3)
        estimator = TransformerFlopsEstimator(config)
        baseline = estimator({"input_ids": torch.zeros(1, 8)})
        self.assertEqual(baseline, transformer_flops(config).estimate(8, 64))
        model = SimpleNamespace(mtp=SimpleNamespace(layers=[
            SimpleNamespace(transformer_layer=SimpleNamespace(mlp=SimpleNamespace(experts=[]))),
        ]))
        actual = TransformerFlopsEstimator(config, model)({"input_ids": torch.zeros(1, 8)})
        self.assertEqual(actual, transformer_flops(config, mtp_layer_types=(True,)).estimate(8, 64))
        self.assertGreater(actual, baseline)

    def test_packed_attention_and_cp_do_not_sync_to_host(self) -> None:
        """Equal token counts need different attention work for lengths [3, 5]."""
        estimator = TransformerFlopsEstimator(decoder_config())
        batch = {"input_ids": torch.zeros(1, 4), "cu_seq_lens": torch.tensor([0, 3, 8])}
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected host sync")):
            actual = estimator(batch, cp_size=2)
        self.assertEqual(actual.item(), transformer_flops(decoder_config()).estimate(8, 34))
        self.assertLess(actual.item(), estimator({"input_ids": torch.zeros(1, 8)}))

    def test_multimodal_composition_and_freeze_factors(self) -> None:
        """Do not silently report just the nested language decoder as the full model."""
        config = SimpleNamespace(text_config=decoder_config(), vision_config=decoder_config(), patch_dim=12)
        self.assertIsNone(TransformerFlopsEstimator(config)({"input_ids": torch.zeros(2, 8)}))
        batch = {"input_ids": torch.zeros(2, 8), "pixel_values": torch.zeros(2, 4, 12)}
        full = CompositeFlopsEstimator(config, vision_components())(batch)
        frozen = CompositeFlopsEstimator(config, vision_components(frozen=True))(batch)
        # Per vision forward: 8*(2*2*(768+1536)) + 32*(2*32*2) = 77824.
        # Patch embedding: 2*8*12*16=3072; projector: 2*8*16*16=4096.
        self.assertEqual(full, 565248 + 3 * (77824 + 3072 + 4096))
        self.assertEqual(full - frozen, 2 * (77824 + 3072))

    def test_unsupported_structures_and_invalid_geometry(self) -> None:
        """Omit unknown work instead of silently estimating a different architecture."""
        for extras in ({"layer_types": ["linear_attention", "full_attention"]}, {"index_topk": 4},
                       {"vision_config": {}}, {"sliding_window": 8}):
            with self.subTest(extras=extras):
                config = decoder_config(**extras)
                if extras == {"vision_config": {}}:
                    config.vision_config = {"hidden_size": 8}
                self.assertIsNone(transformer_flops(config))
        for extras in ({"hidden_size": 0}, {"num_key_value_heads": 3}, {"mlp_layer_types": ["dense"]}):
            with self.subTest(extras=extras), self.assertRaises(ValueError):
                transformer_flops(decoder_config(**extras))
        with self.assertRaises(ValueError):
            transformer_flops(decoder_config()).estimate(1, 1, forward_backward_factor=0)


    def test_variable_audio_lengths_optional_branches_and_input_gradients(self) -> None:
        """Compose executed audio frames with a frozen weight that still needs dgrad."""
        geometry = SimpleNamespace(audio_config=decoder_config())
        components = [
            FlopsComponent("audio_tokens", config_path="audio_config", lengths_key="audio_lengths",
                           forward_backward_factor=2),
            FlopsComponent("audio_tokens", linear_dimensions=(16, 8), lengths_key="audio_lengths"),
            FlopsComponent("image_tokens", linear_dimensions=(12, 8), optional_input=True),
        ]
        estimator = CompositeFlopsEstimator(geometry, components)
        batch = {"audio_lengths": torch.tensor([3, 5])}
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected host sync")):
            actual = estimator(batch, cp_size=2)
        # Frozen parameter GEMMs use 2; attention still needs both activation gradients (3).
        expected = 2 * 9216 * 8 + 3 * 128 * 34 + 3 * 2 * 8 * 16 * 8
        self.assertEqual(actual.item(), expected)
        self.assertIsNone(estimator({}))
        batch["image_tokens"] = torch.zeros(1, 2, 12)
        self.assertEqual(estimator(batch).item() - expected, 3 * 2 * 2 * 12 * 8)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            estimator({"audio_lengths": torch.ones(2, 2)})
        for component in (FlopsComponent("x"), FlopsComponent("x", linear_dimensions=(8,)),
                          FlopsComponent("x", config_path="audio_config", linear_dimensions=(8, 8))):
            with self.subTest(component=component), self.assertRaises(ValueError):
                CompositeFlopsEstimator(geometry, [component])

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
"""CPU regression tests for configurable DeepSeek-V4.1 validation crops."""

import json
import tempfile
import unittest
from pathlib import Path

from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (
    build_deepseek_v41_validation_config,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from tests.common.mark_utils import arg_mark


def _write_engram_assets(
        directory: str,
        num_hidden_layers: int = 4,
        layer_ids: tuple[int, ...] = (1,),
        head_dim: int = 4,
) -> Path:
    """Write a minimal but internally consistent scaled Engram asset."""
    available_primes = (
        [[17, 19], [23, 29]],
        [[31, 37], [41, 43]],
    )
    if len(layer_ids) > len(available_primes):
        raise ValueError("the test asset helper supports at most two Engram layers")
    primes = list(available_primes[:len(layer_ids)])
    assets = {
        "source_model_type": "deepseek_v41",
        "num_hidden_layers": num_hidden_layers,
        "layer_ids": list(layer_ids),
        "bucket_base": 16,
        "max_ngram_size": 3,
        "num_heads": 2,
        "head_dim": head_dim,
        "primes": primes,
        "num_embeddings": [
            sum(value for row in layer_primes for value in row)
            for layer_primes in primes
        ],
        "multipliers": [[101, 103, 107] for _ in layer_ids],
        "token_map": list(range(64)),
        "pad_token_id": 0,
    }
    path = Path(directory) / "engram.json"
    path.write_text(json.dumps(assets), encoding="utf-8")
    return path


def _write_released_config(directory: str) -> Path:
    """Write a release-shaped config whose dimensions have exact crop ratios."""
    text_config = {
        "vocab_size": 64,
        "hidden_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "compress_rope_theta": 160000.0,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 64,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        "engram_layer_ids": [1, 14],
        "engram_head_dim": 32,
    }
    source = {
        "model_type": "deepseek_v41",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "image_token_id": 3,
        "text_config": text_config,
        "vision_config": {
            "num_hidden_layers": 32,
            "hidden_size": 128,
            "num_attention_heads": 16,
            "intermediate_size": 192,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
class TestDeepseekV41ValidationCrop(unittest.TestCase):
    """Depth crops preserve retained model roles and reject inconsistent layouts."""

    def test_validation_builder_supports_released_lightweight_depth_crop(self):
        """Build the released four-text-layer, one-vision-layer validation crop.

        Feature: DeepSeek-V4.1 validation crop depth controls.
        Description: Retain released widths while selecting four text layers and one vision layer.
        Expectation: Shared-attention, Reindex, Engram, and vision roles remain internally consistent.
        """
        with tempfile.TemporaryDirectory() as directory:
            _write_released_config(directory)
            assets_path = _write_engram_assets(
                directory,
                num_hidden_layers=40,
                layer_ids=(1, 14),
                head_dim=32,
            )
            config = build_deepseek_v41_validation_config(
                directory,
                str(assets_path),
                num_hidden_layers=4,
                text_parameter_divisor=1,
                enable_vision=True,
                vision_num_hidden_layers=1,
                vision_parameter_divisor=1,
                num_routed_experts=16,
            )
            model = DeepseekV41ForCausalLM(config)

        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.v41_vision_num_hidden_layers, 1)
        self.assertEqual(config.v41_kv_source_layer_ids, [2])
        self.assertEqual(config.v41_index_source_layer_ids, [2, 3])
        self.assertEqual(config.v41_candidate_source_layer_id, 2)
        self.assertEqual(config.v41_engram_layer_ids, [1])
        self.assertTrue(hasattr(model.model.layers[1], "engram"))
        self.assertFalse(hasattr(model.model.layers[3], "engram"))

    def test_default_depth_preserves_released_layer_roles(self):
        """Omitting the new controls retains the original full-depth behavior."""
        with tempfile.TemporaryDirectory() as directory:
            _write_released_config(directory)
            assets_path = _write_engram_assets(
                directory, num_hidden_layers=40, layer_ids=(1, 14), head_dim=8
            )
            config = build_deepseek_v41_validation_config(directory, str(assets_path))

        self.assertEqual(config.num_hidden_layers, 40)
        self.assertEqual(config.v41_vision_num_hidden_layers, 32)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.o_groups, 8)
        self.assertEqual(len(config.layer_types), 40)
        self.assertEqual(len(config.mlp_layer_types), 40)
        self.assertEqual(config.v41_kv_source_layer_ids, [2, 8, 14, 20])
        self.assertEqual(config.v41_index_source_layer_ids, [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(config.v41_candidate_source_layer_id, 20)
        self.assertEqual(config.v41_engram_layer_ids, [1, 14])
        self.assertFalse(hasattr(config, "v41_validation_reindex_remap"))

    def test_depth_controls_reject_out_of_range_layers(self):
        """Both optional depths must fit inside the corresponding released stack."""
        with tempfile.TemporaryDirectory() as directory:
            _write_released_config(directory)
            assets_path = _write_engram_assets(
                directory, num_hidden_layers=40, layer_ids=(1, 14), head_dim=8
            )
            cases = (
                ({"num_hidden_layers": 0}, "num_hidden_layers must be in"),
                ({"num_hidden_layers": 41}, "num_hidden_layers must be in"),
                ({"enable_vision": True, "vision_num_hidden_layers": 0},
                 "vision_num_hidden_layers must be in"),
                ({"enable_vision": True, "vision_num_hidden_layers": 33},
                 "vision_num_hidden_layers must be in"),
                ({"vision_num_hidden_layers": 1}, "requires enable_vision=true"),
            )
            for overrides, expected_error in cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        build_deepseek_v41_validation_config(
                            directory, str(assets_path), **overrides
                        )

    def test_reindex_requires_a_retained_source_and_following_layer(self):
        """Shallow crops fail clearly when they cannot host the Reindex hierarchy."""
        with tempfile.TemporaryDirectory() as directory:
            _write_released_config(directory)
            assets_path = _write_engram_assets(
                directory, num_hidden_layers=40, layer_ids=(1, 14), head_dim=8
            )
            for depth, expected_error in (
                    (2, "no shared-attention source"),
                    (3, "no layer available for Reindex"),
            ):
                with self.subTest(depth=depth):
                    with self.assertRaisesRegex(ValueError, expected_error):
                        build_deepseek_v41_validation_config(
                            directory, str(assets_path), num_hidden_layers=depth
                        )

    def test_disabled_validation_indexer_only_filters_retained_layers(self):
        """Disabling the smoke indexer avoids remapping a cropped-out candidate source."""
        with tempfile.TemporaryDirectory() as directory:
            _write_released_config(directory)
            assets_path = _write_engram_assets(
                directory, num_hidden_layers=40, layer_ids=(1, 14), head_dim=8
            )
            config = build_deepseek_v41_validation_config(
                directory, str(assets_path), num_hidden_layers=3,
                exercise_post_training_indexer=False,
            )

        self.assertEqual(config.v41_kv_source_layer_ids, [2])
        self.assertEqual(config.v41_index_source_layer_ids, [2])
        self.assertEqual(config.v41_candidate_source_layer_id, -1)
        self.assertEqual(config.v41_candidate_topk_blocks, 2048)
        self.assertEqual(config.v41_engram_layer_ids, [1])
        self.assertEqual(len(config.v41_engram_num_embeddings), 1)
        self.assertEqual(config.v41_compress_ratios, [0, 0, 2])
        self.assertFalse(hasattr(config, "v41_validation_reindex_remap"))

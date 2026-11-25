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
"""Small V4.1 state-boundary fixture retaining all forty decoder roles."""

import json
from pathlib import Path

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM


def make_config(directory: str) -> DeepseekV4Config:
    """Create real Engram assets and a narrow full-depth decoder configuration.

    Args:
        directory: Temporary directory for the per-test Engram assets.
    """
    assets = {
        "layer_ids": [1, 14], "max_ngram_size": 4, "num_heads": 8, "head_dim": 4,
        "token_map": list(range(64)), "pad_token_id": 0,
        "primes": [[[17 + 2 * head for head in range(8)] for _ in range(3)]] * 2,
        "multipliers": [[3, 5, 7, 11], [13, 17, 19, 23]],
        "num_embeddings": [576, 576],
    }
    path = Path(directory) / "engram.json"
    path.write_text(json.dumps(assets), encoding="utf-8")
    ratios = [0, 0] + [2] * 18 + [1] * 20
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg  # Generated Transformers config fields.
        vocab_size=64, hidden_size=32, moe_intermediate_size=32, num_hidden_layers=40,
        num_attention_heads=8, num_key_value_heads=1, head_dim=8, q_lora_rank=16,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=6, hc_mult=4,
        # V4.1 constructs CSA2 sources from v41_compress_ratios and source IDs.
        layer_types=["sliding_attention"] * 40,
        mlp_layer_types=["moe"] * 40, swiglu_limit=1.5, sliding_window=4,
        o_groups=8, o_lora_rank=4, index_n_heads=2, index_head_dim=4,
        index_topk=512, partial_rotary_factor=0.5, tie_word_embeddings=False,
    )
    config.architectures = ["DeepseekV41ForCausalLM"]
    config.v41_compress_ratios = ratios
    config.v41_kv_source_layer_ids = [2, 8, 14, 20]
    config.v41_index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
    config.v41_candidate_source_layer_id = 20
    config.v41_candidate_topk_blocks = 16
    config.v41_candidate_block_size = 64
    config.v41_indexer_loss_coeff = 0.001
    config.v41_engram_layer_ids = [1, 14]
    config.v41_engram_assets_path = str(path)
    config.v41_vision_enabled = False
    config.v41_dspark_depth = 0
    config._attn_implementation = "eager"
    return config


def copy_weights(reference: DeepseekV41ForCausalLM) -> DeepseekV41ForCausalLM:
    """Construct another stage candidate with the same weights and trainable parameters.

    Args:
        reference: Unpartitioned model to reproduce.
    """
    model = DeepseekV41ForCausalLM(reference.config)
    model.load_state_dict(reference.state_dict())
    for name, parameter in reference.named_parameters():
        model.get_parameter(name).requires_grad_(parameter.requires_grad)
    return model

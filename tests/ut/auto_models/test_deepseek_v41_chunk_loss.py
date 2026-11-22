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
"""Native V4.1 family composition for complete-model chunked CE and DSpark."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM


def tiny_config(assets_path: str, draft: bool = False) -> DeepseekV4Config:
    """Keep actual native layers, routing, hyper-connections and an optional complete draft stage."""
    config = DeepseekV4Config.from_dict(
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "moe_intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "q_lora_rank": 8,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "layer_types": ["sliding_attention"] * 2,
            "mlp_layer_types": ["moe"] * 2,
            "swiglu_limit": 1.5,
            "sliding_window": 4,
            "o_groups": 2,
            "o_lora_rank": 4,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "partial_rotary_factor": 0.5,
        }
    )
    config.v41_compress_ratios = [0, 0]
    config.v41_kv_source_layer_ids = []
    config.v41_index_source_layer_ids = []
    config.v41_engram_assets_path = assets_path
    config.v41_vision_enabled = False
    config.v41_dspark_depth = 1 if draft else 0
    config.v41_dspark_n_routed_experts = 4
    config.v41_dspark_top_k = 2
    config.v41_dspark_target_layer_ids = [1]
    config.v41_dspark_block_size = 2
    config.v41_dspark_window = 4
    config.v41_dspark_markov_rank = 4
    config.v41_dspark_noise_token_id = 31
    config.v41_dspark_confidence_coeff = 0.1
    config.v41_dspark_loss_chunk_size = 3
    config.v41_dspark_loss_coeff = 0.7
    config._attn_implementation = "eager"
    return config


class TestDeepseekV41ChunkLoss(unittest.TestCase):
    """Compare complete native forwards and gradients; no runtime forward installation."""

    def setUp(self) -> None:
        """Provide real small hash assets for the complete native Engram branch."""
        # The unittest cleanup registered below owns this resource until the test finishes.
        directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(directory.cleanup)
        self.asset_path = Path(directory.name) / "engram.json"
        self.asset_path.write_text(
            json.dumps(
                {
                    "layer_ids": [0],
                    "num_embeddings": [12],
                    "head_dim": 4,
                    "max_ngram_size": 2,
                    "num_heads": 2,
                    "token_map": list(range(32)),
                    "pad_token_id": 0,
                    "primes": [[[5, 7]]],
                    "multipliers": [[1, 3]],
                }
            )
        )

    def test_chunked_full_model_and_draft_match_full_logits(self) -> None:
        """Native chunked CE and optional draft supervision equal an explicit full-logit oracle."""
        for draft in (False, True):
            with self.subTest(draft=draft):
                torch.manual_seed(31)
                model = DeepseekV41ForCausalLM(tiny_config(str(self.asset_path), draft))
                reference = DeepseekV41ForCausalLM(tiny_config(str(self.asset_path), draft))
                reference.load_state_dict(model.state_dict())
                ids = torch.randint(0, 30, (1, 7))
                targets = torch.randint(0, 30, (1, 7))
                targets[0, -1] = -100
                mask = torch.ones_like(targets, dtype=torch.bool)
                mask[0, 0] = False
                rng = torch.get_rng_state()
                hidden = reference.model(input_ids=ids, use_cache=False).last_hidden_state
                logits = reference.lm_head(hidden)
                supervised = targets.masked_fill(~mask, -100)
                expected = F.cross_entropy(logits.float().flatten(0, 1), supervised.flatten())
                if draft:
                    draft_loss, _ = reference.dspark(
                        reference.model.dspark_target_hiddens,
                        ids,
                        targets,
                        reference.model.dspark_inputs_embeds,
                        reference.lm_head,
                    )
                    expected = expected + reference.dspark_loss_coeff * draft_loss
                torch.set_rng_state(rng)
                output = model(input_ids=ids, chunk_loss_targets=targets, chunk_loss_mask=mask, chunk_loss_chunk_size=3)
                actual = output.loss_sum / output.valid_token_count
                if draft:
                    self.assertIsNotNone(output.aux_loss)
                    actual = actual + output.aux_loss_coef * output.aux_loss
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                expected.backward()
                actual.backward()
                for name, parameter in model.named_parameters():
                    wanted = reference.get_parameter(name).grad
                    if wanted is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        torch.testing.assert_close(parameter.grad, wanted, rtol=1e-4, atol=1e-6, msg=name)

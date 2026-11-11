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
"""Qwen3 family structure; all implementation choices remain explicit in RL YAML."""


QWEN3_MODELS = (
    {
        "model_type": "qwen3",
        "vllm_architecture": "HyperParallelQwen3ForCausalLM",
        "execution_targets": {
            "causal_lm": "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM",
            "decoder": "transformers.models.qwen3.modeling_qwen3.Qwen3Model",
            "layer": "transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer",
            "attention": "transformers.models.qwen3.modeling_qwen3.Qwen3Attention",
            "rotary": "transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding",
        },
        "builder_target": "rl.workers.adapters.qwen3.Qwen3AutoModel",
        "attention_factory": (
            "hyper_parallel.models.qwen3.adapter.conversion.module_replacement.replace_qwen3_flash_attention"
        ),
        "row_projections": ("*.self_attn.o_proj", "*.mlp.down_proj"),
    },
    {
        "model_type": "qwen3_moe",
        "vllm_architecture": "HyperParallelQwen3MoeForCausalLM",
        "execution_targets": {
            "causal_lm": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM",
            "decoder": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel",
            "layer": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeDecoderLayer",
            "attention": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention",
            "rotary": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRotaryEmbedding",
            "router": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeTopKRouter",
            "moe_block": "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
        },
        "builder_target": "rl.workers.adapters.qwen3.Qwen3AutoModel",
        "attention_factory": (
            "hyper_parallel.models.qwen3_moe.adapter.conversion.module_replacement.replace_qwen3_moe_flash_attention"
        ),
        "row_projections": ("*.self_attn.o_proj",),
        "router_adapter": "qwen3moe",
    },
)

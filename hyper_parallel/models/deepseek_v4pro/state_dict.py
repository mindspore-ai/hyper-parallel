# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""HF ↔ hyper state-dict adapter for DeepSeek-V4Pro."""
# pylint: disable=forbidden-backend-import,missing-public-type-hints,missing-public-docstring
from __future__ import annotations

__all__ = ["DeepSeekV4ProStateDictAdapter"]

import re
from typing import Dict, Optional

import torch

from hyper_parallel.models.deepseek_v4pro.checkpoint import load_hf_deepseek_v4pro_state_dict


class DeepSeekV4ProStateDictAdapter:
    """State-dict adapter for DeepSeek-V4Pro."""

    @staticmethod
    def load_hf_state_dict(
        weights_path: str,
        model_config,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        return load_hf_deepseek_v4pro_state_dict(
            weights_path,
            num_hidden_layers=model_config.num_hidden_layers,
            num_experts=model_config.n_routed_experts,
            moe_intermediate_size=model_config.moe_intermediate_size,
            dtype=dtype,
            model_config=model_config,
        )

    @staticmethod
    def save_hf_state_dict(
        state_dict: Dict[str, torch.Tensor],
        model_config,
    ) -> Dict[str, torch.Tensor]:
        del model_config
        hf_sd: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if key == "lm_head.weight":
                hf_sd[key] = tensor
                continue
            if key == "model.embed_tokens.weight":
                hf_sd["embed.weight"] = tensor
                continue
            if key == "model.norm.weight":
                hf_sd["norm.weight"] = tensor
                continue
            if ".hc_head." in key:
                if key.endswith(".hc_head_fn"):
                    hf_sd["hc_head_fn"] = tensor
                elif key.endswith(".hc_head_base"):
                    hf_sd["hc_head_base"] = tensor
                elif key.endswith(".hc_head_scale"):
                    hf_sd["hc_head_scale"] = tensor
                else:
                    hf_sd[key] = tensor
                continue
            if key.startswith("model.layers."):
                match = re.match(r"model\.layers\.(\d+)\.(.+)", key)
                if not match:
                    hf_sd[key] = tensor
                    continue
                layer_idx = int(match.group(1))
                suffix = match.group(2)
                prefix = f"layers.{layer_idx}."
                if suffix == "input_layernorm.weight":
                    hf_sd[prefix + "attn_norm.weight"] = tensor
                elif suffix == "post_attention_layernorm.weight":
                    hf_sd[prefix + "ffn_norm.weight"] = tensor
                elif suffix == "self_attn.attn_sink":
                    hf_sd[prefix + "attn.attn_sink"] = tensor
                elif suffix == "self_attn.q_norm.weight":
                    hf_sd[prefix + "attn.q_norm.weight"] = tensor
                elif suffix == "self_attn.kv_norm.weight":
                    hf_sd[prefix + "attn.kv_norm.weight"] = tensor
                elif suffix == "self_attn.q_lora.weight":
                    hf_sd[prefix + "attn.wq_a.weight"] = tensor
                elif suffix == "self_attn.q_proj.weight":
                    hf_sd[prefix + "attn.wq_b.weight"] = tensor
                elif suffix == "self_attn.wkv.weight":
                    hf_sd[prefix + "attn.wkv.weight"] = tensor
                elif suffix == "self_attn.wo_a.weight":
                    hf_sd[prefix + "attn.wo_a.weight"] = tensor
                elif suffix == "self_attn.wo_b.weight":
                    hf_sd[prefix + "attn.wo_b.weight"] = tensor
                elif suffix == "moe.router.gate.weight":
                    hf_sd[prefix + "ffn.gate.weight"] = tensor
                elif suffix == "moe.expert_bias":
                    hf_sd[prefix + "ffn.gate.bias"] = tensor
                elif suffix == "moe.shared_experts.w1.weight":
                    hf_sd[prefix + "ffn.shared_experts.w1.weight"] = tensor
                elif suffix == "moe.shared_experts.w2.weight":
                    hf_sd[prefix + "ffn.shared_experts.w2.weight"] = tensor
                elif suffix == "moe.shared_experts.w3.weight":
                    hf_sd[prefix + "ffn.shared_experts.w3.weight"] = tensor
                elif suffix in (
                    "hc_attn_base",
                    "hc_attn_fn",
                    "hc_attn_scale",
                    "hc_ffn_base",
                    "hc_ffn_fn",
                    "hc_ffn_scale",
                ):
                    hf_sd[prefix + suffix] = tensor
                elif suffix == "moe.experts.w1":
                    for expert_idx in range(tensor.shape[0]):
                        hf_sd[f"{prefix}ffn.experts.{expert_idx}.w1.weight"] = tensor[expert_idx]
                elif suffix == "moe.experts.w2":
                    for expert_idx in range(tensor.shape[0]):
                        hf_sd[f"{prefix}ffn.experts.{expert_idx}.w2.weight"] = tensor[expert_idx]
                elif suffix == "moe.experts.w3":
                    for expert_idx in range(tensor.shape[0]):
                        hf_sd[f"{prefix}ffn.experts.{expert_idx}.w3.weight"] = tensor[expert_idx]
                else:
                    hf_sd[key] = tensor
                continue
            hf_sd[key] = tensor
        return hf_sd

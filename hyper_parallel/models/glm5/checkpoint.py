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
"""HuggingFace GLM5/GLM4 safetensors to hyper state-dict conversion."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)

_PER_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
)

_SUPPORTED_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.bias",
    "self_attn.kv_lora_a_proj.weight",
    "self_attn.kv_lora_a_proj.bias",
    "self_attn.kv_lora_norm.weight",
    "self_attn.kv_lora_b_proj.weight",
    "self_attn.kv_lora_b_proj.bias",
    "dsa_indexer.query_proj.weight",
    "dsa_indexer.key_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


def _resolve_shard_path(weights_path: str, shard: str) -> str:
    """Return a shard path constrained to the checkpoint directory."""
    if os.path.isabs(shard):
        raise ValueError(f"GLM5 checkpoint shard must be relative: {shard}")
    base_path = os.path.realpath(weights_path)
    shard_path = os.path.realpath(os.path.join(base_path, shard))
    if os.path.commonpath([base_path, shard_path]) != base_path:
        raise ValueError(f"GLM5 checkpoint shard escapes weights_path: {shard}")
    return shard_path


def _remap_key(hf_key: str, max_layer: int) -> Optional[str]:
    """Map a GLM-family HF key to the hyper GLM5 dense module layout."""
    if hf_key == "lm_head.weight":
        return "lm_head.weight"
    standard_prefix = "model."
    if hf_key.startswith(standard_prefix):
        tail = hf_key[len(standard_prefix):]
        if tail.startswith("layers."):
            try:
                layer_i = int(tail.split(".")[1])
            except (IndexError, ValueError):
                return None
            if layer_i > max_layer:
                return None
        if tail.startswith(("embed_tokens.", "layers.", "norm.")):
            return hf_key

    legacy_map = {
        "transformer.embedding.word_embeddings.": "model.embed_tokens.",
        "transformer.encoder.layers.": "model.layers.",
        "transformer.encoder.final_layernorm.": "model.norm.",
        "transformer.output_layer.": "lm_head.",
    }
    for old_prefix, new_prefix in legacy_map.items():
        if hf_key.startswith(old_prefix):
            tail = hf_key[len(old_prefix):]
            mapped = f"{new_prefix}{tail}"
            if mapped.startswith("model.layers."):
                try:
                    layer_i = int(mapped.split(".")[2])
                except (IndexError, ValueError):
                    return None
                if layer_i > max_layer:
                    return None
            if mapped == "lm_head.weight.weight":
                return "lm_head.weight"
            return mapped

    logger.debug("Unmapped GLM-family key dropped: %s", hf_key)
    return None


def _is_structural_glm5_key(hf_key: str, max_layer: int) -> bool:
    """Return True for in-range GLM5 model weights that must not be dropped."""
    if not hf_key.startswith("model."):
        return hf_key == "lm_head.weight"
    tail = hf_key[len("model."):]
    if tail.startswith("layers."):
        try:
            layer_i = int(tail.split(".")[1])
        except (IndexError, ValueError):
            return True
        return layer_i <= max_layer
    return tail.startswith(("embed_tokens.", "norm."))


def _is_supported_mapped_key(mapped_key: str) -> bool:
    """Return True if mapped key belongs to the implemented GLM5 layout."""
    if mapped_key in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ):
        return True
    layer_prefix = "model.layers."
    if not mapped_key.startswith(layer_prefix):
        return False
    tail = mapped_key[len(layer_prefix):]
    parts = tail.split(".", 1)
    if len(parts) != 2:
        return False
    return parts[1] in _SUPPORTED_LAYER_SUFFIXES


def _pack_per_expert_tensors(
    expert_tensors: Dict[Tuple[int, int, str], torch.Tensor],
    num_experts: int,
) -> Dict[str, torch.Tensor]:
    """Pack per-expert HF MoE tensors into GLM5 expert-major parameters."""
    packed: Dict[str, torch.Tensor] = {}
    layers = sorted({layer_i for layer_i, _, _ in expert_tensors})
    for layer_i in layers:
        gate = []
        up = []
        down = []
        for expert_i in range(num_experts):
            try:
                gate_w = expert_tensors[(layer_i, expert_i, "gate_proj")]
                up_w = expert_tensors[(layer_i, expert_i, "up_proj")]
                down_w = expert_tensors[(layer_i, expert_i, "down_proj")]
            except KeyError as exc:
                raise ValueError(
                    f"GLM5 HF MoE layer {layer_i} missing expert {expert_i} "
                    f"tensor for packed expert conversion"
                ) from exc
            if gate_w.shape != up_w.shape:
                raise ValueError(
                    f"GLM5 HF MoE layer {layer_i} expert {expert_i} gate/up "
                    f"shapes differ: {tuple(gate_w.shape)} vs {tuple(up_w.shape)}"
                )
            gate.append(gate_w)
            up.append(up_w)
            target_down_shape = (gate_w.shape[1], gate_w.shape[0])
            if down_w.shape == target_down_shape:
                down.append(down_w)
                continue
            if down_w.shape == gate_w.shape:
                down.append(down_w.transpose(0, 1).contiguous())
                continue
            raise ValueError(
                f"GLM5 HF MoE layer {layer_i} expert {expert_i} down_proj "
                f"shape {tuple(down_w.shape)} does not match gate/up shape "
                f"{tuple(gate_w.shape)}"
            )

        prefix = f"model.layers.{layer_i}.mlp.experts"
        packed[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([g, u], dim=0) for g, u in zip(gate, up)],
            dim=0,
        )
        packed[f"{prefix}.down_proj"] = torch.stack(down, dim=0)
    return packed


def load_hf_glm5_state_dict(
    weights_path: str,
    num_hidden_layers: int,
    num_experts: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load a GLM5/GLM4 safetensors checkpoint into hyper key names."""
    idx_path = os.path.join(weights_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(
            f"GLM5 loader needs {idx_path}; pass a directory containing "
            "model.safetensors.index.json."
        )
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx["weight_map"]
    shard_to_keys: Dict[str, list] = {}
    for hf_key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(hf_key)

    max_layer = num_hidden_layers - 1

    def _cast(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype) if dtype is not None and tensor.dtype != dtype else tensor

    hyper_sd: Dict[str, torch.Tensor] = {}
    expert_tensors: Dict[Tuple[int, int, str], torch.Tensor] = {}
    unsupported = []
    skipped = 0
    for shard in sorted(shard_to_keys.keys()):
        shard_path = _resolve_shard_path(weights_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in shard_to_keys[shard]:
                expert_match = _PER_EXPERT_RE.match(hf_key)
                if expert_match:
                    layer_i = int(expert_match.group("layer"))
                    if layer_i > max_layer:
                        skipped += 1
                        continue
                    expert_tensors[
                        (
                            layer_i,
                            int(expert_match.group("expert")),
                            expert_match.group("kind"),
                        )
                    ] = _cast(f.get_tensor(hf_key))
                    continue
                mapped = _remap_key(hf_key, max_layer)
                if mapped is None:
                    if _is_structural_glm5_key(hf_key, max_layer):
                        unsupported.append(hf_key)
                    else:
                        skipped += 1
                    continue
                if not _is_supported_mapped_key(mapped):
                    unsupported.append(hf_key)
                    continue
                hyper_sd[mapped] = _cast(f.get_tensor(hf_key))

    if expert_tensors:
        if num_experts is None:
            raise ValueError(
                "GLM5 HF checkpoint has per-expert MoE weights; pass "
                "num_experts for packed expert conversion."
            )
        hyper_sd.update(_pack_per_expert_tensors(expert_tensors, num_experts))
    if unsupported:
        examples = ", ".join(unsupported[:5])
        raise ValueError(
            "Unsupported GLM5 HF structural keys encountered; refusing to "
            f"silently random-initialize model weights: {examples}"
        )

    if "lm_head.weight" not in hyper_sd and "model.embed_tokens.weight" in hyper_sd:
        hyper_sd["lm_head.weight"] = hyper_sd["model.embed_tokens.weight"].clone()

    logger.info(
        "GLM-family HF -> hyper state_dict ready: %d keys (%d skipped)",
        len(hyper_sd),
        skipped,
    )
    return hyper_sd


__all__ = ["load_hf_glm5_state_dict"]

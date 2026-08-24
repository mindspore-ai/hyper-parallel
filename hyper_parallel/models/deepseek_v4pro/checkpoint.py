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
"""HF safetensors loader for DeepSeek-V4Pro."""
# pylint: disable=forbidden-backend-import,missing-public-type-hints,unused-argument
from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

_EXPERT_RE = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<kind>w1|w2|w3)\.weight$")


def _resolve_weights_files(weights_path: str) -> Dict[str, list[str]]:
    """Return ``shard -> hf_keys`` grouped from either an index or raw shards."""
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    idx_path = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, "r", encoding="utf-8") as handle:
            idx = json.load(handle)
        weight_map: Dict[str, str] = idx["weight_map"]
        shard_to_keys: Dict[str, list[str]] = {}
        for hf_key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(hf_key)
        return shard_to_keys

    shard_to_keys = {}
    for shard_path in sorted(glob.glob(os.path.join(weights_path, "*.safetensors"))):
        shard_name = os.path.basename(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:  # type: ignore[name-defined]
            shard_to_keys[shard_name] = list(handle.keys())
    if not shard_to_keys:
        raise FileNotFoundError(
            f"No safetensors shards found under {weights_path!r}. "
            "Expected model.safetensors.index.json or one or more *.safetensors files."
        )
    return shard_to_keys


def _normalize_key(hf_key: str) -> str:
    """Strip optional top-level prefixes from a HF checkpoint key."""
    if hf_key.startswith("model.language_model."):
        return hf_key[len("model.language_model."):]
    if hf_key.startswith("model."):
        return hf_key[len("model."):]
    return hf_key


def _cast_tensor(tensor: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor needs a scale-aware dequantization path."""
    quantized_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if dtype is not None
    }
    return tensor.dtype in quantized_dtypes


def _remap_simple_key(hf_key: str, max_layer: int) -> Optional[str]:
    """Map a HuggingFace key to our model layout or return ``None`` to skip."""
    key = _normalize_key(hf_key)
    if key in ("embed.weight", "embed_tokens.weight"):
        return "model.embed_tokens.weight"
    if key in ("head.weight", "lm_head.weight"):
        return "lm_head.weight"
    if key == "norm.weight":
        return "model.norm.weight"
    if key in ("hc_head_base", "hc_head_fn", "hc_head_scale"):
        return f"model.layers.{max_layer}.hc_head.{key}"
    if key.startswith("layers."):
        parts = key.split(".")
        if len(parts) < 3:
            return None
        try:
            layer_idx = int(parts[1])
        except ValueError:
            return None
        if layer_idx > max_layer:
            return None
        suffix = ".".join(parts[2:])
        prefix = f"model.layers.{layer_idx}."
        direct_map = {
            "attn.attn_sink": "self_attn.attn_sink",
            "attn.q_norm.weight": "self_attn.q_norm.weight",
            "attn.kv_norm.weight": "self_attn.kv_norm.weight",
            "attn.wq_a.weight": "self_attn.q_lora.weight",
            "attn.wq_b.weight": "self_attn.q_proj.weight",
            "attn.wkv.weight": "self_attn.wkv.weight",
            "attn.wo_a.weight": "self_attn.wo_a.weight",
            "attn.wo_b.weight": "self_attn.wo_b.weight",
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "ffn.gate.weight": "moe.router.gate.weight",
            "ffn.gate.bias": "moe.expert_bias",
            "ffn.shared_experts.w1.weight": "moe.shared_experts.w1.weight",
            "ffn.shared_experts.w2.weight": "moe.shared_experts.w2.weight",
            "ffn.shared_experts.w3.weight": "moe.shared_experts.w3.weight",
            "hc_attn_base": "hc_attn_base",
            "hc_attn_fn": "hc_attn_fn",
            "hc_attn_scale": "hc_attn_scale",
            "hc_ffn_base": "hc_ffn_base",
            "hc_ffn_fn": "hc_ffn_fn",
            "hc_ffn_scale": "hc_ffn_scale",
        }
        if suffix.startswith("attn.compressor.") or suffix.startswith("attn.indexer."):
            return None
        if suffix in direct_map:
            return prefix + direct_map[suffix]
        if suffix.startswith("ffn.experts."):
            return None
    return None


def _skip_expert_weights(model_config) -> bool:
    """Return ``True`` when the checkpoint experts are quantized and should be skipped."""
    expert_dtype = str(getattr(model_config, "expert_dtype", "")).lower()
    return expert_dtype in {"fp4", "int4", "nf4"}


def load_hf_deepseek_v4pro_state_dict(
    weights_path: str,
    num_hidden_layers: int,
    num_experts: int,
    moe_intermediate_size: int,
    dtype: Optional[torch.dtype] = None,
    model_config=None,
) -> Dict[str, torch.Tensor]:
    """Load a DeepSeek-V4Pro HF checkpoint into our module namespace.

    The loader accepts either a standard HF ``model.safetensors.index.json``
    directory or a directory that only contains a few ``*.safetensors`` shards.
    The latter is convenient for the locally cached DeepSeek-V4Pro snapshot.
    """
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    shard_to_keys = _resolve_weights_files(weights_path)
    max_layer = num_hidden_layers - 1
    skip_experts = bool(model_config is not None and _skip_expert_weights(model_config))

    hyper_sd: Dict[str, torch.Tensor] = {}
    collected_experts: dict[tuple[int, int, str], torch.Tensor] = {}
    skipped = 0

    for shard_name in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(weights_path, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for hf_key in shard_to_keys[shard_name]:
                key = _normalize_key(hf_key)
                expert_match = _EXPERT_RE.match(key)
                if expert_match:
                    if skip_experts:
                        skipped += 1
                        continue
                    layer_idx = int(expert_match.group("layer"))
                    if layer_idx > max_layer:
                        skipped += 1
                        continue
                    expert_idx = int(expert_match.group("expert"))
                    if expert_idx >= num_experts:
                        skipped += 1
                        continue
                    kind = expert_match.group("kind")
                    collected_experts[(layer_idx, expert_idx, kind)] = _cast_tensor(
                        handle.get_tensor(hf_key),
                        dtype,
                    )
                    continue

                mapped = _remap_simple_key(hf_key, max_layer)
                if mapped is None:
                    skipped += 1
                    continue
                tensor = handle.get_tensor(hf_key)
                if _is_quantized_tensor(tensor):
                    skipped += 1
                    continue
                tensor = _cast_tensor(tensor, dtype)
                hyper_sd[mapped] = tensor

    if not skip_experts and collected_experts:
        for layer_idx in range(num_hidden_layers):
            layer_experts = {
                expert_idx: {
                    kind: collected_experts[(layer_idx, expert_idx, kind)]
                    for kind in ("w1", "w2", "w3")
                    if (layer_idx, expert_idx, kind) in collected_experts
                }
                for expert_idx in range(num_experts)
            }
            if any(len(kind_map) != 3 for kind_map in layer_experts.values()):
                continue
            stacked_w1 = []
            stacked_w2 = []
            stacked_w3 = []
            for expert_idx in range(num_experts):
                stacked_w1.append(layer_experts[expert_idx]["w1"])
                stacked_w2.append(layer_experts[expert_idx]["w2"])
                stacked_w3.append(layer_experts[expert_idx]["w3"])
            hyper_sd[f"model.layers.{layer_idx}.moe.experts.w1"] = torch.stack(stacked_w1, dim=0)
            hyper_sd[f"model.layers.{layer_idx}.moe.experts.w2"] = torch.stack(stacked_w2, dim=0)
            hyper_sd[f"model.layers.{layer_idx}.moe.experts.w3"] = torch.stack(stacked_w3, dim=0)

    if "lm_head.weight" not in hyper_sd and "model.embed_tokens.weight" in hyper_sd:
        hyper_sd["lm_head.weight"] = hyper_sd["model.embed_tokens.weight"].clone()

    logger.info(
        "DeepSeek-V4Pro HF load ready: %d tensors (%d skipped) from %s",
        len(hyper_sd),
        skipped,
        weights_path,
    )
    return hyper_sd


__all__ = ["load_hf_deepseek_v4pro_state_dict"]

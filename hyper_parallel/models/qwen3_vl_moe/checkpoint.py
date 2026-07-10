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
"""HF Qwen3-VL-MoE checkpoint loader for hyper native models."""
from __future__ import annotations

__all__ = ["load_hf_qwen3_vl_moe_state_dict"]

import json
import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def _remap_key(
    hf_key: str,
    max_layer: int,
    include_visual: bool = False,
    max_visual_layer: int | None = None,
) -> str | None:
    """Map HF composite-model keys to hyper model keys."""
    if hf_key.startswith("model.visual."):
        if not include_visual:
            return None
        tail = hf_key[len("model.visual."):]
        if tail.startswith("blocks.") and max_visual_layer is not None:
            try:
                layer_idx = int(tail.split(".")[1])
            except (IndexError, ValueError):
                return None
            if layer_idx > max_visual_layer:
                return None
        return hf_key
    if hf_key == "lm_head.weight":
        return hf_key

    prefix = "model.language_model."
    if not hf_key.startswith(prefix):
        return None
    tail = hf_key[len(prefix):]
    if tail.startswith("layers."):
        try:
            layer_idx = int(tail.split(".")[1])
        except (IndexError, ValueError):
            return None
        if layer_idx > max_layer:
            return None
    return hf_key if include_visual else tail


def load_hf_qwen3_vl_moe_state_dict(
    weights_path: str,
    num_hidden_layers: int,
    include_visual: bool = False,
    vision_depth: int | None = None,
    dtype: torch.dtype | None = None,
) -> Dict[str, torch.Tensor]:
    """Load a Qwen3-VL-MoE safetensors dir into hyper key space."""
    # pylint: disable=C0415
    from safetensors import safe_open

    idx_path = os.path.join(weights_path, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_keys: Dict[str, list] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    logger.info(
        "Loading Qwen3-VL-MoE safetensors from %s (%d keys across %d shards), "
        "num_hidden_layers=%d include_visual=%s vision_depth=%s",
        weights_path, len(weight_map), len(shard_to_keys), num_hidden_layers,
        include_visual, vision_depth,
    )

    max_layer = num_hidden_layers - 1
    max_visual_layer = vision_depth - 1 if vision_depth is not None else None
    hyper_sd: Dict[str, torch.Tensor] = {}
    skipped = 0
    for shard in sorted(shard_to_keys):
        with safe_open(os.path.join(weights_path, shard), framework="pt", device="cpu") as f:
            for hf_key in shard_to_keys[shard]:
                mapped = _remap_key(
                    hf_key,
                    max_layer,
                    include_visual=include_visual,
                    max_visual_layer=max_visual_layer,
                )
                if mapped is None:
                    skipped += 1
                    continue
                tensor = f.get_tensor(hf_key)
                if mapped.endswith((
                    ".mlp.experts.gate_up_proj",
                    ".mlp.experts.down_proj",
                )):
                    tensor = tensor.transpose(1, 2).contiguous()
                if dtype is not None and tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                hyper_sd[mapped] = tensor

    logger.info(
        "HF Qwen3-VL-MoE -> hyper state_dict ready: %d keys (%d skipped)",
        len(hyper_sd), skipped,
    )
    return hyper_sd

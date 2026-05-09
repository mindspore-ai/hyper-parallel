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
"""HuggingFace Qwen3.5-MoE → hyper state_dict converter.

Reads the safetensors checkpoint of ``Qwen/Qwen3.5-35B-A3B`` and returns
a state_dict keyed by hyper module paths for
:class:`Qwen3_5MoeForCausalLM` (text-only).

Key remapping:

    model.language_model.embed_tokens.weight  → embed_tokens.weight
    model.language_model.layers.{i}.*         → layers.{i}.*
    model.language_model.norm.weight          → norm.weight
    lm_head.weight                            → lm_head.weight
    model.visual.*                            → (silently dropped)

MoE experts are stored in HF as packed 3-D tensors and split per-expert
to match hyper's ``nn.ModuleList`` layout::

    HF: model.layers.{i}.mlp.experts.gate_up_proj   (E, 2I, H)
        model.layers.{i}.mlp.experts.down_proj      (E, H, I)
    →   layers.{i}.mlp.experts.{e}.gate_proj.weight (I, H)
        layers.{i}.mlp.experts.{e}.up_proj.weight   (I, H)
        layers.{i}.mlp.experts.{e}.down_proj.weight (H, I)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

def _remap_simple_key(hf_key: str, max_layer: int) -> str | None:
    """Return hyper key for simple HF keys, or None to skip.

    ``max_layer`` is ``num_hidden_layers - 1``: any layer index beyond it
    is silently dropped (so hyper can use a smaller model than the
    checkpoint without erroring).
    """
    # vision tower — text-only model, drop.
    if hf_key.startswith("model.visual."):
        return None

    # lm_head sits at root in HF, same in hyper.
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # model.language_model.{embed_tokens|norm|layers.X}.* → model.{...}
    prefix = "model.language_model."
    if hf_key.startswith(prefix):
        tail = hf_key[len(prefix):]
        # layer-index gate
        if tail.startswith("layers."):
            try:
                layer_i = int(tail.split(".")[1])
            except (IndexError, ValueError):
                return None
            if layer_i > max_layer:
                return None
        return f"model.{tail}"

    logger.debug("Unmapped HF key dropped: %s", hf_key)
    return None

def _normalize_packed_experts(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize packed expert tensors to hyper's expected ``(E, 2I, H)`` /
    ``(E, H, I)`` layout, transposing per-expert axes if needed."""
    if gate_up_proj.shape[0] != num_experts:
        raise ValueError(
            f"gate_up_proj first dim {gate_up_proj.shape[0]} != "
            f"num_experts {num_experts}"
        )
    e0 = gate_up_proj[0]
    if e0.shape == (2 * intermediate_size, hidden_size):
        gate_up = gate_up_proj
    elif e0.shape == (hidden_size, 2 * intermediate_size):
        gate_up = gate_up_proj.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"gate_up_proj per-expert shape {tuple(e0.shape)} doesn't match "
            f"(2I={2*intermediate_size}, H={hidden_size}) or transpose"
        )

    d0 = down_proj[0]
    if d0.shape == (hidden_size, intermediate_size):
        down_w = down_proj
    elif d0.shape == (intermediate_size, hidden_size):
        down_w = down_proj.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"down_proj per-expert shape {tuple(d0.shape)} unexpected"
        )
    return gate_up, down_w

def load_hf_qwen3_5_moe_state_dict(
    weights_path: str,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_hidden_layers: int,
    dtype: torch.dtype | None = None,
) -> Dict[str, torch.Tensor]:
    """Load ``Qwen/Qwen3.5-35B-A3B`` safetensors into hyper state_dict.

    Args:
        weights_path: Directory with ``model.safetensors.index.json``.
        num_experts:  ``config.num_experts`` (256 for the 35B-A3B).
        hidden_size:  ``config.hidden_size`` (2048).
        moe_intermediate_size:  ``config.moe_intermediate_size`` (512).
        num_hidden_layers:  Truncate the checkpoint to the first
            ``num_hidden_layers`` text-decoder layers.
        dtype: Optional cast applied to every returned tensor.

    Returns:
        Dict keyed by hyper module paths, values on CPU.
    """
    # pylint: disable=C0415
    from safetensors import safe_open

    idx_path = os.path.join(weights_path, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx["weight_map"]

    # Group keys by shard so we open each file only once.
    shard_to_keys: Dict[str, list] = {}
    for hf_key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(hf_key)

    logger.info(
        "Loading Qwen3.5-MoE safetensors from %s (%d keys across %d shards), "
        "num_hidden_layers=%d",
        weights_path, len(weight_map), len(shard_to_keys), num_hidden_layers,
    )

    hyper_sd: Dict[str, torch.Tensor] = {}
    expert_fused: Dict[Tuple[int, str], torch.Tensor] = {}
    max_layer = num_hidden_layers - 1

    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype) if dtype is not None and t.dtype != dtype else t

    skipped = 0
    for shard in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(weights_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in shard_to_keys[shard]:
                # Detect MoE fused expert tensors first — store aside.
                if (
                    "mlp.experts.gate_up_proj" in hf_key
                    or "mlp.experts.down_proj" in hf_key
                ):
                    parts = hf_key.split(".")
                    try:
                        layers_idx = parts.index("layers")
                        layer_i = int(parts[layers_idx + 1])
                    except (ValueError, IndexError) as exc:
                        raise ValueError(
                            f"Could not parse layer index from {hf_key}"
                        ) from exc
                    if layer_i > max_layer:
                        skipped += 1
                        continue
                    kind = "gate_up" if "gate_up_proj" in hf_key else "down"
                    expert_fused[(layer_i, kind)] = _cast(f.get_tensor(hf_key))
                    continue

                mapped = _remap_simple_key(hf_key, max_layer)
                if mapped is None:
                    skipped += 1
                    continue
                tensor = _cast(f.get_tensor(hf_key))
                hyper_sd[mapped] = tensor

    # Keep packed experts as (E, 2I, H) and (E, H, I) — normalize axes only.
    layer_indices = sorted({li for (li, _) in expert_fused})
    for li in layer_indices:
        gu = expert_fused.get((li, "gate_up"))
        dn = expert_fused.get((li, "down"))
        if gu is None or dn is None:
            raise ValueError(
                f"Layer {li} missing one of (gate_up_proj, down_proj)"
            )
        prefix = f"model.layers.{li}.mlp.experts"
        gu_n, dn_n = _normalize_packed_experts(
            gu, dn,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
        )
        hyper_sd[f"{prefix}.gate_up_proj"] = gu_n
        hyper_sd[f"{prefix}.down_proj"] = dn_n
        del gu, dn
        expert_fused.pop((li, "gate_up"), None)
        expert_fused.pop((li, "down"), None)

    logger.info(
        "HF → hyper state_dict ready: %d keys (%d skipped)",
        len(hyper_sd), skipped,
    )
    return hyper_sd

__all__ = ["load_hf_qwen3_5_moe_state_dict"]

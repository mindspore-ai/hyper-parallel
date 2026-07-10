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
import re
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

def _remap_vl_key(hf_key: str, max_layer: int, vision_depth: int) -> str | None:
    """Identity-style remap for the VL composite (text + vision), or None to skip.

    Unlike the text-only path, the VL composite keeps the ``language_model``
    and ``visual`` prefixes (its ``Qwen3_5MoeVLModel`` owns both), so the
    mapping is identity apart from layer-index gating::

        model.visual.blocks.{j}.*   (j <= vision_depth-1)   → unchanged
        model.visual.{patch_embed|pos_embed|merger}.*       → unchanged
        model.language_model.layers.{i}.* (i <= max_layer)  → unchanged
        model.language_model.{embed_tokens|norm}.*          → unchanged
        lm_head.weight                                      → unchanged
    """
    if hf_key == "lm_head.weight":
        return hf_key

    if hf_key.startswith("model.visual."):
        tail = hf_key[len("model.visual."):]
        if tail.startswith("blocks."):
            try:
                block_j = int(tail.split(".")[1])
            except (IndexError, ValueError):
                return None
            if block_j > vision_depth - 1:
                return None
        return hf_key

    if hf_key.startswith("model.language_model."):
        tail = hf_key[len("model.language_model."):]
        if tail.startswith("layers."):
            try:
                layer_i = int(tail.split(".")[1])
            except (IndexError, ValueError):
                return None
            if layer_i > max_layer:
                return None
        return hf_key

    logger.debug("Unmapped HF key dropped (VL): %s", hf_key)
    return None


def _remap_mtp_key(hf_key: str) -> str | None:
    """Identity-map non-expert MTP keys supported by the composite model."""
    direct_keys = {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    }
    if hf_key in direct_keys:
        return hf_key
    if hf_key.startswith("mtp.layers.0.self_attn."):
        return hf_key
    if hf_key.startswith("mtp.layers.0.mlp.shared_expert."):
        return hf_key
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


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Cast a checkpoint tensor when a target dtype is configured."""
    return tensor.to(dtype) if dtype is not None and tensor.dtype != dtype else tensor


def _group_weight_map_by_shard(weight_map: Dict[str, str]) -> Dict[str, list]:
    """Group checkpoint keys by safetensors shard name."""
    shard_to_keys: Dict[str, list] = {}
    for hf_key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(hf_key)
    return shard_to_keys


def _parse_layer_index(hf_key: str) -> int:
    """Parse the decoder layer index from a HuggingFace checkpoint key."""
    parts = hf_key.split(".")
    try:
        layers_idx = parts.index("layers")
        return int(parts[layers_idx + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not parse layer index from {hf_key}") from exc


def _collect_vl_mtp_key(hf_key: str, reader, dtype, num_experts: int, hyper_sd, mtp_experts) -> bool:
    """Collect one optional MTP key, returning whether it was handled."""
    match = re.fullmatch(
        r"mtp\.layers\.0\.mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.weight",
        hf_key,
    )
    if match:
        expert_idx = int(match.group(1))
        if expert_idx >= num_experts:
            raise ValueError(f"MTP expert index {expert_idx} >= num_experts {num_experts}")
        mtp_experts[(expert_idx, match.group(2))] = _cast_tensor(reader.get_tensor(hf_key), dtype)
        return True
    mapped_mtp = _remap_mtp_key(hf_key)
    if mapped_mtp is not None:
        hyper_sd[mapped_mtp] = _cast_tensor(reader.get_tensor(hf_key), dtype)
        return True
    return False


def _collect_vl_expert_key(hf_key: str, reader, dtype, max_layer: int, expert_fused) -> tuple[bool, int]:
    """Collect one packed VL expert tensor, returning ``(handled, skipped)``."""
    if "mlp.experts.gate_up_proj" not in hf_key and "mlp.experts.down_proj" not in hf_key:
        return False, 0
    layer_i = _parse_layer_index(hf_key)
    if layer_i > max_layer:
        return True, 1
    kind = "gate_up" if "gate_up_proj" in hf_key else "down"
    expert_fused[(layer_i, kind)] = _cast_tensor(reader.get_tensor(hf_key), dtype)
    return True, 0


def _finalize_vl_packed_experts(
    hyper_sd,
    expert_fused,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> None:
    """Normalize and store collected VL packed experts."""
    layer_indices = sorted({layer_i for (layer_i, _) in expert_fused})
    for layer_i in layer_indices:
        gate_up = expert_fused.get((layer_i, "gate_up"))
        down = expert_fused.get((layer_i, "down"))
        if gate_up is None or down is None:
            raise ValueError(f"Layer {layer_i} missing one of (gate_up_proj, down_proj)")
        prefix = f"model.language_model.layers.{layer_i}.mlp.experts"
        gate_up, down = _normalize_packed_experts(
            gate_up,
            down,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
        )
        hyper_sd[f"{prefix}.gate_up_proj"] = gate_up
        hyper_sd[f"{prefix}.down_proj"] = down


def _finalize_vl_mtp_experts(
    hyper_sd,
    mtp_experts,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> None:
    """Pack optional MTP expert tensors into hyper's expert layout."""
    gate_up_weights = []
    down_weights = []
    for expert_idx in range(num_experts):
        gate = mtp_experts.get((expert_idx, "gate_proj"))
        up = mtp_experts.get((expert_idx, "up_proj"))
        down = mtp_experts.get((expert_idx, "down_proj"))
        if gate is None or up is None or down is None:
            raise ValueError(
                f"MTP expert {expert_idx} missing one of (gate_proj, up_proj, down_proj)"
            )
        gate_up_weights.append(torch.cat([gate, up], dim=0))
        down_weights.append(down)
    gate_up, down = _normalize_packed_experts(
        torch.stack(gate_up_weights, dim=0),
        torch.stack(down_weights, dim=0),
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=moe_intermediate_size,
    )
    prefix = "mtp.layers.0.mlp.experts"
    hyper_sd[f"{prefix}.gate_up_proj"] = gate_up
    hyper_sd[f"{prefix}.down_proj"] = down


def load_hf_qwen3_5_moe_state_dict(
    weights_path: str,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_hidden_layers: int,
    dtype: torch.dtype | None = None,
) -> Dict[str, torch.Tensor]:
    """Load text-only Qwen3.5-MoE safetensors into hyper state_dict."""
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

def load_hf_qwen3_5_moe_vl_state_dict(
    weights_path: str,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    num_hidden_layers: int,
    vision_depth: int,
    dtype: torch.dtype | None = None,
    include_mtp: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load the multimodal ``Qwen3_5MoeForConditionalGeneration`` checkpoint.

    Keeps the ``model.visual.*`` vision tower and the ``model.language_model.*``
    text backbone under their native prefixes (the VL composite's
    :class:`Qwen3_5MoeVLModel` owns both). Text layers are truncated to
    ``num_hidden_layers`` and vision blocks to ``vision_depth``; packed MoE
    experts are normalized to hyper's ``(E, 2I, H)`` / ``(E, H, I)`` layout
    under ``model.language_model.layers.{i}.mlp.experts.*``.

    Args:
        weights_path: Directory with ``model.safetensors.index.json``.
        num_experts: ``text_config.num_experts`` (256 for 35B-A3B).
        hidden_size: ``text_config.hidden_size`` (2048).
        moe_intermediate_size: ``text_config.moe_intermediate_size`` (512).
        num_hidden_layers: Truncate to the first N text-decoder layers.
        vision_depth: Truncate to the first N vision blocks (27 for the full ViT).
        dtype: Optional cast applied to every returned tensor.
        include_mtp: Whether to load the optional ``mtp.*`` prediction head.

    Returns:
        Dict keyed by VL composite module paths, values on CPU.
    """
    # pylint: disable=C0415
    from safetensors import safe_open

    idx_path = os.path.join(weights_path, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx["weight_map"]
    shard_to_keys = _group_weight_map_by_shard(weight_map)

    logger.info(
        "Loading Qwen3.5-MoE-VL safetensors from %s (%d keys across %d shards), "
        "num_hidden_layers=%d vision_depth=%d",
        weights_path, len(weight_map), len(shard_to_keys),
        num_hidden_layers, vision_depth,
    )

    hyper_sd: Dict[str, torch.Tensor] = {}
    expert_fused: Dict[Tuple[int, str], torch.Tensor] = {}
    mtp_experts: Dict[Tuple[int, str], torch.Tensor] = {}
    max_layer = num_hidden_layers - 1

    skipped = 0
    for shard in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(weights_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for hf_key in shard_to_keys[shard]:
                if include_mtp and _collect_vl_mtp_key(hf_key, f, dtype, num_experts, hyper_sd, mtp_experts):
                    continue
                handled, skipped_delta = _collect_vl_expert_key(hf_key, f, dtype, max_layer, expert_fused)
                skipped += skipped_delta
                if handled:
                    continue

                mapped = _remap_vl_key(hf_key, max_layer, vision_depth)
                if mapped is None:
                    skipped += 1
                    continue
                hyper_sd[mapped] = _cast_tensor(f.get_tensor(hf_key), dtype)

    _finalize_vl_packed_experts(
        hyper_sd,
        expert_fused,
        num_experts,
        hidden_size,
        moe_intermediate_size,
    )

    if include_mtp:
        _finalize_vl_mtp_experts(
            hyper_sd,
            mtp_experts,
            num_experts,
            hidden_size,
            moe_intermediate_size,
        )

    logger.info(
        "HF → hyper VL state_dict ready: %d keys (%d skipped)",
        len(hyper_sd), skipped,
    )
    return hyper_sd


__all__ = [
    "load_hf_qwen3_5_moe_state_dict",
    "load_hf_qwen3_5_moe_vl_state_dict",
]

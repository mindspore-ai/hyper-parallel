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

"""Per-head Q/K/V views for the existing Muon optimizer (K3 report section 2.5)."""

from functools import partial
import json
from pathlib import Path
import re
from typing import Callable, Optional

import torch  # pylint: disable=forbidden-backend-import


def _reshape_heads(param_fqn: str, ns_input: torch.Tensor, *, kda_layers: frozenset[int],
                   kda_head_dim: int, mla_q_head_dim: int, mla_key_head_dim: int,
                   mla_value_head_dim: int, vision_head_dim: Optional[int]) -> list[torch.Tensor]:
    """Expose disjoint logical matrices, with complete storage coverage."""
    match = re.search(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.(.*)$", param_fqn)
    if match is None:
        if vision_head_dim is not None and re.search(r"(?:^|\.)vision_tower\.encoder\.blocks\.\d+\.wqkv\.weight$",
                                                   param_fqn):
            return [ns_input.view(3, -1, vision_head_dim, ns_input.shape[-1]).flatten(0, 1)]
        # Packed experts already have an independent matrix per expert. In the
        # combined FC1 layout, gate and up remain independent logical matrices.
        if param_fqn.endswith(".experts.w13"):
            experts, rows, width = ns_input.shape
            return [ns_input.view(experts * 2, rows // 2, width)]
        return [ns_input]
    layer, projection = int(match[1]) + 1, match[2]
    if layer in kda_layers:
        if projection not in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            return [ns_input]
        return [ns_input.view(-1, kda_head_dim, ns_input.shape[-1])]
    if projection in ("q_proj.weight", "q_b_proj.weight"):
        return [ns_input.view(-1, mla_q_head_dim, ns_input.shape[-1])]
    if projection == "kv_b_proj.weight":
        # Native MLA stores [head, K then V, latent], rather than [all K, all V].
        heads = ns_input.view(-1, mla_key_head_dim + mla_value_head_dim, ns_input.shape[-1])
        return [heads[:, :mla_key_head_dim], heads[:, mla_key_head_dim:]]
    # Compressed KV, output projections and gates do not represent Q/K/V head
    # matrices. Keep their original logical matrix and optimizer scale.
    return [ns_input]


def build_reshape_fn(*, kda_layers: list[int], kda_head_dim: int,
                     mla_q_head_dim: int, mla_key_head_dim: int, mla_value_head_dim: int,
                     vision_head_dim: Optional[int] = None) -> Callable[[str, torch.Tensor], list[torch.Tensor]]:
    """Build K3 per-head views without changing generic Muon or parameter FQNs.

    Args:
        kda_layers: One-based KDA layer indices from linear_attn_config.
        kda_head_dim: KDA Q/K/V channels per head.
        mla_q_head_dim: MLA query channels, including any configured positional channels.
        mla_key_head_dim: Expanded MLA non-positional key channels per head.
        mla_value_head_dim: Expanded MLA value channels per head.
        vision_head_dim: Optional MoonViT Q/K/V channels per head. None retains
            whole-matrix Muon in the vision tower.

    Returns:
        Callback accepted by Muon's existing reshape_fn option. Every returned
        tensor shares storage with its input, including noncontiguous MLA K/V views.
    """
    widths = (kda_head_dim, mla_q_head_dim, mla_key_head_dim, mla_value_head_dim)
    if vision_head_dim is not None:
        widths += (vision_head_dim,)
    if any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in widths):
        raise ValueError("K3 Muon head dimensions must be positive integers")
    if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 1 for layer in kda_layers):
        raise ValueError("K3 Muon kda_layers must contain one-based layer indices")
    return partial(_reshape_heads, kda_layers=frozenset(kda_layers), kda_head_dim=kda_head_dim,
                   mla_q_head_dim=mla_q_head_dim, mla_key_head_dim=mla_key_head_dim,
                   mla_value_head_dim=mla_value_head_dim, vision_head_dim=vision_head_dim)


def build_reshape_fn_from_config(config_path: str, vision_per_head: bool = False) -> Callable:
    """Read head geometry from the same local text or multimodal model configuration.

    Args:
        config_path: Local K3 training JSON used by the model builder.
        vision_per_head: Also split MoonViT's Q/K/V matrices by head.

    Returns:
        The existing Muon reshape callback, with the exact configured KDA schedule.
    """
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    text = configuration.get("text_config", configuration)
    linear = text["linear_attn_config"]
    vision_head_dim = None
    if vision_per_head:
        vision = configuration["vision_config"]
        vision_head_dim = vision["qkv_hidden_size"] // vision["vt_num_attention_heads"]
    return build_reshape_fn(
        kda_layers=linear["kda_layers"], kda_head_dim=linear["head_dim"],
        mla_q_head_dim=text["qk_nope_head_dim"] + text["qk_rope_head_dim"],
        mla_key_head_dim=text["qk_nope_head_dim"], mla_value_head_dim=text["v_head_dim"],
        vision_head_dim=vision_head_dim,
    )

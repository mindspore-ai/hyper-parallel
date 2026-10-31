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
"""Declare K3 replay scopes over explicit, immutable residual history."""

from dataclasses import replace
from typing import Any

from hyper_parallel.models.adapter_spec import RecomputePolicy


def _build_policy(layer_pattern: str, model: Any = None) -> RecomputePolicy:
    """Keep selected expensive branches resident within the decoder region."""
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    scope = getattr(config, "recompute_scope", "layer")
    if scope not in ("layer", "ffn", "attention", "attn_res"):
        raise ValueError("K3 recompute_scope must be layer, ffn, attention, or attn_res")
    resident = {
        "layer": (),
        "ffn": ("self_attn",),
        "attention": ("mlp", "block_sparse_moe"),
        "attn_res": ("self_attn", "mlp", "block_sparse_moe"),
    }[scope]
    return RecomputePolicy(
        region_patterns=(layer_pattern,),
        swap_input_keep_indices=(1,),  # attention_mask is reused by every decoder layer.
        exclude_patterns=tuple(f"{layer_pattern}.{name}" for name in resident),
    )


def build_recompute_policy(model: Any = None) -> RecomputePolicy:
    """Declare multimodal decoder and optional vision recomputation regions.

    ``layer`` replays each complete decoder; ``ffn`` retains attention's original
    graph, and ``attention`` retains the feed-forward graph. Partial scopes still
    replay AttnRes and norms; ``attn_res`` retains both expensive branches.
    Immutable history and QB guards preserve replay
    semantics. Branch exclusions require eager checkpointing.

    Args:
        model: Model whose text_config.recompute_scope selects the decoder scope.
            vision_config.recompute adds vision blocks before decoder regions.
            None selects complete decoder layers only.

    Returns:
        Model-owned region and exclusion patterns for framework layer selection.

    Raises:
        ValueError: recompute_scope is not layer, ffn, attention or attn_res.
    """
    policy = _build_policy("language_model.model.layers.*", model)
    vision = getattr(getattr(model, "config", None), "vision_config", None)
    if getattr(vision, "recompute", False):
        # Vision blocks reuse angles (index 2); index 1 contains immutable segment lengths.
        policy = replace(policy, region_patterns=("vision_tower.encoder.blocks.*",) + policy.region_patterns,
                         swap_input_keep_indices=(1, 2))
    return policy


def build_text_recompute_policy(model: Any = None) -> RecomputePolicy:
    """Declare decoder recomputation regions without changing layer indices.

    Args:
        model: LLM whose config.recompute_scope selects layer, ffn, attention or attn_res.
            None retains the complete-layer declaration.

    Returns:
        Model-owned region and exclusion patterns for framework layer selection.

    Raises:
        ValueError: recompute_scope is not layer, ffn, attention or attn_res.
    """
    return _build_policy("model.layers.*", model)


__all__ = ["build_recompute_policy", "build_text_recompute_policy"]

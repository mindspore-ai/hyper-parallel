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
"""Build and register DeepSeek-V4.1 checkpoint key mappings."""

from __future__ import annotations

from typing import Any


# Keep decoder renames scoped to text layers: vision and Engram retain native names.
_V41_RENAMES = (
    ('^embed\\.weight$', 'embed_tokens.weight'),
    ('^head\\.weight$', 'lm_head.weight'),
    ('^norm\\.weight$', 'norm.weight'),
    ('^hc_head_fn$', 'hc_head.hc_fn'),
    ('^hc_head_base$', 'hc_head.hc_base'),
    ('^hc_head_scale$', 'hc_head.hc_scale'),
    ('^layers\\.(\\d+)\\.attn_norm\\.', 'layers.\\1.input_layernorm.'),
    ('^layers\\.(\\d+)\\.ffn_norm\\.', 'layers.\\1.post_attention_layernorm.'),
    ('^layers\\.(\\d+)\\.hc_attn_fn$', 'layers.\\1.attn_hc.fn'),
    ('^layers\\.(\\d+)\\.hc_attn_base$', 'layers.\\1.attn_hc.base'),
    ('^layers\\.(\\d+)\\.hc_attn_scale$', 'layers.\\1.attn_hc.scale'),
    ('^layers\\.(\\d+)\\.hc_ffn_fn$', 'layers.\\1.ffn_hc.fn'),
    ('^layers\\.(\\d+)\\.hc_ffn_base$', 'layers.\\1.ffn_hc.base'),
    ('^layers\\.(\\d+)\\.hc_ffn_scale$', 'layers.\\1.ffn_hc.scale'),
    ('^layers\\.(\\d+)\\.attn\\.', 'layers.\\1.self_attn.'),
    ('^layers\\.(\\d+)\\.ffn\\.', 'layers.\\1.mlp.'),
    ('^layers\\.(\\d+)\\.self_attn\\.attn_sink$', 'layers.\\1.self_attn.sinks'),
    ('^layers\\.(\\d+)\\.self_attn\\.indexer\\.wq_b\\.', 'layers.\\1.self_attn.indexer.q_b_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wq_a\\.', 'layers.\\1.self_attn.q_a_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wq_b\\.', 'layers.\\1.self_attn.q_b_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wkv\\.', 'layers.\\1.self_attn.kv_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wo_a\\.', 'layers.\\1.self_attn.o_a_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.wo_b\\.', 'layers.\\1.self_attn.o_b_proj.'),
    ('^layers\\.(\\d+)\\.self_attn\\.q_norm\\.', 'layers.\\1.self_attn.q_a_norm.'),
    ('^layers\\.(\\d+)\\.mlp\\.shared_experts\\.w1\\.', 'layers.\\1.mlp.shared_experts.gate_proj.'),
    ('^layers\\.(\\d+)\\.mlp\\.shared_experts\\.w2\\.', 'layers.\\1.mlp.shared_experts.down_proj.'),
    ('^layers\\.(\\d+)\\.mlp\\.shared_experts\\.w3\\.', 'layers.\\1.mlp.shared_experts.up_proj.'),
)


def build_deepseek_v41_checkpoint_mapping() -> list[Any]:
    """Build model-owned V4.1 names using Transformers conversion operators.

    Upstream V4 rules may rename arbitrary ``.attn.``, ``.norm.`` and router
    biases. V4.1 retains those native names in vision, compressors and gates,
    so its mapping must not depend on upstream V4 pattern spelling or scope.

    Returns:
        Fresh name transforms and expert merge operations for the native model.
    """
    try:
        # Family registration is lazy and remains usable without Transformers installed.
        from transformers.core_model_loading import (  # pylint: disable=import-outside-toplevel
            Concatenate, MergeModulelist, WeightConverter, WeightRenaming,
        )
    except ImportError:
        return []

    mapping = [WeightRenaming(source, target) for source, target in _V41_RENAMES]
    mapping.extend([
        WeightConverter(
            source_patterns=["mlp.experts.*.w1.weight", "mlp.experts.*.w3.weight"],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns="mlp.experts.*.w2.weight",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
    ])
    return mapping



def register_deepseek_v41_checkpoint_mapping() -> bool:
    """Register mappings for the production V4.1 classes."""
    try:
        from transformers.conversion_mapping import (  # pylint: disable=C0415
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        return False

    mapping = build_deepseek_v41_checkpoint_mapping()
    register_checkpoint_conversion_mapping("deepseek_v41", mapping, overwrite=True)
    register_checkpoint_conversion_mapping(
        "DeepseekV41ForCausalLM",
        mapping,
        overwrite=True,
    )
    # The root mapping already covers this nested base model. Registering an
    # empty class-specific mapping prevents its translated DeepseekV4Config
    # from adding a second, V4-only scoped conversion pipeline.
    register_checkpoint_conversion_mapping(
        "DeepseekV41Model",
        [],
        overwrite=True,
    )
    return True


__all__ = [
    "build_deepseek_v41_checkpoint_mapping",
    "register_deepseek_v41_checkpoint_mapping",
]

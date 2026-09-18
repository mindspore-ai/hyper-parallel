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


_V4_INCOMPATIBLE_SOURCE_PATTERNS = {
    r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.norm\.",
    r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.ape$",
    r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.",
    r"^layers\.(\d+)\.self_attn\.indexer\.",
    r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.weights_proj\.",
    r"^layers\.(\d+)\.self_attn\.compressor\.norm\.",
    r"^layers\.(\d+)\.self_attn\.compressor\.ape$",
    r"^layers\.(\d+)\.self_attn\.(.*?)\.wq_a\.",
    r"^layers\.(\d+)\.self_attn\.(.*?)\.wkv\.",
    r"^layers\.(\d+)\.self_attn\.(.*?)\.wgate\.",
    r"^layers\.(\d+)\.self_attn\.(.*?)\.wo_a\.",
    r"^layers\.(\d+)\.self_attn\.(.*?)\.wo_b\.",
}


def build_deepseek_v41_checkpoint_mapping() -> list[Any]:
    """Build V4.1 transforms while reusing common V4/HF conversions.

    V4.1 shares the outer decoder, mHC, attention-projection, and MoE
    namespaces with the upstream V4 mapping. Its compressor and Indexer do
    not share V4's nested ``compressor.indexer`` structure, so those V4-only
    transforms are removed. V4.1-native ``compressor.{wkv,wgate,norm}`` names
    pass through unchanged; only ``indexer.wq_b`` uses the retained standard
    ``q_b_proj`` mapping.
    """
    try:
        from transformers.conversion_mapping import (  # pylint: disable=C0415
            get_checkpoint_conversion_mapping,
        )
    except ImportError:
        return []

    v4_mapping = get_checkpoint_conversion_mapping("deepseek_v4") or []
    return [
        transform
        for transform in v4_mapping
        if not set(transform.source_patterns).intersection(
            _V4_INCOMPATIBLE_SOURCE_PATTERNS
        )
    ]


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

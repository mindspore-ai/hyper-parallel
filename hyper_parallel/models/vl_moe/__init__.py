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
"""VL-MoE model registration."""

__all__ = [
    "VLConfig",
    "VLTextConfig",
    "VLVisionConfig",
    "ForCausalLM",
    "VL",
]

import json
import os

from hyper_parallel.models.vl_moe.model import (
    VLConfig,
    VLTextConfig,
    VLVisionConfig,
    ForCausalLM,
    VL,
)
from hyper_parallel.models.vl_moe.parallelize import (
    parallelize_vl_moe,
    pipeline_vl_moe_for_trainer,
)
from hyper_parallel.models.vl_moe.state_dict import (
    VLStateDictAdapter,
)
from hyper_parallel.models.spec import ModelSpec, register_spec

# ---------------------------------------------------------------------------
# Default config values for text model
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "vocab_size": 153600,
    "hidden_size": 2560,
    "intermediate_size": 2048,
    "num_hidden_layers": 4,
    "num_attention_heads": 48,
    "num_key_value_heads": 48,
    "head_dim": 128,
    "max_position_embeddings": 524288,
    "rms_norm_eps": 1e-05,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "rope_theta": 6400000.0,
    "rope_interleaved": False,
    # MLA
    "use_mla": True,
    "kv_lora_rank": 512,
    "q_lora_rank": 1024,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    # MoE
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 1024,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    # MHC
    "use_mhc": True,
    "mhc_num_stream": 4,
    # MTP
    "num_nextn_predict_layers": 2,
}

_UNIVERSAL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)

# ---------------------------------------------------------------------------
# Default config values for vision model
# ---------------------------------------------------------------------------
_VISION_DEFAULTS = {
    "depth": 24,
    "num_heads": 16,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "hidden_act": "gelu",
    "patch_size": 14,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "spatial_merge_size": 2,
    "out_hidden_size": 3584,
    "tokens_per_second": 2,
    "window_size": 112,
    "use_gatedmerger": True,
    "position_embedding_type": "3d_rope",
    "mrope_section": "8,12,12",
    "rotary_interleaved": True,
}


def _load_text_config_defaults(model_cfg) -> dict:
    """Load text config defaults from checkpoint config.json."""
    weights_path = model_cfg.weights_path
    if not weights_path:
        return {}
    cfg_path = os.path.join(weights_path, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    text = dict(raw.get("text_config", {}))
    rope = text.pop("rope_scaling", None) or {}
    if "rope_theta" in rope:
        text["rope_theta"] = rope["rope_theta"]
    if "mrope_section" in rope:
        text["mrope_section"] = rope["mrope_section"]
    return {k: v for k, v in text.items() if k in _DEFAULTS}


def _load_full_config_defaults(model_cfg) -> dict:
    """Load full config defaults (text + vision + composite) from checkpoint."""
    weights_path = model_cfg.weights_path
    if not weights_path:
        return {}
    cfg_path = os.path.join(weights_path, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    text = dict(raw.get("text_config", {}))
    rope = text.pop("rope_scaling", None) or {}
    if "rope_theta" in rope:
        text["rope_theta"] = rope["rope_theta"]
    if "mrope_section" in rope:
        text["mrope_section"] = rope["mrope_section"]
    vision = dict(raw.get("vision_config", {}))
    composite = {
        "image_token_id": raw.get("image_token_id", 148909),
        "video_token_id": raw.get("video_token_id", 148910),
        "vision_start_token_id": raw.get("vision_start_token_id", 148907),
        "vision_end_token_id": raw.get("vision_end_token_id", 148908),
    }
    return {
        "text": {k: v for k, v in text.items() if k in _DEFAULTS},
        "vision": {k: v for k, v in vision.items() if k in _VISION_DEFAULTS},
        "composite": composite,
    }


def _resolve_kwargs(model_cfg) -> dict:
    """Resolve config with checkpoint config between defaults and YAML."""
    kwargs = dict(_DEFAULTS)
    kwargs.update(_load_text_config_defaults(model_cfg))
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            kwargs[field] = val
    extra = model_cfg.config_overrides
    if isinstance(extra, dict):
        # Extract nested text_config overrides and flatten them into kwargs
        # so that YAML fields like num_hidden_layers, n_routed_experts etc.
        # actually take effect.
        text_extra = extra.pop("text_config", None)
        if isinstance(text_extra, dict):
            kwargs.update(text_extra)
        # Also extract rope_scaling from text_config if present and promote it
        # (VLTextConfig expects rope_scaling as a top-level attr)
        if isinstance(text_extra, dict) and "rope_scaling" in text_extra:
            kwargs["rope_scaling"] = text_extra["rope_scaling"]
        kwargs.update(extra)
        kwargs.pop("vl", None)
        kwargs.pop("vision_config", None)
        kwargs.pop("architectures", None)
        kwargs.pop("model_type", None)
        kwargs.pop("transformers_version", None)
    return kwargs


def _build_vl(cfg) -> VL:
    """Build VL model (internal)."""
    model_cfg = cfg.model
    raw = _load_full_config_defaults(model_cfg)
    text_kwargs = dict(_DEFAULTS)
    vision_kwargs = dict(_VISION_DEFAULTS)
    composite_kwargs = {}
    if raw:
        text_kwargs.update(raw["text"])
        vision_kwargs.update(raw["vision"])
        composite_kwargs.update(raw["composite"])
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            text_kwargs[field] = val
    extra = model_cfg.config_overrides
    if isinstance(extra, dict):
        text_extra = extra.get("text_config", {})
        vision_extra = extra.get("vision_config", {})
        text_kwargs.update(text_extra)
        vision_kwargs.update(vision_extra)
        for key in (
            "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id",
        ):
            if key in extra:
                composite_kwargs[key] = extra[key]
    return VL(
        VLConfig(
            text_config=VLTextConfig(**text_kwargs),
            vision_config=VLVisionConfig(**vision_kwargs),
            **composite_kwargs,
        )
    )


def _build(cfg):
    extra = cfg.model.config_overrides
    is_vl = isinstance(extra, dict) and (
        extra.get("vl", False)
        or extra.get("architectures") == ["VL"]
        or (isinstance(extra.get("architectures"), list) and "VL" in extra.get("architectures", []))
    )
    if is_vl:
        return _build_vl(cfg)
    return ForCausalLM(_build_config(cfg))


def _build_config(cfg) -> VLTextConfig:
    """Construct only the text config from a HyperTrainerConfig.

    Returns the text config object without building the full model.
    Used by the SAPP-ND Hyper YAML parser for memory estimation.
    """
    return VLTextConfig(**_resolve_kwargs(cfg.model))


register_spec(
    "vl_moe",
    ModelSpec(
        name="vl_moe",
        build_model_fn=_build,
        parallelize_fn=parallelize_vl_moe,
        pipelining_fn=pipeline_vl_moe_for_trainer,
        state_dict_adapter=VLStateDictAdapter,
    ),
)
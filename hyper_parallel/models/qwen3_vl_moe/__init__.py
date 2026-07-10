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
"""Qwen3-VL-MoE model registration."""

__all__ = [
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeTextConfig",
    "Qwen3VLMoeVisionConfig",
    "Qwen3VLMoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
]

import json
import os

from hyper_parallel.models.qwen3_vl_moe.model import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeForCausalLM,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from hyper_parallel.models.qwen3_vl_moe.parallelize import (
    parallelize_qwen3_vl_moe,
    pipeline_qwen3_vl_moe_for_trainer,
)
from hyper_parallel.models.qwen3_vl_moe.state_dict import (
    Qwen3VLMoeStateDictAdapter,
)
from hyper_parallel.models.qwen3_5_moe.parallelize import (
    qwen3_5_moe_tp_load_transforms,
)
from hyper_parallel.models.spec import ModelSpec, register_spec

_DEFAULTS = {
    "vocab_size": 151936,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 4,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "rope_theta": 5_000_000.0,
    "mrope_section": [24, 20, 20],
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
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

_VISION_DEFAULTS = {
    "depth": 27,
    "hidden_size": 1152,
    "hidden_act": "gelu_pytorch_tanh",
    "intermediate_size": 4304,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 2048,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [8, 16, 24],
}


def _load_text_config_defaults(model_cfg) -> dict:
    """Load text config defaults (internal)."""
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
    """Load full config defaults (internal)."""
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
        "image_token_id": raw.get("image_token_id", 151655),
        "video_token_id": raw.get("video_token_id", 151656),
        "vision_start_token_id": raw.get("vision_start_token_id", 151652),
        "vision_end_token_id": raw.get("vision_end_token_id", 151653),
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
        kwargs.update(extra)
        kwargs.pop("vl", None)
    return kwargs


def _build_vl(cfg) -> Qwen3VLMoeForConditionalGeneration:
    """Build vl (internal)."""
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
    return Qwen3VLMoeForConditionalGeneration(
        Qwen3VLMoeConfig(
            text_config=Qwen3VLMoeTextConfig(**text_kwargs),
            vision_config=Qwen3VLMoeVisionConfig(**vision_kwargs),
            **composite_kwargs,
        )
    )


def _build(cfg):
    extra = cfg.model.config_overrides
    if isinstance(extra, dict) and extra.get("vl", False):
        return _build_vl(cfg)
    return Qwen3VLMoeForCausalLM(_build_config(cfg))


def _build_config(cfg) -> Qwen3VLMoeTextConfig:
    """Construct only the ``Qwen3VLMoeTextConfig`` from a ``HyperTrainerConfig``.

    Returns the text config object without building the full model.
    Used by the SAPP-ND Hyper YAML parser for memory estimation.
    The vision config is not needed since memory estimation only uses
    text-model hyperparameters.
    """
    return Qwen3VLMoeTextConfig(**_resolve_kwargs(cfg.model))


register_spec(
    "qwen3_vl_moe",
    ModelSpec(
        name="qwen3_vl_moe",
        build_model_fn=_build,
        parallelize_fn=parallelize_qwen3_vl_moe,
        pipelining_fn=pipeline_qwen3_vl_moe_for_trainer,
        state_dict_adapter=Qwen3VLMoeStateDictAdapter,
        tp_load_transform_fn=qwen3_5_moe_tp_load_transforms,
    ),
)

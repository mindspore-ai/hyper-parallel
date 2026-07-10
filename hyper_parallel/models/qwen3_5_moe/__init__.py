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
"""Qwen3.5-MoE (35B-A3B) hyper model registration.

Universal transformer fields (``num_hidden_layers``, ``hidden_size`` ...)
come from ``cfg.model.*``. Qwen-specific knobs — including the MoE expert
geometry, mRoPE section split, linear-attention dims and ``layer_types``
— go through ``cfg.model.config_overrides``. Anything not provided falls
through to the Qwen3.5-35B-A3B defaults below.
"""
import json
import os
from dataclasses import fields as _dataclass_fields

from hyper_parallel.models.qwen3_5_moe.model import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
)
from hyper_parallel.models.qwen3_5_moe.model_vl import (
    Qwen3_5MoeVLConfig,
    Qwen3_5MoeVLForConditionalGeneration,
)
from hyper_parallel.models.qwen3_5_moe.parallelize import (
    parallelize_qwen3_5_moe,
    pipeline_qwen3_5_moe_for_trainer,
    qwen3_5_moe_tp_load_transforms,
)
from hyper_parallel.models.qwen3_5_moe.state_dict import (
    Qwen3_5MoeStateDictAdapter,
)
from hyper_parallel.models.qwen3_vl_vision import Qwen3VLMoeVisionConfig
from hyper_parallel.models.spec import ModelSpec, register_spec

# Default Qwen3.5-35B-A3B architecture values taken from
# https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
# (text_config). Used as the fallback for every field not explicitly set
# via ``cfg.model.*`` (universal fields) or ``cfg.model.config_overrides``
# (Qwen-specific fields).
_DEFAULTS = {
    "vocab_size": 248320,
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "attn_output_gate": True,
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "full_attention_interval": 4,
    "linear_num_value_heads": 32,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_key_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "mtp_loss_weight": 0.0,
}

# Universal transformer fields exposed on ``ModelConfig`` that map straight
# onto :class:`Qwen3_5MoeConfig`. All other architecture fields go through
# ``model.config_overrides``. (``intermediate_size`` is deliberately absent:
# the MoE config has no such field — experts use ``moe_intermediate_size``.)
_UNIVERSAL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)


def _ckpt_text_kwargs(raw: dict) -> dict:
    """Text-config kwargs extracted from a checkpoint's parsed ``config.json``.

    Handles the composite hub layout (architecture under ``text_config``,
    ``tie_word_embeddings`` at the top level, rope knobs nested under
    ``rope_parameters``) with a flat fallback. Keys are filtered to
    :class:`Qwen3_5MoeConfig` fields so hub-only keys (``hidden_act``,
    ``mtp_num_hidden_layers`` ...) cannot reach the dataclass constructor.
    """
    if not raw:
        return {}
    raw_text = dict(raw.get("text_config", raw))
    rope = raw_text.pop("rope_parameters", None) or {}
    field_names = {f.name for f in _dataclass_fields(Qwen3_5MoeConfig)}
    kwargs = {k: v for k, v in raw_text.items() if k in field_names}
    for key in ("rope_theta", "partial_rotary_factor", "mrope_section"):
        if key in rope:
            kwargs[key] = rope[key]
    if "tie_word_embeddings" in raw:
        kwargs["tie_word_embeddings"] = raw["tie_word_embeddings"]
    return kwargs


def _truncate_layer_types(kwargs: dict) -> None:
    """Normalize ``layer_types`` to ``num_hidden_layers`` after all merges.

    A reduced num_hidden_layers (tests shrink the real 40-layer config) must
    also truncate config.json's layer_types to the first N layers — matching
    the checkpoint loader's max_layer gate — else Qwen3_5MoeConfig.__post_init__
    rejects the length mismatch. Some reduced checkpoints store a 1-layer
    config while still carrying more layer weights; when a caller asks for more
    layers than the config lists, regenerate the standard Qwen3.5 layer pattern.
    """
    n_layers = kwargs.get("num_hidden_layers")
    layer_types = kwargs.get("layer_types")
    if n_layers is None or layer_types is None:
        return
    if len(layer_types) > n_layers:
        kwargs["layer_types"] = layer_types[:n_layers]
    elif len(layer_types) < n_layers:
        kwargs["layer_types"] = None


def _resolve_kwargs(model_cfg) -> dict:
    """Build the full kwargs dict for :class:`Qwen3_5MoeConfig`.

    Precedence (low → high): ``_DEFAULTS`` < checkpoint ``config.json`` <
    universal ``ModelConfig`` fields < ``model.config_overrides`` dict.
    ``config.json`` keeps a sibling-size checkpoint from silently building
    the 35B-A3B default architecture; ``config_overrides`` wins so callers
    can clobber any architecture knob without us having to surface it as a
    typed field on ``ModelConfig``.
    """
    kwargs = dict(_DEFAULTS)
    kwargs.update(
        _ckpt_text_kwargs(
            _read_vl_config_json(model_cfg.weights_path)
        )
    )
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            kwargs[field] = val
    extra = model_cfg.config_overrides
    if isinstance(extra, dict):
        kwargs.update(extra)
    _truncate_layer_types(kwargs)
    return kwargs

# Token-id fields that live at the top level of the multimodal config.json.
_VL_TOKEN_FIELDS = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def _read_vl_config_json(weights_path) -> dict:
    """Load ``config.json`` from a multimodal checkpoint, else return ``{}``."""
    if not weights_path:
        return {}
    cfg_path = os.path.join(weights_path, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_vl_kwargs(model_cfg) -> tuple:
    """Build (text_kwargs, vision_kwargs, token_ids) for the VL composite.

    Precedence (low → high): hp ``_DEFAULTS`` < checkpoint ``config.json`` <
    universal ``ModelConfig`` fields < ``model.config_overrides``.
    ``config.json`` is authoritative for the real checkpoint architecture;
    ``config_overrides`` lets tests shrink it.
    """
    raw = _read_vl_config_json(model_cfg.weights_path)

    text_field_names = {f.name for f in _dataclass_fields(Qwen3_5MoeConfig)}
    vision_field_names = {f.name for f in _dataclass_fields(Qwen3VLMoeVisionConfig)}

    text_kwargs = dict(_DEFAULTS)
    text_kwargs.update(_ckpt_text_kwargs(raw))
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            text_kwargs[field] = val

    # DeepStack disabled by default for the Qwen3.5 ViT (the merged ViT).
    vision_kwargs = {
        "deepstack_visual_indexes": [],
        "_attn_implementation": "eager",
    }
    for key, val in raw.get("vision_config", {}).items():
        if key in vision_field_names:
            vision_kwargs[key] = val

    token_ids = {key: raw[key] for key in _VL_TOKEN_FIELDS if key in raw}

    extra = model_cfg.config_overrides
    if isinstance(extra, dict):
        text_extra = extra.get("text_config", {})
        vision_extra = extra.get("vision_config", {})
        text_kwargs.update(
            {k: v for k, v in extra.items()
             if k in text_field_names and k not in ("vl",)}
        )
        text_kwargs.update({k: v for k, v in text_extra.items() if k in text_field_names})
        vision_kwargs.update({k: v for k, v in vision_extra.items() if k in vision_field_names})
        token_ids.update({k: v for k, v in extra.items() if k in _VL_TOKEN_FIELDS})

    _truncate_layer_types(text_kwargs)

    return text_kwargs, vision_kwargs, token_ids


def _build_vl(cfg) -> Qwen3_5MoeVLForConditionalGeneration:
    """Build the multimodal ``Qwen3_5MoeVLForConditionalGeneration``."""
    text_kwargs, vision_kwargs, token_ids = _resolve_vl_kwargs(cfg.model)
    vl_config = Qwen3_5MoeVLConfig(
        text_config=Qwen3_5MoeConfig(**text_kwargs),
        vision_config=Qwen3VLMoeVisionConfig(**vision_kwargs),
        **token_ids,
    )
    return Qwen3_5MoeVLForConditionalGeneration(vl_config)


def _build(cfg):
    """Translate trainer cfg → model.

    A minimal YAML ``model: {name: qwen3_5_moe, weights_path: ...}`` is
    enough — every other field falls through to the 35B-A3B defaults. Set
    ``config_overrides.vl: true`` to build the multimodal (vision + text)
    ``Qwen3_5MoeVLForConditionalGeneration`` instead of the text-only LM.
    """
    extra = cfg.model.config_overrides
    if isinstance(extra, dict) and extra.get("vl", False):
        return _build_vl(cfg)
    kwargs = _resolve_kwargs(cfg.model)
    if kwargs.get("mtp_loss_weight", 0.0) > 0:
        # Only the VL composite constructs the MTP head; accepting the flag
        # here would silently train without the MTP loss.
        raise ValueError(
            "mtp_loss_weight > 0 requires the VL composite "
            "(config_overrides.vl: true) — the text-only Qwen3_5MoeForCausalLM "
            "has no MTP head."
        )
    return Qwen3_5MoeForCausalLM(Qwen3_5MoeConfig(**kwargs))


def _build_config(cfg) -> Qwen3_5MoeConfig:
    """Construct only the ``Qwen3_5MoeConfig`` from a ``HyperTrainerConfig``.

    Returns the model-specific config object without building the full
    model. Used by the SAPP-ND Hyper YAML parser for memory estimation.
    """
    return Qwen3_5MoeConfig(**_resolve_kwargs(cfg.model))

register_spec(
    "qwen3_5_moe",
    ModelSpec(
        name="qwen3_5_moe",
        build_model_fn=_build,
        parallelize_fn=parallelize_qwen3_5_moe,
        pipelining_fn=pipeline_qwen3_5_moe_for_trainer,
        state_dict_adapter=Qwen3_5MoeStateDictAdapter,
        tp_load_transform_fn=qwen3_5_moe_tp_load_transforms,
    ),
)

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
from hyper_parallel.models.qwen3_5_moe.model import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
)
from hyper_parallel.models.qwen3_5_moe.parallelize import parallelize_qwen3_5_moe
from hyper_parallel.models.qwen3_5_moe.state_dict import (
    Qwen3_5MoeStateDictAdapter,
)
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
}

# Universal transformer fields exposed on ``ModelConfig`` that map straight
# onto :class:`Qwen3_5MoeConfig`. All other architecture fields go through
# ``model.config_overrides``.
_UNIVERSAL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)


def _resolve_kwargs(model_cfg) -> dict:
    """Build the full kwargs dict for :class:`Qwen3_5MoeConfig`.

    Precedence (low → high): ``_DEFAULTS`` < universal ``ModelConfig``
    fields < ``model.config_overrides`` dict. ``config_overrides`` wins so
    callers can clobber any architecture knob without us having to surface
    it as a typed field on ``ModelConfig``.
    """
    kwargs = dict(_DEFAULTS)
    if model_cfg is None:
        return kwargs
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            kwargs[field] = val
    extra = getattr(model_cfg, "config_overrides", None)
    if isinstance(extra, dict):
        kwargs.update(extra)
    return kwargs


def _build(cfg) -> Qwen3_5MoeForCausalLM:
    """Translate trainer cfg → Qwen3_5MoeConfig → model.

    A minimal YAML ``model: {name: qwen3_5_moe, weights_path: ...}`` is
    enough — every other field falls through to the 35B-A3B defaults.
    """
    return Qwen3_5MoeForCausalLM(_build_config(cfg))


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
        state_dict_adapter=Qwen3_5MoeStateDictAdapter,
    ),
)

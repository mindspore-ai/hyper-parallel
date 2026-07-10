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
"""Qwen3.5 (dense) model registration.

Registers the ``qwen3_5`` spec under the canonical HF ``model_type`` so
``tasks.discovery.discover_model_spec("qwen3_5")`` auto-imports this
package and populates the registry. Covers all dense Qwen3.5 variants
(0.8B / 2B / 4B / 9B / 27B Base & Instruct) — pass the arch hyperparams
via YAML ``model.*`` (universal fields) or ``model.config_overrides`` for
Qwen-specific ones; everything else falls through to the 0.8B-Base
defaults baked into :class:`Qwen3_5Config`.
"""
import json
import os
from dataclasses import fields as _dataclass_fields

from hyper_parallel.models.qwen3_5.model import (
    Qwen3_5Config,
    Qwen3_5Decoder,
    Qwen3_5ForCausalLM,
)
from hyper_parallel.models.qwen3_5.parallelize import (
    parallelize_qwen3_5,
    pipeline_qwen3_5_for_trainer,
    qwen3_5_tp_load_transforms,
)
from hyper_parallel.models.qwen3_5.state_dict import (
    Qwen3_5StateDictAdapter,
)
from hyper_parallel.models.spec import ModelSpec, register_spec

# Universal transformer fields exposed on ``ModelConfig`` that map straight
# onto :class:`Qwen3_5Config`. Anything Qwen-specific (linear-attn knobs,
# ``layer_types``, ``attention_bias``, ``mrope_section`` ...) goes through
# ``model.config_overrides`` instead — see :func:`_resolve_overrides`.
_UNIVERSAL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)

def _ckpt_config_kwargs(model_cfg) -> dict:
    """Architecture kwargs read from the checkpoint's ``config.json``.

    Dense hub checkpoints use the composite layout (architecture nested
    under ``text_config``, ``tie_word_embeddings`` at the top level, rope
    knobs under ``rope_parameters``); a flat layout is accepted as
    fallback. Keys are filtered to :class:`Qwen3_5Config` fields so
    hub-only keys (``hidden_act``, ``mtp_num_hidden_layers`` ...) never
    reach the dataclass constructor. Without this layer, any dense sibling
    whose geometry differs from the baked 0.8B-Base defaults would
    silently build the wrong architecture unless every field is spelled
    out in YAML.
    """
    weights_path = model_cfg.weights_path
    if not weights_path:
        return {}
    cfg_path = os.path.join(weights_path, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw_text = dict(raw.get("text_config", raw))
    rope = raw_text.pop("rope_parameters", None) or {}
    field_names = {f.name for f in _dataclass_fields(Qwen3_5Config)}
    kwargs = {k: v for k, v in raw_text.items() if k in field_names}
    for key in ("rope_theta", "partial_rotary_factor", "mrope_section"):
        if key in rope:
            kwargs[key] = rope[key]
    if "tie_word_embeddings" in raw:
        kwargs["tie_word_embeddings"] = raw["tie_word_embeddings"]
    return kwargs


def _resolve_overrides(model_cfg) -> dict:
    """Merge universal ``ModelConfig`` fields with ``config_overrides`` dict.

    ``config_overrides`` wins when the same key appears in both — it is
    the explicit per-model escape hatch for overriding generic defaults
    on a per-config basis.
    """
    overrides: dict = {}
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            overrides[field] = val
    extra = model_cfg.config_overrides
    if isinstance(extra, dict):
        overrides.update(extra)
    return overrides


def _build(cfg) -> Qwen3_5ForCausalLM:
    """Construct the dense Qwen3.5 model from a ``HyperTrainerConfig``.

    Precedence (low → high): dataclass defaults < checkpoint
    ``config.json`` < universal ``ModelConfig`` fields <
    ``model.config_overrides``.
    """
    return Qwen3_5ForCausalLM(_build_config(cfg))


def _build_config(cfg) -> Qwen3_5Config:
    """Construct only the ``Qwen3_5Config`` from a ``HyperTrainerConfig``.

    Returns the model-specific config object without building the full
    model. Used by the SAPP-ND Hyper YAML parser for memory estimation.

    Precedence (low → high): dataclass defaults < checkpoint
    ``config.json`` < universal ``ModelConfig`` fields <
    ``model.config_overrides``.
    """
    model_cfg = cfg.model
    overrides = _ckpt_config_kwargs(model_cfg)
    overrides.update(_resolve_overrides(model_cfg))
    # A reduced num_hidden_layers must also clip config.json's layer_types
    # or Qwen3_5Config.__post_init__ rejects the length mismatch.
    n_layers = overrides.get("num_hidden_layers")
    layer_types = overrides.get("layer_types")
    if n_layers is not None and layer_types is not None and len(layer_types) > n_layers:
        overrides["layer_types"] = layer_types[:n_layers]
    return Qwen3_5Config(**overrides) if overrides else Qwen3_5Config()

register_spec(
    "qwen3_5",
    ModelSpec(
        name="qwen3_5",
        build_model_fn=_build,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_qwen3_5_for_trainer,
        state_dict_adapter=Qwen3_5StateDictAdapter,
        tp_load_transform_fn=qwen3_5_tp_load_transforms,
    ),
)

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5Decoder",
    "Qwen3_5ForCausalLM",
]

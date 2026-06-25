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

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5Decoder",
    "Qwen3_5ForCausalLM",
]

from hyper_parallel.models.qwen3_5.model import (
    Qwen3_5Config,
    Qwen3_5Decoder,
    Qwen3_5ForCausalLM,
)
from hyper_parallel.models.qwen3_5.parallelize import parallelize_qwen3_5
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


def _resolve_overrides(model_cfg) -> dict:
    """Merge universal ``ModelConfig`` fields with ``config_overrides`` dict.

    ``config_overrides`` wins when the same key appears in both — it is
    the explicit per-model escape hatch for overriding generic defaults
    on a per-config basis.
    """
    overrides: dict = {}
    if model_cfg is None:
        return overrides
    for field in _UNIVERSAL_FIELDS:
        val = getattr(model_cfg, field, None)
        if val is not None:
            overrides[field] = val
    extra = getattr(model_cfg, "config_overrides", None)
    if isinstance(extra, dict):
        overrides.update(extra)
    return overrides


def _build(cfg) -> Qwen3_5ForCausalLM:
    """Construct the dense Qwen3.5 model from a ``HyperTrainerConfig``."""
    return Qwen3_5ForCausalLM(_build_config(cfg))


def _build_config(cfg) -> Qwen3_5Config:
    """Construct only the ``Qwen3_5Config`` from a ``HyperTrainerConfig``.

    Returns the model-specific config object without building the full
    model. Used by the SAPP-ND Hyper YAML parser for memory estimation.
    """
    overrides = _resolve_overrides(getattr(cfg, "model", None))
    return Qwen3_5Config(**overrides) if overrides else Qwen3_5Config()


register_spec(
    "qwen3_5",
    ModelSpec(
        name="qwen3_5",
        build_model_fn=_build,
        parallelize_fn=parallelize_qwen3_5,
        state_dict_adapter=Qwen3_5StateDictAdapter,
    ),
)

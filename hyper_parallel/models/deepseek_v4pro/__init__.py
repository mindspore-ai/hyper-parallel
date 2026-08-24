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
"""DeepSeek-V4Pro model registration for HyperParallel."""
from __future__ import annotations

import json
import os
from dataclasses import fields

from hyper_parallel.models.deepseek_v4pro.model import (
    DeepSeekV4ProConfig,
    DeepSeekV4ProForCausalLM,
)
from hyper_parallel.models.deepseek_v4pro.parallelize import parallelize_deepseek_v4pro
from hyper_parallel.models.deepseek_v4pro.state_dict import DeepSeekV4ProStateDictAdapter
from hyper_parallel.models.spec import ModelSpec, register_spec


_UNIVERSAL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)


def _checkpoint_config_kwargs(model_cfg) -> dict:
    """Read compatible architecture fields from a local HF config."""
    source_path = getattr(model_cfg, "weights_path", None) or getattr(model_cfg, "tokenizer_path", None)
    if not source_path:
        return {}
    config_path = os.path.join(source_path, "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    field_names = {item.name for item in fields(DeepSeekV4ProConfig)}
    kwargs = {}
    for key, value in raw.items():
        if key in field_names:
            kwargs[key] = value
    rope_scaling = raw.get("rope_scaling", {}) or {}
    if "factor" in rope_scaling:
        kwargs["rope_scaling_factor"] = rope_scaling["factor"]
    if "original_max_position_embeddings" in rope_scaling:
        kwargs["original_seq_len"] = rope_scaling["original_max_position_embeddings"]
    if "sliding_window" in raw:
        kwargs["window_size"] = raw["sliding_window"]
    if "n_routed_experts" in raw:
        kwargs["num_experts"] = raw["n_routed_experts"]
    if "num_hash_layers" in raw:
        kwargs["num_hash_layers"] = raw["num_hash_layers"]
    return kwargs


def _resolve_overrides(model_cfg) -> dict:
    """Merge checkpoint defaults, universal fields and explicit overrides."""
    overrides = _checkpoint_config_kwargs(model_cfg)
    for field_name in _UNIVERSAL_FIELDS:
        value = getattr(model_cfg, field_name, None)
        if value is not None:
            overrides[field_name] = value
    extra = getattr(model_cfg, "config_overrides", None)
    if isinstance(extra, dict):
        overrides.update(extra)
    # The HF field is called ``compress_ratios`` and may contain one trailing
    # zero for the MTP slot; the first smoke path only builds main layers.
    if "compress_ratios" in overrides and overrides["compress_ratios"] is None:
        overrides.pop("compress_ratios")
    return overrides


def _build_config(cfg) -> DeepSeekV4ProConfig:
    overrides = _resolve_overrides(cfg.model)
    return DeepSeekV4ProConfig(**overrides)


def _build(cfg) -> DeepSeekV4ProForCausalLM:
    return DeepSeekV4ProForCausalLM(_build_config(cfg))


register_spec(
    "deepseek_v4pro",
    ModelSpec(
        name="deepseek_v4pro",
        build_model_fn=_build,
        parallelize_fn=parallelize_deepseek_v4pro,
        state_dict_adapter=DeepSeekV4ProStateDictAdapter,
    ),
)


__all__ = [
    "DeepSeekV4ProConfig",
    "DeepSeekV4ProForCausalLM",
]

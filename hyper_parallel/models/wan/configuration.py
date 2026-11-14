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
"""Load Wan configuration as official Diffusers constructor arguments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _load_json_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _convert_veomni_wan_config(config: dict[str, Any]) -> dict[str, Any]:
    if "dim" not in config:
        return dict(config)

    dim = int(config["dim"])
    num_heads = int(config["num_heads"])
    converted = {
        "patch_size": tuple(config.get("patch_size", (1, 2, 2))),
        "num_attention_heads": num_heads,
        "attention_head_dim": dim // num_heads,
        "in_channels": int(config.get("in_dim", config.get("in_channels", 16))),
        "out_channels": int(config.get("out_dim", config.get("out_channels", 16))),
        "text_dim": int(config.get("text_dim", 4096)),
        "freq_dim": int(config.get("freq_dim", 256)),
        "ffn_dim": int(config["ffn_dim"]),
        "num_layers": int(config["num_layers"]),
        "eps": float(config.get("eps", 1e-6)),
    }
    for optional_key in (
        "cross_attn_norm",
        "qk_norm",
        "rope_max_seq_len",
        "pos_embed_seq_len",
        "image_dim",
        "added_kv_proj_dim",
    ):
        if optional_key in config:
            converted[optional_key] = config[optional_key]

    if _as_bool(config.get("has_image_input", False)):
        converted.setdefault("image_dim", int(config.get("image_dim", 1280)))
        converted.setdefault(
            "added_kv_proj_dim",
            int(config.get("added_kv_proj_dim", dim)),
        )
    return converted


def load_wan_transformer_config(
    config_source: str | Path,
    **overrides: Any,
) -> dict[str, Any]:
    """Convert Wan config formats and apply model-constructor overrides."""
    config = _convert_veomni_wan_config(_load_json_config(config_source))
    config.update(overrides)
    return config


__all__ = ["load_wan_transformer_config"]

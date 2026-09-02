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
"""Wan 2.1 Diffusers configuration wrappers for AutoModels DiT training."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import diffusers
from diffusers import WanTransformer3DModel as _WanTransformer3DModel
from transformers import PretrainedConfig


WAN_INIT_SIGNATURE = inspect.signature(_WanTransformer3DModel.__init__)


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
    """Convert VeOmni's compact Wan JSON keys to Diffusers-style keys."""
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
        "tie_word_embeddings": False,
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
            int(config.get("added_kv_proj_dim", converted["num_attention_heads"] * converted["attention_head_dim"])),
        )
    return converted


class WanTransformer3DTrainingConfig(PretrainedConfig):
    """Transformers-compatible config around Diffusers ``WanTransformer3DModel``."""

    model_type = "WanTransformer3DModel"
    condition_model_type = "WanConditionModel"

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        attn_implementation: str = "sdpa",
        task: str = "t2v",
        **kwargs: Any,
    ) -> None:
        self.patch_size = tuple(patch_size)
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.cross_attn_norm = cross_attn_norm
        self.qk_norm = qk_norm
        self.eps = eps
        self.image_dim = image_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.rope_max_seq_len = rope_max_seq_len
        self.pos_embed_seq_len = pos_embed_seq_len
        self.task = task
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(attn_implementation=attn_implementation, **kwargs)

    @classmethod
    def from_config_source(
        cls,
        config_source: str | Path,
        *,
        task: str = "t2v",
        attn_implementation: str = "sdpa",
        **overrides: Any,
    ) -> "WanTransformer3DTrainingConfig":
        config_dict = _convert_veomni_wan_config(_load_json_config(config_source))
        config_dict.update(overrides)
        config_dict["task"] = task
        config_dict["attn_implementation"] = attn_implementation
        return cls(**config_dict)

    def to_diffuser_dict(self) -> dict[str, Any]:
        """Return kwargs accepted by the installed Diffusers Wan transformer."""
        return {
            key: getattr(self, key)
            for key in WAN_INIT_SIGNATURE.parameters
            if key != "self" and hasattr(self, key)
        }

    def to_dict(self) -> dict[str, Any]:
        return_dict = super().to_dict()
        return_dict["_class_name"] = "WanTransformer3DModel"
        return_dict["_diffusers_version"] = diffusers.__version__
        return_dict.pop("dtype", None)
        return return_dict

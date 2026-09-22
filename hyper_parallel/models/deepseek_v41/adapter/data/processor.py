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
"""Build the native DeepSeek-V4.1 multimodal processor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from hyper_parallel.models.deepseek_v41.adapter.data.encoding import encode_messages
from hyper_parallel.models.deepseek_v41.adapter.data.image_processor import (
    prepare_vl_inputs as prepare_native_vl_inputs,
)


@dataclass(frozen=True)
class DeepseekV41Processor:
    """Expose model arguments and the official DeepSeek preprocessing APIs."""

    chat_template = staticmethod(encode_messages)
    image_processor = staticmethod(prepare_native_vl_inputs)

    tokenizer: Any
    vision_n_layers: int
    vision_patch_size: int
    vision_max_wh_ratio: int | None
    vision_min_pixels: int
    vision_downsample_ratio: int
    vision_max_n_token: int
    image_token_id: int

    @property
    def vision_enabled(self) -> bool:
        """Return whether the official DeepSeek configuration enables vision."""
        vision_enabled = self.vision_n_layers > 0
        return vision_enabled


def build_deepseek_v41_processor(
        *,
        config_path: str | None = None,
        pretrained_model_name_or_path: str | None = None,
) -> DeepseekV41Processor:
    """Build native tokenizer, chat template, and image-processing settings.

    Args:
        config_path: Local DeepSeek-V4.1 model directory containing the
            official ``inference/config.json``.
        pretrained_model_name_or_path: Alias for ``config_path`` used by
            generic model-asset configurations.

    Returns:
        Processor consumed by ``DeepseekV41OmniTransform``.

    Raises:
        ValueError: If the model path or its DeepSeek-V4.1 configuration is invalid.
    """
    configured_path = config_path or pretrained_model_name_or_path
    if configured_path is None:
        raise ValueError(
            "build_deepseek_v41_processor requires config_path or "
            "pretrained_model_name_or_path"
        )

    model_path = Path(configured_path).expanduser().resolve()
    config_file = model_path / "inference" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"DeepSeek-V4.1 inference config is missing: {config_file}")

    inference_config = json.loads(config_file.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    processor = DeepseekV41Processor(
        tokenizer=tokenizer,
        vision_n_layers=int(inference_config["vision_n_layers"]),
        vision_patch_size=int(inference_config["vision_patch_size"]),
        vision_max_wh_ratio=inference_config.get("vision_max_wh_ratio"),
        vision_min_pixels=int(inference_config["vision_min_pixels"]),
        vision_downsample_ratio=int(inference_config["vision_downsample_ratio"]),
        vision_max_n_token=int(inference_config["vision_max_n_token"]),
        image_token_id=int(inference_config["image_token_id"]),
    )
    return processor


__all__ = ["DeepseekV41Processor", "build_deepseek_v41_processor"]

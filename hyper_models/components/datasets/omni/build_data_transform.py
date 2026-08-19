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
"""Build the private Omni multimodal data transform."""

from typing import Any

from hyper_models.components.datasets.contracts import SampleTransform


def build_data_transform(
        *,
        tokenizer: Any = None,
        chat_template: Any = None,
        image_processor: Any = None,
        video_processor: Any = None,
        audio_processor: Any = None,
        **transform_options: Any,
) -> SampleTransform:
    """Build the Omni Trainer's modality-specific data transform.

    Args:
        tokenizer: Tokenizer used for multimodal conversations.
        chat_template: Text template used to render conversations.
        image_processor: Optional image processor.
        video_processor: Optional video processor.
        audio_processor: Optional audio processor.
        **transform_options: Model-specific multimodal transform options.

    Returns:
        The configured Omni sample transform.

    Raises:
        NotImplementedError: Until the Omni transform implementation is connected.
    """
    del tokenizer, chat_template, image_processor, video_processor, audio_processor, transform_options
    raise NotImplementedError("Omni data transforms are not implemented")


def build_omni_transform(**kwargs: Any) -> SampleTransform:
    """Build the Omni transform using its previous public name."""
    return build_data_transform(**kwargs)


def build_omni_adapter(**kwargs: Any) -> SampleTransform:
    """Compatibility entry point for the former Omni adapter module."""
    return build_data_transform(**kwargs)


__all__ = ["build_data_transform", "build_omni_adapter", "build_omni_transform"]

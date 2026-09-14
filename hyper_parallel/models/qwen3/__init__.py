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
"""Qwen3 dense model-family integration."""

from typing import Any

__all__ = ["get_adapter_spec", "build_causal_lm"]


def get_adapter_spec() -> Any:
    """Return the Qwen3 dense adapter specification through the shared registry."""
    from hyper_parallel.models.registry import get_model_adapter  # pylint: disable=C0415
    return get_model_adapter("qwen3")


def build_causal_lm(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Load and parallelize a Qwen3 causal language model."""
    from hyper_parallel.models.qwen3.runtime import build_causal_lm as build  # pylint: disable=C0415
    return build(pretrained_model_name_or_path, **kwargs)

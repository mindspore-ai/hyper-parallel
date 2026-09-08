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
"""Stable public entry points for AutoModels model construction.

Programmatic callers use this module instead of the Trainer layer. During
the directory migration the entry points delegate to the private
Transformers integration; this module never imports
``hyper_parallel.trainer`` or ``hyper_parallel.models.trainer``.
"""

from typing import Any, Optional

from hyper_parallel.models.build_options import (
    CompileConfig,
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
    ModelBuildOptions,
    normalize_build_options,
)


def from_pretrained(pretrained_model_name_or_path: str, **kwargs: Any):
    """Load a pretrained model through the HF-compatible facade.

    Migration delegate: forwards to
    ``_transformers.HyperAutoModelForCausalLM.from_pretrained`` until the
    model builder is split out of the private integration package.
    """
    from hyper_parallel.models._transformers import (  # noqa: E402  # lazy: keep api import light
        HyperAutoModelForCausalLM,
    )

    return HyperAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )


def normalize_options(value: Optional[Any]) -> ModelBuildOptions:
    """Normalize programmatic build options at the AutoModels boundary.

    Accepts ``None``, a plain mapping, or a :class:`ModelBuildOptions`;
    never constructs or accepts a Trainer DTO.
    """
    return normalize_build_options(value)


__all__ = [
    "CompileConfig",
    "FSDP2Config",
    "FSDP2MixedPrecisionConfig",
    "ModelBuildOptions",
    "from_pretrained",
    "normalize_options",
]

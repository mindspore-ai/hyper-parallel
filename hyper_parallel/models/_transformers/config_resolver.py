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
"""HF config resolution (01 §5; 05 stage-5 item 4).

HF config parsing and model-family discovery must not share one registry:
the family registry lives in ``models/registry.py`` (M1, adjust doc §7.2),
while this module owns the HF-side config helpers that decide between the
HF native implementation and a Hyper-Parallel custom implementation.
"""

from typing import Any

from transformers import AutoConfig, PretrainedConfig

from hyper_parallel.models.registry import _resolve_custom_model_cls

__all__ = [
    "get_hf_config",
    "get_is_hf_model",
]


def get_is_hf_model(config: PretrainedConfig, force_hf: bool = False) -> bool:
    """Determine whether to use HF native implementation.

    Returns:
        True: Use HF native AutoModel.from_pretrained()
        False: Use Hyper-Parallel custom implementation
    """
    if force_hf:
        return True
    architectures = getattr(config, "architectures", []) or []
    arch_name = architectures[0] if architectures else ""
    return _resolve_custom_model_cls(arch_name) is None


def get_hf_config(
    path: str,
    attn_implementation: str = "sdpa",
    torch_dtype: Any = "auto",
    **kwargs: Any,
) -> PretrainedConfig:
    """Wrap AutoConfig.from_pretrained with unified attn_implementation/dtype injection.

    Following design doc 01 §5.1.
    """
    config_kwargs = dict(kwargs)
    config_kwargs.setdefault("attn_implementation", attn_implementation)
    if torch_dtype != "auto":
        config_kwargs.setdefault("torch_dtype", torch_dtype)
    return AutoConfig.from_pretrained(path, **config_kwargs)

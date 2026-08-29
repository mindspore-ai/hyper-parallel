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
"""MODEL_ARCH_MAPPING registry — HF architectures → Hyper-Parallel custom model implementations.

Following design doc 01_hf_compatibility_layer.md §5.
"""

import importlib
import logging
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional

from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)

# OrderedDict: arch_name → (module_path, class_name)
# Lazy-loaded — only imported on first access.
MODEL_ARCH_MAPPING = OrderedDict([])


@lru_cache(maxsize=128)
def _resolve_custom_model_cls(arch_name: str) -> Optional[type]:
    """Lazy-load model class from MODEL_ARCH_MAPPING.

    Returns None → fall back to HF native.
    """
    entry = MODEL_ARCH_MAPPING.get(arch_name)
    if entry is None:
        return None
    module_path, class_name = entry[0], entry[1]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(
            "Failed to load custom model %s from %s: %s. Falling back to HF native.",
            class_name, module_path, e,
        )
        return None


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

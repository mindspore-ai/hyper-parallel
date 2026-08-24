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
"""_init_model — custom vs HF model path dispatch.

Following design doc 01_hf_compatibility_layer.md §7.
"""

import logging
from typing import Optional

from transformers import PreTrainedModel

from hyper_models._transformers.registry import _resolve_custom_model_cls

logger = logging.getLogger(__name__)


def _init_model(
    cls,                          # HyperAutoModelForCausalLM etc.
    pretrained_model_name_or_path: Optional[str],
    hf_config,                    # AutoConfig.from_pretrained() result
    attn_implementation: str,
    torch_dtype,
    is_hf_model: bool,
    *model_args,
    backend=None,
    **kwargs,
) -> tuple[bool, PreTrainedModel]:
    """Initialize model — dispatching to custom or HF path.

    Following design doc 01 §7.

    Args:
        cls: The HyperAutoModel* class.
        pretrained_model_name_or_path: HF hub repo ID or local path (None for from_config).
        hf_config: PretrainedConfig from AutoConfig.
        attn_implementation: "sdpa" / "flash_attention_2" / "eager".
        torch_dtype: "auto" / "bfloat16" / etc.
        is_hf_model: True = HF native, False = custom implementation.
        *model_args: Extra positional args for model constructor.
        backend: Backend configuration (reserved for interface compatibility
            with HyperAutoModel.from_pretrained; not used yet).
        **kwargs: Extra keyword args.

    Returns:
        (is_custom_model, model)
    """
    _ = backend  # Reserved for interface compatibility; not used yet.
    architectures = getattr(hf_config, "architectures", []) or []
    arch_name = architectures[0] if architectures else ""

    # ── Path A: HF native ──
    if is_hf_model:
        # Use HF parent class from_pretrained (meta device if context is set)
        model = cls._from_pretrained_parent_class(
            pretrained_model_name_or_path,
            *model_args,
            config=hf_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        return False, model

    # ── Path B: Custom model implementation ──
    custom_model_cls = _resolve_custom_model_cls(arch_name)
    if custom_model_cls is None:
        # Fallback to HF native
        logger.warning(
            "Custom model class for %s not found; falling back to HF native.",
            arch_name,
        )
        model = cls._from_pretrained_parent_class(
            pretrained_model_name_or_path,
            *model_args,
            config=hf_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        return False, model

    # Instantiate custom model
    if pretrained_model_name_or_path is not None:
        model = custom_model_cls.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=hf_config,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    else:
        model = custom_model_cls.from_config(
            hf_config,
            *model_args,
            **kwargs,
        )

    return True, model

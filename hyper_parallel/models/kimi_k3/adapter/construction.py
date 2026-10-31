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
"""Local K3 text configuration and standard scratch-training construction."""
from __future__ import annotations

import json
import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.kimi_k3.configuration_kimi_k3 import KimiK3Config, KimiLinearConfig
from hyper_parallel.models.kimi_k3.adapter.policies.sharding import build_training_plan_overrides

logger = logging.getLogger(__name__)


def _merge_training_plan_overrides(defaults: dict[str, Any], user_overrides: dict[str, Any]) -> dict[str, Any]:
    """Preserve unspecified family defaults while merging explicit recipe choices."""
    for pattern, user_spec in user_overrides.items():
        if pattern in defaults:
            inherited = {
                field.name: getattr(defaults[pattern], field.name)
                for field in fields(user_spec)
                if field.init and getattr(user_spec, field.name) is None
            }
            if user_spec.params == "auto":
                inherited["params"] = defaults[pattern].params
            defaults[pattern] = replace(user_spec, **inherited)
        else:
            defaults[pattern] = user_spec
    return defaults

def load_kimi_linear_training_config(config_path: str, *,
                                     strip_inference_quantization: bool = False) -> KimiLinearConfig:
    """Read a native LLM config, optionally extracting K3 text_config.

    Extraction preserves text dimensions and layer roles. No crop, weight loading
    or checkpoint conversion occurs.

    Args:
        config_path: Local config.json or directory containing it.
        strip_inference_quantization: Explicitly discard inference quantization
            metadata before constructing a floating-point training config.

    Returns:
        The complete text architecture as a KimiLinearConfig.

    Raises:
        ValueError: The model type, text section, or quantization policy is invalid.
    """
    path = Path(config_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    model_type = values.get("model_type")
    if model_type == "kimi_k3":
        if not isinstance(values.get("text_config"), dict):
            raise ValueError("K3 config must contain text_config for LLM training")
        outer_quantization = values.get("quantization_config")
        values = dict(values["text_config"])
    elif model_type == "kimi_linear":
        outer_quantization = None
    else:
        raise ValueError("K3 LLM training config must declare model_type=kimi_k3 or kimi_linear")
    if outer_quantization or values.get("quantization_config"):
        if not strip_inference_quantization:
            raise ValueError("Published inference quantization requires explicit strip_inference_quantization=True")
        values.pop("quantization_config", None)
        logger.info("Removed inference quantization metadata for random-init K3 LLM training")
    values["architectures"] = ["KimiLinearForCausalLM"]
    return KimiLinearConfig(**values)


def build_kimi_linear_training_model(config_path: str, *, distributed_setup: Any = None,
                                     model_init_dtype: str = "float32", torch_dtype: str = "bfloat16",
                                     strip_inference_quantization: bool = False, **kwargs: Any) -> Any:
    """Build the portable text model through the shared AutoModel pipeline.

    Args:
        config_path: Local text config JSON or composite K3 config containing text_config.
        distributed_setup: Standard mesh, FSDP and declarative model plan configuration.
        model_init_dtype: Dtype used to initialize persistent parameters.
        torch_dtype: Model compute dtype requested from the common builder.
        strip_inference_quantization: Explicitly discard inference quantization metadata
            for random initialization; this does not load or dequantize weights.
        **kwargs: Additional common AutoModel construction options.

    Returns:
        The native causal LLM, with model-owned AttnRes parameter boundaries.

    Raises:
        ValueError: Model-parallel settings are outside the text baseline scope.
    """
    mesh = getattr(distributed_setup, "mesh_context", None)
    if any(int(getattr(mesh, name, 1)) != 1 for name in ("tp_size", "pp_size")):
        raise ValueError("K3 native text baseline requires TP=PP=1")
    if bool(getattr(mesh, "sequence_parallel", False)) or bool(getattr(mesh, "loss_parallel", False)):
        raise ValueError("K3 native text baseline disables sequence and loss parallelism")
    config = load_kimi_linear_training_config(
        config_path, strip_inference_quantization=strip_inference_quantization,
    )
    if distributed_setup is not None:
        overrides = _merge_training_plan_overrides(
            build_training_plan_overrides(config), distributed_setup.plan_overrides or {},
        )
        distributed_setup = replace(distributed_setup, plan_overrides=overrides)
    return HyperAutoModelForCausalLM.from_config(
        config, distributed_setup=distributed_setup, model_init_dtype=model_init_dtype,
        torch_dtype=torch_dtype, **kwargs,
    )


def load_kimi_k3_training_config(config_path: str, *, strip_inference_quantization: bool = False) -> KimiK3Config:
    """Read a local HF JSON and explicitly opt into floating-point scratch training.

    Args:
        config_path: A local config.json or directory containing it.
        strip_inference_quantization: Remove inference-only quantization metadata;
            this does not load or convert quantized checkpoint weights.
    """
    path = Path(config_path).expanduser()
    if path.is_dir():
        path = path / "config.json"
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    if values.get("model_type") != "kimi_k3":
        raise ValueError("K3 training config must declare model_type=kimi_k3")
    for section in (values, values.get("text_config", {})):
        if section.get("quantization_config"):
            if not strip_inference_quantization:
                raise ValueError("Published inference quantization requires explicit strip_inference_quantization=True")
            section.pop("quantization_config")
            logger.info("Removed inference quantization metadata for random-init K3 training")
    values["architectures"] = ["KimiK3ForConditionalGeneration"]
    return KimiK3Config(**values)


def build_kimi_k3_training_model(config_path: str, *, distributed_setup: Any = None,
                                 model_init_dtype: str = "float32", torch_dtype: str = "bfloat16",
                                 strip_inference_quantization: bool = False, **kwargs: Any) -> Any:
    """Build the native image/video/text model through the standard lifecycle.

    Args:
        config_path: Local composite config.json or its containing directory.
        distributed_setup: Shared FSDP/CP/EP setup with TP=PP=1.
        model_init_dtype: Persistent parameter initialization dtype.
        torch_dtype: Requested model compute dtype.
        strip_inference_quantization: Explicit removal of inference metadata for
            random initialization; this does not load or convert weights.
        **kwargs: Additional common AutoModel construction options.

    Returns:
        The complete model with visual and text parameter boundaries.

    Raises:
        ValueError: Model-parallel settings exceed the CP1 baseline scope.
    """
    mesh = getattr(distributed_setup, "mesh_context", None)
    if any(int(getattr(mesh, name, 1)) != 1 for name in ("tp_size", "pp_size")):
        raise ValueError("K3 media baseline requires TP=PP=1")
    if bool(getattr(mesh, "sequence_parallel", False)) or bool(getattr(mesh, "loss_parallel", False)):
        raise ValueError("K3 media baseline disables sequence and loss parallelism")
    config = load_kimi_k3_training_config(
        config_path, strip_inference_quantization=strip_inference_quantization,
    )
    if distributed_setup is not None:
        overrides = _merge_training_plan_overrides(
            build_training_plan_overrides(config), distributed_setup.plan_overrides or {},
        )
        distributed_setup = replace(distributed_setup, plan_overrides=overrides)
    return HyperAutoModelForCausalLM.from_config(
        config, distributed_setup=distributed_setup, model_init_dtype=model_init_dtype,
        torch_dtype=torch_dtype, **kwargs,
    )

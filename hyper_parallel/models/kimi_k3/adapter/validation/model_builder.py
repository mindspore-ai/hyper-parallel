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
"""Build the production eager class for local, replacement-disabled Gate A."""
# Native Torch model adapters follow the same exception as other HF model families.
# pylint: disable=forbidden-backend-import

import torch

from hyper_parallel.models.kimi_k3.adapter.construction import (
    load_kimi_linear_training_config, load_kimi_k3_training_config,
)
from hyper_parallel.models.kimi_k3.modeling_kimi_linear import KimiLinearForCausalLM
from hyper_parallel.models.kimi_k3.modeling_kimi_k3 import KimiK3ForConditionalGeneration


def build_validation_model(config_path: str, *, dtype: str = "float32", device: str = "cpu", seed: int = 0,
                           strip_inference_quantization: bool = False) -> KimiLinearForCausalLM:
    """Build a local validation model without downloads, crops or replacements.

    Configured optional kernels are retained. The caller supplies an architecture
    that fits the requested device; this is not the distributed Trainer factory.

    Args:
        config_path: Local K3 or Kimi Linear config.json, or its containing directory.
        dtype: Parameter dtype, float32 or bfloat16.
        device: Target device accepted by Module.to.
        seed: Nonnegative initialization seed; the caller's CPU RNG state is restored.
        strip_inference_quantization: Discard inference-only quantization metadata
            for floating-point random initialization.

    Returns:
        The configured native causal LLM on the requested device and dtype.

    Raises:
        ValueError: The dtype, seed or training config is unsupported.
    """
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("K3 validation dtype must be float32 or bfloat16")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("K3 validation seed must be a nonnegative integer")
    config = load_kimi_linear_training_config(
        config_path, strip_inference_quantization=strip_inference_quantization,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = KimiLinearForCausalLM.from_config(config)
    return model.to(device=device, dtype=getattr(torch, dtype))


def build_multimodal_validation_model(
        config_path: str, *, dtype: str = "float32", device: str = "cpu", seed: int = 0,
        strip_inference_quantization: bool = False,
) -> KimiK3ForConditionalGeneration:
    """Build the complete local multimodal architecture without recipe replacements.

    Args:
        config_path: Local K3 config.json, or the directory containing it.
        dtype: Parameter dtype, float32 or bfloat16.
        device: Target device accepted by Module.to.
        seed: Nonnegative initialization seed; the caller's CPU RNG state is restored.
        strip_inference_quantization: Discard inference-only quantization metadata
            for floating-point random initialization.

    Returns:
        The configured native image/text model, including its visual tower and
        projector. No model crop, checkpoint download, CP or EP wrapper is inferred.

    Raises:
        ValueError: The dtype, seed or training configuration is unsupported.
    """
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("K3 validation dtype must be float32 or bfloat16")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("K3 validation seed must be a nonnegative integer")
    config = load_kimi_k3_training_config(
        config_path, strip_inference_quantization=strip_inference_quantization,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = KimiK3ForConditionalGeneration.from_config(config)
    return model.to(device=device, dtype=getattr(torch, dtype))

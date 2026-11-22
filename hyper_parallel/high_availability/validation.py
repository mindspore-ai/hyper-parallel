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
"""Validate configuration and runtime state for process recovery."""

from typing import Any, Mapping

import torch

from hyper_parallel.components.modules.engram.host_state import HostEngramState
from hyper_parallel.components.quantization.functional.base_gmm_func import GroupedLinear
from hyper_parallel.components.quantization.functional.base_linear_func import LinearStrategy
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import HiFloat8LinearStrategy
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import MXFP8LinearStrategy
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.quantizers.base import Quantizer
from hyper_parallel.components.quantization.quantizers.fake_w4a8 import FakeW4A8Quantizer
from hyper_parallel.components.quantization.quantizers.hifloat8 import HiFloat8Quantizer
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.components.quantization.quantizers.w4a8 import W4A8Quantizer
from hyper_parallel.components.quantization.tensor.quantized_tensor import QuantizedTensor

_ROLE_QUANTIZERS = ("input_quantizer", "weight_quantizer", "grad_output_quantizer")
_STRATEGIES = {
    MXFP8LinearStrategy: (MXFP8Quantizer, ("quantizer",)),
    HiFloat8LinearStrategy: (HiFloat8Quantizer, _ROLE_QUANTIZERS),
    MXFP8GroupedLinear: (MXFP8Quantizer, ("quantizer",)),
    HiFloat8GroupedLinear: (HiFloat8Quantizer, _ROLE_QUANTIZERS),
    W4A8GroupedLinear: (W4A8Quantizer, ("quantizer",)),
    FakeW4A8GroupedLinear: (FakeW4A8Quantizer, ("quantizer",)),
}
_QUANTIZERS = (MXFP8Quantizer, HiFloat8Quantizer, W4A8Quantizer, FakeW4A8Quantizer)
_PARAMETER_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def validate_recovery_config(config: Any) -> None:
    """Reject unsupported Trainer configurations before construction.

    Args:
        config: Trainer configuration for a recovery-enabled job.
    """
    if config.accelerator.pp_size != 1:
        raise ValueError("PP schedule state requires a dedicated recovery adapter")
    if config.training.backend not in ("hccl", "nccl"):
        raise ValueError("worker recovery requires an HCCL or NCCL backend")
    if config.compile.enabled:
        raise ValueError("worker recovery requires eager training")
    if config.dataloader is None or config.dataloader.use_background_prefetcher:
        raise ValueError("worker recovery requires a stateful loader without background prefetch")
    if config.model_integration.mode != "off" or config.peft is not None or config.magi is not None:
        raise ValueError("worker recovery does not yet cover integration probes, PEFT or external model state")
    if config.training.eval_iters or config.training.eval_steps or config.training.eval_epochs:
        raise ValueError("worker recovery does not yet cover evaluation data state")


def validate_recovery_model(model: torch.nn.Module) -> None:
    """Check model-owned state and low-precision recovery compatibility before binding.

    Args:
        model: Materialized model with its parallel strategies applied.
    """
    external = getattr(model, "_hp_external_state", None)
    if external is not None and not isinstance(external, HostEngramState):
        raise ValueError("model-owned external state requires its own recovery provider")
    validate_low_precision_state(model)


def validate_low_precision_state(model: torch.nn.Module) -> None:
    """Allow built-in current-scaling computation with ordinary persistent weights.

    Args:
        model: Materialized model, including module-family quantization entries.

    Raises:
        ValueError: A strategy, quantizer or persistent quantized tensor needs a
            dedicated recovery contract. Subclasses are checked explicitly because
            they may introduce unregistered observer or scaling-history state.
    """
    for name, module in model.named_modules():
        has_quantization = False
        for attribute, value in vars(module).items():
            location = f"{name or '<root>'}.{attribute}"
            if isinstance(value, (LinearStrategy, GroupedLinear)):
                specification = _STRATEGIES.get(type(value))
                if specification is None:
                    raise ValueError(f"{location}: custom low-precision strategy requires a recovery adapter")
                quantizer_type, fields = specification
                for field in fields:
                    actual_type = type(getattr(value, field, None))
                    if actual_type is not quantizer_type:
                        raise ValueError(f"{location}.{field}: custom quantizer requires a recovery adapter")
                has_quantization = True
            elif isinstance(value, Quantizer):
                if type(value) not in _QUANTIZERS:
                    raise ValueError(f"{location}: custom quantizer requires a recovery adapter")
                has_quantization = True
        for parameter_name, parameter in module.named_parameters(recurse=False):
            local = getattr(parameter, "_local_tensor", parameter)
            if isinstance(local, QuantizedTensor) or (has_quantization and local.dtype not in _PARAMETER_DTYPES):
                raise ValueError(f"{name or '<root>'}.{parameter_name}: recovery requires high-precision parameters")
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if isinstance(getattr(buffer, "_local_tensor", buffer), QuantizedTensor):
                raise ValueError(f"{name or '<root>'}.{buffer_name}: persistent quantized buffers require an adapter")


def validate_recovery_state_providers(loader: Any, replica_providers: Mapping[str, Any]) -> None:
    """Check that the loader and replica providers expose state save and restore.

    Args:
        loader: Training loader bound to the recovery session.
        replica_providers: Named objects with equivalent state across replicas.
    """
    for provider in (loader, *replica_providers.values()):
        if (not callable(getattr(provider, "state_dict", None))
                or not callable(getattr(provider, "load_state_dict", None))):
            raise ValueError("recovery requires state_dict/load_state_dict for every training state provider")

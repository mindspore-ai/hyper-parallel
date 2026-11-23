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
"""Megatron-aligned fp32 main-parameter optimizer composition."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from hyper_parallel.core.optimizer.dtensor_compat import (
    device_meshes_are_compatible,
    is_dtensor,
    to_local_if_dtensor,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


MIXED_PRECISION_OPTIMIZER_STATE_KEY = "_mixed_precision_optimizer"
FP32_MAIN_PARAM_STATE_KEY = "fp32_from_fp16_params"

logger = logging.getLogger(__name__)


def _copy_tensor(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy one local parameter shard without invoking DTensor dispatch."""
    destination_local = to_local_if_dtensor(destination)
    source_local = to_local_if_dtensor(source)
    source_local = source_local.to(
        device=destination_local.device,
        dtype=destination_local.dtype,
    )
    destination_local.copy_(source_local)


def _gradient_for_param(
        optimizer_param: torch.Tensor,
        gradient: torch.Tensor,
) -> torch.Tensor:
    """Validate gradient layout and cast dtype on the gradient's existing device.

    DTensor optimizer parameters keep a DTensor gradient so lazily-created
    optimizer state inherits its complete global layout. FSDP ``main_grad`` is
    already detached from autograd, so the DTensor object itself is preserved.
    Plain parameters retain the detached local-tensor path used by non-FSDP and
    third-party integrations.
    """
    optimizer_is_dtensor = is_dtensor(optimizer_param)
    gradient_is_dtensor = is_dtensor(gradient)
    if optimizer_is_dtensor != gradient_is_dtensor:
        raise TypeError(
            "Optimizer parameter and gradient must both be DTensors or both be local tensors"
        )

    optimizer_local = to_local_if_dtensor(optimizer_param)
    gradient_local = to_local_if_dtensor(gradient)
    if gradient_local.device != optimizer_local.device:
        raise ValueError(
            "Optimizer gradient must already be on its parameter's device: "
            f"got {gradient_local.device} and {optimizer_local.device}"
        )
    if tuple(gradient_local.shape) != tuple(optimizer_local.shape):
        raise ValueError(
            "Optimizer gradient local shape must match its parameter shard: "
            f"got {tuple(gradient_local.shape)} and {tuple(optimizer_local.shape)}"
        )
    if optimizer_is_dtensor:
        if tuple(gradient.shape) != tuple(optimizer_param.shape):
            raise ValueError(
                "Optimizer gradient global shape must match its parameter: "
                f"got {tuple(gradient.shape)} and {tuple(optimizer_param.shape)}"
            )
        if not device_meshes_are_compatible(
                gradient.device_mesh,
                optimizer_param.device_mesh,
        ):
            raise ValueError("Optimizer gradient and parameter must use the same device mesh")
        if tuple(gradient.placements) != tuple(optimizer_param.placements):
            raise ValueError("Optimizer gradient and parameter must use the same placements")
        # FSDP main_grad is already an autograd-detached accumulation buffer.
        # Keep that DTensor object intact: detach() rebuilds an uneven empty
        # shard without its logical tensor metadata, which makes a subsequent
        # zeros_like() infer the optimizer state's global shape as zero.
        prepared_gradient = gradient
    else:
        prepared_gradient = gradient_local.detach()
    if gradient_local.dtype != optimizer_local.dtype:
        prepared_gradient = prepared_gradient.to(dtype=optimizer_local.dtype)
    return prepared_gradient


@dataclass(frozen=True)
class _ParameterBinding:
    """Describe optimizer storage independently of eager or graph execution."""

    model: nn.Parameter
    optimizer: nn.Parameter
    layout: Optional[Layout] = None
    local_shape: tuple[int, ...] = ()
    owns_storage: bool = False
    storage_ptr: Optional[int] = None
    checkpoint_name: Optional[str] = None

    def local_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the valid local storage without changing distributed metadata.

        Args:
            tensor: Model or gradient tensor containing the physical local shard.
        """
        local = to_local_if_dtensor(tensor)
        if self.layout is not None and local.ndim:
            return local.narrow(0, 0, self.local_shape[0])
        return local


class MixedPrecisionOptimizer:
    """Share gradient, update and checkpoint flows over final parameter bindings."""

    def __init__(
            self, optimizer: ChainedOptimizer, model: nn.Module, *,
            fp32_main_params: bool = True, defer_binding: bool = False,
    ) -> None:
        """Expose the inner optimizer while optionally deferring final storage.

        Args:
            optimizer: Chained leaf-optimizer composition.
            model: Module containing the computation parameters.
            fp32_main_params: Whether low-precision parameters have FP32 main weights.
            defer_binding: Whether final parameter shards are not available yet.
        """
        self.optimizer = optimizer
        self.model = model
        self.optimizers_dict: dict[str, Any] = optimizer.optimizers_dict
        self.chained_optimizers: list[Any] = optimizer.chained_optimizers
        self.param_groups: list[dict[str, Any]] = optimizer.param_groups
        self.fp32_main_params = fp32_main_params
        self._bound = not defer_binding
        self._bindings: list[_ParameterBinding] = []
        self._main_param_by_fqn: dict[str, nn.Parameter] = {}
        self.model_param_by_optimizer_param: dict[nn.Parameter, nn.Parameter] = {}
        self.optimizer_param_by_model_param: dict[nn.Parameter, nn.Parameter] = {}

    def _register_parameter_bindings(self, bindings: list[_ParameterBinding]) -> None:
        """Index independent main weights without mistaking aliases for copies."""
        self._bindings = bindings
        self.model_param_by_optimizer_param.clear()
        self.model_param_by_optimizer_param.update(
            (binding.optimizer, binding.model)
            for binding in bindings if binding.optimizer is not binding.model
        )
        self._main_param_by_fqn = {
            (binding.checkpoint_name if binding.checkpoint_name is not None else binding.model.model_name):
                binding.optimizer
            for binding in bindings if binding.owns_storage
        }

    def _require_bound(self) -> None:
        """Require final shard views before optimizer operations use them."""
        if not self._bound:
            raise RuntimeError(
                "Compile and bind graph optimizer parameters before updating or restoring optimizer state"
            )

    @staticmethod
    def _prepare_bound_gradient(binding: _ParameterBinding, gradient: torch.Tensor) -> torch.Tensor:
        """Validate an existing DTensor or describe a declared local shard."""
        if binding.layout is None or is_dtensor(gradient):
            return _gradient_for_param(binding.optimizer, gradient)
        local = binding.local_view(gradient.detach())
        prepared = _gradient_for_param(to_local_if_dtensor(binding.optimizer), local)
        return DTensor.from_local_with_layout(prepared, binding.layout, shape=binding.layout.tensor_shape)

    def _copy_model_grads_to_main_grads(self) -> None:
        """Prepare final gradients while preserving producer ownership semantics."""
        for binding in self._bindings:
            if binding.storage_ptr is not None and binding.model.data_ptr() != binding.storage_ptr:
                raise RuntimeError(
                    f"Graph parameter storage changed after optimizer binding: {binding.model.model_name}"
                )
            model_gradient = getattr(binding.model, "main_grad", None)
            had_main_grad = model_gradient is not None
            if binding.optimizer is binding.model and not had_main_grad:
                continue
            if model_gradient is None:
                model_gradient = binding.model.grad
            binding.optimizer.grad = (
                self._prepare_bound_gradient(binding, model_gradient)
                if model_gradient is not None else None
            )
            if had_main_grad:
                binding.model.main_grad = binding.optimizer.grad
            if binding.optimizer is not binding.model:
                binding.model.grad = None

    @torch.no_grad()
    def _copy_main_params_to_model_params(self) -> None:
        """Copy owned main weights back and clear invalid physical padding."""
        for binding in self._bindings:
            if binding.owns_storage:
                _copy_tensor(binding.local_view(binding.model), binding.optimizer)
            if binding.layout is not None:
                local = to_local_if_dtensor(binding.model)
                if local.ndim and binding.local_shape[0] < local.shape[0]:
                    local[binding.local_shape[0]:].zero_()

    @torch.no_grad()
    def _copy_model_params_to_main_params(self) -> None:
        """Refresh only optimizer-owned main weights from valid model shards."""
        for binding in self._bindings:
            if binding.owns_storage:
                _copy_tensor(binding.optimizer, binding.local_view(binding.model))

    def prepare_grads(self) -> None:
        """Prepare model gradients for the shared inner optimizer."""
        self._require_bound()
        self._copy_model_grads_to_main_grads()

    def step_with_ready_grads(self, closure: Any = None) -> Any:
        """Update final optimizer parameters and synchronize model storage.

        Args:
            closure: Optional loss closure forwarded to the leaf optimizers.
        """
        self._require_bound()
        loss = self.optimizer.step(closure=closure)
        self._copy_main_params_to_model_params()
        return loss

    def step(self, closure: Any = None) -> Any:
        """Prepare gradients and execute one eager optimizer update.

        Args:
            closure: Optional loss closure forwarded to the leaf optimizers.
        """
        self.prepare_grads()
        return self.step_with_ready_grads(closure=closure)

    def reload_model_params(self) -> None:
        """Initialize owned main parameters from the current model values."""
        self._require_bound()
        self._copy_model_params_to_main_params()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear inner gradients and model-side runtime accumulation state.

        Args:
            set_to_none: Whether inner optimizer gradients should be released.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for binding in self._bindings:
            binding.model.grad = None
            if hasattr(binding.model, "main_grad"):
                binding.model.main_grad = None
            if hasattr(binding.model, "grad_added_to_main_grad"):
                binding.model.grad_added_to_main_grad = False

    def state_dict(self) -> dict[str, Any]:
        """Append independent FP32 main weights to the common FQN state."""
        self._require_bound()
        state_dict = self.optimizer.state_dict()
        if self.fp32_main_params:
            state_dict[MIXED_PRECISION_OPTIMIZER_STATE_KEY] = {
                FP32_MAIN_PARAM_STATE_KEY: dict(self._main_param_by_fqn),
            }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer moments and explicit fp32 main weights.

        Args:
            state_dict: Optimizer state containing leaf and main-param entries.
        """
        self._require_bound()
        optimizer_state = dict(state_dict)
        if not self.fp32_main_params:
            optimizer_state.pop(MIXED_PRECISION_OPTIMIZER_STATE_KEY, None)
            self.optimizer.load_state_dict(optimizer_state)
            self.param_groups = self.optimizer.param_groups
            return
        if MIXED_PRECISION_OPTIMIZER_STATE_KEY not in optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
            self.param_groups = self.optimizer.param_groups
            self.reload_model_params()
            logger.warning(
                "Checkpoint has no %s state; initialized fp32 main parameters "
                "from the loaded model parameters. Values not representable in "
                "the model precision cannot be recovered.",
                MIXED_PRECISION_OPTIMIZER_STATE_KEY,
            )
            return

        mixed_precision_state = optimizer_state.pop(
            MIXED_PRECISION_OPTIMIZER_STATE_KEY
        )
        if not isinstance(mixed_precision_state, dict):
            raise RuntimeError(
                "Mixed-precision optimizer checkpoint state must be a mapping"
            )
        checkpoint_main_params = mixed_precision_state.get(
            FP32_MAIN_PARAM_STATE_KEY
        )
        if not isinstance(checkpoint_main_params, dict):
            raise RuntimeError(
                "Mixed-precision optimizer checkpoint is missing "
                f"{FP32_MAIN_PARAM_STATE_KEY}"
            )
        missing_fqns = sorted(
            set(self._main_param_by_fqn) - set(checkpoint_main_params)
        )
        if missing_fqns:
            raise RuntimeError(
                "Mixed-precision optimizer checkpoint is missing fp32 main "
                f"parameters: {', '.join(missing_fqns)}"
            )

        self.optimizer.load_state_dict(optimizer_state)
        self.param_groups = self.optimizer.param_groups
        with torch.no_grad():
            for parameter_fqn, main_param in self._main_param_by_fqn.items():
                _copy_tensor(main_param, checkpoint_main_params[parameter_fqn])


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Create eager FP32 main weights while sharing optimizer runtime flows."""

    def __init__(self, optimizer: ChainedOptimizer, model: nn.Module) -> None:
        """Replace leaf optimizer params with fp32 main parameters.

        Args:
            optimizer: Chained AdamW/Muon optimizer over model parameters.
            model: Module containing the computation parameters.
        """
        super().__init__(optimizer, model)
        self.float16_groups: list[list[nn.Parameter]] = []
        self.fp32_from_float16_groups: list[list[nn.Parameter]] = []
        self.fp32_from_fp32_groups: list[list[nn.Parameter]] = []
        self.model_param_by_optimizer_param: dict[nn.Parameter, nn.Parameter] = {}
        self._build_model_and_main_param_groups()
        self.optimizer_param_by_model_param = {
            model_param: optimizer_param
            for optimizer_param, model_param in self.model_param_by_optimizer_param.items()
        }
        self.optimizer.reset_optimizer_parameters(
            self.model_param_by_optimizer_param
        )
        self.param_groups = self.optimizer.param_groups

        parameter_names = {parameter: name for name, parameter in model.named_parameters()}
        bindings = [
            _ParameterBinding(model_param, main_param, owns_storage=True,
                              checkpoint_name=parameter_names[model_param])
            for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups)
            for model_param, main_param in zip(model_group, main_group)
        ]
        bindings.extend(
            _ParameterBinding(parameter, parameter)
            for group in self.fp32_from_fp32_groups for parameter in group
        )
        self._register_parameter_bindings(bindings)

    def _build_model_and_main_param_groups(self) -> None:
        """Build Megatron-compatible model and main-parameter group views."""
        parameter_names = {
            parameter: name for name, parameter in self.model.named_parameters()
        }
        for leaf_optimizer in self.chained_optimizers:
            for optimizer_group in leaf_optimizer.param_groups:
                float16_group = []
                fp32_from_float16_group = []
                fp32_from_fp32_group = []
                for param_index, model_param in enumerate(optimizer_group["params"]):
                    model_param.main_grad = None
                    if model_param.dtype == torch.float32:
                        model_param.main_param = model_param
                        fp32_from_fp32_group.append(model_param)
                        continue
                    if model_param.dtype not in (torch.float16, torch.bfloat16):
                        raise TypeError(
                            "Float16OptimizerWithFloat16Params supports trainable "
                            "float16, bfloat16, or float32 parameters; "
                            f"{parameter_names[model_param]} has dtype {model_param.dtype}"
                        )

                    main_param = nn.Parameter(
                        model_param.detach().to(dtype=torch.float32),
                        requires_grad=model_param.requires_grad,
                    )
                    main_param.model_name = model_param.model_name
                    model_param.main_param = main_param
                    optimizer_group["params"][param_index] = main_param
                    if model_param in leaf_optimizer.state:
                        leaf_optimizer.state[main_param] = leaf_optimizer.state.pop(model_param)

                    float16_group.append(model_param)
                    fp32_from_float16_group.append(main_param)
                    self.model_param_by_optimizer_param[main_param] = model_param

                self.float16_groups.append(float16_group)
                self.fp32_from_float16_groups.append(fp32_from_float16_group)
                self.fp32_from_fp32_groups.append(fp32_from_fp32_group)


__all__ = [
    "FP32_MAIN_PARAM_STATE_KEY",
    "Float16OptimizerWithFloat16Params",
    "MIXED_PRECISION_OPTIMIZER_STATE_KEY",
    "MixedPrecisionOptimizer",
]

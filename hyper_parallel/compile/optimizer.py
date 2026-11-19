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
"""Bind graph-local parameters to the eager distributed optimizer runtime."""

from copy import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from hyper_parallel.components.optim.mixed_precision_optimizer import (
    FP32_MAIN_PARAM_STATE_KEY,
    MIXED_PRECISION_OPTIMIZER_STATE_KEY,
    MixedPrecisionOptimizer,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


@dataclass(frozen=True)
class _ParameterBinding:
    model: nn.Parameter
    optimizer: nn.Parameter
    layout: Optional[Layout]
    local_shape: tuple[int, ...]
    owns_storage: bool
    storage_ptr: int

    def local_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Exclude the pass's padding while retaining the local storage alias."""
        if self.layout is not None and tensor.ndim:
            return tensor.narrow(0, 0, self.local_shape[0])
        return tensor


class GraphOptimizer(MixedPrecisionOptimizer):
    """Expose final graph shards to eager Muon without changing graph inputs.

    Binding is delayed until compilation has physically sliced parameters.
    FP32 views share model storage; optional FP32 main parameters own only the
    final local shard. Leaf optimizers and scheduler-facing groups stay stable.
    """

    def __init__(self, optimizer: ChainedOptimizer, model: nn.Module, *, fp32_main_params: bool = False) -> None:
        """Retain the uninitialized optimizer until final layouts are available.

        Args:
            optimizer: Configured chained optimizer, before its first update.
            model: Plain local parameters consumed by the compiled graph.
            fp32_main_params: Update fp16/bf16 shards through FP32 main weights.
        """
        super().__init__(optimizer, model)
        self.fp32_main_params = fp32_main_params
        self._bindings: list[_ParameterBinding] = []
        self._layout_source = None
        self._bound = False
        self.optimizer_param_by_model_param: dict[nn.Parameter, nn.Parameter] = {}
        self.model_param_by_optimizer_param: dict[nn.Parameter, nn.Parameter] = {}

    def bind(self, parameter_layouts: Optional[dict]) -> None:
        """Bind final logical layouts once, before state or gradients are prepared.

        Args:
            parameter_layouts: Model parameter to ``(Layout, valid_local_shape)``;
                None describes an unsharded graph without a distributed mesh.
        """
        if self._bound:
            if parameter_layouts is not self._layout_source:
                raise RuntimeError("Graph optimizer layout changed after binding; rebuild and restore optimizer state")
            return
        if any(leaf.state for leaf in self.chained_optimizers):
            raise RuntimeError("Graph optimizer must bind final shards before initializing optimizer state")
        for leaf in self.chained_optimizers:
            for group in leaf.param_groups:
                for index, model_param in enumerate(group["params"]):
                    layout, local_shape = (None, tuple(model_param.shape))
                    if getattr(leaf, "is_muon", False) and parameter_layouts is not None:
                        if model_param not in parameter_layouts:
                            raise ValueError(f"Missing graph parameter layout for {model_param.model_name}")
                        layout, local_shape = parameter_layouts[model_param]
                    owns_storage = self.fp32_main_params and model_param.dtype in (torch.float16, torch.bfloat16)
                    local = model_param.detach()
                    if layout is not None and local.ndim:
                        local = local.narrow(0, 0, local_shape[0])
                    if owns_storage:
                        local = local.to(dtype=torch.float32)
                    if layout is not None:
                        layout = copy(layout)
                        layout.set_tensor_meta(layout.tensor_shape, layout.tensor_stride, local.dtype)
                        local = DTensor.from_local_with_layout(local, layout, shape=layout.tensor_shape)
                    optimizer_param = nn.Parameter(local) if layout is not None or owns_storage else model_param
                    optimizer_param.model_name = model_param.model_name
                    group["params"][index] = optimizer_param
                    binding = _ParameterBinding(model_param, optimizer_param, layout, tuple(local_shape),
                                                owns_storage, model_param.data_ptr())
                    self._bindings.append(binding)
                    self.optimizer_param_by_model_param[model_param] = optimizer_param
                    if optimizer_param is not model_param:
                        self.model_param_by_optimizer_param[optimizer_param] = model_param
        self.optimizer.reset_optimizer_parameters(self.model_param_by_optimizer_param)
        self.param_groups = self.optimizer.param_groups
        self._layout_source = parameter_layouts
        self._bound = True

    def _require_bound(self) -> None:
        if not self._bound:
            raise RuntimeError(
                "Compile and bind graph optimizer parameters before updating or restoring optimizer state"
            )

    def prepare_grads(self) -> None:
        """Wrap final local gradients without repeating backward reductions."""
        self._require_bound()
        for binding in self._bindings:
            if binding.model.data_ptr() != binding.storage_ptr:
                raise RuntimeError(
                    f"Graph parameter storage changed after optimizer binding: {binding.model.model_name}"
                )
            if binding.optimizer is binding.model:
                continue
            grad = getattr(binding.model, "main_grad", None)
            if grad is None:
                grad = binding.model.grad
            if grad is not None:
                grad = binding.local_view(to_local_if_dtensor(grad).detach())
                grad = grad.to(dtype=binding.optimizer.dtype)
                if binding.layout is not None:
                    grad = DTensor.from_local_with_layout(grad, binding.layout, shape=binding.layout.tensor_shape)
            binding.optimizer.grad = grad
            binding.model.grad = None
            if hasattr(binding.model, "main_grad"):
                binding.model.main_grad = None

    @torch.no_grad()
    def step_with_ready_grads(self, closure: Any = None) -> Any:
        """Update distributed views and copy owned main weights to graph storage.

        Args:
            closure: Optional loss closure forwarded to the leaf optimizers.
        """
        self._require_bound()
        loss = self.optimizer.step(closure=closure)
        for binding in self._bindings:
            if binding.owns_storage:
                binding.local_view(binding.model).copy_(to_local_if_dtensor(binding.optimizer))
            if binding.layout is not None and binding.model.ndim and binding.local_shape[0] < binding.model.shape[0]:
                binding.model[binding.local_shape[0]:].zero_()
        return loss

    def step(self, closure: Any = None) -> Any:
        """Prepare already reduced gradients and perform one eager update.

        Args:
            closure: Optional loss closure forwarded to the leaf optimizers.
        """
        self.prepare_grads()
        return self.step_with_ready_grads(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear both model and optimizer views after an accumulated update.

        Args:
            set_to_none: Whether leaf optimizer gradients should be released.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.model.parameters():
            parameter.grad = None
            if hasattr(parameter, "main_grad"):
                parameter.main_grad = None

    @torch.no_grad()
    def reload_model_params(self) -> None:
        """Refresh FP32 main shards after loading graph model weights."""
        self._require_bound()
        for binding in self._bindings:
            if binding.owns_storage:
                to_local_if_dtensor(binding.optimizer).copy_(binding.local_view(binding.model))

    def state_dict(self) -> dict[str, Any]:
        """Export FQN-keyed state using original model identities."""
        self._require_bound()
        state = self.optimizer.state_dict()
        if self.fp32_main_params:
            state[MIXED_PRECISION_OPTIMIZER_STATE_KEY] = {
                FP32_MAIN_PARAM_STATE_KEY: {binding.model.model_name: binding.optimizer
                                          for binding in self._bindings if binding.owns_storage},
            }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore moments and main shards after graph layout binding.

        Args:
            state_dict: FQN-keyed optimizer state from a compatible graph layout.
        """
        self._require_bound()
        state = dict(state_dict)
        precision_state = state.pop(MIXED_PRECISION_OPTIMIZER_STATE_KEY, None)
        self.optimizer.load_state_dict(state)
        self.param_groups = self.optimizer.param_groups
        if self.fp32_main_params:
            if precision_state is None:
                self.reload_model_params()
            else:
                main_state = precision_state[FP32_MAIN_PARAM_STATE_KEY]
                with torch.no_grad():
                    for binding in self._bindings:
                        if binding.owns_storage:
                            source = main_state[binding.model.model_name]
                            to_local_if_dtensor(binding.optimizer).copy_(to_local_if_dtensor(source))

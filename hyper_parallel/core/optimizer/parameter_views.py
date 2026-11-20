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
"""Bind physically sharded model tensors to optimizer parameter views."""

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor


if TYPE_CHECKING:
    from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


@dataclass(frozen=True)
class _ParameterBinding:
    """Associate a model shard with the parameter the optimizer updates."""
    model: nn.Parameter
    optimizer: nn.Parameter
    layout: Optional[Layout]
    local_shape: tuple[int, ...]
    owns_storage: bool
    storage_ptr: int

    def local_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the valid slice of a physical shard.

        Args:
            tensor: Weight or gradient in the graph storage layout.
        """
        if self.layout is not None and tensor.ndim:
            return tensor.narrow(0, 0, self.local_shape[0])
        return tensor


class OptimizerParameterViews:
    """Retain valid local views after the graph finishes sharding the model."""

    def __init__(self) -> None:
        """Start with no physical shards bound to the optimizer."""
        self.bindings: list[_ParameterBinding] = []
        self.bound = False
        self._layouts = None

    def require_bound(self) -> None:
        """Guard updates and checkpoint operations until final shards are known."""
        if not self.bound:
            raise RuntimeError(
                "Compile and bind graph optimizer parameters before updating or restoring optimizer state"
            )

    def bind(self, optimizer: "ChainedOptimizer", layouts: Optional[dict], fp32_main_params: bool) -> None:
        """Install optimizer views after validating all parameters and layouts.

        Args:
            optimizer: Chain holding the original model parameters.
            layouts: Parameter layouts and valid local shapes, or None for an unsharded model.
            fp32_main_params: Allocate FP32 master weights for low-precision parameters.
        """
        if self.bound:
            if layouts is not self._layouts:
                raise RuntimeError("Graph optimizer layout changed after binding; rebuild and restore optimizer state")
            return
        if any(leaf.state for leaf in optimizer.chained_optimizers):
            raise RuntimeError("Graph optimizer must bind final shards before initializing optimizer state")

        replacements = []
        for leaf in optimizer.chained_optimizers:
            muon_layouts = layouts if getattr(leaf, "is_muon", False) else None
            for group in leaf.param_groups:
                for index, model in enumerate(group["params"]):
                    binding = self._make_binding(model, muon_layouts, fp32_main_params)
                    replacements.append((group, index, binding))

        reverse = {}
        for group, index, binding in replacements:
            group["params"][index] = binding.optimizer
            if binding.optimizer is not binding.model:
                reverse[binding.optimizer] = binding.model
        try:
            optimizer.reset_optimizer_parameters(reverse)
        except Exception:
            for group, index, binding in replacements:
                group["params"][index] = binding.model
            optimizer.reset_optimizer_parameters({})
            raise
        optimizer.optimizer_param_by_model_param.update(
            (binding.model, binding.optimizer) for _, _, binding in replacements
        )
        self.bindings = [binding for _, _, binding in replacements]
        self._layouts = layouts
        self.bound = True

    @staticmethod
    def _make_binding(model: nn.Parameter, layouts: Optional[dict], fp32_main_params: bool) -> _ParameterBinding:
        """Build the parameter view and record the model storage it represents."""
        layout, shape = None, tuple(model.shape)
        local = model.detach()
        if layouts is not None:
            if model not in layouts:
                raise ValueError(f"Missing graph parameter layout for {model.model_name}")
            layout, shape = layouts[model]
            if local.ndim:
                local = local.narrow(0, 0, shape[0])
        owns_storage = fp32_main_params and model.dtype in (torch.bfloat16, torch.float16)
        if owns_storage:
            local = local.to(dtype=torch.float32)
        if layout is not None:
            layout = copy(layout)
            layout.set_tensor_meta(layout.tensor_shape, layout.tensor_stride, local.dtype)
            local = DTensor.from_local_with_layout(local, layout, shape=layout.tensor_shape)
        param = nn.Parameter(local, requires_grad=model.requires_grad) if layout is not None or owns_storage else model
        param.model_name = model.model_name
        return _ParameterBinding(model, param, layout, tuple(shape), owns_storage, model.data_ptr())

    def prepare_grads(self) -> None:
        """Transfer already reduced gradients from physical shards to valid optimizer views."""
        self.require_bound()
        for binding in self.bindings:
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
    def clear_padding(self) -> None:
        """Keep padded graph storage zero after updates to the valid region."""
        for binding in self.bindings:
            if binding.layout is not None and binding.model.ndim and binding.local_shape[0] < binding.model.shape[0]:
                binding.model[binding.local_shape[0]:].zero_()

    def clear_grads(self) -> None:
        """Release model-side accumulation buffers as well as optimizer gradients."""
        for binding in self.bindings:
            binding.model.grad = None
            if hasattr(binding.model, "main_grad"):
                binding.model.main_grad = None

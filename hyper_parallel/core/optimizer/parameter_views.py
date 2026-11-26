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
from hyper_parallel.core.optimizer.swap_optimizer import is_swap_optimizer


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

        Every participating rank must bind at the same step before optimizer
        state is initialized. Leaf resets may run collective communication.

        Args:
            optimizer: Chain holding the original model parameters.
            layouts: Parameter layouts and valid local shapes, or None for an unsharded model.
            fp32_main_params: Allocate FP32 master weights for low-precision parameters.
        """
        if self.bound:
            if layouts is not self._layouts:
                raise RuntimeError("Graph optimizer layout changed after binding; rebuild the trainer")
            if layouts is None:
                registered = set(optimizer.model.parameters())
                if any(binding.model not in registered for binding in self.bindings):
                    raise RuntimeError("Graph model parameter changed after optimizer binding; rebuild the trainer")
            return

        leaves = optimizer.chained_optimizers
        if any(is_swap_optimizer(leaf) for leaf in leaves):
            raise ValueError("optimizer.swap.enabled is unsupported with deferred graph optimizer binding")
        if any(leaf.state for leaf in leaves):
            raise RuntimeError("Graph optimizer must bind final shards before initializing optimizer state")

        replacements = self._build_replacements(optimizer, layouts, fp32_main_params)
        reverse = {binding.optimizer: binding.model for _, _, binding in replacements
                   if binding.optimizer is not binding.model}
        self._install_bindings(optimizer, replacements, reverse)
        optimizer.optimizer_param_by_model_param.update(
            (binding.model, binding.optimizer) for _, _, binding in replacements
        )
        self.bindings = [binding for _, _, binding in replacements]
        self._layouts = layouts
        self.bound = True

    def _build_replacements(
        self, optimizer: "ChainedOptimizer", layouts: Optional[dict], fp32_main_params: bool,
    ) -> list[tuple[dict, int, _ParameterBinding]]:
        """Validate optimizer parameters and build views before replacing any group entries."""
        parameter_names = {parameter: name for name, parameter in optimizer.model.named_parameters()}
        seen = set()
        replacements = []
        for leaf in optimizer.chained_optimizers:
            use_dtensor = getattr(leaf, "is_muon", False)
            for group in leaf.param_groups:
                for index, model in enumerate(group["params"]):
                    if model not in parameter_names:
                        label = getattr(model, "model_name", "<unregistered>")
                        raise ValueError(f"Graph optimizer parameter {label!r} is not registered on the model")
                    name = parameter_names[model]
                    if model in seen:
                        raise ValueError(f"Duplicate graph optimizer parameter {name!r}")
                    seen.add(model)
                    binding = self._make_binding(model, name, layouts, use_dtensor, fp32_main_params)
                    replacements.append((group, index, binding))
        return replacements

    @staticmethod
    def _install_bindings(optimizer: "ChainedOptimizer", replacements: list, reverse: dict) -> None:
        """Replace group parameters and rebuild the leaf caches once.

        Reset hooks for graph binding replace parameter-derived caches. Keep the
        original references so a local failure does not need another reset.
        """
        resetting_leaves = [leaf for leaf in optimizer.chained_optimizers
                            if callable(getattr(leaf, "reset_optimizer_parameters", None))]
        snapshots = [(leaf, vars(leaf).copy()) for leaf in resetting_leaves]
        chain_attrs = vars(optimizer).copy()
        try:
            for group, index, binding in replacements:
                group["params"][index] = binding.optimizer
            optimizer.reset_optimizer_parameters(reverse)
        except Exception:
            for leaf, attrs in snapshots:
                vars(leaf).clear()
                vars(leaf).update(attrs)
                # Binding was checked to start with empty optimizer state.
                leaf.state.clear()
            vars(optimizer).clear()
            vars(optimizer).update(chain_attrs)
            for group, index, binding in replacements:
                group["params"][index] = binding.model
            raise

    @staticmethod
    def _make_binding(
        model: nn.Parameter, name: str, layouts: Optional[dict], use_dtensor: bool, fp32_main_params: bool,
    ) -> _ParameterBinding:
        """Build the optimizer view and track valid rows for all optimizer families."""
        layout, shape = None, tuple(model.shape)
        local = model.detach()
        if layouts is not None and (model.requires_grad or model in layouts):
            if model not in layouts:
                raise ValueError(f"Missing graph parameter layout for {name}")
            logical_layout, shape = layouts[model]
            if len(shape) != local.ndim or (
                local.ndim and (tuple(shape[1:]) != tuple(local.shape[1:]) or not 0 <= shape[0] <= local.shape[0])
            ):
                raise ValueError(
                    f"Invalid graph optimizer local shape {shape} for {name}: storage {tuple(local.shape)}"
                )
            if use_dtensor:
                layout = logical_layout
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
        param.model_name = name
        return _ParameterBinding(model, param, layout, tuple(shape), owns_storage, model.data_ptr())

    def prepare_grads(self) -> None:
        """Transfer already reduced gradients from physical shards to valid optimizer views."""
        self.require_bound()
        for binding in self.bindings:
            if binding.model.data_ptr() != binding.storage_ptr:
                raise RuntimeError(
                    f"Graph parameter storage changed after optimizer binding: {binding.model.model_name}; "
                    "rebuild the trainer"
                )
        for binding in self.bindings:
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

    @torch.no_grad()
    def clear_padding(self) -> None:
        """Keep padding zero in graph storage and in any full-shard main weights."""
        for binding in self.bindings:
            if binding.model.ndim and binding.local_shape[0] < binding.model.shape[0]:
                valid_rows = binding.local_shape[0]
                binding.model[valid_rows:].zero_()
                if binding.owns_storage and binding.layout is None:
                    binding.optimizer[valid_rows:].zero_()

    @torch.no_grad()
    def clear_grads(self, set_to_none: bool) -> None:
        """Apply the requested reset to graph-side gradient accumulation buffers."""
        for binding in self.bindings:
            model = binding.model
            for name in ("grad", "main_grad"):
                if name == "grad" and model is binding.optimizer:
                    continue  # Already cleared by the leaf optimizer.
                grad = getattr(model, name, None)
                if grad is None:
                    continue
                if set_to_none:
                    setattr(model, name, None)
                else:
                    if grad.requires_grad:
                        grad = grad.detach()
                        setattr(model, name, grad)
                    grad.zero_()
            if hasattr(model, "grad_added_to_main_grad"):
                model.grad_added_to_main_grad = False

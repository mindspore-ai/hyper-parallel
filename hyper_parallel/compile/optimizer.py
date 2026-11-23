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
from typing import Optional

import torch
from torch import nn

from hyper_parallel.components.optim.mixed_precision_optimizer import (
    _ParameterBinding,
    MixedPrecisionOptimizer,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


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
        super().__init__(optimizer, model, fp32_main_params=fp32_main_params, defer_binding=True)
        self._layout_source = None

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
        bindings = []
        for leaf in self.chained_optimizers:
            for group in leaf.param_groups:
                for index, model_param in enumerate(group["params"]):
                    layout, local_shape = (None, tuple(model_param.shape))
                    if getattr(leaf, "is_muon", False) and parameter_layouts is not None:
                        if model_param not in parameter_layouts:
                            raise ValueError(f"Missing graph parameter layout for {model_param.model_name}")
                        layout, local_shape = parameter_layouts[model_param]
                    binding = self._bind_parameter(model_param, layout, tuple(local_shape))
                    group["params"][index] = binding.optimizer
                    bindings.append(binding)
                    self.optimizer_param_by_model_param[model_param] = binding.optimizer
        self._register_parameter_bindings(bindings)
        self.optimizer.reset_optimizer_parameters(self.model_param_by_optimizer_param)
        self.param_groups = self.optimizer.param_groups
        self._layout_source = parameter_layouts
        self._bound = True

    def _bind_parameter(self, model_param: nn.Parameter, layout: Optional[Layout],
                        local_shape: tuple[int, ...]) -> _ParameterBinding:
        """Describe one final graph shard, retaining its physical storage guard."""
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
        return _ParameterBinding(model_param, optimizer_param, layout, local_shape,
                                 owns_storage, model_param.data_ptr())

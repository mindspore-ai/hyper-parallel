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
"""Portable expert execution with the same parameter and transport contracts as grouped GEMM."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
    match_grouped_experts_inputs,
)
from hyper_parallel.core.dtensor.dtensor import DTensor


class TorchGroupedExperts(GroupedExpertsModule):
    """Execute source-ordered expert Linear operations without NPU-specific kernels."""

    def __init__(
        self,
        *,
        config: GroupedExpertsConfig | None = None,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        activation_dtype: str = "input", swiglu_limit: float = 0.0,
    ) -> None:
        """Retain the source's actual eager/grouped-Torch compute and accumulation policy."""
        source_config = getattr(module, "config", config)
        self.torch_implementation = getattr(module, "torch_implementation", None)
        if self.torch_implementation is None:
            self.torch_implementation = getattr(source_config, "_experts_implementation", "eager") or "eager"
        if self.torch_implementation not in ("eager", "grouped_mm"):
            raise ValueError("Torch experts support source eager or grouped_mm execution")
        super().__init__(config=config, module=module, module_fqn=module_fqn, context=context,
                         activation_dtype=activation_dtype, swiglu_limit=swiglu_limit)

    @staticmethod

    def _torch_grouped_mm(inputs: Tensor, weights: Tensor, offsets: Tensor) -> Tensor:
        """Use the native Torch grouped kernel selected by the source implementation."""
        kernel = getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)
        if kernel is None:
            raise NotImplementedError("grouped_mm source requires a Torch grouped-MM kernel")
        return kernel(inputs.to(weights.dtype), weights, offs=offsets)

    def _grouped_forward(self, hidden: Tensor, ids: Tensor, weights: Tensor) -> Tensor:
        """Retain token-major sorting, native grouped GEMMs and top-k reduction semantics."""
        experts, order = torch.sort(ids.reshape(-1))
        top_k = ids.shape[1]
        selected = hidden[order // top_k]
        probabilities = weights.reshape(-1)[order]
        histc_ids = experts.float() if experts.device.type in ("cpu", "mps") else experts.int()
        counts = torch.histc(histc_ids, bins=self.config.num_experts, min=0, max=self.config.num_experts - 1)
        offsets = counts.cumsum(dim=0, dtype=torch.int32)
        gate_up = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        gate, up = self._torch_grouped_mm(selected, gate_up.transpose(-2, -1), offsets).chunk(2, dim=-1)
        activated = self.apply_activation(gate, up, hidden.dtype)
        if self.config.routing_weight_mode == "before_fc2":
            activated = (activated * probabilities[:, None]).to(activated.dtype)
        output = self._torch_grouped_mm(activated, down.transpose(-2, -1), offsets)
        if self.config.routing_weight_mode == "output":
            output = output * probabilities[:, None]
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return output[inverse].view(hidden.shape[0], top_k, self.config.hidden_size).sum(dim=1).to(hidden.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor | None = None,
        topk_weights: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """Retain separate expert autograd branches and sequential source accumulation."""
        topk_ids, topk_weights = match_grouped_experts_inputs(topk_ids, topk_weights, kwargs)
        shape = tuple(hidden_states.shape)
        hidden = self.validate_and_flatten_inputs(hidden_states, topk_ids, topk_weights)
        ids = topk_ids.to(torch.int64)
        # Fixed-size scatter rejects out-of-range IDs on device, without a host scalar read.
        ids.new_zeros(self.config.num_experts).scatter_add_(0, ids.reshape(-1), torch.ones_like(ids).reshape(-1))
        if self.torch_implementation == "grouped_mm":
            return self._grouped_forward(hidden, ids, topk_weights).reshape(shape)
        gate_up = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        output = torch.zeros_like(hidden)
        for expert in range(self.config.num_experts):
            slots, tokens = torch.where(ids.transpose(0, 1) == expert)
            if tokens.numel() == 0:
                continue
            gate, up = F.linear(hidden[tokens], gate_up[expert]).chunk(2, dim=-1)  # pylint: disable=not-callable
            activated = self.apply_activation(gate, up, hidden.dtype)
            weights = topk_weights[tokens, slots, None]
            if self.config.routing_weight_mode == "before_fc2":
                activated = (activated * weights).to(activated.dtype)
            values = F.linear(activated, down[expert])  # pylint: disable=not-callable
            if self.config.routing_weight_mode == "output":
                values = values * weights
            output.index_add_(0, tokens, values.to(output.dtype))
        return output.reshape(shape)

    def permute(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Sort top-k-major slots by expert, retaining each expert's source row order."""
        grouped = self.torch_implementation == "grouped_mm"
        ids = (topk_ids.reshape(-1) if grouped else topk_ids.transpose(0, 1).reshape(-1)).to(torch.int64)
        counts = ids.new_zeros(self.config.num_experts).scatter_add_(0, ids, torch.ones_like(ids))
        order = ids.argsort(stable=not grouped)
        token_ids = (
            order // topk_ids.shape[1]
            if grouped
            else torch.arange(hidden_states.shape[0], device=ids.device).repeat(topk_ids.shape[1])[order]
        )
        weights = (topk_weights.reshape(-1) if grouped else topk_weights.transpose(0, 1).reshape(-1))[order]
        return (
            hidden_states[token_ids],
            ids[order],
            counts,
            weights if self.config.routing_weight_mode == "before_fc2" else None,
            torch.stack((order, ids[order])),
        )

    def compute_experts(
        self,
        hidden_states: Tensor,
        tokens_per_expert: Tensor,
        gate_up_proj: Tensor,
        down_proj: Tensor,
        routing_weights: Tensor | None = None,
    ) -> Tensor:
        """Apply complete local expert matrices; EP retains ownership of dispatch and combine."""
        gate_up = gate_up_proj.to_local() if isinstance(gate_up_proj, DTensor) else gate_up_proj
        down = down_proj.to_local() if isinstance(down_proj, DTensor) else down_proj
        if (
            gate_up.ndim != 3
            or gate_up.shape[1:] != (2 * self.config.intermediate_size, self.config.hidden_size)
            or down.shape != (gate_up.shape[0], self.config.hidden_size, self.config.intermediate_size)
            or tokens_per_expert.shape != (gate_up.shape[0],)
        ):
            raise ValueError("Torch experts require complete [E,2I,H] and [E,H,I] expert matrices")
        if self.torch_implementation == "grouped_mm" and hidden_states.shape[0]:
            offsets = tokens_per_expert.cumsum(dim=0, dtype=torch.int32)
            gate, up = self._torch_grouped_mm(hidden_states, gate_up.transpose(-2, -1), offsets).chunk(2, dim=-1)
            activated = self.apply_activation(gate, up, hidden_states.dtype)
            if self.config.routing_weight_mode == "before_fc2":
                if routing_weights is None:
                    raise ValueError("before_fc2 requires differentiable slot routing weights")
                activated = (activated * routing_weights[:, None]).to(activated.dtype)
            return self._torch_grouped_mm(activated, down.transpose(-2, -1), offsets)
        if not hidden_states.shape[0]:
            gate, up = F.linear(hidden_states, gate_up[0]).chunk(2, dim=-1)  # pylint: disable=not-callable
            # Torch exposes this runtime callable through a native or generated descriptor.
            return F.linear(  # pylint: disable=not-callable
                self.apply_activation(gate, up, hidden_states.dtype), down[0]
            )  # pylint: disable=not-callable
        expert_ids = torch.repeat_interleave(
            torch.arange(gate_up.shape[0], device=hidden_states.device),
            tokens_per_expert,
            output_size=hidden_states.shape[0],
        )
        output = torch.zeros_like(hidden_states)
        for expert in range(gate_up.shape[0]):
            rows = torch.where(expert_ids == expert)[0]
            if rows.numel() == 0:
                continue
            gate, up = F.linear(hidden_states[rows], gate_up[expert]).chunk(2, dim=-1)  # pylint: disable=not-callable
            activated = self.apply_activation(gate, up, hidden_states.dtype)
            if self.config.routing_weight_mode == "before_fc2":
                if routing_weights is None:
                    raise ValueError("before_fc2 requires differentiable slot routing weights")
                activated = (activated * routing_weights[rows, None]).to(activated.dtype)
            output = output.index_copy(0, rows, F.linear(activated, down[expert]))  # pylint: disable=not-callable
        return output

    def unpermute(
        self,
        values: Tensor,
        unpermute_indices: Tensor,
        topk_weights: Tensor,
        output_shape: tuple[int, ...],
    ) -> Tensor:
        """Accumulate in expert-major order, matching the source expert loop's rounding order."""
        order, expert_ids = unpermute_indices.unbind(0)
        if self.torch_implementation == "grouped_mm":
            probabilities = topk_weights.reshape(-1)[order]
            dtype = values.dtype
            if self.config.routing_weight_mode == "output":
                values = values * probabilities[:, None]
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=order.device)
            output = values[inverse].view(*topk_weights.shape, self.config.hidden_size).sum(dim=1)
            return output.to(dtype).reshape(output_shape)
        weights = topk_weights.transpose(0, 1).reshape(-1)[order]
        if self.config.routing_weight_mode == "output":
            values = (values * weights[:, None]).to(values.dtype)
        tokens = topk_weights.shape[0]
        token_ids = order.remainder(tokens) if tokens else order
        output = values.new_zeros(tokens, self.config.hidden_size)
        # Each expert updates the accumulation before the next expert starts.
        # A single scatter across all experts changes BF16 addition order on NPU.
        for expert in range(self.config.num_experts):
            rows = torch.where(expert_ids == expert)[0]
            if rows.numel():
                output = output.index_add(0, token_ids[rows], values[rows])
        return output.reshape(output_shape)


class EPTorchGroupedExperts(TorchGroupedExperts, EPGroupedExperts):
    """Use portable expert mathematics with the existing EP dispatch/combine and binding stages."""

    def forward(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor | None = None,
        topk_weights: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """Execute the shared staged protocol so EP always dispatches and combines slots."""
        return GroupedExpertsModule.forward(self, hidden_states, topk_ids, topk_weights, **kwargs)

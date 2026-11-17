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
"""Reserved MoonEP planning, transport and temporary expert-weight lifecycle."""

from __future__ import annotations

from typing import Any, Literal

from torch import Tensor


class MoonEPContext:
    """Borrow a TP-extend-EP mesh while keeping parameter ownership fixed.

    Runtime plans must be invocation-local, distinguish logical expert IDs
    from compute-table rows, and be replayable during activation recomputation.
    This scaffold allocates no communication buffers or temporary Parameters.
    """

    def __init__(self, mesh: Any, *, num_experts: int) -> None:
        """Record existing resources without creating process groups.

        Args:
            mesh: Existing one-dimensional expert mesh, including dense TP ranks.
            num_experts: Global home expert count, divisible by the EP mesh size.
        """
        if mesh is None or mesh.ndim != 1:
            raise ValueError("MoonEPContext requires an existing one-dimensional HP expert mesh")
        if num_experts < 1 or num_experts % mesh.size():
            raise ValueError("num_experts must be positive and divisible by EP size")
        self.mesh = mesh
        self.num_experts = num_experts
        self.local_experts = num_experts // mesh.size()
        self.rank = mesh.get_local_rank()

    def plan(self, hidden_states: Tensor, topk_ids: Tensor, topk_weights: Tensor) -> object:
        """Reserve a per-call allocation/transport plan from original token routes.

        Args:
            hidden_states: Original tokens [T,H].
            topk_ids: Global model-selected expert IDs [T,K].
            topk_weights: Final differentiable routing probabilities [T,K].

        Returns:
            Opaque plan mapping compute rows to home experts, retained through backward/recompute.
        """
        raise NotImplementedError("MoonEP load planning and invocation replay are not implemented")

    def dispatch(
        self, hidden_states: Tensor, topk_ids: Tensor, topk_weights: Tensor, plan: object,
        *, routing_weight_mode: Literal["output", "before_fc2"] = "before_fc2",
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Reserve expert-major dispatch in the plan's compute-table order.

        Args:
            hidden_states: Original tokens [T,H]; do not expand slots twice.
            topk_ids: Global model-selected expert IDs [T,K].
            topk_weights: Final differentiable routing probabilities [T,K].
            plan: This invocation's saved allocation and transport plan.
            routing_weight_mode: Probability placement from the module's algorithm configuration.

        Returns:
            Expert-major activations [R,H], counts [C], optional probabilities [R].
            Preserve router gradients and leave output-mode weighting to combine.
        """
        raise NotImplementedError("MoonEP token dispatch and differentiable transport are not implemented")

    def prefetch_weights(self, gate_up_proj: Tensor, down_proj: Tensor, plan: object) -> tuple[Tensor, Tensor]:
        """Reserve complete local/replica weight views for the shared expert math.

        Args:
            gate_up_proj: Current home-local FC1 weights, read after FSDP unshard hooks.
            down_proj: Current home-local FC2 weights with matching expert rows.
            plan: Per-call compute-row mapping; never a cross-step parameter cache.

        Returns:
            Weights [C,H,2I] / [C,I,H]; temporary rows must return gradients to their home owners.
        """
        raise NotImplementedError("MoonEP weight prefetch and temporary compute views are not implemented")

    def combine(
        self, values: Tensor, topk_weights: Tensor, output_shape: tuple[int, ...], plan: object,
        *, routing_weight_mode: Literal["output", "before_fc2"] = "before_fc2",
    ) -> Tensor:
        """Reserve return transport and exactly one token restoration/top-k sum.

        Args:
            values: Expert outputs in the plan's compute-slot order.
            topk_weights: Original token/top-k probabilities [T,K].
            output_shape: Original hidden-state shape [...,H].
            plan: This invocation's inverse transport information.
            routing_weight_mode: Probability placement from the module's algorithm configuration.

        Returns:
            Routed token output in output_shape, with probabilities applied once.
            This token-level result must not pass through ordinary unpermute again.
        """
        raise NotImplementedError("MoonEP output combine and router-weight backward are not implemented")

    def reduce_grad(self, gate_up_grad: Tensor, down_grad: Tensor, plan: object) -> tuple[Tensor, Tensor]:
        """Reserve autograd-owned replica gradient return before FSDP reduction.

        Args:
            gate_up_grad: Compute-table FC1 gradients [C,H,2I], including temporary rows.
            down_grad: Compute-table FC2 gradients [C,I,H], including temporary rows.
            plan: The same invocation's home mapping, kept alive through backward.

        Returns:
            Home-local gradients with replica contributions summed exactly once.
            The future autograd bridge must complete this before framework gradient reduction.
        """
        raise NotImplementedError("MoonEP replica gradient return and FSDP lifecycle integration are not implemented")

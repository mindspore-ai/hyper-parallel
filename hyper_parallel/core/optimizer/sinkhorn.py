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

"""Momentum updates with distributed Sinkhorn balancing (DeepSeek V4.1, Algorithm 1)."""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.dtensor_compat import is_dtensor, to_local_if_dtensor


def _sum_across(value: torch.Tensor, groups: Tuple[Any, ...]) -> torch.Tensor:
    """Reduce statistics only over the mesh axes partitioning the reduced dimension."""
    for group in groups:
        if value.numel():
            dist.all_reduce(value, group=group)
    return value


def _distributed_norm(
        update: torch.Tensor, dimension: int, groups: Tuple[Any, ...],
) -> torch.Tensor:
    """Reduce squared local norms without materializing a matrix-sized square."""
    norm = torch.linalg.vector_norm(update, dim=dimension, keepdim=True)
    if groups:
        _sum_across(norm.square_(), groups).sqrt_()
    return norm


def _validate_balancing(steps: int, eps: float, tau: float) -> None:
    """Validate algorithm constants before any collective is launched."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0 or steps % 2 != 1:
        raise ValueError("steps must be a positive odd integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")


class Sinkhorn(torch.optim.Optimizer):
    """Nesterov momentum with row/column balancing and no weight decay.

    Gradients must be dense, fully reduced, and consistently present or absent
    across all shards of a matrix. Statistics use float32; momentum is stored as
    a parameter-shaped float32 tensor, preserving DTensor placement metadata.
    """

    def __init__(
            self,
            params: Any,
            lr: float = 1e-3,
            momentum: float = 0.95,
            nesterov: bool = True,
            steps: int = 11,
            eps: float = 1e-20,
            tau: float = 1e-3,
            correction: float = 0.18,
    ) -> None:
        """Initialize independent Sinkhorn parameter groups and momentum state.

        Args:
            params: Matrices or parameter groups with per-group overrides.
            lr: Base learning rate, before correction.
            momentum: Exponential moving-average coefficient.
            nesterov: Use Algorithm 1's Nesterov update; otherwise balance the EMA.
            steps: Positive odd count of normalization steps.
            eps: Positive L2-norm stabilizer.
            tau: Near-zero row masking threshold.
            correction: Learning-rate multiplier matching Adam update magnitude.
        """
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "steps": steps,
                    "eps": eps, "tau": tau, "correction": correction}
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)
            for param in group["params"]:
                self._validate_parameter(param)

    @staticmethod
    def _validate_parameter(param: torch.Tensor) -> None:
        if param.ndim != 2 or min(param.shape) <= 0:
            raise ValueError("Sinkhorn parameters must be non-empty 2D matrices")
        for placement in getattr(param, "placements", ()):
            if not (placement.is_shard() or placement.is_replicate()):
                raise ValueError("Sinkhorn parameters require Shard or Replicate placements")

    @staticmethod
    def _norm_groups(param: torch.Tensor) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Read current parameter layout without parameter-identity caches."""
        groups = ([], [])
        for axis, placement in enumerate(getattr(param, "placements", ())):
            if placement.is_shard():
                groups[placement.dim % 2].append(param.device_mesh.get_group(axis))
        return tuple(groups[0]), tuple(groups[1])

    @staticmethod
    def _validate_group(group: Dict[str, Any]) -> None:
        """Validate parameter-group overrides as well as constructor defaults."""
        _validate_balancing(group["steps"], group["eps"], group["tau"])
        for key in ("lr", "correction"):
            if not math.isfinite(group[key]) or group[key] < 0:
                raise ValueError(f"{key} must be finite and non-negative")
        if not math.isfinite(group["momentum"]) or not 0 <= group["momentum"] < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
        if group.get("weight_decay", 0) != 0:
            raise ValueError("Sinkhorn does not apply weight decay")

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore float32 momentum even when parameters are low precision.

        Args:
            state_dict: Serialized parameter groups and optimizer state.
        """
        super().load_state_dict(state_dict)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for saved_id, param in zip(saved_group["params"], group["params"]):
                momentum = state_dict["state"].get(saved_id, {}).get("momentum_buffer")
                if momentum is not None:
                    self.state[param]["momentum_buffer"] = momentum.to(
                        device=param.device, dtype=torch.float32,
                    ).clone()

    @staticmethod
    def _new_momentum(param: torch.Tensor) -> torch.Tensor:
        """Keep state layout metadata even inside SkipDTensorDispatch."""
        local = torch.zeros_like(to_local_if_dtensor(param), dtype=torch.float32)
        if is_dtensor(param):
            return type(param).from_local(local, param.device_mesh, param.placements,
                                          shape=tuple(param.shape), stride=(param.shape[1], 1))
        return local

    def _update_parameter(self, param: torch.Tensor, group: Dict[str, Any]) -> None:
        """Update one local shard without materializing the full matrix."""
        if param.grad is None:
            return
        self._validate_parameter(param)
        gradient = param.grad
        if hasattr(gradient, "placements") and any(place.is_partial() for place in gradient.placements):
            raise ValueError("Sinkhorn gradients must be reduced before optimizer.step()")
        gradient = to_local_if_dtensor(gradient)
        if gradient.is_sparse:
            raise ValueError("Sinkhorn does not support sparse gradients")
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = self._new_momentum(param)
        buffer = to_local_if_dtensor(state["momentum_buffer"])
        # Explicit allocation avoids same-dtype .to(copy=True) aliasing on NPU.
        # Reuse this workspace for Nesterov and balancing, preserving grad and EMA state.
        update = torch.empty_like(gradient, dtype=torch.float32).copy_(gradient)
        beta = group["momentum"]
        buffer.mul_(beta).add_(update, alpha=1 - beta)
        if group["nesterov"]:
            update.mul_(1 - beta).add_(buffer, alpha=beta)
        else:
            update.copy_(buffer)
        row_groups, column_groups = self._norm_groups(param)
        rows, columns = param.shape
        eps = group["eps"]
        row_norm = _distributed_norm(update, 1, column_groups)
        mean_norm = _sum_across(row_norm.sum(), row_groups) / rows
        update.masked_fill_(row_norm <= group["tau"] * mean_norm, 0)
        # Masking changes only rows that remain zero, so the initial norms can be reused.
        update.div_(row_norm.add_(eps))
        for _ in range((group["steps"] - 1) // 2):
            column_norm = _distributed_norm(update, 0, row_groups)
            update.div_(column_norm.add_(eps))
            row_norm = _distributed_norm(update, 1, column_groups)
            update.div_(row_norm.add_(eps))
        update.mul_(math.sqrt(columns))
        to_local_if_dtensor(param).add_(update, alpha=-group["lr"] * group["correction"])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Any:
        """Apply balancing independently in each replica using shard statistics.

        Args:
            closure: Optional callable reevaluating the training loss.

        Returns:
            Closure loss when supplied, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            self._validate_group(group)
            group["step"] = (group.get("step") or 0) + 1
            for param in group["params"]:
                self._update_parameter(param, group)
        return loss

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
"""Streaming micro-batch or optimizer-step token normalization."""

from __future__ import annotations

from typing import Iterable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.utils.communication import differentiable_all_reduce
from hyper_parallel.distributed.mesh import MeshContext


def _sum(value: torch.Tensor, mesh) -> torch.Tensor:
    """Reduce a detached scalar without accidentally using the world group."""
    if mesh is None or mesh.size() == 1:
        return value
    return differentiable_all_reduce(value, "sum", mesh.get_group())


class StepLossNormalization:
    """Normalize local mean losses without retaining micro-batch inputs.

    Counts describe CP-local targets and are replicated across TP ranks, as in
    TextParallelBatch. Both modes accept only the main LM loss, as a scalar or
    ``foundation_loss``. Models must not fold MTP or auxiliary objectives into
    that scalar; their independent normalization is not supported.
    """

    def __init__(self, mode: str, micro_steps: int, mesh: MeshContext) -> None:
        """Create fresh scalar accounting for one optimizer step."""
        if mode not in ("microbatch", "token"):
            raise ValueError("loss_normalization must be 'microbatch' or 'token'")
        self.mode = mode
        self.micro_steps = micro_steps
        self.mesh = mesh
        self.token_count = None
        self.loss_names = set()

    def reduce(
        self, losses: dict[str, torch.Tensor] | torch.Tensor, counts: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return global reporting values with locally weighted backward edges.

        Args:
            losses: CP-local main LM mean, optionally keyed by ``foundation_loss``.
            counts: Effective local target counts keyed by ``<family>_tokens``.
        """
        if isinstance(losses, torch.Tensor):
            losses = {"foundation_loss": losses}
        if set(losses) != {"foundation_loss"}:
            raise ValueError(
                "microbatch/token normalization supports only foundation_loss (main LM loss); "
                "MTP and auxiliary losses are not supported. Use legacy for existing model-specific objectives."
            )
        if self.mode == "token":
            loss_device = losses["foundation_loss"].device
            count = counts["foundation_tokens"].detach().to(loss_device)
            self.token_count = count if self.token_count is None else self.token_count + count
            self.loss_names.update(losses)

        result = {}
        for key, mean in losses.items():
            family = key.split("_loss", maxsplit=1)[0]
            count = counts[f"{family}_tokens"].detach().to(device=mean.device, dtype=torch.float32)
            # Empty shards carry no loss weight; their loss function must still have a finite backward.
            numerator = torch.where(count > 0, mean.float(), 0.0) * count
            if self.mode == "token":
                backward = numerator
                reported = _sum(numerator.detach(), self.mesh.dp_cp_mesh)
            else:
                denominator = _sum(count, self.mesh.cp_mesh).clamp_min(1) * self.micro_steps
                local = numerator / denominator
                # FSDP averages across DP x CP; CP partitions belong to one micro-batch.
                backward = local * self.mesh.cp_size
                reported = _sum(local.detach(), self.mesh.dp_cp_mesh) / self.mesh.dp_size
            result[key] = reported + (backward - backward.detach())
        return result

    @torch.no_grad()
    def finish(
        self, parameters: Iterable[torch.nn.Parameter], loss: float, metrics: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Normalize accumulated gradients before clipping and optimizer updates.

        Args:
            parameters: Parameters whose reduced gradients will be updated.
            loss: Sum of the reported micro-batch losses.
            metrics: Step metrics; only actual loss keys are divided.
        """
        if self.mode == "microbatch":
            return loss, metrics
        if self.token_count is None:
            raise RuntimeError("token normalization has not seen a micro-batch")
        denominator = _sum(self.token_count, self.mesh.dp_cp_mesh).clamp_min(1)
        factor = (self.mesh.dp_size * self.mesh.cp_size) / denominator.float()
        seen = set()
        with SkipDTensorDispatch():
            for parameter in parameters:
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                # Match the optimizer: FSDP may store FP32 gradients only in main_grad.
                grad = getattr(parameter, "main_grad", None)
                if grad is None:
                    grad = parameter.grad
                if grad is None:
                    continue
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                grad.mul_(factor.to(device=grad.device, dtype=grad.dtype))
        divisor = denominator.item()
        return loss / divisor, {
            key: value / divisor if key in self.loss_names else value for key, value in metrics.items()
        }

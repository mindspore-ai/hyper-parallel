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
"""Reusable auxiliary-loss functions."""

from typing import Any

import torch  # pylint: disable=forbidden-backend-import


class _AuxLossAutoScaler(torch.autograd.Function):
    """Inject auxiliary-loss gradients without changing the forward loss value."""

    main_loss_backward_scale = torch.tensor(1.0)

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        """Save the auxiliary loss and return the output unchanged."""
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass through the output gradient and inject the auxiliary-loss gradient."""
        (aux_loss,) = ctx.saved_tensors
        aux_loss_grad = torch.ones_like(aux_loss) * _AuxLossAutoScaler.main_loss_backward_scale
        return grad_output, aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the auxiliary-loss gradient scale."""
        _AuxLossAutoScaler.main_loss_backward_scale = scale


def aux_loss_auto_scale(output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
    """Attach an auxiliary-loss gradient to an unchanged forward tensor.

    Args:
        output: Main forward output.
        aux_loss: Scalar auxiliary loss whose gradient should be injected.

    Returns:
        ``output`` unchanged in value, with an autograd edge to ``aux_loss``.
    """
    return _AuxLossAutoScaler.apply(output, aux_loss)


def set_aux_loss_scale(scale: torch.Tensor) -> None:
    """Set the gradient multiplier used by :func:`aux_loss_auto_scale`.

    Args:
        scale: Tensor used to scale the injected auxiliary-loss gradient.
    """
    _AuxLossAutoScaler.set_loss_scale(scale)

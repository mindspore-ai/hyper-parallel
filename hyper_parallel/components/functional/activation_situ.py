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
"""SiTU eager reference and first-order autograd over CANN ops-nn."""

import importlib
from collections.abc import Callable
from typing import Any, Optional

# Torch is the native framework; the legacy platform-agnostic lint rule is obsolete here.
# pylint: disable=forbidden-backend-import
import torch
from torch.autograd.function import once_differentiable


def _situ_eager(gate: torch.Tensor, up: torch.Tensor, beta: float,
                linear_beta: Optional[float]) -> torch.Tensor:
    gate_float, up_float = gate.float(), up.float()
    activated_gate = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    if linear_beta:
        up_float = linear_beta * torch.tanh(up_float / linear_beta)
    return (activated_gate * up_float).to(gate.dtype)


def _load_situ_ops() -> tuple[Callable, Callable]:
    """Resolve the optional ops-nn training APIs without hiding native errors."""
    try:
        # CANN is optional for CPU and eager-only callers.
        ops = importlib.import_module("cann_ops_nn")
    except (ImportError, OSError) as error:
        raise ImportError(
            "SiTU backend='ascendc' could not load cann_ops_nn. Install the standalone ops-nn Python package "
            "with compatible PyTorch, torch-npu and CANN libraries. "
            f"See hyper_parallel/components/functional/README.md. Original error: {error}"
        ) from error
    situ_glu = getattr(ops, "situ_glu", None)
    situ_glu_grad = getattr(ops, "situ_glu_grad", None)
    if not callable(situ_glu) or not callable(situ_glu_grad):
        raise ImportError(
            "SiTU backend='ascendc' requires cann_ops_nn.situ_glu and cann_ops_nn.situ_glu_grad. "
            "Install a package containing both training APIs and build both native operators. "
            "See hyper_parallel/components/functional/README.md."
        )
    return situ_glu, situ_glu_grad


class _SituAutograd(torch.autograd.Function):
    """Connect ops-nn's separate SituGlu and SituGluGrad entry points."""

    @staticmethod
    def forward(ctx: Any, packed: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
        """Run dense gate-first rows and retain the native backward inputs."""
        situ_glu, situ_glu_grad = _load_situ_ops()
        rows = packed.reshape(-1, packed.shape[-1]).contiguous()
        ctx.save_for_backward(rows)
        ctx.input_shape = packed.shape
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        ctx.situ_glu_grad = situ_glu_grad
        output = situ_glu(rows, dim=-1, beta=beta, linear_beta=linear_beta, activate_left=True)
        return output.reshape(*packed.shape[:-1], packed.shape[-1] // 2)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Restore the packed gradient with the same saturation parameters."""
        (rows,) = ctx.saved_tensors
        grad_rows = grad_output.reshape(-1, rows.shape[-1] // 2).contiguous()
        gradient = ctx.situ_glu_grad(
            grad_rows, rows, dim=-1, beta=ctx.beta, linear_beta=ctx.linear_beta, activate_left=True,
        )
        return gradient.reshape(ctx.input_shape), None, None


def _situ_fused(packed: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    return _SituAutograd.apply(packed, beta, linear_beta)


def situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    backend: str = "eager",
) -> torch.Tensor:
    """Apply SiTU to separate gate and up projections.

    Args:
        gate: Gate projection with at least one dimension.
        up: Up projection with the same shape, dtype, and device as gate.
        beta: Positive gate saturation scale.
        linear_beta: Up saturation scale; zero or None disables saturation.
        backend: ``eager`` or ``ascendc`` (the fused ops-nn implementation).

    Returns:
        Activated values with gate's shape and dtype.
    """
    # Packing would lose the original split if gate and up had unequal widths.
    if gate.shape != up.shape:
        raise ValueError("SiTU gate and up must have identical shapes")
    if backend == "eager":
        return _situ_eager(gate, up, beta, linear_beta)
    return situ_packed(torch.cat((gate, up), dim=-1), beta, linear_beta, backend)


def situ_packed(
    packed: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    backend: str = "eager",
) -> torch.Tensor:
    """Apply SiTU to a tensor with gate channels followed by up channels.

    Args:
        packed: Tensor of shape [..., 2H]; noncontiguous inputs are supported.
        beta: Positive gate saturation scale.
        linear_beta: Up saturation scale; zero or None disables saturation.
        backend: ``eager`` uses FP32 intermediates. ``ascendc`` calls ops-nn
            on NPU with first-order autograd; native errors propagate unchanged.

    Raises:
        ImportError: The optional CANN package cannot load or lacks a training API.

    Returns:
        Tensor of shape [..., H] in the input dtype. Empty inputs keep connected
        gradients through tensor operations without launching a native kernel.
    """
    if backend not in ("eager", "ascendc"):
        raise ValueError("SiTU backend must be 'eager' or 'ascendc'")
    linear_beta = 0.0 if linear_beta is None else linear_beta
    if backend == "eager" or packed.numel() == 0:
        gate, up = packed.unflatten(-1, (2, packed.shape[-1] // 2)).unbind(-2)
        return _situ_eager(gate, up, beta, linear_beta)
    return _situ_fused(packed, beta, linear_beta)

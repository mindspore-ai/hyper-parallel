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
"""Token-chunked recomputation for packed, bias-free SwiGLU MLPs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# This module is an explicitly PyTorch/NPU high-performance component.
# pylint: disable=forbidden-backend-import,not-callable
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from hyper_parallel.components.functional import swiglu, swiglu_backward
from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models.replacement import module_replacement


def _require_grouped_matmul_add() -> Any:
    """Return the optional torch-npu FP32 weight-gradient accumulator."""
    try:
        import torch_npu  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "Chunked SwiGLU weight gradients require "
            "torch_npu.npu_grouped_matmul_add_."
        ) from exc
    grouped_matmul_add = getattr(torch_npu, "npu_grouped_matmul_add_", None)
    if grouped_matmul_add is None:
        raise RuntimeError(
            "The installed torch-npu does not provide "
            "npu_grouped_matmul_add_; upgrade torch-npu/CANN or disable "
            "Chunked SwiGLU."
        )
    return grouped_matmul_add


def _accumulate_dweight(
    grouped_matmul_add: Any,
    accumulator: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    group_list: torch.Tensor,
) -> None:
    """Accumulate ``left.T @ right`` directly into an FP32 buffer."""
    grouped_matmul_add(
        accumulator,
        left,
        right,
        group_list,
        transpose_x=True,
        transpose_weight=False,
        group_type=2,
    )


class _ChunkedSwiGLU(torch.autograd.Function):
    """Recompute packed SwiGLU token chunks in backward."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Evaluate packed SwiGLU without retaining block intermediates."""
        ctx.chunk_size = chunk_size
        ctx.input_shape = inputs.shape
        ctx.save_for_backward(inputs, gate_up_weight, down_weight)

        flat_inputs = inputs.reshape(-1, inputs.size(-1))
        flat_output = inputs.new_empty((flat_inputs.size(0), down_weight.size(0)))
        for start in range(0, flat_inputs.size(0), chunk_size):
            end = min(start + chunk_size, flat_inputs.size(0))
            gate_up = F.linear(flat_inputs[start:end], gate_up_weight)
            mixed = swiglu(gate_up, dim=-1)
            flat_output[start:end].copy_(F.linear(mixed, down_weight))
        return flat_output.reshape(*inputs.shape[:-1], down_weight.size(0))

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor | None) -> tuple:
        """Recompute each block and return only the requested gradients."""
        if grad_output is None:
            return None, None, None, None

        need_input, need_gate_up, need_down, _ = ctx.needs_input_grad
        inputs, gate_up_weight, down_weight = ctx.saved_tensors
        flat_inputs = inputs.reshape(-1, inputs.size(-1))
        flat_grad_output = grad_output.reshape(-1, grad_output.size(-1))
        flat_grad_inputs = torch.empty_like(flat_inputs) if need_input else None
        grad_gate_up_accumulator = (
            torch.zeros_like(gate_up_weight, dtype=torch.float32)
            if need_gate_up
            else None
        )
        grad_down_accumulator = (
            torch.zeros_like(down_weight, dtype=torch.float32)
            if need_down
            else None
        )
        grouped_matmul_add = (
            _require_grouped_matmul_add()
            if need_gate_up or need_down
            else None
        )
        group_lists = {
            size: torch.tensor([size], device=inputs.device, dtype=torch.int64)
            for size in (ctx.chunk_size, flat_inputs.size(0) % ctx.chunk_size)
            if size > 0
        }

        for start in range(0, flat_inputs.size(0), ctx.chunk_size):
            end = min(start + ctx.chunk_size, flat_inputs.size(0))
            input_chunk = flat_inputs[start:end]
            grad_output_chunk = flat_grad_output[start:end]
            gate_up = F.linear(input_chunk, gate_up_weight)
            group_list = group_lists[end - start]

            if grad_down_accumulator is not None:
                mixed = swiglu(gate_up, dim=-1)
                _accumulate_dweight(
                    grouped_matmul_add,
                    grad_down_accumulator,
                    grad_output_chunk,
                    mixed,
                    group_list,
                )

            if need_input or need_gate_up:
                grad_mixed = F.linear(
                    grad_output_chunk,
                    down_weight.transpose(0, 1),
                )
                grad_gate_up_chunk = swiglu_backward(
                    grad_mixed,
                    gate_up,
                    dim=-1,
                )
                if grad_gate_up_accumulator is not None:
                    _accumulate_dweight(
                        grouped_matmul_add,
                        grad_gate_up_accumulator,
                        grad_gate_up_chunk,
                        input_chunk,
                        group_list,
                    )
                if flat_grad_inputs is not None:
                    flat_grad_inputs[start:end].copy_(
                        torch.mm(grad_gate_up_chunk, gate_up_weight)
                    )

        grad_inputs = (
            flat_grad_inputs.reshape(ctx.input_shape)
            if flat_grad_inputs is not None
            else None
        )
        grad_gate_up = (
            grad_gate_up_accumulator.to(gate_up_weight.dtype)
            if grad_gate_up_accumulator is not None
            else None
        )
        grad_down = (
            grad_down_accumulator.to(down_weight.dtype)
            if grad_down_accumulator is not None
            else None
        )
        return grad_inputs, grad_gate_up, grad_down, None


def _validate_chunk_size(chunk_size: int) -> None:
    """Validate a public token chunk size."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")


def _validate_chunked_swiglu(
    inputs: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    chunk_size: int,
) -> None:
    """Validate the packed, bias-free Chunked SwiGLU contract."""
    _validate_chunk_size(chunk_size)
    if inputs.dim() < 2 or inputs.numel() == 0:
        raise ValueError(
            "inputs must be a non-empty tensor shaped [..., hidden_size], "
            f"got {tuple(inputs.shape)}"
        )
    if inputs.device.type != "npu":
        raise ValueError("Chunked SwiGLU requires Ascend NPU tensors")
    if inputs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "Chunked SwiGLU requires float16 or bfloat16 inputs, "
            f"got {inputs.dtype}"
        )
    if gate_up_weight.dim() != 2 or down_weight.dim() != 2:
        raise ValueError("packed Gate/Up and Down weights must be two-dimensional")
    if gate_up_weight.size(0) % 2:
        raise ValueError("packed Gate/Up output dimension must be even")
    intermediate_size = gate_up_weight.size(0) // 2
    hidden_size = inputs.size(-1)
    if tuple(gate_up_weight.shape) != (2 * intermediate_size, hidden_size):
        raise ValueError("packed Gate/Up input dimension must match inputs")
    if tuple(down_weight.shape) != (hidden_size, intermediate_size):
        raise ValueError(
            "Down weight must be [hidden_size, intermediate_size], got "
            f"{tuple(down_weight.shape)} instead of "
            f"{(hidden_size, intermediate_size)}"
        )
    weights = (gate_up_weight, down_weight)
    if any(weight.device != inputs.device for weight in weights):
        raise ValueError("inputs and both weights must use the same device")
    if any(weight.dtype != inputs.dtype for weight in weights):
        raise ValueError("inputs and both weights must use the same dtype")


def chunked_swiglu(
    inputs: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Apply token-chunked packed SwiGLU recomputation on Ascend NPU.

    Args:
        inputs: Input tensor shaped ``[..., hidden_size]``.
        gate_up_weight: Packed Gate/Up weight shaped ``[2I, H]``.
        down_weight: Down projection weight shaped ``[H, I]``.
        chunk_size: Maximum flattened token count evaluated at once.

    Returns:
        Output tensor with the same shape as ``inputs``.
    """
    _validate_chunked_swiglu(inputs, gate_up_weight, down_weight, chunk_size)
    return _ChunkedSwiGLU.apply(inputs, gate_up_weight, down_weight, chunk_size)


@module_replacement
class ChunkedSwiGLUMLP(SwiGLUMLP):
    """Transformers-compatible packed SwiGLU with token recomputation."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        chunk_size: int,
    ) -> None:
        """Pack a source SwiGLU MLP and configure its token chunk size."""
        if context is not None and context.get("tp", False):
            raise NotImplementedError(
                "Chunked SwiGLU version one does not support tensor parallelism"
            )
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        _validate_chunk_size(chunk_size)
        if self.linear_fc1.bias is not None or self.linear_fc2.bias is not None:
            raise ValueError("Chunked SwiGLU version one supports bias-free MLPs only")
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use bounded recomputation in training and fused eager evaluation."""
        if self.training and torch.is_grad_enabled():
            return chunked_swiglu(
                x,
                self.linear_fc1.weight,
                self.linear_fc2.weight,
                self.chunk_size,
            )
        return super().forward(x)


__all__ = ["ChunkedSwiGLUMLP", "chunked_swiglu"]

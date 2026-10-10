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
"""Torch bindings for HyperMegaGate Route and RouteGrad kernels."""

from functools import lru_cache

import torch
import torch_npu  # pylint: disable=unused-import  # Registers NPU dispatch.

from hyper_parallel.core.multicore._loader import (
    NativeComponentUnavailableError,
    get_multicore_adapter_path,
    preload_vendor_library,
)


@lru_cache(maxsize=1)
def _load_native() -> None:
    """Load the MegaGate ABI adapter on the first native call."""
    vendor_root, adapter_path = get_multicore_adapter_path("hyper_parallel_mega_gate_torch")
    preload_vendor_library(vendor_root)
    try:
        torch.ops.load_library(str(adapter_path))
    except (OSError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED] component=mega_gate "
            f"framework=torch library={adapter_path} error={error}."
        ) from error


def _mega_gate_route(
    logits: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor,
    runtime_config: torch.Tensor,
    profile_buffer: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
    use_vision_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run HyperMegaGateRoute with text-only or per-token vision bias.

    Args:
        logits: Borrowed contiguous FP32 ``[tokens, expert_count]`` tensor.
        text_bias: Borrowed contiguous FP32 ``[expert_count]`` text selection bias.
        vision_bias: Borrowed contiguous FP32 ``[expert_count]`` vision selection bias.
        image_mask: Borrowed contiguous BOOL token mask, or the shared ``[1]`` placeholder.
        runtime_config: Internal UINT8 AIV pipeline descriptors.
        profile_buffer: Internal UINT8 cycle-record buffer or disabled placeholder.
        top_k: Number of selected experts in ``[1, expert_count]``.
        routed_scaling_factor: Finite factor applied to routed weights.
        use_vision_bias: Whether ``image_mask`` selects between text and vision bias.

    Returns:
        FP32 routing weights, route scores, selected scores and normalization
        denominator, plus INT64 expert indices. The final three FP32 tensors
        are internal forward state consumed by MegaGate backward.
        Completion follows current-stream order.
    """
    _load_native()
    return torch.ops.hyper_parallel.mega_gate_route(
        logits,
        text_bias,
        vision_bias,
        image_mask,
        runtime_config,
        profile_buffer,
        top_k,
        routed_scaling_factor,
        use_vision_bias,
    )


def _mega_gate_route_grad(
    logits: torch.Tensor,
    route_scores: torch.Tensor,
    selected_scores: torch.Tensor,
    normalization_denominator: torch.Tensor,
    expert_indices: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    runtime_config: torch.Tensor,
    profile_buffer: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Differentiate the sqrt-softplus Route with RouteGrad and CANN post kernels.

    All tensors are borrowed, contiguous, and complete in current-stream order.
    """
    _load_native()
    return torch.ops.hyper_parallel.mega_gate_route_grad(
        logits,
        route_scores,
        selected_scores,
        normalization_denominator,
        expert_indices,
        grad_routing_weights,
        runtime_config,
        profile_buffer,
        top_k,
        routed_scaling_factor,
    )


__all__: list[str] = []

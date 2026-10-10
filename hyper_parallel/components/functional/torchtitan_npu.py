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
"""Lazy adapters for the public production operators in TorchTitan-NPU."""

import importlib
import importlib.util
from functools import cache
from typing import Any

_OPERATOR_TARGETS = {
    "ascendc.inplace_partial_rotary_mul": (
        "torchtitan_npu.ops.ascendc.inplace_partial_rotary_mul",
        "inplace_partial_rotary_mul",
    ),
    "ascendc.moe_re_routing": (
        "torchtitan_npu.ops.ascendc.moe_re_routing",
        "npu_moe_re_routing",
    ),
    "ascendc.moe_token_permute": (
        "torchtitan_npu.ops.ascendc.moe_token_permute",
        "npu_moe_token_permute",
    ),
    "ascendc.moe_token_unpermute": (
        "torchtitan_npu.ops.ascendc.moe_token_unpermute",
        "npu_moe_token_unpermute",
    ),
    "tilelang.mhc_head_compute_mix": (
        "torchtitan_npu.ops.tilelang",
        "mhc_head_compute_mix_tilelang",
    ),
    "tilelang.mhc_head_compute_mix_a5": (
        "torchtitan_npu.ops.tilelang",
        "tilelang_mhc_head_compute_mix_a5",
    ),
    "tilelang.mhc_post": ("torchtitan_npu.ops.tilelang", "tilelang_mhc_post"),
    "tilelang.mhc_pre": ("torchtitan_npu.ops.tilelang", "tilelang_mhc_pre"),
    "tilelang.mhc_pre_v41": ("torchtitan_npu.ops.tilelang", "tilelang_mhc_pre_v41"),
    "tilelang.swiglu": ("torchtitan_npu.ops.tilelang", "tilelang_swiglu"),
    "tilelang.topk_gate": ("torchtitan_npu.ops.tilelang", "tilelang_topk_gate"),
    "triton.gated_delta_rule": ("torchtitan_npu.ops.triton.gdn", "gated_delta_rule"),
    "triton.mhc_post_bmm1": ("torchtitan_npu.ops.triton.mhc", "mhc_post_bmm1_op"),
    "triton.mhc_post_bmm2": ("torchtitan_npu.ops.triton.mhc", "mhc_post_bmm2_op"),
    "triton.mhc_pre_bmm": ("torchtitan_npu.ops.triton.mhc", "mhc_pre_bmm_op"),
    "triton.mhc_pre_only_sinkhorn": (
        "torchtitan_npu.ops.triton.mhc",
        "mhc_pre_only_sinkhorn_op",
    ),
    "triton.mhc_pre_sinkhorn": ("torchtitan_npu.ops.triton.mhc", "mhc_pre_sinkhorn_op"),
}


def is_torchtitan_npu_available() -> bool:
    """Return whether the optional ``torchtitan_npu`` package is discoverable."""
    return importlib.util.find_spec("torchtitan_npu") is not None


@cache
def _resolve_operator(operator: str) -> Any:
    target = _OPERATOR_TARGETS.get(operator)
    if target is None:
        supported = ", ".join(sorted(_OPERATOR_TARGETS))
        raise ValueError(
            f"Unsupported TorchTitan-NPU operator {operator!r}; supported operators: {supported}"
        )
    module_name, attribute = target
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Unable to load TorchTitan-NPU operator {operator!r}. Install torchtitan-npu and its backend dependencies "
            "for the target Ascend environment."
        ) from exc
    function = getattr(module, attribute, None)
    if not callable(function):
        raise TypeError(
            f"TorchTitan-NPU module {module_name!r} does not expose callable {attribute!r}; "
            "check that the installed revision matches the HyperParallel adapter."
        )
    return function


def torchtitan_npu_op(operator: str, /, *args: Any, **kwargs: Any) -> Any:
    """Call an allowlisted TorchTitan-NPU production operator without changing its arguments.

    Args:
        operator: Qualified adapter name listed in ``_OPERATOR_TARGETS``.
        *args: Positional arguments forwarded to the upstream operator.
        **kwargs: Keyword arguments forwarded to the upstream operator.

    Returns:
        The unmodified upstream operator result.
    """
    if not isinstance(operator, str):
        raise TypeError(f"operator must be str, got {type(operator).__name__}")
    return _resolve_operator(operator)(*args, **kwargs)


def torchtitan_inplace_partial_rotary_mul(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's differentiable in-place partial RoPE adapter."""
    return torchtitan_npu_op("ascendc.inplace_partial_rotary_mul", *args, **kwargs)


def torchtitan_moe_re_routing(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's differentiable MoE re-routing operator."""
    return torchtitan_npu_op("ascendc.moe_re_routing", *args, **kwargs)


def torchtitan_moe_token_permute(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's differentiable MoE token permutation operator."""
    return torchtitan_npu_op("ascendc.moe_token_permute", *args, **kwargs)


def torchtitan_moe_token_unpermute(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's differentiable MoE token restoration operator."""
    return torchtitan_npu_op("ascendc.moe_token_unpermute", *args, **kwargs)


def torchtitan_mhc_head_compute_mix(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's TileLang mHC head-compute-mix operator."""
    return torchtitan_npu_op("tilelang.mhc_head_compute_mix", *args, **kwargs)


def torchtitan_mhc_head_compute_mix_a5(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's A5 TileLang mHC head-compute-mix operator."""
    return torchtitan_npu_op("tilelang.mhc_head_compute_mix_a5", *args, **kwargs)


def torchtitan_mhc_post(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's TileLang mHC post operator."""
    return torchtitan_npu_op("tilelang.mhc_post", *args, **kwargs)


def torchtitan_mhc_pre(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's TileLang mHC pre operator."""
    return torchtitan_npu_op("tilelang.mhc_pre", *args, **kwargs)


def torchtitan_mhc_pre_v41(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's TileLang DeepSeek-V4.1 mHC pre operator."""
    return torchtitan_npu_op("tilelang.mhc_pre_v41", *args, **kwargs)


def torchtitan_swiglu(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's differentiable TileLang SwiGLU operator."""
    return torchtitan_npu_op("tilelang.swiglu", *args, **kwargs)


def torchtitan_topk_gate(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's TileLang Top-K gate operator."""
    return torchtitan_npu_op("tilelang.topk_gate", *args, **kwargs)


def torchtitan_gated_delta_rule(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's differentiable Triton gated-delta-rule operator."""
    return torchtitan_npu_op("triton.gated_delta_rule", *args, **kwargs)


def torchtitan_mhc_post_bmm1(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's first Triton mHC post BMM operator."""
    return torchtitan_npu_op("triton.mhc_post_bmm1", *args, **kwargs)


def torchtitan_mhc_post_bmm2(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's second Triton mHC post BMM operator."""
    return torchtitan_npu_op("triton.mhc_post_bmm2", *args, **kwargs)


def torchtitan_mhc_pre_bmm(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's Triton mHC pre BMM operator."""
    return torchtitan_npu_op("triton.mhc_pre_bmm", *args, **kwargs)


def torchtitan_mhc_pre_only_sinkhorn(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's Triton pre-only mHC Sinkhorn operator."""
    return torchtitan_npu_op("triton.mhc_pre_only_sinkhorn", *args, **kwargs)


def torchtitan_mhc_pre_sinkhorn(*args: Any, **kwargs: Any) -> Any:
    """Call TorchTitan-NPU's Triton mHC pre/post Sinkhorn operator."""
    return torchtitan_npu_op("triton.mhc_pre_sinkhorn", *args, **kwargs)


__all__ = [
    "is_torchtitan_npu_available",
    "torchtitan_gated_delta_rule",
    "torchtitan_inplace_partial_rotary_mul",
    "torchtitan_mhc_head_compute_mix",
    "torchtitan_mhc_head_compute_mix_a5",
    "torchtitan_mhc_post",
    "torchtitan_mhc_post_bmm1",
    "torchtitan_mhc_post_bmm2",
    "torchtitan_mhc_pre",
    "torchtitan_mhc_pre_bmm",
    "torchtitan_mhc_pre_only_sinkhorn",
    "torchtitan_mhc_pre_sinkhorn",
    "torchtitan_mhc_pre_v41",
    "torchtitan_moe_re_routing",
    "torchtitan_moe_token_permute",
    "torchtitan_moe_token_unpermute",
    "torchtitan_npu_op",
    "torchtitan_swiglu",
    "torchtitan_topk_gate",
]

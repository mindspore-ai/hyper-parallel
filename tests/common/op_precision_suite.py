# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Inventory and fixtures for all built-in contracts with a native baseline."""

from collections.abc import Callable

import torch

from hyper_parallel.components.ops.catalog import builtin_registry
from tests.common.op_precision import PrecisionCase
from tests.common.op_precision_cases import (
    attention_cases, swiglu_extended_cases, engram_cases, expert_cases, linear_attention_cases,
    mhc_pre_cases, pointwise_cases,
    quantization_cases, router_cases, tile_kernels_cases,
)
from tests.common.op_precision_dsa import dsa_cases
from tests.common.op_precision_gemm import gemm_cases, grouped_gemm_cases, prenorm_cases, mqa_cases

FAMILY_BUILDERS: dict[str, Callable[..., list[PrecisionCase]]] = {
    "gemm": gemm_cases,
    "rms_norm": pointwise_cases,
    "swiglu": pointwise_cases,
    "rope": pointwise_cases,
    "mhc_post": pointwise_cases,
    "mhc_pre": mhc_pre_cases,
    "sinkhorn": pointwise_cases,
    "grouped_matmul": expert_cases,
    "grouped_swiglu": expert_cases,
    "moe_router": router_cases,
    "engram": engram_cases,
    "quantization": quantization_cases,
    "moe_token_reorder": expert_cases,
    "mome": expert_cases,
    "attention": attention_cases,
    "causal_conv1d": linear_attention_cases,
    "gdn": linear_attention_cases,
    "kda": linear_attention_cases,
    "dsa_indexer": dsa_cases,
    "deepseek_sparse_attention": dsa_cases,
    "dsa_indexer_kl": dsa_cases,
}


def _family_precision_dtypes(family, variant):
    if family == "engram":
        return (torch.int32,) if variant == "hash" else (torch.bfloat16,)
    if family == "quantization":
        return (torch.float8_e4m3fn, torch.int8) if variant == "cast_back" else (torch.float32, torch.bfloat16)
    if family == "moe_router" and variant != "deepseek_v41":
        return (torch.float32,)
    if family == "mhc_pre":
        return (torch.float32,) if variant == "split_mixes" else (torch.bfloat16,)
    return None


def _deep_gemm_precision_dtypes(family, variant):
    if family == "dsa_indexer":
        return (torch.float8_e4m3fn, torch.int8)
    if family == "mhc_pre" or variant == "head_einsum":
        return (torch.bfloat16,)
    return (torch.bfloat16, torch.float8_e4m3fn, torch.int8)


def backend_precision_dtypes(family: str, backend: str, variant: str = "standard") -> tuple[torch.dtype, ...]:
    """Return the explicit training dtype scope of the hardware precision gate.

    DSA and optional linear-attention/expert kernels retain BF16 acceptance
    fixtures; Sinkhorn's coefficients are FP32 regardless of the builder dtype.
    The built-in Torch NPU pointwise, attention and grouped kernels cover all
    three training dtypes. CUDA grouped_mm only accepts BF16.
    """
    if backend == "deep_gemm":
        return _deep_gemm_precision_dtypes(family, variant)
    if family == "swiglu" and variant == "quantized_backward":
        return (torch.float8_e4m3fn,)
    if backend == "flash_mla":
        return (torch.bfloat16,)
    if family == "dsa_indexer" and backend == "deep_select":
        return (torch.float32, torch.bfloat16)
    family_dtypes = _family_precision_dtypes(family, variant)
    if family_dtypes is not None:
        return family_dtypes
    if backend == "tile_kernels":
        if family == "mhc_post":
            return (torch.bfloat16,)
        if family == "sinkhorn":
            return (torch.float32,)
        return (torch.float32, torch.bfloat16)
    if backend == "torch_grouped_mm":
        return (torch.bfloat16,)
    if family in ("rms_norm", "swiglu", "rope", "attention", "grouped_matmul",
                  "grouped_swiglu", "moe_token_reorder"):
        return (torch.float32, torch.float16, torch.bfloat16)
    if family == "sinkhorn":
        return (torch.float32,)
    if family == "mhc_post":
        return (torch.float16, torch.bfloat16)
    return (torch.bfloat16,)


def precision_cases(
    family: str, dtype: torch.dtype, *, hardware: bool = False, backend: str = "native",
) -> list[PrecisionCase]:
    """Return deterministic cases for one family, with fused-compatible ST dimensions."""
    if family == "swiglu" and dtype == torch.float8_e4m3fn:
        return [case for case in swiglu_extended_cases(dtype) if case.variant == "quantized_backward"]
    if backend == "tile_kernels":
        return tile_kernels_cases(family, dtype)
    builders = {"dsa_indexer": mqa_cases, "mhc_pre": prenorm_cases, "grouped_matmul": grouped_gemm_cases}
    if backend == "deep_gemm" and family in builders:
        return builders[family](dtype, hardware)
    cases = FAMILY_BUILDERS[family](dtype, hardware)
    if family == "grouped_matmul":
        cases += grouped_gemm_cases(dtype, hardware)
    if family == "mhc_pre":
        cases += prenorm_cases(dtype, hardware)
    if family == "dsa_indexer" and backend != "deep_select":
        cases += mqa_cases(dtype, hardware)
    return [case for case in cases if case.family == family]


def registered_backends() -> list[tuple[str, str, str, str]]:
    """Inspect registrations to make new native variants/backends fail the coverage audit.

    The registry has no public introspection API; this test-only audit accesses
    its metadata so coverage cannot silently drift from actual registrations.
    """
    return sorted((spec.config_key, spec.variant, backend, device)
                  for spec, backend, device in builtin_registry._implementations)  # pylint: disable=protected-access

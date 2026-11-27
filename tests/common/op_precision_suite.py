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
from tests.common.op_precision_basic import pointwise_cases
from tests.common.op_precision_experts import aggregate_hidden_cases, expert_cases
from tests.common.op_precision_sequence import attention_cases, linear_attention_cases
from tests.common.op_precision_dsa import dsa_cases

FAMILY_BUILDERS: dict[str, Callable[..., list[PrecisionCase]]] = {
    "rms_norm": pointwise_cases,
    "swiglu": pointwise_cases,
    "rope": pointwise_cases,
    "mhc_post": pointwise_cases,
    "sinkhorn": pointwise_cases,
    "grouped_matmul": expert_cases,
    "grouped_swiglu": expert_cases,
    "moe_router": expert_cases,
    "moe_token_permute": expert_cases,
    "moe_token_unpermute": expert_cases,
    "aggregate_hidden": aggregate_hidden_cases,
    "attention": attention_cases,
    "causal_conv1d": linear_attention_cases,
    "gdn": linear_attention_cases,
    "kda": linear_attention_cases,
    "dsa_indexer": dsa_cases,
    "deepseek_sparse_attention": dsa_cases,
    "dsa_indexer_kl": dsa_cases,
}


def backend_precision_dtypes(family: str, backend: str) -> tuple[torch.dtype, ...]:
    """Return the explicit training dtype scope of the hardware precision gate.

    DSA and optional linear-attention/expert kernels retain BF16 acceptance
    fixtures; Sinkhorn's coefficients are FP32 regardless of the builder dtype.
    The built-in Torch NPU pointwise, attention and grouped kernels cover all
    three training dtypes. CUDA grouped_mm only accepts BF16.
    """
    if backend == "torch_grouped_mm":
        return (torch.bfloat16,)
    if family in ("rms_norm", "swiglu", "rope", "attention", "grouped_matmul",
                  "grouped_swiglu", "moe_token_permute", "moe_token_unpermute"):
        return (torch.float32, torch.float16, torch.bfloat16)
    if family == "sinkhorn":
        return (torch.float32,)
    if family == "mhc_post":
        return (torch.float16, torch.bfloat16)
    return (torch.bfloat16,)


def precision_cases(family: str, dtype: torch.dtype, *, hardware: bool = False) -> list[PrecisionCase]:
    """Return deterministic cases for one family, with fused-compatible ST dimensions."""
    return [case for case in FAMILY_BUILDERS[family](dtype, hardware) if case.family == family]


def registered_backends() -> list[tuple[str, str, str, str]]:
    """Inspect registrations to make new native variants/backends fail the coverage audit.

    The registry has no public introspection API; this test-only audit accesses
    its metadata so coverage cannot silently drift from actual registrations.
    """
    return sorted((spec.config_key, spec.variant, backend, device)
                  for spec, backend, device in builtin_registry._implementations)  # pylint: disable=protected-access

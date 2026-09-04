# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""context_parallel: CP collectives, attention adaptations and wrappers.

Public surface: the built-in ``@inner_wrapper`` CP schemes
(``INNER_WRAPPER_REGISTRY`` + the wrapper factories) and the CP collective
primitives (``flex_cp_allgather`` and friends). Split out of
components/distributed/cp_utils.py + cp_wrappers.py in stage 4e.
"""

from hyper_parallel.distributed.context_parallel.collectives import (
    AsyncCPCollective,
    async_cp_allgather_launch,
    async_ulysses_seq_to_head_launch,
    flex_cp_allgather,
    hybrid_cp_attention,
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)
from hyper_parallel.distributed.context_parallel.wrappers import (
    INNER_WRAPPER_REGISTRY,
    INNER_WRAPPER_REQUIREMENTS,
    flex_hf_cp_wrapper,
    flex_hf_hybrid_cp_wrapper,
    flex_hf_ulysses_cp_wrapper,
    flex_qkv_cp_wrapper,
    flex_qkv_hybrid_cp_wrapper,
    flex_qkv_ulysses_cp_wrapper,
    mla_dsa_ulysses_cp_wrapper,
    sdpa_hf_cp_wrapper,
    sdpa_hf_hybrid_cp_wrapper,
    sdpa_hf_load_balance_cp_wrapper,
    sdpa_hf_ulysses_cp_wrapper,
    sdpa_qkv_cp_wrapper,
    sdpa_qkv_hybrid_cp_wrapper,
    sdpa_qkv_load_balance_cp_wrapper,
    sdpa_qkv_ulysses_cp_wrapper,
)

__all__ = [
    "AsyncCPCollective",
    "INNER_WRAPPER_REGISTRY",
    "INNER_WRAPPER_REQUIREMENTS",
    "async_cp_allgather_launch",
    "async_ulysses_seq_to_head_launch",
    "flex_cp_allgather",
    "flex_hf_cp_wrapper",
    "flex_hf_hybrid_cp_wrapper",
    "flex_hf_ulysses_cp_wrapper",
    "flex_qkv_cp_wrapper",
    "flex_qkv_hybrid_cp_wrapper",
    "flex_qkv_ulysses_cp_wrapper",
    "hybrid_cp_attention",
    "mla_dsa_ulysses_cp_wrapper",
    "sdpa_hf_cp_wrapper",
    "sdpa_hf_hybrid_cp_wrapper",
    "sdpa_hf_load_balance_cp_wrapper",
    "sdpa_hf_ulysses_cp_wrapper",
    "sdpa_qkv_cp_wrapper",
    "sdpa_qkv_hybrid_cp_wrapper",
    "sdpa_qkv_load_balance_cp_wrapper",
    "sdpa_qkv_ulysses_cp_wrapper",
    "ulysses_head_to_seq",
    "ulysses_seq_to_head",
]

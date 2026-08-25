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
"""Shared precision-comparison helpers for fully_shard ST (model-agnostic).

A distributed fully_shard run holds, on each rank, the dim-0 shard of every parameter
gradient. To check it against a single-card reference (the full, unsharded gradient),
slice the reference along dim 0 by the rank's shard coordinate and compare. These helpers
capture that mapping and the Tensor/DTensor-to-numpy conversion so every precision test
(whatever its model) shares them.
"""
import numpy as np


def assert_shard_matches_reference(
    case_name: str,
    rank: int,
    what: str,
    reference_full: np.ndarray,
    local_shard: np.ndarray,
    shard_size: int,
    shard_coord: int,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Assert a local shard equals the rank's ceil-chunk reference slice.

    Args:
        case_name: Precision case name used in failures.
        rank: Distributed rank used in failures.
        what: Tensor label used in failures.
        reference_full: Full single-card reference array.
        local_shard: Logical local shard array.
        shard_size: Number of ranks sharding tensor dimension zero.
        shard_coord: This rank's coordinate in the shard mesh.
        rtol: Relative comparison tolerance.
        atol: Absolute comparison tolerance.
    """
    chunk = (reference_full.shape[0] + shard_size - 1) // shard_size
    shard_start = min(shard_coord * chunk, reference_full.shape[0])
    shard_end = min(shard_start + chunk, reference_full.shape[0])
    expected = reference_full[shard_start:shard_end]
    assert np.allclose(expected, local_shard, rtol=rtol, atol=atol), (
        f"{case_name}, rank {rank}, {what}: expected slice {expected}, got {local_shard}"
    )


def _to_numpy(value):
    """Tensor / DTensor -> numpy; a DTensor returns its local shard."""
    return value.to_local().asnumpy() if hasattr(value, "to_local") else value.asnumpy()

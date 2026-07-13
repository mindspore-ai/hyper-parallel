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
"""CP-specific types and helpers shared by memory and perf estimation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import AttentionType


class CPAlgo(Enum):
    """CP algorithm enumeration."""

    COLOSSALAI_CP = "colossalai_cp"
    HYBRID_CP = "hybrid_cp"
    ULYSSES_CP = "ulysses_cp"


_CP_ALGO_STR_MAP = {
    "colossalai_cp": CPAlgo.COLOSSALAI_CP,
    "hybrid_cp": CPAlgo.HYBRID_CP,
    "hybird_cp": CPAlgo.HYBRID_CP,
    "ulysses_cp": CPAlgo.ULYSSES_CP,
}


def _resolve_cp_algo(ccfg) -> CPAlgo:
    """Resolve cp_algo from ccfg, defaulting to COLOSSALAI_CP."""
    raw = getattr(ccfg, 'cp_algo', None)
    if isinstance(raw, CPAlgo):
        return raw
    if isinstance(raw, str):
        return _CP_ALGO_STR_MAP.get(raw, CPAlgo.COLOSSALAI_CP)
    return CPAlgo.COLOSSALAI_CP


@dataclass
class CPMemoryBreakdown:
    """CP activation memory breakdown per layer (all values in bytes)."""

    kv_cache_memory: float
    attention_scores_memory: float
    softmax_outputs_memory: float
    dropout_mask_memory: float
    comm_buffer_memory: float
    kv_reduction: float
    s2_reduction: float
    total_reduction: float
    total_memory: float
    cp_degree: int
    seq_len: int
    attention_type: AttentionType
    cp_algo: CPAlgo = CPAlgo.COLOSSALAI_CP


@dataclass
class CPCommunicationCost:
    """CP communication cost per layer.

    comm_volume uses the same weighted-unit formula as dp/tp/ep comm
    (comm_cp * s * b * dimension_terms), so it can be summed with
    comm[Dim.DP/TP/EP] and fed into estimate_comm_score.
    total_kv_volume is the raw byte volume for diagnostics.
    """

    kv_volume_per_step: float
    total_kv_volume: float
    comm_volume: float
    ring_steps: int
    ring_directions: int
    total_comm_time: float
    exposed_comm_time: float
    overlap_ratio: float
    effective_bandwidth: float
    topology: str
    cp_degree: int
    seq_len: int
    batch_size: int
    attention_type: AttentionType
    kv_dim: int
    cp_algo: CPAlgo = CPAlgo.COLOSSALAI_CP


@dataclass
class CPValidationResult:
    """CP constraint validation result."""

    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    seq_len_divisible: bool = True
    topology_feasible: bool = True
    device_sufficient: bool = True
    memory_within_limit: bool = True
    recommended_cp_max: Optional[int] = None
    topology_penalty: Optional[float] = None
    unsupported_reason: Optional[str] = None


@dataclass
class CPConstraintParams:
    """Parameters for CP constraint validation."""

    seq_len: int
    cp_degree: int
    tp_degree: int = 1
    pp_degree: int = 1
    device_per_node: int = 8
    attention_type_str: str = "mha"
    bw_intra: float = 300.0
    bw_inter: float = 25.0
    total_devices: int = 0
    cp_memory_per_layer: float = 0.0
    device_capacity: float = 0.0
    num_layers: int = 0
    cp_algo: str = "colossalai_cp"
    attention_heads: int = 0
    num_kv_heads: int = 0
    sp_enabled: bool = False

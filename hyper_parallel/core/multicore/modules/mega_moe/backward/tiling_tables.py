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
"""
Backward tiling lookup tables.
Each dict maps split_value -> CSV tiling string (for GMM ops) or
struct fields (for SwiGLU-grad).

C++ source functions:
  GMM1  (pos 20) → get_tiling_data_first_gmm(split_value)
  GMM2  (pos 21) → get_tiling_data_second_gmm(split_value)
  GMM3  (pos 22) → get_tiling_data_third_gmm(split_value)
  GMM4  (pos 23) → get_tiling_data_fourth_gmm(split_value)
  SwiGLU-grad (pos 24) → get_tiling_data_swiglu_grad(split_value)
"""
# pylint: disable=line-too-long
import copy
import ctypes

from hyper_parallel.core.multicore.scheduler.config import (
    SwiGluTilingDataC,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _expand_gmm_string(csv: str, repeat: int = 24) -> bytes:
    """Parse comma-separated uint32 string and tile it `repeat` times."""
    vals = [int(x) for x in csv.split(',')]
    arr  = (ctypes.c_uint32 * (len(vals) * repeat))()
    for i in range(len(vals) * repeat):
        arr[i] = vals[i % len(vals)]
    return bytes(arr)


def _expand_swiglu_struct(td: SwiGluTilingDataC) -> bytes:
    """Tile SwiGluTilingDataC 49 times (= 2 * 24 AI Cube cores + 1)."""
    return bytes(td) * 49


def _find_protected_positions(vals, active_map, split_value, is_weight_grad):
    """Return index positions that must not be replaced by dim_map substitution.

    When split_value collides with a K/N dim in active_map, the M-tile or
    K-tile positions (depending on GMM type) must be shielded.
    T is located via the 393216 anchor which sits at T+19.
    """
    if split_value is None or split_value not in active_map:
        return set()
    for i in range(350, len(vals)):
        if vals[i] == 393216:          # anchor always at t_base+19
            t_base = i - 19
            if is_weight_grad:
                return {t_base + 3, t_base + 4, t_base + 7}
            return {14, t_base + 1, t_base + 5}
    return set()


def _patch_gmm_dims(csv: str, dim_map: dict, num_groups: int,
                    default_groups: int = 8,
                    split_value: int = None,
                    num_cube_cores: int = 24,
                    default_num_cores: int = 24) -> str:
    """Replace matrix dimension values in a GMM tiling CSV string.

    dim_map:         {old_int_value: new_int_value} for K/N dimension replacements.
                     Entries where old == new are no-ops and can be omitted.
    num_groups:      replacement for default_groups; only applied at index > 350.
    split_value:     M tile size (table key). When it equals a K/N value in dim_map,
                     we must protect M-tile positions from accidental replacement.
                     Two GMM types detected via vals[10]:
                       0 → activation GMM: M tile at {14, T+1, T+5}
                       2 → weight-grad GMM: split (K-tile) at {T+3, T+4, T+7}
                     T is found via the 393216 anchor fixed at T+19.
    num_cube_cores:  number of AI Cube cores on the target hardware (default 24 = 910B).
                     Replaces the hardcoded usedCoreNum/blockDim fields in the CSV:
                       vals[1]    — header blockDim field
                       i > 350    — tail usedCoreNum and any derived parallelism fields
    """
    active_map = {k: v for k, v in dim_map.items() if k != v}
    groups_changed = num_groups != default_groups
    cores_changed  = num_cube_cores != default_num_cores
    if not active_map and not groups_changed and not cores_changed:
        return csv  # fast path: nothing to do

    vals = list(map(int, csv.split(',')))
    is_weight_grad = vals[10] == 2

    # Header fields (set before main loop, not touched in loop)
    if groups_changed and vals[0] == default_groups:
        vals[0] = num_groups          # vals[0] = groupNum
    if cores_changed and vals[1] == default_num_cores:
        vals[1] = num_cube_cores      # vals[1] = blockDim

    protected = _find_protected_positions(vals, active_map, split_value, is_weight_grad)

    for i, v in enumerate(vals):
        if i in protected:
            continue
        if v in active_map:
            vals[i] = active_map[v]
        elif cores_changed and i > 350 and v == default_num_cores:
            vals[i] = num_cube_cores   # covers T+0 (usedCoreNum) and derived fields
    return ','.join(map(str, vals))


def _make_swiglu(row_len, col_len, base_row_len, base_col_len):
    """Build a SwiGluTilingDataC for SwiGLU-grad with the given tile dimensions."""
    td = SwiGluTilingDataC()
    td.is32BAligned         = 1
    td.isDoubleBuffer       = 1
    td.rowLen               = row_len
    td.colLen               = col_len
    td.baseRowLen           = base_row_len
    td.baseColLen           = base_col_len
    td.activateLeft         = 0
    td.biasIsEmpty          = 0
    td.quantScaleIsEmpty    = 0
    td.activateScaleIsEmpty = 0
    td.swiColLen            = 0
    td.perRowLen            = 0
    td.modRowLen            = 0
    td.usedCoreNum          = 0
    return td


# ── Backward GMM1: x=[per_rank_seq,7168], weight=[E,2048,7168] ───────────────
# C++: get_tiling_data_first_gmm(split_value)
# x input [*, 7168] → y output [*, 2048]

FIRST_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,2048,7168,7168,4096,256,7168,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward GMM2: x=[per_rank_seq,4096], weight=[E,7168,4096] ───────────────
# C++: get_tiling_data_second_gmm(split_value)
# x input [*, 4096] → y output [*, 7168]

SECOND_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,7168,4096,4096,4096,256,4096,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward GMM3: x1^T=[sre,7168], x2=[sre,4096], weight grad ───────────────
# C++: get_tiling_data_third_gmm(split_value)

THIRD_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,2,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,7168,4096,4096,4096,7168,256,4096,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward GMM4: x1^T=[sre,2048], x2=[sre,7168], weight grad ───────────────
# C++: get_tiling_data_fourth_gmm(split_value)

FOURTH_GMM_TABLE = {
    4096:  "8,24,0,0,0,0,0,1,1,1,2,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,2048,7168,4096,4096,2048,256,4096,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Backward SwiGLU-grad tiling ───────────────────────────────────────────────
# C++: get_tiling_data_swiglu_grad(split_value)  → baseColLen=256

SWIGLU_GRAD_TABLE = {
    32:  _make_swiglu(row_len=32,  col_len=2048, base_row_len=19, base_col_len=256),
    64:  _make_swiglu(row_len=64,  col_len=2048, base_row_len=19, base_col_len=256),
    128: _make_swiglu(row_len=128, col_len=2048, base_row_len=19, base_col_len=256),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_act_grad_tiling_bytes(split_value: int, *,
                                hidden_size: int = 7168,
                                intermediate_size: int = 2048,
                                num_groups: int = 8,
                                num_cube_cores: int = 24) -> bytes:
    """Backward GMM1 [x,hidden]×[E,intermediate,hidden] → y=[x,intermediate]."""
    csv = FIRST_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No first_gmm tiling for split_value={split_value}")
    csv = _patch_gmm_dims(csv,
                          {7168: hidden_size, 2048: intermediate_size},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_gate_grad_tiling_bytes(split_value: int, *,
                                 hidden_size: int = 7168,
                                 intermediate_size: int = 2048,
                                 num_groups: int = 8,
                                 num_cube_cores: int = 24) -> bytes:
    """Backward GMM2 [x,intermediate*2]×[E,hidden,intermediate*2] → y=[x,hidden]."""
    csv = SECOND_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No second_gmm tiling for split_value={split_value}")
    csv = _patch_gmm_dims(csv,
                          {4096: intermediate_size * 2, 7168: hidden_size},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_w2_grad_tiling_bytes(split_value: int, *,
                                hidden_size: int = 7168,
                                intermediate_size: int = 2048,
                                num_groups: int = 8,
                                num_cube_cores: int = 24) -> bytes:
    """Backward GMM3 weight-grad [hidden,sre]×[sre,intermediate*2] → grad=[hidden,intermediate*2]."""
    csv = THIRD_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No third_gmm tiling for split_value={split_value}")
    csv = _patch_gmm_dims(csv,
                          {7168: hidden_size, 4096: intermediate_size * 2},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_w1_grad_tiling_bytes(split_value: int, *,
                                 hidden_size: int = 7168,
                                 intermediate_size: int = 2048,
                                 num_groups: int = 8,
                                 num_cube_cores: int = 24) -> bytes:
    """Backward GMM4 weight-grad [intermediate,sre]×[sre,hidden] → grad=[intermediate,hidden]."""
    csv = FOURTH_GMM_TABLE.get(split_value)
    if csv is None:
        raise KeyError(f"No fourth_gmm tiling for split_value={split_value}")
    csv = _patch_gmm_dims(csv,
                          {2048: intermediate_size, 7168: hidden_size,
                           4096: intermediate_size * 2},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_swiglu_grad_tiling_bytes(split_value: int, *,
                                  intermediate_size: int = 2048) -> bytes:
    """Backward SwiGLU-grad tiling (colLen=intermediate_size)."""
    td = SWIGLU_GRAD_TABLE.get(split_value)
    if td is None:
        raise KeyError(f"No swiglu_grad tiling for split_value={split_value}")
    td = copy.copy(td)
    td.colLen = intermediate_size
    return _expand_swiglu_struct(td)

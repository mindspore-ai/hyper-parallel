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
Forward tiling lookup tables.
Each function mirrors the C++ get_tiling_data_* functions.
String-based GMM tiling is expanded num_cube_cores times (default 24);
struct-based SwiGLU tiling is expanded 49 times (= 2 * 24 + 1 for Ascend 910B).
"""
# pylint: disable=line-too-long
import copy
import ctypes

from hyper_parallel.core.multicore.modules.moe_ffn.common.runtime_structs import (
    SwiGluTilingDataC,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _expand_gmm_string(csv: str, repeat: int = 24) -> bytes:
    """Parse comma-separated uint32 string and tile it `repeat` times."""
    vals = [int(x) for x in csv.split(',')]
    arr = (ctypes.c_uint32 * (len(vals) * repeat))()
    for i in range(len(vals) * repeat):
        arr[i] = vals[i % len(vals)]
    return bytes(arr)


def _expand_swiglu_struct(td: SwiGluTilingDataC) -> bytes:
    """Tile SwiGluTilingDataC 49 times (= 2 * 24 AI Cube cores + 1)."""
    one = bytes(td)
    return one * 49


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


# ── Forward GMM1: x=[per_rank_seq,7168], weight=[E,7168,4096] ─────────────────
# C++: get_tiling_data_gmm_x_7168_g2(split_value)

GMM1_TABLE = {
    4096: "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,4096,7168,7168,4096,256,7168,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Forward GMM2: x=[per_rank_seq,2048], weight=[E,2048,7168] ─────────────────
# C++: get_tiling_data_gmm_x_2048_g2(split_value)

GMM2_TABLE = {
    4096: "8,24,0,0,0,0,0,1,1,1,0,0,0,0,4096,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4294967295,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,4096,7168,2048,2048,4096,256,2048,128,256,64,8,8,1,1,0,0,0,0,393216,131072,0,1,1,1,1,4,4,0,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
}


# ── Forward SwiGLU: colLen=2048 ───────────────────────────────────────────────
# C++: get_tiling_data_swiglu(split_value)  →  SwiGluTilingData struct

def _make_swiglu(is32b, idbl, row_len, col_len, base_row_len, base_col_len,
                 act_left, bias_empty, qs_empty, as_empty,
                 swi_col_len, per_row_len, mod_row_len, used_core_num):
    """Build a SwiGluTilingDataC struct with the given tile parameters."""
    td = SwiGluTilingDataC()
    td.is32BAligned         = is32b
    td.isDoubleBuffer       = idbl
    td.rowLen               = row_len
    td.colLen               = col_len
    td.baseRowLen           = base_row_len
    td.baseColLen           = base_col_len
    td.activateLeft         = act_left
    td.biasIsEmpty          = bias_empty
    td.quantScaleIsEmpty    = qs_empty
    td.activateScaleIsEmpty = as_empty
    td.swiColLen            = swi_col_len
    td.perRowLen            = per_row_len
    td.modRowLen            = mod_row_len
    td.usedCoreNum          = used_core_num
    return td


SWIGLU_TABLE = {
    16:   _make_swiglu(1,1,16,  2048, 16, 512, 0,0,0,0, 0,0,0,0),
    32:   _make_swiglu(1,1,32,  2048, 32, 256, 0,0,0,0, 0,0,0,0),
    64:   _make_swiglu(1,1,64,  2048, 19, 512, 0,0,0,0, 0,0,0,0),
    128:  _make_swiglu(1,1,128, 2048, 19, 512, 0,0,0,0, 0,0,0,0),
    256:  _make_swiglu(1,1,256, 2048, 19, 512, 0,0,0,0, 0,0,0,0),
    512:  _make_swiglu(1,1,512, 2048, 19, 512, 0,0,0,0, 0,0,0,0),
    1024: _make_swiglu(1,1,1024,2048, 19, 512, 0,0,0,0, 0,0,0,0),
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_up_proj_tiling_bytes(split_value: int, *,
                           hidden_size: int = 7168,
                           intermediate_size: int = 2048,
                           num_groups: int = 8,
                           num_cube_cores: int = 24) -> bytes:
    """Forward GMM1 tiling [x,hidden]×[E,hidden,intermediate*2], expanded num_cube_cores times."""
    if split_value not in GMM1_TABLE:
        raise KeyError(f"GMM1 tiling: split_value={split_value} not found")
    csv = _patch_gmm_dims(GMM1_TABLE[split_value],
                          {7168: hidden_size, 4096: intermediate_size * 2},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_down_proj_tiling_bytes(split_value: int, *,
                           hidden_size: int = 7168,
                           intermediate_size: int = 2048,
                           num_groups: int = 8,
                           num_cube_cores: int = 24) -> bytes:
    """Forward GMM2 tiling [x,intermediate]×[E,intermediate,hidden], expanded num_cube_cores times."""
    if split_value not in GMM2_TABLE:
        raise KeyError(f"GMM2 tiling: split_value={split_value} not found")
    csv = _patch_gmm_dims(GMM2_TABLE[split_value],
                          {2048: intermediate_size, 7168: hidden_size},
                          num_groups, split_value=split_value,
                          num_cube_cores=num_cube_cores)
    return _expand_gmm_string(csv, repeat=num_cube_cores)


def get_swiglu_tiling_bytes(split_value: int, *,
                             intermediate_size: int = 2048) -> bytes:
    """Forward SwiGLU tiling (colLen=intermediate_size), expanded 49 times."""
    if split_value not in SWIGLU_TABLE:
        raise KeyError(f"SwiGLU tiling: split_value={split_value} not found")
    td = copy.copy(SWIGLU_TABLE[split_value])
    td.colLen = intermediate_size
    return _expand_swiglu_struct(td)

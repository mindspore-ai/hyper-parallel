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
"""parallel dimensions"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Union

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import (
    CPValidationResult,
    CPConstraintParams,
)

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common._cost_model_variables import _CostModVar


class Dimension:
    """Output dimension"""

    def __init__(
        self,
        acronym,
        cost_model_var_name,
        from_str,
        default=1,
    ):
        self.name = acronym
        self.cost_model_var = cost_model_var_name
        self.default = default
        self.bound = None
        self.from_str = from_str

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def lname(self):
        """lower case name"""
        return self.name.lower()

    def from_config(self, ccfg: _CostModVar):
        """Get dimension value from cost model config"""
        try:
            value = ccfg.__dict__[self.cost_model_var]
        except KeyError:
            logger.error(
                "variable %s does not exist in the cost model: %s",
                self.cost_model_var,
                str(ccfg.__dict__),
            )
            sys.exit(1)
        return value

    def reset_bound(self):
        """Reset bound the dimension space"""
        self.bound = None

    def set_bound(self, bound):
        """Bound the dimension space"""
        if self.bound:
            logger.debug(
                "bound(%s) = min (%d, %d)", self.name, bound, self.bound
            )
            self.bound = min(bound, self.bound)
        else:
            logger.debug("bound(%s) = %d", self.name, bound)
            self.bound = bound

    def get_bound(self):
        """Return dimension bound"""
        return self.bound

    def is_valid(self, value):
        """Check dimension value validity"""
        invalid = False
        if isinstance(value, bool):
            return not invalid
        if isinstance(value, int):
            invalid = self.bound and value > self.bound
            invalid = invalid or value < 1
        if invalid:
            logger.warning(
                "Dimension %s = %s is invalid", self.name, str(value)
            )
        return not invalid


DP = Dimension(
    "DP",
    "d",
    default=1,
    from_str=int,
)
EP = Dimension(
    "EP",
    "ep",
    default=1,
    from_str=int,
)
TP = Dimension(
    "MP",
    "t",
    default=1,
    from_str=int,
)
CP = Dimension(
    "CP",
    "cp",
    default=1,
    from_str=int,
)
PP = Dimension(
    "PP",
    "p",
    default=1,
    from_str=int,
)
MBN = Dimension(
    "MB",
    "m",
    default=1,
    from_str=int,
)
MBS = Dimension(
    "MBS",
    "b",
    default=1,
    from_str=int,
)
SP = Dimension(
    "SP",
    "sp",
    default=True,
    from_str=bool,
)
OP = Dimension(
    "OP",
    "os_max_shard",
    # "op_weight_shard",
    default=1,
    from_str=int,
)
VPP = Dimension(
    "VPP",
    "vp",
    default=1,
    from_str=int,
)

ALL_DIMS = [DP, EP, TP, CP, PP, VPP, MBN, MBS, SP, OP]


class Dimensions:
    """All output dimensions"""

    def __init__(self, config, all_dims=None):
        if isinstance(config, list):
            self.all_dims = [d for d, _ in config]
            self.dims_val = dict(config)
        # elif isinstance(config, dict):
        #     self.all_dims = ALL_DIMS
        #     self.dims_val = {d: d.from_config(config) for d in self.all_dims}
        elif isinstance(config, bool):
            self.all_dims = ALL_DIMS
            self.dims_val = {d: d.default for d in self.all_dims}
        else:
            raise TypeError(
                f"Dimensions cannot be constructed from type {type(config)}"
            )
        if all_dims:
            self.all_dims = all_dims
        self._reset_all_dims()

    def _reset_all_dims(self):
        for d in self.all_dims:
            d.reset_bound()

    def __str__(self):
        return str(self.dims_val)

    def __repr__(self):
        return str(self)

    def keys(self):
        """Return dimensions"""
        return list(self.dims_val)

    def global_batch_size(self):
        """Compute the global batch size"""
        gbs = self.dims_val[DP] * self.dims_val[MBS]
        has_pp = self.has_dim(PP) and self.dims_val[PP] > 1
        has_mbn = self.has_dim(MBN) and self.dims_val[MBN] > 1
        if has_pp and has_mbn:
            gbs *= self.dims_val[MBN]
        return gbs

    def values(self):
        """Return dimension value"""
        return [str(self.dims_val[d]) for d in self.dims_val]

    def unique_name(self):
        """Return all values as a unique string"""
        return "_".join(self.values())

    def has_dim(self, d):
        """Check that this dimension has a value in the parallel config"""
        return d in self.dims_val

    @staticmethod
    def _check_mbn_pp(dims_val, all_dims):
        """Return True if MBN/PP combination is valid."""
        if MBN not in dims_val or PP not in all_dims:
            return True
        valid = dims_val[MBN] >= dims_val[PP]
        valid = valid and not (dims_val[PP] == 1 and dims_val[MBN] > 1)
        if not valid:
            logger.warning("PP and MBN were deemed not suitable")
        return valid

    @staticmethod
    def _check_power_of_two(dim_obj, value):
        """Return True if *value* is a power of 2 (for TP / OP)."""
        if not value & (value - 1) == 0:
            logger.warning("%s must be a power of 2", str(dim_obj))
            return False
        return True

    def is_valid(self):
        """Check if all dimensions values are valid"""
        if not self._check_mbn_pp(self.dims_val, self.all_dims):
            return False
        if TP in self.all_dims and not self._check_power_of_two(TP, self.dims_val[TP]):
            return False
        for d in self.dims_val:
            if not d.is_valid(self.dims_val[d]):
                logger.warning("Dimension %d is not valid", d)
                return False
        if SP in self.all_dims and CP in self.all_dims:
            if self.dims_val[SP] and self.dims_val[CP] > 1:
                logger.warning("SP & CP cannot coexist")
                return False
        if OP in self.all_dims and not self._check_power_of_two(OP, self.dims_val[OP]):
            return False
        return True

    def val(self, dim):
        """Get Dimension value"""
        return self.dims_val[dim]

    def set(self, dim, val):
        """Get Dimension value"""
        self.dims_val[dim] = val

    def steal(self, factor, dim_from, dim_to):
        """Assign a dimension factor to another dimension"""
        self.dims_val[dim_from] = self.dims_val[dim_from] // factor
        self.dims_val[dim_to] = self.dims_val[dim_to] * factor


def get_dim(acronym):
    """Return the dimension of the given string acronym"""
    dname = str(acronym).upper()
    for d in ALL_DIMS:
        if d.name == dname:
            return d
    raise ValueError(f"Dimension {dname} does NOT exist")


def get_dims(dims):
    """Return all dimensions considered"""
    if dims is None:
        return ALL_DIMS
    return [get_dim(acronym) for acronym in dims]


def _cp_params_from_args(
    seq_len, cp_degree, tp_degree, pp_degree, device_per_node,
    attention_type_str, bw_intra, bw_inter, total_devices,
    cp_memory_per_layer, device_capacity, num_layers,
    cp_algo, attention_heads, sp_enabled, num_kv_heads=0,
):
    """Resolve CPConstraintParams from flexible args."""
    if isinstance(seq_len, CPConstraintParams):
        return seq_len
    return CPConstraintParams(
        seq_len=seq_len, cp_degree=cp_degree, tp_degree=tp_degree,
        pp_degree=pp_degree, device_per_node=device_per_node,
        attention_type_str=attention_type_str, bw_intra=bw_intra,
        bw_inter=bw_inter, total_devices=total_devices,
        cp_memory_per_layer=cp_memory_per_layer,
        device_capacity=device_capacity, num_layers=num_layers,
        cp_algo=cp_algo, attention_heads=attention_heads,
        num_kv_heads=num_kv_heads, sp_enabled=sp_enabled,
    )


def _cp_ok_result(**overrides):
    """Build a passing CPValidationResult with sensible defaults."""
    defaults = {
        "is_valid": True, "error_message": None, "warning_message": None,
        "seq_len_divisible": True, "topology_feasible": True,
        "device_sufficient": True, "memory_within_limit": True,
        "recommended_cp_max": None, "topology_penalty": None,
        "unsupported_reason": None,
    }
    defaults.update(overrides)
    return CPValidationResult(**defaults)


def _cp_check_ulysses_heads(p):
    """Check Ulysses KV-head divisibility. Returns error message or None.

    Ulysses CP shards KV heads across tp×cp ranks, so num_kv_heads must be
    divisible by tp×cp. When TP replicates KV (GQA with tp ≤ kv-head-groups)
    the check is conservative but safe — it never lets a crashing config
    through. num_kv_heads=0 falls back to attention_heads (MHA semantics,
    matching compute_kv_dim).
    """
    if p.cp_algo != "ulysses_cp":
        return None
    kv_heads = p.num_kv_heads if p.num_kv_heads > 0 else p.attention_heads
    if kv_heads <= 0:
        return None
    shards = p.tp_degree * p.cp_degree
    if shards <= 0:
        return None
    if kv_heads % shards != 0:
        return (
            f"Ulysses CP requires num_kv_heads ({kv_heads}) divisible by "
            f"tp_degree × cp_degree ({p.tp_degree} × {p.cp_degree} = {shards})."
        )
    return None


def _cp_check_device_sufficiency(p):
    """Check tp*cp*pp <= total_devices. Returns error message or None."""
    if p.total_devices > 0 and p.tp_degree * p.cp_degree * p.pp_degree > p.total_devices:
        return (
            f"tp×cp×pp ({p.tp_degree}×{p.cp_degree}×{p.pp_degree} = "
            f"{p.tp_degree * p.cp_degree * p.pp_degree}) exceeds "
            f"total available devices ({p.total_devices})."
        )
    return None


def _cp_collect_warnings(p):
    """Collect warning messages and derived topology/recommendation info."""
    warnings = []
    topology_feasible = True
    topology_penalty = None
    recommended_cp_max = None

    if p.tp_degree * p.cp_degree > p.device_per_node:
        topology_feasible = False
        topology_penalty = 1.0 - (p.bw_inter / p.bw_intra)
        warnings.append(
            f"CP will cross node boundary (tp={p.tp_degree} × cp={p.cp_degree} = "
            f"{p.tp_degree * p.cp_degree} > {p.device_per_node} devices/node). "
            f"Communication will use slower inter-node bandwidth."
        )

    if p.seq_len < 8192:
        warnings.append(
            f"CP not recommended for short sequences (seq_len={p.seq_len} < 8192). "
            f"Communication overhead may outweigh memory benefits."
        )

    attn_upper = p.attention_type_str.upper()
    if attn_upper == "MLA":
        recommended_cp_max = 16
    elif attn_upper == "GQA":
        recommended_cp_max = 8
    else:
        recommended_cp_max = 4

    if recommended_cp_max is not None and p.cp_degree > recommended_cp_max:
        warnings.append(
            f"cp_degree ({p.cp_degree}) exceeds recommended max ({recommended_cp_max}) "
            f"for {attn_upper} attention."
        )

    return warnings, topology_feasible, topology_penalty, recommended_cp_max


def _cp_check_memory(p, warnings, topology_feasible, topology_penalty, recommended_cp_max):
    """Check CP memory capacity. Returns CPValidationResult or None."""
    if not (p.cp_memory_per_layer > 0 and p.device_capacity > 0 and p.num_layers > 0):
        return None
    total_cp_memory = p.cp_memory_per_layer * p.num_layers
    if total_cp_memory <= p.device_capacity:
        return None
    return _cp_ok_result(
        is_valid=False,
        error_message=(
            f"CP memory per card ({total_cp_memory / 1e6:.1f} MB) exceeds "
            f"device capacity ({p.device_capacity / 1e6:.1f} MB)."
        ),
        warning_message="; ".join(warnings) if warnings else None,
        topology_feasible=topology_feasible,
        topology_penalty=topology_penalty,
        recommended_cp_max=recommended_cp_max,
        memory_within_limit=False,
    )


def validate_cp_constraints(
    seq_len: Union[CPConstraintParams, int],
    cp_degree: int = 1,
    tp_degree: int = 1,
    pp_degree: int = 1,
    device_per_node: int = 8,
    attention_type_str: str = "mha",
    bw_intra: float = 300.0,
    bw_inter: float = 25.0,
    total_devices: int = 0,
    cp_memory_per_layer: float = 0.0,
    device_capacity: float = 0.0,
    num_layers: int = 0,
    cp_algo: str = "colossalai_cp",
    attention_heads: int = 0,
    sp_enabled: bool = False,
    num_kv_heads: int = 0,
) -> CPValidationResult:
    """Validate CP constraints for a given parallel configuration.

    Accepts either a CPConstraintParams dataclass or individual keyword
    arguments for backward compatibility.

    Args:
        seq_len: Sequence length, or a CPConstraintParams dataclass.
        cp_degree: CP degree.
        tp_degree: TP degree.
        pp_degree: PP degree.
        device_per_node: Number of devices per node.
        attention_type_str: Attention type string ("mha", "gqa", "mla").
        bw_intra: Intra-node bandwidth in GB/s (default: 300.0 for Ascend A2).
        bw_inter: Inter-node bandwidth in GB/s (default: 25.0).
        total_devices: Total number of available devices (0 = skip check).
        cp_memory_per_layer: CP memory per layer in bytes (0 = skip check).
        device_capacity: Device memory capacity in bytes (0 = skip check).
        num_layers: Number of transformer layers (0 = skip check).
        cp_algo: CP algorithm ("colossalai_cp", "hybrid_cp", "ulysses_cp").
        attention_heads: Number of attention heads (0 = skip Ulysses head check).
        sp_enabled: Whether sequence parallelism is enabled (SP and CP are incompatible).
        num_kv_heads: Number of KV heads for Ulysses divisibility check
            (0 = fall back to attention_heads, matching compute_kv_dim).

    Returns:
        CPValidationResult with validation outcome.
    """
    p = _cp_params_from_args(
        seq_len, cp_degree, tp_degree, pp_degree, device_per_node,
        attention_type_str, bw_intra, bw_inter, total_devices,
        cp_memory_per_layer, device_capacity, num_layers,
        cp_algo, attention_heads, sp_enabled, num_kv_heads,
    )

    if p.cp_degree <= 1:
        return _cp_ok_result()

    if p.sp_enabled:
        return _cp_ok_result(
            is_valid=False,
            error_message=(
                f"Context Parallelism (cp={p.cp_degree}) and Sequence Parallelism "
                f"are incompatible and cannot be used together."
            ),
            unsupported_reason="CP+SP incompatible",
        )

    if p.seq_len % (p.cp_degree * 2) != 0:
        return _cp_ok_result(
            is_valid=False, seq_len_divisible=False,
            error_message=(
                f"Sequence length {p.seq_len} must be divisible by "
                f"cp_degree × 2 = {p.cp_degree * 2}. "
                f"Current remainder: {p.seq_len % (p.cp_degree * 2)}"
            ),
        )

    ulysses_err = _cp_check_ulysses_heads(p)
    if ulysses_err:
        return _cp_ok_result(
            is_valid=False, error_message=ulysses_err,
            unsupported_reason="Ulysses insufficient heads",
        )

    device_err = _cp_check_device_sufficiency(p)
    if device_err:
        return _cp_ok_result(
            is_valid=False, error_message=device_err, device_sufficient=False,
        )

    warnings, topology_feasible, topology_penalty, recommended_cp_max = (
        _cp_collect_warnings(p)
    )

    mem_result = _cp_check_memory(
        p, warnings, topology_feasible, topology_penalty, recommended_cp_max)
    if mem_result is not None:
        return mem_result

    return _cp_ok_result(
        warning_message="; ".join(warnings) if warnings else None,
        topology_feasible=topology_feasible,
        topology_penalty=topology_penalty,
        recommended_cp_max=recommended_cp_max,
    )

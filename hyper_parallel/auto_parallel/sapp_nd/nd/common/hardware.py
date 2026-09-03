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
"""hardware abstraction"""
# pylint: disable=E0102,W0125,W0612

from __future__ import annotations

import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger


class Type:
    """Machine type"""

    name: str
    levels: int  # levels in hierarchy
    level_bound_number: list[int]  # devices per level
    level_bandwidth: list[int]  # bandwidth (GB/s) per level

    def __init__(self, name, bounds, bandwidths):
        self.name = name
        self.level_bound_number = bounds
        self.level_bandwidth = bandwidths
        if len(bounds) != len(bandwidths):
            raise ValueError("bounds and bandwidths must have the same length")
        self.levels = len(bounds)

    level_efficiency: list[float]  # effective bandwidth utilization fraction per level
    level_latency: list[float]  # collective launch latency (seconds) per level

    def __init__(self, name, bounds, bandwidths,
                 efficiencies=None, latencies=None,
                 flop_coeffs=None, comm_scale_factor=1.0,
                 p2p_bandwidth=None, p2p_efficiency=None,
                 cp_overlap_ratio=0.5, p2p_ratio=0.002):
        self.name = name
        self.level_bound_number = bounds
        self.level_bandwidth = bandwidths
        if len(bounds) != len(bandwidths):
            raise ValueError("bounds and bandwidths must have the same length")
        self.levels = len(bounds)
        self.level_efficiency = efficiencies if efficiencies is not None else [0.005] * self.levels
        self.level_latency = latencies if latencies is not None else [0.00001] * self.levels
        self.flop_coeffs = flop_coeffs if flop_coeffs is not None else {}
        self.comm_scale_factor = comm_scale_factor
        self.p2p_bandwidth = p2p_bandwidth if p2p_bandwidth is not None else bandwidths
        self.p2p_efficiency = p2p_efficiency if p2p_efficiency is not None else self.level_efficiency
        self.cp_overlap_ratio = cp_overlap_ratio
        self.p2p_ratio = p2p_ratio

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def devices_below_level(self, level):
        """Number of devices below the given hierarchy level"""
        devices = 1
        for lvl in range(min(level, self.levels)):
            devices *= self.level_bound_number[lvl]
        return devices

    def intra_node_num(self):
        """Number of devices in a node"""
        return self.devices_below_level(1)

    def levels_used(self, device_number):
        """Number of hierarchy level used"""
        devices = 1
        for lvl in range(self.levels):
            if self.level_bound_number[lvl]:
                devices *= self.level_bound_number[lvl]
                if device_number <= devices:
                    return lvl
            else:
                return lvl
        return self.levels

    def level_assign(self, dp=1, tp=1, cp=1, pp=1, ep=1):
        """device assignment of the different parallel dimensions"""
        # EP borrows from DP, not counted in total devices; kept for
        # topology tracking only — callers should NOT pass ep > 1
        # unless they also account for EP-in-DP convention.
        device_number = dp * tp * cp * pp * ep
        logger.debug("DP = %d, TP = %d, EP = %d, CP = %d, PP = %d", dp, tp, ep, cp, pp)

    def level_assign(
        self, dp = 1, tp = 1, cp = 1, pp = 1,
        ep = 1, d_shard = 1,
    ):
        """device assignment of the different parallel dimensions"""
        # EP borrows from DP, not counted in total devices; kept for
        # topology tracking only — callers should NOT pass ep > 1
        # unless they also account for EP-in-DP convention.
        device_number = dp * tp * cp * pp * ep
        dp_original = dp
        logger.debug("DP = %d, TP = %d, EP = %d, CP = %d, PP = %d, d_shard = %d", dp, tp, ep, cp, pp, d_shard)
        assignment = {}
        assignment[Dim.TP] = []
        assignment[Dim.EP] = []
        assignment[Dim.CP] = []
        assignment[Dim.DP] = []
        assignment[Dim.PP] = []
        assignment[Dim.HSDP] = []
        for level in range(self.levels):
            bound = self.level_bound_number[level]
            if bound:
                level_device_number = min(device_number, bound)
                device_number = device_number // bound
            else:
                level_device_number = device_number
            remaining_devices = max(level_device_number, 1)

            tp_level = min(tp, remaining_devices)
            assignment[Dim.TP].append(tp_level)
            tp = tp // tp_level
            remaining_devices = remaining_devices // tp_level

            ep_level = min(ep, remaining_devices)
            assignment[Dim.EP].append(ep_level)
            ep = ep // ep_level
            remaining_devices = remaining_devices // ep_level

            cp_level = min(cp, remaining_devices)
            assignment[Dim.CP].append(cp_level)
            cp = cp // cp_level
            remaining_devices = remaining_devices // cp_level

            dp_level = min(dp, remaining_devices)
            assignment[Dim.DP].append(dp_level)
            dp = dp // dp_level
            remaining_devices = remaining_devices // dp_level

            pp_level = min(pp, remaining_devices)
            assignment[Dim.PP].append(pp_level)
            pp = pp // pp_level
            remaining_devices = remaining_devices // pp_level

            if d_shard > 1 and dp > 0:
                d_replicate = dp_original // d_shard
                if level == 0:
                    assignment[Dim.HSDP].append(1)
                elif level == 1 and d_replicate > 1:
                    assignment[Dim.HSDP].append(d_replicate)
                else:
                    assignment[Dim.HSDP].append(1)
            else:
                assignment[Dim.HSDP].append(0)

        return assignment


_FLOP_COEFFS_V4 = {
    "shard": {
        "fsdp": {
            "INTERCEPT": 3328.363,
            "INV_SG": -3853.095,
            "CROSS_INV_TP": 4850.947,
            "D_SHARD": -424.267,
            "TP": -179.776,
        },
        "hsdp": {
            "INTERCEPT": 1125.1909088274,
            "TP": -133.9082596473,
            "SG": 84.6312370881,
            "CROSS_AG_VOL": 1478.2295147790,
            "CROSS_INV_TP": -1263.7164058458,
            "INV_SG_D_REP": 800.1637134847,
            "AG_VOL_D_REP": -548.4088908024,
        },
    },
    "tp": {
        "fsdp": {"A": 0.4, "B": 0.03, "C": 1.0},
        "hsdp": {
            "INTERCEPT": -10989.6733755442,
            "INV_TP": 10942.7060704393,
            "INV_SG": -14430.2003957261,
            "LOG2_D_REP": 1811.4219667590,
            "CROSS_D_REP": 139.0623041155,
            "CROSS_SG": 130.2448654531,
            "M": 3662.7206094183,
        },
    },
    "dp": {
        "hsdp": {
            "INTERCEPT": 5521.1264772727,
            "TP": 766.5372727273,
            "DP": 417.1877556818,
            "SG": -269.1334943182,
            "AG_VOL": -7610.5452272727,
            "D_REP": -1019.9627272727,
            "CROSS_AG_VOL": 2412.6679545455,
        },
    },
    "comp": {
        "fsdp": {"A": 0.52, "B": 0.96},
        "hsdp": {
            "INTERCEPT": -3991.3092307692,
            "TP": -149.7816666667,
            "INV_TP": 6368.4866666667,
            "DP": 341.8965384615,
            "AG_VOL": -504.8923076923,
            "CROSS_INV_TP": -2491.9000000000,
            "LOG2_M": 5328.5446153846,
        },
    },
    "pp": {
        "hsdp": {
            "INTERCEPT": 10724.7073995013,
            "AG_VOL_D_REP": -815.5120186005,
            "CROSS_SG": -67.7183075139,
            "M": 658.0207571725,
        },
    },
}

Device_V4 = Type(
    name="V4", bounds=[8, None], bandwidths=[50, 10],
    efficiencies=[0.005, 0.01], latencies=[0.00001, 0.00002],
    flop_coeffs=_FLOP_COEFFS_V4,
    comm_scale_factor=1.0,
    p2p_bandwidth=[300, 25],
    p2p_efficiency=[0.7, 0.9],
)

# Device_A2 = Machine(devices_per_node=8, inter_node_bw=10, intra_node_bw=50)
Device_A2 = Type(name="A2", bounds=[8, None], bandwidths=[50, 10])
Device_A2 = Type(
    name="A2", bounds=[8, None], bandwidths=[50, 10],
    efficiencies=[0.005, 0.01], latencies=[0.00001, 0.00002],
    p2p_bandwidth=[300, 25],
    p2p_efficiency=[0.7, 0.9],
)

if False: Device_A3 = Type(name="A3", bounds=[16, 24, None], bandwidths=[200, 25, 10])
Device_A3 = Type(
    name="A3", bounds=[16, 24, None], bandwidths=[200, 25, 10],
    efficiencies=[0.005, 0.01, 0.01], latencies=[0.00001, 0.00002, 0.00002],
    comm_scale_factor=0.5,
    p2p_bandwidth=[300, 25, 25],
    p2p_efficiency=[0.7, 0.9, 0.9],
)

device_map = {
    "A2": Device_A2,
    "A3": Device_A3,
    "V4": Device_V4,
    "V100": Type(
        name="V100", bounds=[8, None], bandwidths=[50, 10],
        efficiencies=[0.005, 0.01], latencies=[0.00001, 0.00002],
        p2p_bandwidth=[300, 25],
        p2p_efficiency=[0.7, 0.9],
    ),
}


class Machine:
    """Hardware description"""

    number: int
    device: Type

    def __init__(self, number, device):
        self.number = number
        if isinstance(device, int):
            if device == 2:
                self.device = Device_A2
            elif device == 3:
                self.device = Device_A3
            else:
                raise ValueError(f"Ascend A{device} unknown")
        elif isinstance(device, str):
            if device not in device_map:
                raise ValueError(
                    f"Device {device} is not supported. "
                    f"Supported devices: {list(device_map.keys())}"
                )
            self.device = device_map[device]
        else:
            self.device = device

    def update_num_if_none(self, num):
        """Assign number of device if not already precised"""
        if self.number is None:
            self.number = num

    def pipeline_bound(self):
        """Return pipeline bound from hardware topology because as pipeline may currently not cross hierarchy levels"""
        max_bound = 1
        devices = self.number
        while devices > 1:
            max_bound = max(
                max_bound,
                devices
                // self.device.devices_below_level(
                    self.device.levels_used(devices)
                ),
            )
            devices = devices // 2
        # devices = self.devices_below_level(self.levels_used(device_number))
        # return device_number // devices
        return max_bound


def prime_factors(n):
    """Decompose n into a product of prime factors"""
    divisor = 2
    factors = []
    while n > 1:
        while n % divisor != 0:
            divisor += 1
        factors.append(divisor)
        n = n // divisor
    return factors


def all_factors_combinations(factors):
    """Computes all divisors from a prime factor list"""
    def rec_factors(n, factors):
        combinations = {n}
        for u in set(factors):
            remaining = factors.copy()
            remaining.remove(u)
            combinations = combinations.union(rec_factors(n * u, remaining))
        return combinations
    return rec_factors(1, factors)


def all_divisors(n, reverse=False, min_bound=1, max_bound=float("inf")):
    """Computes all divisors of an integer n"""
    divisors = sorted(
        all_factors_combinations(prime_factors(n)), reverse=reverse
    )
    div_in_bound = []
    for d in divisors:
        if min_bound <= d <= max_bound:
            div_in_bound.append(d)

    return div_in_bound


def from_prime_factors(factors):
    """Compute a number from its prime factor decomposition"""
    number = 1
    for f in factors:
        number *= f
    return number


def split_node(n, device):
    """Split decompositions into intra & inter devices"""
    devices_per_node = device.intra_node_num()
    nodes = prime_factors(max(1, n // devices_per_node))
    intra = prime_factors(min(n, devices_per_node))
    return [intra, nodes]


def unique_factors(factors):
    """Remove duplicates. Factors are sorted"""
    offset = 0
    for i, f in enumerate(factors[:-1]):
        j = i - offset
        if factors[j + 1] == f:
            factors.pop(j)
            offset += 1
    return factors


def highest_power_of_2_divisor(divisor_of):
    """Compute the highest number that is both a divisor of 'divisor_of' and a power of 2"""
    divisor = 1
    factors = prime_factors(divisor_of)
    for f in factors:
        if f == 2:
            divisor *= f
    return divisor


def get_cp_topology(tp_degree: int, cp_degree: int, device_per_node: int) -> tuple:
    """Determine CP topology and effective bandwidth.

    Args:
        tp_degree: Tensor parallelism degree.
        cp_degree: Context parallelism degree.
        device_per_node: Number of devices per node.

    Returns:
        Tuple of (topology_type, effective_bandwidth, is_intra_node).
        - topology_type: "intra-node" or "cross-node"
        - effective_bandwidth: Bandwidth in GB/s
        - is_intra_node: True if CP stays within node
    """
    total_devices_needed = tp_degree * cp_degree

    if total_devices_needed <= device_per_node:
        topology_type = "intra-node"
        is_intra_node = True
        effective_bandwidth = 300.0
    else:
        topology_type = "cross-node"
        is_intra_node = False
        effective_bandwidth = 25.0

    return topology_type, effective_bandwidth, is_intra_node


def get_cp_bandwidth(topology_type: str, device_type: str = "A2") -> float:
    """Get effective bandwidth for CP communication based on topology.

    Args:
        topology_type: "intra-node" or "cross-node"
        device_type: Device type string (e.g., "A2", "A3")

    Returns:
        Bandwidth in GB/s
    """
    device = device_map.get(device_type, Device_A2)

    if topology_type == "intra-node":
        return device.level_bandwidth[0] if device.level_bandwidth else 300.0
    return device.level_bandwidth[1] if len(device.level_bandwidth) > 1 else 25.0


def recommend_cp_max_by_attention(attention_type: str) -> int:
    """Recommend maximum CP degree based on attention type.

    Args:
        attention_type: "mla", "gqa", or "mha"

    Returns:
        Recommended maximum CP degree
    """
    attention_type_upper = attention_type.upper()
    if attention_type_upper == "MLA":
        return 16
    if attention_type_upper == "GQA":
        return 8
    return 4

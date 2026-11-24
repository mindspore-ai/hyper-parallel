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

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger


@dataclass(frozen=True)
class HostLink:
    """How fast a device moves activations to its host's memory and back, as offload prices it.

    A3's bandwidth and copy cost below are measured; every other figure is an
    assumption.
    Measure a link with every die of a job copying at once, as
    ``nd_golden/offload/bench_host_link.py`` does, since a job copies at its
    slowest die, and state it here or with ``run_nd --host_link_gibps``.

    Attributes:
        gib_per_s: The sustained copy bandwidth between the device and
            pinned host memory, in GiB/s, with one copy stream carrying both
            directions, as hyper_offload and the trainer's input swap run it.
        sustained_tflops: The device's sustained dense throughput at the
            training precision, in TFLOP/s, which turns a copy's seconds into
            the performance estimate's units where no calibration states
            how long one unit takes.
        overlap: The share of the forward time the copy stream may take:
            1 where the copies meet nothing, less where the step's compute
            or its collectives share what the copies need. 0.8 is an
            assumption; ``run_nd --host_link_overlap`` states another.
        ms_per_unit: The milliseconds one unit of the forward the
            performance estimate prices stands for: a calibration round's
            FORWARD ratio where it measured the forward on its own, else its
            COMPUTE ratio. Stated, it turns a copy's seconds into the
            estimate's units in place of *sustained_tflops*: the estimate
            counts a step's matmul FLOPs, a fraction of its work, so a
            device rate makes every forward look several times shorter than
            it runs; and it may split the forward from the backward
            otherwise than the device does, so that COMPUTE can make the
            forward the copies hide in look longer.
        copy_cost_ms_per_gib: What the copies cost the step, in milliseconds
            a GiB moved either way: the time the compute and the collectives
            beside them lose to the copies, which hide in the forward but
            still take part of what those need. 0 prices offload as free.
        score_ms_per_unit: The milliseconds one unit of the score stands
            for, a calibration round's COMPUTE ratio, which turns that cost
            into the score's units; unset, the cost converts as a copy's
            seconds do.
    """

    gib_per_s: float
    sustained_tflops: float
    overlap: float = 0.8
    ms_per_unit: Optional[float] = None
    copy_cost_ms_per_gib: float = 0.0
    score_ms_per_unit: Optional[float] = None

    def __post_init__(self) -> None:
        """Refuse a figure that cannot price a copy."""
        if self.gib_per_s <= 0 or self.sustained_tflops <= 0 or not 0 < self.overlap <= 1:
            raise ValueError(f"a host link needs positive figures and an overlap in (0, 1], not {self}")
        if self.ms_per_unit is not None and self.ms_per_unit <= 0:
            raise ValueError(f"a calibrated host link needs a positive ms_per_unit, not {self}")
        if self.copy_cost_ms_per_gib < 0 or (self.score_ms_per_unit is not None and self.score_ms_per_unit <= 0):
            raise ValueError(f"a host link needs a copy cost of 0 or more and a positive score_ms_per_unit, not {self}")

    def seconds_per_byte(self) -> float:
        """The seconds one byte takes over the link, one way."""
        return 1.0 / (self.gib_per_s * 2 ** 30)

    def cost_seconds_per_byte(self) -> float:
        """The step time one byte moved costs, either way, in seconds."""
        return self.copy_cost_ms_per_gib / 1000.0 / 2 ** 30

    def flops_per_second(self) -> float:
        """The device's sustained FLOP/s."""
        return self.sustained_tflops * 10 ** 12

    @classmethod
    def of(cls, link: Optional["HostLink"], figures: Mapping[str, Any]) -> "HostLink":
        """*link* with the *figures* stated replacing its own, those ``None`` left as they are.

        Args:
            link: A device's link, or ``None`` for a device that states none.
            figures: Any of the fields, ``None`` for one not stated.

        Raises:
            ValueError: Without *link*, where *figures* leave out the
                bandwidth or the throughput.
        """
        stated = {name: value for name, value in figures.items() if value is not None}
        if link is not None:
            return replace(link, **stated)
        if "gib_per_s" not in stated or "sustained_tflops" not in stated:
            raise ValueError("it states no host link; give both --host_link_gibps and --sustained_tflops for -ao")
        return cls(**stated)

    def units_per_second(self, bytes_p: float) -> float:
        """The performance estimate's units a second, calibrated where *ms_per_unit* is stated.

        Args:
            bytes_p: The training precision's bytes, by which the estimate
                multiplies its FLOPs.
        """
        if self.ms_per_unit is not None:
            return 1000.0 / self.ms_per_unit
        return self.flops_per_second() * bytes_p

    def score_units_per_second(self, bytes_p: float) -> float:
        """The score's units a second, for the copies' cost: *score_ms_per_unit*'s, else as a copy converts.

        Args:
            bytes_p: The training precision's bytes, by which the estimate
                multiplies its FLOPs.
        """
        if self.score_ms_per_unit is not None:
            return 1000.0 / self.score_ms_per_unit
        return self.units_per_second(bytes_p)


class Type:
    """Machine type"""

    name: str
    levels: int  # levels in hierarchy
    level_bound_number: list[int]  # devices per level
    level_bandwidth: list[int]  # bandwidth (GB/s) per level
    host_link: Optional[HostLink]  # the copy path to the host, for offload

    def __init__(self, name, bounds, bandwidths, host_link=None):
        self.name = name
        self.level_bound_number = bounds
        self.level_bandwidth = bandwidths
        if len(bounds) != len(bandwidths):
            raise ValueError("bounds and bandwidths must have the same length")
        self.levels = len(bounds)
        self.host_link = host_link

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
        # Order matters: each dim takes the devices the previous ones left.
        degrees = {Dim.TP: tp, Dim.EP: ep, Dim.CP: cp, Dim.DP: dp, Dim.PP: pp}
        assignment = {dim: [] for dim in degrees}
        for level in range(self.levels):
            bound = self.level_bound_number[level]
            if bound:
                remaining_devices = max(min(device_number, bound), 1)
                device_number = device_number // bound
            else:
                remaining_devices = max(device_number, 1)
            for dim, degree in degrees.items():
                dim_level = min(degree, remaining_devices)
                assignment[dim].append(dim_level)
                degrees[dim] = degree // dim_level
                remaining_devices = remaining_devices // dim_level

        return assignment


# A3's link is measured: 6.92 GiB/s is the slowest die of a two-node job copying
# to the host on 7 October 2026, where one node alone measured 10.33 to 13.29
# (nd_golden/offload/host_link_*_1007.txt). The copies to the host bind, in the
# forward; those back run in the longer backward. Its cost is measured too:
# inside the Demo 2 step on 8 October, four nodes, 4 GiB each way at layer 0's
# points cost the step 22.5 ms, 2.8 ms a GiB moved, where that job's slowest die
# copied 8.61 GiB/s alone (nd_golden/offload/probe_4node_1008.txt). The other links are
# placeholders: 16 GiB/s is hyper_offload's own default before it profiles the
# link. The throughputs are about half of each device's dense BF16 or FP16
# peak, which a calibration replaces (HostLink.ms_per_unit).
# Device_A2 = Machine(devices_per_node=8, inter_node_bw=10, intra_node_bw=50)
Device_A2 = Type(
    name="A2", bounds=[8, None], bandwidths=[50, 10], host_link=HostLink(gib_per_s=16.0, sustained_tflops=140.0)
)
Device_A3 = Type(
    name="A3",
    bounds=[16, 24, None],
    bandwidths=[200, 25, 10],
    host_link=HostLink(gib_per_s=6.92, sustained_tflops=160.0, copy_cost_ms_per_gib=2.8),
)
device_map = {
    "A2": Device_A2,
    "A3": Device_A3,
    "V100": Type(
        name="V100", bounds=[8, None], bandwidths=[50, 10], host_link=HostLink(gib_per_s=12.0, sustained_tflops=60.0)
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

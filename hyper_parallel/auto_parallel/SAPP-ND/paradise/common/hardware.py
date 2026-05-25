# Copyright 2025 Huawei Technologies Co., Ltd
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

import paradise.dimensions as Dim
from paradise.logger import logger


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
        assert len(bounds) == len(bandwidths)
        self.levels = len(bounds)

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

    def level_assign(self, dp=1, tp=1, cp=1, pp=1):
        """device assignment of the different parallel dimensions"""
        device_number = dp * tp * cp * pp
        logger.debug("DP = %d, TP = %d, CP = %d, PP = %d", dp, tp, cp, pp)
        assignment = {}
        assignment[Dim.TP] = []
        assignment[Dim.CP] = []
        assignment[Dim.DP] = []
        assignment[Dim.PP] = []
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

        return assignment


# Device_910B = Machine(devices_per_node=8, inter_node_bw=10, intra_node_bw=50)
Device_910B = Type(name="910B", bounds=[8, None], bandwidths=[50, 10])
Device_A3 = Type(
    name="A3", bounds=[16, 24, None], bandwidths=[200, 25, 10]
)


class Machine:
    """Hardware description"""

    number: int
    device: Type

    def __init__(self, number, device):
        self.number = number
        if isinstance(device, int):
            if device == 2:
                self.device = Device_910B
            elif device == 3:
                self.device = Device_A3
            else:
                raise ValueError(f"Ascend A{device} unknown")
        else:
            self.device = device

    def update_num_if_none(self, num):
        """Assign number of device if not already precised"""
        if self.number is None:
            self.number = num

    def pipeline_bound(self):
        """Return pipeline bound from hardware topology because
        As pipeline may currently not cross hierarchy levels"""
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
    """Compute the highest number that is both
    a divisor of 'divisor_of' and a power of 2"""
    divisor = 1
    factors = prime_factors(divisor_of)
    for f in factors:
        if f == 2:
            divisor *= f
    return divisor

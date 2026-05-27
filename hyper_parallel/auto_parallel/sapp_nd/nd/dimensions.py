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
"""parallel dimensions"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger

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
        if (
            self.has_dim(PP)
            and self.dims_val[PP] > 1
            and self.has_dim(MBN)
            and self.dims_val[MBN] > 1
        ):
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

    def is_valid(self):
        """Check if all dimensions values are valid"""
        if MBN in self.dims_val and PP in self.all_dims:
            valid = self.dims_val[MBN] >= self.dims_val[PP]
            valid = valid and not (
                self.dims_val[PP] == 1 and self.dims_val[MBN] > 1
            )
            if not valid:
                logger.warning("PP and MBN were deemed not suitable")
                return False
        if TP in self.all_dims and not (
            self.dims_val[TP] & (self.dims_val[TP] - 1) == 0
        ):
            logger.warning("%s must be a power of 2", str(TP))
            return False
        for d in self.dims_val:
            if not d.is_valid(self.dims_val[d]):
                logger.warning("Dimension %d is not valid", d)
                return False
        if SP in self.all_dims and CP in self.all_dims:
            if self.dims_val[SP] and self.dims_val[CP] > 1:
                logger.warning("SP & CP cannot coexist")
                return False
        if OP in self.all_dims:
            op = self.dims_val[OP]
            if not op & (op - 1) == 0:
                logger.warning("OP %d must be a power of 2", op)
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

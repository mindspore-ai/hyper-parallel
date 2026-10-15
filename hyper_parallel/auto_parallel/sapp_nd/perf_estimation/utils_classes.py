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
"""Custom Config options"""
from enum import Enum, auto


class RatioType(Enum):
    "comm/comp"

    COMM_ONLY = auto()
    COMPUTE_ONLY = auto()
    STATIC = auto()
    DYNAMIC = auto()


class PerformanceType(Enum):
    """What an estimate is.

    FLOP is the live one, and a relative score: ND weighs each part's work
    and ranks strategies by it, and the ratios fitted on a measured round
    turn the parts into milliseconds, so no rate and no device peak prices
    a part (a host link's sustained_tflops only converts an offload copy
    where no ratio does).  That is a decision rather than a gap (register A5):
    ND stays a relative scorer while it can.  TIME, which would divide by a
    device's rates, has no production caller.
    """

    FLOP = auto()
    TIME = auto()  # no production caller: ND is a relative scorer (A5)


class P2PCommType(Enum):
    """flags"""

    NONE = auto()
    MANUAL = auto()


class RecType(Enum):
    """flags"""

    NONE = auto()
    WITH = auto()
    COMM_ONLY = auto()
    COMPUTE_ONLY = auto()


class NetworkLevel(Enum):
    """device network"""

    NODE = auto()
    CLUSTER = auto()


class CustomConfig:
    r"""Custom Config for Base Performance Estimator"""

    def __init__(
        self,
        rtype=RatioType.DYNAMIC,
        #  ttype = PerformanceType.TIME,
        ttype=PerformanceType.FLOP,
        ptype=P2PCommType.NONE, # MANUAL,
        retype=RecType.WITH,  # recompute re-runs compute and communication
    ):
        self.rtype = rtype
        self.ttype = ttype
        self.ptype = ptype
        self.retype = retype

    def __repr__(self):
        return (
            f"CustomConfig(rtype={self.rtype}, "
            f"ttype={self.ttype}, "
            f"ptype={self.ptype}, "
            f"retype={self.retype})"
        )

    def __str__(self):
        return self.__repr__()

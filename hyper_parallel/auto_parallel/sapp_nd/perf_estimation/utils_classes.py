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
    "metric"

    FLOP = auto()
    TIME = auto()  # to fix


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
        ptype=P2PCommType.NONE, #MANUAL,
        retype=RecType.COMPUTE_ONLY,
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

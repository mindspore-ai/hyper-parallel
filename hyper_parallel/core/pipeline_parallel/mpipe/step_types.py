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
"""Schedule step types for MPipe Transpose.

These are kept out of the core ``MetaStepType`` and dispatched through the
schedule's generic custom-function registry
(:meth:`PipelineScheduleRuntime.register_custom_function`), so the core
scheduler stays free of MPipe-specific steps.  ``MetaStep`` does not validate
its ``type``, and the runtime resolves custom steps by ``_custom_fn_map.get``,
so a separate enum composes cleanly with the built-in ``MetaStepType`` steps.
"""
from enum import Enum, auto


class MpipeStepType(Enum):
    """The ``MPIPE_*`` steps layered around the body schedule.

    A trainable preprocess broadcasts its params (``MPIPE_PARAM_BROADCAST``),
    each owning rank runs the transposed forward (``MPIPE_TRANSPOSE_FWD``) and
    ships its output (``MPIPE_FWD_SEND`` / ``MPIPE_FWD_RECV``) and — for the
    centralized recompute backward — its input (``MPIPE_GRAPH_SEND`` /
    ``MPIPE_GRAPH_RECV``); stage 0 recomputes the backward
    (``MPIPE_TRANSPOSE_BWD``).
    """

    MPIPE_PARAM_BROADCAST = auto()
    MPIPE_TRANSPOSE_FWD = auto()
    MPIPE_FWD_SEND = auto()
    MPIPE_FWD_RECV = auto()
    MPIPE_GRAPH_SEND = auto()
    MPIPE_GRAPH_RECV = auto()
    MPIPE_TRANSPOSE_BWD = auto()

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
"""Debug utilities for DTensor.

Public API:
    CommDebugMode — context manager that traces DTensor ops and collectives.

Internal modules (_call_records, _collective_tracer, _module_tracker) are implementation details and
should not be imported directly by user code.
"""
from hyper_parallel.core.dtensor.debug._comm_debug_mode import CommDebugMode

__all__ = ["CommDebugMode"]

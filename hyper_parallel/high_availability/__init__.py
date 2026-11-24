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
"""Opt-in process recovery; importing this module does not import MindIO."""

from importlib import import_module
from typing import TYPE_CHECKING

from hyper_parallel.high_availability.config import WorkerRecoveryConfig

if TYPE_CHECKING:
    from hyper_parallel.high_availability.recovery_session import RecoverySession
    from hyper_parallel.high_availability.recovery_state import TrainState

__all__ = ["WorkerRecoveryConfig", "RecoverySession", "TrainState"]


def __getattr__(name):  # pylint: disable=invalid-name
    """Load runtime APIs only when requested, keeping configuration imports light."""
    if name == "RecoverySession":
        return import_module("hyper_parallel.high_availability.recovery_session").RecoverySession
    if name == "TrainState":
        return import_module("hyper_parallel.high_availability.recovery_state").TrainState
    raise AttributeError(name)

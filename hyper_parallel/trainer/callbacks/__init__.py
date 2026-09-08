# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trainer Callbacks module.

Provides callback system for customizing trainer behavior at various stages of training.

Moved from ``auto_models/trainer/callbacks`` in stage 7 (05 §15.11 step 2);
``TrainerState`` is defined in ``hyper_parallel.trainer.state`` and
re-exported here. The temporary ``TempLogCallback`` alias was dropped —
``TqdmCallback`` is the logging callback it pointed at.
"""

from hyper_parallel.trainer.state import TrainerState
from .base import Callback
from .environ_meter_callback import EnvironMeterCallback
from .evaluate_callback import EvaluateCallback
from .garbage_collection_callback import GarbageCollectionCallback
from .logging_callback import LoggingCallback
from .checkpoint_callback import CheckpointerCallback
from .profiling_callback import ProfilingCallback
from .tqdm_callback import TqdmCallback


__all__ = [
    "Callback",
    "EnvironMeterCallback",
    "EvaluateCallback",
    "GarbageCollectionCallback",
    "LoggingCallback",
    "CheckpointerCallback",
    "ProfilingCallback",
    "TqdmCallback",
    "TrainerState",
]

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
"""MPipe Transpose: a self-contained Interleaved-1F1B variant that transposes
the preprocess block to shrink the pipeline warmup bubble.

The package is a plugin on top of the generic pipeline schedule runtime: it
defines its own :class:`MpipeStepType` steps (kept out of the core
``MetaStepType``) and registers their handlers through
``PipelineScheduleRuntime.register_custom_function``, so the core scheduler
carries no MPipe-specific code.
"""

__all__ = ["MpipeStepType", "MPipeTransposeExecutorBase", "ScheduleMPipeTranspose"]

from hyper_parallel.core.pipeline_parallel.mpipe.step_types import MpipeStepType
from hyper_parallel.core.pipeline_parallel.mpipe.executor_base import MPipeTransposeExecutorBase
from hyper_parallel.core.pipeline_parallel.mpipe.schedule import ScheduleMPipeTranspose

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
"""Training components: step_scheduler, callback, grad_accum, signal_handler, rng."""

from hyper_models.components.training.callback import (
    CallbackManager,
    CheckpointCallback,
    EvaluateCallback,
    GCCallback,
    LoggingCallback,
    SIGTERMHandler,
    StepState,
    TqdmCallback,
    TrainingCallback,
    WandbCallback,
    build_callback_manager,
)
from hyper_models.components.training.grad_accum import (
    AutoMFU,
    _dp_all_reduce_avg,
    _dp_cp_all_reduce_sum,
    _update_latest_symlink,
    calculate_mfu,
    filter_forward_kwargs,
    get_sync_ctx,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
    set_requires_gradient_sync,
    setup_magi,
)
from hyper_models.components.training.rng import StatefulRNG
from hyper_models.components.training.signal_handler import DistributedSignalHandler
from hyper_models.components.training.step_scheduler import (
    StepScheduler,
    StepSchedulerConfig,
)

__all__ = [
    "StepScheduler",
    "StepSchedulerConfig",
    "StepState",
    "TrainingCallback",
    "CallbackManager",
    "CheckpointCallback",
    "EvaluateCallback",
    "LoggingCallback",
    "TqdmCallback",
    "WandbCallback",
    "GCCallback",
    "SIGTERMHandler",
    "build_callback_manager",
    "get_sync_ctx",
    "prepare_for_grad_accumulation",
    "prepare_for_final_backward",
    "prepare_after_first_microbatch",
    "scale_grads_and_clip_grad_norm",
    "set_requires_gradient_sync",
    "AutoMFU",
    "calculate_mfu",
    "filter_forward_kwargs",
    "setup_magi",
    "_dp_cp_all_reduce_sum",
    "_dp_all_reduce_avg",
    "_update_latest_symlink",
    "DistributedSignalHandler",
    "StatefulRNG",
]
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
"""MindSpore activation checkpoint implementations."""
from hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    ckpt_wrapper,
)
from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import (
    ActivationWrapper,
    AsyncSaveOnCpu,
    SwapWrapper,
    base_check_fn,
    swap_wrapper,
    swap_tensor_wrapper,
)
from hyper_parallel.platform.mindspore.activation_checkpoint.sac import (
    create_selective_checkpoint_contexts
)

__all__ = [
    "CheckpointWrapper",
    "ckpt_wrapper",
    "ActivationWrapper",
    "AsyncSaveOnCpu",
    "SwapWrapper",
    "base_check_fn",
    "swap_wrapper",
    "swap_tensor_wrapper",
    "create_selective_checkpoint_contexts",
]

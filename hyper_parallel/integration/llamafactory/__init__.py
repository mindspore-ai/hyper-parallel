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
"""HyperParallel integration APIs for LlamaFactory."""

from hyper_parallel.core.fully_shard.api import HSDPModule, hsdp_sync_stream
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.integration.llamafactory.activation import (
    find_transformer_blocks,
    setup_activation_optimization,
)
from hyper_parallel.integration.llamafactory.utils import (
    HyperParallelArguments,
    export_to_hf_format,
    fsdp2_prepare_model,
    load_hsdp_model,
    load_hsdp_optimizer_and_scheduler,
    save_hsdp_checkpoint,
    wrap_optimizer_with_skip_dtensor_dispatch,
)

__all__ = [
    "HSDPModule",
    "HyperParallelArguments",
    "clip_grad_norm_",
    "export_to_hf_format",
    "find_transformer_blocks",
    "fsdp2_prepare_model",
    "hsdp_sync_stream",
    "load_hsdp_model",
    "load_hsdp_optimizer_and_scheduler",
    "save_hsdp_checkpoint",
    "setup_activation_optimization",
    "wrap_optimizer_with_skip_dtensor_dispatch",
]

# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MoE Expert-Parallel demo modules.

Import from the ``examples/torch/moe`` directory (see ``expert_parallel_example.py``
for usage).
"""
from .model import MoEDemoConfig, MoEDemoModel
from .parallelize import (
    broadcast_state_dict_from_rank0,
    build_ep_mesh,
    parallelize_moe_ep,
    parallelize_moe_tp,
)
from .pipeline import (
    MicrobatchLossPipelineStage,
    build_moe_pp_chunk,
    build_pipeline_stage,
    count_moe_parameters,
    extract_stage_state_dict,
    layer_range_for_pp_stage,
    split_batch_dim0,
)

__all__ = [
    "MicrobatchLossPipelineStage",
    "MoEDemoConfig",
    "MoEDemoModel",
    "build_ep_mesh",
    "build_moe_pp_chunk",
    "build_pipeline_stage",
    "count_moe_parameters",
    "broadcast_state_dict_from_rank0",
    "extract_stage_state_dict",
    "layer_range_for_pp_stage",
    "parallelize_moe_ep",
    "parallelize_moe_tp",
    "split_batch_dim0",
]

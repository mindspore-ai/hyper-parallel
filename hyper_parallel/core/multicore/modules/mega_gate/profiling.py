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
"""HyperMegaGate row-shard semantics for the generic cycle profiler."""

from hyper_parallel.core.multicore.profiler.profiling import (
    GRAPH_STAGE_DESC_BASE,
    _TaskExecutionMode,
    _set_mega_kernel_profile_metadata,
)
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC


MEGA_GATE_PROFILE_OWNER_LABEL = "TokenRowShard"
MEGA_GATE_ROUTE_STAGE_NAMES = (
    "Softplus",
    "Sqrt",
    "AddBias",
    "TopK",
    "Gather",
    "ReduceSum",
    "AddEpsilon",
    "Div",
    "MulScale",
    "CastIndex",
)

MEGA_GATE_ROUTE_GRAD_STAGE_NAMES = (
    "MulsScaleGrad",
    "BroadcastDenominator",
    "Neg",
    "DivSelected",
    "DivSelectedRatio",
    "MulCross",
    "DivDirect",
    "ReduceSum",
    "BroadcastRowSum",
    "AddSelectedGrad",
    "ZerosLike",
)

MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES = (
    "MulsScaleGrad",
    "ZerosLike",
)


def _configure_mega_gate_profile_metadata(
    runtime_config: RuntimeConfigC,
    stage_names: tuple[str, ...],
    kernel_name: str,
) -> None:
    """Attach MegaGate AIV pipeline stages to one runtime configuration."""
    profile_stage_names = {
        GRAPH_STAGE_DESC_BASE + stage_id: stage_name
        for stage_id, stage_name in enumerate(stage_names)
    }
    _set_mega_kernel_profile_metadata(
        runtime_config,
        kernel_name=kernel_name,
        owner_label=MEGA_GATE_PROFILE_OWNER_LABEL,
        stage_names=profile_stage_names,
        task_stage_names=dict(enumerate(stage_names)),
        execution_mode=_TaskExecutionMode.AIV_PIPELINE,
    )
    for task_id in range(len(stage_names)):
        runtime_config.all_tasks[task_id].profile_desc_id = GRAPH_STAGE_DESC_BASE + task_id

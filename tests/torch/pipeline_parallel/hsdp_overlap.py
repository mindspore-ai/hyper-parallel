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
"""Worker-only HSDP overlap recipe with deterministic parameter communication."""
import os

import torch
import torch.distributed as dist

from hyper_parallel.core.pipeline_parallel import CommComputeOverlap, MetaStepType


def initialize_device() -> torch.device:
    """Select an explicit test device; CPU is available for Gloo correctness checks."""
    device_type = os.environ.get("HYPER_PARALLEL_PP_DEVICE_TYPE", "npu")
    if device_type not in ("npu", "cuda", "cpu"):
        raise ValueError(f"Unsupported pipeline test device: {device_type}")
    if device_type == "npu":
        # The extension registers HCCL; CUDA/CPU workers need no NPU installation.
        import torch_npu
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(device_type, local_rank) if device_type != "cpu" else torch.device("cpu")
    if device_type != "cpu":
        torch.accelerator.set_device_index(device)
    if not dist.is_initialized():
        dist.init_process_group(backend={"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[device_type])
    return device


def register_hsdp_overlap(schedule, stages, device: torch.device) -> None:
    """Use one parameter unit per stage, with no cross-thread parameter prefetch.

    Different stages submit backward then forward AllGather before starting
    their compute threads. Same-stage compute runs sequentially while retaining
    parameters for forward; its gradient communication remains asynchronous.
    """
    stage_by_index = {stage.stage_index: stage for stage in stages}
    overlap = CommComputeOverlap()

    def execute_overlap(step, context):
        backward_step, forward_step = step.sub_steps
        backward_stage = stage_by_index[backward_step.stage_index]
        forward_stage = stage_by_index[forward_step.stage_index]
        controller = context.fsdp_comm_controller

        def execute_backward():
            if device.type != "cpu":
                torch.accelerator.set_device_index(device)
            handle = controller.unshard_handles.pop(backward_stage.submodule.hsdp_scheduler, None)
            if handle is not None:
                handle.wait()
            context.schedule._exec_step(backward_step, context.arg_mbs, context.kwarg_mbs, context.losses)

        def execute_forward():
            if device.type != "cpu":
                torch.accelerator.set_device_index(device)
            handle = controller.unshard_handles.pop(forward_stage.submodule.hsdp_scheduler, None)
            if handle is not None:
                handle.wait()
            context.schedule.execute_fwd_leaf(forward_step, context.arg_mbs, context.kwarg_mbs, context.losses)

        if backward_stage is forward_stage:
            controller.retained_backward_units.add(backward_stage.submodule.hsdp_scheduler)
            execute_backward()
            execute_forward()
            controller.retained_backward_units.remove(backward_stage.submodule.hsdp_scheduler)
        else:
            for stage in (backward_stage, forward_stage):
                controller.unshard_handles[stage.submodule.hsdp_scheduler] = stage.submodule.unshard(async_op=True)
            overlap.run(fwd_fn=execute_forward, bwd_fn=execute_backward)

    schedule.register_custom_function(MetaStepType.OVERLAP_B_F, execute_overlap)

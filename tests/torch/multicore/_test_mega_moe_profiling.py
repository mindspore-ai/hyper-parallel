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
"""Two-rank Torch worker for a steady-state MegaMoe forward profiling capture."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler, schedule as diagnostic_schedule
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import (
    TASK_TYPE_DESC_BASE,
    _configure_trace_metadata,
)
from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID,
    INVALID_PROFILE_DESC_ID,
    RuntimeConfigC,
    TaskType,
)
from tests.torch.multicore import _test_mega_moe as common


_PROFILE_SHAPE = common.MoeShape(
    # The legacy TP=2 test partitions global seq_size=1024 into 512 tokens per rank.
    local_num_tokens=512,
    hidden_size=5120,
    intermediate_size=2048,
    num_experts=16,
    top_k=8,
    ep_size=2,
)


def _build_profile_layer() -> MegaMoeExperts:
    """Construct the representative two-rank MegaMoe layer."""
    return MegaMoeExperts(
        local_num_tokens=_PROFILE_SHAPE.local_num_tokens,
        hidden_size=_PROFILE_SHAPE.hidden_size,
        intermediate_size=_PROFILE_SHAPE.intermediate_size,
        num_experts=_PROFILE_SHAPE.num_experts,
        top_k=_PROFILE_SHAPE.top_k,
        ep_size=_PROFILE_SHAPE.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=common.DEVICE, dtype=torch.bfloat16)


def _expected_forward_records(runtime_config: RuntimeConfigC) -> Counter:
    """Enumerate every profiled task phase in the plan's round-robin worker queues."""
    cube_workers = int(runtime_config.num_workers) // 2
    expected = Counter()
    queues = (("AIC", runtime_config.cube_task_indices), ("AIV", runtime_config.vector_task_indices))
    for queue_index, (core_type, indices) in enumerate(queues):
        for position, task_id in enumerate(indices[:runtime_config.task_index_num[queue_index]]):
            task = runtime_config.all_tasks[task_id]
            worker = position % cube_workers
            block_id = worker if core_type == "AIC" else 2 * worker + 1
            desc_id = int(task.profile_desc_id)
            if desc_id == INVALID_PROFILE_DESC_ID:
                desc_id = TASK_TYPE_DESC_BASE + int(task.task_type)
            expected[(core_type, block_id, task_id, desc_id)] += 1
            if task.dependent_event != EVENT_INVALID_ID:
                expected[(core_type, block_id, task_id, 0x10000)] += 1
            if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                expected[(core_type, block_id, task_id, 0x10007)] += 1
    return expected


def _assert_complete_forward_trace(trace: dict, runtime_config: RuntimeConfigC) -> None:
    """Reject incomplete or ambiguously named device records."""
    metadata = trace["megaKernelCycleTrace"]
    internal_events = [event for event in trace["traceEvents"] if event.get("cat") == "MegaKernelInternal"]
    expected = _expected_forward_records(runtime_config)
    actual = Counter(
        tuple(event["args"][field] for field in ("core_type", "block_id", "task_id", "desc_id"))
        for event in internal_events
    )
    expected_count = sum(expected.values())

    assert metadata["invocationCount"] == 1, f"rank={common.RANK}: expected one forward invocation, got {metadata}."
    assert (
        metadata["recordCount"] == len(internal_events) == expected_count
    ), f"rank={common.RANK}: expected {expected_count} records, got {len(internal_events)}: {metadata}."
    assert metadata["droppedRecordCount"] == 0, f"rank={common.RANK}: profiling dropped records: {metadata}."
    assert actual == expected, (
        f"rank={common.RANK}: missing records={list((expected - actual).items())[:8]}, "
        f"unexpected records={list((actual - expected).items())[:8]}."
    )
    assert all(
        0 < event["args"]["start_cycle"] <= event["args"]["end_cycle"]
        for event in internal_events
    ), f"rank={common.RANK}: profiling exported an invalid cycle record."
    assert {event["args"]["direction"] for event in internal_events} == {"forward"}
    event_names = {event["name"] for event in internal_events}
    assert any("GMM1_WaitDependency" in name for name in event_names)
    assert any("GMM1_TriggerEvent" in name for name in event_names)
    assert any("GMM2_WaitDependency" in name for name in event_names)
    assert any("GMM2_TriggerEvent" in name for name in event_names)


def test_mega_moe_forward_profiling() -> None:
    """Feature: MegaKernel diagnostics.
    Description: Warm up, synchronize ranks, and export one representative forward trace.
    Expectation: Mega moe forward profiling.
    """
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(_PROFILE_SHAPE)
    hidden_states, _ = common.make_data(_PROFILE_SHAPE)
    layer = _build_profile_layer()
    schedule = diagnostic_schedule(wait=0, warmup=0, active=1, repeat=1)
    try:
        with torch.no_grad():
            layer(
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert=tokens_per_expert,
            )
            torch.npu.synchronize()
            dist.barrier()

            with multicore_profiler.mega_kernel_profile(
                schedule=schedule,
                detailed_task_names=True,
            ) as profiler:
                layer(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    tokens_per_expert=tokens_per_expert,
                )
                profiler.step()

        result_dir = os.getenv("HP_MEGA_MOE_PROFILE_RESULT_DIR")
        if not result_dir:
            raise ValueError("HP_MEGA_MOE_PROFILE_RESULT_DIR must name the trace output directory.")
        output_path = Path(result_dir) / f"rank{common.RANK}_mega_kernel_trace.json"
        trace = profiler.export_chrome_trace(output_path)
        # Inspect the bound plan only to derive test expectations for the workers that actually execute.
        plan = layer._resource_group.resources.plan  # pylint: disable=protected-access
        _, forward_config, _, _ = _build_runtime_artifacts(plan.spec)
        _configure_trace_metadata(forward_config, plan.fwd_diagnostics.metadata)
        _assert_complete_forward_trace(trace, forward_config)
        print(f"MEGA_KERNEL_PROFILE_TRACE={output_path}")
    finally:
        layer.close()
    dist.barrier()

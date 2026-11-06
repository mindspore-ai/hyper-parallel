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
"""Rank-synchronized failure propagation and timing."""
import traceback
from typing import Any, Mapping, Optional
import torch.distributed as dist

def run_synchronized(operation: str, callback: Any) -> Any:
    """Run local work and make its failure visible on every training rank."""
    result = None
    local_error = None
    try:
        result = callback()
    except Exception as error:  # pylint: disable=W0718
        local_error = error
    world_size = dist.get_world_size()
    if world_size <= 1:
        if local_error is not None:
            raise local_error
        return result
    errors: list[Optional[str]] = [None] * world_size
    local_traceback = (
        None
        if local_error is None
        else "".join(
            traceback.format_exception(
                type(local_error), local_error, local_error.__traceback__
            )
        )
    )
    dist.all_gather_object(errors, local_traceback)
    if any(error is not None for error in errors):
        raise RuntimeError(f"{operation} failed on at least one rank: {errors}")
    return result

def add_step_timing_metrics(diagnostic_metrics: dict[str, float], timings: Mapping[str, float]) -> None:
    """Record per-stage duration and per-device token throughput."""
    diagnostic_metrics.update({f"timing_s/{name}": value for name, value in timings.items()})
    diagnostic_metrics["perf/time_per_step"] = timings["step"]
    total_tokens = diagnostic_metrics.get("training/total_tokens", 0.0)
    diagnostic_metrics["perf/total_num_tokens"] = total_tokens
    diagnostic_metrics["perf/tokens_per_second_per_device"] = total_tokens / max(
        timings["step"] * dist.get_world_size(), 1.0e-9,
    )

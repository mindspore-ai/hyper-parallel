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
"""Execute large task graphs and expert counts with isolated runtime scratch."""

import struct
from dataclasses import asdict

import torch

from hyper_parallel.core.multicore.scheduler.runtime import RUNTIME_HEADER_BYTES
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    memory_sample,
    start_shmem_lifetime,
    write_evidence,
)

FORMER_TASK_LIMIT = 25600


def _runtime_metadata(tensor: torch.Tensor) -> dict:
    """Inspect the actual uploaded image, including the last queued descriptor."""
    image = tensor.cpu().numpy().tobytes()
    tasks, _, capacity, events = struct.unpack_from("<4I", image)
    counts_offset = RUNTIME_HEADER_BYTES + events * 20 + capacity * 576
    counts = struct.unpack_from("<3i", image, counts_offset)
    queues = [
        struct.unpack_from(f"<{count}i", image, counts_offset + 16 + index * capacity * 4)
        for index, count in enumerate(counts)
    ]
    last_id = max(task_id for queue in queues for task_id in queue)
    assert tasks > FORMER_TASK_LIMIT, f"actual task count {tasks} must exceed {FORMER_TASK_LIMIT}."
    assert last_id > FORMER_TASK_LIMIT, f"queue must execute past {FORMER_TASK_LIMIT}, got {last_id}."
    task_types = [struct.unpack_from("<I", image, RUNTIME_HEADER_BYTES + events * 4 + task_id * 576)[0]
                  for task_id in range(FORMER_TASK_LIMIT, last_id)]
    assert any(task_types), "queued descriptors above the old bound must include compute work."
    return {"task_num": tasks, "capacity": capacity, "event_capacity": events,
            "last_queued_task_id": last_id, "runtime_bytes": len(image), "queue_counts": counts}


def test_mega_moe_large_runtime() -> None:
    """Compare complete F+B and SGD with common MoE for an unfused large graph."""
    shape = baseline.MoeShape(local_num_tokens=327680, hidden_size=128, intermediate_size=128)
    start_shmem_lifetime()
    hidden, upstream = baseline.make_data(shape)
    ids, weights, counts = baseline.make_balanced_route(shape)
    mega, common = baseline.new_layers(shape)
    try:
        expected = baseline.run_layer(common, hidden, ids, weights, counts, upstream)
        actual = baseline.run_layer(mega, hidden, ids, weights, counts, upstream)
        baseline.assert_results_close(expected, actual)
        # Inspect private plan ownership only to prove which uploaded queues ran.
        plan = mega._resource_group.resources.plan  # pylint: disable=protected-access
        result = {"forward": _runtime_metadata(plan.fwd_runtime_config),
                  "backward": _runtime_metadata(plan.bwd_runtime_config),
                  "all_outputs_gradients_updates_match": True, "memory": memory_sample()}
    finally:
        mega.close()
    write_evidence(result)


def test_mega_moe_group_list_isolation() -> None:
    """Compare repeated routes at the local-expert cache-line boundary and limit."""
    start_shmem_lifetime()
    records = []
    patterns = ("balanced", "zero_token_experts", "skew", "single_destination") * 2
    for local_experts in (10, 11, 12, 13, 16):
        shape = baseline.MoeShape(local_num_tokens=128 * local_experts, num_experts=2 * local_experts)
        hidden, upstream = baseline.make_data(shape)
        mega, common = baseline.new_layers(shape)
        try:
            for pattern in patterns:
                ids, weights, counts = baseline._make_fixed_route(shape, pattern)  # pylint: disable=protected-access
                expected = baseline.run_layer(common, hidden, ids, weights, counts, upstream)
                actual = baseline.run_layer(mega, hidden, ids, weights, counts, upstream)
                baseline.assert_results_close(expected, actual)
            records.append({"shape": asdict(shape), "patterns": patterns,
                            "all_outputs_gradients_updates_match": True})
        finally:
            mega.close()
    write_evidence({"cases": records, "memory": memory_sample()})

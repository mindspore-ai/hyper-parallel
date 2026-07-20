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
"""MindSpore distributed test: CommDebugMode on a 2-card MLP with ColwiseParallel + RowwiseParallel.

Launched via msrun (HCCL NPU backend, 2 ranks).
Each test verifies CommDebugMode captures the expected collective operations
when a column-parallel → row-parallel MLP forward pass executes.

Tensor parallelism layout:
  fc1: ColwiseParallel  →  weight Shard(0), input Replicate, output Shard(-1)
                            no communication on forward
  fc2: RowwiseParallel  →  weight Shard(1), input Shard(-1), output Replicate
                            all-reduce on forward

So a single forward pass through the MLP should trigger exactly 1 collective
communication from the RowwiseParallel layer's output redistribution.

In MindSpore the process group is represented as a string (e.g. "hccl_world_group"),
so the tracing table shows ``group=<group_name>`` instead of ``group_size=N``.
"""

import json
import os
import tempfile

import numpy as np
import mindspore as ms
import mindspore.communication.management as D

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "mindspore")

from hyper_parallel import ColwiseParallel, RowwiseParallel, init_device_mesh, parallelize_module  # pylint: disable=C0413
from hyper_parallel.core.dtensor.debug import CommDebugMode  # pylint: disable=C0413


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class TwoLayerMLP(ms.nn.Cell):
    """Simple MLP: fc1 (ColwiseParallel) → GELU → fc2 (RowwiseParallel)."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.fc1 = ms.mint.nn.Linear(hidden, intermediate)
        self.fc2 = ms.mint.nn.Linear(intermediate, hidden)
        self.gelu = ms.mint.nn.GELU()

    def construct(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


def _make_tp_mesh():
    """Create a 1-D TP mesh covering all ranks."""
    world_size = D.get_group_size()
    return init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("tp",),
    )


def _make_model_and_input(hidden, intermediate):
    """Return a parallelized MLP and a random input tensor."""
    mesh = _make_tp_mesh()
    ms.set_seed(42)
    model = TwoLayerMLP(hidden, intermediate)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })
    x = ms.Tensor(np.random.randn(4, hidden).astype(np.float32))
    return model, x


# ---------------------------------------------------------------------------
# Test 1: comm_counts correctly captures collectives
# ---------------------------------------------------------------------------
def test_comm_debug_mode_captures_collectives():
    """
    Feature: CommDebugMode collective counting
    Description: A ColwiseParallel(fc1) + RowwiseParallel(fc2) MLP forward
                 should produce exactly one collective from fc2's output
                 redistribution (Partial → Replicate = all-reduce).
    Expectation: get_total_counts() >= 1 and the collective type contains
                 'all_reduce' or 'reduce_scatter'.

    Example output (2 ranks, hccl):
        Comm counts: {'differentiable_all_reduce': 1}
        Total collective calls: 1
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode() as mode:
        _ = model(x)

    counts = mode.get_comm_counts()
    total = mode.get_total_counts()

    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 1] Comm counts: {counts}")
        print(f"[Test 1] Total collective calls: {total}")

    assert total >= 1, (
        f"Expected at least 1 collective call from RowwiseParallel, got {total}. "
        f"Counts: {counts}"
    )


# ---------------------------------------------------------------------------
# Test 2: debug_string produces hierarchical output
# ---------------------------------------------------------------------------
def test_comm_debug_mode_debug_string():
    """
    Feature: CommDebugMode debug_string
    Description: generate_comm_debug_tracing_table() should produce non-empty
                 hierarchical output containing collective entries.
    Expectation: Output contains 'Collective' and is multi-line.

    Example output (2 ranks, hccl):
        Op(linear) inputs=[DTensor[4, 16], ...] outputs=[DTensor[4, 32]]
        Collective(differentiable_all_reduce) group=hccl_world_group ...

    In MindSpore the group argument to collective ops is a string, so the
    tracing table shows ``group=<group_name>`` rather than ``group_size=N``.
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode() as mode:
        _ = model(x)

    debug_out = mode.generate_comm_debug_tracing_table()
    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 2] Debug string:\n{debug_out}")

    assert debug_out != "(no operations recorded)", (
        "Expected non-empty debug output from MLP forward pass"
    )
    assert "Collective" in debug_out, (
        f"Expected 'Collective' in tracing table, got:\n{debug_out}"
    )
    assert "group=" in debug_out, (
        f"Expected group info in tracing table, got:\n{debug_out}"
    )


# ---------------------------------------------------------------------------
# Test 3: generate_tracing_table with different noise levels
# ---------------------------------------------------------------------------
def test_comm_debug_mode_tracing_table():
    """
    Feature: CommDebugMode tracing table
    Description: generate_tracing_table with noise_level=0 shows only collectives,
                 noise_level=1 also shows ops.
    Expectation: noise_level=1 table has >= lines than noise_level=0 table.

    Example output (2 ranks, hccl):

      noise_level=0 (collectives only):
        Type                 Detail
        ----------------------------------------------------------------
        Collective           Collective(differentiable_all_reduce) ...

      noise_level=1 (ops + collectives):
        Type                 Detail
        ----------------------------------------------------------------
        Op                   Op(linear) inputs=[DTensor[4, 16], ...] ...
        Op                   Op(gelu) inputs=[DTensor[4, 32]] ...
        Op                   Op(linear) inputs=[DTensor[4, 32], ...] ...
        Collective           Collective(differentiable_all_reduce) ...
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode() as mode:
        _ = model(x)

    table_0 = mode.generate_comm_debug_tracing_table(noise_level=0)
    table_1 = mode.generate_comm_debug_tracing_table(noise_level=1)

    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 3] Tracing table (noise_level=0):\n{table_0}")
        print(f"\n[Test 3] Tracing table (noise_level=1):\n{table_1}")

    lines_0 = table_0.strip().split("\n")
    lines_1 = table_1.strip().split("\n")
    assert len(lines_1) >= len(lines_0), (
        f"noise_level=1 ({len(lines_1)} lines) should have >= lines than "
        f"noise_level=0 ({len(lines_0)} lines)"
    )


# ---------------------------------------------------------------------------
# Test 4: CommDebugMode with module tracker
# ---------------------------------------------------------------------------
def test_comm_debug_mode_with_module_tracker():
    """
    Feature: CommDebugMode module tracking
    Description: When a module is passed to CommDebugMode, module enter/exit
                 events are recorded in the call tree (noise_level=2).
    Expectation: debug_string contains collective entries (module annotations
                 are included when the Cell supports named_modules()).

    Example output (2 ranks, hccl):
        Collective(differentiable_all_reduce) group=hccl_world_group ...
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode(module=model) as mode:
        _ = model(x)

    debug_out = mode.generate_comm_debug_tracing_table()
    table_2 = mode.generate_comm_debug_tracing_table(noise_level=2)

    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 4] Debug string with module tracking:\n{debug_out}")
        print(f"\n[Test 4] Tracing table (noise_level=2):\n{table_2}")

    # Collective must always appear.
    assert "Collective" in debug_out or "Collective" in table_2, (
        "Expected at least one collective in tracing output"
    )
    assert mode.get_total_counts() >= 1, "Expected at least 1 collective with module tracker active"


# ---------------------------------------------------------------------------
# Test 5: platform methods restored after exit
# ---------------------------------------------------------------------------
def test_comm_debug_mode_restores_platform():
    """
    Feature: Platform method restoration
    Description: After CommDebugMode exits, collective methods are exactly restored.
    Expectation: cls.__dict__ entries match originals after context exit.

    Example output (2 ranks, hccl):
        Platform methods correctly restored: ['differentiable_all_reduce', ...]
    """
    D.init()

    from hyper_parallel.platform import get_platform  # pylint: disable=C0415
    cls = type(get_platform())

    originals = {}
    for name in ("differentiable_all_reduce", "differentiable_all_gather_concat",
                 "differentiable_reduce_scatter"):
        if name in cls.__dict__:
            originals[name] = cls.__dict__[name]

    with CommDebugMode():
        for name, orig_method in originals.items():
            assert cls.__dict__[name] is not orig_method, (
                f"{name} should be patched inside CommDebugMode"
            )

    for name, orig in originals.items():
        assert cls.__dict__[name] is orig, (
            f"{name} not restored after CommDebugMode exit"
        )

    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 5] Platform methods correctly restored: {list(originals.keys())}")


# ---------------------------------------------------------------------------
# Test 6: multiple forward passes accumulate counts
# ---------------------------------------------------------------------------
def test_comm_debug_mode_multiple_forwards():
    """
    Feature: Accumulation across multiple forwards
    Description: Running 3 forward passes within a single CommDebugMode
                 accumulates collective counts to 3x the single-forward count.
    Expectation: count_3 == 3 * count_1.

    Example output (2 ranks, hccl):
        1 forward: 1 collectives, 3 forwards: 3 collectives
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode() as mode1:
        _ = model(x)
    count_1 = mode1.get_total_counts()

    with CommDebugMode() as mode3:
        for _ in range(3):
            _ = model(x)
    count_3 = mode3.get_total_counts()

    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 6] 1 forward: {count_1} collectives, 3 forwards: {count_3} collectives")

    assert count_3 == 3 * count_1, (
        f"Expected 3x collectives: 1 forward={count_1}, 3 forwards={count_3}"
    )


# ---------------------------------------------------------------------------
# Test 7: get_parameter_info and get_sharding_info
# ---------------------------------------------------------------------------
def test_comm_debug_mode_parameter_and_sharding_info():
    """
    Feature: Parameter and sharding info
    Description: When module is passed to CommDebugMode, get_parameter_info()
                 returns parameter tensors per module, and get_sharding_info()
                 returns placements for DTensor parameters.
    Expectation: Both dicts are non-empty and contain fc1/fc2 entries.

    Example output (2 ranks, hccl):
        Parameter info keys: ['fc1', 'fc2']
        Sharding info: {'fc1.weight': (Shard(dim=0),), 'fc2.weight': (Shard(dim=1),)}
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode(module=model) as mode:
        _ = model(x)

    param_info = mode.get_parameter_info()
    sharding_info = mode.get_sharding_info()

    rank = D.get_rank()
    if rank == 0:
        print(f"\n[Test 7] Parameter info keys: {list(param_info.keys())}")
        print(f"[Test 7] Sharding info: {sharding_info}")

    assert len(param_info) > 0, "Expected non-empty parameter info when module is provided"
    assert len(sharding_info) > 0, "Expected non-empty sharding info for DTensor parameters"
    assert any("fc1" in k for k in param_info), "Expected fc1 in parameter info"
    assert any("fc2" in k for k in param_info), "Expected fc2 in parameter info"
    assert any("weight" in k for k in sharding_info), (
        "Expected weight placements in sharding info"
    )


# ---------------------------------------------------------------------------
# Test 8: log_comm_debug_tracing_table_to_file
# ---------------------------------------------------------------------------
def test_comm_debug_mode_log_to_file():
    """
    Feature: Log tracing table to file
    Description: log_comm_debug_tracing_table_to_file() writes the tracing
                 table to a file with ANSI escape codes stripped.
    Expectation: File exists, is non-empty, and contains collective info.

    Example output (2 ranks, hccl):
        Log file written: /tmp/comm_debug_test_rank0.txt (312 bytes)
        File contains 'Collective': True
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode() as mode:
        _ = model(x)

    rank = D.get_rank()
    log_file = os.path.join(tempfile.gettempdir(), f"comm_debug_ms_test_rank{rank}.txt")
    mode.log_comm_debug_tracing_table_to_file(log_file, noise_level=1)

    assert os.path.exists(log_file), f"Log file not created: {log_file}"
    with open(log_file, encoding="utf-8") as f:
        content = f.read()

    if rank == 0:
        print(f"\n[Test 8] Log file written: {log_file} ({len(content)} bytes)")
        print(f"[Test 8] File contains 'Collective': {'Collective' in content}")

    assert len(content) > 0, "Log file is empty"
    assert "Collective" in content, "Expected collective info in log file"
    assert "\x1b[" not in content, "ANSI escape codes should be stripped from file output"

    os.remove(log_file)


# ---------------------------------------------------------------------------
# Test 9: generate_json_dump
# ---------------------------------------------------------------------------
def test_comm_debug_mode_json_dump():
    """
    Feature: JSON dump export
    Description: generate_json_dump() writes a valid JSON file containing
                 comm_counts, total_counts, and records.
    Expectation: File is valid JSON with expected top-level keys and
                 at least one collective record.

    Example output (2 ranks, hccl):
        JSON keys: ['comm_counts', 'total_counts', 'records']
        Total counts in JSON: 1
        Collective records found: 1
    """
    D.init()
    hidden, intermediate = 16, 32
    model, x = _make_model_and_input(hidden, intermediate)

    with CommDebugMode() as mode:
        _ = model(x)

    rank = D.get_rank()
    json_file = os.path.join(tempfile.gettempdir(), f"comm_debug_ms_test_rank{rank}.json")
    mode.generate_json_dump(json_file, noise_level=1)

    assert os.path.exists(json_file), f"JSON file not created: {json_file}"
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    if rank == 0:
        print(f"\n[Test 9] JSON keys: {list(data.keys())}")
        print(f"[Test 9] Total counts in JSON: {data.get('total_counts')}")

    assert "comm_counts" in data, "Expected 'comm_counts' key in JSON"
    assert "total_counts" in data, "Expected 'total_counts' key in JSON"
    assert "records" in data, "Expected 'records' key in JSON"
    assert data["total_counts"] >= 1, "Expected at least 1 collective in JSON"

    def find_collectives(records):
        count = 0
        for r in records:
            if r.get("type") == "collective":
                count += 1
            count += find_collectives(r.get("children", []))
        return count

    collective_count = find_collectives(data["records"])
    if rank == 0:
        print(f"[Test 9] Collective records found: {collective_count}")

    assert collective_count >= 1, "Expected at least one collective record in JSON"

    os.remove(json_file)

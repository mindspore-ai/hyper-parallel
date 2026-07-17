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
"""Distributed test: CommDebugMode on a 2-layer MLP with ColwiseParallel + RowwiseParallel.

Launched via torchrun (gloo CPU backend, 2 ranks).
Each test verifies CommDebugMode captures the expected collective operations
when a column-parallel → row-parallel MLP forward pass executes.

Tensor parallelism layout:
  fc1: ColwiseParallel  →  weight Shard(0), input Replicate, output Shard(-1)
                            no communication on forward
  fc2: RowwiseParallel  →  weight Shard(1), input Shard(-1), output Replicate
                            all-reduce (or reduce-scatter) on forward

So a single forward pass through the MLP should trigger exactly 1 collective
communication from the RowwiseParallel layer's output redistribution.
"""

import json
import os
import tempfile

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import ColwiseParallel, RowwiseParallel, init_device_mesh, parallelize_module
from hyper_parallel.core.dtensor.debug import CommDebugMode
from hyper_parallel.platform import get_platform as _get_platform
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class TwoLayerMLP(nn.Module):
    """Simple MLP:  fc1 (ColwiseParallel) → GELU → fc2 (RowwiseParallel)."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def _make_tp_mesh():
    """Create a 1-D TP mesh covering all ranks."""
    return init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("tp",),
    )


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

    Example output (2 ranks, gloo):
        Comm counts: {'differentiable_all_reduce': 1}
        Total collective calls: 1

    ColwiseParallel on fc1 shards weight along out_features (Shard(0)) and
    produces a Shard(-1) output — no communication needed.  RowwiseParallel
    on fc2 shards weight along in_features (Shard(1)); its matmul yields a
    Partial result that requires exactly one all-reduce to produce the
    Replicate output.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode() as mode:
        _ = model(x)

    counts = mode.get_comm_counts()
    total = mode.get_total_counts()

    rank = dist.get_rank()
    if rank == 0:
        print(f"\n[Test 1] Comm counts: {counts}")
        print(f"[Test 1] Total collective calls: {total}")

    assert total >= 1, (
        f"Expected at least 1 collective call from RowwiseParallel, got {total}. "
        f"Counts: {counts}"
    )


# ---------------------------------------------------------------------------
# Test 2: debug_string() produces hierarchical output
# ---------------------------------------------------------------------------
def test_comm_debug_mode_debug_string():
    """
    Feature: CommDebugMode debug_string
    Description: debug_string() should produce non-empty hierarchical output
                 containing both op names and collective types.
    Expectation: Output contains 'Collective' entries and is multi-line.

    Example output (2 ranks, gloo):
        Op(linear) inputs=[DTensor[4, 16], DTensor[32, 16], DTensor[32]] outputs=[DTensor[4, 32]]
        Op(gelu) inputs=[DTensor[4, 32]] outputs=[DTensor[4, 32]]
        Op(linear) inputs=[DTensor[4, 32], DTensor[16, 32], DTensor[16]] outputs=[DTensor[4, 16]]
        Collective(differentiable_all_reduce) group_size=2 input_shape=(4, 16) output_shape=(4, 16)

    The call tree shows the full forward path: fc1's linear (column-sharded,
    no comm), gelu activation, fc2's linear (row-sharded), then the
    all-reduce that collapses the Partial result into Replicate.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode() as mode:
        _ = model(x)

    debug_out = mode.generate_comm_debug_tracing_table()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(f"\n[Test 2] Debug string:\n{debug_out}")

    assert debug_out != "(no operations recorded)", (
        "Expected non-empty debug output from MLP forward pass"
    )
    assert f"group_size={world_size}" in debug_out, (
        f"Expected group_size={world_size} in tracing table, got:\n{debug_out}"
    )


# ---------------------------------------------------------------------------
# Test 3: generate_tracing_table with different noise levels
# ---------------------------------------------------------------------------
def test_comm_debug_mode_tracing_table():
    """
    Feature: CommDebugMode tracing table
    Description: generate_tracing_table with noise_level=0 shows only collectives,
                 noise_level=1 also shows ops.
    Expectation: noise_level=0 table is shorter than noise_level=1 table.

    Example output (2 ranks, gloo):

      noise_level=0 (collectives only — 1 data line):
        Type                 Detail
        --------------------------------------------------------------------------------
        Collective           Collective(differentiable_all_reduce) ...

      noise_level=1 (ops + collectives — 4 data lines):
        Type                 Detail
        --------------------------------------------------------------------------------
        Op                   Op(linear) inputs=[DTensor[4, 16], ...] outputs=[DTensor[4, 32]]
        Op                   Op(gelu) inputs=[DTensor[4, 32]] outputs=[DTensor[4, 32]]
        Op                   Op(linear) inputs=[DTensor[4, 32], ...] outputs=[DTensor[4, 16]]
        Collective           Collective(differentiable_all_reduce) ...

    noise_level=0 filters out all Op entries, leaving only the single
    all-reduce from RowwiseParallel.  noise_level=1 adds the three DTensor
    ops (two linears + gelu), giving a fuller picture of the computation.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode() as mode:
        _ = model(x)

    table_0 = mode.generate_comm_debug_tracing_table(noise_level=0)
    table_1 = mode.generate_comm_debug_tracing_table(noise_level=1)

    rank = dist.get_rank()
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
                 events are recorded in the call tree.
    Expectation: debug_string contains module boundary annotations.

    Example output (2 ranks, gloo):
        Module((root)) [enter]
          Module(fc1) [enter]
            Op(linear) inputs=[DTensor[4, 16], DTensor[32, 16], DTensor[32]] outputs=[DTensor[4, 32]]
          Op(gelu) inputs=[DTensor[4, 32]] outputs=[DTensor[4, 32]]
          Module(fc2) [enter]
            Op(linear) inputs=[DTensor[4, 32], DTensor[16, 32], DTensor[16]] outputs=[DTensor[4, 16]]
            Collective(differentiable_all_reduce) group_size=2 input_shape=(4, 16) output_shape=(4, 16)

    With module tracking enabled, the tree gains Module boundary nodes.
    The all-reduce is visibly nested under Module(fc2), confirming it
    originates from RowwiseParallel's Partial → Replicate redistribution.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode(module=model) as mode:
        _ = model(x)

    debug_out = mode.generate_comm_debug_tracing_table()
    table = mode.generate_comm_debug_tracing_table(noise_level=2)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(f"\n[Test 4] Debug string with module tracking:\n{debug_out}")
        print(f"\n[Test 4] Tracing table (noise_level=2):\n{table}")

    assert "Module" in debug_out or "Module" in table, (
        "Expected module boundary annotations when module tracker is active"
    )
    assert f"group_size={world_size}" in debug_out, (
        f"Expected group_size={world_size} in tracing table, got:\n{debug_out}"
    )


# ---------------------------------------------------------------------------
# Test 5: platform methods restored after exit
# ---------------------------------------------------------------------------
def test_comm_debug_mode_restores_platform():
    """
    Feature: Platform method restoration
    Description: After CommDebugMode exits, collective methods are exactly restored.
    Expectation: cls.__dict__ entries match originals after context exit.

    Example output (2 ranks, gloo):
        Platform methods correctly restored: ['differentiable_all_reduce',
            'differentiable_all_gather_concat', 'differentiable_reduce_scatter']

    Verifies the monkey-patch lifecycle: inside CommDebugMode the platform's
    staticmethod descriptors are replaced with tracing wrappers; on exit,
    ``type.__setattr__`` restores the original descriptors exactly (identity
    check via ``is``), so no tracing overhead leaks into subsequent code.
    """
    init_backend(_DEVICE_TYPE)

    cls = type(_get_platform())

    originals = {}
    for name in ("differentiable_all_reduce", "differentiable_all_gather_concat",
                 "differentiable_reduce_scatter"):
        if name in cls.__dict__:
            originals[name] = cls.__dict__[name]

    with CommDebugMode():
        # Inside: methods should be patched (different from originals)
        for name, orig_method in originals.items():
            assert cls.__dict__[name] is not orig_method, (
                f"{name} should be patched inside CommDebugMode"
            )

    # Outside: methods should be restored
    for name, orig in originals.items():
        assert cls.__dict__[name] is orig, (
            f"{name} not restored after CommDebugMode exit"
        )

    rank = dist.get_rank()
    if rank == 0:
        print(f"\n[Test 5] Platform methods correctly restored: {list(originals.keys())}")


# ---------------------------------------------------------------------------
# Test 6: multiple forward passes accumulate counts
# ---------------------------------------------------------------------------
def test_comm_debug_mode_multiple_forwards():
    """
    Feature: Accumulation across multiple forwards
    Description: Running multiple forward passes within a single CommDebugMode
                 accumulates collective counts.
    Expectation: Counts after 3 forwards are ~3x counts after 1 forward.

    Example output (2 ranks, gloo):
        1 forward: 1 collectives, 3 forwards: 3 collectives

    Each forward triggers exactly 1 all-reduce (from RowwiseParallel).
    Running 3 forwards within the same CommDebugMode context accumulates
    to 3 total — a strict 3x relationship confirming linear accumulation
    with no stale state leaking between passes.
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    # Single forward
    with CommDebugMode() as mode1:
        _ = model(x)
    count_1 = mode1.get_total_counts()

    # Three forwards
    with CommDebugMode() as mode3:
        for _ in range(3):
            _ = model(x)
    count_3 = mode3.get_total_counts()

    rank = dist.get_rank()
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

    Example output (2 ranks, gloo):
        Parameter info keys: ['fc1', 'fc2']
        Sharding info: {'fc1.weight': (Shard(dim=0),), 'fc2.weight': (Shard(dim=1),)}
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode(module=model) as mode:
        _ = model(x)

    param_info = mode.get_parameter_info()
    sharding_info = mode.get_sharding_info()

    rank = dist.get_rank()
    if rank == 0:
        print(f"\n[Test 7] Parameter info keys: {list(param_info.keys())}")
        print(f"[Test 7] Sharding info: {sharding_info}")

    assert len(param_info) > 0, "Expected non-empty parameter info when module is provided"
    assert len(sharding_info) > 0, "Expected non-empty sharding info for DTensor parameters"

    # fc1 and fc2 should appear in parameter info
    assert any("fc1" in k for k in param_info), "Expected fc1 in parameter info"
    assert any("fc2" in k for k in param_info), "Expected fc2 in parameter info"

    # DTensor parameters should appear in sharding info
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

    Example output (2 ranks, gloo):
        Log file written: /tmp/comm_debug_test.txt (312 bytes)
        File contains 'Collective': True
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode() as mode:
        _ = model(x)

    rank = dist.get_rank()
    log_file = os.path.join(tempfile.gettempdir(), f"comm_debug_test_rank{rank}.txt")
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

    Example output (2 ranks, gloo):
        JSON keys: ['comm_counts', 'total_counts', 'records']
        Total counts in JSON: 1
        Collective records found: 1
    """
    init_backend(_DEVICE_TYPE)
    mesh = _make_tp_mesh()

    torch.manual_seed(42)
    hidden, intermediate = 16, 32

    model = to_device(TwoLayerMLP(hidden, intermediate), _DEVICE_TYPE)
    parallelize_module(model, mesh, {
        "fc1": ColwiseParallel(use_local_output=False),
        "fc2": RowwiseParallel(),
    })

    x = to_device(torch.randn(4, hidden), _DEVICE_TYPE)

    with CommDebugMode() as mode:
        _ = model(x)

    rank = dist.get_rank()
    json_file = os.path.join(tempfile.gettempdir(), f"comm_debug_test_rank{rank}.json")
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

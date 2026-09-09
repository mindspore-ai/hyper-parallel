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
"""CPU unit tests for direct FSDP-to-vLLM TP resharding."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring

from types import SimpleNamespace
from typing import Any

import torch

import rl.roles.weight_sync.vllm_worker as worker_module
from rl.roles.weight_sync.layout import (
    SourceTensorLayout,
    TensorRegion,
    build_direct_reshard_plan,
    describe_source_tensor,
    resolve_destination_layouts,
    resolve_source_layouts,
)
def _apply_plan(
    plan,
    source_values: dict[tuple[str, int], torch.Tensor],
    destination_values: dict[tuple[str, int], torch.Tensor],
) -> None:
    """Apply one metadata plan to CPU tensors as the transfer oracle harness."""
    for (source_rank, tp_rank), buckets in plan.buckets.items():
        for bucket in buckets:
            for entry in bucket.entries:
                source_slice = tuple(
                    slice(start, start + length)
                    for start, length in zip(entry.source_starts, entry.lengths)
                )
                destination_slice = tuple(
                    slice(start, start + length)
                    for start, length in zip(entry.destination_starts, entry.lengths)
                )
                destination_values[(entry.target_name, tp_rank)][destination_slice].copy_(
                    source_values[(entry.name, source_rank)][source_slice]
                )


def test_direct_reshard_matches_full_gather_reference() -> None:
    """Native fused QKV and gate/up targets equal full-gather then TP slicing."""

    class FakeModel:
        """Expose native vLLM fused parameters for two TP ranks."""

        def __init__(self) -> None:
            self.parameters = {
                "model.layers.0.self_attn.qkv_proj.weight": torch.empty(8, 8, dtype=torch.int64),
                "model.layers.0.mlp.gate_up_proj.weight": torch.empty(12, 8, dtype=torch.int64),
            }

        def named_parameters(self) -> Any:
            """Return physical rollout parameters."""
            return self.parameters.items()

    hf_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        intermediate_size=12,
        vocab_size=10,
    )
    worker_descriptions = [
        {
            "tp_rank": tp_rank,
            "tp_size": 2,
            "tensors": worker_module._native_qwen3_direct_tensors(  # pylint: disable=protected-access
                FakeModel(), hf_config, tp_rank, 2
            ),
        }
        for tp_rank in range(2)
    ]
    full_parameters = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(64).view(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(64, 96).view(4, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.arange(96, 128).view(4, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.arange(128, 224).view(12, 8),
        "model.layers.0.mlp.up_proj.weight": torch.arange(224, 320).view(12, 8),
    }
    destinations = resolve_destination_layouts(
        worker_descriptions,
        {name: tuple(value.shape) for name, value in full_parameters.items()},
    )
    sources = []
    source_values = {}
    for name, full in full_parameters.items():
        for source_rank in range(2):
            local_rows = full.shape[0] // 2
            row_start = source_rank * local_rows
            sources.append(
                SourceTensorLayout(
                    name,
                    "int64",
                    8,
                    tuple(full.shape),
                    source_rank,
                    TensorRegion((row_start, 0), (local_rows, full.shape[1])),
                )
            )
            source_values[(name, source_rank)] = full[row_start : row_start + local_rows]
    plan = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=2,
        bucket_size_bytes=64,
    )
    destination_values = {
        (name, tp_rank): torch.full(shape, -1, dtype=torch.int64)
        for name, shape in (
            ("model.layers.0.self_attn.qkv_proj.weight", (8, 8)),
            ("model.layers.0.mlp.gate_up_proj.weight", (12, 8)),
        )
        for tp_rank in range(2)
    }

    _apply_plan(plan, source_values, destination_values)

    for tp_rank in range(2):
        expected_qkv = torch.cat(
            (
                full_parameters["model.layers.0.self_attn.q_proj.weight"][tp_rank * 4 : (tp_rank + 1) * 4],
                full_parameters["model.layers.0.self_attn.k_proj.weight"][tp_rank * 2 : (tp_rank + 1) * 2],
                full_parameters["model.layers.0.self_attn.v_proj.weight"][tp_rank * 2 : (tp_rank + 1) * 2],
            )
        )
        expected_gate_up = torch.cat(
            (
                full_parameters["model.layers.0.mlp.gate_proj.weight"][tp_rank * 6 : (tp_rank + 1) * 6],
                full_parameters["model.layers.0.mlp.up_proj.weight"][tp_rank * 6 : (tp_rank + 1) * 6],
            )
        )
        torch.testing.assert_close(
            destination_values[("model.layers.0.self_attn.qkv_proj.weight", tp_rank)],
            expected_qkv,
        )
        torch.testing.assert_close(
            destination_values[("model.layers.0.mlp.gate_up_proj.weight", tp_rank)],
            expected_gate_up,
        )


def test_source_and_destination_metadata_build_complete_reshard_plan() -> None:
    """Production metadata resolution builds complete cross-axis FSDP-to-TP routes."""

    class ShardPlacement:
        """Describe one shard placement along tensor dimension zero."""

        dim = 0

        @staticmethod
        def is_shard() -> bool:
            return True

    class Mesh:
        """Expose one FSDP mesh coordinate."""

        ndim = 1

        def __init__(self, coordinate: int) -> None:
            self.coordinate = coordinate

        @staticmethod
        def size(_mesh_dim: int) -> int:
            return 2

        def get_coordinate(self) -> tuple[int]:
            return (self.coordinate,)

    class DistributedValue:
        """Separate global shape metadata from one local tensor shard."""

        shape = (4, 4)
        placements = (ShardPlacement(),)

        def __init__(self, rank: int) -> None:
            self.device_mesh = Mesh(rank)
            self.local = torch.arange(rank * 8, (rank + 1) * 8, dtype=torch.float32).reshape(2, 4)

        def to_local(self) -> torch.Tensor:
            return self.local

    rank_descriptions = [
        [describe_source_tensor("weight", DistributedValue(rank), rank)]
        for rank in range(2)
    ]
    worker_descriptions = [
        {
            "tp_rank": tp_rank,
            "tp_size": 2,
            "tensors": [
                {
                    "name": "weight",
                    "dtype_name": "float32",
                    "element_size": 4,
                    "local_shape": [4, 2],
                    "placement": "shard",
                    "shard_dim": 1,
                }
            ],
        }
        for tp_rank in range(2)
    ]

    sources = resolve_source_layouts(rank_descriptions)
    destinations = resolve_destination_layouts(worker_descriptions, {"weight": (4, 4)})
    plan = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=2,
        bucket_size_bytes=32,
    )

    assert [(layout.source_rank, layout.region.starts, layout.region.lengths) for layout in sources] == [
        (0, (0, 0), (2, 4)),
        (1, (2, 0), (2, 4)),
    ]
    assert [(layout.tp_rank, layout.region.starts, layout.region.lengths) for layout in destinations] == [
        (0, (0, 0), (4, 2)),
        (1, (0, 2), (4, 2)),
    ]
    assert plan.route_count == 4
    assert plan.fragment_count == 4
    assert sum(
        bucket.total_bytes
        for buckets in plan.buckets.values()
        for bucket in buckets
    ) == 64


def test_legacy_single_axis_source_metadata_resolves_contiguous_rank_shards() -> None:
    """Legacy DTensor metadata without mesh coordinates remains a complete FSDP layout."""

    class ShardPlacement:
        dim = 0

        @staticmethod
        def is_shard() -> bool:
            return True

    class DistributedValue:
        shape = (4, 3)
        placements = (ShardPlacement(),)

        def __init__(self, rank: int) -> None:
            self.local = torch.full((2, 3), float(rank))

        def to_local(self) -> torch.Tensor:
            return self.local

    descriptions = [
        [describe_source_tensor("weight", DistributedValue(rank), rank)]
        for rank in range(2)
    ]

    layouts = resolve_source_layouts(descriptions)

    assert [layout.name for layout in layouts] == ["weight", "weight"]
    assert [layout.source_rank for layout in layouts] == [0, 1]
    assert [layout.region.starts for layout in layouts] == [(0, 0), (2, 0)]
    assert [layout.region.lengths for layout in layouts] == [(2, 3), (2, 3)]
    assert all(layout.global_shape == (4, 3) for layout in layouts)
    assert all(layout.dtype_name == "float32" for layout in layouts)

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
"""Metadata-only contracts for FSDP-to-TP Broadcast direct resharding."""

import base64
import gc
import json
import pickle
import sys
import weakref
from copy import deepcopy
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import pytest
import rl.roles.weight_sync.transfer as transfer_module
import rl.roles.weight_sync.vllm_worker as worker_module
import torch
from rl.config import _validate_moe_ep1_topology, _validate_vllm_weight_sync
from rl.roles.weight_sync.config import resolve_weight_sync_config
from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    DirectReshardPlan,
    SourceTensorLayout,
    TensorRegion,
    TransferBucket,
    build_direct_reshard_plan,
    describe_source_tensor,
    resolve_destination_layouts,
    resolve_physical_worker_topology,
    resolve_source_layouts,
)
from rl.roles.weight_sync.model_adapter import (
    aggregate_direct_content_identity,
    build_model_weight_adapter,
    direct_fragment_record,
)
from rl.roles.weight_sync.streaming_full_gather import (
    assemble_streaming_fragment,
    build_streaming_full_gather_plan,
    extract_streaming_contributions,
)
from rl.roles.weight_sync.sync import PolicySnapshot, VLLMWeightSyncClientMixin
from rl.roles.weight_sync.transfer import (
    ColocatedDirectReshardWeightTransfer,
    ColocatedFullGatherWeightTransfer,
    DirectReshardHCCLWeightTransfer,
    FallbackWeightTransfer,
    FullGatherHCCLWeightTransfer,
    build_weight_transfer,
)
from rl.roles.weight_sync.vllm_worker import (
    _hyper_deepseek_v3_direct_tensors,
    _hyper_moe_direct_tensors,
    _hyper_qwen3_moe_direct_tensors,
    _native_moe_direct_tensors,
    _native_moe_ownership_manifest,
)

from hyper_parallel import Replicate, Shard
from hyper_parallel.auto_models.components.distributed import ep_utils


def _source(rank: int, starts: tuple[int, int], lengths: tuple[int, int]) -> SourceTensorLayout:
    return SourceTensorLayout(
        name="model.layers.0.self_attn.q_proj.weight",
        dtype_name="bfloat16",
        element_size=2,
        global_shape=(4, 4),
        source_rank=rank,
        region=TensorRegion(starts, lengths),
    )


def _deepseek_v3_registration() -> SimpleNamespace:
    """Return the minimal native DeepSeek-V3 registration used by adapter tests."""
    return SimpleNamespace(
        family="deepseek_v3",
        is_hyper=False,
        model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )


def _qwen3_moe_registration(*, is_hyper: bool = True) -> SimpleNamespace:
    """Return the minimal Qwen3-MoE registration used by adapter tests."""
    return SimpleNamespace(
        family="qwen3_moe",
        is_hyper=is_hyper,
        model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )


def _destination(
    rank: int,
    starts: tuple[int, int],
    lengths: tuple[int, int],
    *,
    placement: str = "shard",
    shard_dim: int | None = 1,
) -> DestinationTensorLayout:
    return DestinationTensorLayout(
        name="model.layers.0.self_attn.q_proj.weight",
        dtype_name="bfloat16",
        element_size=2,
        global_shape=(4, 4),
        tp_rank=rank,
        tp_size=2,
        placement=placement,
        shard_dim=shard_dim,
        region=TensorRegion(starts, lengths),
    )


def test_cross_axis_reshard_builds_four_source_to_tp_routes() -> None:
    """FSDP row shards intersect both rollout TP column shards."""
    plan = build_direct_reshard_plan(
        (
            _source(0, (0, 0), (2, 4)),
            _source(1, (2, 0), (2, 4)),
        ),
        (
            _destination(0, (0, 0), (4, 2)),
            _destination(1, (0, 2), (4, 2)),
        ),
        source_world_size=2,
        bucket_size_bytes=128,
    )

    assert plan.route_count == 4
    assert plan.fragment_count == 4
    assert plan.for_route(0, 0)[0].entries[0].lengths == (2, 2)
    assert plan.for_route(1, 1)[0].entries[0].destination_starts == (2, 0)


def test_matching_shard_axis_uses_only_diagonal_routes() -> None:
    """Aligned FSDP and TP rows avoid sending either shard to the other TP rank."""
    plan = build_direct_reshard_plan(
        (
            _source(0, (0, 0), (2, 4)),
            _source(1, (2, 0), (2, 4)),
        ),
        (
            _destination(0, (0, 0), (2, 4), shard_dim=0),
            _destination(1, (2, 0), (2, 4), shard_dim=0),
        ),
        source_world_size=2,
        bucket_size_bytes=128,
    )

    assert set(plan.buckets) == {(0, 0), (1, 1)}


def test_replicated_tp_parameters_receive_every_fsdp_fragment() -> None:
    """Each TP rank receives both FSDP pieces for a replicated destination."""
    plan = build_direct_reshard_plan(
        (
            _source(0, (0, 0), (2, 4)),
            _source(1, (2, 0), (2, 4)),
        ),
        (
            _destination(
                0,
                (0, 0),
                (4, 4),
                placement="replicate",
                shard_dim=None,
            ),
            _destination(
                1,
                (0, 0),
                (4, 4),
                placement="replicate",
                shard_dim=None,
            ),
        ),
        source_world_size=2,
        bucket_size_bytes=128,
    )

    assert set(plan.buckets) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(
        sum(entry.numel for bucket in plan.for_route(source_rank, tp_rank) for entry in bucket.entries) == 8
        for source_rank in range(2)
        for tp_rank in range(2)
    )


def test_large_intersection_is_tiled_to_bucket_limit() -> None:
    """One large parameter cannot silently allocate an oversized packed buffer."""
    plan = build_direct_reshard_plan(
        (
            _source(0, (0, 0), (2, 4)),
            _source(1, (2, 0), (2, 4)),
        ),
        (
            _destination(0, (0, 0), (4, 2)),
            _destination(1, (0, 2), (4, 2)),
        ),
        source_world_size=2,
        bucket_size_bytes=4,
    )

    assert all(
        bucket.total_bytes <= 4
        for route_buckets in plan.buckets.values()
        for bucket in route_buckets
    )
    assert plan.fragment_count == 8


def test_direct_plan_declares_bounded_destination_dtype_conversion() -> None:
    """A model-owned BF16-to-FP32 destination cast remains a bounded direct copy."""
    source = SourceTensorLayout(
        name="model.layers.1.mlp.gate.weight",
        dtype_name="bfloat16",
        element_size=2,
        global_shape=(2, 3),
        source_rank=0,
        region=TensorRegion((0, 0), (2, 3)),
    )
    destination = DestinationTensorLayout(
        name=source.name,
        dtype_name="float32",
        element_size=4,
        global_shape=source.global_shape,
        tp_rank=0,
        tp_size=1,
        placement="replicate",
        shard_dim=None,
        region=source.region,
        accepted_source_dtypes=("bfloat16",),
    )

    plan = build_direct_reshard_plan(
        (source,),
        (destination,),
        source_world_size=1,
        bucket_size_bytes=16,
    )
    entry = plan.for_route(0, 0)[0].entries[0]
    metadata = entry.worker_metadata()

    assert entry.dtype_name == "bfloat16"
    assert entry.element_size == 2
    assert metadata["destination_dtype_name"] == "float32"
    assert metadata["destination_element_size"] == 4
    assert metadata["num_bytes"] == 12


def test_deepseek_tp1_direct_plan_matches_ascend_fused_moe_storage() -> None:
    """Packed Trainer experts rebuild transposed Native-vLLM FusedMoE exactly."""

    class FakeNativeModel:
        """Expose the post-load Ascend shapes used by native DeepSeek-V3."""

        def __init__(self) -> None:
            """Initialize FakeNativeModel state."""
            self.parameters = {
                "model.layers.1.mlp.experts.w13_weight": torch.empty(2, 3, 4),
                "model.layers.1.mlp.experts.w2_weight": torch.empty(2, 2, 3),
                "model.layers.1.mlp.shared_experts.gate_up_proj.weight": torch.empty(8, 3),
                "model.layers.1.mlp.gate.weight": torch.empty(2, 3),
                "model.layers.1.mlp.gate.e_score_correction_bias": torch.empty(2),
            }

        def named_parameters(self) -> Any:
            """Return physical native-vLLM parameters."""
            return self.parameters.items()

    hf_config = SimpleNamespace(
        hidden_size=3,
        intermediate_size=5,
        moe_intermediate_size=2,
        n_routed_experts=2,
        n_shared_experts=2,
    )
    gate_up = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    down = torch.arange(24, 36, dtype=torch.float32).view(2, 3, 2)
    shared_gate = torch.arange(36, 48, dtype=torch.float32).view(4, 3)
    shared_up = torch.arange(48, 60, dtype=torch.float32).view(4, 3)
    router = torch.arange(60, 66, dtype=torch.float32).view(2, 3)
    correction = torch.tensor([0.25, -0.25], dtype=torch.float32)
    state_dict = {
        "model.layers.1.mlp.experts.gate_up_proj": gate_up,
        "model.layers.1.mlp.experts.down_proj": down,
        "model.layers.1.mlp.shared_experts.gate_proj.weight": shared_gate,
        "model.layers.1.mlp.shared_experts.up_proj.weight": shared_up,
        "model.layers.1.mlp.gate.weight": router,
        "model.layers.1.mlp.gate.e_score_correction_bias": correction,
    }
    adapter = build_model_weight_adapter(_deepseek_v3_registration())
    rank_descriptions = [adapter.direct_source_descriptions(state_dict, 0)]
    sources = resolve_source_layouts(rank_descriptions)
    worker_descriptions = [
        {
            "tp_rank": 0,
            "tp_size": 1,
            "tensors": worker_module._native_moe_direct_tensors(  # pylint: disable=protected-access
                FakeNativeModel(),
                hf_config,
                1,
                family="deepseek_v3",
            ),
        }
    ]
    global_shapes = {source.name: source.global_shape for source in sources}
    destinations = resolve_destination_layouts(worker_descriptions, global_shapes)
    plan = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=32,
    )
    physical = {
        name: torch.full(tuple(parameter.shape), float("nan"))
        for name, parameter in FakeNativeModel().named_parameters()
    }
    source_fragments = {}
    destination_fragments = {}
    for bucket in plan.for_route(0, 0):
        assert bucket.total_bytes <= 32
        for entry in bucket.entries:
            source_slice = tuple(
                slice(start, start + length)
                for start, length in zip(entry.source_starts, entry.lengths)
            )
            canonical = state_dict[entry.source_key][source_slice]
            physical_fragment = canonical.permute(entry.physical_permutation).contiguous()
            destination_slice = tuple(
                slice(start, start + length)
                for start, length in zip(
                    entry.destination_starts,
                    entry.destination_lengths,
                )
            )
            physical[entry.target_name][destination_slice].copy_(physical_fragment)
            source_key, source_record = direct_fragment_record(
                entry.name,
                entry.logical_starts,
                entry.lengths,
                entry.dtype_name,
                canonical.contiguous().view(torch.uint8).numpy().tobytes(),
            )
            restored = physical[entry.target_name][destination_slice].permute(
                tuple(
                    entry.physical_permutation.index(axis)
                    for axis in range(len(entry.lengths))
                )
            )
            destination_key, destination_record = direct_fragment_record(
                entry.name,
                entry.logical_starts,
                entry.lengths,
                entry.dtype_name,
                restored.contiguous().view(torch.uint8).numpy().tobytes(),
            )
            source_fragments[source_key] = source_record
            destination_fragments[destination_key] = destination_record

    assert torch.equal(
        physical["model.layers.1.mlp.experts.w13_weight"],
        gate_up.transpose(1, 2).contiguous(),
    )
    assert torch.equal(
        physical["model.layers.1.mlp.experts.w2_weight"],
        down.transpose(1, 2).contiguous(),
    )
    assert torch.equal(
        physical["model.layers.1.mlp.shared_experts.gate_up_proj.weight"],
        torch.cat((shared_gate, shared_up), dim=0),
    )
    assert torch.equal(physical["model.layers.1.mlp.gate.weight"], router)
    assert torch.equal(
        physical["model.layers.1.mlp.gate.e_score_correction_bias"],
        correction,
    )
    source_identity = aggregate_direct_content_identity(source_fragments)
    destination_identity = aggregate_direct_content_identity(destination_fragments)
    assert source_identity == destination_identity
    assert source_identity["total_bytes"] == sum(
        tensor.numel() * tensor.element_size() for tensor in state_dict.values()
    )


def test_qwen3_moe_packed_plan_matches_common_fused_moe_storage() -> None:
    """Qwen3-MoE packed Trainer experts use the common bounded schema."""

    class LocalExperts:
        """Mark one physical FusedMoE leaf as communication-free EP1."""

        hyper_local_expert_leaf = True

    class FakeHyperModel:
        """Expose a small Qwen3-MoE HF outer model and physical expert leaf."""

        def __init__(self) -> None:
            """Initialize FakeHyperModel state."""
            self.parameters = {
                "model.layers.0.mlp.experts.w13_weight": torch.empty(2, 3, 4),
                "model.layers.0.mlp.experts.w2_weight": torch.empty(2, 2, 3),
                "model.layers.0.mlp.gate.weight": torch.empty(2, 3),
                "model.norm.weight": torch.empty(3),
            }

        def named_parameters(self) -> Any:
            """Expose fixture parameters through the model interface."""
            return self.parameters.items()

        @staticmethod
        def named_buffers() -> tuple[object, ...]:
            """Expose fixture buffers through the model interface."""
            return ()

        @staticmethod
        def named_modules() -> tuple[tuple[str, object], ...]:
            """Expose fixture modules through the model interface."""
            return (("model.layers.0.mlp.experts", LocalExperts()),)

    hf_config = SimpleNamespace(
        hidden_size=3,
        moe_intermediate_size=2,
        num_experts=2,
    )
    gate_up = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    down = torch.arange(24, 36, dtype=torch.float32).view(2, 3, 2)
    router = torch.arange(36, 42, dtype=torch.float32).view(2, 3)
    norm = torch.arange(3, dtype=torch.float32)
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": down,
        "model.layers.0.mlp.gate.weight": router,
        "model.norm.weight": norm,
    }
    adapter = build_model_weight_adapter(_qwen3_moe_registration())
    sources = resolve_source_layouts(
        [adapter.direct_source_descriptions(state_dict, 0)]
    )
    destination_description = {
        "tp_rank": 0,
        "tp_size": 1,
        "tensors": worker_module._hyper_qwen3_moe_direct_tensors(  # pylint: disable=protected-access
            FakeHyperModel(),
            hf_config,
            1,
        ),
    }
    global_shapes = {source.name: source.global_shape for source in sources}
    destinations = resolve_destination_layouts(
        [destination_description],
        global_shapes,
    )
    plan = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=32,
    )
    physical = {
        name: torch.full(tuple(parameter.shape), float("nan"))
        for name, parameter in FakeHyperModel().named_parameters()
    }
    source_fragments = {}
    destination_fragments = {}
    for bucket in plan.for_route(0, 0):
        assert bucket.total_bytes <= 32
        for entry in bucket.entries:
            source_slice = tuple(
                slice(start, start + length)
                for start, length in zip(entry.source_starts, entry.lengths)
            )
            canonical = state_dict[entry.source_key][source_slice]
            physical_fragment = canonical.permute(entry.physical_permutation).contiguous()
            destination_slice = tuple(
                slice(start, start + length)
                for start, length in zip(
                    entry.destination_starts,
                    entry.destination_lengths,
                )
            )
            physical[entry.target_name][destination_slice].copy_(physical_fragment)
            source_key, source_record = direct_fragment_record(
                entry.name,
                entry.logical_starts,
                entry.lengths,
                entry.dtype_name,
                canonical.contiguous().view(torch.uint8).numpy().tobytes(),
            )
            restored = physical[entry.target_name][destination_slice].permute(
                tuple(
                    entry.physical_permutation.index(axis)
                    for axis in range(len(entry.lengths))
                )
            )
            destination_key, destination_record = direct_fragment_record(
                entry.name,
                entry.logical_starts,
                entry.lengths,
                entry.dtype_name,
                restored.contiguous().view(torch.uint8).numpy().tobytes(),
            )
            source_fragments[source_key] = source_record
            destination_fragments[destination_key] = destination_record

    assert torch.equal(
        physical["model.layers.0.mlp.experts.w13_weight"],
        gate_up.transpose(1, 2).contiguous(),
    )
    assert torch.equal(
        physical["model.layers.0.mlp.experts.w2_weight"],
        down.transpose(1, 2).contiguous(),
    )
    assert torch.equal(physical["model.layers.0.mlp.gate.weight"], router)
    assert torch.equal(physical["model.norm.weight"], norm)
    assert aggregate_direct_content_identity(source_fragments) == (
        aggregate_direct_content_identity(destination_fragments)
    )


def test_hyper_deepseek_layout_keeps_hf_outer_and_router_buffer() -> None:
    """Hyper DeepSeek fuses routed experts but keeps all other HF tensor names."""

    class FakeHyperModel:
        """Expose the mixed HF/FusedMoE physical storage used by Hyper-vLLM."""

        def __init__(self) -> None:
            """Initialize FakeHyperModel state."""
            self.parameters = {
                "model.layers.1.mlp.experts.w13_weight": torch.empty(2, 3, 4),
                "model.layers.1.mlp.experts.w2_weight": torch.empty(2, 2, 3),
                "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.empty(4, 3),
                "model.layers.1.mlp.shared_experts.up_proj.weight": torch.empty(4, 3),
                "model.layers.1.mlp.shared_experts.down_proj.weight": torch.empty(3, 4),
                "model.layers.1.mlp.gate.weight": torch.empty(2, 3),
            }
            self.buffers = {
                "model.layers.1.mlp.gate.e_score_correction_bias": torch.empty(2),
                "model.rotary_emb.inv_freq": torch.empty(2),
            }

        def named_parameters(self) -> Any:
            """Return HF outer and routed FusedMoE parameters."""
            return self.parameters.items()

        def named_buffers(self) -> Any:
            """Return the persistent router bias and a non-policy buffer."""
            return self.buffers.items()

        @staticmethod
        def named_modules() -> tuple[tuple[str, object], ...]:
            """Expose fixture modules through the model interface."""
            return (("model.layers.1.mlp.experts", LocalExperts()),)

    class LocalExperts:
        """Mark one physical FusedMoE leaf as communication-free EP1."""

        hyper_local_expert_leaf = True

    hf_config = SimpleNamespace(
        hidden_size=3,
        intermediate_size=5,
        moe_intermediate_size=2,
        n_routed_experts=2,
        n_shared_experts=2,
    )
    descriptions = worker_module._hyper_deepseek_v3_direct_tensors(  # pylint: disable=protected-access
        FakeHyperModel(),
        hf_config,
        1,
    )
    descriptions_by_name = {description["name"]: description for description in descriptions}

    assert {
        "model.layers.1.mlp.experts.gate_proj.weight",
        "model.layers.1.mlp.experts.up_proj.weight",
        "model.layers.1.mlp.experts.down_proj.weight",
    }.issubset(descriptions_by_name)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.1.mlp.shared_experts.{projection}.weight"
        assert descriptions_by_name[name]["destination_name"] == name
    correction_name = "model.layers.1.mlp.gate.e_score_correction_bias"
    assert descriptions_by_name[correction_name]["destination_name"] == correction_name
    assert "model.rotary_emb.inv_freq" not in descriptions_by_name


def test_hyper_deepseek_ep1_replicates_complete_experts_across_rollout_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hyper EP1 keeps full expert matrices on every request-DP worker."""

    class LocalExperts:
        """Mark one physical FusedMoE leaf as communication-free EP1."""

        hyper_local_expert_leaf = True

    class FakeHyperModel:
        """Expose complete EP1 Ascend expert storage and its module owner."""

        def __init__(self) -> None:
            """Initialize FakeHyperModel state."""
            self.parameters = {
                "model.layers.1.mlp.experts.w13_weight": torch.empty(2, 3, 8),
                "model.layers.1.mlp.experts.w2_weight": torch.empty(2, 4, 3),
                "model.norm.weight": torch.empty(3),
            }

        def named_parameters(self) -> Any:
            """Expose fixture parameters through the model interface."""
            return self.parameters.items()

        @staticmethod
        def named_buffers() -> tuple[object, ...]:
            """Expose fixture buffers through the model interface."""
            return ()

        @staticmethod
        def named_modules() -> tuple[tuple[str, object], ...]:
            """Expose fixture modules through the model interface."""
            return (("model.layers.1.mlp.experts", LocalExperts()),)

    hf_config = SimpleNamespace(
        hidden_size=3,
        intermediate_size=5,
        moe_intermediate_size=4,
        n_routed_experts=2,
        n_shared_experts=2,
    )
    tensors = worker_module._hyper_deepseek_v3_direct_tensors(  # pylint: disable=protected-access
        FakeHyperModel(),
        hf_config,
        1,
        4,
    )
    expert_tensors = [tensor for tensor in tensors if ".mlp.experts." in tensor["name"]]
    assert expert_tensors
    assert all(tensor["placement"] == "replicate" for tensor in expert_tensors)

    workers = [
        {
            "dp_rank": dp_rank,
            "dp_size": 4,
            "tp_rank": 0,
            "tp_size": 1,
            "tensors": tensors,
        }
        for dp_rank in range(4)
    ]

    class Client:
        @staticmethod
        def get_world_size() -> int:
            """Provide the get world size fixture for this regression."""
            return 4

        @staticmethod
        def collective_rpc(method: str) -> list[Mapping[str, Any]]:
            """Provide the fixture response for a collective RPC."""
            assert method == "get_direct_reshard_layout"
            return workers

    monkeypatch.setattr(
        transfer_module,
        "coordinator_call",
        lambda _operation, callback: callback(),
    )
    registration = SimpleNamespace(
        family="deepseek_v3",
        is_hyper=True,
        model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )
    transfer = ColocatedDirectReshardWeightTransfer(
        registration,
        data_parallel_size=4,
        tensor_parallel_size=1,
    )

    representatives = transfer._query_destination_workers(Client())  # pylint: disable=protected-access

    assert len(representatives) == 1
    assert representatives[0]["tp_rank"] == 0
    assert representatives[0]["tp_size"] == 1


def test_deepseek_dp4_direct_plan_targets_flattened_moe_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EP-off DP4 sends routed intermediate shards while replicating dense weights."""

    class FakeNativeModel:
        """Expose one DP rank's refit-time flattened FusedMoE storage."""

        def __init__(self) -> None:
            """Initialize FakeNativeModel state."""
            self.parameters = {
                "model.layers.1.mlp.experts.w13_weight": torch.empty(2, 2, 3),
                "model.layers.1.mlp.experts.w2_weight": torch.empty(2, 3, 1),
                "model.norm.weight": torch.empty(3),
            }

        def named_parameters(self) -> Any:
            """Return physical native-vLLM parameters."""
            return self.parameters.items()

    hf_config = SimpleNamespace(
        hidden_size=3,
        intermediate_size=5,
        moe_intermediate_size=4,
        n_routed_experts=2,
        n_shared_experts=2,
    )
    state_dict = {
        "model.layers.1.mlp.experts.gate_up_proj": torch.arange(
            48,
            dtype=torch.float32,
        ).view(2, 8, 3),
        "model.layers.1.mlp.experts.down_proj": torch.arange(
            24,
            dtype=torch.float32,
        ).view(2, 3, 4),
        "model.norm.weight": torch.arange(3, dtype=torch.float32),
    }
    representative = {
        "dp_rank": 0,
        "dp_size": 4,
        "tp_rank": 0,
        "tp_size": 1,
        "tensors": worker_module._native_moe_direct_tensors(  # pylint: disable=protected-access
            FakeNativeModel(),
            hf_config,
            1,
            4,
            family="deepseek_v3",
        ),
    }

    class Client:
        """Return the representative value exposed by internal-DP RPC."""

        @staticmethod
        def get_world_size() -> int:
            """Provide the get world size fixture for this regression."""
            return 4

        @staticmethod
        def collective_rpc(method: str) -> list[Mapping[str, Any]]:
            """Provide the fixture response for a collective RPC."""
            if method == "prepare_direct_reshard_layout":
                return [{"prepared": True}]
            return [representative]

    monkeypatch.setattr(
        transfer_module,
        "coordinator_call",
        lambda _operation, callback: callback(),
    )
    transfer = ColocatedDirectReshardWeightTransfer(
        _deepseek_v3_registration(),
        data_parallel_size=4,
        tensor_parallel_size=1,
    )
    workers = transfer._query_destination_workers(Client())  # pylint: disable=protected-access
    assert [(worker["tp_rank"], worker["tp_size"]) for worker in workers] == [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ]

    adapter = build_model_weight_adapter(_deepseek_v3_registration())
    sources = resolve_source_layouts(
        [adapter.direct_source_descriptions(state_dict, 0)]
    )
    global_shapes = {source.name: source.global_shape for source in sources}
    destinations = resolve_destination_layouts(workers, global_shapes)
    plan = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=64,
    )

    assert plan.destination_tp_size == 4
    assert set(plan.buckets) == {(0, 0), (0, 1), (0, 2), (0, 3)}
    for target_rank in range(4):
        entries = [
            entry
            for bucket in plan.for_route(0, target_rank)
            for entry in bucket.entries
        ]
        routed = [entry for entry in entries if ".mlp.experts." in entry.name]
        assert {entry.name.rsplit(".", maxsplit=2)[-2] for entry in routed} == {
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        gate_up = [
            entry
            for entry in routed
            if ".gate_proj." in entry.name or ".up_proj." in entry.name
        ]
        assert all(entry.logical_starts[1] == target_rank for entry in gate_up)
        down = next(entry for entry in routed if ".down_proj." in entry.name)
        assert down.logical_starts[2] == target_rank
        norm = [entry for entry in entries if entry.name == "model.norm.weight"]
        assert sum(entry.numel for entry in norm) == 3


def test_direct_source_identity_hashes_bounded_plan_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production source identity follows the plan without a full tensor gather."""
    weight = torch.arange(16, dtype=torch.float32).view(4, 4)
    plan = build_direct_reshard_plan(
        (
            SourceTensorLayout(
                name="weight",
                dtype_name="float32",
                element_size=4,
                global_shape=(4, 4),
                source_rank=0,
                region=TensorRegion((0, 0), (4, 4)),
            ),
        ),
        (
            DestinationTensorLayout(
                name="weight",
                dtype_name="float32",
                element_size=4,
                global_shape=(4, 4),
                tp_rank=0,
                tp_size=1,
                placement="replicate",
                shard_dim=None,
                region=TensorRegion((0, 0), (4, 4)),
            ),
        ),
        source_world_size=1,
        bucket_size_bytes=16,
    )

    def _all_gather_object(output: list[Any], value: Any) -> None:
        """Return the single-rank object through the distributed API shape."""
        output[0] = value

    monkeypatch.setattr(
        transfer_module,
        "platform",
        SimpleNamespace(
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            all_gather_object=_all_gather_object,
            tensor_to_numpy=lambda tensor: tensor.numpy(),
        ),
    )

    identities = (  # pylint: disable=protected-access
        DirectReshardHCCLWeightTransfer._distributed_source_content_identities(
            {"weight": weight},
            plan,
        )
    )

    assert set(identities) == {0}
    assert identities[0]["total_bytes"] == weight.numel() * weight.element_size()
    assert identities[0]["fragment_count"] == plan.fragment_count
    assert all(
        bucket.total_bytes <= 16
        for buckets in plan.buckets.values()
        for bucket in buckets
    )


def test_metadata_resolves_fsdp_rows_and_tp_columns() -> None:
    """Raw rank/worker descriptions resolve to the same global cross-axis plan."""
    rank_descriptions = [
        [
            {
                "name": "weight",
                "dtype_name": "bfloat16",
                "element_size": 2,
                "global_shape": [4, 4],
                "local_shape": [2, 4],
                "source_rank": rank,
                "shard_dim": 0,
            }
        ]
        for rank in range(2)
    ]
    worker_descriptions = [
        {
            "tp_rank": rank,
            "tp_size": 2,
            "tensors": [
                {
                    "name": "weight",
                    "dtype_name": "bfloat16",
                    "element_size": 2,
                    "local_shape": [4, 2],
                    "placement": "shard",
                    "shard_dim": 1,
                }
            ],
        }
        for rank in range(2)
    ]

    sources = resolve_source_layouts(rank_descriptions)
    destinations = resolve_destination_layouts(
        worker_descriptions,
        {"weight": (4, 4)},
    )
    plan = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=2,
        bucket_size_bytes=128,
    )

    assert [source.region.starts for source in sources] == [(0, 0), (2, 0)]
    assert [destination.region.starts for destination in destinations] == [
        (0, 0),
        (0, 2),
    ]
    assert plan.route_count == 4


def test_source_metadata_resolves_multi_axis_fsdp_tp_regions() -> None:
    """FSDP and Trainer TP placements form explicit global source rectangles."""
    shard_row = SimpleNamespace(dim=0, is_shard=lambda: True)
    shard_column = SimpleNamespace(dim=1, is_shard=lambda: True)

    class _Mesh:
        """Expose one rank coordinate in a 2x2 Trainer mesh."""

        ndim = 2

        def __init__(self, coordinate: tuple[int, int]) -> None:
            """Initialize Mesh state."""
            self.coordinate = coordinate

        @staticmethod
        def size(_mesh_dim: int) -> int:
            """Provide the size fixture for this regression."""
            return 2

        def get_coordinate(self) -> tuple[int, int]:
            """Provide the get coordinate fixture for this regression."""
            return self.coordinate

    descriptions = []
    for source_rank, coordinate in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        local = torch.zeros((4, 2), dtype=torch.bfloat16)
        tensor = SimpleNamespace(
            shape=(8, 4),
            placements=(shard_row, shard_column),
            device_mesh=_Mesh(coordinate),
            to_local=lambda local=local: local,
        )
        descriptions.append(
            [describe_source_tensor("weight", tensor, source_rank)]
        )

    layouts = resolve_source_layouts(descriptions)

    assert [(layout.source_rank, layout.region.starts) for layout in layouts] == [
        (0, (0, 0)),
        (1, (0, 2)),
        (2, (4, 0)),
        (3, (4, 2)),
    ]
    assert all(layout.region.lengths == (4, 2) for layout in layouts)


def test_source_metadata_applies_inner_tp_before_outer_fsdp_on_same_dim() -> None:
    """Nested TP then FSDP shards retain TP-major global tensor ordering."""
    shard = SimpleNamespace(dim=0, is_shard=lambda: True)

    class _Mesh:
        ndim = 2

        def __init__(self, coordinate: tuple[int, int]) -> None:
            """Initialize Mesh state."""
            self.coordinate = coordinate

        @staticmethod
        def size(_mesh_dim: int) -> int:
            """Provide the size fixture for this regression."""
            return 2

        def get_coordinate(self) -> tuple[int, int]:
            """Provide the get coordinate fixture for this regression."""
            return self.coordinate

    starts = []
    for coordinate in ((0, 0), (0, 1), (1, 0), (1, 1)):
        local = torch.zeros((2, 4), dtype=torch.bfloat16)
        tensor = SimpleNamespace(
            shape=(8, 4),
            placements=(shard, shard),
            device_mesh=_Mesh(coordinate),
            to_local=lambda local=local: local,
        )
        starts.append(describe_source_tensor("weight", tensor, 0)["region_starts"])

    assert starts == [[0, 0], [4, 0], [2, 0], [6, 0]]


def test_one_tp_worker_rejects_a_disagreeing_layout() -> None:
    """One worker's incompatible placement invalidates the complete TP layout."""
    workers = [
        {
            "tp_rank": 0,
            "tp_size": 2,
            "tensors": [
                {
                    "name": "weight",
                    "dtype_name": "bfloat16",
                    "element_size": 2,
                    "local_shape": [4, 2],
                    "placement": "shard",
                    "shard_dim": 1,
                }
            ],
        },
        {
            "tp_rank": 1,
            "tp_size": 2,
            "tensors": [
                {
                    "name": "weight",
                    "dtype_name": "bfloat16",
                    "element_size": 2,
                    "local_shape": [4, 4],
                    "placement": "replicate",
                    "shard_dim": None,
                }
            ],
        },
    ]

    with pytest.raises(ValueError, match="layout differs across TP workers"):
        resolve_destination_layouts(workers, {"weight": (4, 4)})


def test_internal_dp2_tp2_resolves_explicit_physical_workers() -> None:
    """DP-major vLLM workers retain TP fragments on their actual colocated NPUs."""
    workers = resolve_physical_worker_topology(
        ("host-4", "host-5", "host-6", "host-7"),
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    assert [
        (worker.dp_rank, worker.tp_rank, worker.physical_device_id)
        for worker in workers
    ] == [
        (0, 0, "host-4"),
        (0, 1, "host-5"),
        (1, 0, "host-6"),
        (1, 1, "host-7"),
    ]
    with pytest.raises(ValueError, match="must match DP x TP"):
        resolve_physical_worker_topology(
            ("host-4", "host-5"),
            data_parallel_size=2,
            tensor_parallel_size=2,
        )
    with pytest.raises(ValueError, match="must be unique"):
        resolve_physical_worker_topology(
            ("host-4", "host-5", "host-4", "host-7"),
            data_parallel_size=2,
            tensor_parallel_size=2,
        )


def test_worker_manifests_cover_dp2_tp2_and_compare_with_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each physical worker writes and verifies its own full-policy oracle."""

    def fake_manifest(worker: Any) -> Any:
        """Return a controlled parameter manifest for verification."""
        tensor_digest = f"tp-{worker.tp_rank}"
        return {
            "dp_rank": worker.dp_rank,
            "dp_size": 2,
            "tp_rank": worker.tp_rank,
            "tp_size": 2,
            "physical_device_id": f"host-{4 + worker.dp_rank * 2 + worker.tp_rank}",
            "parameter_count": 1,
            "total_bytes": 8,
            "manifest_sha256": tensor_digest,
            "tensors": {
                "model.norm.weight": {
                    "dtype": "bfloat16",
                    "shape": [4],
                    "num_bytes": 8,
                    "sha256": tensor_digest,
                }
            },
        }

    monkeypatch.setattr(worker_module, "get_all_parameter_manifest", fake_manifest)
    workers = [
        SimpleNamespace(
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            _hyper_loaded_policy_version=1,
        )
        for dp_rank in range(2)
        for tp_rank in range(2)
    ]
    for worker in workers:
        worker_module.write_parameter_manifest(
            worker,
            output_dir=str(tmp_path),
            strategy="full_gather",
            policy_version=1,
            rollout_replica_rank=0,
            expected_data_parallel_size=2,
            oracle_run_id="oracle-1",
        )
    for worker in workers:
        result = worker_module.write_parameter_manifest(
            worker,
            output_dir=str(tmp_path),
            strategy="direct_reshard",
            policy_version=1,
            rollout_replica_rank=0,
            expected_data_parallel_size=2,
            oracle_run_id="oracle-1",
            oracle_dir=str(tmp_path),
            oracle_strategy="full_gather",
        )
        assert result["oracle_match"] is True

    assert len(tuple(tmp_path.glob("full_gather-*.json"))) == 4
    assert len(tuple(tmp_path.glob("direct_reshard-*.json"))) == 4


def test_worker_manifest_compares_trainer_derived_expectation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each worker must match its independent Trainer-derived local state."""
    actual = {
        "dp_rank": 0,
        "dp_size": 1,
        "tp_rank": 0,
        "tp_size": 1,
        "physical_device_id": "host-1",
        "parameter_count": 1,
        "total_bytes": 8,
        "manifest_sha256": "manifest-1",
        "tensors": {
            "model.norm.weight": {
                "dtype": "bfloat16",
                "shape": [4],
                "num_bytes": 8,
                "sha256": "tensor-1",
            }
        },
    }
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    expected = {
        **actual,
        "oracle_run_id": "run-1",
        "policy_version": 1,
        "source_manifest_sha256": "source-1",
    }
    (expected_dir / "version1-dp0-tp0.json").write_text(
        json.dumps(expected),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker_module, "get_all_parameter_manifest", lambda _worker: dict(actual))
    worker = SimpleNamespace(_hyper_loaded_policy_version=1)

    result = worker_module.write_parameter_manifest(
        worker,
        output_dir=str(tmp_path),
        strategy="full_gather",
        policy_version=1,
        rollout_replica_rank=0,
        expected_data_parallel_size=1,
        oracle_run_id="run-1",
        expected_dir=str(expected_dir),
    )

    assert result["source_match"] is True
    written = json.loads(
        (tmp_path / "full_gather-version1-replica0-dp0-tp0.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["source_manifest_sha256"] == "source-1"


def test_worker_manifest_preserves_tied_logical_parameter_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native tied weights remain visible under both state-dict identities."""
    model = torch.nn.Module()
    tied_parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model.register_parameter("embed_tokens_weight", tied_parameter)
    model.register_parameter("lm_head_weight", tied_parameter)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 0,
            "tp_size": 1,
            "physical_device_id": "host-0",
        },
    )

    manifest = worker_module.get_all_parameter_manifest(worker)

    assert manifest["parameter_count"] == 2
    assert set(manifest["tensors"]) == {"embed_tokens_weight", "lm_head_weight"}
    assert (
        manifest["tensors"]["embed_tokens_weight"]
        == manifest["tensors"]["lm_head_weight"]
    )


def test_deepseek_runtime_contract_reports_required_vllm_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P6 exposes absorbed MLA/FusedMoE coverage and rejects implicit EP."""
    class _AbsorbedMLA:
        @staticmethod
        def process_weights_after_loading(_dtype: object) -> None:
            """Expose the pinned vLLM absorbed-weight lifecycle contract."""

    absorbed_type = _AbsorbedMLA
    fused_type = type("FusedMoE", (), {})
    modules = [
        SimpleNamespace(mla_attn=absorbed_type())
        for _ in range(27)
    ]
    modules.extend(
        fused_type()
        for _ in range(26)
    )
    for module in modules[27:]:
        module.w13_weight = object()
        module.w2_weight = object()
    ownership = {
        "outer_model": "Transformers DeepseekV3ForCausalLM",
        "absorbed_mla": "vLLM: paged cache",
        "routed_experts": "vLLM FusedMoE: Ascend kernel",
    }
    model = SimpleNamespace(
        modules=lambda: modules,
        hyper_component_ownership=ownership,
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
        model_config=SimpleNamespace(
            use_mla=True,
            hf_config=SimpleNamespace(
                num_hidden_layers=27,
                first_k_dense_replace=1,
                q_lora_rank=None,
                kv_lora_rank=512,
            ),
        ),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
    )
    monkeypatch.setattr(worker_module, "_is_deepseek_v3_worker", lambda _worker: True)
    monkeypatch.setattr(
        worker_module,
        "_is_hyper_deepseek_v3_worker",
        lambda _worker: True,
    )

    contract = worker_module.get_deepseek_v3_runtime_contract(worker)

    assert contract["use_mla"] is True
    assert contract["q_lora_rank"] is None
    assert contract["kv_lora_rank"] == 512
    assert contract["absorbed_mla_layer_count"] == 27
    assert contract["hyper_mla_layer_count"] == 0
    assert contract["fused_moe_layer_count"] == 26
    assert contract["enable_expert_parallel"] is False
    assert contract["component_ownership"] == ownership


def test_worker_manifest_skips_cross_arm_oracle_for_different_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Independent Trainer states remain source-verified without a false cross-arm mismatch."""
    actual = {
        "dp_rank": 0,
        "dp_size": 1,
        "tp_rank": 0,
        "tp_size": 1,
        "physical_device_id": "host-1",
        "parameter_count": 1,
        "total_bytes": 8,
        "manifest_sha256": "direct-manifest",
        "tensors": {"model.norm.weight": {"sha256": "direct"}},
    }
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    expected = {
        **actual,
        "oracle_run_id": "run-1",
        "policy_version": 2,
        "source_manifest_sha256": "direct-source",
    }
    (expected_dir / "version2-dp0-tp0.json").write_text(
        json.dumps(expected),
        encoding="utf-8",
    )
    oracle = {
        **actual,
        "manifest_sha256": "full-manifest",
        "oracle_run_id": "run-1",
        "source_manifest_sha256": "full-source",
        "tensors": {"model.norm.weight": {"sha256": "full"}},
    }
    (tmp_path / "full_gather-version2-replica0-dp0-tp0.json").write_text(
        json.dumps(oracle),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker_module, "get_all_parameter_manifest", lambda _worker: dict(actual))

    result = worker_module.write_parameter_manifest(
        SimpleNamespace(_hyper_loaded_policy_version=2),
        output_dir=str(tmp_path),
        strategy="direct_reshard",
        policy_version=2,
        rollout_replica_rank=0,
        expected_data_parallel_size=1,
        oracle_run_id="run-1",
        oracle_dir=str(tmp_path),
        oracle_strategy="full_gather",
        expected_dir=str(expected_dir),
    )

    assert result["source_match"] is True
    assert result["oracle_comparable"] is False
    assert result["oracle_match"] is None


def test_colocated_device_topology_uses_visible_order_not_trainer_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permuted Trainer gather cannot change vLLM's physical worker mapping."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2,4,5")

    ordered = ColocatedDirectReshardWeightTransfer._rollout_device_order(  # pylint: disable=protected-access
        ["host-4", "host-1", "host-5", "host-2"]
    )

    assert ordered == ("host-1", "host-2", "host-4", "host-5")


def test_ipc_receiver_retains_rebuilt_buffers_until_stream_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consumer IPC mappings must outlive all asynchronous destination copies."""
    rebuilt_reference = None
    synchronize_count = 0

    def rebuild_npu_tensor(*_args: Any) -> Any:
        """Rebuild the fixture tensor for the simulated IPC receive."""
        nonlocal rebuilt_reference
        packed = torch.tensor([0, 0, 128, 63], dtype=torch.uint8)
        rebuilt_reference = weakref.ref(packed)
        return packed

    def synchronize() -> None:
        """Simulate stream synchronization for the lifecycle test."""
        nonlocal synchronize_count
        assert rebuilt_reference is not None
        assert rebuilt_reference() is not None
        synchronize_count += 1

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-1"),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.multiprocessing.reductions",
        SimpleNamespace(rebuild_npu_tensor=rebuild_npu_tensor),
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {"dp_rank": 0, "tp_rank": 0},
    )
    monkeypatch.setattr(worker_module, "_is_direct_reshard_worker", lambda _worker: True)
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(current_device_index=lambda: 0),
        raising=False,
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=synchronize)),
        raising=False,
    )
    parameter = torch.zeros(1, dtype=torch.float32)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(
                named_parameters=lambda: (("weight", parameter),)
            )
        ),
        _weight_update_active=True,
        _hyper_loaded_policy_version=0,
        _hyper_pending_policy_version=None,
    )
    payload = {
        "worker_topology": [
            {"physical_device_id": "host-1", "dp_rank": 0, "tp_rank": 0}
        ],
        "buckets_by_tp": {
            0: [
                {
                    "ipc_handles": {"host-1": tuple(range(7))},
                    "metadata": {
                        "total_bytes": 4,
                        "entries": [
                            {
                                "name": "weight",
                                "dtype_name": "float32",
                                "element_size": 4,
                                "destination_starts": [0],
                                "lengths": [1],
                                "num_bytes": 4,
                                "buffer_offset": 0,
                            }
                        ],
                    },
                }
            ]
        },
    }

    result = worker_module.receive_ipc_direct_reshard(
        worker,
        payload_pickled=base64.b64encode(pickle.dumps(payload)).decode("ascii"),
        policy_version=1,
    )

    assert result["received"] is True
    assert torch.equal(parameter, torch.ones_like(parameter))
    assert synchronize_count == 1

    payload["buckets_by_tp"][0][0]["metadata"]["entries"].append(
        {
            "name": "missing_weight",
            "dtype_name": "float32",
            "element_size": 4,
            "destination_starts": [0],
            "lengths": [1],
            "num_bytes": 4,
            "buffer_offset": 0,
        }
    )
    with pytest.raises(ValueError, match="parameter 'missing_weight' is missing"):
        worker_module.receive_ipc_direct_reshard(
            worker,
            payload_pickled=base64.b64encode(pickle.dumps(payload)).decode("ascii"),
            policy_version=1,
        )
    assert synchronize_count == 2


def test_middle_bucket_failure_requires_abort_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial IPC write keeps V0 identity until the controller aborts it."""
    rebuild_count = 0

    def rebuild_npu_tensor(*_args: Any) -> Any:
        """Rebuild the fixture tensor for the simulated IPC receive."""
        nonlocal rebuild_count
        rebuild_count += 1
        return torch.tensor([0, 0, 128, 63], dtype=torch.uint8)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-1"),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.multiprocessing.reductions",
        SimpleNamespace(rebuild_npu_tensor=rebuild_npu_tensor),
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {"dp_rank": 0, "tp_rank": 0},
    )
    monkeypatch.setattr(worker_module, "_is_direct_reshard_worker", lambda _worker: True)
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(current_device_index=lambda: 0),
        raising=False,
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
        raising=False,
    )
    first = torch.zeros(1, dtype=torch.float32)
    third = torch.zeros(1, dtype=torch.float32)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(
                named_parameters=lambda: (("first", first), ("third", third))
            )
        ),
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_loaded_policy_version=0,
        _hyper_pending_policy_version=None,
    )

    def bucket(name: str) -> dict[str, object]:
        """Provide the bucket fixture for this regression."""
        return {
            "ipc_handles": {"host-1": tuple(range(7))},
            "metadata": {
                "total_bytes": 4,
                "entries": [
                    {
                        "name": name,
                        "dtype_name": "float32",
                        "element_size": 4,
                        "destination_starts": [0],
                        "lengths": [1],
                        "num_bytes": 4,
                        "buffer_offset": 0,
                    }
                ],
            },
        }

    payload = {
        "worker_topology": [
            {"physical_device_id": "host-1", "dp_rank": 0, "tp_rank": 0}
        ],
        "buckets_by_tp": {0: [bucket("first"), bucket("missing"), bucket("third")]},
    }

    with pytest.raises(ValueError, match="parameter 'missing' is missing"):
        worker_module.receive_ipc_direct_reshard(
            worker,
            payload_pickled=base64.b64encode(pickle.dumps(payload)).decode("ascii"),
            policy_version=1,
        )

    assert torch.equal(first, torch.ones_like(first))
    assert torch.equal(third, torch.zeros_like(third))
    assert rebuild_count == 2
    assert worker._hyper_loaded_policy_version == 0
    assert worker._hyper_pending_policy_version is None

    result = worker_module.abort_weight_update(worker, restore_policy_version=0)
    assert result["aborted"] is True
    assert worker._weight_update_active is False
    assert worker._hyper_loaded_policy_version == 0


def test_full_gather_manifest_does_not_compare_or_overwrite_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct fallback writes a separate manifest without treating itself as oracle."""
    requests = []
    client = SimpleNamespace(
        is_server_owner=True,
        collective_rpc=lambda method, kwargs: requests.append((method, kwargs))
        or [{"written": True}],
    )
    monkeypatch.setenv("HYPER_RL_WEIGHT_MANIFEST_DIR", "/results/manifests")
    monkeypatch.setenv("HYPER_RL_WEIGHT_MANIFEST_ORACLE_DIR", "/results/manifests")
    monkeypatch.setenv("HYPER_RL_WEIGHT_ORACLE_RUN_ID", "oracle-1")
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)

    transfer_module._write_rollout_parameter_manifest(  # pylint: disable=protected-access
        client,
        strategy="full_gather_fallback",
        policy_version=1,
        data_parallel_size=2,
    )

    assert requests[0][0] == "write_parameter_manifest"
    assert requests[0][1]["strategy"] == "full_gather_fallback"
    assert requests[0][1]["oracle_dir"] == "/results/manifests"
    assert requests[0][1]["oracle_strategy"] == "full_gather"


def test_direct_reshard_values_equal_full_gather_then_tp_slice() -> None:
    """Every planned FSDP fragment must rebuild the exact reference TP tensors."""
    parameter_specs = {
        "model.embed_tokens.weight": ((10, 6), 0),
        "model.layers.0.self_attn.q_proj.weight": ((12, 6), 0),
        "model.layers.0.self_attn.o_proj.weight": ((6, 12), 1),
        "model.layers.0.mlp.gate_proj.weight": ((14, 6), 0),
        "model.layers.0.mlp.down_proj.weight": ((6, 14), 1),
        "model.layers.0.input_layernorm.weight": ((6,), None),
        "lm_head.weight": ((10, 6), 0),
    }
    full_parameters = {}
    sources = []
    destinations = []
    source_values = {}
    destination_values = {}
    next_value = 1
    for name, (shape, destination_shard_dim) in parameter_specs.items():
        numel = 1
        for size in shape:
            numel *= size
        full = torch.arange(
            next_value,
            next_value + numel,
            dtype=torch.float32,
        ).view(shape)
        next_value += numel + 17
        full_parameters[name] = full
        source_offset = 0
        for source_rank, local_size in enumerate((shape[0] // 2, shape[0] - shape[0] // 2)):
            starts = (source_offset,) + (0,) * (len(shape) - 1)
            lengths = (local_size,) + shape[1:]
            sources.append(
                SourceTensorLayout(
                    name=name,
                    dtype_name="float32",
                    element_size=4,
                    global_shape=shape,
                    source_rank=source_rank,
                    region=TensorRegion(starts, lengths),
                )
            )
            source_values[(name, source_rank)] = full[
                source_offset : source_offset + local_size
            ].clone()
            source_offset += local_size
        destination_offset = 0
        for tp_rank in range(2):
            if destination_shard_dim is None:
                starts = (0,) * len(shape)
                lengths = shape
                placement = "replicate"
            else:
                local_size = shape[destination_shard_dim] // 2
                starts_list = [0] * len(shape)
                starts_list[destination_shard_dim] = destination_offset
                lengths_list = list(shape)
                lengths_list[destination_shard_dim] = local_size
                starts = tuple(starts_list)
                lengths = tuple(lengths_list)
                destination_offset += local_size
                placement = "shard"
            destinations.append(
                DestinationTensorLayout(
                    name=name,
                    dtype_name="float32",
                    element_size=4,
                    global_shape=shape,
                    tp_rank=tp_rank,
                    tp_size=2,
                    placement=placement,
                    shard_dim=destination_shard_dim,
                    region=TensorRegion(starts, lengths),
                )
            )
            destination_values[(name, tp_rank)] = torch.full(
                lengths,
                float("nan"),
                dtype=torch.float32,
            )

    plan = build_direct_reshard_plan(
        tuple(sources),
        tuple(destinations),
        source_world_size=2,
        bucket_size_bytes=32,
    )
    for (source_rank, tp_rank), buckets in plan.buckets.items():
        for bucket in buckets:
            assert bucket.total_bytes <= 32
            for entry in bucket.entries:
                source_slice = tuple(
                    slice(start, start + length)
                    for start, length in zip(entry.source_starts, entry.lengths)
                )
                destination_slice = tuple(
                    slice(start, start + length)
                    for start, length in zip(entry.destination_starts, entry.lengths)
                )
                destination_values[(entry.name, tp_rank)][destination_slice].copy_(
                    source_values[(entry.name, source_rank)][source_slice]
                )

    for destination in destinations:
        global_slice = tuple(
            slice(start, start + length)
            for start, length in zip(
                destination.region.starts,
                destination.region.lengths,
            )
        )
        assert torch.equal(
            destination_values[(destination.name, destination.tp_rank)],
            full_parameters[destination.name][global_slice],
        )


def test_native_qwen3_fused_parameters_match_reference_tp_slices() -> None:
    """Native QKV and gate/up storage must equal reference TP slicing and fusion."""

    class FakeModel:
        """Expose deterministic native vLLM parameter names and shapes."""

        def __init__(self) -> None:
            """Initialize FakeModel state."""
            self.parameters = {
                "model.layers.0.self_attn.qkv_proj.weight": torch.empty(
                    8, 8, dtype=torch.int64
                ),
                "model.layers.0.mlp.gate_up_proj.weight": torch.empty(
                    12, 8, dtype=torch.int64
                ),
            }

        def named_parameters(self) -> Any:
            """Return the physical native vLLM parameters."""
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
                FakeModel(),
                hf_config,
                tp_rank,
                2,
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
    global_shapes = {
        name: tuple(value.shape) for name, value in full_parameters.items()
    }
    destinations = resolve_destination_layouts(worker_descriptions, global_shapes)
    sources = []
    source_values = {}
    for name, full in full_parameters.items():
        offset = 0
        for source_rank in range(2):
            local_size = full.shape[0] // 2
            sources.append(
                SourceTensorLayout(
                    name=name,
                    dtype_name="int64",
                    element_size=8,
                    global_shape=tuple(full.shape),
                    source_rank=source_rank,
                    region=TensorRegion((offset, 0), (local_size, full.shape[1])),
                )
            )
            source_values[(name, source_rank)] = full[offset : offset + local_size]
            offset += local_size
    plan = build_direct_reshard_plan(
        tuple(sources),
        destinations,
        source_world_size=2,
        bucket_size_bytes=64,
    )
    physical = {
        ("model.layers.0.self_attn.qkv_proj.weight", tp_rank): torch.full(
            (8, 8),
            -1,
            dtype=torch.int64,
        )
        for tp_rank in range(2)
    }
    physical.update(
        {
            ("model.layers.0.mlp.gate_up_proj.weight", tp_rank): torch.full(
                (12, 8),
                -1,
                dtype=torch.int64,
            )
            for tp_rank in range(2)
        }
    )
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
                physical[(entry.target_name, tp_rank)][destination_slice].copy_(
                    source_values[(entry.name, source_rank)][source_slice]
                )

    for tp_rank in range(2):
        qkv_reference = torch.cat(
            (
                full_parameters["model.layers.0.self_attn.q_proj.weight"][
                    tp_rank * 4 : (tp_rank + 1) * 4
                ],
                full_parameters["model.layers.0.self_attn.k_proj.weight"][
                    tp_rank * 2 : (tp_rank + 1) * 2
                ],
                full_parameters["model.layers.0.self_attn.v_proj.weight"][
                    tp_rank * 2 : (tp_rank + 1) * 2
                ],
            )
        )
        gate_up_reference = torch.cat(
            (
                full_parameters["model.layers.0.mlp.gate_proj.weight"][
                    tp_rank * 6 : (tp_rank + 1) * 6
                ],
                full_parameters["model.layers.0.mlp.up_proj.weight"][
                    tp_rank * 6 : (tp_rank + 1) * 6
                ],
            )
        )
        assert torch.equal(
            physical[("model.layers.0.self_attn.qkv_proj.weight", tp_rank)],
            qkv_reference,
        )
        assert torch.equal(
            physical[("model.layers.0.mlp.gate_up_proj.weight", tp_rank)],
            gate_up_reference,
        )


def test_native_qwen3_direct_rejects_grouped_kv_head_replication() -> None:
    """Unsupported grouped KV replication must fail before publishing an invalid layout."""
    hf_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    parameter = torch.empty(6, 8)

    with pytest.raises(ValueError, match="grouped KV-head replication"):
        worker_module._native_qwen3_qkv_descriptions(  # pylint: disable=protected-access
            "model.layers.0.self_attn.qkv_proj.weight",
            parameter,
            hf_config,
            tp_size=4,
        )


def test_direct_transfer_extracts_local_state_without_full_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct strategy requests FSDP local DTensors, never a full state dict."""
    calls = []
    tensor = object()

    def get_model_state_dict(
        payload: object,
        *,
        options: object,
    ) -> dict[str, object]:
        """Expose local fixture state without gathering a full model."""
        calls.append((payload, options))
        return {"model.norm.weight": tensor}

    fake_platform = SimpleNamespace(
        get_model_state_dict=get_model_state_dict,
        is_tensor=lambda value: value is tensor,
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    payload = object()

    state_dict = DirectReshardHCCLWeightTransfer._local_state_dict(payload)

    assert state_dict == {"model.norm.weight": tensor}
    assert len(calls) == 1
    assert calls[0][0] is payload
    assert calls[0][1].full_state_dict is False
    assert calls[0][1].cpu_offload is False


def test_full_transfer_requests_local_device_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded full strategy requests local DTensors, never a full state dict."""
    calls = []
    tensor = object()

    def get_model_state_dict(
        payload: object,
        *,
        options: object,
    ) -> dict[str, object]:
        """Expose local fixture state without gathering a full model."""
        calls.append((payload, options))
        return {"model.norm.weight": tensor}

    fake_platform = SimpleNamespace(
        get_model_state_dict=get_model_state_dict,
        is_tensor=lambda value: value is tensor,
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    payload = object()

    transfer = ColocatedFullGatherWeightTransfer(_deepseek_v3_registration())
    state_dict = transfer._mapped_local_state_dict(payload)  # pylint: disable=protected-access

    assert state_dict == {"model.norm.weight": tensor}
    assert len(calls) == 1
    assert calls[0][0] is payload
    assert calls[0][1].full_state_dict is False
    assert calls[0][1].cpu_offload is False


def test_direct_transfer_aliases_tied_lm_head_without_tensor_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tied rollout lm_head reuses the Actor embedding DTensor as its source."""
    tensor = object()
    model = SimpleNamespace(
        is_hyper=True,
        family="qwen3",
        model=SimpleNamespace(tie_word_embeddings=True),
        actor_weight_name=lambda name: name,
    )
    transfer = DirectReshardHCCLWeightTransfer(model)
    monkeypatch.setattr(
        transfer,
        "_local_state_dict",
        lambda payload: {"model.embed_tokens.weight": tensor},
    )

    state_dict = transfer._mapped_local_state_dict(object())

    assert state_dict["lm_head.weight"] is tensor
    assert state_dict["model.embed_tokens.weight"] is tensor


def test_direct_transfer_reuses_one_compiled_layout_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consecutive policy publications must not rebuild unchanged layout metadata."""
    model = SimpleNamespace(
        family="qwen3",
        model=SimpleNamespace(tie_word_embeddings=False),
    )
    transfer = DirectReshardHCCLWeightTransfer(model)
    plan = object()
    build_calls = []

    def build_plan(client: Any, state_dict: Any) -> Any:
        """Build the fixture transfer plan for the publication test."""
        build_calls.append((client, state_dict))
        transfer._parameter_names = frozenset(state_dict)  # pylint: disable=protected-access
        return plan

    monkeypatch.setattr(transfer, "_build_plan", build_plan)
    client = object()
    state_dict = {"model.norm.weight": object()}

    assert transfer._ensure_plan(client, state_dict) is plan  # pylint: disable=protected-access
    assert transfer._ensure_plan(client, state_dict) is plan  # pylint: disable=protected-access
    assert build_calls == [(client, state_dict)]


def test_disjoint_dp2_tp2_layout_query_accepts_one_returned_dp_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One coordinator query plans from the complete TP engine vLLM returns."""
    calls = []

    def worker(dp_rank: int, tp_rank: int) -> dict[str, object]:
        """Provide the worker fixture for this regression."""
        return {
            "dp_rank": dp_rank,
            "dp_size": 1,
            "tp_rank": tp_rank,
            "tp_size": 2,
            "physical_device_id": f"host-{dp_rank}-{tp_rank}",
            "tensors": [
                {
                    "name": "weight",
                    "dtype_name": "bfloat16",
                    "element_size": 2,
                    "local_shape": [4, 2],
                    "placement": "shard",
                    "shard_dim": 1,
                    "tp_fragment": tp_rank,
                }
            ],
        }

    workers = [worker(0, tp_rank) for tp_rank in range(2)]

    class Client:
        """Expose one shared endpoint topology query."""

        @staticmethod
        def get_world_size() -> int:
            """Provide the get world size fixture for this regression."""
            calls.append("world_size")
            return 4

        @staticmethod
        def collective_rpc(method: str) -> Any:
            """Provide the fixture response for a collective RPC."""
            calls.append(method)
            return workers

    monkeypatch.setattr(
        transfer_module,
        "coordinator_call",
        lambda _operation, callback: callback(),
    )
    transfer = DirectReshardHCCLWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    representatives = transfer._query_destination_workers(  # pylint: disable=protected-access
        Client()
    )

    assert [(worker["dp_rank"], worker["tp_rank"]) for worker in representatives] == [
        (0, 0),
        (0, 1),
    ]
    assert calls == ["world_size", "get_direct_reshard_layout"]


def test_disjoint_layout_query_rejects_incomplete_returned_tp_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible DP engine must return every configured TP worker layout."""
    client = SimpleNamespace(
        get_world_size=lambda: 4,
        collective_rpc=lambda _method: [
            {
                "dp_rank": 0,
                "dp_size": 2,
                "tp_rank": 0,
                "tp_size": 2,
                "tensors": [],
            }
        ],
    )
    monkeypatch.setattr(
        transfer_module,
        "coordinator_call",
        lambda _operation, callback: callback(),
    )
    transfer = DirectReshardHCCLWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    with pytest.raises(RuntimeError, match="incomplete TP engine"):
        transfer._query_destination_workers(client)  # pylint: disable=protected-access


def test_disjoint_layout_query_rejects_same_tp_replica_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A differing DP replica cannot be hidden by TP representative collapse."""
    workers = [
        {
            "dp_rank": dp_rank,
            "dp_size": 2,
            "tp_rank": tp_rank,
            "tp_size": 2,
            "tensors": [{"name": "weight", "local_shape": [4, 2]}],
        }
        for dp_rank in range(2)
        for tp_rank in range(2)
    ]
    workers[2]["tensors"] = [{"name": "weight", "local_shape": [4, 3]}]
    client = SimpleNamespace(
        get_world_size=lambda: 4,
        collective_rpc=lambda _method: workers,
    )
    monkeypatch.setattr(
        transfer_module,
        "coordinator_call",
        lambda _operation, callback: callback(),
    )
    transfer = DirectReshardHCCLWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    with pytest.raises(RuntimeError, match="same-TP DP replicas"):
        transfer._query_destination_workers(client)  # pylint: disable=protected-access


def test_disjoint_direct_control_and_identity_calls_are_coordinator_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only coordinator callbacks mutate or query the shared external endpoint."""
    coordinator_operations = []
    client_calls = []
    plan = SimpleNamespace()

    class Client(VLLMWeightSyncClientMixin):
        """Record shared endpoint control calls."""

        def pause(self) -> None:
            """Provide the pause fixture for this regression."""
            client_calls.append("pause")

        def start_weight_update(self) -> None:
            """Provide the start weight update fixture for this regression."""
            client_calls.append("start")

        def finish_weight_update(self) -> None:
            """Provide the finish weight update fixture for this regression."""
            client_calls.append("finish")

        def verify_policy_weight_identity(self, expected_version: Any, expected_fingerprint: Any) -> None:
            """Provide the verify policy weight identity fixture for this regression."""
            client_calls.append(("verify", expected_version, expected_fingerprint))

        def verify_direct_content_identity(
            self,
            expected_version: int,
            expected: Mapping[int, Mapping[str, Any]],
        ) -> None:
            """Record source-derived identity verification."""
            client_calls.append(("verify_content", expected_version, expected))

    def coordinator_call(operation: Any, callback: Any) -> Any:
        """Execute the coordinator callback in the single-process fixture."""
        coordinator_operations.append(operation)
        return callback()

    fake_platform = SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(transfer_module, "coordinator_call", coordinator_call)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda _operation, callback: callback(),
    )
    transfer = DirectReshardHCCLWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda _payload: {"weight": object()})
    monkeypatch.setattr(transfer, "_ensure_plan", lambda _client, _state_dict: plan)
    monkeypatch.setattr(transfer._transport, "transfer", lambda *_args: None)  # pylint: disable=protected-access
    fingerprint = {"algorithm": "sha256", "digest": "policy-v1"}
    content = {0: {"algorithm": "sha256-canonical-fragments-v1", "digest": "content-v1"}}
    monkeypatch.setattr(transfer, "_distributed_policy_fingerprint", lambda _state_dict: fingerprint)
    monkeypatch.setattr(
        transfer,
        "_distributed_source_content_identities",
        lambda _state_dict, _plan: content,
    )

    transfer.publish(
        Client(),
        PolicySnapshot(version=1, model_name="qwen3", payload=object()),
    )

    assert client_calls == [
        "pause",
        "start",
        "finish",
        ("verify_content", 1, content),
        ("verify", 1, fingerprint),
    ]
    assert coordinator_operations == [
        "direct reshard pause",
        "direct reshard start",
        "direct reshard finish",
        "direct reshard source content verification",
        "direct reshard rollout parameter manifest",
        "direct reshard policy fingerprint",
    ]


def test_failed_direct_buffers_are_owned_only_until_transfer_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception tracebacks must not retain IPC producers after server shutdown."""
    fake_platform = SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda operation, callback: callback(),
    )
    transfer = ColocatedDirectReshardWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
        data_parallel_size=2,
    )

    class DirectClient(VLLMWeightSyncClientMixin):
        """Provide the control methods reached before the injected failure."""

        def wake_up(self, tags: Any) -> None:
            """Provide the wake up fixture for this regression."""
            del tags

        def pause(self) -> None:
            """Provide the pause fixture for this regression."""
            return None

        def start_weight_update(self) -> None:
            """Provide the start weight update fixture for this regression."""
            return None

    client = DirectClient()
    class Producer:
        """Weak-referenceable stand-in for one exported NPU buffer."""

    producer = Producer()
    producer_reference = weakref.ref(producer)

    def fail_stream(*_args: Any) -> None:
        """Retain the exported producer exactly as a failed IPC send would."""
        transfer._failed_buffers.append(producer)  # pylint: disable=protected-access
        raise RuntimeError("planned post-export failure")

    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda payload: {"weight": object()})
    monkeypatch.setattr(
        transfer,
        "_ensure_plan",
        lambda client, state_dict: SimpleNamespace(destination_tp_size=2),
    )
    monkeypatch.setattr(
        transfer,
        "_distributed_source_content_identities",
        lambda _state_dict, _plan: {0: {"digest": "content-v1"}},
    )
    monkeypatch.setattr(
        transfer,
        "_stream_redistribute_and_send",
        fail_stream,
    )
    monkeypatch.setattr(transfer, "_gather_endpoints", lambda client: ("http://server",))

    with pytest.raises(RuntimeError, match="planned post-export failure") as error:
        transfer.publish(
            client,
            PolicySnapshot(version=1, model_name="qwen3", payload=object()),
        )

    del producer
    transfer.close()
    gc.collect()
    assert error.traceback is not None
    assert producer_reference() is None


@pytest.mark.parametrize("fail_after_ack", [False, True])
def test_colocated_direct_releases_each_ipc_bucket_after_ack(
    monkeypatch: pytest.MonkeyPatch,
    fail_after_ack: bool,
) -> None:
    """Direct IPC sends one bucket per RPC without retaining earlier producers."""
    stream = SimpleNamespace(synchronize=lambda: None)

    def all_gather_object(output: Any, value: Any) -> None:
        """Provide the all gather object fixture for this regression."""
        output[0] = value

    fake_platform = SimpleNamespace(
        get_device_handle=lambda _device: SimpleNamespace(current_device=lambda: 0),
        device_type=lambda: "cpu",
        get_rank=lambda: 0,
        get_world_size=lambda: 1,
        get_current_stream=lambda: stream,
        broadcast=lambda _tensor, src: None,
        all_gather_object=all_gather_object,
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    transfer = ColocatedDirectReshardWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
    )
    worker = SimpleNamespace(
        dp_rank=0,
        tp_rank=0,
        physical_device_id="uuid-0",
    )
    monkeypatch.setattr(
        transfer,
        "_resolve_ipc_topology",
        lambda *_args: ("uuid-0", worker, (worker,)),
    )

    class Producer:
        """Weak-referenceable stand-in for one bounded packed tensor."""

    producer_references = []

    def pack_bucket(*_args: Any) -> Any:
        """Provide the pack bucket fixture for this regression."""
        producer = Producer()
        producer_references.append(weakref.ref(producer))
        return producer

    monkeypatch.setattr(transfer_module, "pack_direct_bucket", pack_bucket)
    monkeypatch.setattr(
        transfer_module,
        "_tensor_ipc_rebuild_args",
        lambda _tensor: ("ipc-handle",),
    )
    sent_bucket_indexes = []

    def send_payload(_client: Any, _endpoints: Any, payload: Any, _version: Any, _tp_size: Any) -> None:
        """Provide the send payload fixture for this regression."""
        buckets = payload["buckets_by_target"][0]
        assert len(buckets) == 1
        sent_bucket_indexes.append(buckets[0]["bucket_index"])
        if fail_after_ack:
            raise RuntimeError("injected post-ACK failure")

    monkeypatch.setattr(transfer, "_send_payload", send_payload)
    plan = DirectReshardPlan(
        source_world_size=1,
        destination_tp_size=1,
        bucket_size_bytes=8,
        buckets={(0, 0): (TransferBucket((), 4), TransferBucket((), 8))},
    )

    if fail_after_ack:
        with pytest.raises(RuntimeError, match="post-ACK failure") as error:
            transfer._stream_redistribute_and_send(SimpleNamespace(), ("http://server",), {}, plan, 1)
        gc.collect()
        assert producer_references[0]() is not None
        transfer.release_failed_buffers()
        assert error.traceback is not None
    else:
        transfer._stream_redistribute_and_send(SimpleNamespace(), ("http://server",), {}, plan, 1)
    gc.collect()

    assert sent_bucket_indexes == ([0] if fail_after_ack else [0, 1])
    assert all(reference() is None for reference in producer_references)
    assert not transfer._failed_buffers  # pylint: disable=protected-access


def test_internal_dp1_direct_publish_verifies_every_worker_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared DP1 uses worker-local identity verification rather than rank-local aggregation."""
    expected_fingerprint = {"algorithm": "sha256", "digest": "policy-v1"}
    calls = []
    fake_platform = SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda _operation, callback: callback(),
    )
    transfer = ColocatedDirectReshardWeightTransfer(
        SimpleNamespace(family="qwen3", model=SimpleNamespace(tie_word_embeddings=False)),
        data_parallel_size=1,
    )

    class DirectClient(VLLMWeightSyncClientMixin):
        """Expose the shared control and identity methods used by publication."""

        is_server_owner = True

        def wake_up(self, tags: Any) -> None:
            """Provide the wake up fixture for this regression."""
            calls.append(("wake", tags))

        def pause(self) -> None:
            """Provide the pause fixture for this regression."""
            calls.append(("pause", None))

        def start_weight_update(self) -> None:
            """Provide the start weight update fixture for this regression."""
            calls.append(("start", None))

        def finish_weight_update(self) -> None:
            """Provide the finish weight update fixture for this regression."""
            calls.append(("finish", None))

        def verify_policy_weight_identity(self, version: Any, expected: Any) -> None:
            """Provide the verify policy weight identity fixture for this regression."""
            calls.append(("verify_identity", (version, expected)))

        def verify_direct_content_identity(
            self,
            version: int,
            expected: Mapping[int, Mapping[str, Any]],
        ) -> None:
            """Record source-derived identity verification."""
            calls.append(("verify_content", (version, expected)))

        def get_policy_weight_fingerprints(self) -> Any:
            """Provide the get policy weight fingerprints fixture for this regression."""
            pytest.fail("Shared DP1 used rank-local fingerprint aggregation")

    client = DirectClient()
    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda _payload: {"weight": object()})
    monkeypatch.setattr(
        transfer,
        "_ensure_plan",
        lambda _client, _state_dict: SimpleNamespace(destination_tp_size=2),
    )
    monkeypatch.setattr(
        transfer,
        "_stream_redistribute_and_send",
        lambda *_args: None,
    )
    monkeypatch.setattr(transfer, "_gather_endpoints", lambda _client: ("http://server",))
    monkeypatch.setattr(
        transfer,
        "_distributed_policy_fingerprint",
        lambda _state_dict: expected_fingerprint,
    )
    expected_content = {0: {"digest": "content-v1"}}
    monkeypatch.setattr(
        transfer,
        "_distributed_source_content_identities",
        lambda _state_dict, _plan: expected_content,
    )

    transfer.publish(
        client,
        PolicySnapshot(version=1, model_name="qwen3", payload=object()),
    )

    assert calls == [
        ("wake", ("weights",)),
        ("pause", None),
        ("start", None),
        ("finish", None),
        ("verify_content", (1, expected_content)),
        ("verify_identity", (1, expected_fingerprint)),
    ]


@pytest.mark.parametrize("returned_dp_ranks", [(0,), (0, 1)])
def test_colocated_direct_send_accepts_complete_dp2_tp2_worker_acks(
    monkeypatch: pytest.MonkeyPatch,
    returned_dp_ranks: tuple[int, ...],
) -> None:
    """A shared endpoint may expose one or all complete DP engines after fan-out."""
    workers = [
        {
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "physical_device_id": f"npu-{dp_rank}-{tp_rank}",
        }
        for dp_rank in range(2)
        for tp_rank in range(2)
    ]
    requests = []

    class DirectClient(VLLMWeightSyncClientMixin):
        """Return all physical worker ACKs from one shared endpoint."""

        def collective_rpc(
            self,
            method: str,
            kwargs: Optional[Mapping[str, Any]] = None,
            base_url: Optional[str] = None,
        ) -> list[dict[str, Any]]:
            """Return the DP-engine acknowledgements exposed by the shared endpoint."""
            requests.append((method, kwargs, base_url))
            return [
                {"received": True, **worker}
                for worker in workers
                if worker["dp_rank"] in returned_dp_ranks
            ]

    def synchronize_error(error: Optional[Exception], _operation: str) -> None:
        """Raise the local transfer error as a one-rank synchronized failure."""
        if error is not None:
            raise error

    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module, "synchronize_error", synchronize_error)

    ColocatedDirectReshardWeightTransfer._send_payload(  # pylint: disable=protected-access
        DirectClient(),
        ("http://server",),
        {"worker_topology": workers, "buckets_by_tp": {}},
        policy_version=1,
        destination_tp_size=2,
    )

    assert len(requests) == 1
    assert requests[0][0] == "receive_ipc_direct_reshard"
    assert requests[0][2] == "http://server"


def test_colocated_direct_send_rejects_malformed_worker_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a literal successful ACK may commit a direct IPC transfer."""
    workers = [
        {"dp_rank": 0, "tp_rank": tp_rank, "physical_device_id": f"npu-0-{tp_rank}"}
        for tp_rank in range(2)
    ]
    client = SimpleNamespace(
        collective_rpc=lambda *_args, **_kwargs: [
            {"received": "true", **worker} for worker in workers
        ]
    )

    def synchronize_error(error: Optional[Exception], _operation: str) -> None:
        """Raise the local transfer error as a one-rank synchronized failure."""
        if error is not None:
            raise error

    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module, "synchronize_error", synchronize_error)

    with pytest.raises(RuntimeError, match="returned"):
        ColocatedDirectReshardWeightTransfer._send_payload(  # pylint: disable=protected-access
            client,
            ("http://server",),
            {"worker_topology": workers, "buckets_by_tp": {}},
            policy_version=1,
            destination_tp_size=2,
        )


class _FallbackClient(VLLMWeightSyncClientMixin):
    """Small in-process client used to exercise transaction recovery."""

    def __init__(self, abort_results=None) -> None:
        """Initialize FallbackClient state."""
        self.abort_calls: list[dict[str, int]] = []
        self.abort_results = abort_results

    def collective_rpc(self, method: Any, kwargs: Any=None, base_url: Any=None) -> Any:
        """Record the recovery RPC and report one successfully reset worker."""
        del base_url
        if method == "get_policy_version":
            return [{"version": 3}]
        assert method == "abort_weight_update"
        self.abort_calls.append(dict(kwargs or {}))
        if self.abort_results is not None:
            return self.abort_results
        return [{"aborted": True}]

    @staticmethod
    def is_paused() -> bool:
        """Report fail-closed admission after an aborted publication."""
        return True


class _FailingDirectTransfer:
    """Direct strategy that fails after colocated weights have been restored."""

    weights_awake = True
    last_policy_fingerprint = None

    @staticmethod
    def publish(client: Any, snapshot: Any) -> None:
        """Provide the publish fixture for this regression."""
        del client, snapshot
        raise RuntimeError("planned direct failure")


class _SuccessfulDirectTransfer:
    """Direct strategy stand-in that records one successful publication."""

    last_policy_fingerprint = {"digest": "direct"}

    @staticmethod
    def publish(client: Any, snapshot: Any) -> None:
        """Provide the publish fixture for this regression."""
        del client, snapshot


class _RetainingFailingDirectTransfer(_FailingDirectTransfer):
    """Record whether failed direct IPC buffers are released too early."""

    def __init__(self) -> None:
        """Initialize RetainingFailingDirectTransfer state."""
        self.release_calls = 0

    def release_failed_buffers(self) -> None:
        """Record release after a successful complete-model overwrite."""
        self.release_calls += 1


class _RecordingColocatedFullTransfer(ColocatedFullGatherWeightTransfer):
    """Full strategy stand-in that records fallback residency state."""

    def __init__(self) -> None:
        """Initialize RecordingColocatedFullTransfer state."""
        self.calls = []
        self.last_policy_fingerprint = {"digest": "full"}

    def publish(
        self, client: Any, snapshot: Any, *,
        weights_already_awake: bool = False, manifest_strategy: str = "full_gather",
    ) -> None:
        """Provide the publish fixture for this regression."""
        self.calls.append(
            (client, snapshot, weights_already_awake, manifest_strategy)
        )


class _FailingFullTransfer:
    """Fallback strategy that fails before publication can be acknowledged."""

    last_policy_fingerprint = None

    @staticmethod
    def publish(client: Any, snapshot: Any, **_kwargs: Any) -> None:
        """Provide the publish fixture for this regression."""
        del client, snapshot
        raise RuntimeError("planned fallback failure")


def test_direct_failure_aborts_transaction_then_uses_full_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial direct update is reset before the same version is retried."""
    fake_platform = SimpleNamespace(
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda operation, callback: callback(),
    )
    client = _FallbackClient()
    fallback = _RecordingColocatedFullTransfer()
    transfer = FallbackWeightTransfer(_FailingDirectTransfer(), fallback)
    snapshot = PolicySnapshot(version=4, model_name="qwen3", payload=object())

    transfer.transfer(client, snapshot)

    assert client.abort_calls == [{"restore_policy_version": 3}]
    assert fallback.calls == [(client, snapshot, True, "full_gather_fallback")]
    assert transfer.last_strategy == "full_gather"
    assert transfer.fallback_count == 1
    assert transfer.direct_success_count == 0
    assert transfer.last_policy_fingerprint == {"digest": "full"}
    assert transfer.last_attempted_strategies == (
        "direct_reshard",
        "full_gather",
    )
    assert transfer.last_completed_strategy == "full_gather"
    assert "planned direct failure" in transfer.last_fallback_reason


def test_qwen3_moe_factory_preserves_abort_before_full_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MoE factory binds the common direct/full transaction semantics."""
    fake_platform = SimpleNamespace(
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda operation, callback: callback(),
    )
    transfer = build_weight_transfer(
        "colocated",
        _qwen3_moe_registration(),
        strategy="direct_reshard",
        fallback_strategy="full_gather",
    )
    assert isinstance(transfer, FallbackWeightTransfer)
    assert isinstance(
        transfer._primary,  # pylint: disable=protected-access
        ColocatedDirectReshardWeightTransfer,
    )
    assert isinstance(
        transfer._fallback,  # pylint: disable=protected-access
        ColocatedFullGatherWeightTransfer,
    )
    fallback = _RecordingColocatedFullTransfer()
    transfer._primary = _FailingDirectTransfer()  # pylint: disable=protected-access
    transfer._fallback = fallback  # pylint: disable=protected-access
    client = _FallbackClient()
    snapshot = PolicySnapshot(
        version=4,
        model_name="qwen3_moe",
        payload=object(),
    )

    transfer.transfer(client, snapshot)

    assert client.abort_calls == [{"restore_policy_version": 3}]
    assert fallback.calls == [(client, snapshot, True, "full_gather_fallback")]
    assert transfer.last_completed_strategy == "full_gather"
    assert transfer.last_attempted_strategies == (
        "direct_reshard",
        "full_gather",
    )


def test_successful_direct_publication_updates_strategy_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal direct publication reports direct success and zero fallbacks."""
    fake_platform = SimpleNamespace(
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    client = _FallbackClient()
    transfer = FallbackWeightTransfer(
        _SuccessfulDirectTransfer(),
        _RecordingColocatedFullTransfer(),
    )

    transfer.transfer(
        client,
        PolicySnapshot(version=4, model_name="qwen3", payload=object()),
    )

    assert transfer.configured_strategy == "direct_reshard"
    assert transfer.last_strategy == "direct_reshard"
    assert transfer.direct_success_count == 1
    assert transfer.fallback_count == 0
    assert transfer.last_attempted_strategies == ("direct_reshard",)
    assert transfer.last_completed_strategy == "direct_reshard"
    assert transfer.last_fallback_reason is None


def test_rejected_abort_skips_fallback_and_preserves_direct_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid worker abort ACK fails together instead of splitting ranks."""
    fake_platform = SimpleNamespace(
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda operation, callback: callback(),
    )
    client = _FallbackClient(abort_results=[])
    fallback = _RecordingColocatedFullTransfer()
    transfer = FallbackWeightTransfer(_FailingDirectTransfer(), fallback)

    with pytest.raises(RuntimeError) as error:
        transfer.transfer(
            client,
            PolicySnapshot(version=4, model_name="qwen3", payload=object()),
        )

    assert "planned direct failure" in str(error.value)
    assert "rejected direct-update abort" in str(error.value)
    assert fallback.calls == []


def test_direct_and_fallback_failure_reaborts_and_retains_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fallback restores V and retains direct producers for shutdown."""
    fake_platform = SimpleNamespace(
        get_world_size=lambda: 1,
        get_rank=lambda: 0,
        all_gather_object=lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    monkeypatch.setattr(
        transfer_module,
        "synchronized_call",
        lambda operation, callback: callback(),
    )
    client = _FallbackClient()
    primary = _RetainingFailingDirectTransfer()
    transfer = FallbackWeightTransfer(primary, _FailingFullTransfer())

    with pytest.raises(RuntimeError) as error:
        transfer.transfer(
            client,
            PolicySnapshot(version=4, model_name="qwen3", payload=object()),
        )

    assert "planned direct failure" in str(error.value)
    assert "planned fallback failure" in str(error.value)
    assert client.abort_calls == [
        {"restore_policy_version": 3},
        {"restore_policy_version": 3},
    ]
    assert primary.release_calls == 0
    assert transfer.last_completed_strategy is None
    assert transfer.last_attempted_strategies == (
        "direct_reshard",
        "full_gather",
    )


def test_p7_weight_sync_fault_injector_is_explicit_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance faults require an explicit supported environment value."""
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    injector = transfer_module._WeightSyncTestFaultInjector()  # pylint: disable=protected-access

    injector.trigger("direct_receive_once")
    monkeypatch.setenv(
        "HYPER_RL_TEST_WEIGHT_SYNC_FAULT",
        "direct_receive_once,direct_bucket_once,streaming_bucket_once",
    )
    with pytest.raises(RuntimeError, match="direct_receive_once"):
        injector.trigger("direct_receive_once")
    injector.trigger("direct_receive_once")
    with pytest.raises(RuntimeError, match="direct_bucket_once acknowledged_bytes=128"):
        injector.trigger("direct_bucket_once", context="acknowledged_bytes=128")
    injector.trigger("direct_bucket_once")
    with pytest.raises(RuntimeError, match="streaming_bucket_once"):
        injector.trigger("streaming_bucket_once")

    monkeypatch.setenv("HYPER_RL_TEST_WEIGHT_SYNC_FAULT", "unknown")
    with pytest.raises(ValueError, match="unsupported faults"):
        injector.trigger("unknown")


def test_abort_weight_update_restores_previous_worker_identity() -> None:
    """Worker recovery clears active and pending state without keeping a bad version."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=False,
        _hyper_pending_policy_version=4,
        _hyper_loaded_policy_version=4,
    )

    result = worker_module.abort_weight_update(worker, restore_policy_version=3)

    assert result == {
        "aborted": True,
        "was_active": True,
        "pending_version": 4,
        "restored_version": 3,
    }
    assert worker._weight_update_active is False
    assert worker._hyper_pending_policy_version is None
    assert worker._hyper_loaded_policy_version == 3


def test_finish_weight_update_rejects_worker_without_pending_version() -> None:
    """A worker that received no versioned bucket cannot commit the transaction."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=None,
        _hyper_loaded_policy_version=3,
        _check_weight_transfer_engine=lambda: None,
    )

    with pytest.raises(RuntimeError, match="requires received weights"):
        worker_module._finish_custom_weight_update(worker)  # pylint: disable=protected-access

    assert worker._weight_update_active is True
    assert worker._hyper_loaded_policy_version == 3


def test_finish_weight_update_commits_pending_worker_version() -> None:
    """A complete versioned receive advances identity exactly once."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=4,
        _hyper_loaded_policy_version=3,
        _check_weight_transfer_engine=lambda: None,
    )

    worker_module._finish_custom_weight_update(worker)  # pylint: disable=protected-access

    assert worker._weight_update_active is False
    assert worker._hyper_pending_policy_version is None
    assert worker._hyper_loaded_policy_version == 4


def test_finish_weight_update_commits_source_derived_content_identity() -> None:
    """A direct worker commits canonical fragment content with the policy version."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _hyper_pending_content_tp_rank=0,
        _check_weight_transfer_engine=lambda: None,
    )
    target = torch.arange(6, dtype=torch.float32).view(2, 3)
    entry = {
        "name": "physical.weight",
        "canonical_name": "canonical.weight",
        "canonical_starts": [0, 0],
        "destination_starts": [0, 0],
        "destination_permutation": [0, 1],
        "lengths": [2, 3],
        "dtype_name": "float32",
    }

    worker_module._record_direct_content_fragment(  # pylint: disable=protected-access
        worker,
        1,
        entry,
        target,
    )
    worker_module._finish_custom_weight_update(worker)  # pylint: disable=protected-access
    expected = {"0": dict(worker._hyper_loaded_content_identity)}

    result = worker_module.verify_direct_content_identity(
        worker,
        expected_version=1,
        expected_by_tp_rank=expected,
    )

    assert result["verified"] is True
    assert result["total_bytes"] == target.numel() * target.element_size()
    changed = {"0": {**expected["0"], "digest": "changed"}}
    with pytest.raises(RuntimeError, match="differs from Trainer source"):
        worker_module.verify_direct_content_identity(
            worker,
            expected_version=1,
            expected_by_tp_rank=changed,
        )


def test_deepseek_direct_commit_refreshes_runtime_moe_and_absorbed_mla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Online updates restore Ascend FusedMoE layout and absorbed MLA leaves."""
    refreshes = []

    def _layer(index: int) -> SimpleNamespace:
        """Build one absorbed MLA wrapper with an observable refresh."""
        return SimpleNamespace(
            mla_attn=SimpleNamespace(
                process_weights_after_loading=lambda dtype: refreshes.append(
                    (index, dtype)
                )
            )
        )

    moe = SimpleNamespace(
        w13_weight=torch.nn.Parameter(torch.empty(2, 4, 3)),
        w2_weight=torch.nn.Parameter(torch.empty(2, 3, 2)),
    )

    def process_moe(module: SimpleNamespace) -> None:
        """Model the fixed Ascend unquantized FusedMoE post-load transpose."""
        module.w13_weight = torch.nn.Parameter(
            module.w13_weight.data.transpose(1, 2).contiguous()
        )
        module.w2_weight = torch.nn.Parameter(
            module.w2_weight.data.transpose(1, 2).contiguous()
        )

    moe.quant_method = SimpleNamespace(process_weights_after_loading=process_moe)
    modules = (_layer(0), _layer(1), moe)
    model = SimpleNamespace(modules=lambda: modules)
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV3ForCausalLM"],
                num_hidden_layers=2,
                n_routed_experts=2,
                moe_intermediate_size=2,
                hidden_size=3,
            ),
        ),
        model_runner=SimpleNamespace(get_model=lambda: model),
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _hyper_pending_content_tp_rank=0,
        _hyper_pending_content_fragments={
            1: {
                "fragment": {
                    "sha256": "digest",
                    "num_bytes": 4,
                }
            }
        },
        _check_weight_transfer_engine=lambda: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {"dp_size": 1, "tp_size": 1},
    )

    worker_module._finish_custom_weight_update(worker)  # pylint: disable=protected-access

    assert refreshes == [(0, torch.bfloat16), (1, torch.bfloat16)]
    assert tuple(moe.w13_weight.shape) == (2, 3, 4)
    assert tuple(moe.w2_weight.shape) == (2, 2, 3)
    assert worker._hyper_loaded_moe_refresh_count == 1
    assert worker._hyper_loaded_mla_refresh_count == 2
    assert worker._hyper_loaded_policy_version == 1


def test_qwen3_moe_direct_commit_restores_common_physical_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Qwen3-MoE commit restores every common expert leaf before publish."""
    moe = SimpleNamespace(
        hyper_local_expert_leaf=True,
        w13_weight=torch.nn.Parameter(torch.empty(2, 4, 3)),
        w2_weight=torch.nn.Parameter(torch.empty(2, 3, 2)),
    )

    def ensure_physical_weight_layout() -> None:
        """Model the common leaf's canonical-to-physical conversion."""
        moe.w13_weight = torch.nn.Parameter(
            moe.w13_weight.data.transpose(1, 2).contiguous()
        )
        moe.w2_weight = torch.nn.Parameter(
            moe.w2_weight.data.transpose(1, 2).contiguous()
        )

    moe.ensure_physical_weight_layout = ensure_physical_weight_layout
    model = SimpleNamespace(modules=lambda: (moe,))
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                architectures=["HyperQwen3MoeForCausalLM"],
                num_hidden_layers=1,
                num_experts=2,
                moe_intermediate_size=2,
                hidden_size=3,
            ),
        ),
        model_runner=SimpleNamespace(get_model=lambda: model),
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _hyper_pending_content_tp_rank=0,
        _hyper_pending_content_fragments={
            1: {"fragment": {"sha256": "digest", "num_bytes": 4}}
        },
        _check_weight_transfer_engine=lambda: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {"dp_size": 1, "tp_size": 1},
    )

    worker_module._finish_custom_weight_update(worker)  # pylint: disable=protected-access

    assert tuple(moe.w13_weight.shape) == (2, 3, 4)
    assert tuple(moe.w2_weight.shape) == (2, 2, 3)
    assert worker._hyper_loaded_moe_refresh_count == 1
    assert worker._hyper_loaded_mla_refresh_count == 0
    assert worker._hyper_loaded_policy_version == 1


def test_atomic_weight_wake_prepares_layout_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every weight wake restores executable FusedMoE layout before dummy work."""
    class EngineCore:
        """Minimal import-compatible EngineCore stand-in."""

        @staticmethod
        def wake_up(
            _engine_core: Any,
            tags: Optional[list[str]] = None,
        ) -> None:
            """Accept the patched wake signature without touching runtime state."""
            del tags

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.core",
        SimpleNamespace(EngineCore=EngineCore),
    )

    calls = []
    monkeypatch.setattr(
        EngineCore,
        "wake_up",
        lambda _engine_core, tags=None: calls.append(("original_wake", tags)),
    )
    monkeypatch.setattr(worker_module._patch_state, "engine_core_wake", False)  # pylint: disable=protected-access
    worker_module._patch_engine_core_wake_lifecycle()  # pylint: disable=protected-access

    executor = SimpleNamespace(
        wake_up=lambda tags: calls.append(("memory_wake", tags)),
        collective_rpc=lambda method: calls.append(("prepare", method))
        or [{"prepared": True}],
    )
    engine_core = SimpleNamespace(model_executor=executor)

    EngineCore.wake_up(
        engine_core,
        ["weights", "_hyper_keep_scheduler_paused"],
    )

    assert calls == [
        ("memory_wake", ["weights"]),
        ("prepare", "prepare_direct_reshard_layout"),
    ]


def test_build_weight_transfer_keeps_direct_reshard_for_tp1() -> None:
    """TP1 is the replicated destination case of the same direct planner."""
    model = SimpleNamespace(is_hyper=True, family="qwen3")
    native_model = SimpleNamespace(is_hyper=False, family="qwen3")

    assert isinstance(
        build_weight_transfer("disjoint", model),
        FullGatherHCCLWeightTransfer,
    )
    assert isinstance(
        build_weight_transfer("colocated", model),
        ColocatedFullGatherWeightTransfer,
    )
    assert isinstance(
        build_weight_transfer(
            "disjoint", model, tensor_parallel_size=2, strategy="direct_reshard"
        ),
        DirectReshardHCCLWeightTransfer,
    )
    assert isinstance(
        build_weight_transfer(
            "colocated", model, tensor_parallel_size=2, strategy="direct_reshard"
        ),
        ColocatedDirectReshardWeightTransfer,
    )
    assert isinstance(
        build_weight_transfer("colocated", model, strategy="direct_reshard"),
        ColocatedDirectReshardWeightTransfer,
    )
    internal_direct = build_weight_transfer(
        "colocated",
        model,
        tensor_parallel_size=2,
        data_parallel_size=1,
        strategy="direct_reshard",
        fallback_strategy="none",
    )
    assert isinstance(internal_direct, ColocatedDirectReshardWeightTransfer)
    assert internal_direct._data_parallel_size == 1  # pylint: disable=protected-access
    internal_fallback = build_weight_transfer(
        "colocated",
        model,
        tensor_parallel_size=2,
        data_parallel_size=1,
        strategy="direct_reshard",
        fallback_strategy="full_gather",
    )
    assert isinstance(internal_fallback, FallbackWeightTransfer)
    assert internal_fallback._primary._data_parallel_size == 1  # pylint: disable=protected-access
    assert internal_fallback._fallback._data_parallel_size == 1  # pylint: disable=protected-access
    assert isinstance(
        build_weight_transfer(
            "disjoint",
            native_model,
            tensor_parallel_size=2,
            data_parallel_size=2,
            strategy="direct_reshard",
            fallback_strategy="none",
        ),
        DirectReshardHCCLWeightTransfer,
    )
    disjoint_direct = build_weight_transfer(
        "disjoint",
        native_model,
        tensor_parallel_size=2,
        data_parallel_size=2,
        strategy="direct_reshard",
        fallback_strategy="none",
    )
    assert disjoint_direct._data_parallel_size == 2  # pylint: disable=protected-access
    assert disjoint_direct._tensor_parallel_size == 2  # pylint: disable=protected-access
    disjoint_full = build_weight_transfer(
        "disjoint",
        model,
        tensor_parallel_size=2,
        data_parallel_size=2,
        strategy="full_gather",
    )
    assert disjoint_full._data_parallel_size == 2  # pylint: disable=protected-access
    assert disjoint_full._tensor_parallel_size == 2  # pylint: disable=protected-access
    assert isinstance(
        build_weight_transfer("colocated", model, strategy="full_gather"),
        ColocatedFullGatherWeightTransfer,
    )
    deepseek_direct = build_weight_transfer(
        "colocated",
        _deepseek_v3_registration(),
        tensor_parallel_size=1,
        data_parallel_size=4,
        strategy="direct_reshard",
        fallback_strategy="none",
    )
    assert isinstance(deepseek_direct, ColocatedDirectReshardWeightTransfer)
    assert deepseek_direct._data_parallel_size == 4  # pylint: disable=protected-access
    deepseek_fallback = build_weight_transfer(
        "colocated",
        _deepseek_v3_registration(),
        strategy="direct_reshard",
        fallback_strategy="full_gather",
    )
    assert isinstance(deepseek_fallback, FallbackWeightTransfer)
    assert isinstance(
        deepseek_fallback._fallback,  # pylint: disable=protected-access
        ColocatedFullGatherWeightTransfer,
    )
    with pytest.raises(ValueError, match="Unsupported weight-sync strategy"):
        build_weight_transfer("disjoint", model, strategy="full_broadcast")
    with pytest.raises(ValueError, match="Unsupported weight-sync fallback"):
        build_weight_transfer("colocated", model, fallback_strategy="")
    with pytest.raises(ValueError, match="Unsupported rollout deployment"):
        build_weight_transfer("unknown", model)


def test_direct_reshard_config_accepts_hyper_and_native_qwen3() -> None:
    """Both Qwen3 implementations and residencies use direct reshard."""
    native_qwen3 = SimpleNamespace(is_hyper=False, family="qwen3")
    hyper_qwen3 = SimpleNamespace(is_hyper=True, family="qwen3")
    config = {
        "weight_sync": {
            "strategy": "direct_reshard",
            "bucket_size_mb": 64,
        }
    }

    _validate_vllm_weight_sync(config, "colocated", native_qwen3, {})
    _validate_vllm_weight_sync(config, "disjoint", native_qwen3, {})
    _validate_vllm_weight_sync(config, "colocated", hyper_qwen3, {})
    _validate_vllm_weight_sync(config, "disjoint", hyper_qwen3, {})


def test_weight_sync_defaults_are_normalized_once_by_model_and_tp() -> None:
    """Model and topology never override the upstream full/none defaults."""
    tp1 = resolve_weight_sync_config(
        {},
        deployment="colocated",
        model_family="qwen3",
        rollout_tp=1,
    )
    tp2 = resolve_weight_sync_config(
        {},
        deployment="disjoint",
        model_family="qwen3",
        rollout_tp=2,
    )

    assert (tp1.strategy, tp1.fallback_strategy) == ("full_gather", "none")
    assert (tp2.strategy, tp2.fallback_strategy) == (
        "full_gather",
        "none",
    )
    qwen3_moe = resolve_weight_sync_config(
        {},
        deployment="colocated",
        model_family="qwen3_moe",
        rollout_tp=1,
    )
    assert (qwen3_moe.strategy, qwen3_moe.fallback_strategy) == (
        "full_gather",
        "none",
    )
    direct = resolve_weight_sync_config(
        {"strategy": "direct_reshard"}, deployment="colocated", model_family="deepseek_v3", rollout_tp=2,
    )
    assert (direct.strategy, direct.fallback_strategy) == ("direct_reshard", "none")


def test_qwen3_moe_uses_bounded_colocated_direct_and_full_gather() -> None:
    """Hyper Qwen3-MoE shares the EP1 bounded publication implementations."""
    model = _qwen3_moe_registration()
    config = {
        "tensor_parallel_size": 1,
        "weight_sync": {
            "strategy": "direct_reshard",
            "fallback_strategy": "full_gather",
            "bucket_size_mb": 16,
        },
    }

    _validate_vllm_weight_sync(config, "colocated", model, {"tp": 1})
    fallback = build_weight_transfer(
        "colocated",
        model,
        bucket_size_bytes=16 * 2**20,
        strategy="direct_reshard",
        fallback_strategy="full_gather",
    )
    full = build_weight_transfer(
        "colocated",
        model,
        bucket_size_bytes=16 * 2**20,
        strategy="full_gather",
        fallback_strategy="none",
    )

    assert isinstance(fallback, FallbackWeightTransfer)
    assert isinstance(fallback._primary, ColocatedDirectReshardWeightTransfer)  # pylint: disable=protected-access
    assert isinstance(fallback._fallback, ColocatedFullGatherWeightTransfer)  # pylint: disable=protected-access
    assert isinstance(full, ColocatedFullGatherWeightTransfer)


@pytest.mark.parametrize("strategy", ["direct_reshard", "full_gather"])
def test_native_qwen3_moe_weight_sync_uses_shared_transfers(strategy: str) -> None:
    """P8.6 maps Native storage through the existing colocated transfers."""
    model = _qwen3_moe_registration(is_hyper=False)
    config = {
        "tensor_parallel_size": 1,
        "weight_sync": {
            "strategy": strategy,
            "fallback_strategy": "none",
        },
    }

    _validate_vllm_weight_sync(config, "colocated", model, {"tp": 1})
    transfer = build_weight_transfer(
        "colocated", model, strategy=strategy, fallback_strategy="none",
    )
    expected = (
        ColocatedDirectReshardWeightTransfer if strategy == "direct_reshard"
        else ColocatedFullGatherWeightTransfer
    )
    assert isinstance(transfer, expected)


def test_full_gather_rejects_a_redundant_fallback() -> None:
    """A complete full-gather publication cannot fall back to itself."""
    with pytest.raises(ValueError, match="requires fallback_strategy='none'"):
        resolve_weight_sync_config(
            {
                "strategy": "full_gather",
                "fallback_strategy": "full_gather",
            },
            deployment="colocated",
            model_family="qwen3",
            rollout_tp=1,
        )


def test_deepseek_v3_config_supports_tp2_but_requires_colocated() -> None:
    """Moonlight TP1/TP2 use the shared colocated synchronization runtime."""
    deepseek = SimpleNamespace(is_hyper=False, family="deepseek_v3")
    direct = {
        "tensor_parallel_size": 1,
        "weight_sync": {
            "strategy": "direct_reshard",
            "fallback_strategy": "none",
            "bucket_size_mb": 128,
        },
    }

    _validate_vllm_weight_sync(direct, "colocated", deepseek, {})
    with pytest.raises(ValueError, match="disjoint weight synchronization is not implemented"):
        _validate_vllm_weight_sync(direct, "disjoint", deepseek, {})
    with pytest.raises(ValueError, match="disjoint weight synchronization is not implemented"):
        build_weight_transfer(
            "disjoint",
            _deepseek_v3_registration(),
            strategy="direct_reshard",
            fallback_strategy="none",
        )
    _validate_vllm_weight_sync({**direct, "tensor_parallel_size": 2}, "colocated", deepseek, {})
    with pytest.raises(ValueError, match="supports rollout TP1"):
        _validate_vllm_weight_sync(
            {**direct, "tensor_parallel_size": 3},
            "colocated",
            deepseek,
            {},
        )
    fallback = {
        **direct,
        "weight_sync": {
            "strategy": "direct_reshard",
            "fallback_strategy": "full_gather",
        },
    }
    _validate_vllm_weight_sync(fallback, "colocated", deepseek, {})
    full_gather = {
        **direct,
        "weight_sync": {
            "strategy": "full_gather",
            "fallback_strategy": "none",
        },
    }
    _validate_vllm_weight_sync(full_gather, "colocated", deepseek, {})
    with pytest.raises(ValueError, match="disjoint weight synchronization is not implemented"):
        _validate_vllm_weight_sync(
            {
                **direct,
                "weight_sync": {
                    "strategy": "full_gather",
                    "fallback_strategy": "none",
                },
            },
            "disjoint",
            deepseek,
            {},
        )


def test_full_gather_strategy_and_direct_fallback_are_accepted() -> None:
    """Full gather can be selected or retained only as direct recovery."""
    model = SimpleNamespace(is_hyper=True, family="qwen3")

    _validate_vllm_weight_sync(
        {"weight_sync": {"strategy": "full_gather"}},
        "colocated",
        model,
        {},
    )
    _validate_vllm_weight_sync(
        {
            "tensor_parallel_size": 2,
            "weight_sync": {
                "strategy": "direct_reshard",
                "fallback_strategy": "full_gather",
            },
        },
        "disjoint",
        model,
        {},
    )


@pytest.mark.parametrize("strategy", ["unknown", "full_broadcast"])
def test_invalid_weight_sync_strategy_is_rejected(strategy: str) -> None:
    """Unknown and removed transfer names fail during configuration validation."""
    with pytest.raises(ValueError, match="Unsupported weight-sync strategy"):
        _validate_vllm_weight_sync(
            {"weight_sync": {"strategy": strategy}},
            "colocated",
            SimpleNamespace(is_hyper=True, family="qwen3"),
            {},
        )


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_colocated_ep_configuration(family: str, ep_size: int) -> None:
    """The two families accept the same explicit rollout EP topology."""
    _validate_moe_ep1_topology(
        {"enable_expert_parallel": True, "data_parallel_size": ep_size},
        SimpleNamespace(family=family, is_hyper=True),
        {"dp_replicate": 1, "dp_shard": 4, "tp": 1, "cp": 1, "pp": 1},
    )


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
@pytest.mark.parametrize("transfer_type", [ColocatedDirectReshardWeightTransfer, ColocatedFullGatherWeightTransfer])
@pytest.mark.parametrize("is_hyper", [True, False])
def test_ep_publication_owns_only_expert_axis(family: str, transfer_type: type, is_hyper: bool) -> None:
    """Bounded full-gather and direct resolve the same EP-local expert ranges."""
    registration = SimpleNamespace(
        family=family, is_hyper=is_hyper, model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )
    local = SimpleNamespace(hyper_local_expert_leaf=True, local_expert_count=2, global_expert_start=0)
    parameters = {
        "model.layers.0.mlp.experts.w13_weight": torch.empty(2, 4, 6),
        "model.layers.0.mlp.experts.w2_weight": torch.empty(2, 3, 4),
        "model.layers.0.mlp.gate.weight": torch.empty(8, 4),
    }
    model = SimpleNamespace(
        named_parameters=parameters.items, named_buffers=lambda: (),
        named_modules=lambda: (("model.layers.0.mlp.experts", local),),
    )
    if is_hyper:
        tensors = _hyper_moe_direct_tensors(
            model, family=family, num_experts=8, intermediate_size=3, hidden_size=4, tp_size=1,
        )
    else:
        local.expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, -1, -1])
        local.w13_weight = parameters["model.layers.0.mlp.experts.w13_weight"]
        local.w2_weight = parameters["model.layers.0.mlp.experts.w2_weight"]
        tensors = _native_moe_direct_tensors(
            model, SimpleNamespace(n_routed_experts=8, num_experts=8, moe_intermediate_size=3, hidden_size=4),
            1, 4, family=family, ep_rank=2, ep_size=4,
        )
    # vLLM DP collective_rpc can return only one engine's result. Each worker
    # independently checks its actual physical DP/EP coordinate before receipt.
    worker = {"dp_rank": 2, "dp_size": 4, "tp_rank": 0, "tp_size": 1,
              "ep_rank": 2, "ep_size": 4, "tensors": tensors}
    client = SimpleNamespace(
        get_world_size=lambda: 4,
        collective_rpc=lambda method: [{"prepared": True}] if method == "prepare_direct_reshard_layout" else [worker],
    )
    transfer = transfer_type(registration, data_parallel_size=4)
    descriptions = transfer._query_destination_workers(client)
    state = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.empty(8, 6, 4),
        "model.layers.0.mlp.experts.down_proj": torch.empty(8, 4, 3),
        "model.layers.0.mlp.gate.weight": torch.empty(8, 4),
    }
    adapter = build_model_weight_adapter(registration)
    sources = resolve_source_layouts([adapter.direct_source_descriptions(state, 0)])
    destinations = resolve_destination_layouts(descriptions, {s.name: s.global_shape for s in sources})
    for destination in destinations:
        if ".experts." in destination.name:
            assert destination.shard_dim == 0
            assert destination.region.starts[0] == destination.tp_rank * 2
            assert destination.region.lengths[0] == 2
            assert destination.region.lengths[1:] == destination.global_shape[1:]
        else:
            assert destination.placement == "replicate"
            assert destination.region.lengths == destination.global_shape


def test_qwen_ep_combine_keeps_fp32_router_weights() -> None:
    """EP reordering cannot round Qwen3 routing weights before multiplication."""
    pytest.importorskip("vllm")
    from rl.roles.rollout.vllm_moe import qwen3_combine  # pylint: disable=import-outside-toplevel

    output = torch.tensor([[11.0, 0.25], [-5.0, 32.0], [4.0, -3.0], [2.0, 7.0]], dtype=torch.bfloat16)
    weights = torch.tensor([0.201234, 0.798766, 0.701234, 0.298766])
    order = torch.tensor([2, 0, 3, 1])
    actual = qwen3_combine(output[order], weights, torch.tensor([0, 0, 1, 1]), order, (1, 2, 2))
    expected = (output.float() * weights[:, None]).bfloat16().view(1, 2, 2, 2).sum(2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("transfer_type", [DirectReshardHCCLWeightTransfer, FullGatherHCCLWeightTransfer])
def test_disjoint_transport_rejects_ep_destinations(transfer_type: type) -> None:
    """Inheritance from the shared planner must not imply disjoint EP support."""
    registration = SimpleNamespace(
        family="qwen3_moe", is_hyper=True, model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )
    worker = {"dp_rank": 0, "dp_size": 4, "tp_rank": 0, "tp_size": 1,
              "ep_rank": 0, "ep_size": 4, "tensors": []}
    client = SimpleNamespace(get_world_size=lambda: 4, collective_rpc=lambda _method: [worker])
    transfer = transfer_type(registration, data_parallel_size=4)
    with pytest.raises(ValueError, match="requires colocated IPC"):
        transfer._query_destination_workers(client)


@pytest.mark.parametrize("custom", [False, True])
def test_ep_primitive_keeps_default_and_custom_gradient_contract(
    monkeypatch: pytest.MonkeyPatch, custom: bool,
) -> None:
    """The optional combine does not change default weighting or break autograd."""
    pytest.importorskip("vllm")
    from rl.roles.rollout.vllm_moe import qwen3_combine  # pylint: disable=import-outside-toplevel

    class Experts(torch.nn.Module):
        """Simple differentiable leaf isolating dispatch and combine semantics."""

        local_expert_count = 4

        def forward(self, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            """Apply a distinguishable local-expert multiplier."""
            return hidden * (indices[:, None] + 1).to(hidden.dtype)

    group = SimpleNamespace(size=lambda: 1)
    monkeypatch.setattr(ep_utils.dist, "get_rank", lambda **_kwargs: 0)
    monkeypatch.setattr(ep_utils.dist, "all_to_all_single", lambda out, src, **_kwargs: out.copy_(src))
    monkeypatch.setattr(ep_utils, "ep_all_to_all", lambda x, *_args: x)
    module = torch.nn.Module()
    module.experts = Experts()
    hidden = torch.tensor([[[2.0, 7.0], [3.0, -4.0]]], dtype=torch.bfloat16, requires_grad=True)
    weights = torch.tensor([[0.201234, 0.798766], [0.701234, 0.298766]], requires_grad=True)
    indices = torch.tensor([[3, 1], [2, 0]])
    output = ep_utils.ep_routed_forward(
        module, hidden, router_fn=lambda *_args: (indices, weights), ep_group=group,
        combine_fn=qwen3_combine if custom else None,
    )
    expanded = hidden.view(2, 1, 2) * (indices[..., None] + 1).bfloat16()
    expected = (
        (expanded.float() * weights[..., None]).bfloat16()
        if custom else expanded * weights.bfloat16()[..., None]
    ).sum(1).view_as(hidden)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(output.sum(), (hidden, weights), retain_graph=True)
    reference_gradients = torch.autograd.grad(expected.sum(), (hidden, weights))
    for actual, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
@pytest.mark.parametrize("ep_size", [1, 4])
@pytest.mark.parametrize("runtime", [False, True])
def test_native_moe_ownership_axes(family: str, ep_size: int, runtime: bool) -> None:
    """EP-off shards I; EP-on shards E while QKV and router stay replicated."""
    config = SimpleNamespace(
        num_experts=8, n_routed_experts=8, hidden_size=6, moe_intermediate_size=12,
        num_attention_heads=3, num_key_value_heads=1, head_dim=2,
    )
    local_e, local_i = (2, 12) if ep_size == 4 else (8, 3)
    w13 = torch.empty(local_e, 2 * local_i, 6)
    w2 = torch.empty(local_e, 6, local_i)
    if runtime:
        w13, w2 = w13.transpose(1, 2), w2.transpose(1, 2)
    expert_map = torch.tensor([-1, -1, -1, -1, 0, 1, -1, -1]) if ep_size == 4 else None
    leaf = SimpleNamespace(w13_weight=w13, w2_weight=w2, expert_map=expert_map)
    parameters = {
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
        "model.layers.0.mlp.gate.weight": torch.empty(8, 6),
    }
    if family == "qwen3_moe":
        parameters["model.layers.0.self_attn.qkv_proj.weight"] = torch.empty(10, 6)
    model = SimpleNamespace(
        named_parameters=parameters.items,
        named_modules=lambda: [("model.layers.0.mlp.experts", leaf)],
    )
    descriptions = _native_moe_direct_tensors(
        model, config, 1, 4, family=family, ep_size=ep_size, ep_rank=2 if ep_size == 4 else 0,
    )
    config.architectures = ["Qwen3MoeForCausalLM" if family == "qwen3_moe" else "DeepseekV3ForCausalLM"]
    manifest = _native_moe_ownership_manifest(
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config), model_runner=SimpleNamespace(get_model=lambda: model),
        ),
        {"ep_size": ep_size, "ep_rank": 2 if ep_size == 4 else 0},
    )["model.layers.0.mlp.experts"]
    assert manifest["local_experts"] == local_e
    assert manifest["global_to_local"] == (None if expert_map is None else expert_map.tolist())
    experts = [item for item in descriptions if ".experts." in item["name"]]
    assert len(experts) == 3
    for item in descriptions:
        if ".experts." in item["name"]:
            assert item["placement"] == "shard"
            assert item["shard_dim"] == (0 if ep_size == 4 else (2 if "down_proj" in item["name"] else 1))
        else:
            assert item["placement"] == "replicate"
            assert item["shard_dim"] is None
    if ep_size == 4:
        leaf.expert_map = torch.tensor([0, -1, -1, -1, -1, 1, -1, -1])
        with pytest.raises(ValueError, match="static contiguous"):
            _native_moe_direct_tensors(model, config, 1, 4, family=family, ep_size=4, ep_rank=2)
        leaf.expert_map = None
        with pytest.raises(ValueError, match="actual global-to-local"):
            _native_moe_direct_tensors(model, config, 1, 4, family=family, ep_size=4, ep_rank=2)
    leaf.expert_map = expert_map
    leaf.dynamic_eplb = True
    with pytest.raises(ValueError, match="EPLB disabled"):
        _native_moe_direct_tensors(
            model, config, 1, 4, family=family, ep_size=ep_size, ep_rank=2 if ep_size == 4 else 0,
        )
    leaf.dynamic_eplb = False
    leaf.w13_weight = w13[:1]
    with pytest.raises(ValueError, match="physical expert count"):
        _native_moe_direct_tensors(
            model, config, 1, 4, family=family, ep_size=ep_size, ep_rank=2 if ep_size == 4 else 0,
        )


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
def test_native_ep_keeps_original_optimized_path_selectable(family: str) -> None:
    """Native may use graph mode; EPLB and rollout TP remain outside P8.6."""
    config = {"enable_expert_parallel": True, "data_parallel_size": 4, "enforce_eager": False}
    model = SimpleNamespace(family=family, is_hyper=False)
    accelerator = {"dp_replicate": 1, "dp_shard": 4, "tp": 1, "cp": 1, "pp": 1}
    _validate_moe_ep1_topology(config, model, accelerator)
    with pytest.raises(ValueError, match="EPLB"):
        _validate_moe_ep1_topology({**config, "enable_eplb": True}, model, accelerator)


def _model_worker(tp_rank: int, is_hyper: bool = False, family: str = "qwen3_moe") -> dict:
    config = SimpleNamespace(
        num_experts=8, moe_intermediate_size=3, hidden_size=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=2, vocab_size=10,
        n_routed_experts=8, n_shared_experts=2, intermediate_size=8,
    )
    prefix = "model.layers.0"
    parameters = {
        f"{prefix}.self_attn.qkv_proj.weight": torch.empty(8, 4, dtype=torch.bfloat16),
        f"{prefix}.self_attn.o_proj.weight": torch.empty(4, 4, dtype=torch.bfloat16),
        f"{prefix}.mlp.gate.weight": torch.empty(8, 4, dtype=torch.bfloat16),
        f"{prefix}.mlp.experts.w13_weight": torch.empty(2, 4, 6, dtype=torch.bfloat16),
        f"{prefix}.mlp.experts.w2_weight": torch.empty(2, 3, 4, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.empty(5, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.empty(5, 4, dtype=torch.bfloat16),
    }
    expert_map = torch.full((8,), -1)
    expert_map[tp_rank * 2:tp_rank * 2 + 2] = torch.arange(2)
    expert = SimpleNamespace(
        expert_map=expert_map, w13_weight=parameters[f"{prefix}.mlp.experts.w13_weight"],
        w2_weight=parameters[f"{prefix}.mlp.experts.w2_weight"],
        hyper_local_expert_leaf=True, local_expert_count=2, global_expert_start=tp_rank * 2,
    )
    if is_hyper:
        fused = parameters.pop(f"{prefix}.self_attn.qkv_proj.weight")
        for projection, value in zip(("q_proj", "k_proj", "v_proj"), fused.split((4, 2, 2))):
            parameters[f"{prefix}.self_attn.{projection}.weight"] = value
    if family == "deepseek_v3":
        for projection in ("qkv_proj", "q_proj", "k_proj", "v_proj"):
            parameters.pop(f"{prefix}.self_attn.{projection}.weight", None)
        for name, shape in {
            "self_attn.q_proj.weight": (4, 4), "self_attn.kv_b_proj.weight": (4, 2),
            "self_attn.kv_a_proj_with_mqa.weight": (3, 4), "self_attn.kv_a_layernorm.weight": (2,),
            "self_attn.o_proj.weight": (4, 2), "mlp.shared_experts.down_proj.weight": (4, 3),
            "mlp.gate.e_score_correction_bias": (8,),
        }.items():
            parameters[f"{prefix}.{name}"] = torch.empty(shape, dtype=torch.bfloat16)
        projections = ("gate_proj", "up_proj") if is_hyper else ("gate_up_proj",)
        for projection in projections:
            parameters[f"{prefix}.mlp.shared_experts.{projection}.weight"] = torch.empty(
                3 if is_hyper else 6, 4, dtype=torch.bfloat16,
            )
    model = SimpleNamespace(
        named_parameters=parameters.items,
        named_buffers=lambda: (),
        named_modules=lambda: ((f"{prefix}.mlp.experts", expert),),
    )
    if is_hyper:
        model._tp_placements = {  # pylint: disable=protected-access
            name: (Replicate() if any(part in name for part in (".mlp.gate.", ".kv_a_"))
                   else Shard(1) if any(part in name for part in (".o_proj.", ".shared_experts.down_proj."))
                   else Shard(0),)
            for name in parameters if ".experts." not in name
        }
    return {
        "dp_rank": 0, "dp_size": 2, "tp_rank": tp_rank, "tp_size": 2,
        "ep_rank": tp_rank, "ep_size": 4,
        "tensors": (
            _hyper_deepseek_v3_direct_tensors(model, config, 2) if family == "deepseek_v3"
            else _hyper_qwen3_moe_direct_tensors(model, config, 2)
        ) if is_hyper else _native_moe_direct_tensors(
            model, config, 2, 2, family=family, ep_size=4, ep_rank=tp_rank, tp_rank=tp_rank,
        ),
    }


def _destinations(transfer_type: type, is_hyper: bool = False, family: str = "qwen3_moe") -> tuple[list[dict], dict]:
    workers = [_model_worker(rank, is_hyper, family) for rank in range(2)]
    client = SimpleNamespace(
        get_world_size=lambda: 4,
        collective_rpc=lambda method: [{"prepared": True}] if method == "prepare_direct_reshard_layout" else workers,
    )
    registration = SimpleNamespace(family=family, is_hyper=is_hyper, actor_weight_name=lambda name: name)
    transfer = transfer_type(registration, data_parallel_size=2, tensor_parallel_size=2)
    descriptions = transfer._query_destination_workers(client)  # pylint: disable=protected-access
    shapes = {}
    for tensor in descriptions[0]["tensors"]:
        shape = list(tensor["local_shape"])
        if tensor["placement"] == "shard":
            shape[tensor["shard_dim"]] *= tensor["shard_group_size"]
        shapes[tensor["name"]] = tuple(shape)
    return descriptions, shapes


@pytest.mark.parametrize("transfer_type", [ColocatedDirectReshardWeightTransfer, ColocatedFullGatherWeightTransfer])
@pytest.mark.parametrize("is_hyper", [False, True])
@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
def test_tp2_ep4_keeps_distinct_shard_domains(transfer_type: type, is_hyper: bool, family: str) -> None:
    descriptions, shapes = _destinations(transfer_type, is_hyper, family)
    layouts = resolve_destination_layouts(descriptions, shapes)
    for layout in layouts:
        physical_rank = layout.tp_rank
        if ".experts." in layout.name:
            assert layout.region.starts[0] == physical_rank * 2
            assert layout.region.lengths[0] == 2
        elif layout.placement == "shard":
            dim = layout.shard_dim
            assert layout.region.starts[dim] == (physical_rank % 2) * layout.region.lengths[dim]
        else:
            assert layout.region.starts == (0,) * len(layout.global_shape)


@pytest.mark.parametrize("is_hyper", [False, True])
def test_qwen_tp2_ep4_configuration(is_hyper: bool) -> None:
    """Only the implemented first Native combination is admitted."""
    rollout = {"tensor_parallel_size": 2, "data_parallel_size": 2, "enable_expert_parallel": True}
    model = SimpleNamespace(family="qwen3_moe", is_hyper=is_hyper)
    accelerator = {"dp_replicate": 1, "dp_shard": 4, "tp": 1, "cp": 1, "pp": 1}
    _validate_moe_ep1_topology(rollout, model, accelerator)
    with pytest.raises(ValueError, match="requires rollout TP1"):
        _validate_moe_ep1_topology({**rollout, "enable_expert_parallel": False}, model, accelerator)


@pytest.mark.parametrize("fault", ["missing_rank", "out_of_range", "wrong_size", "missing_replica"])
def test_mixed_shard_layout_rejects_incomplete_ownership(fault: str) -> None:
    descriptions, shapes = _destinations(ColocatedDirectReshardWeightTransfer)
    descriptions = deepcopy(descriptions)
    tensor = next(t for t in descriptions[0]["tensors"] if t["name"].endswith("q_proj.weight"))
    if fault == "missing_rank":
        tensor.pop("shard_rank")
    elif fault == "out_of_range":
        tensor["shard_rank"] = 2
    elif fault == "wrong_size":
        tensor["shard_group_size"] = 4
    else:
        tensor["shard_rank"] = 1
    with pytest.raises(ValueError):
        resolve_destination_layouts(descriptions, shapes)


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("is_hyper", [False, True])
@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
def test_mixed_direct_and_streaming_values_match_source(changed: bool, is_hyper: bool, family: str) -> None:
    """Four FSDP sources feed TP replicas and EP owners with the same exact values."""
    descriptions, shapes = _destinations(ColocatedDirectReshardWeightTransfer, is_hyper, family)
    destinations = resolve_destination_layouts(descriptions, shapes)
    states = {rank: {} for rank in range(4)}
    global_state = {}
    sources = []
    for name, shape in shapes.items():
        value = (torch.arange(prod(shape)).reshape(shape) + (17 if changed else 0)).bfloat16()
        global_state[name] = value
        start = 0
        for rank, chunk in enumerate(torch.tensor_split(value, 4, dim=0)):
            states[rank][name] = chunk.clone()
            sources.append(SourceTensorLayout(
                name=name, dtype_name="bfloat16", element_size=2, global_shape=shape, source_rank=rank,
                region=TensorRegion((start,) + (0,) * (len(shape) - 1), tuple(chunk.shape)),
            ))
            start += chunk.shape[0]
    direct = build_direct_reshard_plan(sources, destinations, source_world_size=4, bucket_size_bytes=32)
    full = build_streaming_full_gather_plan(sources, destinations, source_world_size=4, bucket_size_bytes=32)
    by_target = {(d.tp_rank, d.name): d for d in destinations}
    for strategy in ("direct", "full"):
        received = {
            key: torch.full(d.region.lengths, float("nan"), dtype=torch.bfloat16)
            for key, d in by_target.items()
        }

        def copy_fragment(target: int, name: str, starts: tuple, lengths: tuple, value: torch.Tensor) -> None:
            """Provide the copy fragment fixture for this regression."""
            destination = by_target[(target, name)]
            local_slices = tuple(
                slice(start - base, start - base + length)
                for start, base, length in zip(starts, destination.region.starts, lengths)
            )
            received[(target, name)][local_slices].copy_(value)

        for target in range(4):
            if strategy == "direct":
                for rank in range(4):
                    for bucket in direct.for_route(rank, target):
                        assert bucket.total_bytes <= 32
                        for entry in bucket.entries:
                            slices = tuple(
                                slice(start, start + length)
                                for start, length in zip(entry.source_starts, entry.lengths)
                            )
                            copy_fragment(
                                target, entry.name, entry.logical_starts, entry.lengths,
                                states[rank][entry.source_key][slices],
                            )
            else:
                for bucket in full.for_target(target):
                    assert bucket.total_bytes <= 32
                    for fragment in bucket.entries:
                        contributions = [
                            item for rank in range(4)
                            for item in extract_streaming_contributions(fragment, rank, states[rank])
                        ]
                        value = assemble_streaming_fragment(fragment, contributions)
                        copy_fragment(target, fragment.name, fragment.canonical_starts, fragment.lengths, value)
        for key, destination in by_target.items():
            slices = tuple(
                slice(start, start + length)
                for start, length in zip(destination.region.starts, destination.region.lengths)
            )
            torch.testing.assert_close(received[key], global_state[destination.name][slices], rtol=0, atol=0)


def test_direct_reshard_config_requires_pure_fsdp_source_layout() -> None:
    """HSDP cannot be interpreted as one dense FSDP shard axis."""
    config = {"weight_sync": {"strategy": "direct_reshard"}}

    with pytest.raises(ValueError, match="pure FSDP"):
        _validate_vllm_weight_sync(
            config,
            "disjoint",
            SimpleNamespace(is_hyper=True, family="qwen3"),
            {"dp_replicate": 2, "tp": 1, "cp": 1, "pp": 1},
        )


def test_trainer_tp2_accepts_qwen3_deployments_independent_of_rollout_tp() -> None:
    """Normal Qwen3 weight sync allows both deployments and mismatched rollout TP."""
    hyper_qwen3 = SimpleNamespace(is_hyper=True, family="qwen3")
    native_qwen3 = SimpleNamespace(is_hyper=False, family="qwen3")
    full_gather = {
        "tensor_parallel_size": 2,
        "weight_sync": {
            "strategy": "full_gather",
            "fallback_strategy": "none",
        },
    }
    accelerator = {"dp_replicate": 1, "tp": 2, "cp": 1, "pp": 1}

    for deployment in ("colocated", "disjoint"):
        for model in (hyper_qwen3, native_qwen3):
            for rollout_tp in (1, 2):
                _validate_vllm_weight_sync(
                    {**full_gather, "tensor_parallel_size": rollout_tp},
                    deployment,
                    model,
                    accelerator,
                )
            for fallback in ("none", "full_gather"):
                _validate_vllm_weight_sync(
                    {
                        "tensor_parallel_size": 2,
                        "weight_sync": {
                            "strategy": "direct_reshard",
                            "fallback_strategy": fallback,
                        },
                    },
                    deployment,
                    model,
                    accelerator,
                )


def test_direct_reshard_config_rejects_invalid_bucket_size() -> None:
    """A transfer buffer must hold at least one positive number of MiB."""
    config = {
        "weight_sync": {
            "strategy": "direct_reshard",
            "bucket_size_mb": 0,
        }
    }

    with pytest.raises(ValueError, match="bucket_size_mb"):
        _validate_vllm_weight_sync(
            config,
            "disjoint",
            SimpleNamespace(is_hyper=True, family="qwen3"),
            {},
        )

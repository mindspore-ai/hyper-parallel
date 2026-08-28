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
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Optional
import weakref

import pytest
import torch

import rl.roles.weight_sync.transfer as transfer_module
import rl.roles.weight_sync.vllm_worker as worker_module
from rl.config import _validate_vllm_weight_sync
from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    SourceTensorLayout,
    TensorRegion,
    build_direct_reshard_plan,
    describe_source_tensor,
    resolve_destination_layouts,
    resolve_physical_worker_topology,
    resolve_source_layouts,
)
from rl.roles.weight_sync.transfer import (
    ColocatedDirectReshardWeightTransfer,
    ColocatedFullGatherWeightTransfer,
    DirectReshardHCCLWeightTransfer,
    FallbackWeightTransfer,
    FullGatherHCCLWeightTransfer,
    build_weight_transfer,
)
from rl.roles.weight_sync.sync import PolicySnapshot, VLLMWeightSyncClientMixin


def _source(rank: int, starts: tuple[int, int], lengths: tuple[int, int]) -> SourceTensorLayout:
    return SourceTensorLayout(
        name="model.layers.0.self_attn.q_proj.weight",
        dtype_name="bfloat16",
        element_size=2,
        global_shape=(4, 4),
        source_rank=rank,
        region=TensorRegion(starts, lengths),
    )


def test_colocated_ipc_handles_use_torch_reduction_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both colocated strategies export the rebuild arguments produced by Torch."""
    weight = object()
    monkeypatch.setattr(
        transfer_module,
        "reduce_tensor",
        lambda tensor: ("rebuild", ("storage", tensor)),
    )

    handles = ColocatedFullGatherWeightTransfer._build_local_handles(  # pylint: disable=protected-access
        {"weight": weight},
        lambda: "npu-0",
    )

    assert handles == [{"npu-0": ("storage", weight)}]


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
            self.coordinate = coordinate

        @staticmethod
        def size(_mesh_dim: int) -> int:
            return 2

        def get_coordinate(self) -> tuple[int, int]:
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
            self.coordinate = coordinate

        @staticmethod
        def size(_mesh_dim: int) -> int:
            return 2

        def get_coordinate(self) -> tuple[int, int]:
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

    def fake_manifest(worker):
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

    def rebuild_npu_tensor(*_args):
        nonlocal rebuilt_reference
        packed = torch.tensor([0, 0, 128, 63], dtype=torch.uint8)
        rebuilt_reference = weakref.ref(packed)
        return packed

    def synchronize() -> None:
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

    def rebuild_npu_tensor(*_args):
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
            self.parameters = {
                "model.layers.0.self_attn.qkv_proj.weight": torch.empty(
                    8, 8, dtype=torch.int64
                ),
                "model.layers.0.mlp.gate_up_proj.weight": torch.empty(
                    12, 8, dtype=torch.int64
                ),
            }

        def named_parameters(self):
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


def test_full_transfer_requests_complete_device_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recovery strategy explicitly materializes full tensors on every rank."""
    calls = []
    tensor = object()

    def get_model_state_dict(
        payload: object,
        *,
        options: object,
    ) -> dict[str, object]:
        calls.append((payload, options))
        return {"model.norm.weight": tensor}

    fake_platform = SimpleNamespace(
        get_model_state_dict=get_model_state_dict,
        is_tensor=lambda value: value is tensor,
    )
    monkeypatch.setattr(transfer_module, "platform", fake_platform)
    payload = object()

    state_dict = transfer_module._full_state_dict(  # pylint: disable=protected-access
        payload,
        operation="test full gather",
    )

    assert state_dict == {"model.norm.weight": tensor}
    assert len(calls) == 1
    assert calls[0][0] is payload
    assert calls[0][1].full_state_dict is True
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

    def build_plan(client, state_dict):
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
            calls.append("world_size")
            return 4

        @staticmethod
        def collective_rpc(method: str):
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
            client_calls.append("pause")

        def start_weight_update(self) -> None:
            client_calls.append("start")

        def finish_weight_update(self) -> None:
            client_calls.append("finish")

        def verify_policy_weight_identity(self, expected_version, expected_fingerprint) -> None:
            client_calls.append(("verify", expected_version, expected_fingerprint))

    def coordinator_call(operation, callback):
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
    monkeypatch.setattr(transfer, "_distributed_policy_fingerprint", lambda _state_dict: fingerprint)

    transfer.publish(
        Client(),
        PolicySnapshot(version=1, model_name="qwen3", payload=object()),
    )

    assert client_calls == [
        "pause",
        "start",
        "finish",
        ("verify", 1, fingerprint),
    ]
    assert coordinator_operations == [
        "direct reshard pause",
        "direct reshard start",
        "direct reshard finish",
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

        def wake_up(self, tags) -> None:
            del tags

        def pause(self) -> None:
            return None

        def start_weight_update(self) -> None:
            return None

    client = DirectClient()
    class Producer:
        """Weak-referenceable stand-in for one exported NPU buffer."""

    producer = Producer()
    producer_reference = weakref.ref(producer)
    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda payload: {"weight": object()})
    monkeypatch.setattr(
        transfer,
        "_ensure_plan",
        lambda client, state_dict: SimpleNamespace(destination_tp_size=2),
    )
    monkeypatch.setattr(
        transfer,
        "_redistribute_and_export",
        lambda state_dict, plan, data_parallel_size: ({}, [producer]),
    )
    monkeypatch.setattr(transfer, "_gather_endpoints", lambda client: ("http://server",))

    def fail_send(*_args) -> None:
        raise RuntimeError("planned post-export failure")

    monkeypatch.setattr(transfer, "_send_payload", fail_send)

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

        def wake_up(self, tags) -> None:
            calls.append(("wake", tags))

        def pause(self) -> None:
            calls.append(("pause", None))

        def start_weight_update(self) -> None:
            calls.append(("start", None))

        def finish_weight_update(self) -> None:
            calls.append(("finish", None))

        def verify_policy_weight_identity(self, version, expected) -> None:
            calls.append(("verify_identity", (version, expected)))

        def get_policy_weight_fingerprints(self):
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
        "_redistribute_and_export",
        lambda _state_dict, _plan, _data_parallel_size: ({}, []),
    )
    monkeypatch.setattr(transfer, "_gather_endpoints", lambda _client: ("http://server",))
    monkeypatch.setattr(transfer, "_send_payload", lambda *_args: None)
    monkeypatch.setattr(
        transfer,
        "_distributed_policy_fingerprint",
        lambda _state_dict: expected_fingerprint,
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
        self.abort_calls: list[dict[str, int]] = []
        self.abort_results = abort_results

    def collective_rpc(self, method, kwargs=None, base_url=None):
        """Record the recovery RPC and report one successfully reset worker."""
        del base_url
        if method == "get_policy_version":
            return [{"version": 3}]
        assert method == "abort_weight_update"
        self.abort_calls.append(dict(kwargs or {}))
        if self.abort_results is not None:
            return self.abort_results
        return [{"aborted": True}]


class _FailingDirectTransfer:
    """Direct strategy that fails after colocated weights have been restored."""

    weights_awake = True
    last_policy_fingerprint = None

    @staticmethod
    def publish(client, snapshot) -> None:
        del client, snapshot
        raise RuntimeError("planned direct failure")


class _SuccessfulDirectTransfer:
    """Direct strategy stand-in that records one successful publication."""

    last_policy_fingerprint = {"digest": "direct"}

    @staticmethod
    def publish(client, snapshot) -> None:
        del client, snapshot


class _RetainingFailingDirectTransfer(_FailingDirectTransfer):
    """Record whether failed direct IPC buffers are released too early."""

    def __init__(self) -> None:
        self.release_calls = 0

    def release_failed_buffers(self) -> None:
        """Record release after a successful complete-model overwrite."""
        self.release_calls += 1


class _RecordingColocatedFullTransfer(ColocatedFullGatherWeightTransfer):
    """Full strategy stand-in that records fallback residency state."""

    def __init__(self) -> None:
        self.calls = []
        self.last_policy_fingerprint = {"digest": "full"}

    def publish(
        self,
        client,
        snapshot,
        *,
        weights_already_awake: bool = False,
        manifest_strategy: str = "full_gather",
    ) -> None:
        self.calls.append(
            (client, snapshot, weights_already_awake, manifest_strategy)
        )


class _FailingFullTransfer:
    """Fallback strategy that fails before publication can be acknowledged."""

    last_policy_fingerprint = None

    @staticmethod
    def publish(client, snapshot) -> None:
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
    assert "planned direct failure" in transfer.last_primary_error


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


def test_build_weight_transfer_skips_reshard_for_pure_dp() -> None:
    """TP1 replicas receive full weights without constructing a reshard path."""
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
        build_weight_transfer("disjoint", model, tensor_parallel_size=2),
        FallbackWeightTransfer,
    )
    assert isinstance(
        build_weight_transfer("colocated", model, tensor_parallel_size=2),
        FallbackWeightTransfer,
    )
    internal_direct = build_weight_transfer(
        "colocated",
        model,
        tensor_parallel_size=2,
        data_parallel_size=1,
        fallback_strategy="none",
    )
    assert isinstance(internal_direct, ColocatedDirectReshardWeightTransfer)
    assert internal_direct._data_parallel_size == 1  # pylint: disable=protected-access
    internal_fallback = build_weight_transfer(
        "colocated",
        model,
        tensor_parallel_size=2,
        data_parallel_size=1,
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
            fallback_strategy="none",
        ),
        DirectReshardHCCLWeightTransfer,
    )
    disjoint_direct = build_weight_transfer(
        "disjoint",
        native_model,
        tensor_parallel_size=2,
        data_parallel_size=2,
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
    with pytest.raises(ValueError, match="Unsupported weight-sync strategy"):
        build_weight_transfer("disjoint", model, strategy="full_broadcast")
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
    with pytest.raises(ValueError, match="strategy must be"):
        _validate_vllm_weight_sync(
            {"weight_sync": {"strategy": strategy}},
            "colocated",
            SimpleNamespace(is_hyper=True, family="qwen3"),
            {},
        )


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

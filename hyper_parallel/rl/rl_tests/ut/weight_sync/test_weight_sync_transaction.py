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
"""CPU unit tests for successful policy publication transactions."""
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access
# pylint: disable=missing-public-type-hints

from types import SimpleNamespace
import sys
from typing import Any, Optional

import pytest
import torch

import rl.roles.weight_sync.sync as sync_module
import rl.roles.weight_sync.transfer as transfer_module
from rl.roles.model import ModelRegistration, resolve_vllm_model
from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    DirectReshardPlan,
    SourceTensorLayout,
    TensorRegion,
    TransferBucket,
)
from rl.roles.weight_sync.streaming_full_gather import (
    StreamingBucketAck,
    StreamingMaterializedBucket,
    build_streaming_full_gather_plan,
)
from rl.roles.weight_sync.sync import (
    ActorRolloutWeightSync,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
)
from rl.roles.weight_sync.transfer import (
    ColocatedDirectReshardWeightTransfer,
    ColocatedFullGatherWeightTransfer,
    DirectReshardHCCLWeightTransfer,
    FallbackWeightTransfer,
    FullGatherHCCLWeightTransfer,
)


def _rollout_model():
    """Return a tied Hyper Qwen3 registration."""
    model = ModelRegistration(
        "qwen",
        "qwen3",
        "/model",
        "/tokenizer",
        "Qwen3ForCausalLM",
        "qwen3",
        "qwen3",
        True,
    )
    return resolve_vllm_model(model, "hyper")


def _fingerprint(version: int) -> dict[str, Any]:
    """Return one complete worker identity accepted by publication code."""
    return {
        "algorithm": "test",
        "tensor_count": 1,
        "value_count": 2,
        "digest": f"digest-v{version}",
        "tensors": {"model.norm.weight": "tensor-digest"},
        "version": version,
    }


class _Client(VLLMWeightSyncClientMixin):
    """Record shared-endpoint control and verification calls."""

    base_url = "http://rollout"
    is_server_owner = True

    def __init__(self) -> None:
        self.events: list[str] = []
        self.paused = False
        self.sleeping = False
        self.version = 0
        self.fingerprint = _fingerprint(0)

    def pause(self) -> None:
        self.events.append("pause")
        self.paused = True

    def start_weight_update(self) -> None:
        self.events.append("start")

    def finish_weight_update(self) -> None:
        self.events.append("finish")

    def wake_up(self, tags: tuple[str, ...]) -> None:
        self.events.append(f"wake:{','.join(tags)}")
        self.sleeping = False

    def sleep(self, level: int = 1, mode: str = "wait") -> None:
        self.events.append(f"sleep:{level}:{mode}")
        self.sleeping = True

    def resume(self) -> None:
        self.events.append("resume")
        self.paused = False

    def is_paused(self) -> bool:
        return self.paused

    def is_sleeping(self) -> bool:
        return self.sleeping

    def verify_policy_weight_identity(
        self,
        expected_version: int,
        expected_fingerprint: dict[str, Any],
    ) -> None:
        self.events.append(
            f"verify:{expected_version}:{expected_fingerprint['digest']}"
        )
        self.version = expected_version
        self.fingerprint = dict(expected_fingerprint)

    def collective_rpc(
        self,
        method: str,
        kwargs: Any = None,
        base_url: Any = None,
    ) -> list[dict[str, Any]]:
        del kwargs, base_url
        self.events.append(f"rpc:{method}")
        if method == "get_policy_version":
            return [{"version": self.version}]
        if method == "get_policy_weight_fingerprint":
            return [dict(self.fingerprint)]
        if method == "get_weight_sync_memory_stats":
            return [
                {
                    "current_memory_allocated_bytes": 0,
                    "current_memory_reserved_bytes": 0,
                    "current_host_rss_bytes": 0,
                    "max_memory_allocated_bytes": 0,
                    "max_memory_reserved_bytes": 0,
                    "host_max_rss_bytes": 0,
                }
            ]
        if method == "abort_weight_update":
            return [{"aborted": True, "restored_version": self.version}]
        if method == "write_parameter_manifest":
            return [{"written": True}]
        return []


class _FakeTransfer:
    """Publish one fingerprint while leaving transport details to their own tests."""

    def __init__(self, strategy: str) -> None:
        self.configured_strategy = strategy
        self.last_strategy = strategy
        self.last_completed_strategy = strategy
        self.last_attempted_strategies = (strategy,)
        self.last_fallback_reason = None
        self.last_streaming_stats = None
        self.fallback_count = 0
        self.direct_success_count = int(strategy == "direct_reshard")
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None

    def publish(self, client: _Client, snapshot: PolicySnapshot) -> None:
        self.last_policy_fingerprint = _fingerprint(snapshot.version)
        client.verify_policy_weight_identity(
            snapshot.version,
            self.last_policy_fingerprint,
        )


def test_disjoint_direct_publish_executes_current_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct publication plans, transfers, verifies content, and commits identity."""
    transfer = DirectReshardHCCLWeightTransfer(
        _rollout_model(),
        data_parallel_size=1,
        tensor_parallel_size=2,
    )
    client = _Client()
    events: list[str] = []
    state = {"model.norm.weight": torch.tensor([1.0, 2.0])}
    plan = object()
    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda _payload: dict(state))
    monkeypatch.setattr(transfer, "_ensure_plan", lambda _client, _state: events.append("plan") or plan)
    monkeypatch.setattr(
        transfer,
        "_distributed_source_content_identities",
        lambda _state, _plan: {0: {"digest": "content-v1"}},
    )
    monkeypatch.setattr(
        transfer._transport,
        "transfer",
        lambda _client, _state, _plan, version: events.append(f"transfer:{version}"),
    )
    monkeypatch.setattr(
        transfer,
        "_distributed_policy_fingerprint",
        lambda _state: _fingerprint(1),
    )
    monkeypatch.setattr(
        transfer_module.platform,
        "get_current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append("stream")),
    )
    controller = ActorRolloutWeightSync(
        "qwen",
        "disjoint",
        lambda: client,
        transfer,
    )

    controller.update_weights(PolicySnapshot(1, "qwen", object()))

    assert events == ["plan", "stream", "transfer:1"]
    assert client.events == [
        "pause",
        "start",
        "finish",
        "rpc:verify_direct_content_identity",
        "verify:1:digest-v1",
        "resume",
    ]
    assert transfer.last_completed_strategy == "direct_reshard"
    assert transfer.direct_success_count == 1
    assert controller.configured_strategy == "direct_reshard"
    assert controller.completed_strategy == "direct_reshard"
    assert controller.attempted_strategies == ("direct_reshard",)
    assert controller.policy_version == 1
    assert controller.policy_fingerprint == "digest-v1"
    assert controller.policy_fingerprint_changed


def test_colocated_direct_publish_preserves_residency_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colocated direct publication wakes, streams, verifies, and retains no stable buffer."""
    transfer = ColocatedDirectReshardWeightTransfer(
        _rollout_model(),
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    client = _Client()
    state = {"model.norm.weight": torch.tensor([1.0, 2.0])}
    events: list[str] = []
    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda _payload: dict(state))
    monkeypatch.setattr(
        transfer,
        "_ensure_plan",
        lambda _client, _state: SimpleNamespace(destination_tp_size=1),
    )
    monkeypatch.setattr(
        transfer,
        "_distributed_source_content_identities",
        lambda _state, _plan: {0: {"digest": "content-v2"}},
    )
    monkeypatch.setattr(transfer, "_gather_endpoints", lambda _client: (client.base_url,))
    monkeypatch.setattr(
        transfer,
        "_stream_redistribute_and_send",
        lambda *_args: events.append("redistribute"),
    )
    monkeypatch.setattr(
        transfer,
        "_distributed_policy_fingerprint",
        lambda _state: _fingerprint(2),
    )
    monkeypatch.setattr(
        transfer_module.platform,
        "get_current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append("stream")),
    )

    transfer.publish(client, PolicySnapshot(2, "qwen", object()))

    assert events == ["redistribute", "stream"]
    assert client.events == [
        "wake:weights",
        "pause",
        "start",
        "finish",
        "rpc:verify_direct_content_identity",
        "verify:2:digest-v2",
    ]
    assert transfer.weights_awake
    assert transfer.last_completed_strategy == "direct_reshard"
    transfer.release_failed_buffers()
    transfer.close()


def _streaming_plan():
    """Return one source and replicated destination for streaming publication."""
    sources = (
        SourceTensorLayout(
            "weight",
            "float32",
            4,
            (2,),
            0,
            TensorRegion((0,), (2,)),
        ),
    )
    destinations = (
        DestinationTensorLayout(
            "weight",
            "float32",
            4,
            (2,),
            0,
            1,
            "replicate",
            None,
            TensorRegion((0,), (2,)),
        ),
    )
    return build_streaming_full_gather_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=8,
    )


def test_streaming_full_gather_publish_executes_ack_gated_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full gather materializes one bounded bucket and commits after its ACK."""
    transfer = ColocatedFullGatherWeightTransfer(
        _rollout_model(),
        bucket_size_bytes=8,
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    client = _Client()
    state = {"weight": torch.tensor([1.0, 2.0])}
    plan = _streaming_plan()
    events: list[str] = []
    monkeypatch.setattr(transfer, "_mapped_local_state_dict", lambda _payload: dict(state))
    monkeypatch.setattr(transfer, "_ensure_streaming_plan", lambda _client, _state: plan)
    monkeypatch.setattr(transfer, "_prepare_streaming_transport", lambda _client, _plan: ())
    monkeypatch.setattr(
        transfer,
        "_send_streaming_transport",
        lambda _client, _context, tp_rank, index, bucket, _payload, _version, _size: (
            events.append(f"send:{tp_rank}:{index}")
            or StreamingBucketAck(tp_rank, index, bucket.total_bytes, 1)
        ),
    )
    monkeypatch.setattr(
        transfer,
        "_distributed_policy_fingerprint",
        lambda _state: _fingerprint(3),
    )
    monkeypatch.setattr(transfer_module, "_current_trainer_memory_stats", lambda: {})
    monkeypatch.setattr(transfer_module, "_streaming_memory_summary", lambda *_args: {})
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module.platform, "device_type", lambda: "cpu")
    monkeypatch.setattr(
        transfer_module.platform,
        "get_device_handle",
        lambda _kind: SimpleNamespace(current_device=lambda: 0),
    )
    monkeypatch.setattr(
        transfer_module.platform,
        "get_current_stream",
        lambda: SimpleNamespace(synchronize=lambda: events.append("stream")),
    )
    monkeypatch.setattr(transfer_module.platform, "broadcast", lambda _tensor, src: None)

    transfer.publish(client, PolicySnapshot(3, "qwen", object()))

    assert events == ["stream", "stream", "send:0:0", "stream", "stream"]
    assert client.events == [
        "rpc:get_policy_version",
        "wake:weights",
        "rpc:get_weight_sync_memory_stats",
        "pause",
        "start",
        "finish",
        "rpc:verify_direct_content_identity",
        "rpc:get_weight_sync_memory_stats",
        "verify:3:digest-v3",
    ]
    assert transfer.last_completed_strategy == "full_gather"
    assert transfer.last_streaming_stats["bucket_count"] == 1
    assert transfer.last_streaming_stats["acked_buckets"] == 1
    assert transfer.last_streaming_stats["released_buckets"] == 1


def test_weight_sync_controller_completes_colocated_phase_cycle() -> None:
    """The controller exposes a colocated policy only after wake and verification."""
    client = _Client()
    transfer = _FakeTransfer("direct_reshard")
    controller = ActorRolloutWeightSync(
        "qwen",
        "colocated",
        lambda: client,
        transfer,
    )

    controller.prepare_for_training()
    controller.update_weights(PolicySnapshot(1, "qwen", object()))
    controller.prepare_for_rollout()
    identity = controller.generation_identity(client)

    assert controller.phase == "rollout"
    assert identity == (1, "digest-v1")
    assert client.events == [
        "rpc:get_policy_weight_fingerprint",
        "verify:0:digest-v0",
        "sleep:1:wait",
        "verify:1:digest-v1",
        "wake:kv_cache",
        "pause",
        "verify:1:digest-v1",
        "resume",
        "verify:1:digest-v1",
    ]


def test_transfer_maps_local_state_and_computes_distributed_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State extraction uses canonical names and hashes one distributed norm."""
    transfer = DirectReshardHCCLWeightTransfer(
        _rollout_model(),
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    state = {
        "model.norm.weight": torch.tensor([1.0, 2.0]),
        "model.embed_tokens.weight": torch.arange(4, dtype=torch.float32),
    }
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        transfer_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(
        transfer_module.platform,
        "broadcast",
        lambda _tensor, src: None,
    )

    mapped = transfer._mapped_local_state_dict(state)
    fingerprint = transfer._distributed_policy_fingerprint(mapped)

    assert list(mapped) == [
        "model.norm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert fingerprint["tensor_count"] == 1
    assert fingerprint["value_count"] == 2
    assert fingerprint["digest"]


def test_direct_and_streaming_transfers_build_plans_from_shared_dp_tp_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both strategies collapse rollout DP replicas into the same TP destinations."""
    tensor_layout = {
        "name": "model.norm.weight",
        "dtype_name": "float32",
        "element_size": 4,
        "local_shape": [2],
        "placement": "replicate",
        "shard_dim": None,
    }
    workers = [
        {
            "dp_rank": dp_rank,
            "dp_size": 2,
            "tp_rank": tp_rank,
            "tp_size": 2,
            "physical_device_id": f"host-{dp_rank}-{tp_rank}",
            "tensors": [dict(tensor_layout)],
        }
        for dp_rank in range(2)
        for tp_rank in range(2)
    ]
    client = _Client()
    client.get_world_size = lambda: 4
    client.collective_rpc = lambda method, kwargs=None, base_url=None: workers
    state = {"model.norm.weight": torch.tensor([1.0, 2.0])}
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(sync_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(sync_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        transfer_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    direct = DirectReshardHCCLWeightTransfer(
        _rollout_model(),
        data_parallel_size=2,
        tensor_parallel_size=2,
        bucket_size_bytes=8,
    )
    full = FullGatherHCCLWeightTransfer(
        _rollout_model(),
        data_parallel_size=2,
        tensor_parallel_size=2,
        bucket_size_bytes=8,
    )
    direct_plan = direct._ensure_plan(client, state)
    assert direct._ensure_plan(client, state) is direct_plan
    streaming_plan = full._ensure_streaming_plan(client, state)
    assert full._ensure_streaming_plan(client, state) is streaming_plan
    identities = direct._distributed_source_content_identities(state, direct_plan)

    assert direct_plan.source_world_size == 1
    assert direct_plan.destination_tp_size == 2
    assert direct_plan.route_count == 2
    assert streaming_plan.destination_tp_size == 2
    assert streaming_plan.bucket_count == 2
    assert streaming_plan.total_bytes == 16
    assert set(identities) == {0, 1}
    assert all(identity["total_bytes"] == 8 for identity in identities.values())


def test_colocated_direct_streams_each_bucket_to_its_target_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Colocated direct redistribution sends one bounded IPC payload per bucket."""
    transfer = ColocatedDirectReshardWeightTransfer(
        _rollout_model(),
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    worker = SimpleNamespace(
        dp_rank=0,
        tp_rank=0,
        physical_device_id="host-0",
    )
    sent = []
    stream_events = []
    monkeypatch.setattr(
        transfer,
        "_resolve_ipc_topology",
        lambda *_args: ("host-0", worker, (worker,)),
    )
    monkeypatch.setattr(
        transfer_module,
        "pack_direct_bucket",
        lambda _state, bucket, _device: torch.zeros(
            bucket.total_bytes,
            dtype=torch.uint8,
        ),
    )
    monkeypatch.setattr(
        transfer_module,
        "_tensor_ipc_rebuild_args",
        lambda _tensor: ("ipc-handle",),
    )
    monkeypatch.setattr(
        transfer,
        "_send_payload",
        lambda _client, endpoints, payload, version, tp_size: sent.append(
            (endpoints, payload, version, tp_size)
        ),
    )
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(transfer_module.platform, "device_type", lambda: "cpu")
    monkeypatch.setattr(
        transfer_module.platform,
        "get_device_handle",
        lambda _kind: SimpleNamespace(current_device=lambda: 0),
    )
    monkeypatch.setattr(
        transfer_module.platform,
        "get_current_stream",
        lambda: SimpleNamespace(
            synchronize=lambda: stream_events.append("sync")
        ),
    )
    monkeypatch.setattr(
        transfer_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(transfer_module.platform, "broadcast", lambda _tensor, src: None)
    plan = DirectReshardPlan(
        source_world_size=1,
        destination_tp_size=1,
        bucket_size_bytes=8,
        buckets={(0, 0): (TransferBucket((), 4), TransferBucket((), 8))},
    )

    transfer._stream_redistribute_and_send(
        _Client(),
        ("http://rollout",),
        {},
        plan,
        4,
    )

    assert [item[1]["buckets_by_target"][0][0]["bucket_index"] for item in sent] == [0, 1]
    assert [item[2:] for item in sent] == [(4, 1), (4, 1)]
    assert stream_events == ["sync"] * 6
    assert not transfer._failed_buffers


def test_colocated_streaming_bucket_uses_resolved_ipc_worker_and_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming full gather exports one bounded buffer to its physical worker."""
    transfer = ColocatedFullGatherWeightTransfer(
        _rollout_model(),
        bucket_size_bytes=8,
        data_parallel_size=1,
        tensor_parallel_size=1,
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-0"),
    )
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(transfer_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        transfer_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )
    monkeypatch.setattr(
        transfer_module,
        "_tensor_ipc_rebuild_args",
        lambda _tensor: ("ipc-handle",),
    )
    npu_uuid, local_worker, workers = transfer._resolve_ipc_topology(1, 1)
    calls = []

    class IPCClient(_Client):
        def collective_rpc(self, method: str, kwargs: Any = None, base_url: Any = None):
            calls.append((method, kwargs, base_url))
            if method == "receive_ipc_direct_reshard":
                return [
                    {
                        "received": True,
                        "dp_rank": 0,
                        "tp_rank": 0,
                        "physical_device_id": "host-0",
                    }
                ]
            return super().collective_rpc(method, kwargs, base_url)

    bucket = _streaming_plan().for_target(0)[0]
    materialized = StreamingMaterializedBucket(
        value=torch.zeros(8, dtype=torch.uint8),
        gathered_bytes=8,
        packed_bytes=8,
    )

    ack = transfer._send_streaming_bucket(
        IPCClient(),
        ("http://rollout",),
        workers,
        npu_uuid,
        local_worker,
        0,
        0,
        bucket,
        materialized,
        5,
        1,
    )

    assert ack == StreamingBucketAck(0, 0, 8, 1)
    assert calls[0][0] == "receive_ipc_direct_reshard"
    assert calls[0][2] == "http://rollout"
    assert calls[0][1]["policy_version"] == 5


def test_colocated_moe_query_flattens_dp_tp_into_static_ep_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MoE layout query keeps physical DP/TP identity while exposing EP4 destinations."""
    registration = SimpleNamespace(
        family="qwen3_moe",
        is_hyper=True,
        model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )
    transfer = ColocatedDirectReshardWeightTransfer(
        registration,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    tensors_by_tp = {
        tp_rank: [
            {
                "name": "model.layers.0.mlp.experts.gate_proj.weight",
                "dtype_name": "bfloat16",
                "element_size": 2,
                "local_shape": [2, 3, 4],
                "placement": "shard",
                "shard_dim": 0,
            }
        ]
        for tp_rank in range(2)
    }
    workers = [
        {
            "dp_rank": dp_rank,
            "dp_size": 2,
            "tp_rank": tp_rank,
            "tp_size": 2,
            "ep_rank": dp_rank * 2 + tp_rank,
            "ep_size": 4,
            "physical_device_id": f"host-{dp_rank}-{tp_rank}",
            "tensors": tensors_by_tp[tp_rank],
        }
        for dp_rank in range(2)
        for tp_rank in range(2)
    ]
    client = _Client()
    client.get_world_size = lambda: 4
    client.collective_rpc = lambda method, kwargs=None, base_url=None: workers
    monkeypatch.setattr(sync_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(sync_module.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        sync_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    destinations = transfer._query_destination_workers(client)

    assert [destination["tp_rank"] for destination in destinations] == [0, 1, 2, 3]
    assert [destination["ep_rank"] for destination in destinations] == [0, 1, 2, 3]
    for rank, destination in enumerate(destinations):
        tensor = destination["tensors"][0]
        assert tensor["shard_rank"] == rank
        assert tensor["shard_group_size"] == 4


def test_direct_fallback_completes_with_streaming_full_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed direct attempt aborts before a complete full-gather publication."""
    client = _Client()
    client.version = 2
    released = []

    class Direct:
        weights_awake = True

        @staticmethod
        def publish(_client: Any, _snapshot: PolicySnapshot) -> None:
            raise RuntimeError("direct transport unavailable")

        @staticmethod
        def release_failed_buffers() -> None:
            released.append("buffers")

    fallback = _FakeTransfer("full_gather")

    def publish(
        used_client: Any,
        snapshot: PolicySnapshot,
        *,
        weights_already_awake: bool,
        manifest_strategy: str,
    ) -> None:
        assert used_client is client
        assert weights_already_awake is True
        assert manifest_strategy == "full_gather_fallback"
        fallback.last_policy_fingerprint = _fingerprint(snapshot.version)

    monkeypatch.setattr(fallback, "publish", publish)
    monkeypatch.setattr(transfer_module.platform, "get_rank", lambda: 0)
    wrapper = FallbackWeightTransfer(Direct(), fallback)

    wrapper.publish(client, PolicySnapshot(3, "qwen", object()))

    assert wrapper.last_completed_strategy == "full_gather"
    assert wrapper.last_attempted_strategies == ("direct_reshard", "full_gather")
    assert wrapper.last_fallback_reason == "RuntimeError('direct transport unavailable')"
    assert wrapper.fallback_count == 1
    assert wrapper.last_policy_fingerprint == _fingerprint(3)
    assert released == ["buffers"]


def test_weight_sync_coordinator_and_close_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator results, no-error synchronization, and transfer close remain usable."""
    monkeypatch.setattr(sync_module.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(sync_module.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(
        sync_module.platform,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, None]),
    )
    closed = []
    controller = ActorRolloutWeightSync(
        "qwen",
        "disjoint",
        _Client,
        SimpleNamespace(close=lambda: closed.append("close")),
    )

    result = sync_module.coordinator_call("build", lambda: {"value": 3})
    sync_module.synchronize_error(None, "success")
    controller.close()

    assert result == {"value": 3}
    assert closed == ["close"]


def test_streaming_memory_summary_tracks_baseline_peak_and_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming metrics retain Trainer and rollout memory boundaries."""
    handle = SimpleNamespace(
        memory_allocated=lambda: 11,
        memory_reserved=lambda: 22,
        max_memory_allocated=lambda: 33,
        max_memory_reserved=lambda: 44,
    )
    monkeypatch.setattr(transfer_module.platform, "device_type", lambda: "cpu")
    monkeypatch.setattr(
        transfer_module.platform,
        "get_device_handle",
        lambda _kind: handle,
    )
    monkeypatch.setattr(
        transfer_module.resource,
        "getrusage",
        lambda _kind: SimpleNamespace(ru_maxrss=55),
    )
    trainer_samples = [
        {"allocated_bytes": 10, "reserved_bytes": 20, "host_rss_bytes": 30},
        {"allocated_bytes": 15, "reserved_bytes": 25, "host_rss_bytes": 35},
        {"allocated_bytes": 12, "reserved_bytes": 21, "host_rss_bytes": 31},
    ]
    worker_before = [
        {
            "current_memory_allocated_bytes": 5,
            "current_memory_reserved_bytes": 6,
            "current_host_rss_bytes": 7,
        }
    ]
    worker_after = [
        {
            "current_memory_allocated_bytes": 8,
            "current_memory_reserved_bytes": 9,
            "current_host_rss_bytes": 10,
            "max_memory_allocated_bytes": 18,
            "max_memory_reserved_bytes": 19,
            "host_max_rss_bytes": 20,
        }
    ]

    current = transfer_module._current_trainer_memory_stats()
    summary = transfer_module._streaming_memory_summary(
        trainer_samples,
        worker_before,
        worker_after,
    )

    assert current["allocated_bytes"] == 11
    assert current["reserved_bytes"] == 22
    assert current["host_rss_bytes"] > 0
    assert summary["trainer_baseline_allocated_bytes"] == 10
    assert summary["trainer_peak_current_allocated_bytes"] == 15
    assert summary["trainer_post_release_allocated_bytes"] == 12
    assert summary["rollout_baseline_memory_allocated_bytes"] == 5
    assert summary["rollout_post_release_memory_allocated_bytes"] == 8
    assert summary["trainer_max_memory_allocated_bytes"] == 33
    assert summary["rollout_max_memory_allocated_bytes"] == 18

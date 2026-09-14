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
"""CPU unit tests for production HCCL weight-transfer control logic."""
# Local fakes replace HCCL/NPU boundaries while production transport methods run.
# pylint: disable=forbidden-backend-import,missing-public-docstring,missing-public-type-hints,protected-access

import base64
import pickle
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

import rl.roles.weight_sync.hccl as hccl_module
import rl.roles.weight_sync.ipc as ipc_module
import rl.roles.weight_sync.sync as sync_module
import rl.roles.weight_sync.vllm_client as client_module
from rl.roles.weight_sync.hccl import HCCLWeightTransport
from rl.roles.weight_sync.ipc import PhysicalRolloutWorker
from rl.roles.weight_sync.layout import DirectReshardPlan, TransferBucket, TransferEntry


class _Collectives:
    """Provide deterministic single-rank object collectives and CPU streams."""

    @staticmethod
    def get_rank() -> int:
        return 0

    @staticmethod
    def get_world_size() -> int:
        return 1

    @staticmethod
    def all_gather_object(output: list[Any], value: Any) -> None:
        output[0] = value

    @staticmethod
    def broadcast(unused_tensor, src):
        del unused_tensor
        assert src == 0


def _mock_cpu_device(monkeypatch):
    """Isolate native Torch accelerator APIs without initializing device hardware."""
    stream = SimpleNamespace(synchronize=lambda: None)
    handle = SimpleNamespace(current_device=lambda: 0, current_stream=lambda: stream)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch, "get_device_module", lambda device=None: handle)


@pytest.mark.parametrize("backend", ["ipc", "hccl"])
def test_packed_cpu_buffer_is_staged_before_transport(monkeypatch, backend):
    """CPU-offload output is moved onto the transport device before broadcast."""
    staged = torch.arange(16, dtype=torch.uint8)
    moves = []

    class OffloadedBuffer:
        @staticmethod
        def numel():
            return 16

        @staticmethod
        def to(device):
            moves.append(device)
            return staged

    if backend == "ipc":
        transport, client, unused_exported = _configure_ipc(monkeypatch)
    else:
        transport, client, unused_group = _configure_transport(monkeypatch)
    context = transport.prepare_packed(client)
    ack = transport.send_packed_bucket(client, context, 0, [], 16, OffloadedBuffer(), 1)
    assert ack.total_bytes == 16
    assert len(moves) == 1
    assert moves[0].type == "cpu"


class _Group:
    """Capture the exact packed tensors broadcast by the transport."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.buffers: list[torch.Tensor] = []

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        self.buffers.append(tensor.clone())


class _Client:
    """Acknowledge group initialization and receives for two rollout DP workers."""

    base_url = "http://shared"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any], str]] = []

    def collective_rpc(
        self,
        method: str,
        kwargs: Mapping[str, Any],
        base_url: str,
    ) -> list[dict[str, Any]]:
        """Emulate worker RPC acknowledgements and configured transport failures."""
        self.calls.append((method, kwargs, base_url))
        if method == "init_direct_reshard_group":
            target_tp = int(kwargs["target_tp_rank"])
            return [
                {
                    "joined": True,
                    "dp_rank": dp_rank,
                    "tp_rank": target_tp,
                    "group_rank": 1 + dp_rank,
                }
                for dp_rank in range(2)
            ]
        if method == "init_packed_weight_group":
            return [
                {
                    "joined": True,
                    "dp_rank": dp_rank,
                    "tp_rank": 0,
                    "group_rank": 1 + dp_rank,
                }
                for dp_rank in range(2)
            ]
        if method == "receive_packed_weights":
            return [
                {
                    "received": True,
                    "dp_rank": dp_rank,
                    "tp_rank": 0,
                    "bytes": int(kwargs["total_bytes"]),
                }
                for dp_rank in range(2)
            ]
        target_tp = int(kwargs["target_tp_rank"])
        transferred = sum(
            int(entry["num_bytes"])
            for bucket in kwargs["buckets"]
            for entry in bucket["entries"]
        )
        return [
            {
                "received": True,
                "dp_rank": dp_rank,
                "tp_rank": target_tp,
                "bytes": transferred,
            }
            for dp_rank in range(2)
        ]


def _plan() -> DirectReshardPlan:
    """Return one 16-byte source-to-TP route split into two entries."""
    entries = (
        TransferEntry("weight", "float32", 4, (0, 0), (0, 0), (1, 2), buffer_offset=0),
        TransferEntry("weight", "float32", 4, (1, 0), (1, 0), (1, 2), buffer_offset=8),
    )
    return DirectReshardPlan(
        source_world_size=1,
        destination_tp_size=1,
        bucket_size_bytes=16,
        buckets={(0, 0): (TransferBucket(entries, total_bytes=16),)},
    )


def _configure_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[HCCLWeightTransport, _Client, _Group]:
    """Patch only the port, synchronization, and HCCL group boundaries."""
    _mock_cpu_device(monkeypatch)
    group = _Group()
    transport = HCCLWeightTransport(
        data_parallel_size=2,
        tensor_parallel_size=1,
    )
    client = _Client()
    monkeypatch.setattr(hccl_module, "dist", _Collectives())
    monkeypatch.setattr(client_module, "dist", _Collectives())
    monkeypatch.setattr(hccl_module, "_open_port", lambda: 12345)
    monkeypatch.setattr(
        hccl_module,
        "synchronize_error",
        lambda error, _operation: (
            None if error is None else (_ for _ in ()).throw(error)
        ),
    )
    monkeypatch.setattr(transport, "_trainer_init", lambda _info: group)
    return transport, client, group


def test_hccl_transport_sends_complete_route_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production transfer initializes, packs, sends, acknowledges, and closes its route."""
    transport, client, group = _configure_transport(monkeypatch)

    transport.transfer_direct(
        client,
        {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        _plan(),
        policy_version=2,
    )

    assert transport._group_ids[(0, 0)] == "hyper-direct-s0-t0-d2-p12345"
    assert transport._groups[(0, 0)] is group
    assert len(group.buffers) == 1
    torch.testing.assert_close(
        group.buffers[0].view(torch.float32),
        torch.arange(4, dtype=torch.float32),
    )
    assert [call[0] for call in client.calls] == [
        "init_direct_reshard_group",
        "receive_direct_reshard",
    ]
    transport.close()
    assert not transport._groups
    assert not transport._group_ids
    assert transport._endpoint is None


def test_hccl_transport_broadcasts_one_packed_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed full gather uses one producer-to-all-workers HCCL group."""
    transport, client, group = _configure_transport(monkeypatch)
    endpoint = transport.prepare_packed(client)
    metadata = [{
        "name": "weight",
        "dtype_name": "float32",
        "shape": [4],
        "buffer_offset": 0,
        "num_bytes": 16,
    }]
    packed = torch.arange(16, dtype=torch.uint8)

    ack = transport.send_packed_bucket(
        client,
        endpoint,
        0,
        metadata,
        16,
        packed,
        3,
    )

    assert endpoint == client.base_url
    assert ack.worker_count == 2
    assert [call[0] for call in client.calls] == [
        "init_packed_weight_group",
        "receive_packed_weights",
    ]
    torch.testing.assert_close(group.buffers[0], packed)


class _IPCClient:
    """Decode a real sender payload and acknowledge its intended physical worker."""
    base_url = "http://shared"

    def __init__(self):
        self.payloads = []
        self.fail = False

    def collective_rpc(self, method, kwargs, endpoint):
        """Emulate worker RPC acknowledgements and configured transport failures."""
        assert endpoint == self.base_url
        if method == "receive_ipc_packed_weights":
            payload = pickle.loads(base64.b64decode(kwargs["payload_pickled"]))
            self.payloads.append(payload)
            if self.fail:
                raise RuntimeError("receive interrupted after exporting storage")
            return [{
                "received": True,
                "dp_rank": 0,
                "tp_rank": 0,
                "physical_device_id": "npu-0",
                "bytes": int(payload["total_bytes"]),
            }]
        assert method == "receive_ipc_direct_reshard"
        payload = pickle.loads(base64.b64decode(kwargs["payload_pickled"]))
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("receive interrupted after exporting storage")
        bucket = payload["buckets_by_target"][0][0]
        return [{
            "received": True, "dp_rank": 0, "tp_rank": 0, "physical_device_id": "npu-0",
            "bytes": sum(entry["num_bytes"] for entry in bucket["metadata"]["entries"]),
        }]


def _configure_ipc(monkeypatch):
    """Install CPU runtime and IPC doubles while recording exported storage."""
    _mock_cpu_device(monkeypatch)
    monkeypatch.setattr(ipc_module, "dist", _Collectives())
    monkeypatch.setattr(client_module, "dist", _Collectives())
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
                        SimpleNamespace(npu_generate_uuid=lambda: "npu-0"))
    exported = []
    def export(tensor):
        exported.append(tensor.clone())
        return ("shared-handle",)
    monkeypatch.setattr(ipc_module, "tensor_ipc_rebuild_args", export)
    return ipc_module.IPCWeightTransport(), _IPCClient(), exported


def test_ipc_direct_packs_and_delivers_through_one_endpoint(monkeypatch):
    """A direct payload preserves bytes, physical ownership and one RPC per bucket."""
    transport, client, exported = _configure_ipc(monkeypatch)
    state = {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    transport.transfer_direct(client, state, _plan(), 1)
    assert len(client.payloads) == len(exported) == 1
    torch.testing.assert_close(exported[0].view(torch.float32), state["weight"].flatten())
    payload = client.payloads[0]
    assert payload["worker_topology"] == [{"dp_rank": 0, "tp_rank": 0, "physical_device_id": "npu-0"}]
    assert set(payload["buckets_by_target"][0][0]["ipc_handles"]) == {"npu-0"}
    assert transport.failed_buffer_count == 0
    context = transport.prepare(client, 1)
    assert transport.prepare(client, 1) is context
    transport.close()
    assert transport._context is None


def test_ipc_retains_unacknowledged_storage_until_close(monkeypatch):
    """A failed RPC keeps its producer alive until the failed run closes."""
    transport, client, unused_exported = _configure_ipc(monkeypatch)
    context = transport.prepare(client, 1)
    packed = torch.arange(16, dtype=torch.uint8)
    metadata = [{
        "name": "weight",
        "dtype_name": "float32",
        "shape": [4],
        "buffer_offset": 0,
        "num_bytes": 16,
    }]
    client.fail = True
    with pytest.raises(RuntimeError, match="receive interrupted"):
        transport.send_packed_bucket(
            client, context, 0, metadata, 16, packed, 1,
        )
    assert transport._failed_buffers[0] is packed
    assert transport.failed_buffer_count == 1
    client.fail = False
    ack = transport.send_packed_bucket(
        client, context, 0, metadata, 16, packed, 1,
    )
    assert ack.worker_count == 1 and ack.total_bytes == 16
    assert transport.failed_buffer_count == 1
    transport.close()
    assert transport.failed_buffer_count == 0


@pytest.mark.parametrize("ranks, valid", [([0, 1], True), ([0, 1, 2, 3], True), ([0], False), ([0, 0], False)])
def test_ipc_ack_checks_complete_tp_representatives(ranks, valid):
    """Representative DP replies are valid only with complete, unique TP ownership."""
    workers = tuple(PhysicalRolloutWorker(index // 2, index % 2, f"npu-{index}") for index in range(4))
    context = ipc_module.IPCContext("http://shared", "npu-0", workers[0], workers)
    transport = ipc_module.IPCWeightTransport(data_parallel_size=2, tensor_parallel_size=2)
    results = [{"received": True, "dp_rank": rank // 2, "tp_rank": rank % 2,
                "physical_device_id": f"npu-{rank}", "bytes": 16 if rank % 2 == 0 else 0} for rank in ranks]
    if valid:
        transport._validate_results(results, context, 0, 16)
    else:
        with pytest.raises(RuntimeError, match="IPC"):
            transport._validate_results(results, context, 0, 16)


def test_shared_endpoint_rejects_different_trainer_endpoints(monkeypatch):
    """Endpoint disagreement is rejected before either transport creates resources."""
    fake = SimpleNamespace(
        get_world_size=lambda: 2,
        all_gather_object=lambda values, _value: values.__setitem__(
            slice(None), ["http://a", "http://b"]
        ),
    )
    monkeypatch.setattr(client_module, "dist", fake)
    with pytest.raises(RuntimeError, match="one shared"):
        client_module.shared_endpoint(SimpleNamespace(base_url="http://a"))


@pytest.mark.parametrize("strategy", ["direct", "packed"])
def test_ipc_export_failure_is_synchronized_before_handle_collective(monkeypatch, strategy):
    """An export failure enters error gathering instead of the handle collective."""
    transport, client, unused_exported = _configure_ipc(monkeypatch)
    context = transport.prepare_packed(client)
    packed = torch.arange(16, dtype=torch.uint8)
    events = []

    class TwoRanks(_Collectives):
        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather_object(output, value):
            assert value is None or isinstance(value, str), "Entered handle gather after export failure"
            events.append(value)
            output[:] = [value, None]

    def fail_export(unused_tensor):
        del unused_tensor
        raise RuntimeError("export failed")

    monkeypatch.setattr(ipc_module, "dist", TwoRanks())
    monkeypatch.setattr(sync_module, "dist", TwoRanks())
    monkeypatch.setattr(ipc_module, "tensor_ipc_rebuild_args", fail_export)
    with pytest.raises(RuntimeError, match="export failed"):
        if strategy == "direct":
            transport.send_bucket(client, context, 0, 0, _plan().for_route(0, 0)[0].worker_metadata(), packed, 1)
        else:
            transport.send_packed_bucket(client, context, 0, [], 16, packed, 1)
    assert events[-1] == "export failed"
    assert not client.payloads
    assert transport._failed_buffers == [packed]


def test_packed_ipc_allocation_failure_precedes_broadcast(monkeypatch):
    """Receiver allocation failure is propagated before any tensor broadcast."""
    transport, client, unused_exported = _configure_ipc(monkeypatch)
    context = transport.prepare_packed(client)
    errors = []

    class Receiver(_Collectives):
        """Represent a non-producer rank during synchronized allocation failure."""
        @staticmethod
        def get_rank():
            return 1

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather_object(output, value):
            errors.append(value)
            output[:] = [None, value]

        @staticmethod
        def broadcast(unused_tensor, src):
            del unused_tensor
            pytest.fail("Broadcast started after receiver allocation failed")

    def fail_allocate(*_args, **_kwargs):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(ipc_module, "dist", Receiver())
    monkeypatch.setattr(sync_module, "dist", Receiver())
    monkeypatch.setattr(torch, "empty", fail_allocate)
    with pytest.raises(RuntimeError, match="allocation failed"):
        transport.send_packed_bucket(client, context, 0, [], 16, None, 1)
    assert errors == ["allocation failed"]
    assert not client.payloads


@pytest.mark.parametrize("failure", ["rpc", "broadcast"])
def test_packed_hccl_propagates_failures_and_retains_buffer(monkeypatch, failure):
    """RPC and broadcast failures leave the producer buffer owned by the transport."""
    transport, client, group = _configure_transport(monkeypatch)
    endpoint = transport.prepare_packed(client)
    packed = torch.arange(16, dtype=torch.uint8)

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure} failed")

    if failure == "rpc":
        monkeypatch.setattr(client, "collective_rpc", fail)
    else:
        monkeypatch.setattr(group, "broadcast", fail)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        transport.send_packed_bucket(client, endpoint, 0, [], 16, packed, 1)
    assert transport._failed_buffers == [packed]
    transport.close()
    assert not transport._failed_buffers


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_direct_hccl_keeps_coordinator_separate_from_producer(monkeypatch, rank):
    """Rank zero issues RPC, only rank one packs, and idle ranks share the ACK."""
    transport, client, group = _configure_transport(monkeypatch)
    metadata = _plan().for_route(0, 0)[0].worker_metadata()
    packed = torch.arange(16, dtype=torch.uint8)
    materialized = []
    replies = [{"received": True, "dp_rank": dp, "tp_rank": 0, "bytes": 16} for dp in range(2)]

    class TrainerRanks(_Collectives):
        """Expose the selected trainer rank for collective coordination assertions."""
        @staticmethod
        def get_rank():
            return rank

        @staticmethod
        def get_world_size():
            return 3

        @staticmethod
        def all_gather_object(output, value):
            assert (value is not None) is (rank == 0)
            output[:] = [replies, None, None]

    def materialize(index):
        materialized.append(index)
        return packed

    monkeypatch.setattr(hccl_module, "dist", TrainerRanks())
    transport._group_ids[(1, 0)] = "source-1"
    if rank == 1:
        transport._groups[(1, 0)] = group
    sent, copied = transport._broadcast_buffers(client, client.base_url, 1, 0, [metadata], materialize, 1)
    assert len(client.calls) == int(rank == 0)
    assert materialized == ([0] if rank == 1 else [])
    assert len(group.buffers) == int(rank == 1)
    assert (sent, copied) == (16 if rank == 1 else 0, 16)

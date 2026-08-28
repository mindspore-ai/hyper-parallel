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
"""CPU contracts for FSDP-to-rollout direct reshard over HCCL."""

import sys
import threading
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

import rl.roles.weight_sync.hccl as hccl_module
import rl.roles.weight_sync.vllm_worker as worker_module
from rl.roles.weight_sync.hccl import BroadcastDirectReshardHCCLTransport
from rl.roles.weight_sync.layout import (
    DirectReshardPlan,
    TransferBucket,
    TransferEntry,
)
from rl.roles.weight_sync.transfer import FullGatherHCCLWeightTransfer


def test_direct_reshard_uses_one_shared_rpc_for_all_dp_receivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One source/TP route sends once while one shared RPC fans out to DP2."""

    class FakeStream:
        """Expose the synchronization method used around one HCCL broadcast."""

        @staticmethod
        def synchronize() -> None:
            """Represent a completed CPU test stream."""

    class FakePlatform:
        """Allocate route buffers on CPU for this metadata/packing test."""

        @staticmethod
        def get_current_stream() -> FakeStream:
            """Return the synchronous fake stream."""
            return FakeStream()

        @staticmethod
        def get_rank() -> int:
            """Run the source and coordinator roles on this test thread."""
            return 0

        @staticmethod
        def get_world_size() -> int:
            """Represent one Trainer process for this route-level test."""
            return 1

        @staticmethod
        def all_gather_object(output: list[Any], value: Any) -> None:
            """Complete one single-rank synchronization."""
            output[0] = value

    class FakeGroup:
        """Capture the packed bytes supplied to PyHCCL Broadcast."""

        device = torch.device("cpu")

        def __init__(self) -> None:
            self.buffers: list[torch.Tensor] = []

        def broadcast(self, tensor: torch.Tensor, src: int) -> None:
            """Record one immutable copy of the source buffer."""
            assert src == 0
            self.buffers.append(tensor.clone())

    class FakeDirectClient:
        """Return one visible ACK for a worker-local all-engine operation."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, Mapping[str, Any], str]] = []

        def collective_rpc(
            self,
            method: str,
            kwargs: Mapping[str, Any],
            base_url: str,
        ) -> list[dict[str, Any]]:
            """Acknowledge the target worker and reject no route metadata."""
            assert method == "receive_direct_reshard"
            self.calls.append((method, kwargs, base_url))
            assert kwargs["expected_data_parallel_size"] == 2
            assert kwargs["expected_tensor_parallel_size"] == 2
            transferred_bytes = sum(
                int(entry["num_bytes"])
                for bucket in kwargs["buckets"]
                for entry in bucket["entries"]
            )
            return [
                {
                    "received": True,
                    "dp_rank": 0,
                    "tp_rank": kwargs["target_tp_rank"],
                    "bytes": transferred_bytes,
                }
            ]

    entry = TransferEntry(
        name="weight",
        dtype_name="bfloat16",
        element_size=2,
        source_starts=(0, 0),
        destination_starts=(0, 0),
        lengths=(2, 2),
    )
    bucket = TransferBucket((entry,), total_bytes=8)
    plan = DirectReshardPlan(
        source_world_size=1,
        destination_tp_size=2,
        bucket_size_bytes=8,
        buckets={(0, 0): (bucket,)},
    )
    transport = BroadcastDirectReshardHCCLTransport(
        data_parallel_size=2,
        tensor_parallel_size=2,
    )
    group = FakeGroup()
    transport._groups[(0, 0)] = group  # pylint: disable=protected-access
    transport._group_ids[(0, 0)] = "route"  # pylint: disable=protected-access
    client = FakeDirectClient()
    monkeypatch.setattr(hccl_module, "platform", FakePlatform())

    sent_bytes, fragment_bytes = transport._broadcast_route(  # pylint: disable=protected-access
        client,
        "http://rollout",
        {"weight": torch.arange(4, dtype=torch.bfloat16).view(2, 2)},
        plan,
        source_rank=0,
        tp_rank=0,
        policy_version=1,
    )

    assert sent_bytes == 8
    assert fragment_bytes == 8
    assert len(client.calls) == 1
    assert client.calls[0][2] == "http://rollout"
    assert len(group.buffers) == 1
    assert torch.equal(
        group.buffers[0].view(torch.bfloat16),
        torch.arange(4, dtype=torch.bfloat16),
    )


def test_direct_route_runs_coordinator_rpc_concurrently_with_nonzero_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer rank 0 submits the RPC while source rank 1 enters HCCL Broadcast."""

    class FakePlatform:
        """Provide deterministic two-rank object collectives to two threads."""

        def __init__(self) -> None:
            self.local = threading.local()
            self.barrier = threading.Barrier(2)
            self.values: dict[tuple[int, int], Any] = {}
            self.lock = threading.Lock()

        def set_rank(self, rank: int) -> None:
            """Bind one simulated Trainer rank to the current thread."""
            self.local.rank = rank
            self.local.collective_index = 0

        def get_rank(self) -> int:
            """Return this thread's simulated Trainer rank."""
            return self.local.rank

        @staticmethod
        def get_world_size() -> int:
            """Return the simulated Trainer world size."""
            return 2

        @staticmethod
        def get_current_stream() -> Any:
            """Return a synchronous CPU stream stand-in."""
            return SimpleNamespace(synchronize=lambda: None)

        def all_gather_object(self, output: list[Any], value: Any) -> None:
            """Gather one Python value in identical per-rank call order."""
            index = self.local.collective_index
            self.local.collective_index += 1
            rank = self.get_rank()
            with self.lock:
                self.values[(index, rank)] = value
            self.barrier.wait()
            for gathered_rank in range(2):
                output[gathered_rank] = self.values[(index, gathered_rank)]
            self.barrier.wait()
            if rank == 0:
                with self.lock:
                    for gathered_rank in range(2):
                        del self.values[(index, gathered_rank)]
            self.barrier.wait()

    rpc_started = threading.Event()
    broadcast_completed = threading.Event()

    class FakeClient:
        """Block the receiver RPC until source rank 1 broadcasts."""

        calls = 0

        def collective_rpc(self, method, kwargs, base_url):
            assert method == "receive_direct_reshard"
            assert base_url == "http://shared"
            self.calls += 1
            rpc_started.set()
            assert broadcast_completed.wait(timeout=5)
            return [
                {
                    "received": True,
                    "dp_rank": 0,
                    "tp_rank": kwargs["target_tp_rank"],
                    "bytes": 8,
                }
            ]

    class FakeGroup:
        """Require the coordinator request before completing the source broadcast."""

        device = torch.device("cpu")

        @staticmethod
        def broadcast(_tensor: torch.Tensor, src: int) -> None:
            assert src == 0
            assert rpc_started.wait(timeout=5)
            broadcast_completed.set()

    entry = TransferEntry(
        name="weight",
        dtype_name="bfloat16",
        element_size=2,
        source_starts=(0, 0),
        destination_starts=(0, 0),
        lengths=(2, 2),
    )
    plan = DirectReshardPlan(
        source_world_size=2,
        destination_tp_size=2,
        bucket_size_bytes=8,
        buckets={(1, 0): (TransferBucket((entry,), total_bytes=8),)},
    )
    fake_platform = FakePlatform()
    monkeypatch.setattr(hccl_module, "platform", fake_platform)
    client = FakeClient()
    results: list[Any] = [None, None]
    errors: list[Any] = [None, None]

    def run(rank: int) -> None:
        fake_platform.set_rank(rank)
        transport = BroadcastDirectReshardHCCLTransport(
            data_parallel_size=2,
            tensor_parallel_size=2,
        )
        transport._group_ids[(1, 0)] = "route"  # pylint: disable=protected-access
        if rank == 1:
            transport._groups[(1, 0)] = FakeGroup()  # pylint: disable=protected-access
        try:
            results[rank] = transport._broadcast_route(  # pylint: disable=protected-access
                client,
                "http://shared",
                {"weight": torch.arange(4, dtype=torch.bfloat16).view(2, 2)},
                plan,
                source_rank=1,
                tp_rank=0,
                policy_version=1,
            )
        except Exception as error:  # pylint: disable=W0718
            errors[rank] = error

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert results == [(0, 8), (8, 8)]
    assert client.calls == 1


def test_direct_group_initializes_one_shared_dp2_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One RPC joins matching DP workers to a 1+DP route group."""
    calls = []
    group = object()

    class FakePlatform:
        """Represent Trainer rank 0 for route initialization."""

        @staticmethod
        def get_rank() -> int:
            """Return the coordinator/source rank."""
            return 0

        @staticmethod
        def get_world_size() -> int:
            """Return the one-process Trainer world."""
            return 1

        @staticmethod
        def all_gather_object(output: list[Any], value: Any) -> None:
            """Complete one single-rank object collective."""
            output[0] = value

    class FakeClient:
        """Return one visible ACK from an all-worker collective RPC."""

        @staticmethod
        def collective_rpc(method, kwargs, base_url):
            """Record and acknowledge one worker-local group initialization."""
            calls.append((method, kwargs, base_url))
            return [
                {
                    "joined": True,
                    "dp_rank": 0,
                    "tp_rank": 1,
                    "group_rank": 1,
                }
            ]

    monkeypatch.setattr(hccl_module, "platform", FakePlatform())
    monkeypatch.setattr(
        BroadcastDirectReshardHCCLTransport,
        "_trainer_init",
        staticmethod(lambda init_info: calls.append(("trainer", init_info)) or group),
    )
    transport = BroadcastDirectReshardHCCLTransport(
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    transport._initialize_route(  # pylint: disable=protected-access
        FakeClient(),
        "http://shared",
        source_rank=0,
        tp_rank=1,
    )

    rpc_call = next(call for call in calls if call[0] == "init_direct_reshard_group")
    assert rpc_call[1]["world_size"] == 3
    assert rpc_call[1]["expected_data_parallel_size"] == 2
    assert rpc_call[1]["expected_tensor_parallel_size"] == 2
    assert rpc_call[2] == "http://shared"
    assert transport._groups[(0, 1)] is group  # pylint: disable=protected-access


def test_worker_direct_group_uses_dp_major_receiver_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DP2 TP1 worker joins group rank 2 from its worker-local topology."""
    init_calls = []

    class FakeEngine:
        """Capture stateless HCCL receiver initialization."""

        @staticmethod
        def _stateless_init_process_group(*args, **kwargs):
            init_calls.append((args, kwargs))
            return object()

    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 1,
            "dp_size": 2,
            "tp_rank": 0,
            "tp_size": 2,
            "physical_device_id": "host-2",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        SimpleNamespace(HCCLWeightTransferEngine=FakeEngine),
    )
    monkeypatch.setattr(
        worker_module.platform,
        "get_device_handle",
        lambda _device_type: SimpleNamespace(
            current_device=lambda: 3,
            synchronize=lambda: None,
        ),
    )
    monkeypatch.setattr(worker_module.platform, "device_type", lambda: "npu")
    worker = SimpleNamespace()

    result = worker_module.init_direct_reshard_group(
        worker,
        group_id="route",
        target_tp_rank=0,
        master_address="127.0.0.1",
        master_port=12345,
        world_size=3,
        expected_data_parallel_size=2,
        expected_tensor_parallel_size=2,
    )

    assert result == {
        "joined": True,
        "dp_rank": 1,
        "tp_rank": 0,
        "group_rank": 2,
        "group_id": "route",
    }
    assert init_calls[0][0][2:4] == (2, 3)

    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 3,
            "tp_rank": 0,
            "tp_size": 2,
            "physical_device_id": "host-0",
        },
    )
    with pytest.raises(ValueError, match="topology differs"):
        worker_module.init_direct_reshard_group(
            SimpleNamespace(),
            group_id="mismatch",
            target_tp_rank=0,
            master_address="127.0.0.1",
            master_port=12346,
            world_size=3,
            expected_data_parallel_size=2,
            expected_tensor_parallel_size=2,
        )


def test_full_gather_uses_one_dp2_tp2_group_and_one_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full gather initializes and updates one shared 1+DP*TP HCCL group."""
    calls = []
    group = object()

    class FakeEngine:
        """Capture trainer group initialization and send calls."""

        @staticmethod
        def trainer_init(init_info):
            calls.append(("trainer_init", dict(init_info)))
            return group

        @staticmethod
        def trainer_send_weights(iterator, trainer_args):
            calls.append(("trainer_send", list(iterator), trainer_args))

    class FakeClient:
        """Expose the shared full-gather HTTP operations."""

        @staticmethod
        def get_world_size(base_url=None):
            assert base_url == "http://shared"
            calls.append(("world_size", base_url))
            return 4

        @staticmethod
        def collective_rpc(method, kwargs, base_url=None):
            assert method == "init_full_gather_group"
            calls.append(("server_init", dict(kwargs), base_url))
            return [
                {
                    "joined": True,
                    "dp_rank": dp_rank,
                    "tp_rank": tp_rank,
                    "group_rank": 1 + dp_rank * 2 + tp_rank,
                }
                for dp_rank in range(2)
                for tp_rank in range(2)
            ]

        @staticmethod
        def receive_weights(update_info, policy_version, base_url=None):
            calls.append(("server_update", dict(update_info), policy_version, base_url))

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        SimpleNamespace(
            HCCLTrainerSendWeightsArgs=lambda **kwargs: SimpleNamespace(**kwargs),
            HCCLWeightTransferEngine=FakeEngine,
        ),
    )
    transfer = FullGatherHCCLWeightTransfer(
        SimpleNamespace(),
        bucket_size_bytes=16,
        data_parallel_size=2,
        tensor_parallel_size=2,
    )

    transfer._send_shared(  # pylint: disable=protected-access
        FakeClient(),
        "http://shared",
        {"weight": torch.ones(2, dtype=torch.float32)},
        policy_version=1,
    )

    server_init = [call for call in calls if call[0] == "server_init"]
    server_update = [call for call in calls if call[0] == "server_update"]
    trainer_send = [call for call in calls if call[0] == "trainer_send"]
    assert len(server_init) == len(server_update) == len(trainer_send) == 1
    assert server_init[0][1]["world_size"] == 5
    assert server_init[0][1]["expected_data_parallel_size"] == 2
    assert server_init[0][1]["expected_tensor_parallel_size"] == 2
    assert trainer_send[0][2].group is group


def test_worker_full_gather_uses_global_dp_major_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DP1 TP0 joins a DP2 x TP2 full-gather group as rank 3."""
    init_calls = []

    class FakeTransferEngine:
        """Capture the worker-side stateless communicator arguments."""

        model_update_group = None

        @staticmethod
        def _stateless_init_process_group(*args, **kwargs):
            init_calls.append((args, kwargs))
            return "group"

    worker = SimpleNamespace(
        weight_transfer_engine=FakeTransferEngine(),
        _check_weight_transfer_engine=lambda: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 1,
            "dp_size": 2,
            "tp_rank": 0,
            "tp_size": 2,
            "physical_device_id": "host-4",
        },
    )
    monkeypatch.setattr(
        worker_module.platform,
        "get_device_handle",
        lambda _device_type: SimpleNamespace(
            current_device=lambda: 4,
            synchronize=lambda: None,
        ),
    )
    monkeypatch.setattr(worker_module.platform, "device_type", lambda: "npu")

    result = worker_module.init_full_gather_group(
        worker,
        master_address="127.0.0.1",
        master_port=12345,
        world_size=5,
        expected_data_parallel_size=2,
        expected_tensor_parallel_size=2,
    )

    assert result == {
        "joined": True,
        "dp_rank": 1,
        "tp_rank": 0,
        "group_rank": 3,
    }
    assert init_calls[0][0][2:4] == (3, 5)
    assert worker.weight_transfer_engine.model_update_group == "group"

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
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access

from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

import rl.roles.weight_sync.hccl as hccl_module
from rl.roles.weight_sync.hccl import BroadcastDirectReshardHCCLTransport
from rl.roles.weight_sync.layout import DirectReshardPlan, TransferBucket, TransferEntry


class _Platform:
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
    def get_current_stream() -> Any:
        return SimpleNamespace(synchronize=lambda: None)


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
        self.calls.append((method, kwargs, base_url))
        target_tp = int(kwargs["target_tp_rank"])
        if method == "init_direct_reshard_group":
            return [
                {
                    "joined": True,
                    "dp_rank": dp_rank,
                    "tp_rank": target_tp,
                    "group_rank": 1 + dp_rank,
                }
                for dp_rank in range(2)
            ]
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
) -> tuple[BroadcastDirectReshardHCCLTransport, _Client, _Group]:
    """Patch only the port, synchronization, and HCCL group boundaries."""
    group = _Group()
    transport = BroadcastDirectReshardHCCLTransport(
        data_parallel_size=2,
        tensor_parallel_size=1,
    )
    client = _Client()
    monkeypatch.setattr(hccl_module, "platform", _Platform())
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

    transport.transfer(
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


def test_hccl_transport_streams_one_acknowledged_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming full gather reuses one HCCL route and returns all receiver ACKs."""
    transport, client, group = _configure_transport(monkeypatch)
    endpoint = transport.ensure_streaming_groups(client, destination_tp_size=1)
    metadata = _plan().for_route(0, 0)[0].worker_metadata()
    packed = torch.arange(16, dtype=torch.uint8)

    workers = transport.broadcast_streaming_bucket(
        client,
        endpoint,
        packed,
        metadata,
        target_tp_rank=0,
        bucket_index=0,
        policy_version=3,
    )

    assert endpoint == client.base_url
    assert workers == 2
    assert [call[0] for call in client.calls] == [
        "init_direct_reshard_group",
        "receive_direct_reshard",
    ]
    torch.testing.assert_close(group.buffers[0], packed)

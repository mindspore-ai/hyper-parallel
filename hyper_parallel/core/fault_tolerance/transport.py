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
"""Direct device tensor repair; only rank-owned metadata is mirrored during training."""

import io
import threading
from collections import OrderedDict
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.fault_tolerance.state import freeze

_TRANSFER_CHUNK_BYTES = 4 * 1024 * 1024
_RETAINED_VERSIONS = 2
_METADATA_TIMEOUT_SECONDS = 10


def encode(value: Any) -> bytes:
    """Serialize Python training metadata, never model weights or optimizer moments."""
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def decode(payload: bytes) -> Any:
    """Decode small metadata received from a trusted training peer."""
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def device_bytes(value: Any, device: torch.device) -> torch.Tensor:
    """Place serialized metadata in a device byte buffer."""
    return torch.frombuffer(bytearray(encode(value)), dtype=torch.uint8).to(device)


class DeviceMetadataBank:
    """Mirror rank-specific device metadata and keep donor providers local until repair."""

    def __init__(self, rank: int, device: torch.device) -> None:
        """Bind this rank's device and allocate a bounded metadata version ring."""
        self.rank, self.device = rank, device
        self._entries = {}
        self._replica_entries = OrderedDict()
        self._lock = threading.RLock()

    def put(self, step: int, metadata: Any, buffers: dict[str, torch.Tensor]) -> None:
        """Separate local replica providers from mirrored rank state and device buffers.

        Args:
            step: Committed training step represented by this snapshot.
            metadata: Training snapshot including replica_state and module extra state.
            buffers: Rank-owned buffer snapshots on the worker device.
        """
        for value in buffers.values():
            if value.device != self.device:
                raise ValueError("recovery buffers must reside on the worker device")
        state = metadata["state"]
        with self._lock:
            self._replica_entries[step] = freeze(state["replica_state"])
            self._replica_entries.move_to_end(step)
            while len(self._replica_entries) > _RETAINED_VERSIONS:
                self._replica_entries.popitem(last=False)
        metadata = {**metadata, "state": {"rank_state": state["rank_state"]}}
        self.install(self.rank, step, device_bytes(metadata, self.device), buffers)

    def install(self, owner: int, step: int, payload: torch.Tensor, buffers: dict[str, torch.Tensor]) -> None:
        """Install device buffers without retaining a CPU checkpoint."""
        with self._lock:
            slots = self._entries.setdefault(owner, OrderedDict())
            slots[step] = {"owner": owner, "step": step, "payload": payload, "buffers": buffers}
            slots.move_to_end(step)
            while len(slots) > _RETAINED_VERSIONS:
                slots.popitem(last=False)

    def get(self, owner: int, step: int) -> dict[str, Any]:
        """Require exact target-rank metadata; a donor's own RNG is not interchangeable."""
        with self._lock:
            if step not in self._entries.get(owner, {}):
                raise RuntimeError(f"exact training metadata unavailable: owner={owner}, step={step}")
            return self._entries[owner][step]

    def tensors(self) -> list[torch.Tensor]:
        """List retained device allocations for torch-npu safety recovery."""
        with self._lock:
            return [value for slots in self._entries.values() for entry in slots.values()
                    for value in [entry["payload"], *entry["buffers"].values()]]

    def metadata(self, owner: int, step: int) -> dict[str, Any]:
        """Combine target-rank state with the donor's exact-step replica providers.

        Args:
            owner: Logical rank whose rank-specific state is required.
            step: Exact committed step required for every restored state component.

        Returns:
            Decoded rank state combined with the local replica-provider snapshot.
        """
        entry = self.get(owner, step)
        metadata = decode(entry["payload"].cpu().numpy().tobytes())
        with self._lock:
            if step not in self._replica_entries:
                raise RuntimeError(f"exact replica provider metadata unavailable: step={step}")
            metadata["state"]["replica_state"] = freeze(self._replica_entries[step])
        return metadata


class ReplicaTransport:
    """Send the donor's live device model/main/moment tensors directly over HCCL."""

    def __init__(self, group: Any, bank: DeviceMetadataBank, state: Any, device: torch.device) -> None:
        """Bind the exact replica collective group and persistent live device state."""
        self.group, self.bank, self.state, self.device = group, bank, state, device
        self.ranks = tuple(dist.get_process_group_ranks(group))

    @staticmethod
    def _wait(work: Any) -> None:
        try:
            work.wait(timeout=timedelta(seconds=_METADATA_TIMEOUT_SECONDS))
        except RuntimeError as error:
            raise RuntimeError("SUSPECT REMOTE ERROR: training metadata communication failed") from error

    def mirror_metadata(self, step: int) -> None:
        """Mirror RNG/cursor and rank-local buffers; never mirror parameter or moment values.

        Args:
            step: Exact committed step whose rank metadata must be mirrored.
        """
        entry = self.bank.get(self.bank.rank, step)
        size = torch.tensor([entry["payload"].numel()], dtype=torch.int64, device=self.device)
        sizes = [torch.empty_like(size) for _ in self.ranks]
        self._wait(dist.all_gather(sizes, size, group=self.group, async_op=True))
        lengths = [int(value.item()) for value in sizes]
        payload = entry["payload"]
        if payload.numel() < max(lengths):
            padded = torch.empty(max(lengths), dtype=torch.uint8, device=self.device)
            padded[:payload.numel()].copy_(payload)
            padded[payload.numel():].zero_()
            payload = padded
        payloads = [torch.empty_like(payload) for _ in self.ranks]
        self._wait(dist.all_gather(payloads, payload, group=self.group, async_op=True))
        buffers = [{} for _ in self.ranks]
        for name, tensor in sorted(entry["buffers"].items()):
            values = [torch.empty_like(tensor) for _ in self.ranks]
            self._wait(dist.all_gather(values, tensor.contiguous(), group=self.group, async_op=True))
            for saved, value in zip(buffers, values):
                saved[name] = value
        for owner, length, value, saved in zip(self.ranks, lengths, payloads, buffers):
            if owner != self.bank.rank:
                self.bank.install(owner, step, value[:length], saved)
        self._wait(dist.barrier(group=self.group, async_op=True))

    def _send_tensor(self, tensor: torch.Tensor, destination: int, group: Any = None) -> None:
        flat = tensor.detach().contiguous().view(-1)
        chunk = max(1, _TRANSFER_CHUNK_BYTES // tensor.element_size())
        for start in range(0, flat.numel(), chunk):
            dist.send(flat[start:start + chunk], dst=destination, group=group)

    @torch.no_grad()
    def _receive_tensor(self, tensor: torch.Tensor, source: int, group: Any = None) -> None:
        flat = torch.empty(tensor.numel(), dtype=tensor.dtype, device=self.device)
        chunk = max(1, _TRANSFER_CHUNK_BYTES // tensor.element_size())
        for start in range(0, flat.numel(), chunk):
            dist.recv(flat[start:start + chunk], src=source, group=group)
        tensor.copy_(flat.view(tensor.shape))

    def send(self, destination: int, step: int, group: Any = None) -> None:
        """Send metadata, then the donor's current device tensors, never a stored weight copy.

        Args:
            destination: Global rank of the replacement full replica.
            step: Exact committed step selected by MindIO.
            group: MindIO-selected temporary repair group, or the default group.
        """
        if destination not in self.ranks or destination == self.bank.rank:
            raise ValueError("destination is not an equivalent device replica")
        entry = self.bank.get(destination, step)
        header = {"optimizer": self.state.metadata(), "training": self.bank.metadata(destination, step),
                  "owner": destination, "step": step}
        payload = device_bytes(header, self.device)
        dist.send(torch.tensor([payload.numel()], dtype=torch.int64, device=self.device),
                  dst=destination, group=group)
        self._send_tensor(payload, destination, group)
        for _, tensor in self.state.tensor_items():
            self._send_tensor(tensor, destination, group)
        for _, tensor in sorted(entry["buffers"].items()):
            self._send_tensor(tensor, destination, group)

    def receive(self, source: int, step: int, group: Any = None) -> None:
        """Receive weights directly into device storage, rebuilding optimizer slots in place.

        Args:
            source: Global rank of the full device-state donor selected by MindIO.
            step: Exact committed step selected by MindIO.
            group: MindIO-selected temporary repair group, or the default group.
        """
        if source not in self.ranks or source == self.bank.rank:
            raise ValueError("source is not an equivalent device replica")
        length = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(length, src=source, group=group)
        if length.item() <= 0:
            raise RuntimeError("empty device repair header")
        payload = torch.empty(int(length.item()), dtype=torch.uint8, device=self.device)
        self._receive_tensor(payload, source, group)
        header = decode(payload.cpu().numpy().tobytes())
        if header["owner"] != self.bank.rank or header["step"] != step:
            raise RuntimeError("repair metadata belongs to another logical rank or step")
        self.state.prepare_receive(header["optimizer"])
        for _, tensor in self.state.tensor_items():
            self._receive_tensor(tensor, source, group)
        buffers = {name: torch.empty_like(tensor) for name, tensor in self.state.buffers.items()}
        for _, tensor in sorted(buffers.items()):
            self._receive_tensor(tensor, source, group)
        self.bank.put(step, header["training"], buffers)

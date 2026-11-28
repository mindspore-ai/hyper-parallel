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
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.high_availability.recovery_state import local_tensor

_TRANSFER_CHUNK_BYTES = 4 * 1024 * 1024
_METADATA_TIMEOUT_SECONDS = 10


def serialize_metadata(value: Any) -> bytes:
    """Serialize Python training metadata, never model weights or optimizer moments."""
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def deserialize_metadata(encoded_metadata: torch.Tensor) -> Any:
    """Decode a serialized byte tensor from a trusted peer into CPU metadata."""
    payload = encoded_metadata.cpu().numpy().tobytes()
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def metadata_to_device(value: Any, device: torch.device) -> torch.Tensor:
    """Place serialized metadata in a device byte buffer."""
    return torch.frombuffer(bytearray(serialize_metadata(value)), dtype=torch.uint8).to(device)


class DeviceMetadataBank:
    """Retain one committed metadata and buffer snapshot per rank for the live weights."""

    def __init__(self, rank: int, device: torch.device) -> None:
        """Keep local metadata unencoded alongside CPU buffer snapshots."""
        self.rank = rank
        self.device = device
        self._rank_snapshots = {}
        self._replica_snapshot = None

    def put_snapshot(self, step: int, rank_metadata: dict[str, Any], replica_state: dict[str, Any]) -> None:
        """Take ownership of an independent snapshot without copying it again.

        Args:
            step: Committed training step represented by this snapshot.
            rank_metadata: Independently captured or decoded rank metadata including CPU
                buffer snapshots. The caller must not mutate snapshot values after this call.
            replica_state: Independent same-step state shared by full replicas, no longer
                mutated by the caller after this call.
        """
        self.store_rank_snapshot(self.rank, step, rank_metadata)
        self._replica_snapshot = {"step": step, "state": replica_state}

    def store_rank_snapshot(
        self,
        owner: int,
        step: int,
        rank_metadata: dict[str, Any],
    ) -> None:
        """Replace rank metadata with owned metadata and CPU buffer snapshots."""
        self._rank_snapshots[owner] = {
            "step": step,
            "rank_metadata": rank_metadata,
        }

    def get_rank_metadata(self, owner: int, step: int) -> dict[str, Any]:
        """Return rank metadata only when the stored snapshot matches the exact owner and step."""
        rank_snapshot = self._rank_snapshots.get(owner)
        if rank_snapshot is None or rank_snapshot["step"] != step:
            raise RuntimeError(f"exact training metadata unavailable: owner={owner}, step={step}")
        return rank_snapshot["rank_metadata"]

    def get_replica_state(self, step: int) -> dict[str, Any]:
        """Return the local replica-provider snapshot for the exact committed step.

        Args:
            step: Exact committed step required for the replica providers.

        Returns:
            Read-only replica state. Restore adapters copy at load boundaries.
        """
        if self._replica_snapshot is None or self._replica_snapshot["step"] != step:
            raise RuntimeError(f"exact replica provider metadata unavailable: step={step}")
        return self._replica_snapshot["state"]


class ReplicaTransport:
    """Send the donor's live device model/main/moment tensors directly over HCCL."""

    def __init__(
        self,
        group: Any,
        bank: DeviceMetadataBank,
        state: Any,
        device: torch.device,
        mirror_metadata: bool = True,
    ) -> None:
        """Bind the exact replica collective group and persistent live device state."""
        self.group = group
        self.metadata_bank = bank
        self.state = state
        self.device = device
        self.ranks = tuple(dist.get_process_group_ranks(group))
        self._mirror_metadata = mirror_metadata

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
        if not self._mirror_metadata:
            return
        rank_metadata = self.metadata_bank.get_rank_metadata(self.metadata_bank.rank, step)
        training_metadata = {name: value for name, value in rank_metadata.items() if name != "buffers"}
        encoded_metadata = metadata_to_device(training_metadata, self.device)
        size = torch.tensor([encoded_metadata.numel()], dtype=torch.int64, device=self.device)
        sizes = [torch.empty_like(size) for _ in self.ranks]
        self._wait(dist.all_gather(sizes, size, group=self.group, async_op=True))
        lengths = [int(value.item()) for value in sizes]
        if encoded_metadata.numel() < max(lengths):
            padded = torch.empty(max(lengths), dtype=torch.uint8, device=self.device)
            padded[:encoded_metadata.numel()].copy_(encoded_metadata)
            padded[encoded_metadata.numel():].zero_()
            encoded_metadata = padded
        gathered_metadata = [torch.empty_like(encoded_metadata) for _ in self.ranks]
        self._wait(dist.all_gather(gathered_metadata, encoded_metadata, group=self.group, async_op=True))
        buffers = [{} for _ in self.ranks]
        for name, tensor in rank_metadata["buffers"].items():
            communication_tensor = tensor.to(device=self.device).contiguous()
            values = [torch.empty_like(communication_tensor) for _ in self.ranks]
            self._wait(dist.all_gather(values, communication_tensor, group=self.group, async_op=True))
            for owner, saved, value in zip(self.ranks, buffers, values):
                if owner != self.metadata_bank.rank:
                    saved[name] = value.cpu()
        for owner, length, value, saved in zip(self.ranks, lengths, gathered_metadata, buffers):
            if owner != self.metadata_bank.rank:
                peer_metadata = deserialize_metadata(value[:length])
                peer_metadata["buffers"] = saved
                self.metadata_bank.store_rank_snapshot(owner, step, peer_metadata)
        self._wait(dist.barrier(group=self.group, async_op=True))

    def _send_tensor(self, tensor: torch.Tensor, destination: int, group: Any = None) -> None:
        """Send live storage with at most one CPU-to-device staging block."""
        flat = tensor.detach().contiguous().view(-1)
        chunk = max(1, _TRANSFER_CHUNK_BYTES // tensor.element_size())
        for start in range(0, flat.numel(), chunk):
            # Host tables can exceed device memory; stage only the current block.
            dist.send(
                flat[start:start + chunk].to(device=self.device),
                dst=destination,
                group=group,
            )

    @torch.no_grad()
    def _receive_tensor(self, tensor: torch.Tensor, source: int, group: Any = None) -> None:
        """Receive bounded device blocks and write them into the original destination storage."""
        target = tensor if tensor.is_contiguous() else torch.empty_like(tensor, memory_format=torch.contiguous_format)
        flat = target.view(-1)
        chunk = max(1, _TRANSFER_CHUNK_BYTES // tensor.element_size())
        staging = (torch.empty(min(chunk, flat.numel()), dtype=tensor.dtype, device=self.device)
                   if tensor.device != self.device else None)
        for start in range(0, flat.numel(), chunk):
            destination = flat[start:start + chunk]
            block = destination if staging is None else staging[:destination.numel()]
            dist.recv(block, src=source, group=group)
            if staging is not None:
                destination.copy_(block)
        if target is not tensor:
            tensor.copy_(target)

    def send(self, destination: int, step: int, group: Any = None) -> None:
        """Send metadata, then the donor's current device tensors, never a stored weight copy.

        Args:
            destination: Global rank of the replacement full replica.
            step: Exact committed step selected by MindIO.
            group: MindIO-selected temporary repair group, or the default group.
        """
        if destination not in self.ranks or destination == self.metadata_bank.rank:
            raise ValueError("destination is not an equivalent device replica")
        owner = destination if self._mirror_metadata else self.metadata_bank.rank
        rank_metadata = self.metadata_bank.get_rank_metadata(owner, step)
        training_metadata = {
            "state": {
                "rank_state": rank_metadata["rank_state"],
                "replica_state": self.metadata_bank.get_replica_state(step),
            },
            "extra": rank_metadata["extra"],
        }
        header = {
            "optimizer": self.state.metadata(),
            "training": training_metadata,
            "owner": destination,
            "step": step
        }
        encoded_metadata = metadata_to_device(header, self.device)
        dist.send(torch.tensor([encoded_metadata.numel()], dtype=torch.int64, device=self.device),
                  dst=destination, group=group)
        self._send_tensor(encoded_metadata, destination, group)
        for _, tensor in self.state.tensor_items():
            self._send_tensor(tensor, destination, group)
        for tensor in rank_metadata["buffers"].values():
            self._send_tensor(tensor, destination, group)

    def receive(self, source: int, step: int, group: Any = None) -> None:
        """Receive weights directly into device storage, rebuilding optimizer slots in place.

        Args:
            source: Global rank of the full device-state donor selected by MindIO.
            step: Exact committed step selected by MindIO.
            group: MindIO-selected temporary repair group, or the default group.
        """
        if source not in self.ranks or source == self.metadata_bank.rank:
            raise ValueError("source is not an equivalent device replica")
        length = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(length, src=source, group=group)
        if length.item() <= 0:
            raise RuntimeError("empty device repair header")
        encoded_metadata = torch.empty(int(length.item()), dtype=torch.uint8, device=self.device)
        self._receive_tensor(encoded_metadata, source, group)
        header = deserialize_metadata(encoded_metadata)
        if header["owner"] != self.metadata_bank.rank or header["step"] != step:
            raise RuntimeError("repair metadata belongs to another logical rank or step")
        self.state.prepare_receive(header["optimizer"])
        for _, tensor in self.state.tensor_items():
            self._receive_tensor(tensor, source, group)
        buffers = {
            name: torch.empty_like(local_tensor(buffer), device="cpu")
            for name, buffer in self.state.buffers.items()
        }
        for tensor in buffers.values():
            self._receive_tensor(tensor, source, group)
        training_metadata = header["training"]
        training_state = training_metadata["state"]
        rank_metadata = {
            "rank_state": training_state["rank_state"],
            "extra": training_metadata["extra"],
            "buffers": buffers,
        }
        self.metadata_bank.put_snapshot(step, rank_metadata, training_state["replica_state"])

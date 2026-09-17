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
"""CPU-only communication queues and an independent affine-scan oracle."""
from queue import Queue
from threading import Lock
from types import SimpleNamespace

import torch


class Mailbox:
    """Replace P2P transport with CPU mailboxes, preserving payload and ordering."""

    def __init__(self, size: int) -> None:
        """Create per-peer FIFO queues and per-collective epochs."""
        self.queues = {(src, dst): Queue() for src in range(size) for dst in range(size)}
        self.packets = []
        self.collective_queues = {}
        self.collective_counts = {}
        self.lock = Lock()

    def isend(self, tensor: torch.Tensor, dst: int, group: int) -> SimpleNamespace:
        """Publish a copy so the test catches missing or reordered messages.

        Args:
            tensor: CPU payload or destination buffer.
            dst: Global destination rank.
            group: Existing process group or CPU mock group metadata.
        """
        self.packets.append(tuple(tensor.shape))
        rank = group if isinstance(group, int) else group[0]
        self.queues[rank, dst].put(tensor.clone())
        return SimpleNamespace(wait=lambda: None)

    def irecv(self, tensor: torch.Tensor, src: int, group: int) -> SimpleNamespace:
        """Defer reading until wait, as required by the communication contract.

        Args:
            tensor: CPU payload or destination buffer.
            src: Global source rank.
            group: Existing process group or CPU mock group metadata.
        """
        rank = group if isinstance(group, int) else group[0]
        return SimpleNamespace(wait=lambda: tensor.copy_(self.queues[src, rank].get(timeout=20)))

    def _collective(self, group):
        rank, members = group
        with self.lock:
            epoch = self.collective_counts.get((rank, members), 0)
            self.collective_counts[rank, members] = epoch + 1
            for src in members:
                for dst in members:
                    self.collective_queues.setdefault((members, epoch, src, dst), Queue())
        return rank, members, epoch

    def all_gather_into_tensor(self, output: torch.Tensor, tensor: torch.Tensor, group: tuple) -> None:
        """Deliver each rank's payload in communicator order.

        Args:
            output: Destination CPU buffer.
            tensor: Local CPU payload.
            group: Rank and ordered global members.
        """
        rank, members, epoch = self._collective(group)
        for dst in members:
            self.collective_queues[members, epoch, rank, dst].put(tensor.clone())
        output.copy_(torch.cat([self.collective_queues[members, epoch, src, rank].get(timeout=20)
                               for src in members], dim=0))

    def broadcast(self, tensor: torch.Tensor, src: int, group: tuple) -> torch.Tensor:
        """Deliver the leader boundary in the same collective order on every rank.

        Args:
            tensor: CPU payload or destination buffer.
            src: Global source rank.
            group: Existing process group or CPU mock group metadata.
        """
        rank, members, epoch = self._collective(group)
        if rank == src:
            for dst in members:
                if dst != src:
                    self.collective_queues[members, epoch, src, dst].put(tensor.clone())
        else:
            tensor.copy_(self.collective_queues[members, epoch, src, rank].get(timeout=20))
        return tensor


def serial_reference(states: list[torch.Tensor], matrices: list[torch.Tensor],
                     gradients: list[torch.Tensor], step: int) -> tuple:
    """Use independent chronological FP64 arithmetic and autograd as the oracle.

    Args:
        states: Chronological affine offsets [P,B,H,128,128].
        matrices: Matching chronological FP32 transfer matrices.
        gradients: Local G tensors for the independent backward oracle.
        step: Invocation index used to vary S and M deterministically.
    """
    # Inputs are rounded to FP32 before protocol execution.
    serial_states = [(s * (1 + .1 * step)).double().requires_grad_() for s in states]
    serial_matrices = [(m * (1 - .03 * step)).double() for m in matrices]
    state = torch.zeros_like(serial_states[0])
    expected = []
    loss = sum(s.sum() * 0 for s in serial_states)
    for local, matrix, gradient in zip(serial_states, serial_matrices, gradients):
        expected.append(state)
        loss = loss + (state * gradient.double()).sum()
        state = matrix @ state + local
    adjoints = torch.autograd.grad(loss, serial_states)
    return expected, adjoints

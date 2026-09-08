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
"""Shared fakes for ``tests/ut/auto_models/distributed`` (Gate-1).

The fakes here never create a process group: ``FakeDeviceMesh`` records mesh
shape/dim/group requests, ``CollectiveRecorder`` stands in for the collective
entry points, and ``FakeDTensor`` records layout transitions. Production code
bound to the real classes is patched per-module with these fakes (the same
pattern as ``tests/ut/auto_models/losses/test_loss_parallel.py``).
"""

import torch


class FakeSubMesh:
    """Metadata-only mesh: dims, ranks and group requests are recorded only."""

    def __init__(self, device_type, shape, mesh_dim_names, rank_list=None, parent=None):
        self.device_type = device_type
        self.mesh_shape = tuple(shape)
        self.mesh_dim_names = tuple(mesh_dim_names)
        size = 1
        for extent in self.mesh_shape:
            size *= extent
        self.rank_list = (
            tuple(rank_list) if rank_list is not None else tuple(range(size))
        )
        self.parent = parent
        self.group_requests = []

    def size(self, dim=None):
        """Total size, or the extent of one named dimension."""
        if dim is None:
            return len(self.rank_list)
        return self.mesh_shape[self.mesh_dim_names.index(dim)]

    def get_group(self, dim=None):
        """Record the group request and return a recorded stand-in handle."""
        handle = ("fake-group", dim, self.rank_list)
        self.group_requests.append(handle)
        return handle

    def __getitem__(self, dims):
        """Sub-slice by dim name(s), row-major like ``DeviceMesh``."""
        names = (dims,) if isinstance(dims, str) else tuple(dims)
        axes = [self.mesh_dim_names.index(name) for name in names]
        shape = tuple(self.mesh_shape[axis] for axis in axes)
        # Fake metadata only: the rank list is not re-derived per coordinate.
        return FakeSubMesh(self.device_type, shape, names, parent=self)


class FakeDeviceMesh(FakeSubMesh):
    """Top-level fake mesh (``mesh_dim_names=()`` means no active axis)."""

    def __init__(self, shape=(), mesh_dim_names=(), device_type="cpu", rank_list=None):
        super().__init__(device_type, shape, mesh_dim_names, rank_list=rank_list)


class CollectiveRecorder:
    """Stand-in collectives: record op/shape/group and return canned tensors.

    Every entry point returns its input unchanged, so single-rank reference
    numerics flow through while the call sequence stays assertable.
    """

    def __init__(self):
        self.records = []

    def _record(self, op, tensor, group, async_op):
        self.records.append(
            {
                "op": op,
                "shape": tuple(tensor.shape) if torch.is_tensor(tensor) else None,
                "group": group,
                "async_op": async_op,
            }
        )
        return tensor

    def all_reduce(self, tensor, group=None, async_op=False):
        return self._record("all_reduce", tensor, group, async_op)

    def all_gather(self, tensor, group=None, async_op=False):
        return self._record("all_gather", tensor, group, async_op)

    def reduce_scatter(self, tensor, group=None, async_op=False):
        return self._record("reduce_scatter", tensor, group, async_op)

    def all_to_all(self, tensor, group=None, async_op=False):
        return self._record("all_to_all", tensor, group, async_op)

    def broadcast(self, tensor, group=None, async_op=False):
        return self._record("broadcast", tensor, group, async_op)

    def assert_ops(self, expected):
        """Assert the exact recorded op sequence."""
        actual = [record["op"] for record in self.records]
        assert actual == list(expected), f"collective ops {actual} != {list(expected)}"


class FakeDTensor:
    """Local-tensor wrapper recording placements and redistributions.

    Not a ``torch.Tensor`` subclass: production modules under test patch their
    own ``DTensor`` reference with this class, so ``isinstance`` checks keep
    working without touching the real dispatch machinery.
    """

    def __init__(self, local_tensor, mesh, placements, global_shape=None):
        self.local_tensor = local_tensor
        self.device_mesh = mesh
        self.placements = tuple(placements)
        self.global_shape = (
            tuple(global_shape) if global_shape is not None else tuple(local_tensor.shape)
        )
        self.redistribute_log = []

    @property
    def shape(self):
        return torch.Size(self.global_shape)

    def to_local(self):
        return self.local_tensor

    @classmethod
    def from_local(cls, local_tensor, mesh, placements, global_shape=None):
        return cls(local_tensor, mesh, placements, global_shape=global_shape)

    def redistribute(self, mesh, placements):
        """Record the requested transition and return the retargeted fake."""
        self.redistribute_log.append((mesh, tuple(placements)))
        return FakeDTensor(self.local_tensor, mesh, placements, self.global_shape)

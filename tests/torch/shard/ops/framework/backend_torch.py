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
"""Torch backends for shard-ops cases — Ascend (hccl) and CPU (gloo)."""
from typing import Any, Tuple

import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.shard_ops.framework.backend import ShardBackend
from tests.shard_ops.framework.case_spec import CompareSpec, InputSpec
from tests.shard_ops.framework.utils import build_numpy
from tests.torch.shard.utils import local_to_global as _local_to_global


_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _to_torch_dtype(name: str) -> torch.dtype:
    if name not in _TORCH_DTYPE:
        raise ValueError(f"unsupported torch dtype name: {name!r}")
    return _TORCH_DTYPE[name]


class _TorchBackendBase(ShardBackend):
    """Shared Torch backend — Ascend GPU (hccl) and CPU (gloo)."""
    framework = "torch"
    device_type = "<override>"

    def __init__(self) -> None:
        """Lazy state — populated on first ``maybe_init_dist`` / mesh call."""
        self._dist_inited = False
        self._mesh_cache: dict = {}

    # ---- shared ----
    def make_tensor(self, spec: InputSpec) -> torch.Tensor:
        """Build numpy array per spec and move to the local device."""
        arr = build_numpy(spec)
        tensor = torch.from_numpy(arr).to(_to_torch_dtype(spec.dtype))
        return self._to_device(tensor)

    def distribute(self, full_tensor: torch.Tensor, mesh: Any,
                   placements: Tuple[Any, ...]) -> Any:
        """Shard ``full_tensor`` across ``mesh`` per ``placements``."""
        plist = list(placements) if placements else [Replicate()] * mesh.ndim
        return distribute_tensor(full_tensor, mesh, plist)

    def local_to_global(self, dist_tensor: Any) -> torch.Tensor:
        """Gather a DTensor back to a full tensor."""
        return _local_to_global(dist_tensor)

    def get_or_init_mesh(self, shape: Tuple[int, ...],
                         names: Tuple[str, ...]) -> Any:
        """Cache device mesh keyed by ``(shape, names)``."""
        key = (shape, names)
        if key not in self._mesh_cache:
            self._mesh_cache[key] = init_device_mesh(
                device_type=self.device_type,
                mesh_shape=shape,
                mesh_dim_names=names,
            )
        return self._mesh_cache[key]

    def recover_after_failure(self) -> bool:
        """Probe the comm group with a short barrier.

        Returns False (group considered broken) when the barrier raises
        or times out — the next case will be skipped instead of
        deadlocking on the next collective. A clean barrier means the
        group can still service subsequent cases.
        """
        if not dist.is_initialized():
            return True
        try:
            dist.barrier()
            return True
        except Exception:  # pylint: disable=W0703
            return False

    def assert_close(self, expected: Any, actual: Any,
                     spec: CompareSpec) -> None:
        """Compare full tensors per ``spec`` and raise on mismatch."""
        if spec.kind == "shape":
            if tuple(expected.shape) != tuple(actual.shape) or expected.dtype != actual.dtype:
                raise AssertionError(
                    f"shape/dtype mismatch: expected {tuple(expected.shape)}/{expected.dtype}, "
                    f"actual {tuple(actual.shape)}/{actual.dtype}"
                )
            return
        if spec.kind == "equal":
            if not torch.equal(expected, actual):
                raise AssertionError(
                    f"torch.equal mismatch: expected.shape={tuple(expected.shape)}, "
                    f"actual.shape={tuple(actual.shape)}"
                )
            return
        # allclose
        if not torch.allclose(expected, actual, rtol=spec.rtol, atol=spec.atol):
            diff = (expected - actual).abs()
            raise AssertionError(
                f"torch.allclose failed: max_abs_diff={diff.max().item():.6g}, "
                f"rtol={spec.rtol}, atol={spec.atol}"
            )

    # ---- platform hooks ----
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TorchHcclBackend(_TorchBackendBase):
    """Torch Ascend NPU backend (hccl)."""
    device_type = "npu"

    def maybe_init_dist(self) -> None:
        """Initialise hccl + bind the local NPU device once per process."""
        if self._dist_inited:
            return
        # pylint: disable=C0415
        from tests.torch.utils import init_dist
        init_dist()
        self._dist_inited = True

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.npu()


class TorchGlooBackend(_TorchBackendBase):
    """Torch CPU backend (gloo). Uses standard torch ops only — no torch_npu."""
    device_type = "cpu"

    def maybe_init_dist(self) -> None:
        """Initialise gloo process group once per process."""
        if self._dist_inited:
            return
        # pylint: disable=C0415
        from tests.torch.utils import init_dist_gloo
        init_dist_gloo()
        self._dist_inited = True

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

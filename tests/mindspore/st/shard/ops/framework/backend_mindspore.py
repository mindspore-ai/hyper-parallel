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
"""MindSpore Ascend backend for shard-ops cases."""
from typing import Any, Tuple

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from tests.shard_ops.framework.backend import ShardBackend
from tests.shard_ops.framework.case_spec import CompareSpec, InputSpec
from tests.shard_ops.framework.utils import build_numpy


_MS_DTYPE = {
    "float32": ms.float32,
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "float64": ms.float64,
    "int32": ms.int32,
    "int64": ms.int64,
    "bool": ms.bool_,
}


def _to_ms_dtype(name: str):
    if name not in _MS_DTYPE:
        raise ValueError(f"unsupported mindspore dtype name: {name!r}")
    return _MS_DTYPE[name]


def _as_numpy(t):
    if isinstance(t, Tensor):
        # numpy cannot promote bfloat16 (ml_dtypes) against Python floats in
        # allclose; cast to float32 on-device first.
        if t.dtype == ms.bfloat16:
            t = t.astype(ms.float32)
        return t.asnumpy()
    if isinstance(t, np.ndarray):
        return t
    return np.asarray(t)


class MindSporeAscendBackend(ShardBackend):
    """MindSpore Ascend backend for shard-ops distributed tests."""
    framework = "mindspore"
    device_type = "npu"

    def __init__(self) -> None:
        """Lazy state — populated on first ``maybe_init_dist`` / mesh call."""
        self._dist_inited = False
        self._mesh_cache: dict = {}

    def maybe_init_dist(self) -> None:
        """Initialise the MS Ascend runtime + communication once per process."""
        if self._dist_inited:
            return
        ms.set_device("Ascend")
        D.init()
        self._dist_inited = True

    def get_or_init_mesh(self, shape: Tuple[int, ...],
                         names: Tuple[str, ...]) -> Any:
        """Cache device mesh keyed by ``(shape, names)``."""
        key = (shape, names)
        if key not in self._mesh_cache:
            self._mesh_cache[key] = init_device_mesh(
                device_type="npu",
                mesh_shape=shape,
                mesh_dim_names=names,
            )
        return self._mesh_cache[key]

    def make_tensor(self, spec: InputSpec) -> Tensor:
        """Build numpy array per spec and wrap as MS Tensor."""
        arr = build_numpy(spec)
        return Tensor(arr, dtype=_to_ms_dtype(spec.dtype))

    def distribute(self, full_tensor: Any, mesh: Any,
                   placements: Tuple[Any, ...]) -> Any:
        """Shard ``full_tensor`` across ``mesh`` per ``placements``."""
        plist = list(placements) if placements else [Replicate()] * mesh.ndim
        return distribute_tensor(full_tensor, mesh, plist)

    def local_to_global(self, dist_tensor: Any) -> Any:
        """Gather a DTensor back to a full MS Tensor."""
        return dist_tensor.full_tensor()

    def recover_after_failure(self) -> bool:
        """Probe the comm group with a tiny all-reduce as a heartbeat.

        Returns False if the collective raises — the next case will be
        skipped instead of hanging on its own collective. MS does not
        expose a general ``barrier`` API; a 1-element all-reduce is the
        cheapest equivalent.
        """
        try:
            ms.communication.comm_func.all_reduce(Tensor([0], dtype=ms.float32))
            return True
        except Exception:  # pylint: disable=W0703
            return False

    def assert_close(self, expected: Any, actual: Any,
                     spec: CompareSpec) -> None:
        """Compare via numpy after asnumpy() conversion."""
        a = _as_numpy(expected)
        b = _as_numpy(actual)
        if a.shape != b.shape:
            raise AssertionError(
                f"shape mismatch: expected={a.shape}, actual={b.shape}"
            )
        if spec.kind == "shape":  # shape (checked above) and dtype only, no values
            if a.dtype != b.dtype:
                raise AssertionError(f"dtype mismatch: expected={a.dtype}, actual={b.dtype}")
            return
        if spec.kind == "equal":
            if not np.array_equal(a, b):
                raise AssertionError("np.array_equal mismatch")
            return
        if not np.allclose(a, b, rtol=spec.rtol, atol=spec.atol):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            raise AssertionError(
                f"np.allclose failed: max_abs_diff={diff.max():.6g}, "
                f"rtol={spec.rtol}, atol={spec.atol}"
            )

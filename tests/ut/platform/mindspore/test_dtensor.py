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
"""Unit tests for MindSpore DTensorBase.to() method."""

from types import SimpleNamespace

import mindspore as ms

from hyper_parallel.platform.mindspore.dtensor import DTensorBase
from hyper_parallel.core.dtensor.placement_types import Shard


class _FakeDTensor:
    """Fake DTensor that records constructor calls for assertion.

    Avoids Tensor._make_subclass which requires NPU hardware.
    """

    constructor_calls = []

    def __init__(self, local_tensor=None, device_mesh=None, placements=None):
        self._local_tensor = local_tensor
        self._device_mesh = device_mesh
        self._placements = placements
        self._layout = None
        _FakeDTensor.constructor_calls.append({
            "local_tensor": local_tensor,
            "device_mesh": device_mesh,
            "placements": placements,
        })


def _make_fake_dtensor(local_tensor=None, mesh=None, placements=None, alias_placements=None):
    """Build a lightweight fake DTensorBase for unit testing."""
    if local_tensor is None:
        local_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
    if placements is None:
        placements = [Shard(0)]

    fake_layout = SimpleNamespace(
        alias_placements=alias_placements if alias_placements else placements
    )
    fake = _FakeDTensor.__new__(_FakeDTensor)
    fake._local_tensor = local_tensor
    fake._layout = fake_layout
    fake._placements = placements
    fake._device_mesh = mesh or "fake_mesh"
    return fake


def test_to_dtype_converts_local_tensor():
    """to(dtype) should delegate to _local_tensor.to and return a new DTensor."""
    _FakeDTensor.constructor_calls.clear()
    local_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
    fake = _make_fake_dtensor(local_tensor)

    DTensorBase.to(fake, ms.float16)

    assert len(_FakeDTensor.constructor_calls) == 1, (
        f"Expected 1 constructor call, got {len(_FakeDTensor.constructor_calls)}"
    )
    call = _FakeDTensor.constructor_calls[0]
    assert call["local_tensor"].dtype == ms.float16, (
        f"Expected dtype {ms.float16}, got {call['local_tensor'].dtype}"
    )


def test_to_preserves_device_mesh():
    """to() should pass the original device_mesh to the new DTensor."""
    _FakeDTensor.constructor_calls.clear()
    fake = _make_fake_dtensor(mesh="my_mesh")

    DTensorBase.to(fake, ms.float16)

    call = _FakeDTensor.constructor_calls[0]
    assert call["device_mesh"] == "my_mesh", (
        f"Expected device_mesh 'my_mesh', got {call['device_mesh']}"
    )


def test_to_uses_alias_placements_when_layout_exists():
    """to() should prefer layout.alias_placements over _placements."""
    _FakeDTensor.constructor_calls.clear()
    alias_p = [Shard(1)]
    fake = _make_fake_dtensor(placements=[Shard(0)], alias_placements=alias_p)

    DTensorBase.to(fake, ms.float16)

    call = _FakeDTensor.constructor_calls[0]
    assert call["placements"] == alias_p, (
        f"Expected alias_placements {alias_p}, got {call['placements']}"
    )


def test_to_falls_back_to_placements_when_no_layout():
    """to() should use _placements when _layout is None."""
    _FakeDTensor.constructor_calls.clear()
    local_tensor = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
    placements = [Shard(0)]
    fake = _FakeDTensor.__new__(_FakeDTensor)
    fake._local_tensor = local_tensor
    fake._layout = None
    fake._placements = placements
    fake._device_mesh = "fake_mesh"

    DTensorBase.to(fake, ms.float16)

    call = _FakeDTensor.constructor_calls[0]
    assert call["placements"] == placements, (
        f"Expected placements {placements}, got {call['placements']}"
    )

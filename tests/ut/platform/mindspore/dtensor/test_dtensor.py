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
"""UT for :mod:`hyper_parallel.platform.mindspore.dtensor`.

Covers the consolidated ``DTensorBase.__new__`` flow that runs without an
Ascend device — input-argument validation, the ``has_init`` ``init_device``
side effect, and wrapping an existing DTensor.

Device-handling on real Ascend tensors (the optimisation that avoids
re-wrapping an already-on-Ascend tensor) is covered by the ST suite.
"""
from copy import copy
from unittest.mock import patch

import pytest

pytest.importorskip("mindspore")

import mindspore as ms
from mindspore import Parameter
from mindspore.common.initializer import initializer

from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.platform.mindspore.dtensor import DTensorBase


# pylint: disable=redefined-outer-name


def _stub_init_data(self, local_tensor, device_mesh, placements, layout=None, shape=None):
    """Stand-in for ``DTensor.__init_data__`` so DTensorBase can be tested in
    isolation, without going through the core ``DTensor`` subclass."""
    self._local_tensor = local_tensor
    self._device_mesh = device_mesh
    self._placements = tuple(placements) if placements is not None else None
    self._global_shape = tuple(shape) if shape is not None else None


def _stub_to_local(self):
    return self._local_tensor


def _stub_alias_placements(self):
    return self._placements


def _stub_device_mesh_get(self):
    return self._device_mesh


def _keep_on_current_device(tensor, *_args, **_kwargs):
    """Keep CPU tensors local while exercising device-agnostic copy logic."""
    return tensor


def _mock_clone_on_cpu(tensor):
    """Bypass MindSpore's unavailable CPU Clone kernel in wrapper-only tests."""
    # MindSpore does not register the Clone kernel on CPU. These tests verify
    # HyperParallel's dispatch and wrapper semantics, not MindSpore's kernel.
    return tensor


# Attach the stubs for the lifetime of this test module.
DTensorBase.__init_data__ = _stub_init_data
DTensorBase.to_local = _stub_to_local
DTensorBase._alias_placements = _stub_alias_placements
DTensorBase.device_mesh = property(_stub_device_mesh_get)


@pytest.fixture
def fake_mesh():
    return object()


@pytest.fixture
def fake_placements():
    return [object()]


# ----------------------------------------------------------------------
# Input validation: synchronous ValueError on missing required args
# ----------------------------------------------------------------------

def test_none_local_tensor_raises(fake_mesh, fake_placements):
    with pytest.raises(ValueError, match="local_tensor"):
        DTensorBase(None, fake_mesh, fake_placements)


def test_none_device_mesh_raises(fake_placements):
    local = ms.Tensor([[1.0, 2.0]], dtype=ms.float32)
    with pytest.raises(ValueError, match="device_mesh"):
        DTensorBase(local, None, fake_placements)


def test_none_placements_raises(fake_mesh):
    local = ms.Tensor([[1.0, 2.0]], dtype=ms.float32)
    with pytest.raises(ValueError, match="placements"):
        DTensorBase(local, fake_mesh, None)


# ----------------------------------------------------------------------
# has_init initializer: sets init_device, identity preserved
# ----------------------------------------------------------------------

def test_has_init_initializer_sets_init_device(fake_mesh, fake_placements):
    """``has_init`` tensors must have ``init_device`` set to Ascend without
    triggering an actual device move (the optimisation's "no .to()" branch)."""
    init_t = initializer("zeros", (4, 4), ms.float32)
    assert init_t.has_init

    dt = DTensorBase(init_t, fake_mesh, fake_placements)

    assert dt._local_tensor is init_t
    assert dt._local_tensor.init_device == "Ascend"


# ----------------------------------------------------------------------
# Wrapping an existing DTensorBase: reuses mesh / placements / storage
# ----------------------------------------------------------------------

def test_wrap_existing_dtensor_reuses_mesh_and_placements(fake_mesh, fake_placements):
    """Constructing from another DTensorBase should preserve mesh / placements
    and share the inner ``_local_tensor`` (the consolidated "wrap" branch).

    Uses a ``has_init`` initializer so the constructor stays inside the
    device-agnostic fast path (no Ascend runtime needed)."""
    # `_alias_placements` is invoked on the wrapped src; stub it with a
    # minimal layout that exposes the attribute the wrapping branch reads.
    class _Layout:
        alias_placements = tuple(fake_placements)

    init_t = initializer("zeros", (4, 4), ms.float32)
    src = DTensorBase(init_t, fake_mesh, fake_placements, shape=(8, 4))
    src._layout = _Layout()

    wrapped = DTensorBase(src)

    assert wrapped._device_mesh is src._device_mesh
    assert wrapped._placements == src._placements
    assert wrapped._global_shape == src._global_shape
    assert wrapped._local_tensor is src._local_tensor


@patch.object(ms.Tensor, "clone", _mock_clone_on_cpu)
def test_copy_dtensor_with_dispatch_disabled_returns_local_tensor(fake_mesh, fake_placements):
    """SkipDTensorDispatch must make copy return a plain local Tensor."""
    local = ms.Tensor([[1.0, 2.0]], dtype=ms.float32)
    with patch.object(ms.Tensor, "to", _keep_on_current_device):
        src = DTensorBase(local, fake_mesh, fake_placements, shape=(2, 2))

        with SkipDTensorDispatch():
            copied = copy(src)

    assert isinstance(copied, ms.Tensor)
    assert not isinstance(copied, DTensorBase)
    assert copied.shape == src._local_tensor.shape
    assert copied.dtype == src._local_tensor.dtype


@patch.object(ms.Tensor, "clone", _mock_clone_on_cpu)
def test_copy_parameter_dtensor_with_dispatch_disabled_returns_parameter(fake_mesh, fake_placements):
    """SkipDTensorDispatch must unwrap a ParameterDTensor to Parameter."""
    local = ms.Tensor([[1.0, 2.0]], dtype=ms.float32)
    with patch.object(ms.Tensor, "to", _keep_on_current_device):
        src = DTensorBase(local, fake_mesh, fake_placements, shape=(2, 2))
        src = Parameter(src, name="weight", requires_grad=False)

        with SkipDTensorDispatch():
            copied = copy(src)

    assert isinstance(copied, Parameter)
    assert not isinstance(copied, DTensorBase)
    assert copied.name == src.name
    assert copied.requires_grad is False
    assert copied.shape == src._local_tensor.shape


@patch.object(ms.Tensor, "clone", _mock_clone_on_cpu)
def test_copy_dtensor_with_dispatch_enabled_preserves_dtensor(fake_mesh, fake_placements):
    """The existing DTensor copy behavior must remain unchanged."""
    class _Layout:
        mesh = fake_mesh
        alias_placements = tuple(fake_placements)

    local = ms.Tensor([[1.0, 2.0]], dtype=ms.float32)
    with patch.object(ms.Tensor, "to", _keep_on_current_device):
        src = DTensorBase(local, fake_mesh, fake_placements, shape=(2, 2))
        src._layout = _Layout()

        copied = copy(src)

    assert isinstance(copied, DTensorBase)
    assert copied._device_mesh is src._device_mesh
    assert copied._placements == src._placements
    assert copied._global_shape == src._global_shape


def test_copy_uninitialized_dtensor_raises(fake_mesh, fake_placements):
    """Copying a lazy initializer must fail instead of creating another initializer."""
    init_t = initializer("zeros", (4, 4), ms.float32)
    src = DTensorBase(init_t, fake_mesh, fake_placements, shape=(8, 4))

    with pytest.raises(RuntimeError, match="uninitialized local tensor"):
        copy(src)

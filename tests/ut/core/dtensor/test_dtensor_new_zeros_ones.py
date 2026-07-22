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
"""Unit tests for DTensor.new_zeros() and DTensor.new_ones() — platform-agnostic."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.platform.platform import PlatformType

# Bypass C extension by pulling the Python function from DTensor.__dict__.
_new_zeros_fn = DTensor.__dict__["new_zeros"]
_new_ones_fn = DTensor.__dict__["new_ones"]
_new_const_tensor_op_fn = DTensor.__dict__["_new_const_tensor_op"]
_validate_factory_device_fn = DTensor.__dict__["_validate_factory_device"]


def _make_mock_dtensor(mesh=None, placements=None, is_partial=False):
    """Build a mock DTensor for unit testing without hardware."""
    if placements is None:
        placements = [Shard(0)]
    if mesh is None:
        mesh = Mock(name="device_mesh")
        mesh.ndim = 2

    mock_local = Mock(name="local_tensor")
    mock_local.device = torch.device("cpu")

    fake_layout = SimpleNamespace(
        alias_placements=placements,
        is_partial=lambda: is_partial,
    )

    class _Recorder:
        _new_const_tensor_op = _new_const_tensor_op_fn
        _validate_factory_device = _validate_factory_device_fn

    fake = _Recorder.__new__(_Recorder)
    fake._local_tensor = mock_local
    fake._layout = fake_layout
    fake._placements = placements
    fake._device_mesh = mesh

    return fake, mock_local


# =========================================================================
# new_zeros  tests
# =========================================================================

@patch.object(DTensor, "from_local")
def test_new_zeros_tuple_size(mock_from_local):
    """new_zeros((3, 4)) — tuple size."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4))
    mock_local.new_zeros.assert_called_once_with((3, 4))


@patch.object(DTensor, "from_local")
def test_new_zeros_int_size(mock_from_local):
    """new_zeros(5) — bare int normalised to (5,)."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 1
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, 5)
    mock_local.new_zeros.assert_called_once_with((5,))


@patch.object(DTensor, "from_local")
def test_new_zeros_replicate_output(mock_from_local):
    """Sharded self → output placements all-Replicate."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, _ = _make_mock_dtensor(mesh=mesh, placements=[Shard(0)])
    _new_zeros_fn(fake, (3, 4))
    placements = mock_from_local.call_args[0][2]
    assert len(placements) == 2
    assert all(isinstance(p, Replicate) for p in placements)


@patch.object(DTensor, "from_local")
def test_new_zeros_scalar_output(mock_from_local):
    """new_zeros(()) → 0-d tensor."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 1
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, ())
    mock_local.new_zeros.assert_called_once_with(())


@patch.object(DTensor, "from_local")
def test_new_zeros_zero_length_output(mock_from_local):
    """new_zeros((0, 16)) → zero-length output."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (0, 16))
    mock_local.new_zeros.assert_called_once_with((0, 16))


@patch.object(DTensor, "from_local")
def test_new_zeros_dtype_forwarded(mock_from_local):
    """dtype=float64 reaches local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4), dtype=torch.float64)
    mock_local.new_zeros.assert_called_once_with((3, 4), dtype=torch.float64)


@patch.object(DTensor, "from_local")
def test_new_zeros_dtype_none_not_forwarded(mock_from_local):
    """dtype=None → not included in local_kwargs."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4))
    mock_local.new_zeros.assert_called_once_with((3, 4))


@patch.object(DTensor, "from_local")
def test_new_zeros_partial_input(mock_from_local):
    """Partial self — no error."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh, is_partial=True)
    _new_zeros_fn(fake, (3, 4))
    mock_local.new_zeros.assert_called_once()


@patch.object(DTensor, "from_local")
def test_new_zeros_layout_forwarded(mock_from_local):
    """layout=torch.sparse_coo forwarded to local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4), layout=torch.sparse_coo)
    mock_local.new_zeros.assert_called_once_with(
        (3, 4), layout=torch.sparse_coo,
    )


@patch.object(DTensor, "from_local")
def test_new_zeros_pin_memory_forwarded(mock_from_local):
    """pin_memory=True forwarded to local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4), pin_memory=True)
    mock_local.new_zeros.assert_called_once_with(
        (3, 4), pin_memory=True,
    )


@patch.object(DTensor, "from_local")
def test_new_zeros_device_mismatch(mock_from_local):
    """Explicit different device → ValueError."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    mock_local.device = torch.device("cpu")
    try:
        _new_zeros_fn(fake, (3, 4), device=torch.device("cuda:1"))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "requires device to match" in str(e)


@patch.object(DTensor, "from_local")
def test_new_zeros_device_match(mock_from_local):
    """Explicit same torch.device — forwarded."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    fake_dev = torch.device("cpu")
    mock_local.device = fake_dev
    _new_zeros_fn(fake, (3, 4), device=fake_dev)
    mock_local.new_zeros.assert_called_once_with((3, 4), device=fake_dev)


@patch.object(DTensor, "from_local")
def test_new_zeros_device_str_match(mock_from_local):
    """Device as string matching local — forwarded."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    mock_local.device = torch.device("cpu")
    _new_zeros_fn(fake, (3, 4), device="cpu")
    mock_local.new_zeros.assert_called_once_with(
        (3, 4), device=torch.device("cpu"),
    )


@patch.object(DTensor, "from_local")
def test_new_zeros_unindexed_device_uses_local_device(mock_from_local):
    """An unindexed device is pinned to the DTensor's local device."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    mock_local.device = torch.device("cuda:1")
    _new_zeros_fn(fake, (3, 4), device="cuda")
    mock_local.new_zeros.assert_called_once_with(
        (3, 4), device=torch.device("cuda:1"),
    )


@patch.object(DTensor, "from_local")
def test_new_zeros_requires_grad_forwarded(mock_from_local):
    """requires_grad=True forwarded to local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4), requires_grad=True)
    mock_local.new_zeros.assert_called_once_with(
        (3, 4), requires_grad=True,
    )


# =========================================================================
# MindSpore  tests
# =========================================================================

@patch.object(DTensor, "from_local")
@patch("hyper_parallel.core.dtensor.dtensor.platform")
def test_new_zeros_mindspore_only_dtype(mock_platform, mock_from_local):
    """MindSpore: only dtype is forwarded."""
    mock_platform.platform_type = PlatformType.MINDSPORE
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_zeros_fn(fake, (3, 4), dtype=torch.float16)
    mock_local.new_zeros.assert_called_once_with((3, 4), dtype=torch.float16)
    mock_from_local.assert_called_once()


@patch("hyper_parallel.core.dtensor.dtensor.platform")
def test_new_zeros_mindspore_no_dtype(mock_platform):
    """MindSpore: no kwargs at all when no params set."""
    mock_platform.platform_type = PlatformType.MINDSPORE
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    with patch.object(DTensor, "from_local"):
        _new_zeros_fn(fake, (3, 4))
    mock_local.new_zeros.assert_called_once_with((3, 4))


@patch("hyper_parallel.core.dtensor.dtensor.platform")
def test_new_zeros_mindspore_rejects_device(mock_platform):
    """MindSpore: device=... → ValueError."""
    mock_platform.platform_type = PlatformType.MINDSPORE
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, _ = _make_mock_dtensor(mesh=mesh)
    try:
        _new_zeros_fn(fake, (3, 4), device=torch.device("cpu"))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "only supports size and dtype" in str(e)


@patch("hyper_parallel.core.dtensor.dtensor.platform")
def test_new_zeros_mindspore_rejects_requires_grad(mock_platform):
    """MindSpore: requires_grad=True → ValueError."""
    mock_platform.platform_type = PlatformType.MINDSPORE
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, _ = _make_mock_dtensor(mesh=mesh)
    try:
        _new_zeros_fn(fake, (3, 4), requires_grad=True)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "only supports size and dtype" in str(e)


@patch("hyper_parallel.core.dtensor.dtensor.platform")
def test_new_zeros_mindspore_rejects_layout(mock_platform):
    """MindSpore: layout=... → ValueError."""
    mock_platform.platform_type = PlatformType.MINDSPORE
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, _ = _make_mock_dtensor(mesh=mesh)
    try:
        _new_zeros_fn(fake, (3, 4), layout=torch.strided)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "only supports size and dtype" in str(e)


@patch("hyper_parallel.core.dtensor.dtensor.platform")
def test_new_zeros_mindspore_rejects_pin_memory(mock_platform):
    """MindSpore: pin_memory=True → ValueError."""
    mock_platform.platform_type = PlatformType.MINDSPORE
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, _ = _make_mock_dtensor(mesh=mesh)
    try:
        _new_zeros_fn(fake, (3, 4), pin_memory=True)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "only supports size and dtype" in str(e)


# =========================================================================
# new_ones  tests
# =========================================================================

@patch.object(DTensor, "from_local")
def test_new_ones_tuple_size(mock_from_local):
    """new_ones((3, 4)) — tuple size."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_ones_fn(fake, (3, 4))
    mock_local.new_ones.assert_called_once_with((3, 4))


@patch.object(DTensor, "from_local")
def test_new_ones_int_size(mock_from_local):
    """new_ones(3) — bare int normalised to (3,)."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 1
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_ones_fn(fake, 3)
    mock_local.new_ones.assert_called_once_with((3,))


@patch.object(DTensor, "from_local")
def test_new_ones_replicate_output(mock_from_local):
    """Sharded self → output placements all-Replicate."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, _ = _make_mock_dtensor(mesh=mesh, placements=[Shard(0), Shard(1)])
    _new_ones_fn(fake, (3, 4))
    placements = mock_from_local.call_args[0][2]
    assert len(placements) == 2
    assert all(isinstance(p, Replicate) for p in placements)


@patch.object(DTensor, "from_local")
def test_new_ones_scalar_output(mock_from_local):
    """new_ones(()) → 0-d tensor."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 1
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_ones_fn(fake, ())
    mock_local.new_ones.assert_called_once_with(())


@patch.object(DTensor, "from_local")
def test_new_ones_dtype_forwarded(mock_from_local):
    """dtype=float64 forwarded to local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_ones_fn(fake, (3, 4), dtype=torch.float64)
    mock_local.new_ones.assert_called_once_with((3, 4), dtype=torch.float64)


@patch.object(DTensor, "from_local")
def test_new_ones_partial_input(mock_from_local):
    """Partial self — no error."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh, is_partial=True)
    _new_ones_fn(fake, (3, 4))
    mock_local.new_ones.assert_called_once()


@patch.object(DTensor, "from_local")
def test_new_ones_layout_forwarded(mock_from_local):
    """layout=torch.sparse_coo forwarded to local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_ones_fn(fake, (3, 4), layout=torch.sparse_coo)
    mock_local.new_ones.assert_called_once_with(
        (3, 4), layout=torch.sparse_coo,
    )


@patch.object(DTensor, "from_local")
def test_new_ones_pin_memory_forwarded(mock_from_local):
    """pin_memory=True forwarded to local factory."""
    mesh = Mock(name="device_mesh")
    mesh.ndim = 2
    fake, mock_local = _make_mock_dtensor(mesh=mesh)
    _new_ones_fn(fake, (3, 4), pin_memory=True)
    mock_local.new_ones.assert_called_once_with(
        (3, 4), pin_memory=True,
    )

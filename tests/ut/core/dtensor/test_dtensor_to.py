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
"""Unit tests for DTensor.to() and DTensor.float() — platform-agnostic."""

from types import SimpleNamespace
from unittest.mock import Mock

import hyper_parallel.core.dtensor.dtensor as dtensor_mod
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard

# DTensor inherits from DTensorBase which inherits from a C extension Tensor.
# On MindSpore the C-level descriptor enforces isinstance(self, Tensor).
# Bypass it by pulling the Python function directly from DTensor.__dict__.
_to_fn = DTensor.__dict__["to"]
_float_fn = DTensor.__dict__["float"]
_alias_placements_fn = DTensor.__dict__["_alias_placements"]
_from_converted_local_fn = DTensor.__dict__["_from_converted_local"]


def _make_mock_dtensor(mesh="fake_mesh", placements=None, alias_placements=None):
    """Build a mock DTensor for unit testing without hardware.

    Returns a ``SimpleNamespace`` whose attributes mimic the internal state
    set by ``DTensor.__init_data__``.  ``_local_tensor`` is a ``Mock`` so
    that ``.to()`` / ``.float()`` never touch real hardware.

    Because ``DTensor.to()`` calls ``self.__class__(...)``, the returned
    object's type must accept arbitrary ``__init__`` arguments.  A small
    recording class (NOT a DTensor subclass) is used so that tests can
    assert on the forwarded constructor arguments.

    Args:
        mesh: device_mesh value.
        placements: _placements list.
        alias_placements: If provided, ``_layout.alias_placements`` returns
            this value instead of ``placements``.

    Returns:
        tuple: ``(fake, mock_local, calls)``.
    """
    if placements is None:
        placements = [Shard(0)]

    mock_local = Mock(name="local_tensor")
    mock_local.to.return_value = Mock(name="new_local")
    mock_local.float.return_value = Mock(name="new_float_local")

    fake_layout = SimpleNamespace(
        alias_placements=alias_placements if alias_placements else placements
    )

    calls = []

    class _Recorder:
        """Plain Python class — records constructor calls, no Tensor baggage."""

        _alias_placements = _alias_placements_fn
        _from_converted_local = _from_converted_local_fn

        def __init__(self, local_tensor=None, device_mesh=None, placements=None):
            calls.append({
                "local_tensor": local_tensor,
                "device_mesh": device_mesh,
                "placements": placements,
            })

    fake = _Recorder.__new__(_Recorder)
    fake._local_tensor = mock_local
    fake._layout = fake_layout
    fake._placements = placements
    fake._device_mesh = mesh
    return fake, mock_local, calls


def test_to_delegates_dtype_conversion_to_local_tensor():
    """
    Feature: DTensor.to() dtype conversion
    Description: Call to("float16_arg") and verify _local_tensor.to receives it.
    Expectation: _local_tensor.to called once with "float16_arg".
    """
    fake, mock_local, calls = _make_mock_dtensor()

    _to_fn(fake, "float16_arg")

    mock_local.to.assert_called_once_with("float16_arg")


def test_to_preserves_device_mesh():
    """
    Feature: DTensor.to() device_mesh propagation
    Description: Call to() on a DTensor with device_mesh="my_mesh".
    Expectation: Constructor receives device_mesh="my_mesh" unchanged.
    """
    fake, mock_local, calls = _make_mock_dtensor(mesh="my_mesh")

    _to_fn(fake, "dtype_arg")

    assert len(calls) == 1, (
        f"Expected 1 constructor call, got {len(calls)}"
    )
    assert calls[0]["device_mesh"] == "my_mesh", (
        f"Expected device_mesh 'my_mesh', got {calls[0]['device_mesh']}"
    )


def test_to_uses_alias_placements_when_layout_exists():
    """
    Feature: DTensor.to() alias_placements priority
    Description: Construct mock with _placements=[Shard(0)] but
        layout.alias_placements=[Shard(1)].
    Expectation: Constructor receives [Shard(1)], not [Shard(0)].
    """
    alias_p = [Shard(1)]
    fake, mock_local, calls = _make_mock_dtensor(
        placements=[Shard(0)], alias_placements=alias_p
    )

    _to_fn(fake, "dtype_arg")

    assert calls[0]["placements"] == alias_p, (
        f"Expected alias_placements {alias_p}, got {calls[0]['placements']}"
    )


def test_to_falls_back_to_placements_when_no_layout():
    """
    Feature: DTensor.to() fallback to _placements
    Description: Construct mock with _layout=None and _placements=[Shard(0)].
    Expectation: Constructor receives [Shard(0)] from _placements.
    """
    mock_local = Mock(name="local_tensor")
    mock_local.to.return_value = Mock(name="new_local")
    placements = [Shard(0)]

    calls = []

    class _Recorder:
        _alias_placements = _alias_placements_fn
        _from_converted_local = _from_converted_local_fn

        def __init__(self, local_tensor=None, device_mesh=None, placements=None):
            calls.append({
                "local_tensor": local_tensor,
                "device_mesh": device_mesh,
                "placements": placements,
            })

    fake = _Recorder.__new__(_Recorder)
    fake._local_tensor = mock_local
    fake._layout = None
    fake._placements = placements
    fake._device_mesh = "fake_mesh"

    _to_fn(fake, "dtype_arg")

    assert calls[0]["placements"] == placements, (
        f"Expected placements {placements}, got {calls[0]['placements']}"
    )


def test_float_delegates_to_local_tensor():
    """
    Feature: DTensor.float() dtype conversion
    Description: Call float() and verify _local_tensor.float is invoked.
    Expectation: _local_tensor.float called once; constructor receives correct args.
    """
    fake, mock_local, calls = _make_mock_dtensor(mesh="my_mesh")

    _float_fn(fake)

    mock_local.float.assert_called_once_with()
    assert calls[0]["device_mesh"] == "my_mesh", (
        f"Expected device_mesh 'my_mesh', got {calls[0]['device_mesh']}"
    )


def _make_parameter_dtensor(monkeypatch, mesh="fake_mesh", placements=None):
    """Build a fake ParameterDTensor that must be rebuilt as base DTensor."""
    if placements is None:
        placements = [Shard(0)]

    class _FakeParameter:
        pass

    class _BaseDTensor:
        def __init__(self, local_tensor=None, device_mesh=None, placements=None):
            calls.append({
                "local_tensor": local_tensor,
                "device_mesh": device_mesh,
                "placements": placements,
            })

    class _ParameterDTensor(_FakeParameter):
        _alias_placements = _alias_placements_fn
        _from_converted_local = _from_converted_local_fn

        def __init__(self, *args, **kwargs):
            raise AssertionError("ParameterDTensor constructor should not be used")

    calls = []
    monkeypatch.setattr(dtensor_mod.platform, "Parameter", _FakeParameter)
    monkeypatch.setattr(dtensor_mod, "DTensor", _BaseDTensor)

    mock_local = Mock(name="local_tensor")
    mock_local.to.return_value = Mock(name="new_local")
    mock_local.float.return_value = Mock(name="new_float_local")

    fake = _ParameterDTensor.__new__(_ParameterDTensor)
    fake._local_tensor = mock_local
    fake._layout = SimpleNamespace(alias_placements=placements)
    fake._placements = placements
    fake._device_mesh = mesh
    return fake, mock_local, calls


def test_to_on_parameter_dtensor_returns_base_dtensor(monkeypatch):
    """
    Feature: DTensor.to() on ParameterDTensor
    Description: Call to() on a parameter-wrapped DTensor.
    Expectation: Rebuild as base DTensor instead of calling ParameterDTensor constructor.
    """
    fake, mock_local, calls = _make_parameter_dtensor(monkeypatch, mesh="my_mesh")

    _to_fn(fake, "dtype_arg")

    mock_local.to.assert_called_once_with("dtype_arg")
    assert len(calls) == 1
    assert calls[0]["device_mesh"] == "my_mesh"


def test_float_on_parameter_dtensor_returns_base_dtensor(monkeypatch):
    """
    Feature: DTensor.float() on ParameterDTensor
    Description: Call float() on a parameter-wrapped DTensor.
    Expectation: Rebuild as base DTensor instead of calling ParameterDTensor constructor.
    """
    fake, mock_local, calls = _make_parameter_dtensor(monkeypatch, mesh="my_mesh")

    _float_fn(fake)

    mock_local.float.assert_called_once_with()
    assert len(calls) == 1
    assert calls[0]["device_mesh"] == "my_mesh"

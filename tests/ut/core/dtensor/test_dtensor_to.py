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
"""Unit tests for DTensor.to(), DTensor.float() and DTensor.type_as()
— platform-agnostic."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import hyper_parallel.core.dtensor.dtensor as dtensor_mod
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard

# DTensor inherits from DTensorBase which inherits from a C extension Tensor.
# On MindSpore the C-level descriptor enforces isinstance(self, Tensor).
# Bypass it by pulling the Python function directly from DTensor.__dict__.
_to_fn = DTensor.__dict__["to"]
_float_fn = DTensor.__dict__["float"]
_type_as_fn = DTensor.__dict__["type_as"]
_alias_placements_fn = DTensor.__dict__["_alias_placements"]
_from_converted_local_fn = DTensor.__dict__["_from_converted_local"]

# The platform Tensor alias used by the public DTensor module.
_Tensor = dtensor_mod.Tensor


def _make_mock_dtensor(mesh="fake_mesh", placements=None, alias_placements=None,
                       dtype=None, is_partial=False):
    """Build a mock DTensor for unit testing without hardware.

    Returns a ``SimpleNamespace`` whose attributes mimic the internal state
    set by ``DTensor.__init_data__``.  ``_local_tensor`` is a ``Mock`` so
    that ``.to()`` / ``.float()`` / ``.type_as()`` never touch real hardware.

    Because ``DTensor.to()`` calls ``self.__class__(...)``, the returned
    object's type must accept arbitrary ``__init__`` arguments.  A small
    recording class (NOT a DTensor subclass) is used so that tests can
    assert on the forwarded constructor arguments.

    Args:
        mesh: device_mesh value.
        placements: _placements list.
        alias_placements: If provided, ``_layout.alias_placements`` returns
            this value instead of ``placements``.
        dtype: torch.dtype attribute on both fake DTensor and mock_local.
        is_partial: If True, ``_layout.is_partial()`` returns True.

    Returns:
        tuple: ``(fake, mock_local, calls)``.
    """
    if placements is None:
        placements = [Shard(0)]

    mock_local = Mock(name="local_tensor")
    new_local_mock = Mock(name="new_local")
    mock_local.to.return_value = new_local_mock
    mock_local.float.return_value = Mock(name="new_float_local")
    mock_local.device = torch.device("cpu")

    fake_layout = SimpleNamespace(
        alias_placements=alias_placements if alias_placements else placements,
        is_partial=lambda: is_partial,
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
    if dtype is not None:
        fake.dtype = dtype
        mock_local.dtype = dtype
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


# ---------------------------------------------------------------------------
# DTensor.type_as() tests
# ---------------------------------------------------------------------------

def test_type_as_delegates_dtype_conversion():
    """
    Feature: DTensor.type_as() dtype conversion
    Description: Call type_as() with a plain Tensor other; verify delegation.
    Expectation: _local_tensor.to(dtype=other.dtype) is called once.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float16)
    other = torch.empty((), dtype=torch.float32)

    _type_as_fn(fake, other)

    mock_local.to.assert_called_once_with(dtype=torch.float32)


def test_type_as_preserves_mesh():
    """
    Feature: DTensor.type_as() device_mesh propagation
    Description: Call type_as() on a DTensor with device_mesh="my_mesh".
    Expectation: Reconstructed DTensor receives device_mesh="my_mesh".
    """
    fake, mock_local, calls = _make_mock_dtensor(mesh="my_mesh", dtype=torch.float16)
    other = torch.empty((), dtype=torch.float32)

    _type_as_fn(fake, other)

    assert len(calls) == 1, (
        f"Expected 1 constructor call, got {len(calls)}"
    )
    assert calls[0]["device_mesh"] == "my_mesh", (
        f"Expected device_mesh 'my_mesh', got {calls[0]['device_mesh']}"
    )


def test_type_as_preserves_placements():
    """
    Feature: DTensor.type_as() placements propagation
    Description: Construct with _layout.alias_placements=[Shard(1)].
    Expectation: Reconstructed DTensor receives [Shard(1)].
    """
    alias_p = [Shard(1)]
    fake, mock_local, calls = _make_mock_dtensor(
        placements=[Shard(0)], alias_placements=alias_p, dtype=torch.float16
    )
    other = torch.empty((), dtype=torch.float32)

    _type_as_fn(fake, other)

    assert calls[0]["placements"] == alias_p, (
        f"Expected alias_placements {alias_p}, got {calls[0]['placements']}"
    )


def test_type_as_with_plain_tensor_other():
    """
    Feature: DTensor.type_as() with plain Tensor other
    Description: Pass a plain torch.Tensor (not DTensor).  other must be
        accepted, its dtype read, its device compared against self.
    Expectation: Conversion succeeds; target dtype = other.dtype.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float16)
    other = torch.empty((5,), dtype=torch.float32)

    _type_as_fn(fake, other)

    mock_local.to.assert_called_once_with(dtype=torch.float32)


def test_type_as_with_dtensor_other(monkeypatch):
    """
    Feature: DTensor.type_as() with DTensor other
    Description: Pass a mocked DTensor as other.  type_as() must call
        other.to_local() to get the local tensor for the device check.
    Expectation: other.to_local() is invoked; dtype is read from other.dtype.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float16)

    mock_other_local = Mock(name="other_local")
    mock_other_local.device = torch.device("cpu")

    mock_other = Mock(name="other_dtensor")
    mock_other.dtype = torch.float32
    mock_other.to_local.return_value = mock_other_local

    # Patch isinstance so mock_other is recognised as a Tensor and DTensor.
    import builtins
    orig_isinstance = builtins.isinstance

    def _custom_isinstance(obj, classinfo):
        if obj is mock_other:
            if classinfo is DTensor or classinfo is _Tensor:
                return True
        return orig_isinstance(obj, classinfo)

    monkeypatch.setattr(builtins, 'isinstance', _custom_isinstance)

    _type_as_fn(fake, mock_other)

    mock_other.to_local.assert_called_once_with()
    mock_local.to.assert_called_once_with(dtype=torch.float32)


def test_type_as_cross_layout():
    """
    Feature: DTensor.type_as() ignores other's layout
    Description: other has different placements / layout than self.
        type_as() must NOT read other's layout — only its dtype and device.
    Expectation: Conversion succeeds; no AttributeError about layout on other.
    """
    fake, mock_local, calls = _make_mock_dtensor(
        placements=[Shard(0), Shard(1)], dtype=torch.float16
    )
    # other is a plain Tensor → isinstance(other, DTensor) is False,
    # so other.to_local() is never called and other.layout is never accessed.
    other = torch.empty((3, 7), dtype=torch.float32)

    _type_as_fn(fake, other)

    mock_local.to.assert_called_once_with(dtype=torch.float32)


def test_type_as_non_tensor_other_raises():
    """
    Feature: DTensor.type_as() with non-Tensor other
    Description: Pass a float (3.14) as other.
    Expectation: Raises ValueError with message containing argument type.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float16)

    try:
        _type_as_fn(fake, 3.14)
    except ValueError as e:
        assert "type_as() argument must be a Tensor" in str(e), (
            f"Unexpected message: {e}"
        )
        assert "float" in str(e), (
            f"Expected type name in message, got: {e}"
        )
    else:
        raise AssertionError("Expected ValueError for non-Tensor other, but no exception was raised.")


def test_type_as_partial_self_raises():
    """
    Feature: DTensor.type_as() rejects Partial input
    Description: self._layout.is_partial() returns True.
    Expectation: Raises ValueError asking user to call reduce_partial() first.
    """
    fake, mock_local, calls = _make_mock_dtensor(
        dtype=torch.float16, is_partial=True
    )
    other = torch.empty((), dtype=torch.float32)

    try:
        _type_as_fn(fake, other)
    except ValueError as e:
        assert "Partial" in str(e), f"Expected Partial in message, got: {e}"
        assert "reduce_partial" in str(e), (
            f"Expected reduce_partial hint, got: {e}"
        )
    else:
        raise AssertionError("Expected ValueError for Partial self, but no exception was raised.")


def test_type_as_cross_device_raises():
    """
    Feature: DTensor.type_as() rejects cross-device input
    Description: self._local_tensor.device != other.device.
    Expectation: Raises ValueError asking user to use to() instead.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float16)
    mock_local.device = torch.device("cpu")

    other = torch.empty((), dtype=torch.float32, device="meta")

    try:
        _type_as_fn(fake, other)
    except ValueError as e:
        assert "same device" in str(e), (
            f"Expected 'same device' in message, got: {e}"
        )
    else:
        raise AssertionError("Expected ValueError for cross-device, but no exception was raised.")


def test_type_as_same_dtype_no_op():
    """
    Feature: DTensor.type_as() same-dtype no-op
    Description: self.dtype == other.dtype (both float32).
    Expectation: Returns self unchanged; _local_tensor.to() is NOT called.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float32)
    other = torch.empty((), dtype=torch.float32)

    result = _type_as_fn(fake, other)

    assert result is fake, (
        "Expected type_as() to return self when dtypes match."
    )
    mock_local.to.assert_not_called()


def test_type_as_output_dtype_matches_other():
    """
    Feature: DTensor.type_as() output dtype verification
    Description: self=float16, other=float32.
    Expectation: _local_tensor.to(dtype=torch.float32) is called and its
        return value is forwarded to _from_converted_local.
    """
    fake, mock_local, calls = _make_mock_dtensor(dtype=torch.float16)
    other = torch.empty((), dtype=torch.float32)

    _type_as_fn(fake, other)

    mock_local.to.assert_called_once_with(dtype=torch.float32)
    assert len(calls) == 1, (
        f"Expected 1 constructor call, got {len(calls)}"
    )
    # The local_tensor passed to the constructor must be the result of .to().
    new_local = mock_local.to.return_value
    assert calls[0]["local_tensor"] is new_local, (
        "_from_converted_local should receive the output of _local_tensor.to()"
    )

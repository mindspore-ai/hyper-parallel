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
"""Unit tests for :meth:`DTensor.tolist` — platform-agnostic."""

from types import SimpleNamespace
from unittest.mock import Mock

from hyper_parallel.core.dtensor.dtensor import DTensor

# Pull the Python function directly from DTensor.__dict__ to bypass
# C-level Tensor descriptor checks on MindSpore.
_tolist_fn = DTensor.__dict__["tolist"]


def _make_mock_dtensor():
    """Build a mock DTensor whose ``full_tensor()`` returns a mock tensor.

    Returns:
        tuple: ``(fake, mock_full_tensor)`` where ``fake`` is a SimpleNamespace
        with enough attributes to stand in for a DTensor, and ``mock_full_tensor``
        is the Mock that ``full_tensor()`` returns.
    """
    mock_local = Mock(name="local_tensor")
    mock_full = Mock(name="full_tensor")
    mock_local.tolist.return_value = [[1, 2], [3, 4]]
    mock_full.tolist.return_value = [[1, 2], [3, 4]]

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
    )
    return fake, mock_full


def test_tolist_delegates_to_full_tensor():
    """
    Feature: DTensor.tolist() delegation
    Description: Call tolist() on a mock DTensor.
    Expectation: full_tensor() is invoked and its tolist() return value is passed through.
    """
    fake, mock_full = _make_mock_dtensor()

    result = _tolist_fn(fake)

    mock_full.tolist.assert_called_once_with()
    assert result == [[1, 2], [3, 4]], (
        f"Expected [[1, 2], [3, 4]], got {result}"
    )


def test_tolist_returns_list_for_2d_tensor():
    """
    Feature: DTensor.tolist() return type for 2-d tensor
    Description: Verify tolist() returns a nested list when the full tensor is 2-d.
    Expectation: Result is a list of lists.
    """
    mock_local = Mock(name="local_tensor")
    mock_full = Mock(name="full_tensor")
    mock_full.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
    )

    result = _tolist_fn(fake)

    assert isinstance(result, list), (
        f"Expected list, got {type(result).__name__}"
    )
    assert len(result) == 2, f"Expected length 2, got {len(result)}"
    assert result == [[0.1, 0.2], [0.3, 0.4]], (
        f"Expected [[0.1, 0.2], [0.3, 0.4]], got {result}"
    )


def test_tolist_returns_scalar_number_for_0d_tensor():
    """
    Feature: DTensor.tolist() for scalar tensor
    Description: Verify tolist() returns a Python number when the full tensor is 0-d.
    Expectation: Result is an int or float, not a list.
    """
    mock_local = Mock(name="local_tensor")
    mock_full = Mock(name="full_tensor")
    mock_full.tolist.return_value = 3.14

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
    )

    result = _tolist_fn(fake)

    assert not isinstance(result, list), (
        f"Expected number for scalar, got {type(result).__name__}: {result}"
    )
    assert result == 3.14, f"Expected 3.14, got {result}"


def test_tolist_returns_flat_list_for_1d_tensor():
    """
    Feature: DTensor.tolist() for 1-d tensor
    Description: Verify tolist() returns a flat list for a 1-d full tensor.
    Expectation: Result is a single-level list.
    """
    mock_local = Mock(name="local_tensor")
    mock_full = Mock(name="full_tensor")
    mock_full.tolist.return_value = [1, 2, 3, 4]

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
    )

    result = _tolist_fn(fake)

    assert result == [1, 2, 3, 4], (
        f"Expected [1, 2, 3, 4], got {result}"
    )


def test_tolist_returns_nested_lists_for_3d_tensor():
    """
    Feature: DTensor.tolist() for 3-d tensor
    Description: Verify tolist() returns 3-level nested lists for a 3-d full tensor.
    Expectation: Result is a list of lists of lists.
    """
    mock_local = Mock(name="local_tensor")
    mock_full = Mock(name="full_tensor")
    mock_full.tolist.return_value = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
    )

    result = _tolist_fn(fake)

    assert len(result) == 2, f"Expected length 2, got {len(result)}"
    assert len(result[0]) == 2, f"Expected sub-length 2, got {len(result[0])}"
    assert result == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (
        f"Expected [[[1,2],[3,4]],[[5,6],[7,8]]], got {result}"
    )


def test_tolist_full_tensor_equivalence():
    """
    Feature: DTensor.tolist() equals full_tensor().tolist()
    Description: Verify tolist() always matches the result of calling
        full_tensor().tolist() directly.
    Expectation: ``dt.tolist() == dt.full_tensor().tolist()``.
    """
    mock_local = Mock(name="local_tensor")
    mock_full = Mock(name="full_tensor")
    expected = [[10, 20, 30], [40, 50, 60]]
    mock_full.tolist.return_value = expected

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
    )

    result_via_tolist = _tolist_fn(fake)
    result_via_full = fake.full_tensor().tolist()

    assert result_via_tolist == result_via_full, (
        f"tolist() returned {result_via_tolist} but full_tensor().tolist() "
        f"returned {result_via_full}"
    )


def test_tolist_to_local_tolist_difference():
    """
    Feature: DTensor.tolist() vs to_local().tolist()
    Description: Verify that tolist() returns the full (gathered) data,
        while to_local().tolist() returns only the local shard.
    Expectation: The two paths can produce different results because
        tolist() gathers all shards first.
    """
    mock_local = Mock(name="local_tensor")
    mock_local.tolist.return_value = [1, 2]  # local shard only

    mock_full = Mock(name="full_tensor")
    mock_full.tolist.return_value = [1, 2, 3, 4]  # full data after gather

    fake = SimpleNamespace(
        _local_tensor=mock_local,
        _layout=None,
        _placements=[],
        _device_mesh="fake_mesh",
        full_tensor=lambda: mock_full,
        to_local=lambda: mock_local,
    )

    result_full = _tolist_fn(fake)
    result_local = fake.to_local().tolist()

    assert result_full == [1, 2, 3, 4], (
        f"tolist() should return full data [1,2,3,4], got {result_full}"
    )
    assert result_local == [1, 2], (
        f"to_local().tolist() should return local shard [1,2], got {result_local}"
    )
    assert result_full != result_local, (
        "tolist() and to_local().tolist() should differ when tensor is sharded"
    )
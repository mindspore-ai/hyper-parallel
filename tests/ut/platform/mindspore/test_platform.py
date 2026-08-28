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
"""Unit tests for MindSporePlatform communication behavior."""
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pytest

ms = pytest.importorskip("mindspore")
nn = pytest.importorskip("mindspore.nn")
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

from hyper_parallel.platform.mindspore.platform import (  # pylint: disable=wrong-import-position
    MindSporePlatform,
    _MSDifferentiableAllToAllSingle,
    _mindspore_variable_all_gather,
    _mindspore_variable_all_to_all,
    _validate_variable_row_splits,
)


def _tensor(rows: int = 3, width: int = 4):
    """Create a deterministic two-dimensional MindSpore tensor."""
    return ms.Tensor(np.arange(rows * width, dtype=np.float32).reshape(rows, width))


def test_prepare_batch_p2p_group_does_not_synchronize():
    """MindSpore batch P2P preparation must not introduce a group barrier."""
    with mock.patch("hyper_parallel.platform.mindspore.platform.dist.barrier") as barrier:
        result = MindSporePlatform.prepare_batch_p2p_group(mock.sentinel.pp_group)

    barrier.assert_not_called()
    assert result is None


def test_buffers_dict_includes_all_registered_buffers():
    """MindSpore buffer enumeration includes persistent and non-persistent buffers."""
    cell = nn.Cell()
    cell.register_buffer("persistent", ms.Tensor([1.0]))
    cell.register_buffer("scratch", ms.Tensor([2.0]), persistent=False)

    buffers = dict(MindSporePlatform.buffers_dict(cell))

    assert set(buffers) == {"persistent", "scratch"}
    assert buffers["persistent"] is cell.persistent
    assert buffers["scratch"] is cell.scratch


def test_variable_all_gather_uses_element_splits() -> None:
    """Convert N-D row counts into flattened AllGatherV element counts."""
    input_tensor = _tensor(2)
    flat_output = ms.Tensor(np.arange(24, dtype=np.float32))
    all_gather_v = mock.MagicMock(return_value=flat_output)

    with mock.patch.object(
        ms.ops, "AllGatherV", return_value=all_gather_v
    ) as all_gather_v_cls:
        result = _mindspore_variable_all_gather(
            input_tensor, [2, 0, 4], "group"
        )

    all_gather_v_cls.assert_called_once_with(group="group")
    flat_input, element_splits = all_gather_v.call_args.args
    assert tuple(flat_input.shape) == (8,)
    assert element_splits.asnumpy().tolist() == [8, 0, 16]
    assert tuple(result.shape) == (6, 4)


def test_variable_all_gather_public_api_accepts_zero_local_rows() -> None:
    """A zero-row local tensor still uses the shared row-split contract."""
    input_tensor = _tensor(0)
    flat_output = ms.Tensor(np.arange(8, dtype=np.float32))
    all_gather_v = mock.MagicMock(return_value=flat_output)

    with mock.patch.object(ms.ops, "AllGatherV", return_value=all_gather_v):
        result = MindSporePlatform.differentiable_variable_all_gather(
            input_tensor, [0, 2], "group"
        )

    flat_input, element_splits = all_gather_v.call_args.args
    assert tuple(flat_input.shape) == (0,)
    assert element_splits.asnumpy().tolist() == [0, 8]
    assert tuple(result.shape) == (2, 4)


@pytest.mark.parametrize(
    "splits, message",
    [([], "at least one"), ([1, -1], "non-negative")],
)
def test_variable_all_gather_rejects_invalid_metadata(splits, message) -> None:
    """Invalid split metadata fails before constructing AllGatherV."""
    with mock.patch.object(
        ms.ops, "AllGatherV"
    ) as all_gather_v, pytest.raises(ValueError, match=message):
        MindSporePlatform.differentiable_variable_all_gather(
            _tensor(1), splits, "group"
        )
    all_gather_v.assert_not_called()


def test_variable_split_validation_uses_dim_zero_rows():
    """Validate row sums, metadata lengths and the group size."""
    with mock.patch(
        "hyper_parallel.platform.mindspore.platform.get_group_size", return_value=2
    ):
        input_splits, output_splits = _validate_variable_row_splits(
            _tensor(), [1, 2], [2, 1], "group"
        )

    assert input_splits == [1, 2]
    assert output_splits == [2, 1]

    with mock.patch(
        "hyper_parallel.platform.mindspore.platform.get_group_size", return_value=2
    ), pytest.raises(ValueError, match=r"sum\(input_splits\)"):
        _validate_variable_row_splits(_tensor(), [1, 1], [1, 1], "group")


def test_differentiable_all_to_all_forwards_row_splits():
    """The public API accepts N-D input and row-count splits."""
    input_tensor = _tensor()
    sentinel = object()
    with mock.patch(
        "hyper_parallel.platform.mindspore.platform.get_group_size", return_value=2
    ), mock.patch.object(
        _MSDifferentiableAllToAllSingle, "apply", return_value=sentinel
    ) as mock_apply:
        result = MindSporePlatform.differentiable_all_to_all_single(
            input_tensor, [1, 2], [2, 1], "group"
        )

    assert result is sentinel
    assert mock_apply.call_args.args == (input_tensor, [2, 1], [1, 2], "group")


def test_variable_all_to_all_allocates_nd_output_from_row_splits():
    """The communication wrapper receives an N-D output shape and row splits."""
    input_tensor = _tensor()
    expected = _tensor()
    with mock.patch(
        "hyper_parallel.platform.mindspore.platform.comm_func.all_to_all_single",
        return_value=(expected, None),
    ) as mock_all_to_all:
        result = _mindspore_variable_all_to_all(
            input_tensor, [1, 2], [2, 1], "group"
        )

    assert result is expected
    assert mock_all_to_all.call_args.args == ((3, 4), input_tensor)
    assert mock_all_to_all.call_args.kwargs == {
        "input_split_sizes": [1, 2],
        "output_split_sizes": [2, 1],
        "group": "group",
        "async_op": False,
    }


def test_variable_all_to_all_backward_swaps_splits():
    """Backward routes gradients through the exact reverse A2A."""
    grad_output = _tensor()
    grad_input = _tensor()
    ctx = SimpleNamespace(
        input_splits=[1, 2],
        output_splits=[2, 1],
        group="group",
    )
    with mock.patch(
        "hyper_parallel.platform.mindspore.platform._mindspore_variable_all_to_all",
        return_value=grad_input,
    ) as mock_all_to_all:
        result = _MSDifferentiableAllToAllSingle.backward(ctx, grad_output)

    assert result[0] is grad_input
    assert result[1:] == (None, None, None)
    assert mock_all_to_all.call_args.args == (
        grad_output,
        [2, 1],
        [1, 2],
        "group",
    )

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
"""
UT for :mod:`hyper_parallel.platform.mindspore.pipeline_parallel._utils` ``_MicroBatch``.
"""
import os

# Minimal torch.distributed env for ``custom_shard`` single-process UT.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK_ID", "0")
os.environ.setdefault("DEVICE_ID", "0")
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import pytest

pytest.importorskip("mindspore")

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_shard_and_dtensor,
)

ensure_mindspore_platform_for_shard_and_dtensor()

import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.pipeline_parallel.utils import BatchDimSpec
from hyper_parallel.core.pipeline_parallel.scheduler import PipelineScheduleRuntime
from hyper_parallel.platform.mindspore.pipeline_parallel._utils import _MicroBatch

# ``_MicroBatch`` exercises ``custom_shard`` collectives, so its tests need a real
# MindSpore distributed init (HCCL/msrun) and are covered in pipeline-parallel ST.
# The batch-dim normalization tests at the bottom are pure (no collectives) and run
# as plain UT, so the skip is scoped to the ``_MicroBatch`` tests only.
requires_distributed = pytest.mark.skip(
    reason="Microbatch uses ``custom_shard`` collectives; requires MindSpore distributed "
    "init (HCCL/msrun). Covered in pipeline parallel ST.",
)


@requires_distributed
def test_microbatch_basic_args_only():
    """
    Feature: _microbatch
    Description: Test the function of args
    Expectation: Run success
    """
    batch_size = 8
    micro_batch_num = 4
    dp = 1
    mp = 1
    cp = 1
    layout = Layout((dp, mp, cp), ("dp", "mp", "cp"))
    local_tensor1 = Tensor(np.ones((batch_size, 10, 20)), ms.float32)
    local_tensor2 = Tensor(np.ones((batch_size, 5)), ms.float32)
    input_tensor1_layout = layout("dp", "mp", "cp")
    input_tensor2_layout = layout("dp", "mp")
    input_tensor1 = DTensor.from_local(local_tensor1, input_tensor1_layout.mesh, input_tensor1_layout.placements)
    input_tensor2 = DTensor.from_local(local_tensor2, input_tensor2_layout.mesh, input_tensor2_layout.placements)
    original_layout1 = input_tensor1.layout
    original_layout2 = input_tensor2.layout
    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        args_batch_dim=BatchDimSpec.from_tuple((0, 0))
    )
    args = [input_tensor1, input_tensor2]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)
    assert len(args_after_split) == micro_batch_num
    for micro_args in args_after_split:
        assert micro_args[0].layout == original_layout1
        assert micro_args[1].layout == original_layout2
    return True

@requires_distributed
def test_microbatch_kwargs_only():
    """
    Feature: _microbatch
    Description:Test the function of kwargs
    Expectation:Run success
    """
    total_batch_size = 12
    micro_batch_num = 3

    dp = 1
    mp = 1
    batch_per_device = total_batch_size // dp

    layout = Layout((dp, mp), ("dp", "mp"))

    local_tensor1 = Tensor(np.ones((batch_per_device, 32)), ms.float32)
    local_tensor2 = Tensor(np.ones((20, batch_per_device)), ms.float32)

    input_tensor1_layout = layout("dp", "mp")
    input_tensor2_layout = layout("dp", "mp")

    input_tensor1 = DTensor.from_local(local_tensor1, input_tensor1_layout.mesh, input_tensor1_layout.placements)
    input_tensor2 = DTensor.from_local(local_tensor2, input_tensor2_layout.mesh, input_tensor2_layout.placements)

    original_layout1 = input_tensor1.layout
    original_layout2 = input_tensor2.layout

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        kwargs_batch_dim=BatchDimSpec.from_dict({
            "input1": 0,
            "input2": 1
        })
    )

    args = []
    kwargs = {
        "input1": input_tensor1,
        "input2": input_tensor2
    }

    args_after_split, kwargs_after_split = microbatch(args, kwargs)

    assert len(args_after_split) == micro_batch_num
    assert len(kwargs_after_split) == micro_batch_num

    for i, micro_kwargs in enumerate(kwargs_after_split):
        assert micro_kwargs['input1'].shape == (total_batch_size // micro_batch_num, 32)
        assert micro_kwargs['input2'].shape == (20, total_batch_size // micro_batch_num)

        assert micro_kwargs['input1'].layout == original_layout1, f"Micro batch {i} tensor1 layout mismatch"
        assert micro_kwargs['input2'].layout == original_layout2, f"Micro batch {i} tensor2 layout mismatch"

    for micro_args in args_after_split:
        assert micro_args == []

@requires_distributed
def test_microbatch_mixed_args_kwargs():
    """
    Feature: _microbatch
    Description:Test the mix function of args and kwargs
    Expectation:Run success
    """
    batch_size = 16
    micro_batch_num = 2
    local_arg_tensor = Tensor(np.ones((batch_size, 64)), ms.float32)
    local_kwarg_tensor1 = Tensor(np.ones((8, batch_size, 16)), ms.float32)
    local_kwarg_tensor2 = Tensor(np.ones((batch_size,)), ms.float32)

    dp = 1
    mp = 1
    cp = 1
    layout = Layout((dp, mp, cp), ("dp", "mp", "cp"))
    arg_tensor_layout = layout("dp", "mp")
    kwarg_tensor1_layout = layout("dp", "mp", "cp")
    kwarg_tensor2_layout = layout("dp")

    input_arg_tensor = DTensor.from_local(local_arg_tensor, arg_tensor_layout.mesh, arg_tensor_layout.placements)
    input_kwarg_tensor1 = DTensor.from_local(local_kwarg_tensor1,
                                             kwarg_tensor1_layout.mesh,
                                             kwarg_tensor1_layout.placements
                                             )
    input_kwarg_tensor2 = DTensor.from_local(local_kwarg_tensor2,
                                             kwarg_tensor2_layout.mesh,
                                             kwarg_tensor2_layout.placements
                                             )

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        args_batch_dim=BatchDimSpec.from_tuple((0,)),
        kwargs_batch_dim=BatchDimSpec.from_dict({
            "data1": 1,
            "data2": 0
        })
    )

    args = [input_arg_tensor]
    kwargs = {
        "data1": input_kwarg_tensor1,
        "data2": input_kwarg_tensor2
    }
    args_after_split, kwargs_after_split = microbatch(args, kwargs)

    assert len(args_after_split) == micro_batch_num
    assert len(kwargs_after_split) == micro_batch_num

    for micro_args, micro_kwargs in zip(args_after_split, kwargs_after_split):
        assert len(micro_args) == 1
        assert micro_args[0].shape == (batch_size // micro_batch_num, 64)

        assert micro_kwargs['data1'].shape == (8, batch_size // micro_batch_num, 16)
        assert micro_kwargs['data2'].shape == (batch_size // micro_batch_num,)

@requires_distributed
def test_microbatch_default_batch_dim():
    """
    Feature: _microbatch
    Description:Test the function of default_batch_dim
    Expectation:Run success
    """
    batch_size = 6
    micro_batch_num = 3
    local_tensor = Tensor(np.ones((batch_size, 10)), ms.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    tensor_layout = layout("dp", "mp")
    input_tensor = DTensor.from_local(local_tensor, tensor_layout.mesh, tensor_layout.placements)

    microbatch = _MicroBatch(micro_batch_num=micro_batch_num)

    args = [input_tensor]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)

    for micro_args in args_after_split:
        assert micro_args[0].shape == (batch_size // micro_batch_num, 10)

@requires_distributed
def test_microbatch_partial_batch_dim_spec():
    """
    Feature: _microbatch
    Description:Test the function of partial_batch_dim
    Expectation:Run success
    """
    batch_size = 9
    micro_batch_num = 3
    local_arg_tensor1 = Tensor(np.ones((batch_size, 10)), ms.float32)
    local_arg_tensor2 = Tensor(np.ones((batch_size, 20)), ms.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    input_tensor_layout = layout("dp", "mp")
    arg_tensor1 = DTensor.from_local(local_arg_tensor1, input_tensor_layout.mesh, input_tensor_layout.placements)
    arg_tensor2 = DTensor.from_local(local_arg_tensor2, input_tensor_layout.mesh, input_tensor_layout.placements)

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        args_batch_dim=[BatchDimSpec(1), None]
    )  # BatchDimSpec.from_tuple((1,)):IndexError: tuple index out of range

    args = [arg_tensor1, arg_tensor2]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)

    for micro_args in args_after_split:
        assert len(micro_args) == 2

@requires_distributed
def test_microbatch_edge_cases():
    """
    Feature: _microbatch
    Description:Test the function of edge_cases
    Expectation:Run success
    """
    batch_size = 8
    local_input_tensor = Tensor(np.ones((batch_size, 10)), ms.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    input_tensor_layout = layout("dp", "mp")
    input_tensor = DTensor.from_local(local_input_tensor, input_tensor_layout.mesh, input_tensor_layout.placements)

    microbatch = _MicroBatch(micro_batch_num=1)
    args = [input_tensor]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)

    assert len(args_after_split) == 1
    assert args_after_split[0][0].shape == (batch_size, 10)


# --------------------------------------------------------------------------- #
# Batch-dim normalization (``PipelineScheduleRuntime._normalize_*``).
# Pure helpers that let a single-input model pass a bare ``int``/``BatchDimSpec``
# and let list/dict entries be plain ints; no collectives, so plain UT.
# --------------------------------------------------------------------------- #
_norm_args = PipelineScheduleRuntime._normalize_args_batch_dim
_norm_kwargs = PipelineScheduleRuntime._normalize_kwargs_batch_dim


def _dims(spec_seq):
    """Map a tuple of ``BatchDimSpec | None`` to its ``.batch_dim`` ints for asserts."""
    return tuple(None if s is None else s.batch_dim for s in spec_seq)


def test_normalize_args_bare_scalar_wrapped():
    """
    Feature: args_batch_dim normalization
    Description: A single-input model may pass a bare int / BatchDimSpec (no list).
    Expectation: Wrapped into a one-element tuple of BatchDimSpec.
    """
    assert _dims(_norm_args(0)) == (0,)
    assert _dims(_norm_args(3)) == (3,)
    assert _dims(_norm_args(BatchDimSpec(2))) == (2,)


def test_normalize_args_list_int_and_none():
    """
    Feature: args_batch_dim normalization
    Description: List/tuple entries may be plain int, BatchDimSpec, or None.
    Expectation: ints become BatchDimSpec, None and BatchDimSpec pass through.
    """
    assert _dims(_norm_args([0, 1])) == (0, 1)
    assert _dims(_norm_args((0, None, BatchDimSpec(2)))) == (0, None, 2)
    assert _dims(_norm_args([BatchDimSpec(1), 0])) == (1, 0)


def test_normalize_args_none_and_empty_and_from_tuple():
    """
    Feature: args_batch_dim normalization
    Description: None / empty stay benign; legacy ``from_tuple`` still works.
    Expectation: None -> None, [] -> (), from_tuple unchanged.
    """
    assert _norm_args(None) is None
    assert _norm_args([]) == ()
    assert _dims(_norm_args(BatchDimSpec.from_tuple((0, 1)))) == (0, 1)


@pytest.mark.parametrize("bad", [True, False, 1.5, "0", object()])
def test_normalize_args_rejects_bad_top_type(bad):
    """
    Feature: args_batch_dim normalization
    Description: A non-int/spec/list scalar (incl. bool) is rejected at the top.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_args(bad)


@pytest.mark.parametrize("seq", [[0, True], [1.5], [BatchDimSpec(0), "x"]])
def test_normalize_args_rejects_bad_element(seq):
    """
    Feature: args_batch_dim normalization
    Description: A bad element inside the list (incl. bool) is rejected.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_args(seq)


def test_normalize_kwargs_int_and_spec():
    """
    Feature: kwargs_batch_dim normalization
    Description: Dict values may be plain int or BatchDimSpec.
    Expectation: ints become BatchDimSpec; legacy from_dict still works.
    """
    out = _norm_kwargs({"x": 0, "y": BatchDimSpec(1)})
    assert {k: v.batch_dim for k, v in out.items()} == {"x": 0, "y": 1}
    out2 = _norm_kwargs(BatchDimSpec.from_dict({"a": 2}))
    assert out2["a"].batch_dim == 2


def test_normalize_kwargs_none_passthrough():
    """
    Feature: kwargs_batch_dim normalization
    Description: None stays None.
    Expectation: None -> None.
    """
    assert _norm_kwargs(None) is None


@pytest.mark.parametrize("bad", [[0], (0,), 0, "x"])
def test_normalize_kwargs_rejects_non_dict(bad):
    """
    Feature: kwargs_batch_dim normalization
    Description: A non-dict is rejected.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_kwargs(bad)


@pytest.mark.parametrize("bad", [True, 1.5, "0"])
def test_normalize_kwargs_rejects_bad_value(bad):
    """
    Feature: kwargs_batch_dim normalization
    Description: A bad dict value (incl. bool) is rejected.
    Expectation: TypeError.
    """
    with pytest.raises(TypeError):
        _norm_kwargs({"x": bad})


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    test_microbatch_basic_args_only()
    test_microbatch_kwargs_only()
    test_microbatch_mixed_args_kwargs()
    test_microbatch_default_batch_dim()
    test_microbatch_partial_batch_dim_spec()
    test_microbatch_edge_cases()

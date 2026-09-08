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
"""Unit tests for pipeline-stage P2P buffer lifecycle."""
# pylint: disable=protected-access

from unittest.mock import patch

import hyper_parallel.core.pipeline_parallel.stage as stage_module
from hyper_parallel.core.pipeline_parallel.stage import PipelineStage


class _FakeTensor:
    """Tensor metadata used without allocating framework tensors."""

    def __init__(self, shape: tuple[int, ...], dtype: str, requires_grad: bool) -> None:
        """Store the metadata consumed by pipeline P2P setup."""
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad


class _FakeDTensor(_FakeTensor):
    """DTensor metadata with a distinct local communication shape."""

    def __init__(self, shape: tuple[int, ...], local_shape: tuple[int, ...], dtype: str) -> None:
        """Store global and local shapes used by pipeline metadata extraction."""
        super().__init__(shape, dtype, requires_grad=True)
        self.local_shape = local_shape
        self.layout = object()


def _make_stage(outputs: tuple[_FakeTensor, ...], micro_index: int = 0) -> PipelineStage:
    """Create a device-free stage with cached forward outputs."""
    stage = object.__new__(PipelineStage)
    stage.stage_index = 0
    stage.stage_num = 2
    stage.dst_stage = 1
    stage.device = "test_device"
    stage._has_backward = True
    stage._virtual_chunk_num = 1
    stage.grad_recv_info = {}
    stage._fwd_output_meta = {}
    stage.fwd_outputs_cache = {}
    stage.fwd_outputs_cache[micro_index] = outputs
    return stage


def test_backward_recv_buffer_is_allocated_when_recv_is_posted() -> None:
    """Reserve gradient slots at forward-send time and allocate them at backward-recv time."""
    detached_output = _FakeTensor((2,), "float32", requires_grad=False)
    grad_output = _FakeTensor((3, 4), "float64", requires_grad=True)
    recv_buffer = object()
    stage = _make_stage((detached_output, grad_output))

    with patch.object(stage, "_global_rank", return_value=1), \
            patch.object(stage, "_communicate_meta"), \
            patch.object(stage_module.platform, "empty", return_value=recv_buffer) as empty_mock:
        send_specs = stage.fwd_send_specs(0)

        assert empty_mock.call_count == 0, (
            f"FWD_SEND must not allocate a backward recv buffer: expected=0, got={empty_mock.call_count}"
        )
        assert len(send_specs) == 2, f"Expected two forward-send specs, got={len(send_specs)}"
        recv_infos = stage.grad_recv_info[0]
        assert len(recv_infos) == 1, f"Expected one gradient recv slot, got={len(recv_infos)}"
        assert recv_infos[0].buffer is None, f"Expected an unallocated recv slot, got={recv_infos[0].buffer}"

        recv_specs = stage.bwd_recv_specs(0)
        assert empty_mock.call_count == 1, (
            f"First BWD_RECV must allocate one buffer: expected=1, got={empty_mock.call_count}"
        )
        empty_mock.assert_called_once_with(grad_output.shape, dtype=grad_output.dtype, device=stage.device)
        assert len(recv_specs) == 1, f"Expected one backward-recv spec, got={len(recv_specs)}"
        op_type, tensor, peer = recv_specs[0]
        assert op_type == "irecv", f"Expected op_type=irecv, got={op_type}"
        assert tensor is recv_buffer, f"Expected recv_buffer={recv_buffer}, got={tensor}"
        assert peer == 1, f"Expected peer=1, got={peer}"

        repeated_specs = stage.bwd_recv_specs(0)
        assert empty_mock.call_count == 1, (
            f"Repeated specs construction must reuse the buffer: expected=1, got={empty_mock.call_count}"
        )
        assert repeated_specs[0][1] is recv_buffer, (
            f"Expected repeated spec to reuse recv_buffer={recv_buffer}, got={repeated_specs[0][1]}"
        )


def test_reused_micro_index_allocates_from_latest_output_meta() -> None:
    """Use the latest shape and dtype when a cleared micro-batch slot is reused next step."""
    first_output = _FakeTensor((2, 3), "float32", requires_grad=True)
    second_output = _FakeDTensor((10, 7), (5, 7), "float64")
    first_buffer = object()
    second_buffer = object()
    stage = _make_stage((first_output,))

    with patch.object(stage, "_global_rank", return_value=1), \
            patch.object(stage, "_communicate_meta"), \
            patch.object(stage_module, "DTensor", _FakeDTensor), \
            patch.object(stage_module.platform, "empty", side_effect=[first_buffer, second_buffer]) as empty_mock:
        stage.fwd_send_specs(0)
        first_specs = stage.bwd_recv_specs(0)
        stage._clear_recv_buffer(stage.grad_recv_info, 0)

        stage.fwd_outputs_cache[0] = (second_output,)
        stage.fwd_send_specs(0)
        assert empty_mock.call_count == 1, (
            f"Reused FWD_SEND must leave the recv slot empty: expected=1, got={empty_mock.call_count}"
        )
        second_specs = stage.bwd_recv_specs(0)

    assert first_specs[0][1] is first_buffer, (
        f"Expected first recv buffer={first_buffer}, got={first_specs[0][1]}"
    )
    assert second_specs[0][1] is second_buffer, (
        f"Expected second recv buffer={second_buffer}, got={second_specs[0][1]}"
    )
    second_call = empty_mock.call_args_list[1]
    assert second_call.args == (second_output.local_shape,), (
        f"Expected latest local-shape args={(second_output.local_shape,)}, got={second_call.args}"
    )
    assert second_call.kwargs == {"dtype": second_output.dtype, "device": stage.device}, (
        f"Expected latest dtype/device={{'dtype': {second_output.dtype}, 'device': {stage.device}}}, "
        f"got={second_call.kwargs}"
    )

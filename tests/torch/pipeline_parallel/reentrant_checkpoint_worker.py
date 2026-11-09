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
"""NPU VPP worker comparing reentrant split with a full uncheckpointed model."""
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel import MetaStepType, PipelineStage, ScheduleInterleaved1F1B
from hyper_parallel.core.activation_memory import checkpoint_wrapper, checkpoint_exclude_wrapper
from hyper_parallel.core.activation_memory.swap import SwapManager
from hyper_parallel.core.pipeline_parallel import CommComputeOverlap
from hyper_parallel.core.pipeline_parallel._stage import PipelineStageBase
from tests.torch.pipeline_parallel.simple_mlp import init_hccl


class _Layer(torch.nn.Module):
    """Nonlinear checkpoint with one excluded operator."""

    def __init__(self, index: int, counts: Optional[dict[str, int]] = None) -> None:
        """Build a layer and optionally track its checkpoint replay."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(8, device="npu") * (1 + index * 0.02))
        self.bias = torch.nn.Parameter(torch.full((8,), 0.01 * (index + 1), device="npu"))
        self.counts = counts
        self.excluded = checkpoint_exclude_wrapper(self._excluded)
        self.checkpointed = checkpoint_wrapper(self._body, use_reentrant=True) if counts is not None else None

    def _excluded(self, value):
        if self.counts is not None:
            self.counts["exclude"] += 1
        return value + 0.1 * torch.tanh(value @ self.weight)

    def _body(self, value):
        if self.counts is not None:
            self.counts["body"] += 1
        return self.excluded(value) + self.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run either a checkpointed pipeline layer or an ordinary reference."""
        return self.checkpointed(value) if self.checkpointed is not None else self._body(value)


def _overlap(schedule):
    """Register an overlapped forward and backward pipeline callback."""
    overlap = CommComputeOverlap()

    def callback(step: object, ctx: object) -> None:
        """Run the paired forward and input-backward steps concurrently."""
        backward_step, forward_step = step.sub_steps

        def forward() -> None:
            """Execute the overlapped forward step."""
            ctx.schedule.execute_fwd_leaf(forward_step, ctx.arg_mbs, ctx.kwarg_mbs, ctx.losses)

        def backward() -> None:
            """Execute input backward on this worker's NPU."""
            torch.npu.set_device(dist.get_rank() % torch.npu.device_count())
            ctx.schedule._exec_step(  # pylint: disable=protected-access
                backward_step, ctx.arg_mbs, ctx.kwarg_mbs, ctx.losses,
            )

        overlap.run(fwd_fn=forward, bwd_fn=backward)

    schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)


def _run(pp_size):
    """Run the reentrant checkpoint pipeline case for a pipeline size."""
    init_hccl()
    rank = dist.get_rank()
    torch.npu.set_device(rank % torch.npu.device_count())
    virtual_count = pp_size * 2
    micro_count = max(8, pp_size * 2)
    counts = {"body": 0, "exclude": 0}
    layers = [_Layer(index) for index in range(virtual_count)]
    stage_indices = (rank, rank + pp_size)
    chunks = [torch.nn.Sequential(_Layer(index, counts)) for index in stage_indices]
    stages = [PipelineStage(chunk, index, virtual_count, torch.device("npu"))
              for chunk, index in zip(chunks, stage_indices)]
    for chunk, index in zip(chunks, stage_indices):
        with torch.no_grad():
            chunk[0].weight.copy_(layers[index].weight)
            chunk[0].bias.copy_(layers[index].bias)
    schedule = ScheduleInterleaved1F1B(stages, micro_count, overlap_b_f=True, enable_dxdw_split=True)
    _overlap(schedule)

    # Sum each reference microbatch separately, matching the schedule's unit sensitivities.
    data = torch.arange(micro_count * 8, device="npu", dtype=torch.float32).reshape(micro_count, 8)
    data = data / (micro_count * 8)
    for iteration in range(2):
        for layer in layers:
            layer.zero_grad(set_to_none=True)
        for chunk in chunks:
            chunk.zero_grad(set_to_none=True)
        for value in data.split(1):
            result = value
            for layer in layers:
                result = layer(result)
            result.sum().backward()
        outputs = schedule.run(data) if rank == 0 else schedule.run()
        if rank == pp_size - 1:
            with torch.no_grad():
                expected = data
                for layer in layers:
                    expected = layer(expected)
                torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-4, atol=1e-5)
        with torch.no_grad():
            for chunk, index in zip(chunks, stage_indices):
                for name in ("weight", "bias"):
                    param = getattr(chunk[0], name)
                    reference = getattr(layers[index], name)
                    torch.testing.assert_close(param.grad, reference.grad, rtol=1e-4, atol=1e-5)
                    param.add_(param.grad, alpha=-0.01)
            for layer in layers:
                for param in (layer.weight, layer.bias):
                    param.add_(param.grad, alpha=-0.01)
        assert counts == {"body": 2 * micro_count * len(chunks) * (iteration + 1),
                          "exclude": micro_count * len(chunks) * (iteration + 1)}, counts


def test_reentrant_pp2():
    """Exercise warmup, split steady state and cooldown with two ranks."""
    _run(2)


def test_reentrant_pp4():
    """Exercise warmup, split steady state and cooldown with four ranks."""
    _run(4)


def test_reentrant_pp8():
    """Exercise two virtual chunks per rank across eight NPU pipeline ranks."""
    _run(8)


class _SwapSplitBlock(torch.nn.Module):
    """A checkpoint with internal saved tensors and a single excluded region."""

    def __init__(self, counts: dict[str, int], index: int, *, checkpointed: bool) -> None:
        """Build either the swapped stage block or its plain reference."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(8, device="npu") * (1 + index * 0.02))
        self.counts = counts
        self.excluded = checkpoint_exclude_wrapper(self._exclude, swap=checkpointed)
        self.checkpointed = checkpoint_wrapper(self._body, use_reentrant=True) if checkpointed else None

    def _exclude(self, value: torch.Tensor) -> torch.Tensor:
        self.counts["exclude"] += 1
        hidden = torch.tanh(value @ self.weight)
        return hidden.square() + value * 0.1

    def _body(self, value: torch.Tensor) -> torch.Tensor:
        self.counts["body"] += 1
        return self.excluded(value) + value * 0.5

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the checkpointed or plain block."""
        return self.checkpointed(value) if self.checkpointed is not None else self._body(value)


def _swap_split_storages(layer: torch.nn.Module) -> tuple[int, int]:
    """Inspect actual group storage and offloaded tensors after stream completion."""
    manager = SwapManager()
    group_name = layer._swap_group_name  # pylint: disable=protected-access
    group = manager._groups[group_name]  # pylint: disable=protected-access
    storages = tuple(group._storages)  # pylint: disable=protected-access
    host_tensors = sum(
        item.state == "host" for storage in storages for item in storage.iter_swap_tensors()
    )
    return len(storages), host_tensors


def test_reentrant_swap_stage_split_npu() -> None:
    """Check real swap prefetch and stage dx/dw on each rank, without cross-rank PP ordering."""
    init_hccl()
    torch.npu.set_device(dist.get_rank() % torch.npu.device_count())
    counts = {"body": 0, "exclude": 0}
    blocks = torch.nn.Sequential(*(_SwapSplitBlock(counts, index, checkpointed=True) for index in range(2)))
    reference = torch.nn.Sequential(*(
        _SwapSplitBlock({"body": 0, "exclude": 0}, index, checkpointed=False) for index in range(2)
    ))
    manager = SwapManager()
    first, second = (block.checkpointed for block in blocks)
    manager.set_forward_prefetch_layer(first, second)
    first_group_name = first._swap_group_name  # pylint: disable=protected-access
    second_group_name = second._swap_group_name  # pylint: disable=protected-access
    stage = PipelineStageBase(blocks, stage_index=1, stage_num=3)
    stage.args_recv_info = {}
    stage.grad_recv_info = {}
    hits = []
    handles = [param.register_hook(lambda grad, index=index: hits.append(index))
               for index, param in enumerate(blocks.parameters())]
    try:
        for iteration in range(2):
            for micro in range(4):
                blocks.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                hits.clear()
                value = torch.arange(8, device="npu", dtype=torch.float32) / 8 + micro * 0.02
                input_value = value.detach().requires_grad_()
                expected_input = value.detach().requires_grad_()
                expected = reference(expected_input)
                expected.sum().backward()
                stage.args_recv_info[micro] = [SimpleNamespace(buffer=input_value, requires_grad=True)]
                stage.grad_recv_info[micro] = [SimpleNamespace(buffer=torch.ones_like(expected))]
                actual = stage.forward_one_chunk(micro)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(actual.sum(), expected.sum(), rtol=1e-4, atol=1e-5)
                torch.npu.synchronize()
                storage_count, host_count = _swap_split_storages(first)
                assert storage_count > 0 and host_count > 0, (storage_count, host_count)

                with patch.object(manager, "launch_load", wraps=manager.launch_load) as launch_load:
                    stage.backward_input_one_chunk(micro)
                    torch.testing.assert_close(input_value.grad, expected_input.grad, rtol=1e-4, atol=1e-5)
                    assert not hits and all(param.grad is None for param in blocks.parameters())
                    stage.backward_weight_one_chunk(micro)
                assert any(call.args[0] == first_group_name for call in launch_load.call_args_list)
                torch.npu.synchronize()
                assert len(hits) == len(list(blocks.parameters())) and len(set(hits)) == len(hits), hits
                for param, expected_param in zip(blocks.parameters(), reference.parameters()):
                    torch.testing.assert_close(param.grad, expected_param.grad, rtol=1e-4, atol=1e-5)
                assert _swap_split_storages(first) == (0, 0)
                assert _swap_split_storages(second) == (0, 0)
                assert not stage._reentrant_frames and not stage._dw_cache  # pylint: disable=protected-access
                assert counts == {"body": 2 * (iteration * 4 + micro + 1) * 2,
                                  "exclude": (iteration * 4 + micro + 1) * 2}, counts
    finally:
        stage.clear_cache()
        for handle in handles:
            handle.remove()
        manager.unregister_forward_prefetch_layer(first)
        manager.unregister_forward_prefetch_layer(second)
    assert first_group_name not in manager._groups  # pylint: disable=protected-access
    assert second_group_name not in manager._groups  # pylint: disable=protected-access

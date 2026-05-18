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
"""MindSpore pipeline parallel + shard end-to-end test."""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Parameter, mint, nn
from mindspore.common.initializer import initializer

from hyper_parallel import DTensor, PipelineStage, ScheduleInterleaved1F1B, init_device_mesh, shard_module
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat


class MLP(nn.Cell):
    """Single MLP block used by both standalone and pipeline tests."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = Parameter(initializer("ones", [in_size, out_size], ms.float32), name="weight")
        self.relu = mint.nn.ReLU()

    def construct(self, x):
        x = mint.matmul(x, self.weight)
        if isinstance(x, DTensor):
            x = x.reduce_partial()
        x = self.relu(x)
        return x


class SimpleMLP(nn.Cell):
    """Simple stacked MLP network."""

    def __init__(self, num_layers, in_size, out_size):
        super().__init__()
        self.mlp_layers = nn.CellList([MLP(in_size, out_size) for _ in range(num_layers)])

    def construct(self, x):
        for mlp in self.mlp_layers:
            x = mlp(x)
        return x


def _run_standalone(batch_size, model):
    """Run the standalone model and materialize gradients on parameters."""
    x = mint.ones((batch_size, 16), dtype=ms.float32)
    for param in model.trainable_params():
        param.grad = None
    out = model(x)
    sens = mint.ones(out.shape, dtype=out.dtype)
    out.backward(sens)
    return out


def _get_local_tensor(tensor):
    """Return local tensor data for Tensor/DTensor/grad objects."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _expected_local_grad(full_grad, placement, tp_rank, tp_size):
    """Slice standalone full grad to the local TP shard expected on this rank."""
    full_grad_np = full_grad.asnumpy()
    shard_size = full_grad_np.shape[placement.dim] // tp_size
    start = tp_rank * shard_size
    end = start + shard_size
    if placement.dim == 0:
        return full_grad_np[start:end, :]
    return full_grad_np[:, start:end]


def _placement_for_layer(layer_idx):
    """Mirror the torch pipeline_shard alternating matmul layouts."""
    return Shard(1) if layer_idx % 2 == 0 else Shard(0)


def test_pipeline_shard():
    """
    Feature: PipelineParallel + shard.
    Description: Run interleaved pipeline stages on a (pp, tp) mesh and compare
        last-stage outputs plus local weight grads with a standalone model.
    Expectation: Pipeline output and local sharded grads match standalone slices.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()
    D.init()

    world_size = D.get_group_size()
    assert world_size == 4, f"Expected 4 ranks for (pp, tp)=(2, 2), but got {world_size}"

    root_mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("pp", "tp"))
    pp_index, tp_index = root_mesh.get_coordinate()
    pp_mesh = root_mesh["pp"]
    tp_mesh = root_mesh["tp"]
    tp_size = tp_mesh.mesh_shape[0]

    micro_batch_num = 8
    num_stages = 2
    total_virtual_stages = num_stages + 2
    hidden_size = 16

    stage_index = pp_index
    stage0_layer_idx = stage_index
    stage1_layer_idx = stage_index + num_stages

    model0 = MLP(hidden_size, hidden_size)
    model1 = MLP(hidden_size, hidden_size)
    shard_module(
        model0,
        device_mesh=tp_mesh,
        sharding_plan=ShardingPlan(plan={"weight": (_placement_for_layer(stage0_layer_idx),)}),
    )
    shard_module(
        model1,
        device_mesh=tp_mesh,
        sharding_plan=ShardingPlan(plan={"weight": (_placement_for_layer(stage1_layer_idx),)}),
    )

    pipeline_stage0 = PipelineStage(model0, stage_index, total_virtual_stages, mesh=pp_mesh)
    pipeline_stage1 = PipelineStage(model1, stage_index + num_stages, total_virtual_stages, mesh=pp_mesh)
    schedule = ScheduleInterleaved1F1B([pipeline_stage0, pipeline_stage1], micro_batch_num)

    standalone_model = SimpleMLP(total_virtual_stages, hidden_size, hidden_size)
    standalone_out = _run_standalone(micro_batch_num, standalone_model)

    x = mint.ones((micro_batch_num, hidden_size), dtype=ms.float32)
    d_x = DTensor.from_local(x, tp_mesh, (Replicate(),))
    if stage_index == 0:
        pp_losses = schedule.run(d_x)
    else:
        pp_losses = schedule.run()

    if stage_index == num_stages - 1:
        pp_out = mint.cat(tuple(_get_local_tensor(loss) for loss in pp_losses), dim=0)
        np.testing.assert_allclose(pp_out.asnumpy(), standalone_out.asnumpy(), rtol=1e-5, atol=1e-5)

    standalone_grads = [layer.weight.grad for layer in standalone_model.mlp_layers]
    local_grad0 = _get_local_tensor(model0.weight.grad)
    local_grad1 = _get_local_tensor(model1.weight.grad)
    expected0 = _expected_local_grad(
        standalone_grads[stage0_layer_idx], _placement_for_layer(stage0_layer_idx), tp_index, tp_size
    )
    expected1 = _expected_local_grad(
        standalone_grads[stage1_layer_idx], _placement_for_layer(stage1_layer_idx), tp_index, tp_size
    )
    np.testing.assert_allclose(local_grad0.asnumpy(), expected0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(local_grad1.asnumpy(), expected1, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_pipeline_shard()

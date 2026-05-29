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
"""MindSpore GPipe activation swap end-to-end test."""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Parameter, Tensor, mint, nn
from mindspore.common.initializer import initializer

from hyper_parallel import PipelineStage
from hyper_parallel.core.activation_checkpoint import swap_wrapper
from hyper_parallel.core.pipeline_parallel import ScheduleGPipe
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat


class LinearRelu(nn.Cell):
    """Single deterministic linear + relu block."""

    def __init__(self, hidden_size):
        super().__init__()
        self.weight = Parameter(initializer("ones", [hidden_size, hidden_size], ms.float32), name="weight")

    def construct(self, x):
        return mint.nn.functional.relu(mint.matmul(x, self.weight))


def _run_standalone(x, hidden_size):
    """Run a two-stage serial reference and return output plus grads."""
    layers = [LinearRelu(hidden_size), LinearRelu(hidden_size)]
    for param in tuple(layer.weight for layer in layers):
        param.grad = None
    out = x
    for layer in layers:
        out = layer(out)
    out.backward(mint.ones(out.shape, dtype=out.dtype))
    return out, [layer.weight.grad for layer in layers]


def _cat_losses(losses):
    return mint.cat(tuple(losses), dim=0)


def test_gpipe_pipeline_swap():
    """
    Feature: GPipe activation swap.
    Description: Run a two-rank GPipe schedule whose first stage is wrapped by
        activation swap, then compare output and local gradients with a serial
        reference.
    Expectation: Run success.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    enable_mindspore_backward_compat()
    D.init()

    rank = D.get_rank()
    world_size = D.get_group_size()
    assert world_size == 2, f"Expected 2 ranks for GPipe swap, but got {world_size}"

    micro_batch_num = 4
    hidden_size = 8
    x = Tensor(np.arange(micro_batch_num * hidden_size, dtype=np.float32).reshape(micro_batch_num, hidden_size) / 100)
    ref_out, ref_grads = _run_standalone(x, hidden_size)

    stage_model = LinearRelu(hidden_size)
    if rank == 0:
        stage_model = swap_wrapper(stage_model)
    stage = PipelineStage(stage_model, stage_index=rank, stage_num=world_size)
    schedule = ScheduleGPipe(stage, micro_batch_num=micro_batch_num, swap=True)

    if rank == 0:
        pp_losses = schedule.run(x)
    else:
        pp_losses = schedule.run()

    if rank == world_size - 1:
        np.testing.assert_allclose(_cat_losses(pp_losses).asnumpy(), ref_out.asnumpy(), rtol=1e-5, atol=1e-5)

    local_param = next(iter(stage_model.trainable_params()))
    np.testing.assert_allclose(local_param.grad.asnumpy(), ref_grads[rank].asnumpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_gpipe_pipeline_swap()

import time
import numpy as np
import mindspore as ms
from collections import defaultdict
from mindspore import nn, Tensor, mint
from mindspore.communication.management import init, get_rank, get_group_size
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStage
from hyper_parallel.platform.mindspore.pipeline_parallel.schedule import Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B
from hyper_parallel import Layout, DTensor
from hyper_parallel.core.hsdp import hsdp
from hyper_parallel import shard
from mindspore.common.initializer import initializer

import time
import numpy as np
import mindspore as ms
from collections import defaultdict
from mindspore import nn, Tensor, mint
from mindspore.communication.management import init, get_rank, get_group_size
from hyper_parallel.platform.mindspore.pipeline_parallel.stage import PipelineStage
from hyper_parallel.platform.mindspore.pipeline_parallel.schedule import Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B
from hyper_parallel import Layout, DTensor
from hyper_parallel.core.hsdp import hsdp
from hyper_parallel import shard
from mindspore.common.initializer import initializer

def hook_fn(grad):
    print(f"grad:{grad}",flush=True)
    return grad

class MLP(nn.Cell):
    """MLP net."""
    def __init__(self, in_size, out_size, compute_dtype=np.float32, w_layout=None):
        super().__init__()
        # initializing with "ones" is for the convenience of precision compare
        if not w_layout:
            self.weight = ms.Parameter(
                initializer("ones", [in_size, out_size],ms.float32),
                name="weight")
        else:
            self.weight = ms.Parameter(
                # ms.parallel.DTensor.from_local(Tensor(np.ones([in_size, out_size]).astype(compute_dtype)), w_layout),
                DTensor.from_local(initializer("ones", [in_size, out_size],ms.float32), w_layout),
                name="weight")
        self.relu = mint.nn.ReLU()

    def construct(self, x):
        #self.weight.register_hook(hook_fn)
        x = mint.matmul(x, self.weight)
        #x.register_hook(hook_fn)
        x = self.relu(x)
        #x.register_hook(hook_fn)
        return x


class SimpleMLP(nn.Cell):
    """Simple MLP net."""

    def __init__(self, num_layers, in_size, out_size, w_layout=None):
        super().__init__()
        self.mlp_layers = nn.CellDict()
        for mlp_id in range(num_layers):
            self.mlp_layers[str(mlp_id)] = MLP(in_size, out_size, w_layout=w_layout)

    def construct(self, x):
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        return x


def run_standalone(dim):
    """run standalone."""
    model = SimpleMLP(4, 16, 16)

    def forward_fn(data):
        logits = model(data)
        return logits

    grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=False)

    steps = 1
    ret_loss = None
    ret_grads = None
    data = Tensor(np.ones((dim, 16)), dtype=ms.float32)
    for _ in range(steps):
        ret_loss, ret_grads = grad_fn(data)
    return ret_loss, ret_grads


def model0_split_manual(model, stage_index, num_layers):
    """pipeline parallel split."""
    for i in range(num_layers):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def model1_split_manual(model, stage_index, num_layers):
    """pipeline parallel split."""
    for i in range(num_layers):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def run_parallel(micro_batch_num):
    """
    Feature: HSDP + SHARD + PP.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    # export RANK_ID = 3
    ms.set_seed(0)
    # ms.runtime.launch_blocking()
    init("hccl")

    # pp config
    num_stages = 4
    micro_batch_num = micro_batch_num
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    local_batch_size = micro_batch_num
    local_hidden_size = 16

    model0 = SimpleMLP(4, 16, 16)
    model1 = SimpleMLP(4, 16, 16)
    model0_split_manual(model0, stage_index, 4)
    model1_split_manual(model1, stage_index, 4)
    # shard config
    dp = 1
    mp = 2
    rank_list = [device_num_per_stage * stage_index + i for i in range(device_num_per_stage)]
    layout = Layout((dp, mp), ("dp", "mp"), rank_list)
    if stage_index == 0:
        in_layout = layout("dp", "mp")
        w_layout = layout("mp", "None")
        out_layout = layout("dp", "None")
    else:
        in_layout = layout("dp", "None")
        w_layout = layout("None", "None")
        out_layout = layout("dp", "None")

    model_stra = {"forward": {"input": (in_layout,), "output": (out_layout,)},
                  "parameter": {"weight": w_layout}}
    print("\nmodel0 层数:", len(model0.mlp_layers))
    print("model0 结构:")
    for name, cell in model0.mlp_layers.items():
        print(f"  层 {name}: {cell}")
        if hasattr(cell, 'weight'):
            print(f"    有 weight 参数")

    print("\nmodel1 层数:", len(model1.mlp_layers))
    print("model1 结构:")
    for name, cell in model1.mlp_layers.items():
        print(f"  层 {name}: {cell}")
        if hasattr(cell, 'weight'):
            print(f"    有 weight 参数")
    shard(model0, model_stra)
    shard(model1, model_stra)

    model0 = hsdp(model0, 2, 0, "level1", True)
    model1 = hsdp(model1, 2, 0, "level1", True)

    # optimizer = nn.Adam(model0.trainable_params() + model1.trainable_params(), learning_rate=learning_rate)

    # pp stage
    pipeline_stage0 = PipelineStage(model0,stage_index , num_stages + 4)
    pipeline_stage1 = PipelineStage(model1,stage_index + 4,num_stages + 4)
    schedule = ScheduleInterleaved1F1B([pipeline_stage0, pipeline_stage1], micro_batch_num)

    # input
    x_layout = layout("dp", "mp")
    local_batch_size = 8
    local_hidden_size = 16
    x = DTensor.from_local(Tensor(np.ones((local_batch_size, local_hidden_size)), dtype=ms.float32), x_layout)

    #optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)

    # train config
    epochs = 1
    for epoch in range(epochs):
        model0.zero_grads()
        model1.zero_grads()
        if stage_index == 0:
            loss, grads = schedule.run(x)
        else:
            loss, grads = schedule.run()
        # optimizer(grads)
    return loss, grads


def vpp(micro):
    """
    Feature: HSDP + SHARD + PP.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    standalone_loss, standalone_grads = run_standalone(micro*2)
    pp_loss, pp_grads = run_parallel(micro)
    num_stages = 4
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    if stage_index == 3:
        assert np.allclose(standalone_loss[:1, :].asnumpy(), pp_loss[0].asnumpy())
    expect_grad = None
    if stage_index in [1, 2]:
        expect_grad = standalone_grads[stage_index].asnumpy()
    else:
        expect_grad = standalone_grads[0].asnumpy()
    assert np.allclose(expect_grad[:8, :], pp_grads[0].asnumpy())

def test():
    micro = 8
    vpp(micro)
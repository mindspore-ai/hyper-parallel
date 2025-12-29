# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test data parallel"""
import numpy as np
import torch
from torch import nn
from torch import optim
# pylint: disable=W0611
import torch_npu
from hyper_parallel import hsdp, DTensor, Layout, SkipDTensorDispatch
from tests.torch.common_net import SimpleModel
from tests.torch.utils import init_dist
torch.manual_seed(0)
standalone_x = torch.rand(8, 8)


def get_standalone_result(step, acc_grad=False):
    """
    Get results from standalone (non-distributed) model training for comparison.
    
    Args:
        step (int): Number of training steps to execute
        acc_grad (bool): Whether to accumulate gradients without zeroing, defaults to False
        
    Returns:
        tuple: (loss tensor, gradient tensor) from standalone training
    """
    standalone_model = SimpleModel().npu()
    standalone_optimizer = optim.SGD(standalone_model.parameters(), lr=0.01)
    acc_step = 2
    if step < acc_step:
        step = 2 * acc_step
    acc_epoch = step // acc_step
    for _ in range (acc_epoch):
        for _ in range(acc_step):
            standalone_loss = standalone_model(standalone_x.npu())
            standalone_loss.backward()
            standalone_grad = standalone_model.weight.grad.data.clone()  # using for validation
            if not acc_grad:
                standalone_optimizer.step()
                standalone_optimizer.zero_grad()
        if acc_grad:
            standalone_optimizer.step()
            standalone_optimizer.zero_grad()
    return standalone_loss, standalone_grad

def get_hsdp_result(step, shard_size=1, optimizer_level="level1", acc_grad=False):
    """
    Get results from HSDP (Hybrid Sharded Data Parallel) distributed training.
    
    Args:
        step (int): Number of training steps to execute
        shard_size (int): Size of parameter sharding, defaults to 1
        optimizer_level (str): Optimization level ("level1", "level2", "level3"), defaults to "level1"
        acc_grad (bool): Whether to accumulate gradients without zeroing, defaults to False
        
    Returns:
        tuple: (loss tensor, gradient tensor) from distributed HSDP training
    """
    dist_model = SimpleModel().npu()
    layout = Layout((4, 2), ("dp", "tp"))
    x_layout = layout("None", "None")
    w_layout = layout("None", "tp")
    dist_x = DTensor.from_local(standalone_x.npu(), x_layout)
    local_w = torch.ones(8, 4).npu()
    # pylint: disable=W0212
    for key, param in dist_model._parameters.items():
        if param is not None and not isinstance(param, DTensor):
            dist_model.register_parameter(
                key,
                nn.Parameter(DTensor.from_local(local_w, w_layout)),
            )
    dist_model = hsdp(dist_model, shard_size=shard_size, threshold=0, enable_grad_accumulation=acc_grad,
                      optimizer_level=optimizer_level)
    dist_optimizer = optim.SGD(dist_model.parameters(), lr=0.01)

    acc_step = 2
    if step < acc_step:
        step = 2 * acc_step
    acc_epoch = step // acc_step
    for _ in range (acc_epoch):
        for i in range(acc_step):
            if i == acc_step - 1:
                dist_model.set_requires_grad_sync(True)
            else:
                dist_model.set_requires_grad_sync(False)
            dist_loss = dist_model(dist_x)
            dist_loss = dist_loss.reduce_partial()  # handle partial state

            # handle backward input
            repeat_num = dist_loss.layout.repeat_num()
            backward_input = torch.tensor(1.0 / repeat_num)
            dist_loss.backward(backward_input)

            dist_grad = dist_model.weight.grad.data.clone()
            if not acc_grad:
                with SkipDTensorDispatch():
                    dist_optimizer.step()
                    dist_model.zero_grads()
        if acc_grad:
            with SkipDTensorDispatch():
                dist_optimizer.step()
                dist_model.zero_grads()
    return dist_loss, dist_grad

def test_pure_data_parallel():
    """test data parallel"""
    rank, _ = init_dist()
    step = 4
    standalone_loss, standalone_grad = get_standalone_result(step)
    dist_loss, dist_grad = get_hsdp_result(step)

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.to_local().cpu().detach().numpy(),
                       0.001, 0.001)
    offset = rank % 2 * 4
    assert np.allclose(standalone_grad.cpu().detach().numpy()[:, offset: offset + 4],
                       dist_grad.cpu().detach().numpy(),
                       0.001, 0.001)

def shard_param_data_parallel(shard_size, optimizer_level="level1", acc_grad=False):
    """shard param data parallel"""
    rank, _ = init_dist()
    step = 4
    standalone_loss, standalone_grad = get_standalone_result(step, acc_grad=acc_grad)
    dist_loss, dist_grad = get_hsdp_result(step, shard_size=shard_size, optimizer_level=optimizer_level,
                                           acc_grad=acc_grad)

    assert np.allclose(standalone_loss.cpu().detach().numpy(),
                       dist_loss.to_local().cpu().detach().numpy(),
                       0.001, 0.001)
    dp_stride = 8 // shard_size
    tp_offset = rank % 2 * 4
    dp_offset = rank // 2 % shard_size * dp_stride
    assert np.allclose(standalone_grad.cpu().detach().numpy()[dp_offset: dp_offset+dp_stride, tp_offset: tp_offset+4],
                       dist_grad.cpu().detach().numpy(),
                       0.001, 0.001)

def test_zero1_fully_shard():
    """test zero1 fully shard parallel"""
    shard_param_data_parallel(shard_size=4)

def test_zero1_partial_shard():
    """test zero1 partial shard parallel"""
    shard_param_data_parallel(shard_size=2)

def test_zero2_fully_shard():
    """test zero2 fully shard parallel"""
    shard_param_data_parallel(shard_size=4, optimizer_level="level2")

def test_zero2_partial_shard():
    """test zero2 partial shard parallel"""
    shard_param_data_parallel(shard_size=2, optimizer_level="level2")

def test_zero3_fully_shard():
    """test zero3 fully shard parallel"""
    shard_param_data_parallel(shard_size=4, optimizer_level="level3")

def test_zero3_partial_shard():
    """test zero3 partial shard parallel"""
    shard_param_data_parallel(shard_size=2, optimizer_level="level3")

def test_zero1_fully_shard_with_acc_grad():
    """test zero1 fully shard parallel with acc grad"""
    shard_param_data_parallel(shard_size=4, acc_grad=True)

def test_zero1_partial_shard_with_acc_grad():
    """test zero1 partial shard parallel with acc grad"""
    shard_param_data_parallel(shard_size=2, acc_grad=True)

def test_zero2_fully_shard_with_acc_grad():
    """test zero2 fully shard parallel with acc grad"""
    shard_param_data_parallel(shard_size=4, optimizer_level="level2", acc_grad=True)

def test_zero2_partial_shard_with_acc_grad():
    """test zero2 partial shard parallel with acc grad"""
    shard_param_data_parallel(shard_size=2, optimizer_level="level2", acc_grad=True)

def test_zero3_fully_shard_with_acc_grad():
    """test zero3 fully shard parallel with acc grad"""
    shard_param_data_parallel(shard_size=4, optimizer_level="level3", acc_grad=True)

def test_zero3_partial_shard_with_acc_grad():
    """test zero3 partial shard parallel with acc grad"""
    shard_param_data_parallel(shard_size=2, optimizer_level="level3", acc_grad=True)

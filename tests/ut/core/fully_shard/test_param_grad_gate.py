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
"""独立审查回归：预建参数视图、真实梯度门控、Hyper 重算与连续迭代。"""

import types

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.core.activation_memory.api import checkpoint
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState




@pytest.fixture(autouse=True)
def cpu_checkpoint_device():
    """固定 CPU 回归的重计算设备；例如安装 torch_npu 时也不触发设备初始化。"""
    previous = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        yield
    finally:
        DefaultDeviceType.set_device_type(previous)


@pytest.mark.parametrize("checkpoint_enabled", [False, True])
def test_param_grad_gate_handles_prebuilt_views(checkpoint_enabled: bool) -> None:
    """复现输入边先到的反例；例如两个步骤只能分别提交一次完整参数梯度。"""
    p = torch.nn.Parameter(torch.tensor([[2.0]]))
    q = torch.nn.Parameter(torch.tensor([[3.0]]))
    scheduler = object.__new__(HSDPSchedulerV2)
    scheduler._param_grad_gate = True
    scheduler._param_grad_expected = set()
    scheduler._param_grad_ready = set()
    scheduler._param_grad_handles = {}
    scheduler._param_grad_input_done = False
    scheduler.hsdp_state = types.SimpleNamespace(
        hsdp_params=[types.SimpleNamespace(unsharded_param=value) for value in (p, q)],
    )
    scheduler.cell = None
    scheduler.scheduler_state = None
    commits = []

    def commit(self, *_args):
        """替代真正的 RS 入口，确认门控放行时两个 Parameter 已累积。"""
        assert p.grad is not None and q.grad is not None
        commits.append((p.grad.clone(), q.grad.clone()))
        self.scheduler_state = FSDPSchedulerState.BACKWARD

    scheduler._hsdp_backward_hook = types.MethodType(commit, scheduler)

    class Expert(torch.autograd.Function):
        """同时返回输入与两参数梯度；保存输入以便触发真正的重计算。"""

        @staticmethod
        def forward(ctx, x, gate, down):
            ctx.save_for_backward(x)
            return x * gate * down

        @staticmethod
        def backward(ctx, gradient):
            x, = ctx.saved_tensors
            return gradient * 6, gradient * x * 3, gradient * x * 2

    for step in range(2):
        p.grad = q.grad = None
        scheduler.scheduler_state = None
        # 先创建 view：旧输入边钩子会在这两个 view 的梯度传播之前执行。
        cached = (p.t(), q.t())

        def body(x):
            if not is_recomputing():
                scheduler._arm_param_grad_gate()
            x, = PostBackwardFunction.apply(scheduler, x)
            return Expert.apply(x, *cached)

        x = torch.tensor([[4.0 + step]], requires_grad=True)
        output = (checkpoint(body, x, use_reentrant=False, early_stop=False)
                  if checkpoint_enabled else body(x))
        output.sum().backward()
        assert len(commits) == step + 1
        assert p.grad.item() == 3 * (4 + step)
        assert q.grad.item() == 2 * (4 + step)
        assert len(scheduler._param_grad_handles) == 2


@pytest.mark.parametrize("checkpoint_enabled", [False, True])
def test_param_grad_gate_preserves_no_sync_accumulation(checkpoint_enabled: bool) -> None:
    """真实HSDP累积方法与gate联测；例如两个微批12+15只在最后归约为27。"""
    p = torch.nn.Parameter(torch.tensor([[2.0]]))
    q = torch.nn.Parameter(torch.tensor([[3.0]]))
    params = []
    for value in (p, q):
        parameter = object.__new__(HSDPParamV2)
        parameter._unsharded_param = value
        parameter.unsharded_accumulated_grad = None
        parameter.reduce_dtype = torch.float32
        params.append(parameter)
    state = object.__new__(HSDPStateV2)
    state.hsdp_params = params
    state.module_name = "review_no_sync"
    state.comm_fusion_policy = types.SimpleNamespace(enable_comm_fusion=False)
    state.reshard_after_backward = False
    state._wait_prev_reduce_scatter = lambda: []
    state._wait_prev_reduce_scatter_without_all_reduce = lambda: None
    state._issue_prev_fused_all_reduce = lambda _groups: None
    reduced = []

    def reduce_once():
        """替换通信入口，真实post_backward和两种梯度累积方法仍照常运行。"""
        values = []
        for param in params:
            gradient = param.unsharded_accumulated_grad
            if gradient is None:
                gradient = param.unsharded_param.grad
            values.append(gradient.clone())
        reduced.append(values)

    state._issue_reduce_scatter_for_current_module = reduce_once
    scheduler = object.__new__(HSDPSchedulerV2)
    scheduler._param_grad_gate = True
    scheduler._param_grad_expected = set()
    scheduler._param_grad_ready = set()
    scheduler._param_grad_handles = {}
    scheduler._param_grad_input_done = False
    scheduler._gradient_extension = None
    scheduler.hsdp_state = state
    scheduler.cell = None
    scheduler._fsdp_group_post_pending = None
    scheduler.scheduler_state = None

    class Expert(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gate, down):
            ctx.save_for_backward(x)
            return x * gate * down

        @staticmethod
        def backward(ctx, gradient):
            x, = ctx.saved_tensors
            return gradient * 6, gradient * x * 3, gradient * x * 2

    for step in range(2):
        state.reduce_grads = step == 1
        scheduler.scheduler_state = None
        cached = (p.t(), q.t())

        def body(x):
            if not is_recomputing():
                scheduler._arm_param_grad_gate()
            x, = PostBackwardFunction.apply(scheduler, x)
            return Expert.apply(x, *cached)

        x = torch.tensor([[4.0 + step]], requires_grad=True)
        result = (checkpoint(body, x, use_reentrant=False, early_stop=False)
                  if checkpoint_enabled else body(x))
        result.sum().backward()
        assert scheduler.scheduler_state == FSDPSchedulerState.BACKWARD
        if step == 0:
            assert not reduced and p.grad is None and q.grad is None
            assert params[0].unsharded_accumulated_grad.item() == 12
            assert params[1].unsharded_accumulated_grad.item() == 8
    assert len(reduced) == 1
    assert reduced[0][0].item() == 27 and reduced[0][1].item() == 18
    assert len(scheduler._param_grad_handles) == 2













def test_backward_pre_hook_rearms_gate_once_per_backward():
    """两次独立反向不复用旧ready；例如 B2 完成后 B1 仍须等待本次两个 AccumulateGrad。"""
    scheduler = object.__new__(HSDPSchedulerV2)
    scheduler.scheduler_state = FSDPSchedulerState.BACKWARD
    scheduler._param_grad_gate = True
    scheduler._param_grad_expected = {11, 22}
    scheduler._param_grad_ready = {11, 22}
    scheduler._param_grad_input_done = True
    scheduler._is_root = False
    scheduler.cell = None
    scheduler.scheduler_ctx = types.SimpleNamespace(root_bp_state=False)
    scheduler._hsdp_backward_pre_hook = lambda *_args: setattr(
        scheduler, 'scheduler_state', FSDPSchedulerState.PRE_BACKWARD)
    gradient = torch.ones(1)
    assert scheduler._backward_pre_hook(gradient) is gradient
    assert not scheduler._param_grad_ready and not scheduler._param_grad_input_done
    assert scheduler._param_grad_expected == {11, 22}
    scheduler._param_grad_ready.add(11)
    # 同图另一个输出或重放再次进入 pre-hook，不能抹掉已经就绪的本次参数。
    assert scheduler._backward_pre_hook(gradient) is gradient
    assert scheduler._param_grad_ready == {11}


@pytest.mark.parametrize("late_hook", [False, True])
def test_user_post_accumulate_hooks_finish_before_fsdp_consumes(late_hook):
    """用户钩子先于归约消费；例如 p 的12被钩子改为24，不能提前清空成None。"""
    p = torch.nn.Parameter(torch.tensor([[2.0]]))
    q = torch.nn.Parameter(torch.tensor([[3.0]]))
    scheduler = object.__new__(HSDPSchedulerV2)
    scheduler._param_grad_gate = True
    scheduler._param_grad_expected = set()
    scheduler._param_grad_ready = set()
    scheduler._param_grad_handles = {}
    scheduler._param_grad_input_done = False
    scheduler.hsdp_state = types.SimpleNamespace(
        hsdp_params=[types.SimpleNamespace(unsharded_param=value) for value in (p, q)])
    scheduler.cell = None
    scheduler.scheduler_state = None
    events = []

    def user_hook(parameter):
        """模拟原位梯度修正；例如完整梯度12必须仍存在，随后变成24。"""
        assert parameter.grad is not None
        events.append(('user', parameter.grad.item()))
        parameter.grad.mul_(2)

    if not late_hook:
        for parameter in (p, q):
            parameter.register_post_accumulate_grad_hook(user_hook)
    # 原始参数视图先建立，保证输入边与真实参数累积是不同的 autograd 分支。
    cached = p.t(), q.t()
    scheduler._arm_param_grad_gate()
    x = torch.tensor([[4.0]], requires_grad=True)
    wrapped, = PostBackwardFunction.apply(scheduler, x)
    output = wrapped * cached[0] * cached[1]
    if late_hook:
        for parameter in (p, q):
            parameter.register_post_accumulate_grad_hook(user_hook)

    def consume(self, *_args):
        """模拟 FSDP 读取并清除梯度；例如必须读到修正后的24和16。"""
        events.append(('fsdp', p.grad.item(), q.grad.item()))
        p.grad = q.grad = None
        self.scheduler_state = FSDPSchedulerState.BACKWARD

    scheduler._hsdp_backward_hook = types.MethodType(consume, scheduler)
    output.sum().backward()
    assert len(events) == 2 and all(event[0] == 'user' for event in events)
    scheduler._backward_hook(force=True)
    assert events[-1] == ('fsdp', 24.0, 16.0)
    assert p.grad is None and q.grad is None

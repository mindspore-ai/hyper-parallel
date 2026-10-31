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
"""不保存专家权重的两次前向/四次反向GMM与激活导数。"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn import functional


GroupedOp = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ExpertSaved:
    """保存反向必要激活；字段中没有专家权重。

    Example:
        rank 1 的 [d,b,c,d] 专家主序输入产生 x[4,2]、preact[4,2]、
        intermediate[4,1] 和 boundaries=[1,2,4]，权重随后可释放。
    """

    x: torch.Tensor
    preact: torch.Tensor
    intermediate: torch.Tensor
    boundaries: torch.Tensor
    limit: float
    activation: str = "clipped"


@torch.no_grad()
def expert_forward_values(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    boundaries: torch.Tensor,
    limit: float,
    gmm: GroupedOp,
    activation: str = "clipped",
) -> tuple[torch.Tensor, ExpertSaved]:
    """执行两次 GMM，并只返回保存激活的反向上下文。

    Args:
        x: [C,H] 专家主序输入。
        gate_up: [A,H,2I] 当前物化的权重。
        down: [A,I,H] 当前物化的权重，与 gate_up 专家顺序相同。
        boundaries: [A] int64 累计分组边界，已在设备上。
        limit: SwiGLU 限幅；非正数表示不限幅。
        gmm: 不保存 autograd 图的 group_type=0 grouped GEMM 接口。

    Returns:
        [C,H] 专家输出，以及仅含激活/边界/限幅配置的 ExpertSaved。

    Example:
        x=[d,b,c,d]、boundaries=[1,2,4] 时，依次计算 e0/e2/e3；
        输出为 [F0(d),F2(b),F3(c),F3(d)]，等待外部恢复接收顺序。
    """
    preact = gmm(x, gate_up, boundaries)
    if activation == "swiglu":
        intermediate = swiglu_values(preact)
        output = gmm(intermediate, down, boundaries)
        return output, ExpertSaved(x, preact, intermediate, boundaries, limit, activation)
    if activation != "clipped":
        raise ValueError("未知的专家激活类型")
    gate, up = preact.float().chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    intermediate = (functional.silu(gate) * up).to(x.dtype)
    output = gmm(intermediate, down, boundaries)
    return output, ExpertSaved(x, preact, intermediate, boundaries, limit)


@torch.no_grad()
def expert_backward_values(
    grad_output: torch.Tensor,
    saved: ExpertSaved,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gmm: GroupedOp,
    gmm_weight_grad: GroupedOp,
    on_weight_gradients: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用重新物化的权重计算四次 GMM 反向，不重跑前向矩阵乘。

    Args:
        grad_output: [C,H] 专家主序输出梯度。
        saved: 对应同一层、同一 microbatch 前向保存的激活。
        gate_up: 与前向数值/版本相同的 [A,H,2I] 重建权重。
        down: 与前向数值/版本相同的 [A,I,H] 重建权重。
        gmm: 输入梯度所需的 group_type=0 接口。
        gmm_weight_grad: 输入 X[N,J],dY[N,K]，返回 [A,J,K] 分组梯度。
        on_weight_gradients: 两份 dW 就绪后调用；例如先发 EP 回传再计算 dX GMM。

    Returns:
        dx[C,H]、dgate[A,H,2I]、ddown[A,I,H]；后两者尚未归还 home。

    Example:
        rank 1 产生 e0/e2/e3 三组梯度；外部先将 e0 梯度回传 rank 0，
        e2/e3 留作本地参数贡献，再交给 FSDP 归约。
    """
    grad_output = grad_output.contiguous()
    boundaries = saved.boundaries
    grad_mid = gmm(grad_output, down.transpose(-1, -2), boundaries)
    grad_down = gmm_weight_grad(saved.intermediate, grad_output, boundaries)
    if saved.activation == "swiglu":
        grad_preact = swiglu_grad(grad_mid, saved.preact)
        if on_weight_gradients is None:
            grad_x = gmm(grad_preact, gate_up.transpose(-1, -2), boundaries)
            grad_gate_up = gmm_weight_grad(saved.x, grad_preact, boundaries)
        else:
            grad_gate_up = gmm_weight_grad(saved.x, grad_preact, boundaries)
            on_weight_gradients(grad_gate_up, grad_down)
            grad_x = gmm(grad_preact, gate_up.transpose(-1, -2), boundaries)
        return grad_x, grad_gate_up, grad_down
    raw_gate, raw_up = saved.preact.float().chunk(2, dim=-1)
    gate, up = raw_gate, raw_up
    if saved.limit > 0:
        gate = raw_gate.clamp(max=saved.limit)
        up = raw_up.clamp(-saved.limit, saved.limit)
    grad_gate = torch.ops.aten.silu_backward(grad_mid.float() * up, gate)
    grad_up = grad_mid.float() * functional.silu(gate)
    if saved.limit > 0:
        # 与 Torch clamp 的端点导数保持一致：恰等于边界时仍传递梯度。
        grad_gate = grad_gate * (raw_gate <= saved.limit)
        grad_up = grad_up * ((raw_up >= -saved.limit) & (raw_up <= saved.limit))
    grad_preact = torch.cat((grad_gate, grad_up), dim=-1).to(saved.preact.dtype)
    if on_weight_gradients is None:
        grad_x = gmm(grad_preact, gate_up.transpose(-1, -2), boundaries)
        grad_gate_up = gmm_weight_grad(saved.x, grad_preact, boundaries)
    else:
        grad_gate_up = gmm_weight_grad(saved.x, grad_preact, boundaries)
        on_weight_gradients(grad_gate_up, grad_down)
        grad_x = gmm(grad_preact, gate_up.transpose(-1, -2), boundaries)
    return grad_x, grad_gate_up, grad_down


def swiglu_values(preact: torch.Tensor) -> torch.Tensor:
    """复用基线相同的SwiGLU；例如[C,1536]变为[C,768]且保持BF16舍入。"""
    if preact.device.type == "npu":
        # 可选昇腾依赖只在实际设备路径加载，CPU逻辑测试不要求安装。
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_swiglu(preact, dim=-1)
    gate, up = preact.chunk(2, dim=-1)
    return functional.silu(gate) * up


def swiglu_grad(gradient: torch.Tensor, preact: torch.Tensor) -> torch.Tensor:
    """复用现有SwiGLU反向；例如[C,I]梯度恢复为两投影的[C,2I]梯度。"""
    if preact.device.type == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_swiglu_backward(gradient.contiguous(), preact, dim=-1)
    gate, up = preact.chunk(2, dim=-1)
    return torch.cat((torch.ops.aten.silu_backward(gradient * up, gate),
                      gradient * functional.silu(gate)), dim=-1)


@torch.no_grad()
def grouped_values(x: torch.Tensor, weight: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """不构造权重autograd缓存的分组乘法；例如两组输入分别乘weight[0]/weight[1]。"""
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[-1]))
    if x.device.type == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_grouped_matmul(
            x=[x], weight=[weight], bias=[], group_list=boundaries,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
    begin = 0
    outputs = []
    for index, end in enumerate(boundaries.tolist()):
        outputs.append(x[begin:end] @ weight[index])
        begin = end
    return torch.cat(outputs)


@torch.no_grad()
def grouped_weight_grad(x: torch.Tensor, gradient: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """group_type=2聚合每专家dW；例如X[C,H]与dU[C,2I]生成[A,H,2I]。"""
    if x.shape[0] == 0:
        return x.new_zeros((boundaries.numel(), x.shape[1], gradient.shape[1]))
    if x.device.type == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_grouped_matmul(
            x=[x.transpose(-1, -2)], weight=[gradient.contiguous()], bias=[], group_list=boundaries,
            split_item=2, group_type=2, group_list_type=0,
        )[0]
    begin = 0
    outputs = []
    for end in boundaries.tolist():
        outputs.append(x[begin:end].T @ gradient[begin:end])
        begin = end
    return torch.stack(outputs)

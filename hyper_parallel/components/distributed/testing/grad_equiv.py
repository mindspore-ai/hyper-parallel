# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""testing.grad_equiv: M_D.15a 双模式梯度等价工具（05 §5.5 修订版）。

自研 DTensor 前向-only（05 §1.0）：production（FSDP/tp_grad_info 旁路）与
validate（local autograd 直出）的 backward **均为 local tensor 路径**——不存在
"DTensor backward" 对照组。因此双模式梯度等价直接逐参数比较：

- TP-Shard 参数：两模式梯度天然是 local shard，逐 rank 相等（免同步）；
- TP-Replicate 参数：两模式梯度同为 Partial 贡献，逐 rank 相等；
  与单卡参考梯度比较前需先经 tp_grad_info 旁路 all-reduce（本模块提供模拟；
  真实 FSDP2 fork 路径属 M_M.2a 联调）。
"""

import torch
import torch.distributed as dist


def run_one_step(model, input_ids, labels, vocab_size):
    """单步 forward+backward，返回 {param_fqn: grad}。"""
    model.zero_grad()
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size).float(), labels.reshape(-1))
    loss.backward()
    return loss, {
        name: (param.grad.clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }


def assert_grad_equivalence(prod_grads, val_grads, *, rtol=1e-3, atol=1e-5):
    """双模式梯度逐参数 assert_close（跳过两侧均缺失的参数）。"""
    for name, gp in prod_grads.items():
        gv = val_grads.get(name)
        if gp is None and gv is None:
            continue
        assert gp is not None, f"{name}: production 缺梯度"
        assert gv is not None, f"{name}: validate 缺梯度"
        torch.testing.assert_close(gp, gv, rtol=rtol, atol=atol)


def simulate_tp_replicate_grad_sync(grad, tp_group):
    """模拟 tp_grad_info 旁路：TP-Replicate 参数梯度的 TP all-reduce。

    真实路径由 FSDP2 fork 的 all_reduce_grad 完成（M_M.2a 联调）；此处用于
    独立开发阶段的梯度等价验证。
    """
    synced = grad.clone()
    dist.all_reduce(synced, group=tp_group)
    return synced

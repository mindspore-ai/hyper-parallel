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
"""FunctionModule — autograd.Function 的模块壳（教程 §10.8）。

边界机制的作用粒度是 ``nn.Module.forward``；自定义 ``autograd.Function``
以 ``A.apply(...)`` 裸调用时对框架不可见（没有 FQN 就没有 spec 挂载点）。
``FunctionModule`` 给 Function 一个模块形态：无参数、无状态，forward 透传
``fn.apply``，backward 走 Function 自己的静态 backward——壳对 autograd
透明。

用法（plan_overrides 声明边界；自定义 Function 不在 DTensor dispatch 覆盖
范围，必须 ``region_dispatch=False``）::

    self.a_fn = FunctionModule(A)
    plan_overrides={"...a_fn": ModuleShardingSpec(
        params={}, region_dispatch=False,
        in_src={"x": ...}, in_dst={"x": ...},
        out_src={...}, out_dst={...})}

契约 key 绑定（与所有边界一致，``_bind_input_indices``）：先按壳 forward
的形参名绑定；**单输入契约回退绑定到第 0 个位置参数**——因此单张量输入
直接用本壳的 ``*args`` 透传即可。多输入（如额外权重张量）请子类化并给出
显式签名::

    class SeqNormWithWeight(FunctionModule):
        def forward(self, x, weight):
            return self._fn.apply(x, weight)
"""

import torch.nn as nn


class FunctionModule(nn.Module):
    """Wrap an ``autograd.Function`` class as an ``nn.Module`` (boundary FQN
    mount point).  Transparent to autograd: backward is the Function's own
    static ``backward``."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn.apply(*args, **kwargs)

    def extra_repr(self) -> str:
        return f"fn={self._fn.__name__}"

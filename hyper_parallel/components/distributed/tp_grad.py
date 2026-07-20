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
"""tp_grad: build_tp_grad_info（05 §6.7.1）。

tp_grad_info 从 ShardingPlan 读取（而非 DTensor——production 下参数已被
_local_params_context 解包，只有 plan 保留完整 placement 信息）。
"""

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def build_tp_grad_info(plan, tp_mesh, *, tied_pairs=None):
    """{param_fqn: (tp_placement, tp_mesh)}，tp_placement in {Shard, Replicate}。

    tied_pairs: 共享存储的参数对（默认取 plan.tied_pairs）。tied 对必须映射到
    同一 tp_placement——placement 不一致时取较细分片（Shard 优先于 Replicate），
    保证两端 TP all-reduce / reduce-scatter 语义一致。
    """
    info = {}
    for fqn, spec in plan.modules.items():
        for param_name, named_placement in spec.params.items():
            full_fqn = f"{fqn}.{param_name}"
            tp_placement = named_placement.get("tp", Replicate())
            info[full_fqn] = (tp_placement, tp_mesh)

    pairs = tied_pairs if tied_pairs is not None else getattr(plan, "tied_pairs", None)
    if pairs:
        for a, b in pairs:
            if a in info and b in info:
                pa, _ = info[a]
                pb, _ = info[b]
                if pa != pb:
                    norm = pa if isinstance(pa, Shard) else pb
                    info[a] = (norm, tp_mesh)
                    info[b] = (norm, tp_mesh)
    return info

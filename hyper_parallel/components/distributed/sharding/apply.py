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
"""sharding.apply: _local_params_context / 路径工具（canonical 定义，05 §4.4）。

06 的 dtensor_utils.py re-export 本模块定义，勿另起副本。
"""

import logging

import torch.nn as nn

from hyper_parallel.core.dtensor.dtensor import DTensor

logger = logging.getLogger(__name__)


def _get_attr_by_path(model, fqn):
    """沿点分 FQN 取属性（数字段走 ModuleList 索引）。"""
    obj = model
    for p in fqn.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _set_param_by_path(model: nn.Module, fqn: str, new_param) -> None:
    """沿点分 FQN 定位父模块并替换 leaf 参数。

    object.__setattr__(model, dotted_name, ...) 只会在 model 上设一个怪属性，
    不会替换子模块参数——必须沿路径定位到真正的父模块再赋值。
    """
    *path, leaf = fqn.split(".")
    obj = model
    for p in path:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    if hasattr(obj, "register_parameter"):
        obj.register_parameter(leaf, new_param)
    else:
        object.__setattr__(obj, leaf, new_param)


def _resolve_module(model, fqn):
    """按 FQN 取模块（不剥离末段，调用点传模块 FQN）。

    与 _get_attr_by_path 同语义——所有调用点（Phase A/B/C）传入的 fqn 均为
    模块完全限定名（如 `model.layers.0.self_attn`），而非参数 FQN，故不做
    末段剥离（剥离会错误返回父模块）。
    """
    obj = model
    for p in fqn.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _local_params_context(model: nn.Module):
    """build 期一次性解包：把 DTensor 参数替换为 _local_tensor（plain），零拷贝。

    在 apply_sharding_plan 的 Phase C 入口、fully_shard 之前调用，永久解包不恢复。
    _local_tensor 与原 DTensor 共享存储（data_ptr 相同）。

    返回 {fqn: placements} 解包前的 placement 快照（仅诊断用途；tp_grad_info 的
    canonical 来源是 ShardingPlan，见 build_tp_grad_info）。
    """
    tp_grad_records = {}
    for name, param in list(model.named_parameters()):
        if isinstance(param, DTensor):
            tp_grad_records[name] = param.placements
            _set_param_by_path(model, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    return tp_grad_records

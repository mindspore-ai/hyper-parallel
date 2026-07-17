# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.1: 路径工具 _resolve_module / _get_attr_by_path / _set_param_by_path。"""

import torch
import torch.nn as nn

from hyper_models.components.distributed.sharding.apply import (
    _get_attr_by_path,
    _resolve_module,
    _set_param_by_path,
)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.lm_head = nn.Linear(4, 4)


class TestPathUtils:
    def test_resolve_module_nested_modulelist(self):
        net = _Net()
        assert _resolve_module(net, "model.layers.0") is net.model.layers[0]
        assert _resolve_module(net, "model.layers.1") is net.model.layers[1]
        assert _resolve_module(net, "lm_head") is net.lm_head

    def test_resolve_module_no_leaf_strip(self):
        """不剥离末段：传模块 FQN 返回模块本身，而非父模块。"""
        net = _Net()
        mod = _resolve_module(net, "model.layers.0")
        assert isinstance(mod, nn.Linear)

    def test_get_attr_by_path_param(self):
        net = _Net()
        w = _get_attr_by_path(net, "model.layers.0.weight")
        assert w is net.model.layers[0].weight

    def test_set_param_by_path_register_parameter(self):
        net = _Net()
        new_w = nn.Parameter(torch.ones(4, 4))
        _set_param_by_path(net, "model.layers.1.weight", new_w)
        assert net.model.layers[1].weight is new_w
        # register_parameter 路径：在 _parameters 中
        assert net.model.layers[1]._parameters["weight"] is new_w

    def test_set_param_by_path_setattr_branch(self):
        class Plain:
            pass
        obj = Plain()
        p = nn.Parameter(torch.ones(2))
        _set_param_by_path(obj, "w", p)
        assert obj.w is p

    def test_set_param_by_path_numeric_segment(self):
        net = _Net()
        new_w = nn.Parameter(torch.zeros(4, 4))
        _set_param_by_path(net, "model.layers.0.bias",
                           nn.Parameter(torch.zeros(4)))
        _set_param_by_path(net, "model.layers.0.weight", new_w)
        assert net.model.layers[0].weight is new_w

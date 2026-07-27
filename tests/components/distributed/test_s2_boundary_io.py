# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.5: redistribute_inputs/outputs + _get_arg/_set_arg 双通道（单进程 mock mesh）。

identity op 不需要真实通信——用 1-rank mesh 即可覆盖参数通道逻辑；
通信数值由 S2.3 多进程用例覆盖。
"""

import torch
import torch.distributed as dist
import pytest

from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
    RedistOp,
    _get_arg,
    _set_arg,
)
from hyper_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard


@pytest.fixture(scope="module")
def mesh():
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29721", rank=0, world_size=1)
    return init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))


class TestGetSetArg:
    def test_kwargs_hit_priority(self):
        args, kwargs = (torch.tensor([1]),), {"x": torch.tensor([2])}
        got = _get_arg(args, kwargs, "x", 0)
        assert got.item() == 2  # kwargs 优先于 args[0]

    def test_args_idx_fallback(self):
        args, kwargs = (torch.tensor([1]),), {}
        got = _get_arg(args, kwargs, "x", 0)
        assert got.item() == 1

    def test_missing_returns_default(self):
        assert _get_arg((), {}, "x", None, default="d") == "d"

    def test_idx_out_of_range_falls_to_kwargs(self):
        args, kwargs = (), {}
        args, kwargs = _set_arg(args, kwargs, "x", 5, "v")
        assert kwargs == {"x": "v"}

    def test_set_arg_kwargs_channel(self):
        args, kwargs = (), {"x": 1}
        args, kwargs = _set_arg(args, kwargs, "x", None, 2)
        assert kwargs["x"] == 2

    def test_set_arg_args_channel(self):
        args, kwargs = (1, 2), {}
        args, kwargs = _set_arg(args, kwargs, "x", 1, 9)
        assert args == (1, 9)


class TestRedistributeIO:
    def test_inputs_kwargs_hit(self, mesh):
        spec = ModuleShardingSpec(
            in_src={"x": {TP: Replicate()}}, in_dst={"x": {TP: Replicate()}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        t = torch.tensor([1.0])
        args, kwargs = b.redistribute_inputs((), {"x": t})
        assert kwargs["x"] is t  # identity 直通

    def test_inputs_missing_arg_skipped(self, mesh):
        """arg 未找到（None）→ 跳过，不向 kwargs 注入 None。"""
        spec = ModuleShardingSpec(
            in_src={"input": {TP: Replicate()}}, in_dst={"input": {TP: Replicate()}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        args, kwargs = b.redistribute_inputs((torch.tensor([1]),), {})
        assert "input" not in kwargs

    def test_outputs_single(self, mesh):
        spec = ModuleShardingSpec(
            out_src={"output": {TP: Shard(1)}}, out_dst={"output": {TP: Shard(1)}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        # identity out_plan 为空 → 原样返回
        t = torch.tensor([1.0])
        assert b.redistribute_outputs(t) is t

    def test_outputs_tuple_order_preserved(self, mesh):
        spec = ModuleShardingSpec(
            out_src={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
            out_dst={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
            out_names=["a", "b"],
        )
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        ta, tb = torch.tensor([1.0]), torch.tensor([2.0])
        out = b.redistribute_outputs((ta, tb))
        assert isinstance(out, tuple) and out[0] is ta and out[1] is tb

    def test_outputs_index_out_of_range_warns_and_skips(self, mesh, caplog):
        spec = ModuleShardingSpec(
            out_src={"a": {TP: Shard(1)}, "b": {TP: Partial()}},
            out_dst={"a": {TP: Replicate()}, "b": {TP: Replicate()}},
            out_names=["a", "b"],
        )
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        import logging
        with caplog.at_level(logging.WARNING):
            out = b.redistribute_outputs((torch.tensor([[1.0]]),))
        assert "Skipping" in caplog.text
        assert len(out) == 1

    def test_inputs_as_dtensor(self, mesh):
        spec = ModuleShardingSpec(
            in_src={"x": {TP: Replicate()}}, in_dst={"x": {TP: Replicate()}})
        b = PrecompiledBoundary(spec, mesh, ("tp",))
        from hyper_parallel.core.dtensor.dtensor import DTensor
        args, kwargs = b.redistribute_inputs((), {"x": torch.tensor([1.0])},
                                             as_dtensor=True)
        assert isinstance(kwargs["x"], DTensor)

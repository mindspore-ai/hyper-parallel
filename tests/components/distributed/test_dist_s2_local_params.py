# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.8（2 进程）: _local_params_context 零拷贝解包 + placement 快照。"""

import torch
import torch.nn as nn

from hyper_models.components.distributed.sharding.apply import (
    _local_params_context,
)
from hyper_models.components.distributed.sharding_config import TP
from hyper_models.components.distributed.sharding_applier import (
    _shard_module_params,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.components.distributed.conftest import run_dist


def _worker(rank, world_size):
    torch.manual_seed(0)
    lin = nn.Linear(16, 16, bias=False)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    _shard_module_params(lin, {"weight": {TP: Shard(0)}}, mesh, ("tp",))
    dt_local = lin.weight.to_local()
    dt_ptr = dt_local.data_ptr()

    records = _local_params_context(lin)
    # 解包后为 plain tensor
    assert not isinstance(lin.weight, DTensor)
    assert isinstance(lin.weight, nn.Parameter)
    # 零拷贝：data_ptr 与 DTensor._local_tensor 共享存储
    assert lin.weight.data_ptr() == dt_ptr
    # requires_grad 保留
    assert lin.weight.requires_grad
    # placement 快照
    assert "weight" in records
    assert tuple(records["weight"]) == (Shard(0),)
    # 数值不变
    torch.testing.assert_close(lin.weight.data, dt_local)

    # 无 DTensor 时 no-op
    records2 = _local_params_context(lin)
    assert records2 == {}


def test_local_params_context_2proc():
    run_dist(2, _worker)

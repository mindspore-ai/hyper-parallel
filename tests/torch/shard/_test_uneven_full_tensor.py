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
"""Real Gloo collectives and safetensors round-trip for uneven full weights."""

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout, _get_slice_tensor_by_layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.api import get_model_state_dict


def _check_export(full, mesh, placements):
    """Verify export values, detached storage, public state dict and serialization."""
    layout = Layout.from_device_mesh(mesh)
    layout.set_placements(placements)
    layout.placement_to_tensor_map(full.ndim)
    local = _get_slice_tensor_by_layout(full, layout).clone().requires_grad_()
    expected = full
    tensor = DTensor.from_local_with_layout(local, layout, shape=tuple(full.shape))
    with torch.no_grad():
        output = tensor.full_tensor()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not output.requires_grad, f"Unexpected export requires_grad={output.requires_grad}"
    assert output._base is None, f"Export must own storage, got base={output._base}"
    assert output.untyped_storage().nbytes() == output.numel() * output.element_size(), (
        f"Storage is not compact: bytes={output.untyped_storage().nbytes()}, numel={output.numel()}"
    )

    model = torch.nn.Module()
    model.register_parameter("A_log", torch.nn.Parameter(tensor))
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        torch.testing.assert_close(state["A_log"], expected, rtol=0, atol=0)
        with TemporaryDirectory(prefix="hp-uneven-export-") as directory:
            path = Path(directory) / "model.safetensors"
            save_file(state, path)
            restored = load_file(path)
            torch.testing.assert_close(restored["A_log"], expected, rtol=0, atol=0)
    else:
        assert state == {}, f"Non-writing rank retained weights: keys={list(state)}"
    dist.barrier()


def test_uneven_full_tensor_gloo():
    """Reconstruct and save uneven tensors using four real CPU processes."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        size = dist.get_world_size()
        mesh = DeviceMesh("cpu", list(range(size)), mesh_dim_names=("fsdp",))
        for length in (0, 2, 5, 6, 8, 48):
            _check_export(torch.arange(length, dtype=torch.float32), mesh, (Shard(0, uneven_shard=True),))
        _check_export(torch.arange(5, dtype=torch.bfloat16), mesh, (Shard(0, uneven_shard=True),))
        rank_grid = np.arange(size).reshape(2, size // 2)
        hybrid = DeviceMesh("cpu", rank_grid, mesh_dim_names=("dp", "tp"))
        reversed_mesh = DeviceMesh("cpu", rank_grid[:, ::-1].copy(), mesh_dim_names=("dp", "tp"))
        placements = (
            (Replicate(), Shard(0, uneven_shard=True)),
            (Shard(1), Shard(0, uneven_shard=True)),
            (Shard(0), Shard(0, uneven_shard=True)),
            (StridedShard(0, split_factor=size // 2, uneven_shard=True), Shard(0)),
        )
        for current_mesh in (hybrid, reversed_mesh):
            for placement in placements:
                _check_export(torch.arange(30, dtype=torch.float32).reshape(5, 6), current_mesh, placement)
        if dist.get_rank() == 0:
            print("Uneven full_tensor, detached storage and safetensors round-trip: PASS")
    finally:
        dist.destroy_process_group()

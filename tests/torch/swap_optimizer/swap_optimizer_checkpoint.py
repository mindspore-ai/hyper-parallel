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
"""Distributed checkpoint coverage for swap optimizer DTensor state."""

import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.torch.utils import init_dist_gloo


def _shared_checkpoint_dir() -> Path:
    """Create one temporary checkpoint directory shared by every rank."""
    path_holder = [None]
    if dist.get_rank() == 0:
        path_holder[0] = tempfile.mkdtemp(prefix="swap_optimizer_dtensor_")
    dist.broadcast_object_list(path_holder, src=0)
    return Path(path_holder[0])


def _dtensor(local: torch.Tensor, mesh: object, placements: tuple[object, ...]) -> DTensor:
    """Wrap a local shard with the test tensor's fixed global metadata."""
    return DTensor.from_local(
        local,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
        shape=(4, 4),
        stride=(4, 1),
    )


def _build_host_resident_swap_state(mesh: object, expected: torch.Tensor):
    """Build a swap optimizer whose Adam moments differ by global shard."""
    rank = dist.get_rank()
    local_moment = expected.chunk(dist.get_world_size(), dim=0)[rank].clone()
    placements = (Shard(0),)
    param = torch.nn.Parameter(_dtensor(torch.ones_like(local_moment), mesh, placements))
    optimizer = torch.optim.AdamW([param], lr=0.01)
    optimizer.state[param].update({
        "step": torch.tensor(1.0),
        "exp_avg": _dtensor(local_moment, mesh, placements),
        "exp_avg_sq": _dtensor(local_moment.square(), mesh, placements),
    })
    wrapped = swap_optimizer(
        optimizer,
        SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False),
    )
    # The host-resident test slots have no device work to synchronize.
    wrapped.runtime.current_stream = lambda: None
    for slot in wrapped.adapter.all_slots():
        slot.swappable = True
        slot.cpu_tensor = slot.tensor.to_local().detach().cpu().clone()
        slot.state = "host"
    return wrapped, local_moment


def test_swap_optimizer_checkpoint_reshards_native_dtensor() -> None:
    """Export distinct optimizer shards and reload them as replicated state."""
    init_dist_gloo()
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("dp",))
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    checkpoint_path = _shared_checkpoint_dir()

    try:
        wrapped, local_moment = _build_host_resident_swap_state(mesh, expected)
        exported = wrapped.state_dict()["state"][0]["exp_avg"]
        assert isinstance(exported, DTensor), (
            f"Swap checkpoint state must remain a DTensor: expected={DTensor}, got={type(exported)}"
        )
        assert tuple(exported.shape) == tuple(expected.shape), (
            f"Swap checkpoint global shape mismatch: expected={tuple(expected.shape)}, got={tuple(exported.shape)}"
        )
        torch.testing.assert_close(exported.to_local(), local_moment, rtol=0, atol=0)

        copied_state = {"exp_avg": exported}
        ChainedOptimizer._localize_state_dict_tensors(copied_state)  # pylint: disable=protected-access
        copied = copied_state["exp_avg"]
        assert isinstance(copied, DTensor), (
            f"Chained checkpoint copy must retain DTensor metadata: expected={DTensor}, got={type(copied)}"
        )
        assert tuple(copied.shape) == tuple(expected.shape), (
            f"Chained checkpoint global shape mismatch: expected={tuple(expected.shape)}, got={tuple(copied.shape)}"
        )
        torch.testing.assert_close(copied.to_local(), local_moment, rtol=0, atol=0)

        dcp.save({"exp_avg": exported}, checkpoint_id=checkpoint_path)
        metadata = dcp.FileSystemReader(checkpoint_path).read_metadata()
        saved_shape = tuple(metadata.state_dict_metadata["exp_avg"].size)
        assert saved_shape == tuple(expected.shape), (
            f"DCP recorded a local optimizer shard shape: expected={tuple(expected.shape)}, got={saved_shape}"
        )

        restored = _dtensor(torch.zeros_like(expected), mesh, (Replicate(),))
        target = {"exp_avg": restored}
        dcp.load(target, checkpoint_id=checkpoint_path)
        torch.testing.assert_close(restored.to_local(), expected, rtol=0, atol=0)

        restored_shard = _dtensor(torch.zeros_like(local_moment), mesh, (Shard(0),))
        dcp.load({"exp_avg": restored_shard}, checkpoint_id=checkpoint_path)
        checkpoint_state = wrapped.state_dict()
        checkpoint_state["state"][0]["exp_avg"] = restored_shard
        new_param = torch.nn.Parameter(_dtensor(torch.ones_like(local_moment), mesh, (Shard(0),)))
        new_wrapped = swap_optimizer(
            torch.optim.AdamW([new_param], lr=0.01),
            SwapOptimizerConfig(swap_times=2, min_numel=1, packed_swap=False),
        )
        new_wrapped.runtime.is_swappable_tensor = lambda _tensor, _min_numel: True
        new_wrapped.load_state_dict(checkpoint_state)
        loaded = new_wrapped.state[new_param]["exp_avg"]
        loaded_slot = next(slot for slot in new_wrapped.adapter.all_slots() if slot.name == "exp_avg")
        assert isinstance(loaded, DTensor), (
            f"Loaded swap state must keep its DTensor shell: expected={DTensor}, got={type(loaded)}"
        )
        assert tuple(loaded.shape) == tuple(expected.shape), (
            f"Loaded swap state global shape mismatch: expected={tuple(expected.shape)}, got={tuple(loaded.shape)}"
        )
        torch.testing.assert_close(loaded_slot.cpu_tensor, local_moment, rtol=0, atol=0)
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        dist.barrier()
        dist.destroy_process_group()

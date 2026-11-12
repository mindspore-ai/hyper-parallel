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
"""Real CPU/Gloo Hyper layout, loading and local-autograd boundary checks."""

import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor as TorchDTensor

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh as HyperMesh
from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor
from hyper_parallel.core.dtensor.placement_types import Replicate as HyperReplicate
from hyper_parallel.core.dtensor.placement_types import Shard as HyperShard
from hyper_parallel.integration import torch as compatibility
from hyper_parallel.integration.torch import DTensorAdapter
from hyper_parallel.integration.torch.dtensor import from_torch_view, to_torch_view
from hyper_parallel.integration.veomni.optimizer import HyperOptimizerAdapter


def test_adapter_gloo():
    """Compare both backends without replacing the training tensor or Torch API."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        global_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        configurations = ((HyperDTensor, HyperMesh, HyperShard, HyperReplicate),)
        for tensor_type, mesh_type, shard_type, replicate_type in configurations:
            mesh = mesh_type.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("shard",))
            for dim in (None, 0, 1):
                placements = (replicate_type() if dim is None else shard_type(dim),)
                local = global_tensor.clone() if dim is None else global_tensor.chunk(2, dim)[rank].clone().contiguous()
                tensor = tensor_type.from_local(local, mesh, placements, shape=(4, 4), stride=(4, 1))
                adapter = DTensorAdapter(tensor)
                pointer = adapter.to_local().data_ptr()
                for source_rank in (None, 0):
                    payload = global_tensor + 7 if source_rank is None or rank == 0 else torch.zeros_like(global_tensor)
                    adapter.copy_from_full_(payload, src_data_rank=source_rank)
                    expected_local = global_tensor + 7
                    if dim is not None:
                        expected_local = expected_local.chunk(2, dim)[rank]
                    torch.testing.assert_close(adapter.to_local(), expected_local)
                self_local = global_tensor.clone() if dim is None else global_tensor.chunk(2, dim)[rank].contiguous()
                adapter.copy_local_(self_local)
                adapter.copy_local_(local + 1)
                rebuilt = adapter.from_local(adapter.to_local())
                torch.testing.assert_close(rebuilt.to_local(), local)
                assert rebuilt.shape == (4, 4), f"Expected global shape (4, 4), got {rebuilt.shape}"
                assert rebuilt.device_mesh is mesh, f"Expected original mesh {mesh}, got {rebuilt.device_mesh}"
                assert isinstance(rebuilt, tensor_type), f"Expected {tensor_type}, got {type(rebuilt)}"

                grad = torch.ones_like(local)
                tensor.grad = grad
                torch.testing.assert_close(adapter.local_grad(), grad)
                tensor.grad = None
                adapter.to_local().requires_grad_(True)
                loss = adapter.to_local().square().sum()
                loss.backward()
                torch.testing.assert_close(adapter.local_grad(), 2 * adapter.to_local())
                actual_pointer = adapter.to_local().data_ptr()
                assert actual_pointer == pointer, f"Storage changed: expected {pointer}, got {actual_pointer}"
        from torch.distributed import tensor as torch_api  # pylint: disable=import-outside-toplevel

        assert torch_api.DTensor is TorchDTensor, f"Torch export changed to {torch_api.DTensor}"
    finally:
        dist.destroy_process_group()


def test_uneven_optimizer_checkpoint_gloo():
    """Restore empty/uneven FSDP states and match the next finite AdamW update."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    root = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        paths = [tempfile.mkdtemp(prefix="hyper-uneven-optim-") if rank == 0 else None]
        dist.broadcast_object_list(paths, src=0)
        root = Path(paths[0])
        mesh = HyperMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp_shard_sp",))
        for rows, width in ((4, 8), (1, 128), (1, 4096), (5, 8)):
            chunk = (rows + world - 1) // world
            local_rows = max(0, min(chunk, rows - rank * chunk))
            placements = (HyperShard(0, uneven_shard=rows % world != 0),)
            initial = torch.full((local_rows, width), 0.5)

            def build(initial=initial, placements=placements, rows=rows, width=width):
                model = torch.nn.Module()
                parameter = torch.nn.Parameter(HyperDTensor.from_local(
                    initial.clone(), mesh, placements, shape=(rows, width), stride=(width, 1)
                ))
                model.register_parameter("weight", parameter)
                leaf = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
                return model, HyperOptimizerAdapter(model, {"adamw": leaf})

            def step(model, optimizer, gradient, initial=initial, placements=placements, rows=rows, width=width):
                model.weight.grad = HyperDTensor.from_local(
                    torch.full_like(initial, gradient), mesh, placements,
                    shape=(rows, width), stride=(width, 1)
                )
                optimizer.step()
                optimizer.zero_grad()

            reference, reference_optimizer = build()
            step(reference, reference_optimizer, 1.0)
            checkpoint = {
                "model": compatibility.get_model_state_dict(reference),
                "optimizer": reference_optimizer.state_dict(),
            }
            destination = root / f"{rows}-{width}"
            dcp.save(checkpoint, checkpoint_id=destination)
            step(reference, reference_optimizer, 2.0)

            model, optimizer = build()
            parameter = model.weight
            pointer = parameter.to_local().data_ptr()
            state = {
                "model": compatibility.get_model_state_dict(model),
                "optimizer": optimizer.state_dict(),
            }
            dcp.load(state, checkpoint_id=destination)
            compatibility.set_model_state_dict(model, state["model"])
            optimizer.load_state_dict(state["optimizer"])
            assert model.weight is parameter, "Restore replaced the live parameter"
            assert parameter.to_local().data_ptr() == pointer, "Restore replaced local storage"
            assert parameter.placements == placements, "Restore lost FSDP chunk geometry"
            step(model, optimizer, 2.0)
            torch.testing.assert_close(model.weight.to_local(), reference.weight.to_local(), rtol=0, atol=0)
            expected = reference_optimizer.optimizers_dict["adamw"].state[reference.weight]
            actual = optimizer.optimizers_dict["adamw"].state[model.weight]
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        dist.barrier()
    finally:
        if dist.get_rank() == 0 and root is not None:
            shutil.rmtree(root)
        dist.destroy_process_group()


def test_checkpoint_compatibility_gloo():
    """Round-trip real shared-storage DCP views without importing VeOmni."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    root = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        paths = [tempfile.mkdtemp(prefix="hyper-torch-compat-") if rank == 0 else None]
        dist.broadcast_object_list(paths, src=0)
        root = Path(paths[0])
        mesh = compatibility.DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp",))
        for dim in (None, 0, 1):
            full = torch.arange(28, dtype=torch.float32).reshape(7, 4)
            local = full.clone() if dim is None else full.chunk(world, dim)[rank].clone().contiguous()
            placement = compatibility.Replicate() if dim is None else compatibility.Shard(dim)
            parameter = torch.nn.Parameter(compatibility.DTensor.from_local(
                local, mesh, [placement], shape=full.shape, stride=full.stride()
            ))
            model = torch.nn.Module()
            model.register_parameter("weight", parameter)
            optimizer = torch.optim.AdamW(model.parameters())
            pointer = parameter.to_local().data_ptr()
            view = to_torch_view(parameter)
            restored = from_torch_view(view)
            assert isinstance(parameter, HyperDTensor), f"Expected Hyper parameter, got {type(parameter)}"
            assert isinstance(view, TorchDTensor), f"Expected Torch transport, got {type(view)}"
            for value in (view, restored):
                actual = value.to_local().data_ptr()
                assert actual == pointer, f"Storage changed: expected {pointer}, got {actual}"
                assert value.shape == full.shape, f"Expected shape {full.shape}, got {value.shape}"
            state = compatibility.get_model_state_dict(model)
            destination = root / f"dim-{dim}"
            dcp.save(state, checkpoint_id=destination)
            with torch.no_grad():
                parameter.to_local().zero_()
            dcp.load(state, checkpoint_id=destination)
            compatibility.set_model_state_dict(model, state)
            expected = full if dim is None else full.chunk(world, dim)[rank]
            torch.testing.assert_close(parameter.to_local(), expected)
            assert model.weight is parameter, f"Parameter identity changed to {id(model.weight)}"
            bound = optimizer.param_groups[0]["params"][0]
            assert bound is parameter, f"Optimizer reference changed to {id(bound)}"
            actual = parameter.to_local().data_ptr()
            assert actual == pointer, f"Storage changed: expected {pointer}, got {actual}"
        dist.barrier()
    finally:
        if dist.get_rank() == 0 and root is not None:
            shutil.rmtree(root)
        dist.destroy_process_group()

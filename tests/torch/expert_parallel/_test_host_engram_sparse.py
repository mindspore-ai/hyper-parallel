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
"""Four-rank Host Engram toy fixture from the design document."""

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.adapter.optim.host_sparse import HostSparseOptimizerCoordinator
from hyper_parallel.models.external_state import CheckpointRuntime


class _DenseOptimizer:
    """Expose a single SGD leaf to the Host coordinator."""

    def __init__(self, parameter: nn.Parameter) -> None:
        """Build the minimal SGD leaf used by the sparse coordinator."""
        self.leaf = torch.optim.SGD([parameter], lr=0.01)
        self.optimizers_dict = {"dense": self.leaf}

    def step(self, closure: Any = None) -> Any:
        """Update the dense fixture parameter.

        Args:
            closure: Optional optimizer closure.
        """
        return self.leaf.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear the dense fixture gradient.

        Args:
            set_to_none: Clear gradients by assigning None when true.
        """
        self.leaf.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return SGD state for the coordinator's dense branch."""
        return self.leaf.state_dict()


class _Model(nn.Module):
    """One dense parameter and one owner-local Host table."""

    def __init__(self, table: HostEngramTable, device: torch.device) -> None:
        """Register one owner table and one replicated dense parameter."""
        super().__init__()
        self.embed = table
        self.dense = nn.Parameter(torch.ones(1, device=device))


def _assert_aligned_failures(state, table, rank, device, runtime):
    """Require invalid ID, budget, non-finite, and checksum faults to align."""
    invalid_ids = torch.tensor([15 if rank == 0 else 2], device=device)
    try:
        table(invalid_ids)
    except IndexError:
        pass
    else:
        raise AssertionError("Every stage rank must reject one invalid Host ID")

    if rank == 0:
        table.max_pending_entries = 0
        table.append_rows(torch.tensor([2]), torch.ones((1, 2)))
    try:
        state.prepare_optimizer_step(max_norm=0)
    except RuntimeError as exc:
        assert "Host sparse gradient validation failed" in str(exc)
    else:
        raise AssertionError("Every stage rank must reject a pending budget overflow")

    table.clear_step()
    table.max_pending_entries = 1
    if rank == 0:
        table.append_rows(torch.tensor([2]), torch.tensor([[float("nan"), 0.0]]))
    try:
        state.prepare_optimizer_step(max_norm=0)
    except RuntimeError as exc:
        assert "Host sparse gradient validation failed" in str(exc)
    else:
        raise AssertionError("Every stage rank must reject a non-finite Host gradient")
    table.clear_step()

    dist.barrier()
    if rank == 0:
        sidecar = next((Path(runtime.step_dir) / "engram_host").glob("*.pt"))
        with sidecar.open("ab") as target:
            target.write(b"corrupt")
    dist.barrier()
    try:
        state.before_checkpoint_load(runtime)
    except RuntimeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("Corrupt Host owner sidecar must reject the checkpoint")


def _run_host_engram(backend, device_type):
    """Run the same fixture on CPU/Gloo or Ascend/HCCL."""
    if device_type == "npu":
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend)
    try:
        rank = dist.get_rank()
        device = torch.device(device_type)
        mesh = MeshContext(dp_size=2, cp_size=1, tp_size=2, ep_size=2,
                           dp_replicate_size=1, dp_shard_size=2,
                           edp_shard_size=2, loss_parallel=True)
        mesh.build_meshs(device_type, 4)
        ep_mesh = mesh.fsdp_moe_mesh["ep"]
        ep_rank = ep_mesh.get_local_rank()
        table = HostEngramTable(
            source_weight=nn.Parameter(torch.empty((16, 2), device="meta")),
            logical_rows=15, physical_rows=16, width=2,
        )
        table.weight = nn.Parameter(torch.empty((8, 2), device="meta"))
        table.bind_planned_shard(ep_rank=ep_rank, ep_size=2, ep_group=ep_mesh.get_group())
        initial = torch.zeros((8, 2), dtype=torch.float32)
        initial[2] = torch.tensor([1.0, 1.0] if ep_rank == 0 else [2.0, 2.0])
        torch.utils.swap_tensors(table.weight, nn.Parameter(initial))
        model = _Model(table, device)
        state = DeepseekV41HostState(model, {"embed": table}, mesh)
        sparse = torch.optim.SparseAdam([table.weight], lr=0.1, betas=(0.9, 0.95), eps=1.0e-6)
        coordinator = HostSparseOptimizerCoordinator(_DenseOptimizer(model.dense), sparse, state.tables)
        state.optimizer = coordinator
        state.sparse_optimizer = sparse

        ids = torch.tensor([2, 10, 2] if rank in (0, 1) else [2, 10], device=device)
        upstream = {
            0: [[1.0, 1.0], [0.0, 5.0], [1.0, 3.0]],
            1: [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            2: [[1.0, 0.0], [0.0, 1.0]],
            3: [[0.0, 0.0], [0.0, 1.0]],
        }[rank]
        device_table = nn.Embedding(8, 2, device=device)
        with torch.no_grad():
            device_table.weight.copy_(table.weight.to(device))
        device_output = EngramModule._sparse_lookup(
            ids, device_table, ep_group=ep_mesh.get_group(), ep_rank=ep_rank,
            ep_size=2, padded_num_embeddings=16,
        )
        device_output.backward(torch.tensor(upstream, device=device))
        output = table(ids)
        torch.testing.assert_close(output, device_output)
        torch.testing.assert_close(output, torch.tensor([[1.0, 1.0], [2.0, 2.0]] +
                                                        ([[1.0, 1.0]] if rank in (0, 1) else []), device=device))
        output.backward(torch.tensor(upstream, device=device))
        model.dense.grad = torch.tensor([3.0], device=device)
        local = table.coalesce_pending()
        torch.testing.assert_close(local.to_dense(), device_table.weight.grad.cpu())
        expected_local = {0: [3.0, 4.0], 1: [0.0, 6.0],
                          2: [1.0, 0.0], 3: [0.0, 2.0]}[rank]
        torch.testing.assert_close(local.values()[0], torch.tensor(expected_local))
        prepared = state.prepare_optimizer_step(max_norm=2.0)
        torch.testing.assert_close(prepared.global_norm, torch.tensor(math.sqrt(33.0), device=device))
        expected_synced = [2.0, 2.0] if ep_rank == 0 else [0.0, 4.0]
        coefficient = 2.0 / (math.sqrt(33.0) + 1.0e-6)
        torch.testing.assert_close(table.pending.values()[0],
                                   torch.tensor(expected_synced) * coefficient)
        coordinator.step()
        expected_weight = [0.9, 0.9] if ep_rank == 0 else [2.0, 1.9]
        torch.testing.assert_close(table.weight[2], torch.tensor(expected_weight), atol=1.0e-5, rtol=1.0e-5)
        coordinator.zero_grad()

        checkpoint_dir = [tempfile.mkdtemp() if rank == 0 else None]
        dist.broadcast_object_list(checkpoint_dir, src=0)
        runtime = CheckpointRuntime(
            step_dir=checkpoint_dir[0], global_step=1, mesh_context=mesh,
            save_optimizer=True, save_train_state=True,
            restore_optimizer=True, restore_train_state=True,
        )
        state.before_checkpoint_save(runtime)
        state.before_checkpoint_load(runtime)
        with torch.no_grad():
            table.weight.zero_()
        state.after_checkpoint_load(runtime)
        torch.testing.assert_close(table.weight[2], torch.tensor(expected_weight), atol=1.0e-5, rtol=1.0e-5)
        _assert_aligned_failures(state, table, rank, device, runtime)
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir[0])
    finally:
        dist.destroy_process_group()


def test_host_engram_four_rank_gloo():
    """Verify the complete toy step on CPU/Gloo."""
    _run_host_engram("gloo", "cpu")


def test_host_engram_four_rank_hccl():
    """Verify the same toy step on Ascend/HCCL."""
    _run_host_engram("hccl", "npu")


def test_host_engram_ep1_generalization_gloo():
    """Check EP1 DP/CP/TP loss-domain variants against device lookups."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        variants = ((2, 1, 1, False), (1, 2, 1, False),
                    (1, 1, 2, False), (1, 1, 2, True))
        for dp_size, cp_size, tp_size, loss_parallel in variants:
            mesh = MeshContext(
                dp_size=dp_size, cp_size=cp_size, tp_size=tp_size,
                ep_size=1, dp_replicate_size=1,
                dp_shard_size=dp_size * cp_size,
                loss_parallel=loss_parallel,
            )
            mesh.build_meshs("cpu", 2)
            table = HostEngramTable(
                source_weight=nn.Parameter(torch.empty((16, 2), device="meta")),
                logical_rows=15, physical_rows=16, width=2,
            )
            table.bind_planned_shard(ep_rank=0, ep_size=1)
            initial = torch.zeros((16, 2), dtype=torch.float32)
            initial[2] = torch.tensor([1.0, 1.0])
            initial[10] = torch.tensor([2.0, 2.0])
            torch.utils.swap_tensors(table.weight, nn.Parameter(initial))
            reference = nn.Embedding(16, 2)
            with torch.no_grad():
                reference.weight.copy_(initial)
            model = _Model(table, torch.device("cpu"))
            state = DeepseekV41HostState(model, {"embed": table}, mesh)
            sparse = torch.optim.SparseAdam([table.weight], lr=0.1)
            state.optimizer = HostSparseOptimizerCoordinator(
                _DenseOptimizer(model.dense), sparse, state.tables,
            )
            state.sparse_optimizer = sparse
            ids = torch.tensor([2, 10, 2] if rank == 0 else [2, 10])
            upstream = torch.tensor(
                [[1.0, 1.0], [0.0, 5.0], [1.0, 3.0]] if rank == 0
                else [[1.0, 0.0], [0.0, 1.0]],
            )
            host_output = table(ids)
            device_output = reference(ids)
            torch.testing.assert_close(host_output, device_output)
            host_output.backward(upstream)
            device_output.backward(upstream)
            torch.testing.assert_close(table.coalesce_pending().to_dense(), reference.weight.grad)
            model.dense.grad = torch.tensor([3.0])
            prepared = state.prepare_optimizer_step(max_norm=0)
            denominator = dp_size * cp_size * (1 if loss_parallel else tp_size)
            expected = torch.zeros((16, 2))
            expected[2] = torch.tensor([3.0, 4.0]) / denominator
            expected[10] = torch.tensor([0.0, 6.0]) / denominator
            torch.testing.assert_close(table.pending.to_dense(), expected)
            norm_sq = 9.0 + expected.square().sum()
            torch.testing.assert_close(prepared.global_norm, norm_sq.sqrt())
            state.optimizer.zero_grad()
    finally:
        dist.destroy_process_group()


def test_host_engram_pp_empty_stage_gloo():
    """Check a one-rank PP stage without Engram still joins norm and save."""
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        mesh = MeshContext(dp_size=1, cp_size=1, tp_size=1, pp_size=2,
                           dp_replicate_size=1, dp_shard_size=1)
        mesh.pp_rank = rank
        mesh.build_meshs("cpu", 2)
        tables = {}
        sparse = None
        if rank == 0:
            table = HostEngramTable(
                source_weight=nn.Parameter(torch.empty((16, 2), device="meta")),
                logical_rows=15, physical_rows=16, width=2,
            )
            table.bind_planned_shard(ep_rank=0, ep_size=1)
            torch.utils.swap_tensors(table.weight, nn.Parameter(torch.ones((16, 2))))
            model = _Model(table, torch.device("cpu"))
            tables = {"embed": table}
            sparse = torch.optim.SparseAdam([table.weight], lr=0.1)
            table.append_rows(torch.tensor([2]), torch.tensor([[2.0, 0.0]]))
        else:
            model = nn.Module()
            model.dense = nn.Parameter(torch.ones(1))
        model.dense.grad = torch.tensor([3.0 if rank == 0 else 5.0])
        state = DeepseekV41HostState(model, tables, mesh)
        state.optimizer = HostSparseOptimizerCoordinator(_DenseOptimizer(model.dense), sparse, tables)
        state.sparse_optimizer = sparse
        prepared = state.prepare_optimizer_step(max_norm=0)
        torch.testing.assert_close(prepared.global_norm, torch.tensor(math.sqrt(38.0)))

        checkpoint_dir = [tempfile.mkdtemp() if rank == 0 else None]
        dist.broadcast_object_list(checkpoint_dir, src=0)
        runtime = CheckpointRuntime(
            step_dir=checkpoint_dir[0], global_step=0, mesh_context=mesh,
            save_optimizer=True, save_train_state=True,
            restore_optimizer=True, restore_train_state=True,
        )
        state.before_checkpoint_save(runtime)
        state.before_checkpoint_load(runtime)
        if rank == 0:
            with torch.no_grad():
                table.weight.zero_()
        state.after_checkpoint_load(runtime)
        if rank == 0:
            torch.testing.assert_close(table.weight, torch.ones_like(table.weight))
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir[0])
    finally:
        dist.destroy_process_group()

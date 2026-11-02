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
"""Model-owned Host Engram state across build, load and optimization."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import logging
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.utils.clip_grad import (
    build_norm_shard_plan,
    dense_grad_norm_sq_and_refs,
    scale_grad_refs_,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed.host_sparse_grad import (
    sparse_sum_same_ep,
    stage_check,
    stage_reduce_scalar,
)
from hyper_parallel.models.external_state import (
    CheckpointRuntime,
    CheckpointTensorShard,
    ExternalBuildContext,
    ExternalLoadContext,
    ExternalMaterializationResult,
    GradientPreparation,
)
from hyper_parallel.models.deepseek_v41.adapter.checkpoint.host_sidecar import (
    HostSidecarLoadPlan,
    after_checkpoint_load,
    after_weights_only_load,
    before_checkpoint_load,
    before_checkpoint_save,
)
from hyper_parallel.models.deepseek_v41.adapter.engram.host_source import materialize_owner_rows
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.adapter.optim.host_sparse import (
    HostSparseOptimizerCoordinator,
)

logger = logging.getLogger(__name__)


class DeepseekV41HostState:
    """Own final CPU table parameters for this pipeline stage."""

    supports_async_checkpoint = False

    def __init__(self, model: nn.Module, tables: Mapping[str, HostEngramTable], mesh_context: Any) -> None:
        """Record final local table identities and the stage mesh."""
        self.model = model
        self.tables = dict(sorted(tables.items()))
        self.mesh_context = mesh_context
        for table in self.tables.values():
            table.mesh_context = mesh_context
        self.parameters = {f"{fqn}.weight": table.weight for fqn, table in self.tables.items()}
        self.optimizer = None
        self.sparse_optimizer = None
        self.norm_shard_plan = None

    def check_identities(self) -> None:
        """Reject FSDP, loading or dtype conversion that replaced a Host parameter."""
        named = dict(self.model.named_parameters(remove_duplicate=False))
        for fqn, parameter in self.parameters.items():
            table = self.tables[fqn.removesuffix(".weight")]
            if (named.get(fqn) is not parameter or parameter.device.type != "cpu"
                    or parameter.dtype != torch.float32
                    or tuple(parameter.shape) != (table.rows_per_owner, table.width)):
                raise RuntimeError(f"Host Engram parameter identity or device changed: {fqn}")

    def weight_digests(self) -> dict[str, str]:
        """Fingerprint owner-local row bytes around ordinary model initialization."""
        self.check_identities()
        return {
            fqn: hashlib.sha256(memoryview(parameter.detach().contiguous().numpy()).cast("B")).hexdigest()
            for fqn, parameter in self.parameters.items()
        }

    def checkpoint_tensor_shards(self) -> dict[str, CheckpointTensorShard]:
        """Expose owner-local rows and padding to checkpoint verification tools."""
        self.check_identities()
        return {
            f"{fqn}.weight": CheckpointTensorShard(
                tensor=table.weight,
                global_shape=(table.physical_rows, table.width),
                offsets=(table.global_row_start, 0),
                logical_shape=(table.logical_rows, table.width),
            )
            for fqn, table in self.tables.items()
        }

    def materialize(self, context: ExternalLoadContext) -> ExternalMaterializationResult:
        """Install owner-local CPU rows, optionally slicing safetensors by row.

        Args:
            context: Build or load lifecycle context.
        """
        source_keys = materialize_owner_rows(self.model, self.tables, context)
        self.check_identities()
        return ExternalMaterializationResult(frozenset(self.parameters), source_keys)

    def build_optimizer(self, optimizer_config: Any, *, wrap_dense: Callable[[Any], Any]) -> Any:
        """Build dense optimizer once, then combine it with CPU SparseAdam.

        Args:
            optimizer_config: Dense optimizer configuration.
        """
        host_ids = {id(parameter) for parameter in self.parameters.values()}
        dense_raw = optimizer_config.target.build(
            model=self.model, excluded_param_ids=host_ids,
        ).get_optimizer()
        dense_ids = {id(param) for group in dense_raw.param_groups for param in group["params"]}
        all_ids = {id(param) for param in self.model.parameters() if param.requires_grad}
        if dense_ids & host_ids or dense_ids | host_ids != all_ids:
            raise ValueError("Dense and Host optimizers do not partition trainable parameters")
        dense = wrap_dense(dense_raw)
        hparams = getattr(getattr(self.model, "config", None), "engram_sparse_adam_config", None)
        if hparams is None:
            hparams = {"lr": 1.0e-3}
        sparse = torch.optim.SparseAdam(list(self.parameters.values()), **hparams) if self.parameters else None
        self.optimizer = HostSparseOptimizerCoordinator(dense, sparse, self.tables)
        self.sparse_optimizer = sparse
        return self.optimizer

    def optimizer_for_dcp(self) -> Any:
        """Return the dense optimizer that owns the DCP state."""
        return self.optimizer.optimizer_for_dcp()

    def before_checkpoint_save(self, runtime: CheckpointRuntime) -> None:
        """Write and verify CPU owner sidecars before dense DCP save.

        Args:
            runtime: Checkpoint runtime and topology context.
        """
        before_checkpoint_save(self, runtime)

    def before_checkpoint_load(self, runtime: CheckpointRuntime) -> HostSidecarLoadPlan:
        """Validate all Host files before dense DCP state is read.

        Args:
            runtime: Checkpoint runtime and topology context.
        """
        return before_checkpoint_load(self, runtime)

    def after_checkpoint_load(self, runtime: CheckpointRuntime, load_plan: HostSidecarLoadPlan) -> None:
        """Restore this rank's owner rows and optional sparse moments.

        Args:
            runtime: Checkpoint runtime and topology context.
            load_plan: Validated sidecar schema from this restore attempt.
        """
        after_checkpoint_load(self, runtime, load_plan)

    def after_weights_only_load(self) -> None:
        """Reset sparse moments after a weights-only restore."""
        after_weights_only_load(self)

    def is_sparse_norm_representative(self) -> bool:
        """Count each owner's synchronized row gradient on one edp replica."""
        mesh = self.mesh_context
        if mesh is None or mesh.device_mesh is None or not dist.is_initialized():
            return True
        moe = mesh.fsdp_moe_mesh
        if moe is None:
            return dist.get_rank() == min(mesh.device_mesh.rank_list)
        return all(moe.get_local_rank(name) == 0 for name in
                   ("edp_shard", "edp_replicate") if name in (moe.mesh_dim_names or ()))

    def _prepare_sparse_rows(self) -> None:
        """Coalesce and synchronize each table before the shared failure gate."""
        mesh = self.mesh_context
        failed = False
        denominator = 1 if mesh is None else (
            mesh.dp_size * mesh.cp_size * (1 if mesh.loss_parallel else mesh.tp_size)
        )
        for table in self.tables.values():
            try:
                started = time.perf_counter()
                local = table.coalesce_pending()
                table.coalesce_seconds += time.perf_counter() - started
                local_bad = False
            except (ValueError, IndexError) as exc:
                logger.error("Host sparse local row validation failed: %s", exc)
                local = None
                local_bad = True
            if stage_check(local_bad, mesh):
                failed = True
                break
            synced = sparse_sum_same_ep(local, mesh, table.width)
            synced_nnz = synced._nnz()  # pylint: disable=protected-access
            table.max_synced_nnz = max(table.max_synced_nnz, synced_nnz)
            if stage_check(synced_nnz > table.max_sparse_rows_per_step or
                           not bool(torch.isfinite(synced.values()).all()), mesh):
                failed = True
                break
            table.pending = torch.sparse_coo_tensor(
                synced.indices(), synced.values() / denominator, synced.shape,
            ).coalesce()
        if mesh is not None and mesh.pp_size > 1:
            device_type = mesh.device_mesh.device_type
            failure_flag = torch.tensor(int(failed), device=torch.device(device_type))
            dist.all_reduce(failure_flag, op=dist.ReduceOp.MAX)
            failed = bool(failure_flag.item())
        if failed:
            raise RuntimeError("Host sparse gradient validation failed")

    def _combined_gradient_norm(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Count each dense shard and synchronized sparse owner exactly once."""
        mesh = self.mesh_context
        if self.norm_shard_plan is None:
            self.norm_shard_plan = build_norm_shard_plan(
                self.model, self.parameters.values(), mesh,
            )
        dense_sq, dense_refs = dense_grad_norm_sq_and_refs(
            self.model, self.parameters.values(), self.norm_shard_plan, mesh,
        )
        sparse_sq = torch.zeros_like(dense_sq)
        if self.is_sparse_norm_representative():
            for table in self.tables.values():
                sparse_sq.add_(table.pending.values().square().sum().to(dense_sq.device))
        stage_reduce_scalar(sparse_sq, mesh)
        global_sq = dense_sq + sparse_sq
        if mesh is not None and mesh.pp_size > 1:
            dist.all_reduce(global_sq, op=dist.ReduceOp.SUM)
            global_sq.div_(mesh.dp_size * mesh.cp_size * mesh.tp_size)
        if not bool(torch.isfinite(global_sq)):
            raise RuntimeError("Non-finite combined Engram gradient norm")
        return global_sq.sqrt(), dense_refs

    def prepare_optimizer_step(self, *, max_norm: float) -> GradientPreparation:
        """Synchronize sparse rows and clip them with dense effective gradients."""
        if self.optimizer is None or self.optimizer.state != "ACCUMULATING":
            raise RuntimeError("Host gradient preparation requires ACCUMULATING state")
        self._prepare_sparse_rows()
        norm, dense_refs = self._combined_gradient_norm()
        coefficient = (torch.clamp(max_norm / (norm + 1.0e-6), max=1.0)
                       if max_norm > 0 else torch.ones_like(norm))
        scale_grad_refs_(dense_refs, coefficient)
        cpu_coefficient = coefficient.cpu()
        for table in self.tables.values():
            sparse = table.pending
            table.pending = torch.sparse_coo_tensor(
                sparse.indices(), sparse.values() * cpu_coefficient, sparse.shape,
            ).coalesce()
        self.optimizer.state = "PREPARED"
        return GradientPreparation(norm, coefficient, norm * coefficient)


def _validate_host_source_layout(target: str, layout: Any, ep_size: int) -> None:
    """Require the planner's sole EP row shard and reject any second shard."""
    if ep_size == 1:
        if layout is not None and any(isinstance(place, Shard) for place in layout[0]):
            raise ValueError(f"EP1 Host Engram cannot be sharded: {target}")
        return
    if layout is None:
        raise ValueError(f"Missing EP source layout for {target}")
    placements, source_mesh = layout
    dim_names = source_mesh.mesh_dim_names or ()
    if len(placements) != len(dim_names) or sum(
            name == "ep" and isinstance(place, Shard) and place.dim == 0
            for name, place in zip(dim_names, placements)) != 1:
        raise ValueError(f"Host Engram requires exactly one EP Shard(0): {target}")
    if any(isinstance(place, Shard) and name != "ep" for name, place in zip(dim_names, placements)):
        raise ValueError(f"Host Engram has an unexpected second shard: {target}")


def _validate_host_model_mode(model: nn.Module, tables: Mapping[str, HostEngramTable],
                              context: ExternalBuildContext) -> bool:
    """Reject mixed device/Host replacements and unsupported build settings."""
    backend = getattr(getattr(model, "config", None), "engram_storage_backend", "device")
    if backend not in ("device", "host"):
        raise ValueError("engram_storage_backend must be 'device' or 'host'")
    if not tables and backend != "host":
        return False
    if backend != "host":
        raise ValueError("HostEngramModule requires engram_storage_backend='host'")
    from hyper_parallel.components.modules.engram import EngramModule  # pylint: disable=C0415
    from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (  # pylint: disable=C0415
        DeepseekV41Engram,
    )
    if any(isinstance(module, (EngramModule, DeepseekV41Engram))
           and not isinstance(module.embed, HostEngramTable)
           for module in model.modules()):
        raise ValueError("Host Engram configuration still contains a device replacement")
    if (context.validate_placement or context.init_device is None
            or context.model_init_dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("Host Engram requires meta production build with model_init_dtype=float32 or bfloat16")
    return True


def build_deepseek_v41_external_state(model: nn.Module,
                                     context: ExternalBuildContext) -> DeepseekV41HostState | None:
    """Bind every Host table to the EP row interval selected by the planner.

    Args:
        model: Model being built or inspected.
        context: Build or load lifecycle context.
    """
    tables = {
        fqn: module for fqn, module in model.named_modules()
        if isinstance(module, HostEngramTable)
    }
    if not _validate_host_model_mode(model, tables, context):
        return None
    mesh = context.mesh_context
    ep_size = mesh.ep_size if mesh is not None else 1
    ep_rank = mesh.ep_rank if mesh is not None else 0
    ep_mesh = mesh.fsdp_moe_mesh["ep"] if ep_size > 1 else None
    if ep_mesh is not None:
        ep_rank = ep_mesh.get_local_rank()
    for fqn, table in tables.items():
        target = f"{fqn}.weight"
        layout = (context.source_shard_info or {}).get(target)
        _validate_host_source_layout(target, layout, ep_size)
        table.bind_planned_shard(
            ep_rank=ep_rank, ep_size=ep_size,
            ep_group=ep_mesh.get_group() if ep_mesh is not None else None,
            compute_dtype=context.model_init_dtype,
        )
    return DeepseekV41HostState(model, tables, mesh)

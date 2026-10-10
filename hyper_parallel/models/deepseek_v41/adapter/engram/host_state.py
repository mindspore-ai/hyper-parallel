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
import json
import logging
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    rename_source_key,
)
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
    CheckpointRequirements,
    CheckpointRuntime,
    ExternalBuildContext,
    ExternalLoadContext,
    ExternalMaterializationResult,
    GradientPreparation,
)
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.adapter.optim.host_sparse import (
    HostSparseOptimizerCoordinator,
)

logger = logging.getLogger(__name__)


def _row_manifest(checkpoint_index: Any) -> tuple[Path, dict[str, Any]]:
    """Read optional logical-table to physical-safetensors row mapping."""
    root = next(iter(checkpoint_index.files_by_key.values())).parent
    path = root / "engram_host_rows.json"
    if not path.is_file():
        return root, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not isinstance(payload.get("tables"), dict):
        raise ValueError("Invalid Engram Host row manifest")
    return root, payload["tables"]


def _read_manifest_rows(root: Path, source_key: str, spec: dict[str, Any],
                        table: HostEngramTable) -> torch.Tensor:
    """Validate complete physical row coverage and read only this owner."""
    if spec.get("shape") != [table.physical_rows, table.width]:
        raise ValueError(f"Host Engram manifest shape mismatch: {source_key}")
    fragments = spec.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError(f"Host Engram manifest has no row fragments: {source_key}")
    rows = torch.empty((table.rows_per_owner, table.width), dtype=torch.float32)
    owner_start = table.global_row_start
    owner_end = owner_start + table.rows_per_owner
    cursor = 0
    for fragment in sorted(fragments, key=lambda item: item["start"]):
        start, length = int(fragment["start"]), int(fragment["rows"])
        end = start + length
        if start != cursor or length <= 0:
            raise ValueError(f"Host Engram manifest row gap or overlap: {source_key}")
        cursor = end
        left, right = max(start, owner_start), min(end, owner_end)
        if left >= right:
            continue
        path = (root / fragment["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.suffix != ".safetensors":
            raise ValueError(f"Host Engram manifest has invalid fragment path: {source_key}")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            tensor_slice = source.get_slice(fragment["key"])
            if tuple(tensor_slice.get_shape()) != (length, table.width):
                raise ValueError(f"Host Engram manifest fragment shape mismatch: {source_key}")
            part = tensor_slice[left - start:right - start]
        if part.dtype != torch.float32:
            raise ValueError(f"Host Engram manifest fragment dtype mismatch: {source_key}")
        rows[left - owner_start:right - owner_start] = part
    if cursor != table.physical_rows:
        raise ValueError(f"Host Engram manifest row coverage incomplete: {source_key}")
    return rows


def _host_source_keys(model: nn.Module, checkpoint_index: Any,
                      manifest: dict[str, Any],
                      weights_mapping: Any, targets: frozenset[str]) -> dict[str, str]:
    """Resolve one sliceable source tensor for each Host target FQN."""
    from hyper_parallel.models._transformers.checkpoint_loader import (  # pylint: disable=C0415
        _base_weight_mapping,
        _build_load_targets,
    )

    replacement_mapping = getattr(model, "_hp_replacement_weight_conversions", None)
    mapping = _base_weight_mapping(weights_mapping or [], replacement_mapping or [])
    renamings = [item for item in mapping if isinstance(item, WeightRenaming)]
    converters = [item for item in mapping if isinstance(item, WeightConverter)]
    all_targets = _build_load_targets(model)
    found: dict[str, str] = {}
    for source_key in sorted(set(checkpoint_index.keys()) | manifest.keys()):
        target, source_pattern = rename_source_key(
            source_key, renamings, converters, getattr(model, "base_model_prefix", None),
            meta_state_dict=all_targets,
        )
        if target not in targets and source_key in targets:
            target, source_pattern = rename_source_key(
                source_key, [], [], getattr(model, "base_model_prefix", None),
                meta_state_dict=all_targets,
            )
        if target not in targets:
            continue
        if source_pattern is not None:
            raise ValueError(f"Host Engram source conversion cannot be sliced: {source_key}")
        if target in found:
            raise ValueError(f"Multiple Host Engram source keys map to {target}")
        found[target] = source_key
    missing = targets - found.keys()
    if missing:
        raise KeyError(f"Missing Host Engram source key for {sorted(missing)}")
    return found


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
        self._host_sidecar_schema = None

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

    def materialize(self, context: ExternalLoadContext) -> ExternalMaterializationResult:
        """Install owner-local CPU rows, optionally slicing safetensors by row.

        Args:
            context: Build or load lifecycle context.
        """
        source_keys = set()
        index = None
        source_by_target = {}
        manifest_root = None
        manifest = {}
        if context.load_base_model and self.tables:
            from hyper_parallel.models._transformers.checkpoint_loader import (  # pylint: disable=C0415
                _resolve_checkpoint_index,
            )
            index = _resolve_checkpoint_index(context.pretrained_path)
            manifest_root, manifest = _row_manifest(index)
            source_by_target = _host_source_keys(
                self.model, index, manifest, context.weights_mapping, frozenset(self.parameters),
            )
        for fqn, table in self.tables.items():
            target = f"{fqn}.weight"
            if index is None:
                # Row-wise seeds keep each EP replica identical without allocating the full table.
                rows = torch.empty((table.rows_per_owner, table.width), dtype=torch.float32)
                generator = torch.Generator(device="cpu")
                layer_seed = int.from_bytes(hashlib.sha256(fqn.encode()).digest()[:8], "little")
                for local_row in range(table.rows_per_owner):
                    generator.manual_seed((1729 + layer_seed + table.global_row_start + local_row) % (2**63 - 1))
                    rows[local_row] = torch.randn(table.width, generator=generator)
            else:
                source_key = source_by_target[target]
                if source_key in manifest:
                    rows = _read_manifest_rows(manifest_root, source_key, manifest[source_key], table)
                else:
                    with safe_open(str(index.files_by_key[source_key]), framework="pt", device="cpu") as source:
                        tensor_slice = source.get_slice(source_key)
                        if tuple(tensor_slice.get_shape()) != (table.physical_rows, table.width):
                            raise ValueError(f"Host Engram source shape mismatch: {target}")
                        rows = tensor_slice[table.global_row_start:table.global_row_start + table.rows_per_owner]
                source_keys.add(source_key)
            if rows.dtype != torch.float32 or tuple(rows.shape) != tuple(table.weight.shape):
                raise ValueError(f"Host Engram source dtype or local shape mismatch: {target}")
            torch.utils.swap_tensors(table.weight, nn.Parameter(rows.contiguous()))
            table.weight._is_hf_initialized = True  # pylint: disable=W0212
        self.check_identities()
        return ExternalMaterializationResult(frozenset(self.parameters), frozenset(source_keys))

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
        from hyper_parallel.models.deepseek_v41.adapter.checkpoint.host_sidecar import (  # pylint: disable=C0415
            before_checkpoint_save,
        )
        before_checkpoint_save(self, runtime)

    def before_checkpoint_load(self, runtime: CheckpointRuntime) -> CheckpointRequirements:
        """Validate all Host files before dense DCP state is read.

        Args:
            runtime: Checkpoint runtime and topology context.
        """
        from hyper_parallel.models.deepseek_v41.adapter.checkpoint.host_sidecar import (  # pylint: disable=C0415
            before_checkpoint_load,
        )
        return before_checkpoint_load(self, runtime)

    def after_checkpoint_load(self, runtime: CheckpointRuntime) -> None:
        """Restore this rank's owner rows and optional sparse moments.

        Args:
            runtime: Checkpoint runtime and topology context.
        """
        from hyper_parallel.models.deepseek_v41.adapter.checkpoint.host_sidecar import (  # pylint: disable=C0415
            after_checkpoint_load,
        )
        after_checkpoint_load(self, runtime)

    def after_weights_only_load(self) -> None:
        """Reset sparse moments after a weights-only restore."""
        from hyper_parallel.models.deepseek_v41.adapter.checkpoint.host_sidecar import (  # pylint: disable=C0415
            after_weights_only_load,
        )
        after_weights_only_load(self)

    def _is_sparse_norm_representative(self) -> bool:
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
            table.max_synced_nnz = max(table.max_synced_nnz, synced._nnz())
            if stage_check(synced._nnz() > table.max_sparse_rows_per_step or
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
        if self._is_sparse_norm_representative():
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
    if context.validate_placement or context.init_device is None or context.model_init_dtype != torch.float32:
        raise ValueError("Host Engram requires meta production build with model_init_dtype=float32")
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
        )
    return DeepseekV41HostState(model, tables, mesh)

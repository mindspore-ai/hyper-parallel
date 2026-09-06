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
"""parameter_sharding: parameter-side apply machinery (05 §4.2 + Phase D).

Owns parameter placement/localize/stack (merged from the legacy
``components/distributed/sharding/apply.py`` — there is exactly one parameter
sharding implementation), source-mesh resolution, runtime source_shard_info
construction, and the tied-weights replication pass (same-rank shared
storage, no cross-rank shard broadcast).
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.distributed.recipe_spec import (
    PlacementMismatchError,
    resolve_placements,
)
from hyper_parallel.distributed._builder.source_shard import (
    build_source_shard_info,
)
from hyper_parallel.distributed.tensor_parallel.head_count import (
    maybe_update_head_counts,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Legacy path helpers (merged from components/distributed/sharding/apply.py)
# ────────────────────────────────────────────────────────────────────────────

def _get_attr_by_path(model, fqn):
    """Fetch an attribute along a dotted FQN (numeric segments index into ModuleLists)."""
    obj = model
    for p in fqn.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _set_param_by_path(model: nn.Module, fqn: str, new_param) -> None:
    """Locate the parent module along a dotted FQN and replace the leaf parameter.

    object.__setattr__(model, dotted_name, ...) would only set a stray
    attribute on model and would not replace the submodule parameter — you
    must walk the path to the true parent module before assigning.
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
    """Fetch a module by FQN (the last segment is NOT stripped; call sites pass module FQNs).

    Same semantics as _get_attr_by_path — every call site (Phase A/B/C)
    passes a module fully-qualified name (e.g. `model.layers.0.self_attn`),
    not a parameter FQN, so no last-segment stripping is done (stripping
    would incorrectly return the parent module). The empty FQN "" resolves
    to the model itself (D-14: a root-level outer spec, e.g. a whole-LM
    contract, 05 §13.4).
    """
    obj = model
    if not fqn:
        return obj
    for p in fqn.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _local_params_context(model: nn.Module):
    """One-shot unwrap at build time: replace DTensor parameters with their
    _local_tensor (plain), zero-copy.

    Called at the Phase C entry of apply_sharding_plan, before fully_shard;
    the permanent unwrap is not restored. _local_tensor shares storage with
    the original DTensor (same data_ptr).

    Returns a {fqn: placements} snapshot of the placements before unwrapping
    (diagnostic use only; the canonical source for source_shard_info is the
    ShardingPlan, see build_source_shard_info).
    """
    source_shard_records = {}
    for name, param in list(model.named_parameters()):
        if isinstance(param, DTensor):
            source_shard_records[name] = param.placements
            _set_param_by_path(model, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    return source_shard_records


@contextmanager
def _temp_local_params(module, exclude=()):
    """Temporarily unwrap DTensor parameters inside a validate-mode local region (restored on exit).

    Under production the parameters were already permanently unwrapped at build
    time, so this context is unnecessary. A validate-mode local region (MoE
    all-to-all / HF CP attention) computes on local tensors internally and
    needs local parameters; after restoration the DTensor propagation chain is
    unbroken.

    exclude: relative FQN prefixes of nested-boundary subtrees whose
    parameters must stay DTensors (D-14 invariant 3, 05 §13.3 — inner
    validate islands dispatch via __torch_function__ and break if the outer
    region unwraps their parameters).
    """
    excluded = tuple(e.rstrip(".") + "." for e in exclude)
    saved = []
    for name, param in list(module.named_parameters(recurse=True)):
        if excluded and name.startswith(excluded):
            continue
        if isinstance(param, DTensor):
            saved.append((name, param))
            *path, leaf = name.split(".")
            owner = module
            for part in path:
                owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
            owner._parameters[leaf] = param.to_local()  # pylint: disable=protected-access
    try:
        yield
    finally:
        for name, param in saved:
            _set_param_by_path(module, name, param)


class _StackedExperts(nn.Module):
    """Container for stacked per-expert weights: gate_proj/up_proj/down_proj
    (or w1/w2/w3) become Parameters of shape [E, ...], sharded uniformly by
    EP Shard(0) + TP (the D-08 ndim=3 rule)."""


def _stack_moe_experts(module: nn.Module, ep_stack: Dict[str, List[str]]) -> None:
    """Per-expert parameters → stacked 3D parameters (stack is concat, values
    exactly equal).

    Executed before _shard_module_params in Phase A:
    - fetch weights by source path → torch.stack(dim=0), register onto the
      replaced experts holder;
    - the original ModuleList is replaced as a whole (memory freed);
    - v1 asserts no bias and consistent source-parameter shapes; meta tensors
      work the same way (concat of metas).

    ep_stack: {stacked relative path: [source parameter relative paths
    (ordered by expert idx)]}, e.g.
    {"experts.gate_proj": ["experts.0.gate_proj.weight", ...]}.
    """
    holders: Dict[str, Dict[str, nn.Parameter]] = {}
    for stacked_path, sources in ep_stack.items():
        parent_path, param_name = stacked_path.rsplit(".", 1)
        tensors = []
        requires_grad = True
        for src in sources:
            owner = _resolve_module(module, src.rsplit(".", 1)[0])
            if getattr(owner, "bias", None) is not None:
                raise NotImplementedError(
                    f"D-09 v1 does not support experts with bias "
                    f"({src.rsplit('.', 1)[0]}); use an EP-aware MoE module instead"
                )
            t = _get_attr_by_path(module, src)
            tensors.append(t.data if hasattr(t, "data") else t)
            requires_grad = getattr(t, "requires_grad", True)
        stacked = torch.stack(tensors, dim=0)
        holders.setdefault(parent_path, {})[param_name] = nn.Parameter(
            stacked, requires_grad=requires_grad)

    for parent_path, params in holders.items():
        holder = _StackedExperts()
        for name, p in params.items():
            holder.register_parameter(name, p)
        *path, leaf = parent_path.split(".")
        obj = module
        for seg in path:
            obj = obj[int(seg)] if seg.isdigit() else getattr(obj, seg)
        setattr(obj, leaf, holder)   # replace the original ModuleList (original expert params freed)


# ────────────────────────────────────────────────────────────────────────────
# Source-mesh resolution + Phase A parameter sharding (05 §4.2)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_parameter_source_meshes(plan, mesh_context, full_mesh, tp_mesh):
    """Resolve dense TP and routed-expert source meshes for one sharding plan."""
    # Lazy import: applier imports parameter_sharding (_resolve_module) at
    # module level — importing applier here at module level would cycle.
    from hyper_parallel.distributed._builder.applier import (  # pylint: disable=C0415
        _build_expert_mesh,
    )
    ep_size = next(
        (
            spec._ep_size  # pylint: disable=protected-access
            for spec in plan.modules.values()
            if spec._ep_size > 0  # pylint: disable=protected-access
        ),
        0,
    )
    if mesh_context is not None:
        expert_mesh = mesh_context.fsdp_moe_mesh
    elif ep_size > 0:
        expert_mesh = _build_expert_mesh(
            full_mesh,
            full_mesh.mesh_dim_names,
            ep_size,
        )
    else:
        expert_mesh = None
    if ep_size > 0 and expert_mesh is None:
        raise ValueError("Routed expert plan requires MeshContext.fsdp_moe_mesh")
    if expert_mesh is not None:
        logger.info(
            "expert mesh: using %s for parameter sharding, explicit compute injection, "
            "and FSDP source metadata",
            dict(zip(tuple(expert_mesh.mesh_dim_names), tuple(expert_mesh.mesh_shape))),
        )
    # Pass the full region meshes: build_source_shard_info strips the FSDP-owned
    # axes itself and records the complete source (non-FSDP) layout.
    dense_source_mesh = (
        mesh_context.fsdp_non_moe_mesh
        if mesh_context is not None and mesh_context.fsdp_non_moe_mesh is not None
        else tp_mesh
    )
    expert_source_mesh = (
        expert_mesh
        if expert_mesh is not None and "ep" in expert_mesh.mesh_dim_names
        else None
    )
    return expert_mesh, dense_source_mesh, expert_source_mesh


def _shard_planned_parameters(models, plan, mesh, expert_mesh, validate_mode):
    """Shard dense and expert parameters, then update local attention metadata."""
    for model in models:
        for module_fqn, spec in plan.modules.items():
            module = _resolve_module(model, module_fqn)
            # Runtime planner metadata is internal to the sharding pipeline.
            if spec._ep_stack:  # pylint: disable=protected-access
                _stack_moe_experts(
                    module,
                    spec._ep_stack,  # pylint: disable=protected-access
                )
            if spec._ep_size > 0:  # pylint: disable=protected-access
                expert_params = {
                    name: placement
                    for name, placement in spec.params.items()
                    if name.startswith("experts.")
                }
                dense_params = {
                    name: placement
                    for name, placement in spec.params.items()
                    if not name.startswith("experts.")
                }
                _shard_module_params(
                    module,
                    expert_params,
                    expert_mesh,
                    expert_mesh.mesh_dim_names,
                )
                _shard_module_params(
                    module,
                    dense_params,
                    mesh,
                    plan.mesh_dim_names,
                )
            else:
                _shard_module_params(
                    module,
                    spec.params,
                    mesh,
                    plan.mesh_dim_names,
                )
            if not validate_mode:
                maybe_update_head_counts(
                    module,
                    spec,
                    module_fqn,
                    mesh,
                    plan.mesh_dim_names,
                )


def _build_runtime_source_shard_info(
    models,
    plan,
    dense_source_mesh,
    expert_source_mesh,
    validate_mode,
):
    """Unwrap production parameters and build FSDP source-layout metadata."""
    if validate_mode:
        return None
    source_shard_records = {}
    for model in models:
        source_shard_records.update(_local_params_context(model))
    if not source_shard_records:
        return None
    if dense_source_mesh is None and expert_source_mesh is None:
        return None
    return build_source_shard_info(
        plan,
        dense_source_mesh,
        expert_source_mesh=expert_source_mesh,
    )


def _shard_module_params(module, param_specs, mesh, mesh_dim_names):
    """distribute_tensor() converts parameters into DTensors.

    - meta tensor -> DTensor: _local_tensor remains meta (zero-memory path);
    - real tensor -> DTensor: physically split; each rank holds a local shard;
    - already a DTensor: skipped if the placement matches, otherwise raises
      PlacementMismatchError.
    """
    for param_path, named in param_specs.items():
        param = _get_attr_by_path(module, param_path)
        placements = tuple(resolve_placements(named, mesh_dim_names))
        if not placements:
            continue  # no active DTensor axes (all size 1) -- no sharding needed

        if isinstance(param, DTensor):
            if tuple(param.placements) != placements:
                raise PlacementMismatchError(
                    f"{type(module).__name__}.{param_path}",
                    placements, tuple(param.placements), "params",
                )
            continue

        src = param.data if hasattr(param, "data") else param
        dt = distribute_tensor(src, mesh, placements)
        requires_grad = getattr(param, "requires_grad", True)
        _set_param_by_path(module, param_path,
                           nn.Parameter(dt, requires_grad=requires_grad))


# ────────────────────────────────────────────────────────────────────────────
# Phase D: tied weights
# ────────────────────────────────────────────────────────────────────────────

def detect_tied_weights(model: Any) -> List[Tuple[str, str]]:
    """Detect tied-weight pairs (embed_tokens.weight <-> lm_head.weight).

    In PP scenarios cross-stage pairs cannot be detected; the user must
    explicitly declare plan.tied_pairs.

    Args:
        model: The model to inspect (an HF-style ``nn.Module`` with a
            ``config`` carrying ``tie_word_embeddings``).

    Returns:
        A list of ``(embed_fqn, lm_head_fqn)`` tied-parameter FQN pairs.
    """
    tied = []
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        embed_fqn = lm_head_fqn = None
        # remove_duplicate=False: under the default dedup of named_parameters
        # a tied parameter appears only once; duplicates must be explicitly
        # retained to discover the FQNs of both ends.
        for name, _ in model.named_parameters(remove_duplicate=False):
            if name.endswith("embed_tokens.weight"):
                embed_fqn = name
            elif name.endswith("lm_head.weight"):
                lm_head_fqn = name
        if embed_fqn and lm_head_fqn:
            tied.append((embed_fqn, lm_head_fqn))
    return tied


def _broadcast_tied_param(model, tied_pair):
    """A tied-weight pair shares storage within this rank (end A's storage is authoritative; end B shares it).

    Cross-rank broadcast would be **wrong**: a tied pair (embed/lm_head) is
    usually Shard(0)-sharded on both ends, and each rank's local shard carries
    a different vocab interval -- broadcasting rank0's shard to rank1 would
    corrupt rank1's sharding. Tied semantics require that **within the same
    rank** the two ends are the same physical parameter (shared gradients),
    not cross-rank consistency (sharding is naturally consistent: same global
    source, same placement).
    """
    fqn_a, fqn_b = tied_pair
    try:
        param_a = _get_attr_by_path(model, fqn_a)
        param_b = _get_attr_by_path(model, fqn_b)
    except AttributeError:
        return
    if param_a is None or param_b is None:
        return
    tensor_a = param_a.to_local() if isinstance(param_a, DTensor) else param_a.data
    # B shares storage with A (a tied weight is the same physical parameter)
    if isinstance(param_b, DTensor):
        param_b._local_tensor = tensor_a  # pylint: disable=protected-access
    else:
        param_b.data = tensor_a


def _replicate_tied_weights(model, tied_pairs=None):
    """Phase D: replicate tied weights across ranks."""
    for tied_pair in (tied_pairs if tied_pairs is not None
                      else detect_tied_weights(model)):
        _broadcast_tied_param(model, tied_pair)

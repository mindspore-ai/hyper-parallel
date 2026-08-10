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
"""sharding.apply: _local_params_context / path utilities (canonical definitions, 05 §4.4).

06's dtensor_utils.py re-exports the definitions of this module — do not
create another copy.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn as nn

from hyper_parallel.core.dtensor.dtensor import DTensor

logger = logging.getLogger(__name__)


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
    (diagnostic use only; the canonical source for tp_grad_info is the
    ShardingPlan, see build_tp_grad_info).
    """
    tp_grad_records = {}
    for name, param in list(model.named_parameters()):
        if isinstance(param, DTensor):
            tp_grad_records[name] = param.placements
            _set_param_by_path(model, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    return tp_grad_records


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
            _set_param_by_path(module, name, nn.Parameter(
                param.to_local(), requires_grad=param.requires_grad))
    try:
        yield
    finally:
        for name, param in saved:
            _set_param_by_path(module, name, param)


# ────────────────────────────────────────────────────────────────────────────
# D-09: HF-native MoE parameter stacking (05 §6.4.7 D-09b)
# ────────────────────────────────────────────────────────────────────────────

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

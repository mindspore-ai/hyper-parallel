# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen3.5-MoE parallelization — self-contained per-model logic.

There
is no shared "default_parallelize" template — this file owns Qwen3.5-MoE's
full AC / FSDP / Prefetch pipeline. TP and EP are explicitly rejected (the
generic ``ParallelStyle``-based path produces incorrect numerics for grouped
experts, and full EP requires per-model dispatch/combine wiring that lives
here, not in a shared helper).
"""
from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel import (
    ColwiseParallel,
    ContextParallel,
    DTensor,
    HSDPModule,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    SkipDTensorDispatch,
    fully_shard,
    parallelize_module,
)
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from hyper_parallel.core.pipeline_parallel import (
    BatchDimSpec, PipelineStage, Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B)
from hyper_parallel.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM, Qwen3_5MoeStageModule
from hyper_parallel.platform import get_platform
from hyper_parallel.trainer.utils.logging import get_logger

if TYPE_CHECKING:
    from hyper_parallel.trainer.config import HyperTrainerConfig

platform = get_platform()
logger = get_logger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}

_PP_SCHEDULES = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
}


def _group_info(group):
    """Return the platform collectives group-info wrapper for a raw group."""
    return group if isinstance(group, str) else SimpleNamespace(group=group)


def _world_size() -> int:
    """Return world size, treating an uninitialized backend as single-rank."""
    try:
        return platform.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def _model_text_config(model: nn.Module):
    """Return the text config for text-only and VL Qwen3.5-MoE wrappers."""
    return getattr(model.config, "text_config", model.config)


def _all_reduce_sum_(tensor, group) -> None:
    """SUM-reduce ``tensor`` in-place through the platform collective API."""
    result = platform.all_reduce(tensor, _group_info(group))
    reduced = result[0] if isinstance(result, tuple) else result
    if reduced is not tensor:
        tensor.copy_(reduced)


def _broadcast_from_rank0_(tensor) -> None:
    """Broadcast ``tensor`` from global rank 0 when distributed is initialized."""
    if _world_size() <= 1:
        return
    platform.broadcast(tensor, src=0, group=None)


def _resolve_pp_schedule(name: str) -> type:
    """Return the single-stage pipeline schedule class selected by config."""
    key = str(name).lower().replace("-", "_")
    try:
        return _PP_SCHEDULES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pp_schedule {name!r}; choose from {sorted(_PP_SCHEDULES)}."
        ) from exc


def _resolve_stage_layer_counts(
    n_layers: int,
    num_stages: int,
    layer_split: Optional[list[int]] = None,
    *,
    allow_empty_stages: bool = True,
) -> list[int]:
    """Return decoder-layer counts for this model's global pipeline stages."""
    if layer_split:
        counts = [int(c) for c in layer_split]
        if len(counts) != num_stages:
            raise ValueError(
                f"pp_layer_split has {len(counts)} entries but pp*pp_vpp = "
                f"{num_stages} global stages."
            )
        if sum(counts) != n_layers:
            raise ValueError(
                f"pp_layer_split sums to {sum(counts)} but the model has "
                f"{n_layers} decoder layers."
            )
        if min(counts) < 0 or (not allow_empty_stages and min(counts) < 1):
            floor = 0 if allow_empty_stages else 1
            raise ValueError(
                f"pp_layer_split entries must be >= {floor}; got {counts}."
            )
        return counts
    base, rem = divmod(n_layers, num_stages)
    return [base + (1 if i >= num_stages - rem else 0) for i in range(num_stages)]


def _seq_full(x):
    """Gather a sequence-sharded DTensor to a local full-sequence tensor."""
    if isinstance(x, DTensor):
        full = x.redistribute(x.device_mesh, [Replicate()]).to_local()
        return full, (x.device_mesh, x.placements)
    return x, None


def _cp_gather(x, cp_mesh: DeviceMesh):
    """Gather a CP sequence shard (dim 1) to a local full-sequence tensor."""
    return DTensor.from_local(x, cp_mesh, [Shard(1)]).redistribute(
        cp_mesh, [Replicate()],
    ).to_local()


def _cp_slice(x, cp_mesh: DeviceMesh, dim: int = 1):
    """Return this CP rank's contiguous slice along ``dim``."""
    rank = cp_mesh.get_local_rank()
    size = cp_mesh.size()
    chunk = x.shape[dim] // size
    sl = [slice(None)] * x.dim()
    sl[dim] = slice(rank * chunk, (rank + 1) * chunk)
    return x[tuple(sl)].contiguous()


def _seq_restore(x, info):
    """Restore a full-sequence local tensor to its original DTensor layout."""
    if info is not None:
        mesh, placements = info
        local = _cp_slice(x, mesh, dim=1)
        return DTensor.from_local(local, mesh, list(placements))
    return x


def _has_shard_dim(mesh) -> bool:
    """Return ``True`` when the mesh has an explicit FSDP shard axis."""
    for dim in ("fsdp", "dp_shard"):
        try:
            submesh = mesh[dim]
            del submesh
            return True
        except (KeyError, TypeError):
            continue
    return False


def _has_nontrivial_shard_dim(mesh) -> bool:
    """Return ``True`` when the FSDP shard axis has more than one rank."""
    for dim in ("fsdp", "dp_shard"):
        try:
            if mesh[dim].size() > 1:
                return True
        except (KeyError, TypeError, AttributeError, RuntimeError):
            continue
    return False


def _resolve_qwen3_5_moe_pp_fsdp_mesh(mesh, cfg):
    """Return a safe per-stage FSDP mesh, or ``None`` for plain DP fallback."""
    if int(cfg.train.accelerator.dp_replicate or 1) > 1:
        logger.info_rank0(
            "Qwen3.5-MoE PP+HSDP leaves stages replicated so the trainer can "
            "synchronize plain gradients over the combined DP group."
        )
        return None
    return _resolve_fsdp_mesh(mesh) if _has_nontrivial_shard_dim(mesh) else None


def _resolve_mp_policy(cfg):
    """Build FSDP mixed-precision policy from the Qwen3.5-MoE YAML."""
    mp_cfg = cfg.train.mixed_precision
    if not mp_cfg.enabled:
        return None
    output_dtype_str = mp_cfg.output_dtype
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP.get(mp_cfg.param_dtype),
        reduce_dtype=_DTYPE_MAP.get(mp_cfg.reduce_dtype),
        output_dtype=_DTYPE_MAP.get(output_dtype_str) if output_dtype_str else None,
    )


def _build_fsdp_kwargs(module: nn.Module, dp_mesh: DeviceMesh, cfg) -> dict:
    """Assemble Qwen3.5-MoE FSDP kwargs over ``dp_mesh``."""
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": cfg.train.accelerator.reshard_after_forward,
        "comm_fusion": cfg.train.accelerator.comm_fusion,
    }
    mp_policy = _resolve_mp_policy(cfg)
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    if cfg.train.accelerator.cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    mesh_shape = getattr(dp_mesh, "mesh_shape", None)
    try:
        shard_size = int(mesh_shape[-1]) if mesh_shape else dp_mesh.size()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        shard_size = 1
    shard_dim_overrides: dict = {}
    replicate_params = set()
    if shard_size > 1:
        for _, param in module.named_parameters():
            if param.dim() == 0 or param.size(0) % shard_size == 0:
                continue
            shardable_dim = next(
                (dim for dim in range(1, param.dim()) if param.size(dim) % shard_size == 0),
                None,
            )
            if shardable_dim is not None:
                shard_dim_overrides[id(param)] = shardable_dim
            else:
                replicate_params.add(param)
    if shard_dim_overrides:
        overrides = shard_dim_overrides

        def _shard_placement_fn(param):
            dim = overrides.get(id(param))
            return Shard(dim) if dim is not None else None

        fsdp_kwargs["shard_placement_fn"] = _shard_placement_fn
    if replicate_params:
        fsdp_kwargs["replicate_params"] = replicate_params
    return fsdp_kwargs


def _without_forward_input_cast(fsdp_kwargs: dict) -> dict:
    """Return FSDP kwargs whose mixed-precision policy preserves input dtype."""
    boundary_kwargs = dict(fsdp_kwargs)
    mp_policy = boundary_kwargs.get("mp_policy")
    if mp_policy is not None:
        boundary_kwargs["mp_policy"] = replace(mp_policy, cast_forward_inputs=False)
    return boundary_kwargs


def broadcast_state_dict_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters from rank 0 so every rank starts from the same weights.

    Buffers are intentionally not broadcast: the rotary ``inv_freq`` table is
    rebuilt deterministically from the same theta / head_dim on every rank, so
    a broadcast would be redundant.
    """
    for param in model.parameters():
        _broadcast_from_rank0_(param.data)


def _is_rank_local_moe_tp_param(name: str) -> bool:
    """Return ``True`` for routed-expert params manually sharded over TP."""
    return (
        ".mlp.experts.gate_up_proj" in name
        or ".mlp.experts.down_proj" in name
    )


def _is_rank_local_linear_attention_param(name: str) -> bool:
    """Return ``True`` for GatedDeltaNet params manually sharded over TP heads."""
    return (
        ".linear_attn.in_proj_qkv.weight" in name
        or ".linear_attn.conv1d.weight" in name
        or ".linear_attn.dt_bias" in name
        or ".linear_attn.A_log" in name
    )


def _is_rank_local_tp_param(name: str) -> bool:
    """Return ``True`` for params whose TP shard is owned locally."""
    return _is_rank_local_moe_tp_param(name) or _is_rank_local_linear_attention_param(name)


def _is_sequence_parallel_norm_param(name: str) -> bool:
    """Return ``True`` for norm params whose DTensor backward already syncs TP."""
    return (
        name.endswith(".input_layernorm.weight")
        or name.endswith(".post_attention_layernorm.weight")
        or name in ("model.norm.weight", "model.language_model.norm.weight")
    )


def _is_ep_replicated_moe_param(name: str) -> bool:
    """Return ``True`` for MoE-local replicated params in a pure-EP block."""
    return ".mlp.gate.weight" in name or ".mlp.shared_expert." in name


def _is_ep_replicated_mtp_param(name: str) -> bool:
    """Return ``True`` for MTP-owned replicated params in a pure-EP block."""
    if not name.startswith("mtp."):
        return False
    return ".mlp.experts." not in name


def _needs_tp_grad_avg(name: str, gate_grad_avg: bool = True) -> bool:
    """Return ``True`` for MoE TP replicas that see the full token stream."""
    return (
        (gate_grad_avg and ".mlp.gate.weight" in name)
        or ".mlp.shared_expert_gate.weight" in name
    )


def _is_ep_replicated_avg_param(name: str) -> bool:
    """Return ``True`` for pure-EP replicated params that need AVG sync."""
    return _is_ep_replicated_moe_param(name) or _is_ep_replicated_mtp_param(name)


def _chain_grad_reducers(*reducers):
    """Compose optional post-backward gradient reducers."""
    active = tuple(reducer for reducer in reducers if reducer is not None)
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _reduce() -> None:
        for reducer in active:
            reducer()

    return _reduce


def _moe_expert_tp_slices(experts: nn.Module, tp_mesh: DeviceMesh):
    """Compute this TP rank's blockwise slices for packed routed experts."""
    tp_size = tp_mesh.size()
    intermediate_size = getattr(experts, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = getattr(experts, "intermediate_dim", None)
    if intermediate_size is None:
        raise ValueError(
            "Packed MoE expert params must expose intermediate_size or "
            "intermediate_dim for TP slicing."
        )
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"moe_intermediate_size ({intermediate_size}) must divide TP size ({tp_size})."
        )
    tp_rank = tp_mesh.get_local_rank()
    local_intermediate = intermediate_size // tp_size
    start = tp_rank * local_intermediate
    end = start + local_intermediate
    gate_slice = slice(start, end)
    up_slice = slice(intermediate_size + start, intermediate_size + end)
    down_slice = slice(start, end)
    return gate_slice, up_slice, down_slice


def _moe_expert_tp_layout(experts: nn.Module, intermediate_size: int, local_intermediate: int) -> str:
    """Return the packed expert layout used by ``experts`` after/full TP slicing."""
    gate_up = experts.gate_up_proj
    down = experts.down_proj
    if (
        gate_up.shape[1] in (2 * intermediate_size, 2 * local_intermediate)
        and down.shape[2] in (intermediate_size, local_intermediate)
    ):
        return "gate_up_out_in"
    if (
        gate_up.shape[2] in (2 * intermediate_size, 2 * local_intermediate)
        and down.shape[1] in (intermediate_size, local_intermediate)
    ):
        return "gate_up_in_out"
    raise ValueError(
        "Packed MoE expert params have unexpected shapes for TP slicing: "
        f"gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}, "
        f"intermediate={intermediate_size}, local={local_intermediate}."
    )


def _shard_moe_expert_tp_params(experts: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Slice packed routed-expert weights over the TP intermediate dimension.

    ``gate_up_proj`` is fused as ``[gate | up]`` along dim 1, so a plain
    contiguous ``Shard(1)`` would put all gate rows on rank 0 and all up rows on
    rank 1 when ``tp=2``. Slice the gate and up halves independently and
    concatenate this rank's shards back into the local packed layout.
    """
    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return
    intermediate_size = getattr(experts, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = getattr(experts, "intermediate_dim", None)
    if intermediate_size is None:
        raise ValueError(
            "Packed MoE expert params must expose intermediate_size or "
            "intermediate_dim for TP slicing."
        )
    local_intermediate = intermediate_size // tp_size
    layout = _moe_expert_tp_layout(experts, intermediate_size, local_intermediate)
    gate_up = experts.gate_up_proj
    down = experts.down_proj
    if layout == "gate_up_out_in":
        current_gate_up = gate_up.shape[1]
        current_down = down.shape[2]
    else:
        current_gate_up = gate_up.shape[2]
        current_down = down.shape[1]
    if current_gate_up == 2 * local_intermediate and current_down == local_intermediate:
        return
    if current_gate_up != 2 * intermediate_size or current_down != intermediate_size:
        raise ValueError(
            "Packed MoE expert params have unexpected shapes for TP slicing: "
            f"gate_up dim={current_gate_up}, down dim={current_down}, "
            f"expected full dims {2 * intermediate_size}/{intermediate_size}."
        )
    gate_slice, up_slice, down_slice = _moe_expert_tp_slices(experts, tp_mesh)
    if layout == "gate_up_out_in":
        gate_up_local = torch.cat(
            [gate_up.data[:, gate_slice, :], gate_up.data[:, up_slice, :]],
            dim=1,
        ).contiguous()
        down_local = down.data[:, :, down_slice].contiguous()
    else:
        gate_up_local = torch.cat(
            [gate_up.data[:, :, gate_slice], gate_up.data[:, :, up_slice]],
            dim=2,
        ).contiguous()
        down_local = down.data[:, down_slice, :].contiguous()
    experts.gate_up_proj = nn.Parameter(gate_up_local)
    experts.down_proj = nn.Parameter(down_local)


def _gated_delta_tp_slices(linear_attn: nn.Module, tp_mesh: DeviceMesh):
    """Compute rank-local channel / head slices for GatedDeltaNet TP."""
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    head_k_dim = linear_attn.head_k_dim
    head_v_dim = linear_attn.head_v_dim
    key_dim = linear_attn.key_dim
    n_v_local = linear_attn.num_v_heads // tp_size
    head_start = tp_rank * n_v_local
    head_end = head_start + n_v_local
    n_k_local = linear_attn.num_k_heads // tp_size
    k_head_start = tp_rank * n_k_local
    k_head_end = k_head_start + n_k_local
    q_slice = slice(k_head_start * head_k_dim, k_head_end * head_k_dim)
    k_slice = slice(key_dim + k_head_start * head_k_dim, key_dim + k_head_end * head_k_dim)
    v_slice = slice(2 * key_dim + head_start * head_v_dim, 2 * key_dim + head_end * head_v_dim)
    return q_slice, k_slice, v_slice, head_start, head_end


def _shard_gated_delta_local_params(linear_attn: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Slice GatedDeltaNet's fused QKV / conv / SSM params over TP heads."""
    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return
    q_slice, k_slice, v_slice, head_start, head_end = _gated_delta_tp_slices(linear_attn, tp_mesh)

    qkv = linear_attn.in_proj_qkv
    qkv_local = torch.cat(
        [qkv.weight.data[q_slice], qkv.weight.data[k_slice], qkv.weight.data[v_slice]],
        dim=0,
    ).contiguous()
    qkv.weight = nn.Parameter(qkv_local)
    qkv.out_features = qkv_local.shape[0]

    conv = linear_attn.conv1d
    conv_local = torch.cat(
        [conv.weight.data[q_slice], conv.weight.data[k_slice], conv.weight.data[v_slice]],
        dim=0,
    ).contiguous()
    conv.weight = nn.Parameter(conv_local)
    conv_local_groups = conv_local.shape[0]
    conv.in_channels = conv_local_groups
    conv.out_channels = conv_local_groups
    conv.groups = conv_local_groups

    linear_attn.dt_bias = nn.Parameter(
        linear_attn.dt_bias.data[head_start:head_end].clone()
    )
    linear_attn.A_log = nn.Parameter(
        linear_attn.A_log.data[head_start:head_end].clone()
    )


# TP helper for modules that expose Qwen3_5Attention /
# Qwen3_5GatedDeltaNet-style attributes.


def _register_tp_replicated_param_grad_sum(model, tp_mesh: DeviceMesh, eager: bool = False) -> None:
    """Reduce, across the TP/SP mesh, the grads of every replicated weight.

    Under the SequenceParallel TP plan each rank routes a different token
    (sequence) shard, so a weight that is *replicated* across the mesh receives
    only its own rank's partial gradient unless the module first gathers the
    sequence back to Replicate. Partial replicated grads are SUM-reduced; MoE
    gate replicas that run on the full gathered sequence are AVG-reduced.
    Two parameter classes are replicated and need this:

    * plain ``nn.Parameter`` replicas — the per-head norms, the MoE router
      ``gate``, the shared-expert gate, and the ``linear_attn`` projections;
    SequenceParallel norm DTensors already carry their TP layout through
    autograd, so they are skipped. Other replicated weights still need this SUM;
    MoE router and shared-expert gates see the full sequence on every TP rank,
    so their TP replica grads are averaged instead of summed;
    sharded weights (``embed_tokens`` / ``lm_head``, the column/row-parallel
    attention projections, the shared expert, and the routed experts' manual TP
    intermediate shards) already own only this rank's shard, so they are skipped.

    The hooks are attached lazily through a one-shot forward pre-hook: the
    trainer's ``model.to_empty`` meta-materialisation replaces every
    ``nn.Parameter`` with a fresh object, silently dropping any grad hook
    registered at parallelize time. By the first forward the parameters are
    final, and the per-parameter closure defers the sharded-or-replicated test to
    backward time (when the placements are settled), so the registration order of
    the various ``parallelize_module`` conversions does not matter.
    """
    if tp_mesh.size() <= 1:
        return
    tp_group = tp_mesh.get_group()
    skip_moe_expert_grads = getattr(model, "_hp_moe_experts_tp_sharded", True)

    def _sharded_over_tp_mesh(param) -> bool:
        # A weight whose grad is already correct per rank is one that is sharded
        # *along the TP/SP mesh itself* (``embed_tokens`` / ``lm_head`` on the
        # vocab dim, or the column/row-parallel attention projections). A weight
        # sharded over a *different* mesh (e.g. the EP experts on the ``ep`` dim)
        # is still replicated w.r.t. TP, so its grad is a per-rank partial that
        # must be summed across TP.
        if not isinstance(param, DTensor):
            return False
        if not any(pl.is_shard() for pl in param.placements):
            return False
        try:
            return param.device_mesh.get_group() == tp_group
        except (RuntimeError, ValueError, AttributeError):
            return False

    def _make_grad_hook(name, param):
        # ``_sharded_over_tp_mesh`` is constant per param but the placements are
        # only settled by the first backward, so classify lazily on first fire
        # and cache the boolean for every later step.
        decision = {"reduce": None}
        gate_grad_avg = getattr(model, "_hp_moe_tp_gate_grad_avg", True)
        grad_avg = _needs_tp_grad_avg(name, gate_grad_avg)

        def _hook(grad):
            if _is_sequence_parallel_norm_param(name):
                return grad
            if decision["reduce"] is None:
                decision["reduce"] = not _sharded_over_tp_mesh(param)
            if not decision["reduce"]:
                return grad
            # Replicated weight → SUM its per-rank partial grad across the TP
            # mesh. SequenceParallel norm DTensors are skipped above because
            # their placement-aware backward already handles TP layout semantics.
            if isinstance(grad, DTensor):
                local = grad.to_local()
                _all_reduce_sum_(local, tp_group)
                if grad_avg:
                    local.div_(tp_mesh.size())
                return grad
            if not grad.is_contiguous():
                grad = grad.contiguous()
            _all_reduce_sum_(grad, tp_group)
            if grad_avg:
                grad.div_(tp_mesh.size())
            return grad
        return _hook

    attached = {"done": False}

    def _attach_grad_hooks(module, hook_args):
        del hook_args
        if attached["done"]:
            return
        for name, param in module.named_parameters():
            if skip_moe_expert_grads and _is_rank_local_tp_param(name):
                continue
            if param.requires_grad:
                if getattr(param, "_hp_tp_grad_sum_hook_attached", False):
                    continue
                param.register_hook(_make_grad_hook(name, param))
                param._hp_tp_grad_sum_hook_attached = True  # pylint: disable=protected-access
        attached["done"] = True

    if eager:
        # PP stages: the schedule drives stage fragments, never the module
        # root, so the lazy forward hook would not fire. The PP-alone path
        # materializes BEFORE the pipeline split, so the params are final and
        # eager registration survives.
        _attach_grad_hooks(model, None)
        return
    model.register_forward_pre_hook(_attach_grad_hooks)


def _register_ep_replicated_param_grad_sum(model, ep_mesh: DeviceMesh, eager: bool = False) -> None:
    """SUM-reduce non-expert replicated grads over the EP mesh.

    In this implementation pure EP is an independent model-parallel axis: every
    EP rank runs the same token stream, while routed experts are sharded by
    expert id. When no TP axis is present, the trainer divides the replicated
    loss by ``ep_size``; routed-expert shards then receive the correct gradient
    through token dispatch, while router / shared-expert / attention / norm
    replicas need one EP SUM to match the single-card update. When TP is also
    present, the TP loss scale already accounts for the duplicated expert
    gradient path, so these EP-only hooks are not installed.
    """
    if ep_mesh.size() <= 1:
        return
    ep_group = ep_mesh.get_group()

    def _make_grad_hook():

        def _hook(grad):
            if isinstance(grad, DTensor):
                _all_reduce_sum_(grad.to_local(), ep_group)
                return grad
            if not grad.is_contiguous():
                grad = grad.contiguous()
            _all_reduce_sum_(grad, ep_group)
            return grad
        return _hook

    attached = {"done": False}

    def _attach_grad_hooks(module, hook_args):
        del hook_args
        if attached["done"]:
            return
        for name, param in module.named_parameters():
            if _is_rank_local_tp_param(name):
                continue
            if param.requires_grad:
                if getattr(param, "_hp_ep_grad_sum_hook_attached", False):
                    continue
                param.register_hook(_make_grad_hook())
                param._hp_ep_grad_sum_hook_attached = True  # pylint: disable=protected-access
        attached["done"] = True

    if eager:
        _attach_grad_hooks(model, None)
        return
    model.register_forward_pre_hook(_attach_grad_hooks)


def _register_ep_replicated_moe_grad_avg(model, ep_mesh: DeviceMesh, eager: bool = False) -> None:
    """AVG-reduce MoE/MTP-local replicated grads over the EP mesh."""
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        return
    ep_group = ep_mesh.get_group()

    def _make_grad_hook(fp32_reduce: bool):

        def _avg_reduce(local_grad: platform.Tensor) -> None:
            reduce_target = local_grad.contiguous() if not local_grad.is_contiguous() else local_grad
            if fp32_reduce and reduce_target.dtype != torch.float32:
                buf = reduce_target.to(torch.float32)
                _all_reduce_sum_(buf, ep_group)
                buf.div_(ep_size)
                reduce_target.copy_(buf.to(reduce_target.dtype))
            else:
                _all_reduce_sum_(reduce_target, ep_group)
                reduce_target.div_(ep_size)
            if reduce_target is not local_grad:
                local_grad.copy_(reduce_target)

        def _hook(grad):
            if isinstance(grad, DTensor):
                _avg_reduce(grad.to_local())
                return grad
            _avg_reduce(grad)
            return grad
        return _hook

    attached = {"done": False}

    def _attach_grad_hooks(module, hook_args):
        del hook_args
        if attached["done"]:
            return
        for name, param in module.named_parameters():
            if param.requires_grad and _is_ep_replicated_avg_param(name):
                if getattr(param, "_hp_ep_moe_grad_avg_hook_attached", False):
                    continue
                param.register_hook(_make_grad_hook(fp32_reduce=True))
                param._hp_ep_moe_grad_avg_hook_attached = True  # pylint: disable=protected-access
        attached["done"] = True

    if eager:
        _attach_grad_hooks(model, None)
        return
    model.register_forward_pre_hook(_attach_grad_hooks)


def _make_post_fsdp_tp_reducer(model, tp_mesh: DeviceMesh):
    """Build a post-FSDP callback that syncs TP-replicated grads over TP.

    Under FSDP the per-parameter ``register_hook`` path (which fixes pure TP)
    never fires — FSDP reduce-scatters the *unsharded* grad before the sharded
    leaf's hook would run. The trainer invokes this once the FSDP reduction has
    drained (after ``hsdp_sync_stream``); it syncs every TP-replicated grad,
    while skipping SequenceParallel norm DTensors whose layout-aware backward
    already handled the TP mesh and parameters whose weight shard is rank-local
    over TP. MoE gate replicas that run on a full gathered sequence are averaged;
    the remaining replicated grads are summed.

    Returns ``None`` when ``tp_mesh`` is trivial so the trainer skips the call.
    """
    if tp_mesh.size() <= 1:
        return None
    tp_group = tp_mesh.get_group()
    skip_moe_expert_grads = getattr(model, "_hp_moe_experts_tp_sharded", True)

    def _sharded_over_tp_mesh(param) -> bool:
        if not isinstance(param, DTensor):
            return False
        device_mesh = param.device_mesh
        for dim in range(device_mesh.ndim):
            try:
                if device_mesh.get_group(dim) == tp_group and param.placements[dim].is_shard():
                    return True
            except (RuntimeError, ValueError, KeyError):
                continue
        return False

    def _should_reduce(name, param) -> bool:
        """Return whether a parameter needs the post-FSDP TP reduction."""
        return (
            param.requires_grad
            and not _is_sequence_parallel_norm_param(name)
            and not (skip_moe_expert_grads and _is_rank_local_tp_param(name))
            and not _sharded_over_tp_mesh(param)
        )

    # The replicated/sharded split is constant once the layout is materialized,
    # so resolve the param list on the first call (params are final by then) and
    # reuse it every step instead of re-walking placements each train step.
    to_reduce: list = []
    resolved = {"done": False}

    def _reduce() -> None:
        if not resolved["done"]:
            to_reduce.extend(
                (name, param) for name, param in model.named_parameters()
                if _should_reduce(name, param)
            )
            resolved["done"] = True
        with SkipDTensorDispatch():
            for name, param in to_reduce:
                grad = param.grad
                if grad is None:
                    continue
                local = grad._local_tensor if isinstance(grad, DTensor) else grad  # pylint: disable=protected-access
                reduce_target = local.contiguous() if not local.is_contiguous() else local
                if local.device.type == "cpu":
                    # cpu_offload keeps the sharded grad on CPU, but the TP
                    # group is HCCL/NCCL-only: reduce a device copy and
                    # write the result back into the offloaded storage.
                    buf = reduce_target.to(tp_mesh.device_type)
                    _all_reduce_sum_(buf, tp_group)
                    if _needs_tp_grad_avg(name, getattr(model, "_hp_moe_tp_gate_grad_avg", True)):
                        buf.div_(tp_mesh.size())
                    local.copy_(buf)
                else:
                    _all_reduce_sum_(reduce_target, tp_group)
                    if _needs_tp_grad_avg(name, getattr(model, "_hp_moe_tp_gate_grad_avg", True)):
                        reduce_target.div_(tp_mesh.size())
                    if reduce_target is not local:
                        local.copy_(reduce_target)

    return _reduce


def _make_post_fsdp_ep_reducer(model, ep_mesh: DeviceMesh):
    """Build a post-FSDP EP SUM reducer for non-expert replicated grads."""
    if ep_mesh.size() <= 1:
        return None
    ep_group = ep_mesh.get_group()
    to_reduce: list = []
    resolved = {"done": False}

    def _reduce() -> None:
        if not resolved["done"]:
            to_reduce.extend(
                param for name, param in model.named_parameters()
                if param.requires_grad and not _is_rank_local_tp_param(name)
            )
            resolved["done"] = True
        with SkipDTensorDispatch():
            for param in to_reduce:
                grad = param.grad
                if grad is None:
                    continue
                local = grad._local_tensor if isinstance(grad, DTensor) else grad  # pylint: disable=protected-access
                reduce_target = local.contiguous() if not local.is_contiguous() else local
                if local.device.type == "cpu":
                    buf = reduce_target.to(ep_mesh.device_type)
                    _all_reduce_sum_(buf, ep_group)
                    local.copy_(buf)
                else:
                    _all_reduce_sum_(reduce_target, ep_group)
                    if reduce_target is not local:
                        local.copy_(reduce_target)

    return _reduce


def _make_post_fsdp_ep_avg_reducer(model, ep_mesh: DeviceMesh):
    """Build a post-FSDP EP AVG reducer for non-expert replicated grads."""
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        return None
    ep_group = ep_mesh.get_group()
    to_reduce: list = []
    resolved = {"done": False}

    def _reduce() -> None:
        if not resolved["done"]:
            to_reduce.extend(
                param for name, param in model.named_parameters()
                if param.requires_grad and not _is_rank_local_tp_param(name)
            )
            resolved["done"] = True
        with SkipDTensorDispatch():
            for param in to_reduce:
                grad = param.grad
                if grad is None:
                    continue
                local = grad._local_tensor if isinstance(grad, DTensor) else grad  # pylint: disable=protected-access
                reduce_target = local.contiguous() if not local.is_contiguous() else local
                if local.device.type == "cpu":
                    buf = reduce_target.to(ep_mesh.device_type)
                    _all_reduce_sum_(buf, ep_group)
                    buf.div_(ep_size)
                    local.copy_(buf)
                else:
                    _all_reduce_sum_(reduce_target, ep_group)
                    reduce_target.div_(ep_size)
                    if reduce_target is not local:
                        local.copy_(reduce_target)

    return _reduce


def _make_post_fsdp_ep_moe_avg_reducer(model, ep_mesh: DeviceMesh):
    """Build a post-FSDP EP AVG reducer for MoE/MTP-local replicated grads."""
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        return None
    ep_group = ep_mesh.get_group()
    to_reduce: list = []
    resolved = {"done": False}

    def _reduce() -> None:
        if not resolved["done"]:
            to_reduce.extend(
                (param, True)
                for name, param in model.named_parameters()
                if param.requires_grad and _is_ep_replicated_avg_param(name)
            )
            resolved["done"] = True
        with SkipDTensorDispatch():
            for param, fp32_reduce in to_reduce:
                grad = param.grad
                if grad is None:
                    continue
                local = grad._local_tensor if isinstance(grad, DTensor) else grad  # pylint: disable=protected-access
                reduce_target = local.contiguous() if not local.is_contiguous() else local
                if local.device.type == "cpu":
                    buf = reduce_target.to(ep_mesh.device_type)
                    if fp32_reduce and buf.dtype != torch.float32:
                        buf = buf.to(torch.float32)
                    _all_reduce_sum_(buf, ep_group)
                    buf.div_(ep_size)
                    local.copy_(buf.to(local.dtype))
                else:
                    if fp32_reduce and reduce_target.dtype != torch.float32:
                        buf = reduce_target.to(torch.float32)
                        _all_reduce_sum_(buf, ep_group)
                        buf.div_(ep_size)
                        reduce_target.copy_(buf.to(reduce_target.dtype))
                    else:
                        _all_reduce_sum_(reduce_target, ep_group)
                        reduce_target.div_(ep_size)
                    if reduce_target is not local:
                        local.copy_(reduce_target)

    return _reduce


def _make_post_fsdp_ep_expert_grad_divider(model, ep_mesh: DeviceMesh):
    """Build a post-FSDP divider for pure-EP duplicated expert gradients."""
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        return None
    to_divide: list = []
    resolved = {"done": False}

    def _divide() -> None:
        if not resolved["done"]:
            to_divide.extend(
                param for name, param in model.named_parameters()
                if param.requires_grad and _is_rank_local_moe_tp_param(name)
            )
            resolved["done"] = True
        with SkipDTensorDispatch():
            for param in to_divide:
                grad = param.grad
                if grad is None:
                    continue
                local = grad._local_tensor if isinstance(grad, DTensor) else grad  # pylint: disable=protected-access
                local.div_(ep_size)

    return _divide


def _apply_ac(model, cfg) -> None:
    """Apply ac (internal)."""
    ac_mode = cfg.train.gradient_checkpointing.activation_checkpoint
    if ac_mode in ("off", "none", None, False, ""):
        return
    if not hasattr(model, "layers"):
        logger.warning("AC enabled but model has no .layers; skipping.")
        return

    layers = list(model.layers)
    for i, layer in enumerate(layers):
        model.layers[i] = checkpoint_wrapper(layer)
    logger.info_rank0("AC applied to %d Qwen3.5-MoE layers (mode=%s)", len(layers), ac_mode)


def _resolve_fsdp_mesh(mesh):
    """Return the 1-D FSDP ``DeviceMesh`` (``fsdp`` → ``dp_shard`` → whole mesh)."""
    try:
        return mesh["fsdp"]
    except (KeyError, TypeError):
        try:
            return mesh["dp_shard"]
        except (KeyError, TypeError):
            return mesh


def _resolve_qwen3_5_moe_dp_sizes(mesh, cfg) -> tuple[int, int]:
    """Return configured replicate size and the resolved shard-axis size."""
    accelerator = cfg.train.accelerator
    replicate_size = int(accelerator.dp_replicate or 1)
    shard_size = int(accelerator.dp_shard or 1)
    if shard_size == -1:
        try:
            shard_size = int(mesh["dp_shard"].size())
        except (KeyError, TypeError, AttributeError, RuntimeError, ValueError):
            shard_size = 1
    return replicate_size, shard_size


def _resolve_qwen3_5_moe_dp_mesh(mesh, cfg):
    """Resolve the FSDP/HSDP mesh for data-parallel synchronization.

    HSDP requires the explicit two-dimensional mesh so ``fully_shard`` can
    reduce-scatter over ``dp_shard`` and then all-reduce over
    ``dp_replicate``. A flattened one-dimensional alias loses that distinction.
    """
    replicate_size, _ = _resolve_qwen3_5_moe_dp_sizes(mesh, cfg)
    if replicate_size > 1:
        try:
            return mesh[("dp_replicate", "dp_shard")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Qwen3.5-MoE HSDP requires a DeviceMesh with ordered "
                "'dp_replicate' and 'dp_shard' dimensions."
            ) from exc
    return _resolve_fsdp_mesh(mesh)


def _has_configured_parallel_axis(cfg, names) -> bool:
    """Return whether any configured parallel dimension is larger than one."""
    accelerator = cfg.train.accelerator
    values = {
        "dp_replicate": accelerator.dp_replicate,
        "dp_shard": accelerator.dp_shard,
        "tp": accelerator.tp,
        "cp": accelerator.cp,
        "pp": accelerator.pp,
        "ep": accelerator.ep,
        "etp": accelerator.etp,
    }
    return any(int(values[name] or 1) > 1 for name in names)


def _is_qwen3_5_moe_vl(model, cfg) -> bool:
    """Return whether this run owns the Qwen3.5-MoE VL visual tower."""
    extra = cfg.model.config_overrides
    return (
        isinstance(extra, dict)
        and extra.get("vl", False)
        and hasattr(model, "model")
        and hasattr(model.model, "visual")
    )


def _apply_vl_visual_tower(model, mesh, cfg) -> None:
    """FSDP-wrap the Qwen3.5 VL visual tower separately."""
    if not _is_qwen3_5_moe_vl(model, cfg):
        return
    has_parallel_axis = _has_configured_parallel_axis(
        cfg,
        ("dp_shard", "dp_replicate", "tp", "ep", "cp", "pp"),
    )
    if _world_size() == 1 and not has_parallel_axis:
        logger.info_rank0(
            "Single-rank Qwen3.5-MoE VL run has no parallel axes; "
            "skipping visual tower FSDP wrap."
        )
        return

    visual = model.model.visual
    if not any(param.requires_grad for param in visual.parameters()):
        logger.info_rank0(
            "Qwen3.5-MoE VL visual tower is frozen; keeping it replicated "
            "instead of FSDP-wrapping frozen ViT blocks."
        )
        return

    dp_mesh = _resolve_qwen3_5_moe_dp_mesh(mesh, cfg)
    fsdp_kwargs = _build_fsdp_kwargs(visual, dp_mesh, cfg)
    mp_policy = fsdp_kwargs.get("mp_policy")
    if mp_policy is not None:
        # The frozen visual tower should keep the same bf16 activation flow as
        # the single-card path. The text backbone asks FSDP to cast outputs to
        # fp32 for residual math; applying that to each vision block changes the
        # merger input dtype and drifts the multimodal loss.
        mp_policy.output_dtype = None
    if hasattr(visual, "blocks"):
        for block in visual.blocks:
            fully_shard(block, **fsdp_kwargs)
    merger = getattr(visual, "merger", None)
    if merger is not None:
        fully_shard(merger, **fsdp_kwargs)
    for merger in getattr(visual, "deepstack_merger_list", ()):
        fully_shard(merger, **fsdp_kwargs)
    fully_shard(visual, **fsdp_kwargs)
    logger.info_rank0("Qwen3.5-MoE VL visual tower wrapped (per-block + merger + root)")


def _redistribute_first_tensor(
    tensor: platform.Tensor,
    mesh: DeviceMesh,
    input_layout,
    desired_layout,
    *,
    use_local_output: bool,
):
    """Redistribute one tensor at a Qwen3.5-MoE model-parallel hook boundary."""
    if isinstance(tensor, DTensor):
        dtensor = tensor
    else:
        dtensor = DTensor.from_local(tensor, mesh, [input_layout])
    if tuple(dtensor.placements) != (desired_layout,):
        dtensor = dtensor.redistribute(mesh, [desired_layout])
    return dtensor.to_local() if use_local_output else dtensor


def _apply_linear_attention_cp(module: nn.Module, cp_mesh: DeviceMesh) -> None:
    """Gather full sequence for MoE linear attention and slice before ``out_proj``."""

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        if args:
            hidden_states = args[0]
            rest = args[1:]
        else:
            hidden_states = kwargs.get("hidden_states")
            rest = None
        if hidden_states is None:
            raise ValueError("linear attention CP hook expects hidden_states")
        hidden_states = _redistribute_first_tensor(
            hidden_states,
            cp_mesh,
            Shard(1),
            Replicate(),
            use_local_output=True,
        )
        if rest is None:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hidden_states
            return args, kwargs
        return (hidden_states, *rest), kwargs

    def _post_hook(hook_module, hook_args, output):
        del hook_module, hook_args
        return _redistribute_first_tensor(
            output,
            cp_mesh,
            Replicate(),
            Shard(1),
            use_local_output=True,
        )

    module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    module.out_proj_input.register_forward_hook(_post_hook)


def _apply_shared_expert_cp_full_sequence(module: nn.Module, cp_mesh: DeviceMesh) -> None:
    """Run the shared expert on full CP sequence while keeping routed experts local."""

    def forward(mlp, x: platform.Tensor) -> platform.Tensor:
        """Forward with the shared expert evaluated on the full CP sequence."""
        bsz, seq_len, hidden_dim = x.shape
        local_flat = x.reshape(-1, hidden_dim)
        router_logits, top_value, top_index = mlp.gate(local_flat)
        mlp.router_logits = router_logits
        routed = mlp.experts(local_flat, top_index, top_value)

        full_x = _redistribute_first_tensor(
            x,
            cp_mesh,
            Shard(1),
            Replicate(),
            use_local_output=True,
        )
        full_flat = full_x.reshape(-1, hidden_dim)
        shared_out = mlp.shared_expert(full_flat)
        shared_out = torch.sigmoid(mlp.shared_expert_gate(full_flat)) * shared_out
        shared_out = _cp_slice(
            shared_out.view(full_x.shape[0], full_x.shape[1], hidden_dim),
            cp_mesh,
            dim=1,
        ).reshape(-1, hidden_dim)
        return (routed + shared_out).view(bsz, seq_len, hidden_dim)

    module.forward = MethodType(forward, module)


def _apply_text_input_cp_slice(module: nn.Module, cp_mesh: DeviceMesh) -> None:
    """Slice VL text-backbone inputs to the local CP sequence shard."""

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        kwargs = dict(kwargs)
        inputs_embeds = kwargs.get("inputs_embeds")
        position_ids = kwargs.get("position_ids")
        if inputs_embeds is None:
            return args, kwargs
        inputs_embeds, tp_placement = _seq_full(inputs_embeds)
        inputs_embeds = _cp_slice(inputs_embeds, cp_mesh, dim=1)
        kwargs["inputs_embeds"] = _seq_restore(inputs_embeds, tp_placement)
        if position_ids is not None:
            kwargs["position_ids"] = _cp_slice(position_ids, cp_mesh, dim=position_ids.ndim - 1)
        return args, kwargs

    module.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def _apply_full_sequence_input_cp_gather(module: nn.Module, cp_mesh: DeviceMesh) -> None:
    """Gather CP-sharded VL token inputs to full sequence before visual fusion."""

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        args = list(args)
        kwargs = dict(kwargs)
        if args:
            args[0] = _cp_gather(args[0], cp_mesh)
            if len(args) > 1 and args[1] is not None:
                args[1] = _cp_gather(args[1], cp_mesh)
        elif kwargs.get("input_ids") is not None:
            kwargs["input_ids"] = _cp_gather(kwargs["input_ids"], cp_mesh)
            if kwargs.get("attention_mask") is not None:
                kwargs["attention_mask"] = _cp_gather(kwargs["attention_mask"], cp_mesh)
        return tuple(args), kwargs

    module.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def _apply_visual_sequence_roundtrip(module: nn.Module) -> None:
    """Gather/restore sequence-sharded VL embeddings around visual token injection."""
    placements = []

    def _pre_hook(hook_module, args):
        del hook_module
        hidden_states = args[0]
        hidden_states, tp_placement = _seq_full(hidden_states)
        placements.append(tp_placement)
        return (hidden_states,)

    def _post_hook(hook_module, hook_args, output):
        del hook_module, hook_args
        tp_placement = placements.pop() if placements else None
        return _seq_restore(output, tp_placement)

    module.visual_injection_input.register_forward_pre_hook(_pre_hook)
    module.visual_injection_output.register_forward_hook(_post_hook)


def _apply_replicated_linear_out_proj_tp(module: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Run linear-attention ``out_proj`` as a replicated full matmul in deterministic TP."""

    def _pre_hook(hook_module, args):
        del hook_module
        hidden_states = args[0]
        full_hidden = DTensor.from_local(
            hidden_states,
            tp_mesh,
            [Shard(-1)],
        ).redistribute(tp_mesh, [Replicate()]).to_local()
        return (full_hidden,)

    def _post_hook(hook_module, hook_args, output):
        del hook_module, hook_args
        return DTensor.from_local(
            output,
            tp_mesh,
            [Replicate()],
        ).redistribute(tp_mesh, [Shard(1)])

    module.register_forward_pre_hook(_pre_hook)
    module.register_forward_hook(_post_hook)


def _apply_replicated_attention_out_proj_tp(module: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Run attention ``o_proj`` on gathered local heads when it is not sharded."""

    def _pre_hook(hook_module, args):
        if not args:
            return args
        hidden_states = args[0]
        in_features = getattr(hook_module, "in_features", None)
        if isinstance(hidden_states, DTensor):
            if tuple(hidden_states.placements) == (Replicate(),):
                return (hidden_states.to_local(), *args[1:])
            return (
                hidden_states.redistribute(tp_mesh, [Replicate()]).to_local(),
                *args[1:],
            )
        if (
            platform.is_tensor(hidden_states)
            and in_features is not None
            and hidden_states.shape[-1] * tp_mesh.size() == in_features
        ):
            full_hidden = DTensor.from_local(
                hidden_states,
                tp_mesh,
                [Shard(-1)],
            ).redistribute(tp_mesh, [Replicate()]).to_local()
            return (full_hidden, *args[1:])
        return args

    module.register_forward_pre_hook(_pre_hook)


def _local_expert_params(experts: nn.Module) -> tuple[platform.Tensor, platform.Tensor]:
    """Return packed expert weights as local tensors for routed EP grouped matmul."""
    gate_up = experts.gate_up_proj
    down = experts.down_proj
    gate_up = gate_up.to_local() if isinstance(gate_up, DTensor) else gate_up
    down = down.to_local() if isinstance(down, DTensor) else down
    return gate_up, down


def _make_shared_expert_tp_forward():
    """Build the TP-only MoE forward that returns a local hidden partial."""

    def _forward(mlp, x: platform.Tensor) -> platform.Tensor:
        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        if hasattr(mlp, "shared_expert"):
            shared_out = mlp.shared_expert(x_flat)
            router_logits, top_value, top_index = mlp.gate(x_flat)
            mlp.router_logits = router_logits
            routed = mlp.experts(x_flat, top_index, top_value)
            shared_out = torch.sigmoid(mlp.shared_expert_gate(x_flat)) * shared_out
            out = routed + shared_out
        else:
            router_logits, top_value, top_index = mlp.gate(x_flat)
            mlp.router_logits = router_logits
            out = mlp.experts(x_flat, top_index, top_value)

        # Routed experts and shared experts are both TP partials over the hidden
        # output. The MoE boundary hook below wraps this local value as
        # DTensor(Partial) and reduce-scatters it back to SequenceParallel.
        return out.view(bsz, seq_len, hidden_dim)

    return _forward


def _apply_shared_expert_tp_forward(mlp: nn.Module) -> None:
    """Install the TP MoE forward on a shared-expert MoE block."""
    mlp._hp_moe_tp_enabled = True  # pylint: disable=protected-access
    mlp.experts._hp_moe_tp_enabled = True  # pylint: disable=protected-access
    mlp.forward = MethodType(_make_shared_expert_tp_forward(), mlp)


def _expert_parallel_routed_forward(
    mlp: nn.Module,
    x_flat: platform.Tensor,
    top_index: platform.Tensor,
    top_value: platform.Tensor,
    ep_style: ExpertParallel,
    ep_mesh: DeviceMesh,
) -> platform.Tensor:
    """Run sorted routed tokens through local experts and combine EP outputs."""
    num_tokens, hidden_dim = x_flat.shape
    num_top_k = top_index.size(-1)
    device = x_flat.device

    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    )
    expert_ids = top_index.reshape(-1)
    sample_weights = top_value.reshape(-1)

    # Stable expert-id sort preserves token/top-k order before all-to-all.
    # Enable it for VL pure-EP groups; TP/CP compositions keep the local
    # grouped-mm tie order because they already change the token partition path.
    perm = torch.argsort(
        expert_ids,
        stable=getattr(mlp, "_hp_moe_ep_stable_sort", False),
    )
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    sorted_hidden = x_flat[token_idx[perm]]

    histc_input = expert_ids.float() if device.type == "cpu" else expert_ids.int()
    num_tokens_per_expert = torch.histc(
        histc_input, bins=mlp.num_experts, min=0, max=mlp.num_experts - 1,
    ).to(torch.int64)

    permuted, local_counts = ep_style._token_dispatch(  # pylint: disable=W0212
        mlp.experts, (sorted_hidden, num_tokens_per_expert), ep_mesh,
    )
    gate_up, down = _local_expert_params(mlp.experts)
    expert_out = mlp.experts.grouped_forward(
        permuted,
        local_counts,
        gate_up=gate_up,
        down=down,
    )
    combined = ep_style._token_combine(mlp.experts, expert_out, ep_mesh)  # pylint: disable=W0212

    use_fp32_combine = (
        getattr(mlp, "_hp_moe_tp_enabled", False)
        or getattr(mlp, "_hp_moe_ep_fp32_routing", False)
    )
    if use_fp32_combine:
        weighted = (
            combined.to(torch.float32) * sample_weights[perm].to(torch.float32).unsqueeze(-1)
        ).to(combined.dtype)
    else:
        weighted = combined * sample_weights[perm].unsqueeze(-1)
    unsorted = weighted[inv_perm]
    if use_fp32_combine:
        return unsorted.view(
            num_tokens, num_top_k, hidden_dim,
        ).sum(dim=1, dtype=torch.float32).to(x_flat.dtype)
    return unsorted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(x_flat.dtype)


def _make_shared_expert_ep_forward(
    ep_style: ExpertParallel,
    ep_mesh: DeviceMesh,
    cp_mesh: Optional[DeviceMesh] = None,
):
    """Build an EP forward for shared-expert MoE blocks."""

    def _forward(mlp, x: platform.Tensor) -> platform.Tensor:
        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        shared_out = mlp.shared_expert(x_flat)
        router_logits, top_value, top_index = mlp.gate(x_flat)
        mlp.router_logits = router_logits
        routed = _expert_parallel_routed_forward(
            mlp, x_flat, top_index, top_value, ep_style, ep_mesh,
        )
        if cp_mesh is not None:
            full_x = _redistribute_first_tensor(
                x,
                cp_mesh,
                Shard(1),
                Replicate(),
                use_local_output=True,
            )
            full_flat = full_x.reshape(-1, hidden_dim)
            shared_out = mlp.shared_expert(full_flat)
            shared_out = torch.sigmoid(mlp.shared_expert_gate(full_flat)) * shared_out
            shared_out = _cp_slice(
                shared_out.view(full_x.shape[0], full_x.shape[1], hidden_dim),
                cp_mesh,
                dim=1,
            ).reshape(-1, hidden_dim)
        else:
            shared_out = torch.sigmoid(mlp.shared_expert_gate(x_flat)) * shared_out

        out = routed + shared_out
        return out.view(bsz, seq_len, hidden_dim)

    return _forward


def _make_sparse_ep_forward(ep_style: ExpertParallel, ep_mesh: DeviceMesh):
    """Build an EP forward for Qwen3-VL sparse MoE blocks."""

    def _forward(mlp, hidden_states: platform.Tensor) -> platform.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)
        router_logits = F.linear(hidden_states_2d, mlp.gate.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(
            routing_weights, mlp.top_k, dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        next_states = _expert_parallel_routed_forward(
            mlp, hidden_states_2d, router_indices, routing_weights, ep_style, ep_mesh,
        )
        return next_states.reshape(batch_size, sequence_length, hidden_dim)

    return _forward


def _apply_expert_parallel_forward(
    mlp: nn.Module,
    ep_style: ExpertParallel,
    ep_mesh: DeviceMesh,
    cp_mesh: Optional[DeviceMesh] = None,
) -> None:
    """Install the EP MoE forward owned by the model parallel plan."""
    if hasattr(mlp, "shared_expert"):
        forward = _make_shared_expert_ep_forward(ep_style, ep_mesh, cp_mesh)
    else:
        forward = _make_sparse_ep_forward(ep_style, ep_mesh)
    mlp.forward = MethodType(forward, mlp)


def _should_skip_qwen3_5_moe_fsdp(cfg) -> bool:
    """Return whether a single-rank MoE run can keep plain parameters."""
    single_rank = _world_size() == 1
    has_parallel_axis = _has_configured_parallel_axis(
        cfg,
        ("dp_replicate", "dp_shard", "tp", "cp", "pp", "ep", "etp"),
    )
    mp_cfg = cfg.train.mixed_precision
    has_output_dtype = bool(mp_cfg.output_dtype)
    param_dtype = mp_cfg.param_dtype
    loss_aggregation = cfg.train.optimizer.loss_aggregation
    needs_rank_average_mp_wrap = (
        loss_aggregation == "rank_average"
        and mp_cfg.enabled
        and param_dtype not in ("float32", "fp32")
    )
    return single_rank and not has_parallel_axis and not has_output_dtype and not needs_rank_average_mp_wrap


def _resolve_qwen3_5_moe_fsdp_mesh(mesh, cfg, cp_size: int):
    """Resolve the FSDP mesh for a MoE model, or ``None`` when it should skip."""
    replicate_size, shard_size = _resolve_qwen3_5_moe_dp_sizes(mesh, cfg)
    if replicate_size > 1 and cp_size > 1:
        raise ValueError(
            "Qwen3.5-MoE non-PP training does not support HSDP with CP: the "
            "current fully_shard mesh cannot express both the outer "
            "dp_replicate synchronization and the CP gradient domain."
        )
    if cp_size > 1:
        try:
            return mesh["loss"]
        except (KeyError, TypeError):
            return mesh["cp"]
    other_parallel = max(
        int(cfg.train.accelerator.tp),
        int(cfg.train.accelerator.ep),
    )
    has_model_parallel = other_parallel > 1
    if replicate_size * shard_size <= 1 and has_model_parallel and not _has_shard_dim(mesh):
        return None
    return _resolve_qwen3_5_moe_dp_mesh(mesh, cfg)


def _mark_moe_tp_fsdp_enabled(model, cfg, dp_mesh) -> None:
    """Mark TP-sharded experts when FSDP also shards over a real DP mesh."""
    try:
        fsdp_world = dp_mesh.size()
    except (AttributeError, RuntimeError):
        fsdp_world = 1
    if int(cfg.train.accelerator.tp or 1) <= 1 or fsdp_world <= 1:
        return
    for block in model.layers:
        block.mlp.experts._hp_moe_tp_fsdp_enabled = True  # pylint: disable=protected-access


def _wrap_qwen3_5_moe_layers(model, layer_fsdp_kwargs) -> int:
    """FSDP-wrap decoder layers and optional MTP layer."""
    layers = list(model.layers)
    for layer in layers:
        fully_shard(layer, **layer_fsdp_kwargs)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        fully_shard(mtp.layers[0], **layer_fsdp_kwargs)
    return len(layers)


def _wrap_qwen3_5_moe_pp_boundaries(model, cfg, fsdp_kwargs) -> None:
    """Wrap PP boundary modules so stage forwards trigger FSDP all-gather."""
    if int(cfg.train.accelerator.pp) <= 1:
        return
    backbone = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for sub in (
        getattr(backbone, "embed_tokens", None),
        getattr(backbone, "norm", None),
        getattr(model, "lm_head", None),
    ):
        if sub is not None:
            fully_shard(sub, **fsdp_kwargs)


def _set_qwen3_5_moe_reduce_policy(model, cfg, cp_size: int) -> None:
    """Set the FSDP reduce policy required by CP/EP loss scaling."""
    if cp_size > 1:
        model.set_reduce_op_type("sum")
        model.hp_token_loss_scale_size = 1
        logger.info_rank0("CP FSDP reduce op set to SUM for Qwen3.5-MoE token-weighted loss.")
    elif int(cfg.train.accelerator.ep or 1) > 1:
        model.set_reduce_op_type("avg")
        logger.info_rank0("EP FSDP reduce op set to AVG for Qwen3.5-MoE token-weighted loss.")


def _apply_fsdp(model, mesh, cfg) -> None:
    """Per-layer + root FSDP wrap for real parallel axes.

    Wraps each Qwen3.5-MoE decoder layer (attention + MoE expert block) as
    its own FSDP unit so per-layer all-gather scheduling stays stable. Pure
    single-rank runs with no parallel axis skip FSDP so the baseline uses the
    same plain ``nn.Parameter`` AdamW path. If the YAML
    explicitly requests an FSDP output dtype, keep the size-1 FSDP wrap because
    that policy has no non-FSDP equivalent.
    """
    if _should_skip_qwen3_5_moe_fsdp(cfg):
        logger.info_rank0(
            "Single-rank Qwen3.5-MoE run has no parallel axes; skipping FSDP wrap. "
            "Post-load param_dtype casting handles mixed-precision storage."
        )
        return

    cp_size = int(cfg.train.accelerator.cp or 1)
    dp_mesh = _resolve_qwen3_5_moe_fsdp_mesh(mesh, cfg, cp_size)
    if dp_mesh is None:
        logger.info_rank0("dp_size==1 with tp/ep>1; skipping FSDP wrap.")
        return
    fsdp_kwargs = _build_fsdp_kwargs(model, dp_mesh, cfg)
    layer_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    _mark_moe_tp_fsdp_enabled(model, cfg, dp_mesh)

    if not hasattr(model, "layers"):
        logger.warning(
            "Qwen3_5MoeForCausalLM has no ``.layers`` — root-only FSDP wrap. "
            "This is usually wrong for transformer-style models."
        )
        fully_shard(model, **fsdp_kwargs)
        return

    num_layers = _wrap_qwen3_5_moe_layers(model, layer_fsdp_kwargs)
    _wrap_qwen3_5_moe_pp_boundaries(model, cfg, fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    _set_qwen3_5_moe_reduce_policy(model, cfg, cp_size)
    logger.info_rank0(
        "FSDP applied to Qwen3.5-MoE: %d layers + root  replicate=%d",
        num_layers, len(fsdp_kwargs.get("replicate_params", ())),
    )


def qwen3_5_moe_tp_load_transforms(
    model: Qwen3_5MoeForCausalLM,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> dict:
    """``ModelSpec.tp_load_transform_fn`` — slice TP-local MoE / GatedDeltaNet weights."""
    tp_size = int(cfg.train.accelerator.tp or 1)
    if tp_size <= 1:
        return {}
    tp_mesh = mesh["tp"]
    is_vl = hasattr(model.model, "language_model")
    layer_prefix = "model.language_model.layers" if is_vl else "model.layers"
    transforms: dict = {}
    for idx, block in enumerate(model.layers):
        experts = block.mlp.experts
        intermediate_size = getattr(experts, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(experts, "intermediate_dim", None)
        if intermediate_size is None:
            raise ValueError(
                "Packed MoE expert params must expose intermediate_size or "
                "intermediate_dim for TP load transforms."
            )
        local_intermediate = intermediate_size // tp_size
        layout = _moe_expert_tp_layout(experts, intermediate_size, local_intermediate)
        gate_up_dim = 1 if layout == "gate_up_out_in" else 2
        down_dim = 2 if layout == "gate_up_out_in" else 1
        if experts.gate_up_proj.shape[gate_up_dim] == 2 * intermediate_size:
            continue
        if experts.down_proj.shape[down_dim] != local_intermediate:
            raise ValueError(
                "Qwen3.5-MoE TP load transform found unexpected expert "
                f"shape at layer {idx}: gate_up={tuple(experts.gate_up_proj.shape)}, "
                f"down={tuple(experts.down_proj.shape)}."
            )
        gate_slice, up_slice, down_slice = _moe_expert_tp_slices(experts, tp_mesh)
        prefix = f"{layer_prefix}.{idx}.mlp.experts"
        if layout == "gate_up_out_in":
            transforms[f"{prefix}.gate_up_proj"] = (
                lambda w, g=gate_slice, u=up_slice: torch.cat(
                    [w[:, g, :], w[:, u, :]], dim=1,
                ).contiguous()
            )
            transforms[f"{prefix}.down_proj"] = (
                lambda w, d=down_slice: w[:, :, d].contiguous()
            )
        else:
            transforms[f"{prefix}.gate_up_proj"] = (
                lambda w, g=gate_slice, u=up_slice: torch.cat(
                    [w[:, :, g], w[:, :, u]], dim=2,
                ).contiguous()
            )
            transforms[f"{prefix}.down_proj"] = (
                lambda w, d=down_slice: w[:, d, :].contiguous()
            )
        if block.layer_type == "full_attention":
            continue
        linear_attn = block.linear_attn
        full_conv_dim = (
            2 * linear_attn.key_dim
            + linear_attn.num_v_heads * linear_attn.head_v_dim
        )
        if linear_attn.conv1d.weight.shape[0] == full_conv_dim:
            continue
        q_slice, k_slice, v_slice, head_start, head_end = _gated_delta_tp_slices(
            linear_attn, tp_mesh,
        )
        linear_prefix = f"{layer_prefix}.{idx}.linear_attn"
        transforms[f"{linear_prefix}.in_proj_qkv.weight"] = (
            lambda w, q=q_slice, k=k_slice, v=v_slice: torch.cat(
                [w[q], w[k], w[v]], dim=0,
            ).contiguous()
        )
        transforms[f"{linear_prefix}.conv1d.weight"] = (
            lambda w, q=q_slice, k=k_slice, v=v_slice: torch.cat(
                [w[q], w[k], w[v]], dim=0,
            ).contiguous()
        )
        transforms[f"{linear_prefix}.dt_bias"] = lambda w, s=head_start, e=head_end: w[s:e].clone()
        transforms[f"{linear_prefix}.A_log"] = lambda w, s=head_start, e=head_end: w[s:e].clone()
    return transforms


def _validate_qwen3_5_moe_tp_config(model: Qwen3_5MoeForCausalLM, tp_world: int) -> None:
    """Validate TP divisibility for full-attention MoE layers."""
    cfg = _model_text_config(model)
    layer_types = getattr(cfg, "layer_types", None)
    has_full_attention = (
        layer_types is None
        or any(layer_type == "full_attention" for layer_type in layer_types)
    )
    if not has_full_attention:
        return
    if cfg.num_attention_heads % tp_world != 0:
        raise ValueError(
            f"num_attention_heads ({cfg.num_attention_heads}) must divide TP size ({tp_world})."
        )
    if cfg.num_key_value_heads % tp_world != 0:
        raise ValueError(
            f"num_key_value_heads ({cfg.num_key_value_heads}) must divide TP size ({tp_world})."
        )


def _build_qwen3_5_moe_tp_plans(
    backbone: str,
    sp_layout,
    enable_loss_parallel: bool,
    replicate_out_proj: bool,
    replicate_attention_out_proj: bool = False,
):
    """Build reusable TP plans for Qwen3.5-MoE blocks."""
    rowwise_reduce_output_plan = RowwiseParallel(
        input_layouts=Shard(-1),
        output_layouts=sp_layout,
        reduce_dtype=torch.float32,
        use_local_output=False,
    )
    norm_plan = SequenceParallel(sequence_dim=1, use_local_output=False)
    sp_to_replicate = PrepareModuleInput(
        input_layouts=(sp_layout,),
        desired_input_layouts=(Replicate(),),
        use_local_output=True,
    )
    colwise = ColwiseParallel()
    moe_boundary_plan = PrepareModuleInputOutput(
        input_layouts=(sp_layout,),
        desired_input_layouts=(Replicate(),),
        use_local_input=True,
        output_layouts=(Partial(),),
        desired_output_layouts=(sp_layout,),
        reduce_dtype=torch.float32,
        use_local_output=False,
    )
    shared_expert_rowwise_plan = RowwiseParallel(
        input_layouts=Shard(-1),
        output_layouts=Partial(),
        reduce_dtype=torch.float32,
        use_local_output=True,
    )
    norm_and_mlp_plan = {
        "input_layernorm": norm_plan,
        "post_attention_layernorm": norm_plan,
        "mlp": moe_boundary_plan,
        "mlp.shared_expert.gate_proj": colwise,
        "mlp.shared_expert.up_proj": colwise,
        "mlp.shared_expert.down_proj": shared_expert_rowwise_plan,
    }
    linear_layer_plan = {
        **norm_and_mlp_plan,
        "linear_attn": sp_to_replicate,
        "linear_attn.in_proj_z": colwise,
        "linear_attn.in_proj_b": colwise,
        "linear_attn.in_proj_a": colwise,
    }
    if not replicate_out_proj:
        linear_layer_plan["linear_attn.out_proj"] = rowwise_reduce_output_plan
    full_layer_plan = {
        **norm_and_mlp_plan,
        "self_attn": sp_to_replicate,
        "self_attn.q_proj": colwise,
        "self_attn.k_proj": colwise,
        "self_attn.v_proj": colwise,
    }
    if not replicate_attention_out_proj:
        full_layer_plan["self_attn.o_proj"] = rowwise_reduce_output_plan
    return SimpleNamespace(
        root={
            f"{backbone}.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=sp_layout,
                use_local_output=False,
            ),
            f"{backbone}.norm": norm_plan,
            "lm_head": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
        full_layer=full_layer_plan,
        linear_layer=linear_layer_plan,
    )


def _apply_qwen3_5_moe_tp_layer(
    block: nn.Module,
    tp_mesh: DeviceMesh,
    plans,
    replicate_out_proj: bool,
    replicate_attention_out_proj: bool = False,
) -> None:
    """Apply TP plan and local expert sharding for one MoE decoder block."""
    if block.layer_type == "full_attention":
        parallelize_module(block, tp_mesh, plans.full_layer)
        if replicate_attention_out_proj:
            _apply_replicated_attention_out_proj_tp(block.self_attn.o_proj, tp_mesh)
    else:
        parallelize_module(block, tp_mesh, plans.linear_layer)
        _shard_gated_delta_local_params(block.linear_attn, tp_mesh)
        if replicate_out_proj:
            _apply_replicated_linear_out_proj_tp(block.linear_attn.out_proj, tp_mesh)
    _shard_moe_expert_tp_params(block.mlp.experts, tp_mesh)
    _apply_shared_expert_tp_forward(block.mlp)


def _apply_tp_checkpoint_local_sequence_shard(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Keep TP checkpoint-wrapper inputs as local sequence shards."""

    def _target_seq_len(kwargs) -> Optional[int]:
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            return int(position_ids.shape[-1])
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask.shape[-1])
        return None

    def _to_local_sequence_shard(hidden_states, target_seq_len: Optional[int]):
        if isinstance(hidden_states, DTensor):
            return hidden_states.redistribute(tp_mesh, [Shard(1)]).to_local()
        if target_seq_len is None or not platform.is_tensor(hidden_states):
            return hidden_states
        seq_len = hidden_states.shape[1]
        if seq_len == target_seq_len:
            return _cp_slice(hidden_states, tp_mesh, dim=1)
        return hidden_states

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        args = list(args)
        kwargs = dict(kwargs)
        target_seq_len = _target_seq_len(kwargs)
        if args:
            args[0] = _to_local_sequence_shard(args[0], target_seq_len)
        elif kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = _to_local_sequence_shard(
                kwargs["hidden_states"],
                target_seq_len,
            )
        return tuple(args), kwargs

    for block in model.layers:
        if getattr(block, "_is_wrapped", False):
            block.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def parallelize_qwen3_5_moe_tp(
    model: Qwen3_5MoeForCausalLM,
    tp_mesh: DeviceMesh,
    *,
    enable_loss_parallel: bool = False,
    register_grad_hooks: bool = True,
    replicate_linear_out_proj: bool = False,
    replicate_attention_out_proj: bool = False,
) -> Qwen3_5MoeForCausalLM:
    """Apply 1-D SequenceParallel tensor parallelism to a Qwen3.5-MoE backbone.

    Uses the dense Qwen3.5 attention plan for
    :class:`Qwen3_5Attention` / :class:`Qwen3_5GatedDeltaNet`, and handles the
    MoE ``mlp`` (:class:`Qwen3_5SharedExpertMoE`) by gathering the
    SequenceParallel token stream to Replicate at the MoE boundary, running
    router/gate on the full token set, sharding the routed and shared experts
    over the TP intermediate dimension, then reduce-scattering the local hidden
    partial back to ``Shard(1)``.

    Requires full-attention ``num_attention_heads`` / ``num_key_value_heads``
    divisible by ``tp_size`` and ``seq_len % tp_size == 0`` for the sequence shard.
    """
    tp_world = tp_mesh.size()
    cfg = _model_text_config(model)
    _validate_qwen3_5_moe_tp_config(model, tp_world)

    tie_embeddings = bool(cfg.tie_word_embeddings)
    sp_layout = Shard(1)
    is_vl = hasattr(model.model, "language_model")
    backbone = "model.language_model" if is_vl else "model"
    plans = _build_qwen3_5_moe_tp_plans(
        backbone,
        sp_layout,
        enable_loss_parallel,
        replicate_linear_out_proj,
        replicate_attention_out_proj,
    )
    parallelize_module(model, tp_mesh, plans.root)
    if tie_embeddings:
        model.tie_weights()

    for block in model.layers:
        _apply_qwen3_5_moe_tp_layer(
            block,
            tp_mesh,
            plans,
            replicate_linear_out_proj,
            replicate_attention_out_proj,
        )

    model._hp_moe_experts_tp_sharded = True  # pylint: disable=protected-access
    model._hp_moe_tp_gate_grad_avg = True  # pylint: disable=protected-access
    if register_grad_hooks:
        _register_tp_replicated_param_grad_sum(model, tp_mesh)
    model.hp_loss_tp_scale_size = tp_world
    logger.info_rank0(
        "TP applied to Qwen3.5-MoE: SequenceParallel plan, tp_size=%d, layers=%d "
        "(routed/shared experts intermediate-sharded over TP)",
        tp_world,
        len(model.layers),
    )
    return model


def parallelize_qwen3_5_moe_cp(
    model: Qwen3_5MoeForCausalLM,
    cp_mesh: DeviceMesh,
    *,
    ulysses_degree: Optional[int] = None,
) -> Qwen3_5MoeForCausalLM:
    """Apply context parallelism across the Qwen3.5-MoE hybrid decoder.

    Uses the dense attention CP strategy: full-attention layers hook their SDPA
    core with
    :class:`ContextParallel` (pure Ulysses by default — the safe choice that
    avoids the right-aligned causal-mask leak of the ``ulysses_degree=1``
    variant); linear-attention layers gather the full sequence and slice the
    output through model-level hooks around the module body. The MoE routed
    experts stay local per token (HF grouped-mm order), while the shared expert
    is computed on the full CP sequence and sliced back so its dense matmul
    backward follows the single-card row grouping more closely.

    Unlike the TP ``SequenceParallel`` plan, CP consumes the trainer's
    pre-sharded sequence directly with explicit global ``position_ids`` and
    trainer-owned pre-shifted loss targets. Replicated-weight gradients are
    reduced across the CP ranks by FSDP over the ``"loss"`` mesh (see
    :func:`_apply_fsdp`), so no SequenceParallel grad hook is registered here.
    """
    # Only pure Ulysses (``ulysses_degree`` None or == cp_size) is wired here.
    # A smaller ``ulysses_degree`` makes each rank attend over gathered K/V with
    # ``is_causal=True`` but without a per-rank causal mask — leaking future
    # tokens to every CP rank but the last. Reject it loudly until the per-rank
    # mask is wired, rather than silently leaking.
    if ulysses_degree is not None and ulysses_degree < cp_mesh.size():
        raise NotImplementedError(
            f"Qwen3.5-MoE CP: ulysses_degree={ulysses_degree} < cp={cp_mesh.size()} "
            "non-pure Ulysses CP is not supported — it needs a per-rank causal mask "
            "that is not yet wired. Use pure Ulysses (omit ulysses_degree or set it "
            "equal to cp)."
        )
    cp_plan = ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=ulysses_degree)
    expand_kv_for_cp = _needs_gqa_kv_expand_for_cp(model, cp_mesh)
    full_attached = 0
    linear_attached = 0
    for block in model.layers:
        if block.layer_type == "full_attention":
            block.self_attn._hp_cp_expand_kv_before_core = expand_kv_for_cp  # pylint: disable=protected-access
            cp_plan.apply(block.self_attn.sdpa_core, cp_mesh)
            full_attached += 1
        else:
            _apply_linear_attention_cp(block.linear_attn, cp_mesh)
            _apply_shared_expert_cp_full_sequence(block.mlp, cp_mesh)
            linear_attached += 1
    logger.info_rank0(
        "CP applied to Qwen3.5-MoE: cp_size=%d, ulysses_degree=%s, full-attn hooks=%d, "
        "linear-attn gather/slice=%d",
        cp_mesh.size(), ulysses_degree, full_attached, linear_attached,
    )
    return model


def _needs_gqa_kv_expand_for_cp(model, cp_mesh: DeviceMesh) -> bool:
    """Return whether GQA K/V heads must be expanded before Ulysses CP."""
    tp_size = int(getattr(model, "hp_loss_tp_scale_size", 1) or 1)
    if tp_size <= 1:
        return False
    kv_heads = int(_model_text_config(model).num_key_value_heads or 1)
    local_kv_heads = max(kv_heads // tp_size, 1)
    return local_kv_heads % cp_mesh.size() != 0


def parallelize_qwen3_5_moe_ep(
    model: Qwen3_5MoeForCausalLM,
    ep_mesh: DeviceMesh,
    *,
    cp_mesh: Optional[DeviceMesh] = None,
    register_grad_hooks: bool = True,
    set_loss_scale: bool = True,
    stable_sort: bool = False,
    fp32_routing: bool = False,
) -> Qwen3_5MoeForCausalLM:
    """Apply expert parallelism to Qwen3.5-MoE given the EP sub-mesh.

    Each decoder block's ``mlp.experts`` packed weights are sharded ``Shard(0)``
    (expert dim) so each rank owns ``num_experts // ep_size`` experts; the block
    is handed the EP mesh + a :class:`ExpertParallel` style whose tested
    all-to-all dispatch/combine wraps its EP forward around the local grouped
    matmul. EP is a model-parallel axis, so every rank computes a replicated
    loss while routed experts are sharded. The full trainer path installs EP
    loss scaling and replicated-gradient SUM only for pure EP / EP+CP; when TP
    is also present, the TP loss scale is already the required model-parallel
    normalizer for the routed expert path.

    Args:
        model: Fully constructed :class:`Qwen3_5MoeForCausalLM`.
        ep_mesh: 1-D EP sub-mesh (``mesh["ep"]``).
        register_grad_hooks: Whether to attach the pure-EP replicated-gradient
            SUM hooks immediately. The trainer's full parallelize path disables
            this before FSDP wrapping and installs the appropriate reducer after
            FSDP has finalized parameters.
        set_loss_scale: Whether to set the trainer EP loss-scale marker.
        stable_sort: Whether to keep equal-expert routed rows in token/top-k
            order before EP dispatch.
        fp32_routing: Whether to multiply routed expert outputs by routing
            weights in fp32 before casting back to the expert output dtype.

    Returns:
        The same ``model`` with experts sharded and EP dispatch wired.

    Raises:
        ValueError: When ``num_experts`` is not divisible by ``ep_mesh.size()``.
    """
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        return model
    text_config = _model_text_config(model)
    if text_config.num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({text_config.num_experts}) must divide EP size ({ep_size})."
        )
    blocks = list(model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        blocks.append(mtp.layers[0])
    for block in blocks:
        mlp = block.mlp
        mlp._hp_moe_ep_stable_sort = stable_sort  # pylint: disable=protected-access
        mlp._hp_moe_ep_fp32_routing = fp32_routing  # pylint: disable=protected-access
        # One ExpertParallel style PER block: its ``_dispatch_ctx`` is per-dispatch
        # mutable state, so a single shared instance across all layers would let
        # one layer's dispatch context be overwritten by the next layer before its
        # combine runs (e.g. under activation checkpointing / non-strict pairing).
        ep_style = ExpertParallel()
        # Shard the packed expert weights (gate_up_proj / down_proj) on dim 0.
        ep_style._partition_fn(None, mlp.experts, ep_mesh)  # pylint: disable=W0212
        _apply_expert_parallel_forward(mlp, ep_style, ep_mesh, cp_mesh)
    logger.info_rank0(
        "EP applied to Qwen3.5-MoE: ep_size=%d, experts/rank=%d, layers=%d",
        ep_size, text_config.num_experts // ep_size, len(model.layers),
    )
    if set_loss_scale:
        model.hp_loss_ep_scale_size = ep_size
    if register_grad_hooks:
        _register_ep_replicated_param_grad_sum(model, ep_mesh)
    return model


def _apply_ep(model, mesh, cfg) -> None:
    """Resolve the ``ep`` sub-mesh from the run config and delegate to
    :func:`parallelize_qwen3_5_moe_ep` (the canonical EP entry shared by direct
    callers).
    """
    ep_size = int(cfg.train.accelerator.ep)
    if ep_size <= 1:
        return
    tp_size = int(cfg.train.accelerator.tp or 1)
    cp_size = int(cfg.train.accelerator.cp or 1)
    is_vl = hasattr(model.model, "language_model")
    try:
        ep_mesh = mesh["ep"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parallel.ep>1 requires a DeviceMesh with an 'ep' dim.") from exc
    cp_mesh = None
    if cp_size > 1:
        try:
            cp_mesh = mesh["cp"]
        except (KeyError, TypeError) as exc:
            raise ValueError("parallel.cp>1 requires a DeviceMesh with a 'cp' dim.") from exc
    parallelize_qwen3_5_moe_ep(
        model,
        ep_mesh,
        cp_mesh=cp_mesh,
        register_grad_hooks=False,
        set_loss_scale=False,
        stable_sort=is_vl and tp_size <= 1,
        fp32_routing=is_vl,
    )


def _validate_qwen3_5_moe_parallel_combo(
    model,
    tp_size: int,
    cp_size: int,
    pp_size: int,
    ep_size: int,
) -> None:
    """Validate unsupported TP/CP/PP combinations for Qwen3.5-MoE."""
    text_config = _model_text_config(model)
    mtp_loss_weight = text_config.mtp_loss_weight
    if mtp_loss_weight > 0 and (tp_size > 1 or cp_size > 1 or pp_size > 1):
        raise NotImplementedError(
            "Qwen3.5-MoE MTP currently supports only FSDP and EP; "
            "set tp=1, cp=1, and pp=1 when mtp_loss_weight > 0."
        )
    if text_config.output_router_logits and (
        tp_size > 1 or cp_size > 1 or pp_size > 1
    ):
        raise NotImplementedError(
            "MoE router aux loss (output_router_logits=True) currently supports only "
            "FSDP / EP / single-card; disable output_router_logits or set tp=1, cp=1, pp=1."
        )
    if pp_size > 1 and ep_size > 1:
        raise NotImplementedError(
            "Qwen3.5-MoE PP with EP is not supported: the pipeline stage does "
            "not yet install the parameter-class-aware EP gradient reducers. "
            "Set pp=1 or ep=1."
        )
    if tp_size > 1 and cp_size > 1:
        heads = int(text_config.num_attention_heads)
        if heads and (heads // tp_size) % cp_size != 0:
            raise ValueError(
                f"TP+CP requires (num_attention_heads / tp) divisible by cp "
                f"for the Ulysses all-to-all; got {heads} heads / tp={tp_size}"
                f" = {heads // tp_size}, cp={cp_size}."
            )


def _apply_qwen3_5_moe_tp(model, mesh, cfg, is_vl: bool, ep_size: int) -> None:
    """Apply Qwen3.5-MoE tensor parallelism when requested."""
    tp_size = int(cfg.train.accelerator.tp)
    if tp_size <= 1:
        return
    try:
        tp_mesh = mesh["tp"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parallel.tp>1 requires a DeviceMesh with a 'tp' dim.") from exc
    data_cfg = cfg.data
    deterministic = bool(cfg.train.debug.deterministic)
    replicate_linear_out_proj = bool(
        deterministic and is_vl and (data_cfg.vl_video or ep_size <= 1)
    )
    parallelize_qwen3_5_moe_tp(
        model,
        tp_mesh,
        register_grad_hooks=False,
        replicate_linear_out_proj=replicate_linear_out_proj,
    )
    if is_vl:
        _apply_visual_sequence_roundtrip(model.model)


def _apply_qwen3_5_moe_cp(model, mesh, cfg, is_vl: bool, cp_size: int) -> None:
    """Apply Qwen3.5-MoE context parallelism when requested."""
    if cp_size <= 1:
        return
    try:
        cp_mesh = mesh["cp"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parallel.cp>1 requires a DeviceMesh with a 'cp' dim.") from exc
    ulysses_degree = cfg.train.accelerator.ulysses_degree
    parallelize_qwen3_5_moe_cp(model, cp_mesh, ulysses_degree=ulysses_degree)
    if is_vl:
        _apply_full_sequence_input_cp_gather(model.model, cp_mesh)
        _apply_text_input_cp_slice(model.model.language_model, cp_mesh)


def _apply_tp_cp_ep(model, mesh, cfg) -> None:
    """Apply in-layer TP (SequenceParallel) / CP (Ulysses) and expert parallelism.

    TP shards the token (sequence) dimension via :class:`SequenceParallel` (the
    plan gathers the full sequence for the SSM / attention and re-shards, so it
    needs no trainer-side input sharding). CP instead consumes the trainer's
    pre-sharded sequence with :class:`ContextParallel` (Ulysses on full attention,
    sequence-gather on the linear-attention SSM), mirroring the dense model.
    Combined TP+CP nests the same way as dense: every TP rank of a CP group
    receives the same trainer-presharded CP slice, TP's SequenceParallel then
    shards within that slice, and attention gathers the TP shard back to the CP
    slice before the Ulysses all-to-all. EP then shards each layer's experts
    across the ``ep`` dim. Shared by the non-PP ``parallelize_fn`` and the PP
    pipelining adapter so both compose the in-layer parallelism identically
    before FSDP / the pipeline split.
    """
    is_vl = hasattr(model.model, "language_model")
    tp_size = int(cfg.train.accelerator.tp)
    cp_size = int(cfg.train.accelerator.cp)
    ep_size = int(cfg.train.accelerator.ep or 1)
    pp_size = int(cfg.train.accelerator.pp)
    _validate_qwen3_5_moe_parallel_combo(
        model,
        tp_size,
        cp_size,
        pp_size,
        ep_size,
    )
    _apply_qwen3_5_moe_tp(model, mesh, cfg, is_vl, ep_size)
    _apply_qwen3_5_moe_cp(model, mesh, cfg, is_vl, cp_size)
    _apply_ep(model, mesh, cfg)


def parallelize_qwen3_5_moe(
    model: Qwen3_5MoeForCausalLM,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> Qwen3_5MoeForCausalLM:
    """Apply TP / EP / CP / AC / FSDP to a Qwen3.5-MoE model.

    Order: TP first (per-layer weights sharded before AC/FSDP wrap), then EP
    (experts sharded across the ep dim + token dispatch/combine), AC next, FSDP
    last so wrap units close over the already-parallelized modules.

    Args:
        model: Qwen3.5-MoE model to parallelize.
        mesh: Device mesh containing the configured parallel dimensions.
        cfg: Trainer configuration controlling parallel degrees and policies.

    Returns:
        The parallelized model.
    """
    _resolve_qwen3_5_moe_fsdp_mesh(
        mesh, cfg, int(cfg.train.accelerator.cp or 1),
    )
    _apply_vl_visual_tower(model, mesh, cfg)
    _apply_tp_cp_ep(model, mesh, cfg)
    # PP is driven by the trainer's pipeline path via ``pipelining_fn`` (it calls
    # this ``parallelize_fn`` only for the FSDP/TP/EP/CP part when composing PP
    # with those), so pp>1 is handled there, not rejected here.
    _apply_ac(model, cfg)
    tp_size = int(cfg.train.accelerator.tp or 1)
    if tp_size > 1:
        _apply_tp_checkpoint_local_sequence_shard(model, mesh["tp"])
    _apply_fsdp(model, mesh, cfg)
    # Under TP+FSDP the per-param SequenceParallel grad hook cannot fire (FSDP
    # owns the grad reduction), so the trainer drives an explicit post-FSDP TP
    # all-reduce for the plain TP-replicated weights. Needed whenever FSDP
    # actually wrapped the model alongside TP — including FSDP over the
    # ``loss`` (dp_shard × cp) mesh under CP and the size-1 dp_shard wrap a
    # low-precision run injects; pure TP (no FSDP wrap) is covered by the
    # in-graph hook.
    ep_size = int(cfg.train.accelerator.ep or 1)
    cp_size = int(cfg.train.accelerator.cp or 1)
    reducers = []
    if tp_size > 1:
        if isinstance(model, HSDPModule):
            reducers.append(_make_post_fsdp_tp_reducer(model, mesh["tp"]))
        else:
            _register_tp_replicated_param_grad_sum(model, mesh["tp"])
    if ep_size > 1:
        if tp_size > 1:
            model.hp_loss_ep_scale_size = 1
            if isinstance(model, HSDPModule):
                reducers.append(_make_post_fsdp_ep_avg_reducer(model, mesh["ep"]))
                reducers.append(_make_post_fsdp_ep_expert_grad_divider(model, mesh["ep"]))
        elif cp_size <= 1:
            model.hp_loss_ep_scale_size = 1
            if isinstance(model, HSDPModule):
                reducers.append(_make_post_fsdp_ep_moe_avg_reducer(model, mesh["ep"]))
            else:
                _register_ep_replicated_moe_grad_avg(model, mesh["ep"])
        else:
            model.hp_loss_ep_scale_size = 1
            if isinstance(model, HSDPModule):
                reducers.append(_make_post_fsdp_ep_moe_avg_reducer(model, mesh["ep"]))
            else:
                _register_ep_replicated_moe_grad_avg(model, mesh["ep"])
    post_fsdp_grad_reduce = _chain_grad_reducers(*reducers)
    if post_fsdp_grad_reduce is not None:
        model.hp_post_fsdp_grad_reduce = post_fsdp_grad_reduce
    return model


def _fsdp_wrap_stage(stage_module, dp_mesh, cfg) -> None:
    """Wrap a pipeline stage's children as FSDP units + a root coordinator.

    Each decoder layer plus the embed / norm / lm_head this stage owns becomes
    its own ``HSDPModule``, and the stage module itself is then wrapped as the
    *root* FSDP unit. The root wrap is required, not cosmetic: the schedule's
    ``FSDP_REDUCE_GRAD`` finalization (``_root_backward_hook``) drains the
    process-wide fused-reduction state once per step, and the per-unit reductions
    are only fully applied through that single root drain. Without a root,
    independently-wrapped boundary units (the vocab-sized embed / lm_head) have
    their reduce-scatter left pending and their ``sharded_param.grad`` is applied
    only every other step — silently mis-training those params. This matches the
    non-PP ``_apply_fsdp`` shape: per-layer units plus one root unit.

    Wrapping runs while the model is still on meta (the trainer's PP+FSDP path
    materializes + loads the shards afterwards), so ``fully_shard`` builds
    correctly-sized meta shards instead of tripping the sharded-size check.
    """
    fsdp_kwargs = _build_fsdp_kwargs(stage_module, dp_mesh, cfg)
    # Under the 1F1B schedule, unshard / reshard is driven explicitly by the
    # schedule (FSDP_UNSHARD / FSDP_RESHARD steps), not by the per-forward hook:
    # params must stay unsharded across all micro-batches of a step. A
    # ``reshard_after_forward=True`` hook would reshard after micro 0's forward,
    # leaving micro 1's forward with sharded params (tripping the schedule's
    # in-unshard assertion). Force it off so the schedule owns resharding.
    fsdp_kwargs["reshard_after_forward"] = False
    layer_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    boundary_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    root_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    for layer in stage_module.layers:
        fully_shard(layer, **layer_fsdp_kwargs)
    for sub in (stage_module.embed_tokens, stage_module.norm, stage_module.lm_head):
        if sub is not None:
            fully_shard(sub, **boundary_fsdp_kwargs)
    fully_shard(stage_module, **root_fsdp_kwargs)
    logger.info_rank0(
        "PP+FSDP: wrapped %d layer(s) + head/embed + root as stage FSDP units (dp=%d)",
        len(stage_module.layers), dp_mesh.size() if hasattr(dp_mesh, "size") else 1,
    )


def _fsdp_wrap_stage_vl(stage_module, dp_mesh, cfg) -> None:
    """Wrap a VL pipeline stage's text children and root as FSDP units."""
    fsdp_kwargs = _build_fsdp_kwargs(stage_module, dp_mesh, cfg)
    fsdp_kwargs["reshard_after_forward"] = False
    layer_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    boundary_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    root_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    for layer in stage_module.layers:
        fully_shard(layer, **layer_fsdp_kwargs)
    for sub in (stage_module.embed_tokens, stage_module.norm, stage_module.lm_head):
        if sub is not None:
            fully_shard(sub, **boundary_fsdp_kwargs)
    fully_shard(stage_module, **root_fsdp_kwargs)
    logger.info_rank0(
        "PP+FSDP: Qwen3.5-MoE VL stage wrapped %d layer(s) + "
        "embed/norm/lm_head + root (dp=%d)",
        len(stage_module.layers), dp_mesh.size() if hasattr(dp_mesh, "size") else 1,
    )


def _load_qwen3_5_moe_vl_stage_cls():
    """Load the VL stage class lazily to avoid circular model imports."""
    from hyper_parallel.models.qwen3_5_moe.model_vl import (  # pylint: disable=C0415
        Qwen3_5MoeVLStageModule,
    )

    return Qwen3_5MoeVLStageModule


def _build_qwen3_5_moe_pp_stage_module(
    model,
    backbone,
    layers,
    stage_index: int,
    num_stages: int,
    start: int,
    end: int,
    is_vl: bool,
    vl_stage_cls,
):
    """Build one Qwen3.5-MoE PP stage module."""
    is_first = stage_index == 0
    is_last = stage_index == num_stages - 1
    if is_vl:
        return vl_stage_cls(
            layers=layers[start:end],
            config=model.config if is_first else None,
            visual=model.model.visual if is_first else None,
            embed_tokens=backbone.embed_tokens if is_first else None,
            norm=backbone.norm if is_last else None,
            lm_head=model.lm_head if is_last else None,
        )
    return Qwen3_5MoeStageModule(
        layers=layers[start:end],
        embed_tokens=backbone.embed_tokens if is_first else None,
        norm=backbone.norm if is_last else None,
        lm_head=model.lm_head if is_last else None,
    )


def _wrap_qwen3_5_moe_pp_stage(stage_module, is_vl: bool, fsdp_mesh, cfg) -> None:
    """Apply per-stage FSDP wrapping when PP composes with FSDP."""
    if fsdp_mesh is None:
        return
    if is_vl:
        _fsdp_wrap_stage_vl(stage_module, fsdp_mesh, cfg)
    else:
        _fsdp_wrap_stage(stage_module, fsdp_mesh, cfg)


def _apply_qwen3_5_moe_pp_stage_tp(stage_module, model, tp_mesh, fsdp_mesh) -> None:
    """Attach TP grad handling to a local PP stage."""
    if tp_mesh is None or tp_mesh.size() <= 1:
        return
    stage_module._hp_moe_experts_tp_sharded = getattr(  # pylint: disable=protected-access
        model,
        "_hp_moe_experts_tp_sharded",
        True,
    )
    stage_module.hp_loss_tp_scale_size = tp_mesh.size()
    if fsdp_mesh is None:
        _register_tp_replicated_param_grad_sum(stage_module, tp_mesh, eager=True)
    else:
        stage_module.hp_post_fsdp_grad_reduce = _make_post_fsdp_tp_reducer(
            stage_module,
            tp_mesh,
        )


def _build_qwen3_5_moe_pp_schedule(stages, micro_batch_num: int, schedule_name: str, vpp: int, is_vl: bool):
    """Build a Qwen3.5-MoE PP schedule with the right batch-dim schema."""
    if is_vl:
        kwargs_batch_dim = {
            "targets": BatchDimSpec(0),
            "attention_mask": BatchDimSpec(0),
            "pixel_values": BatchDimSpec(0),
            "image_grid_thw": BatchDimSpec(0),
            "pixel_values_videos": BatchDimSpec(0),
            "video_grid_thw": BatchDimSpec(0),
            "mm_token_type_ids": BatchDimSpec(0),
        }
    else:
        kwargs_batch_dim = {
            "targets": BatchDimSpec(0),
            "position_ids": BatchDimSpec(0),
            "attention_mask": BatchDimSpec(0),
        }
    if vpp > 1:
        return ScheduleInterleaved1F1B(
            stages,
            micro_batch_num,
            kwargs_batch_dim=kwargs_batch_dim,
        )
    return _resolve_pp_schedule(schedule_name)(
        stages,
        micro_batch_num,
        kwargs_batch_dim=kwargs_batch_dim,
    )


def pipelining_qwen3_5_moe(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    *,
    device: Optional[torch.device] = None,
    micro_batch_num: int = 1,
    fsdp_mesh: Optional[DeviceMesh] = None,
    cfg: Optional["HyperTrainerConfig"] = None,
    schedule_name: str = "1f1b",
    vpp: int = 1,
    tp_mesh: Optional[DeviceMesh] = None,
) -> tuple[object, list[PipelineStage]]:
    """Split a Qwen3.5-MoE or VL-MoE backbone into local PP stages."""
    num_ranks = pp_mesh.size()
    if num_ranks < 2:
        raise ValueError(f"pipelining_qwen3_5_moe requires pp>=2; got {num_ranks}.")
    vpp = max(1, int(vpp))
    num_stages = num_ranks * vpp  # total interleaved global stages (P * V)
    is_vl = hasattr(model.model, "language_model")
    backbone = model.model.language_model if is_vl else model.model
    layers = list(model.layers)
    n_layers = len(layers)

    # Uneven contiguous slabs over all ``num_stages`` global stages (a model
    # with fewer layers than stages still splits — leading stages just hold the
    # embed / visual with no decoder layer). ``pp_layer_split`` overrides.
    layer_split = cfg.train.accelerator.pp_layer_split if cfg is not None else None
    counts = _resolve_stage_layer_counts(n_layers, num_stages, layer_split)
    starts = [sum(counts[:i]) for i in range(num_stages)]
    pp_rank = pp_mesh.get_local_rank()
    vl_stage_cls = _load_qwen3_5_moe_vl_stage_cls() if is_vl else None

    stages = []
    for v in range(vpp):
        stage_index = pp_rank + v * num_ranks
        start = starts[stage_index]
        end = start + counts[stage_index]
        stage_module = _build_qwen3_5_moe_pp_stage_module(
            model,
            backbone,
            layers,
            stage_index,
            num_stages,
            start,
            end,
            is_vl,
            vl_stage_cls,
        )
        if device is not None:
            stage_module = stage_module.to(device=device)
        _wrap_qwen3_5_moe_pp_stage(stage_module, is_vl, fsdp_mesh, cfg)
        _apply_qwen3_5_moe_pp_stage_tp(stage_module, model, tp_mesh, fsdp_mesh)
        stages.append(
            PipelineStage(stage_module, stage_index, num_stages,
                          device=device, mesh=pp_mesh))

    schedule = _build_qwen3_5_moe_pp_schedule(
        stages,
        micro_batch_num,
        schedule_name,
        vpp,
        is_vl,
    )
    logger.info_rank0(
        "PP(%s) applied to Qwen3.5-MoE: pp=%d vpp=%d stages=%d, this rank owns %s",
        schedule_name, num_ranks, vpp, num_stages, [s.stage_index for s in stages],
    )
    return schedule, stages


def pipeline_qwen3_5_moe_for_trainer(
    model: Qwen3_5MoeForCausalLM,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> tuple[object, list[PipelineStage]]:
    """Split Qwen3.5-MoE into pipeline stages for the trainer.

    Args:
        model: Full Qwen3.5-MoE model before pipeline partitioning.
        mesh: Device mesh containing the pipeline and data-parallel dimensions.
        cfg: Trainer configuration controlling the pipeline schedule.

    Returns:
        The pipeline schedule and the stages owned by the current rank.
    """
    try:
        pp_mesh = mesh["pp"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parallel.pp>1 requires a DeviceMesh with a 'pp' dim.") from exc
    micro_batch_num = int(cfg.train.accelerator.pp_micro_batch_num)
    schedule_name = cfg.train.accelerator.pp_schedule or "1f1b"
    vpp = int(cfg.train.accelerator.pp_vpp or 1)
    device = next(model.parameters()).device
    is_vl = hasattr(model.model, "language_model")
    # TP / CP / EP shard *within* each decoder layer (sequence or expert dim),
    # so apply them to the full model before the PP split — the per-stage layer
    # slabs then reference the already-parallelized modules. FSDP (the dp dim)
    # is applied last, per-stage, inside ``pipelining_qwen3_5_moe``.
    _apply_tp_cp_ep(model, mesh, cfg)
    # Resolve only when a non-trivial shard axis exists. Pure dp_replicate
    # materializes a size-1 dp_shard to form a 2-D non-PP HSDP mesh, but PP
    # stages must stay plain so the trainer's PP DP reducer synchronizes them.
    fsdp_mesh = _resolve_qwen3_5_moe_pp_fsdp_mesh(mesh, cfg)
    if is_vl and fsdp_mesh is not None:
        _apply_vl_visual_tower(model, mesh, cfg)
    tp_size = int(cfg.train.accelerator.tp or 1)
    tp_mesh = mesh["tp"] if tp_size > 1 else None
    return pipelining_qwen3_5_moe(
        model, pp_mesh, device=device, micro_batch_num=micro_batch_num,
        fsdp_mesh=fsdp_mesh, cfg=cfg, schedule_name=schedule_name, vpp=vpp,
        tp_mesh=tp_mesh,
    )


__all__ = [
    "broadcast_state_dict_from_rank0",
    "parallelize_qwen3_5_moe",
    "parallelize_qwen3_5_moe_ep",
    "parallelize_qwen3_5_moe_tp",
    "pipelining_qwen3_5_moe",
    "pipeline_qwen3_5_moe_for_trainer",
    "qwen3_5_moe_tp_load_transforms",
]

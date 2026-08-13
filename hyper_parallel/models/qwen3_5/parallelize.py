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
"""Qwen3.5 (dense) parallelization — self-contained per-model logic.

This file owns Qwen3.5's TP / CP / AC / FSDP pipeline. The TP plan keeps
parameter shards on ``head`` or ``feature`` dimensions and norms on the
sequence axis so per-step gradients stay slice-faithful to the single-card
run.
"""
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

from hyper_parallel import (
    ColwiseParallel,
    ContextParallel,
    DTensor,
    HSDPModule,
    PipelineStage,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    SkipDTensorDispatch,
    fully_shard,
    parallelize_module,
)
from hyper_parallel.core.context_parallel.linear_attention_context_parallel import (
    LinearAttentionContextParallel,
)
from hyper_parallel.core.pipeline_parallel import (
    BatchDimSpec, Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B)
from hyper_parallel.core.pipeline_parallel.stage import SharedParameterInfo
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from hyper_parallel.models.qwen3_5.model import Qwen3_5ForCausalLM, Qwen3_5StageModule
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


def _resolve_mp_policy(cfg):
    """Build FSDP mixed-precision policy from the dense model YAML."""
    mp_cfg = cfg.train.mixed_precision
    if not mp_cfg.enabled:
        return None
    output_dtype_str = mp_cfg.output_dtype
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP.get(mp_cfg.param_dtype),
        reduce_dtype=_DTYPE_MAP.get(mp_cfg.reduce_dtype),
        output_dtype=_DTYPE_MAP.get(output_dtype_str) if output_dtype_str else None,
    )


def _build_dense_fsdp_kwargs(module: nn.Module, dp_mesh: DeviceMesh, cfg) -> dict:
    """Assemble dense Qwen3.5 FSDP kwargs over ``dp_mesh``."""
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


def _has_linear_attention_layers(model: Qwen3_5ForCausalLM) -> bool:
    """Return ``True`` if any layer is configured as ``linear_attention``.

    Qwen3.5 dense interleaves GatedDeltaNet (linear-attention) and full
    GQA (full-attention) layers. The TP / CP plans below cover the
    full-attention layers only; mixed configurations would silently fall
    back to local compute on the linear-attention layers and miss the
    gradient reduction across the TP mesh, so the helpers reject the
    combination instead.
    """
    layer_types = getattr(model.config, "layer_types", None) or []
    return any(layer_type == "linear_attention" for layer_type in layer_types)


def broadcast_state_dict_from_rank0(model: nn.Module) -> None:
    """Broadcast parameters from rank 0 so every rank starts from the same weights.

    Buffers are intentionally **not** broadcast: the rotary
    ``inv_freq`` table is deterministically rebuilt from the same theta /
    head_dim on every rank, so a broadcast would be redundant.
    """
    for param in model.parameters():
        _broadcast_from_rank0_(param.data)


def _is_rank_local_linear_attention_param(name: str) -> bool:
    """Return ``True`` for GatedDeltaNet params sliced locally over TP heads."""
    return (
        ".linear_attn.in_proj_qkv.weight" in name
        or ".linear_attn.conv1d.weight" in name
        or ".linear_attn.dt_bias" in name
        or ".linear_attn.A_log" in name
    )


def _register_tp_replicated_param_grad_sum(
    model: Qwen3_5ForCausalLM, tp_mesh: DeviceMesh,
) -> None:
    """All-reduce SUM gradients of parameters replicated over the TP mesh.

    DTensor-sharded parameters that span the TP mesh carry their layout in the
    tensor metadata and are reduced by DTensor autograd. Replicated parameters
    — either plain ``nn.Parameter`` values or ``Replicate`` DTensors such as
    SequenceParallel norms — only see this rank's sequence/head contribution
    after the trainer scales the replicated TP loss by ``1 / tp_size``, so their
    gradients must be summed over TP. Rank-local GatedDeltaNet state slices
    (``in_proj_qkv`` / ``conv1d`` / ``dt_bias`` / ``A_log``) are intentionally
    excluded.
    """
    if tp_mesh.size() <= 1:
        return
    tp_group = tp_mesh.get_group()

    def _sharded_over_tp_mesh(param: platform.Parameter) -> bool:
        if not isinstance(param, DTensor):
            return False
        if not any(pl.is_shard() for pl in param.placements):
            return False
        try:
            return param.device_mesh.get_group() == tp_group
        except (RuntimeError, ValueError, AttributeError):
            return False

    def _make_grad_hook(param: platform.Parameter):
        decision = {"reduce": None}

        def _hook(grad: platform.Tensor) -> platform.Tensor:
            if decision["reduce"] is None:
                decision["reduce"] = not _sharded_over_tp_mesh(param)
            if not decision["reduce"]:
                return grad
            if isinstance(grad, DTensor):
                _all_reduce_sum_(grad.to_local(), tp_group)
                return grad
            if not grad.is_contiguous():
                grad = grad.contiguous()
            _all_reduce_sum_(grad, tp_group)
            return grad

        return _hook

    def _attach_grad_hooks(module, hook_args):
        del hook_args
        for name, param in module.named_parameters():
            if _is_rank_local_linear_attention_param(name):
                continue
            if not param.requires_grad:
                continue
            if getattr(param, "_hp_tp_grad_sum_hook_attached", False):
                continue
            param.register_hook(_make_grad_hook(param))
            param._hp_tp_grad_sum_hook_attached = True  # pylint: disable=protected-access

    _attach_grad_hooks(model, None)
    model.register_forward_pre_hook(_attach_grad_hooks)


def _make_post_fsdp_tp_replicated_param_reducer(
    model: nn.Module, tp_mesh: DeviceMesh,
):
    """Build a post-FSDP TP reducer for grads not already reduced over TP."""
    if tp_mesh.size() <= 1:
        return None
    tp_group = tp_mesh.get_group()
    to_reduce = []
    resolved = {"done": False}

    def _layout_spans_tp(param: platform.Parameter) -> bool:
        if not isinstance(param, DTensor):
            return False
        device_mesh = param.device_mesh
        for dim in range(device_mesh.ndim):
            try:
                if device_mesh.get_group(dim) == tp_group:
                    return True
            except (RuntimeError, ValueError, KeyError):
                continue
        return False

    def _resolve() -> None:
        for name, param in model.named_parameters():
            if _is_rank_local_linear_attention_param(name):
                continue
            if getattr(param, "requires_grad", False) and not _layout_spans_tp(param):
                to_reduce.append(param)
        resolved["done"] = True

    def _reduce() -> None:
        if not resolved["done"]:
            _resolve()
        with SkipDTensorDispatch():
            for param in to_reduce:
                grad = param.grad
                if grad is None:
                    continue
                local = grad._local_tensor if isinstance(grad, DTensor) else grad  # pylint: disable=protected-access
                if local.device.type == "cpu":
                    buf = local.contiguous().to(tp_mesh.device_type)
                    _all_reduce_sum_(buf, tp_group)
                    local.copy_(buf.to(local.device))
                else:
                    _all_reduce_sum_(local, tp_group)

    return _reduce


def _gated_delta_tp_slices(linear_attn, tp_mesh: DeviceMesh):
    """Compute the rank-local channel / head slices for GatedDeltaNet's manually
    TP-sharded ``conv1d`` / ``dt_bias`` / ``A_log``.

    ``conv1d.weight`` is laid out ``[Q_channels | K_channels | V_channels]`` over
    the depthwise axis 0; Q/K use ``head_k_dim`` channels per (key) head and V
    uses ``head_v_dim`` per (value) head. Returns the three conv1d channel slices
    plus the value-head ``(start, end)`` used by ``dt_bias`` / ``A_log``. Shared
    by the in-place slicer and the load-time transform so they stay in lockstep.
    """
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


def _shard_gated_delta_local_params(linear_attn, tp_mesh: DeviceMesh) -> None:
    """Slice the per-rank head subset out of fused QKV / conv / SSM params.

    ``in_proj_qkv`` is a fused checkpoint ``[Q | K | V]`` block, so a plain
    ColwiseParallel split would cut through the wrong block boundaries. Slice
    each Q/K/V block by head, concatenate the rank-local blocks back into a
    fused local projection, and do the same for the depthwise conv channels.
    """
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

    # Concat the three rank-local conv slices and replace the parameter in place
    # so downstream callers that hold a reference to ``linear_attn.conv1d`` see
    # the sharded weight without a module rebuild.
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


def _build_gated_delta_tp_load_transforms(
    model: Qwen3_5ForCausalLM,
    tp_mesh: DeviceMesh,
    *,
    require_pre_sharded_params: bool,
) -> dict:
    """Build the canonical load-time transforms for TP-local GDN parameters."""
    if tp_mesh.size() <= 1 or not _has_linear_attention_layers(model):
        return {}

    transforms: dict = {}
    for idx, block in enumerate(model.layers):
        if block.layer_type == "full_attention":
            continue
        linear_attn = block.linear_attn
        # Only transform when the conv weight is *already* TP-sliced to the
        # rank-local shard (the meta-init → parallelize → load order). When it is
        # still full-size (the load-then-parallelize order, e.g. the PP-alone
        # path), the in-place ``_shard_gated_delta_local_params`` slices the
        # loaded weight afterwards, so the full checkpoint weight must load unchanged.
        full_conv_dim = (
            2 * linear_attn.key_dim
            + linear_attn.num_v_heads * linear_attn.head_v_dim
        )
        if require_pre_sharded_params and linear_attn.conv1d.weight.shape[0] == full_conv_dim:
            continue
        q_slice, k_slice, v_slice, head_start, head_end = _gated_delta_tp_slices(
            linear_attn, tp_mesh,
        )
        prefix = f"model.layers.{idx}.linear_attn"
        transforms[f"{prefix}.in_proj_qkv.weight"] = (
            lambda w, q=q_slice, k=k_slice, v=v_slice: torch.cat(
                [w[q], w[k], w[v]], dim=0,
            ).contiguous()
        )
        transforms[f"{prefix}.conv1d.weight"] = (
            lambda w, q=q_slice, k=k_slice, v=v_slice: torch.cat(
                [w[q], w[k], w[v]], dim=0,
            ).contiguous()
        )
        transforms[f"{prefix}.dt_bias"] = lambda w, s=head_start, e=head_end: w[s:e].clone()
        transforms[f"{prefix}.A_log"] = lambda w, s=head_start, e=head_end: w[s:e].clone()
    return transforms


def qwen3_5_tp_load_transforms(
    model: Qwen3_5ForCausalLM,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> dict:
    """``ModelSpec.tp_load_transform_fn`` — slice GatedDeltaNet TP-local weights.

    Under the trainer's meta-init → parallelize → load order the GatedDeltaNet
    ``in_proj_qkv`` / ``conv1d`` / ``dt_bias`` / ``A_log`` params are sliced to
    a rank-local head subset on meta (see
    :func:`_shard_gated_delta_local_params`), so the full checkpoint weight no longer
    matches their shape. This returns transforms that map the full checkpoint
    tensors to this rank's slices before load. No-op for ``tp <= 1`` or a
    full-attention-only model.
    """
    tp_size = int(cfg.train.accelerator.tp or 1)
    if tp_size <= 1 or not _has_linear_attention_layers(model):
        return {}
    return _build_gated_delta_tp_load_transforms(
        model,
        mesh["tp"],
        require_pre_sharded_params=True,
    )


def qwen3_5_inference_tp_load_transforms(
    model: Qwen3_5ForCausalLM,
    tp_mesh: DeviceMesh,
) -> dict:
    """Return transforms for native GDN parameters sliced outside DTensor.

    Args:
        model: Native Qwen3.5 model whose GDN parameters are already TP-local.
        tp_mesh: One-dimensional inference tensor-parallel mesh.

    Returns:
        Checkpoint-key transforms that select this rank's GDN blocks.
    """
    return _build_gated_delta_tp_load_transforms(
        model,
        tp_mesh,
        require_pre_sharded_params=False,
    )


def _apply_linear_attention_cp(module: nn.Module, cp_mesh: DeviceMesh, mode: str) -> None:
    """Apply CP to a Qwen3.5 linear-attention module."""
    LinearAttentionContextParallel(mode=mode).apply(module, cp_mesh)


def _validate_qwen3_5_tp_config(model: Qwen3_5ForCausalLM, tp_world: int) -> None:
    """Validate TP divisibility constraints for Qwen3.5 dense layers."""
    cfg = model.config
    if cfg.num_attention_heads % tp_world != 0:
        raise ValueError(
            f"num_attention_heads ({cfg.num_attention_heads}) must divide TP size ({tp_world})."
        )
    if cfg.num_key_value_heads % tp_world != 0:
        raise ValueError(
            f"num_key_value_heads ({cfg.num_key_value_heads}) must divide TP size ({tp_world})."
        )
    if not _has_linear_attention_layers(model):
        return
    for field, value in (
        ("linear_num_value_heads", cfg.linear_num_value_heads),
        ("linear_num_key_heads", cfg.linear_num_key_heads),
    ):
        if value % tp_world != 0:
            raise ValueError(
                f"{field} ({value}) must divide TP size ({tp_world}); "
                "GatedDeltaNet's TP plan shards the per-head dim."
            )


def _validate_qwen3_5_inference_tp_config(
    model: Qwen3_5ForCausalLM,
    tp_world: int,
) -> None:
    """Validate inference TP dimensions that Hyper shards without padding."""
    _validate_qwen3_5_tp_config(model, tp_world)
    cfg = model.config
    for field, value in (
        ("vocab_size", cfg.vocab_size),
        ("intermediate_size", cfg.intermediate_size),
    ):
        if value % tp_world != 0:
            raise ValueError(
                f"{field} ({value}) must be divisible by TP size ({tp_world})."
            )


_TP_PROFILE_TRAINING_SP = "training_sp"
_TP_PROFILE_INFERENCE_REPLICATED = "inference_replicated"


def _build_qwen3_5_tp_plans(
    activation_profile: str,
    *,
    enable_loss_parallel: bool = False,
):
    """Build canonical parameter-sharding plans with profile-specific activations."""
    if activation_profile == _TP_PROFILE_TRAINING_SP:
        output_layout = Shard(1)
        colwise = ColwiseParallel()
        rowwise_output = RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=output_layout,
            use_local_output=False,
        )
        rowwise_reduce_output = RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=output_layout,
            reduce_dtype=torch.float32,
            use_local_output=False,
        )
        norm = SequenceParallel(sequence_dim=1, use_local_output=False)
        input_boundary = PrepareModuleInput(
            input_layouts=(output_layout,),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        )
        root = {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=output_layout,
                use_local_output=False,
            ),
            "model.norm": norm,
            "lm_head": ColwiseParallel(
                input_layouts=output_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        }
        layer_boundaries = {
            "input_layernorm": norm,
            "post_attention_layernorm": norm,
            "mlp": input_boundary,
        }
    elif activation_profile == _TP_PROFILE_INFERENCE_REPLICATED:
        output_layout = Replicate()
        colwise = ColwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(-1),
            use_local_output=True,
        )
        rowwise_output = RowwiseParallel(
            input_layouts=Shard(-1),
            output_layouts=output_layout,
            use_local_output=True,
        )
        rowwise_reduce_output = rowwise_output
        input_boundary = None
        root = {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=output_layout,
                use_local_output=True,
            ),
            "lm_head": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=output_layout,
                use_local_output=True,
            ),
        }
        layer_boundaries = {}
    else:
        raise ValueError(
            f"Unknown Qwen3.5 TP activation profile {activation_profile!r}; expected "
            f"{_TP_PROFILE_TRAINING_SP!r} or {_TP_PROFILE_INFERENCE_REPLICATED!r}."
        )

    mlp = {
        "mlp.gate_proj": colwise,
        "mlp.up_proj": colwise,
        "mlp.down_proj": rowwise_output,
    }
    full_attention = {}
    if input_boundary is not None:
        full_attention["self_attn"] = input_boundary
    full_attention.update({
        "self_attn.q_proj": colwise,
        "self_attn.k_proj": colwise,
        "self_attn.v_proj": colwise,
    })
    if input_boundary is not None:
        full_attention.update({
            "self_attn.q_norm": SequenceParallel(sequence_dim=2, use_local_output=True),
            "self_attn.k_norm": SequenceParallel(sequence_dim=2, use_local_output=True),
        })
    full_attention["self_attn.o_proj"] = rowwise_reduce_output

    linear_attention = {}
    if input_boundary is not None:
        linear_attention["linear_attn"] = input_boundary
    linear_attention.update({
        "linear_attn.in_proj_z": colwise,
        "linear_attn.in_proj_b": colwise,
        "linear_attn.in_proj_a": colwise,
        "linear_attn.out_proj": rowwise_output,
    })
    norm_and_mlp = {**layer_boundaries, **mlp}
    return SimpleNamespace(
        root=root,
        norm_and_mlp=norm_and_mlp,
        norm_and_mlp_reduce={**norm_and_mlp, "mlp.down_proj": rowwise_reduce_output},
        full_attention=full_attention,
        linear_attention=linear_attention,
        linear_attention_reduce={**linear_attention, "linear_attn.out_proj": rowwise_reduce_output},
    )


def _apply_qwen3_5_tp_layer_plan(
    block: nn.Module,
    layer_idx: int,
    tp_mesh: DeviceMesh,
    plans,
    mlp_reduce_layer_indices,
    linear_reduce_layer_indices,
    *,
    src_data_rank: Optional[int],
) -> None:
    """Apply the dense TP plan for one decoder block."""
    norm_and_mlp = (
        plans.norm_and_mlp_reduce
        if layer_idx in mlp_reduce_layer_indices
        else plans.norm_and_mlp
    )
    if block.layer_type == "full_attention":
        plan = {**norm_and_mlp, **plans.full_attention}
        if src_data_rank is None:
            parallelize_module(block, tp_mesh, plan, src_data_rank=None)
        else:
            parallelize_module(block, tp_mesh, plan)
        return

    linear_attention = (
        plans.linear_attention_reduce
        if layer_idx in linear_reduce_layer_indices
        else plans.linear_attention
    )
    plan = {**norm_and_mlp, **linear_attention}
    if src_data_rank is None:
        parallelize_module(block, tp_mesh, plan, src_data_rank=None)
    else:
        parallelize_module(block, tp_mesh, plan)
    _shard_gated_delta_local_params(block.linear_attn, tp_mesh)
    bind_runtime = getattr(block.linear_attn, "bind_state_runtime_parameters", None)
    if bind_runtime is not None:
        bind_runtime()


def _apply_qwen3_5_tp(
    model: Qwen3_5ForCausalLM,
    tp_mesh: DeviceMesh,
    *,
    activation_profile: str,
    enable_loss_parallel: bool,
    register_grad_hooks: bool,
) -> Qwen3_5ForCausalLM:
    """Apply canonical Qwen3.5 TP parameter shards under one activation profile."""
    tp_world = tp_mesh.size()
    inference = activation_profile == _TP_PROFILE_INFERENCE_REPLICATED
    if inference:
        _validate_qwen3_5_inference_tp_config(model, tp_world)
        if tp_world <= 1:
            return model
    else:
        _validate_qwen3_5_tp_config(model, tp_world)

    plans = _build_qwen3_5_tp_plans(
        activation_profile,
        enable_loss_parallel=enable_loss_parallel,
    )
    src_data_rank = None if inference else 0
    if src_data_rank is None:
        parallelize_module(model, tp_mesh, plans.root, src_data_rank=None)
    else:
        parallelize_module(model, tp_mesh, plans.root)
    if model.config.tie_word_embeddings:
        model.tie_weights()

    full_attention_layer_indices = {
        idx for idx, block in enumerate(model.layers)
        if block.layer_type == "full_attention"
    }
    if inference:
        mlp_reduce_layer_indices = set()
        linear_reduce_layer_indices = set()
    else:
        mlp_reduce_layer_indices = full_attention_layer_indices
        linear_reduce_layer_indices = {
            idx for idx, block in enumerate(model.layers)
            if block.layer_type == "linear_attention"
        }

    for layer_idx, block in enumerate(model.layers):
        _apply_qwen3_5_tp_layer_plan(
            block,
            layer_idx,
            tp_mesh,
            plans,
            mlp_reduce_layer_indices,
            linear_reduce_layer_indices,
            src_data_rank=src_data_rank,
        )

    if inference:
        logger.info_rank0(
            "Inference TP applied to Qwen3.5: replicated activations, tp_size=%d, layers=%d",
            tp_world,
            len(model.layers),
        )
    else:
        if register_grad_hooks:
            _register_tp_replicated_param_grad_sum(model, tp_mesh)
        model.hp_loss_tp_scale_size = tp_world
        logger.info_rank0(
            "TP applied to Qwen3.5: SequenceParallel plan, tp_size=%d, layers=%d",
            tp_world, len(model.layers),
        )
    return model


def parallelize_qwen3_5_tp(
    model: Qwen3_5ForCausalLM,
    tp_mesh: DeviceMesh,
    *,
    enable_sequence_parallel: bool = True,
    enable_loss_parallel: bool = False,
    register_grad_hooks: bool = True,
) -> Qwen3_5ForCausalLM:
    """Apply 1-D tensor parallelism to Qwen3.5 dense (SequenceParallel plan).

    The plan keeps parameter shards and sequence-sharded activations aligned
    so single-card and TP-N runs see identical per-step gradients:

    * ``model.embed_tokens`` — RowwiseParallel ``Replicate`` → ``Shard(1)``
      (sequence-sharded output)
    * ``model.norm`` / per-layer ``input_layernorm`` /
      ``post_attention_layernorm`` — SequenceParallel on dim 1
    * ``lm_head`` — ColwiseParallel ``Shard(1)`` → ``Replicate`` (or
      ``Shard(-1)`` under loss parallel)
    * Each ``self_attn``: ``PrepareModuleInput`` from ``Shard(1)`` to
      ``Replicate``; ``q_proj`` / ``k_proj`` / ``v_proj`` Colwise;
      ``o_proj`` Rowwise ``Shard(-1)`` → ``Shard(1)``
    * Each ``mlp``: ``PrepareModuleInput`` from ``Shard(1)`` to
      ``Replicate``; ``gate_proj`` / ``up_proj`` Colwise; ``down_proj``
      Rowwise ``Shard(-1)`` → ``Shard(1)``

    Requires ``num_attention_heads % tp_size == 0`` **and**
    ``num_key_value_heads % tp_size == 0``; SP additionally requires
    ``seq_len % tp_size == 0`` so the sequence shard divides evenly.
    """
    if not enable_sequence_parallel:
        raise NotImplementedError(
            "Qwen3.5 TP currently only supports the SequenceParallel path."
        )

    return _apply_qwen3_5_tp(
        model,
        tp_mesh,
        activation_profile=_TP_PROFILE_TRAINING_SP,
        enable_loss_parallel=enable_loss_parallel,
        register_grad_hooks=register_grad_hooks,
    )


def parallelize_qwen3_5_inference_tp(
    model: Qwen3_5ForCausalLM,
    tp_mesh: DeviceMesh,
) -> Qwen3_5ForCausalLM:
    """Shard native Qwen3.5 weights while retaining replicated packed tokens.

    Args:
        model: Native Hyper Qwen3.5 model, optionally containing vLLM runtime
            leaves that do not own checkpoint parameters.
        tp_mesh: One-dimensional mesh backed by vLLM's existing TP group.

    Returns:
        The input model with native Hyper parameters sharded in place.
    """
    return _apply_qwen3_5_tp(
        model,
        tp_mesh,
        activation_profile=_TP_PROFILE_INFERENCE_REPLICATED,
        enable_loss_parallel=False,
        register_grad_hooks=False,
    )


def parallelize_qwen3_5_cp(
    model: Qwen3_5ForCausalLM,
    cp_mesh: DeviceMesh,
    *,
    ulysses_degree: Optional[int] = None,
    linear_attention_cp_mode: str = "ulysses",
) -> Qwen3_5ForCausalLM:
    """Apply context parallelism across the Qwen3.5 hybrid decoder.

    Full-attention layers hook :class:`Qwen3_5SdpaCore` with
    :class:`hyper_parallel.ContextParallel` in BHSD mode (``seq_dim=2``,
    ``head_dim=1``) because the core consumes ``[B, H, S, D]`` Q/K/V.
    ``ulysses_degree`` defaults to ``None`` which the underlying
    :class:`ContextParallel` resolves to **Pure Ulysses** (``cp_size`` =
    sequence→head all-to-all). Pure Ulysses is the safe default because the
    ``ulysses_degree=1`` all-gathers K/V on every CP rank, so
    ``F.scaled_dot_product_attention(is_causal=True)`` falls back to
    a right-aligned causal mask that **leaks future tokens** to every CP rank
    except the last — a silent correctness bug unless the caller passes an
    explicit per-rank causal mask. Pure Ulysses sidesteps it because each
    rank ends up with the full sequence on a head-shard and a square causal
    mask is correct again.

    Linear-attention (:class:`Qwen3_5GatedDeltaNet`) layers use a matching
    pure-Ulysses execution wrapper: project local sequence shards, all-to-all
    the projected Q/K/V/B/A tensors to full-sequence local-head shards, run
    the per-head conv and gated delta rule on local heads, then all-to-all the
    result back to sequence shards before the output projection.
    """
    # Only pure Ulysses is wired here; a smaller ``ulysses_degree`` makes each
    # rank attend over gathered K/V with ``is_causal=True`` but without a
    # per-rank causal mask, leaking future tokens to every CP rank but the last.
    # Reject it loudly until that mask is wired.
    if ulysses_degree is not None and ulysses_degree < cp_mesh.size():
        raise NotImplementedError(
            f"Qwen3.5 CP: ulysses_degree={ulysses_degree} < cp={cp_mesh.size()} "
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
            _apply_linear_attention_cp(block.linear_attn, cp_mesh, linear_attention_cp_mode)
            linear_attached += 1
    logger.info_rank0(
        "CP applied to Qwen3.5: cp_size=%d, ulysses_degree=%s, full-attn hooks=%d, "
        "linear-attn %s hooks=%d",
        cp_mesh.size(), ulysses_degree, full_attached, linear_attention_cp_mode, linear_attached,
    )
    return model


def _needs_gqa_kv_expand_for_cp(model: Qwen3_5ForCausalLM, cp_mesh: DeviceMesh) -> bool:
    """Return whether GQA K/V heads must be expanded before Ulysses CP."""
    tp_size = int(getattr(model, "hp_loss_tp_scale_size", 1) or 1)
    q_heads = int(model.config.num_attention_heads or 1)
    kv_heads = int(model.config.num_key_value_heads or 1)
    local_kv_heads = max(kv_heads // tp_size, 1)
    if local_kv_heads % cp_mesh.size() == 0:
        return False
    local_q_heads = max(q_heads // tp_size, 1)
    return local_q_heads % cp_mesh.size() == 0 and local_q_heads % local_kv_heads == 0


def _resolve_fsdp_mesh(mesh):
    """Return the dp-shard ``DeviceMesh`` to FSDP-wrap a dense PP stage over, or
    ``None`` to leave the stage replicated.

    A stage is FSDP-sharded only when ``dp_shard`` is the *entire* data-parallel
    group (pure FSDP, ``dp_replicate == 1``): it is then sharded + reduce-
    scattered over the ``fsdp`` / ``dp_shard`` axis (always the 1-D shard mesh —
    ``mesh["fsdp"]`` equals ``dp_shard``). ``None`` is returned, leaving the
    stage replicated (plain params, the trainer's PP DP grad all-reduce covers
    the full dp mesh), when:

    * there is no shard axis with size > 1 (pure ``dp_replicate`` / no data
      parallel) — else the whole-mesh fallback would shard a stage across other
      PP stages' ranks; or
    * ``dp_replicate > 1`` (HSDP) — FSDP over the 1-D ``dp_shard`` axis alone
      would reduce-scatter within each shard group but never sync the
      ``dp_replicate`` groups; per-stage HSDP sharding is not wired, so those
      stages stay replicated (correct, just not memory-optimal).

    Size-1 dims are normally omitted from the mesh, so presence of the
    ``fsdp`` / ``dp_shard`` axis is what distinguishes a real shard axis. The
    one deliberate exception: a low-precision run injects a size-1
    ``dp_shard`` (``build_mesh(force_dp_shard=True)``) precisely so the
    stages get the FSDP wrap that carries ``MixedPrecisionPolicy`` — its
    communication is a same-rank no-op, so it is accepted here too.
    """
    try:
        if mesh["dp_replicate"].size() > 1:
            logger.info_rank0(
                "PP+HSDP: dp_replicate>1 — leaving stages replicated; per-stage "
                "FSDP sharding over the dp_shard axis alone is not wired for HSDP."
            )
            return None
    except (KeyError, TypeError):
        pass
    for dim in ("fsdp", "dp_shard"):
        try:
            return mesh[dim]
        except (KeyError, TypeError):
            continue
    return None


def _fsdp_wrap_stage(stage_module, dp_mesh, cfg) -> None:
    """Wrap a dense pipeline stage's children + root as FSDP units.

    Each decoder layer plus the embed / norm / lm_head this stage owns becomes
    its own FSDP unit, then the stage module itself is wrapped as the root unit
    so the schedule's ``FSDP_REDUCE_GRAD`` root drain applies the boundary
    units' reduce-scatter. This uses the same per-layer + root shape as the
    non-PP ``_apply_fsdp`` path. The dense
    :class:`Qwen3_5StageModule` only sets ``embed_tokens`` / ``norm`` /
    ``lm_head`` on the stages that own them, so they are fetched with
    ``getattr(..., None)`` rather than read directly.

    ``reshard_after_forward`` is forced off: the pipeline scheduler drives
    unshard / reshard explicitly (FSDP_UNSHARD / FSDP_RESHARD) so params must
    stay unsharded across all micro-batches of a step under both gpipe and 1F1B.
    """
    fsdp_kwargs = _build_dense_fsdp_kwargs(stage_module, dp_mesh, cfg)
    fsdp_kwargs["reshard_after_forward"] = False
    boundary_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    root_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    for layer in stage_module.layers:
        fully_shard(layer, **fsdp_kwargs)
    for sub in (
        getattr(stage_module, "embed_tokens", None),
        getattr(stage_module, "norm", None),
        getattr(stage_module, "lm_head", None),
    ):
        if sub is not None:
            fully_shard(sub, **boundary_fsdp_kwargs)
    fully_shard(stage_module, **root_fsdp_kwargs)
    logger.info_rank0(
        "PP+FSDP: wrapped %d layer(s) + head/embed + root as stage FSDP units (dp=%d)",
        len(stage_module.layers),
        dp_mesh.size() if hasattr(dp_mesh, "size") else 1,
    )


def _validate_qwen3_5_pp_shape(model, num_ranks: int, vpp: int, cfg):
    """Validate dense PP partitioning and return per-stage layer counts."""
    tie_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    if num_ranks < 2:
        raise ValueError(
            f"pipelining_qwen3_5 requires pp_mesh.size() >= 2; got {num_ranks}."
        )
    if tie_embeddings and num_ranks > 2:
        raise NotImplementedError(
            "Qwen3.5 tied-embedding PP supports exactly 2 pp ranks (got "
            f"pp={num_ranks}); the cross-stage shared embed / lm_head group "
            "spans only the first and last ranks. Use pp=2 for tied models, "
            "or build with tie_word_embeddings=False for pp>2."
        )
    n_layers = len(model.layers)
    num_stages = num_ranks * vpp
    layer_split = cfg.train.accelerator.pp_layer_split if cfg is not None else None
    if not layer_split and n_layers % num_stages != 0:
        raise ValueError(
            f"num_hidden_layers ({n_layers}) must divide the total stage count "
            f"pp*pp_vpp ({num_ranks}*{vpp}={num_stages}) so the per-stage layer "
            "slabs stay balanced — or set pp_layer_split for an explicit uneven split."
        )
    return _resolve_stage_layer_counts(
        n_layers,
        num_stages,
        layer_split,
        allow_empty_stages=False,
    )


def _qwen3_5_stage_shared_parameters(
    tie_embeddings: bool,
    stage_module,
    is_first: bool,
    is_last: bool,
    num_stages: int,
):
    """Return tied-embedding shared parameter info for one dense PP stage."""
    if not tie_embeddings:
        return None
    if is_first:
        return SharedParameterInfo(
            stage_module.embed_tokens.weight,
            [0, num_stages - 1],
            owner_module=stage_module.embed_tokens,
            param_name="weight",
        )
    if is_last:
        return SharedParameterInfo(
            stage_module.lm_head.weight,
            [0, num_stages - 1],
            owner_module=stage_module.lm_head,
            param_name="weight",
        )
    return None


def _build_qwen3_5_pp_stage(
    model,
    stage_index: int,
    num_stages: int,
    start: int,
    end: int,
    device,
    fsdp_mesh,
    cfg,
):
    """Build one dense Qwen3.5 pipeline stage module."""
    is_first = stage_index == 0
    is_last = stage_index == num_stages - 1
    stage_module = Qwen3_5StageModule(
        layers=list(model.layers[start:end]),
        embed_tokens=model.model.embed_tokens if is_first else None,
        rotary_emb=model.model.rotary_emb,
        norm=model.model.norm if is_last else None,
        lm_head=model.lm_head if is_last else None,
    )
    if device is not None:
        stage_module = stage_module.to(device=device)
    if fsdp_mesh is not None:
        _fsdp_wrap_stage(stage_module, fsdp_mesh, cfg)
    return stage_module, is_first, is_last


def _apply_qwen3_5_pp_stage_tp(stage_module, tp_mesh, fsdp_mesh) -> None:
    """Attach TP gradient handling to one local dense pipeline stage."""
    if tp_mesh is None or tp_mesh.size() <= 1:
        return
    stage_module.hp_loss_tp_scale_size = tp_mesh.size()
    if fsdp_mesh is None:
        _register_tp_replicated_param_grad_sum(stage_module, tp_mesh)
        return
    reducer = _make_post_fsdp_tp_replicated_param_reducer(stage_module, tp_mesh)
    if reducer is not None:
        stage_module.hp_post_fsdp_grad_reduce = reducer


def _build_qwen3_5_pp_schedule(stages, micro_batch_num: int, schedule_name: str, vpp: int):
    """Build the dense PP schedule for local stages."""
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


def pipelining_qwen3_5(
    model: Qwen3_5ForCausalLM,
    pp_mesh: DeviceMesh,
    *,
    device: Optional[torch.device] = None,
    micro_batch_num: int = 1,
    fsdp_mesh: Optional[DeviceMesh] = None,
    tp_mesh: Optional[DeviceMesh] = None,
    cfg: Optional["HyperTrainerConfig"] = None,
    schedule_name: str = "gpipe",
    vpp: int = 1,
) -> tuple[object, list[PipelineStage]]:
    """Split dense Qwen3.5 into local pipeline stages and a schedule.

    Args:
        model: Fully constructed :class:`Qwen3_5ForCausalLM`.
        pp_mesh: 1-D PP sub-mesh (``mesh["pp"]``).
        device: Device tensors live on.
        micro_batch_num: Number of micro-batches per global step.
        tp_mesh: Optional TP sub-mesh used to synchronize replicated stage
            parameters after the pipeline split.
        schedule_name: ``"gpipe"`` or ``"1f1b"`` when ``vpp == 1``.
        vpp: Virtual-pipeline stage chunks per rank.

    Returns:
        ``(schedule, stages)`` for this rank.

    Note:
        When ``tp_mesh`` is provided, direct callers remain responsible for
        dividing the backward sensitivity of a local replicated loss by the
        TP size. The Trainer applies that normalization after all stage
        gradient reducers; core pipeline schedules retain unit sensitivity.
    """
    num_ranks = pp_mesh.size()
    vpp = max(1, int(vpp))
    num_stages = num_ranks * vpp  # total interleaved global stages (P * V)
    tie_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    counts = _validate_qwen3_5_pp_shape(model, num_ranks, vpp, cfg)
    starts = [sum(counts[:i]) for i in range(num_stages)]
    pp_rank = pp_mesh.get_local_rank()

    stages = []
    for v in range(vpp):
        stage_index = pp_rank + v * num_ranks
        start = starts[stage_index]
        end = start + counts[stage_index]
        stage_module, is_first, is_last = _build_qwen3_5_pp_stage(
            model,
            stage_index,
            num_stages,
            start,
            end,
            device,
            fsdp_mesh,
            cfg,
        )
        _apply_qwen3_5_pp_stage_tp(stage_module, tp_mesh, fsdp_mesh)
        shared_parameters = _qwen3_5_stage_shared_parameters(
            tie_embeddings,
            stage_module,
            is_first,
            is_last,
            num_stages,
        )
        stages.append(
            PipelineStage(
                stage_module,
                stage_index,
                num_stages,
                device=device,
                mesh=pp_mesh,
                shared_parameters=shared_parameters,
            )
        )

    schedule = _build_qwen3_5_pp_schedule(stages, micro_batch_num, schedule_name, vpp)
    logger.info_rank0(
        "PP applied to Qwen3.5: pp=%d vpp=%d stages=%d, this rank owns %s",
        num_ranks, vpp, num_stages, [s.stage_index for s in stages],
    )
    return schedule, stages


def pipeline_qwen3_5_for_trainer(
    model: Qwen3_5ForCausalLM,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> tuple[object, list[PipelineStage]]:
    """``ModelSpec.pipelining_fn`` entry — split Qwen3.5 for the trainer.

    The trainer hands the **full** :class:`DeviceMesh` plus the run config;
    this adapter extracts the ``pp`` sub-mesh and ``pp_micro_batch_num`` and
    delegates to :func:`pipelining_qwen3_5` (the canonical splitter shared by
    direct pipeline callers). The model is already materialized on its
    device by the time the trainer calls this, so the stage device is read from
    the model's parameters rather than moved.

    Args:
        model: Fully constructed + weight-loaded :class:`Qwen3_5ForCausalLM`.
        mesh: The trainer's root ``DeviceMesh`` (must expose a ``pp`` dim).
        cfg: The trainer run config (reads ``train.accelerator.pp_micro_batch_num``).

    Returns:
        ``(schedule, stages)`` from :func:`pipelining_qwen3_5`.

    Raises:
        ValueError: When the mesh has no ``pp`` dimension.
    """
    try:
        pp_mesh = mesh["pp"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "parallel.pp>1 requires a DeviceMesh with a 'pp' dim."
        ) from exc
    # TP shards the per-layer weights / sequence within each stage, so apply it
    # to the full model before the PP split — the per-stage slabs then reference
    # the already TP-sharded modules (the GatedDeltaNet conv / SSM weights are
    # filled by the spec's ``tp_load_transform_fn`` at load time).
    tp_size = int(cfg.train.accelerator.tp or 1)
    tp_mesh = None
    if tp_size > 1:
        try:
            tp_mesh = mesh["tp"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "parallel.tp>1 with pp>1 requires a DeviceMesh with a 'tp' dim."
            ) from exc
        parallelize_qwen3_5_tp(
            model,
            tp_mesh,
            register_grad_hooks=False,
        )
    micro_batch_num = int(cfg.train.accelerator.pp_micro_batch_num)
    schedule_name = cfg.train.accelerator.pp_schedule or "gpipe"
    vpp = int(cfg.train.accelerator.pp_vpp or 1)
    device = next(model.parameters()).device
    # FSDP (the dp-shard axis) is applied last, per-stage, inside
    # ``pipelining_qwen3_5`` so the stage's params are sharded over dp_shard.
    # ``_resolve_fsdp_mesh`` returns ``None`` for a pure replicate / no-dp run, so
    # those stay replicated (the trainer's DP grad all-reduce handles them) rather
    # than being mis-sharded across PP stages.
    fsdp_mesh = _resolve_fsdp_mesh(mesh)
    if fsdp_mesh is None and tp_size > 1:
        # HSDP (dp_replicate>1) leaves the stage replicated, but TP has already
        # made the stage params DTensors. The trainer's PP DP grad all-reduce
        # skips DTensor params (treating them as FSDP-sharded), so the
        # dp_replicate groups would silently diverge. Reject this composite
        # rather than train it incorrectly (use dp_shard for data parallelism).
        try:
            replicate_gt1 = mesh["dp_replicate"].size() > 1
        except (KeyError, TypeError):
            replicate_gt1 = False
        if replicate_gt1:
            raise NotImplementedError(
                "Qwen3.5 dense PP with tp>1 and dp_replicate>1 is not supported: "
                "the TP-sharded stage params are DTensors, so the trainer's PP DP "
                "grad all-reduce would skip them and the dp_replicate groups would "
                "diverge. Use dp_shard for data parallelism, or set tp=1."
            )
    return pipelining_qwen3_5(
        model, pp_mesh, device=device, micro_batch_num=micro_batch_num,
        fsdp_mesh=fsdp_mesh, tp_mesh=tp_mesh, cfg=cfg,
        schedule_name=schedule_name, vpp=vpp,
    )


def _apply_ac(model: Qwen3_5ForCausalLM, cfg) -> None:
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
    logger.info_rank0("AC applied to %d Qwen3.5 layers (mode=%s)", len(layers), ac_mode)


def _should_skip_qwen3_5_fsdp(cfg) -> bool:
    """Return whether a dense single-rank run can keep plain parameters."""
    accelerator = cfg.train.accelerator
    single_rank = _world_size() == 1
    has_parallel_axis = any(
        int(value or 1) > 1
        for value in (
            accelerator.dp_replicate,
            accelerator.dp_shard,
            accelerator.tp,
            accelerator.cp,
            accelerator.pp,
            accelerator.ep,
            accelerator.etp,
        )
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


def _resolve_qwen3_5_fsdp_mesh(mesh, cfg, cp_size: int):
    """Resolve a synchronization-complete mesh for the dense FSDP wrap.

    A two-dimensional HSDP mesh must preserve ``dp_replicate`` as axis zero
    and ``dp_shard`` as axis one. The flattened ``"dp"`` alias is not
    sufficient: ``fully_shard`` would interpret that one-dimensional mesh as
    plain FSDP and omit synchronization between replica groups.
    """
    accelerator = cfg.train.accelerator
    replicate_size = int(accelerator.dp_replicate or 1)
    if replicate_size > 1 and cp_size > 1:
        raise ValueError(
            "Qwen3.5 dense non-PP training does not support HSDP with CP: "
            "the current fully_shard mesh cannot express both the outer "
            "dp_replicate synchronization and the CP gradient domain."
        )
    if replicate_size > 1:
        try:
            return mesh[("dp_replicate", "dp_shard")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Qwen3.5 dense HSDP requires a DeviceMesh with ordered "
                "'dp_replicate' and 'dp_shard' dimensions."
            ) from exc
    if cp_size > 1:
        try:
            return mesh["loss"]
        except (KeyError, TypeError):
            return mesh["cp"]
    try:
        return mesh["fsdp"]
    except (KeyError, TypeError):
        try:
            return mesh["dp_shard"]
        except (KeyError, TypeError):
            return None


def _apply_fsdp(model: Qwen3_5ForCausalLM, mesh, cfg) -> None:
    """Per-layer + root FSDP wrap for real data/sequence parallel axes.

    Pure single-rank runs with no parallel axis skip FSDP so the trainer
    baseline uses the same plain ``nn.Parameter`` AdamW path as HF. Mixed
    precision storage is handled by the post-load dtype cast. If the YAML
    explicitly requests an FSDP output dtype, keep the size-1 FSDP wrap because
    that policy has no non-FSDP equivalent.
    """
    if _should_skip_qwen3_5_fsdp(cfg):
        logger.info_rank0(
            "Single-rank Qwen3.5 run has no parallel axes; skipping FSDP wrap. "
            "Post-load param_dtype casting handles mixed-precision storage."
        )
        return

    cp_size = int(cfg.train.accelerator.cp or 1)
    dp_mesh = _resolve_qwen3_5_fsdp_mesh(mesh, cfg, cp_size)
    if dp_mesh is None:
        logger.info_rank0("No FSDP/dp_shard/cp mesh dim; skipping FSDP wrap.")
        return

    fsdp_kwargs = _build_dense_fsdp_kwargs(model, dp_mesh, cfg)

    if not hasattr(model, "layers"):
        logger.warning(
            "Qwen3_5ForCausalLM has no ``.layers`` — root-only FSDP wrap. "
            "This is usually wrong for transformer-style models."
        )
        fully_shard(model, **fsdp_kwargs)
        return

    layers = list(model.layers)
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    if cp_size > 1:
        model.set_reduce_op_type("sum")
        model.hp_token_loss_scale_size = 1
        logger.info_rank0("CP FSDP reduce op set to SUM for Qwen3.5 token-weighted loss.")
    logger.info_rank0("FSDP applied to Qwen3.5: %d layers + root", len(layers))


def parallelize_qwen3_5(
    model: Qwen3_5ForCausalLM,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> Qwen3_5ForCausalLM:
    """Apply TP / CP / AC / FSDP to a Qwen3.5 dense model.

    Order: TP first so per-layer weights are sharded before CP attaches to
    ``sdpa_core``; AC wraps the (possibly TP-sharded) layers, and FSDP
    wrap units close over the already-parallelized modules.

    Args:
        model: Qwen3.5 dense model to parallelize.
        mesh: Device mesh containing the configured parallel dimensions.
        cfg: Trainer configuration controlling parallel degrees and policies.

    Returns:
        The parallelized model.
    """
    _resolve_qwen3_5_fsdp_mesh(
        mesh, cfg, int(cfg.train.accelerator.cp or 1),
    )
    tp_size = int(cfg.train.accelerator.tp)
    if tp_size > 1:
        try:
            tp_mesh = mesh["tp"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "parallel.tp>1 requires a DeviceMesh with a 'tp' dim."
            ) from exc
        parallelize_qwen3_5_tp(
            model,
            tp_mesh,
            register_grad_hooks=False,
        )

    cp_size = int(cfg.train.accelerator.cp)
    if tp_size > 1 and cp_size > 1 and _has_linear_attention_layers(model):
        raise NotImplementedError(
            "Qwen3.5 TP+CP for linear-attention layers is not supported in the "
            "initial linear-attention CP path. Set parallel.tp=1 when parallel.cp>1."
        )
    if cp_size > 1:
        try:
            cp_mesh = mesh["cp"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "parallel.cp>1 requires a DeviceMesh with a 'cp' dim."
            ) from exc
        ulysses_degree = cfg.train.accelerator.ulysses_degree
        # ``None`` (the default) resolves to pure Ulysses (degree == cp_size)
        # inside ``parallelize_qwen3_5_cp``; only coerce when explicitly set.
        ulysses_degree = int(ulysses_degree) if ulysses_degree is not None else None
        linear_attention_cp_mode = getattr(
            cfg.train.accelerator,
            "linear_attention_cp_mode",
            "ulysses",
        )
        parallelize_qwen3_5_cp(
            model,
            cp_mesh,
            ulysses_degree=ulysses_degree,
            linear_attention_cp_mode=linear_attention_cp_mode,
        )

    if cfg.train.accelerator.ep > 1:
        raise NotImplementedError("Qwen3.5 dense has no experts; set parallel.ep=1.")

    if cfg.train.accelerator.pp > 1:
        raise NotImplementedError(
            "Qwen3.5 PP is wired through ``pipelining_qwen3_5`` — register "
            "it as ``ModelSpec.pipelining_fn`` rather than calling "
            "``parallelize_qwen3_5`` with pp>1."
        )

    _apply_ac(model, cfg)
    _apply_fsdp(model, mesh, cfg)
    if tp_size > 1:
        if isinstance(model, HSDPModule):
            reducer = _make_post_fsdp_tp_replicated_param_reducer(
                model, mesh["tp"],
            )
            if reducer is not None:
                model.hp_post_fsdp_grad_reduce = reducer
                logger.info_rank0("TP post-FSDP reducer attached for Qwen3.5 replicated params.")
        else:
            _register_tp_replicated_param_grad_sum(model, mesh["tp"])
    return model


__all__ = [
    "broadcast_state_dict_from_rank0",
    "parallelize_qwen3_5",
    "parallelize_qwen3_5_cp",
    "parallelize_qwen3_5_inference_tp",
    "parallelize_qwen3_5_tp",
    "pipeline_qwen3_5_for_trainer",
    "qwen3_5_tp_load_transforms",
    "qwen3_5_inference_tp_load_transforms",
    "pipelining_qwen3_5",
]

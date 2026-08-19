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
"""Qwen3-VL-MoE parallelization — self-contained per-model logic.

This file owns Qwen3-VL-MoE's visual-tower handling plus TP / CP / EP / FSDP /
AC / PP wiring. Text-backbone TP / CP / EP reuse the Qwen3.5-MoE layer plans;
VL-specific hooks gather full-sequence visual inputs and restore the original
parallel layouts around visual token and DeepStack injection.
"""
from dataclasses import replace
import logging
from typing import Optional, TYPE_CHECKING

import torch

from hyper_parallel import (
    AsyncContextParallel,
    ContextParallel,
    DTensor,
    HSDPModule,
    fully_shard,
)
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
from hyper_parallel.core.pipeline_parallel import (
    BatchDimSpec, PipelineStage, Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B)
from hyper_parallel.models.qwen3_vl_moe.model import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeStageModule,
)
from hyper_parallel.platform import get_platform
# The VL text backbone shares the Qwen3.5-MoE GQA and packed-expert modules, so
# the same in-layer TP / CP / EP plans and post-FSDP TP reducer are reused here.
from hyper_parallel.models.qwen3_5_moe.parallelize import (  # pylint: disable=C0412
    _chain_grad_reducers,
    _make_post_fsdp_ep_avg_reducer,
    _make_post_fsdp_ep_expert_grad_divider,
    _make_post_fsdp_ep_moe_avg_reducer,
    _make_post_fsdp_tp_reducer,
    _register_ep_replicated_moe_grad_avg,
    _register_tp_replicated_param_grad_sum,
    parallelize_qwen3_5_moe_cp,
    parallelize_qwen3_5_moe_ep,
    parallelize_qwen3_5_moe_tp,
)
from hyper_parallel.trainer.config import get_vision_parallel_config

if TYPE_CHECKING:
    from hyper_parallel.trainer.config import HyperTrainerConfig

logger = logging.getLogger(__name__)
platform = get_platform()

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}

_PP_SCHEDULES = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
}


def _world_size() -> int:
    """Return world size, treating an uninitialized backend as single-rank."""
    try:
        return platform.get_world_size()
    except (RuntimeError, ValueError):
        return 1


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
        return DTensor.from_local(x, mesh, [Replicate()]).redistribute(
            mesh, list(placements),
        )
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


def _resolve_qwen3_vl_moe_pp_fsdp_mesh(mesh, cfg):
    """Return a safe per-stage FSDP mesh, or ``None`` for plain DP fallback."""
    if int(cfg.train.accelerator.dp_replicate or 1) > 1:
        logger.info_rank0(
            "Qwen3-VL-MoE PP+HSDP leaves stages replicated so the trainer can "
            "synchronize plain gradients over the combined DP group."
        )
        return None
    return _resolve_dp_mesh(mesh) if _has_nontrivial_shard_dim(mesh) else None


def _resolve_mp_policy(cfg):
    """Build FSDP mixed-precision policy from the Qwen3-VL-MoE YAML."""
    mp_cfg = cfg.train.mixed_precision
    if not mp_cfg.enabled:
        return None
    output_dtype_str = mp_cfg.output_dtype
    return MixedPrecisionPolicy(
        param_dtype=_DTYPE_MAP.get(mp_cfg.param_dtype),
        reduce_dtype=_DTYPE_MAP.get(mp_cfg.reduce_dtype),
        output_dtype=_DTYPE_MAP.get(output_dtype_str) if output_dtype_str else None,
    )


def _build_fsdp_kwargs(module, dp_mesh, cfg) -> dict:
    """Assemble Qwen3-VL-MoE FSDP kwargs over ``dp_mesh``."""
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
        _extend_visual_replicate_params(module, dp_mesh, cfg, shard_size, replicate_params)
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


def _extend_visual_replicate_params(module, mesh, cfg, shard_size: int, replicate_params) -> None:
    """Keep visual params replicated when ``vision_parallel.dp_shard=1`` is requested."""
    if shard_size <= 1:
        return
    if not _get_model_extra(cfg).get("vl", False):
        return
    if not (hasattr(module, "model") and hasattr(module.model, "visual")):
        return
    if _resolve_vision_dp_shard_size(mesh, cfg) != 1:
        return
    replicate_params.update(module.model.visual.parameters())
    logger.info_rank0(
        "VL visual params routed through replicate_params for encoder-only DP "
        "(vision_parallel.dp_shard=1)."
    )


def _resolve_dp_mesh(mesh):
    try:
        return mesh["fsdp"]
    except (KeyError, TypeError):
        pass
    try:
        return mesh["dp_shard"]
    except (KeyError, TypeError):
        return mesh


def _get_model_extra(cfg) -> dict:
    """Return model.config_overrides as a plain dict."""
    extra = getattr(cfg.model, "config_overrides", None)
    return extra if isinstance(extra, dict) else {}


def _get_vision_parallel_cfg(cfg) -> dict:
    """Return visual Encoder local parallel config when present."""
    return get_vision_parallel_config(cfg.model)


def _mesh_size(mesh) -> int:
    """Return mesh size when available, otherwise treat it as single rank."""
    try:
        return int(mesh.size())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 1


def _resolve_vision_dp_shard_size(mesh, cfg) -> int:
    """Resolve visual Encoder DP shard size from config or global DP."""
    vision_parallel = _get_vision_parallel_cfg(cfg)
    dp_mesh = _resolve_dp_mesh(mesh)
    global_dp_size = _mesh_size(dp_mesh)
    requested = vision_parallel.get("dp_shard", None)
    if requested in (None, "", 0, "0"):
        return global_dp_size

    requested_size = int(requested)
    if requested_size <= 0:
        raise ValueError(f"vision_parallel.dp_shard must be >= 1, got {requested_size}.")
    if requested_size not in (1, global_dp_size):
        raise ValueError(
            "Current qwen3_vl_moe visual Encoder DP supports only "
            f"vision_parallel.dp_shard in {{1, {global_dp_size}}}, got {requested_size}."
        )
    return requested_size


def _resolve_vision_cp_mesh(mesh, cfg):
    """Resolve visual Encoder CP mesh, including explicit dp_shard reuse mode."""
    vision_parallel = _get_vision_parallel_cfg(cfg)
    cp_size = int(vision_parallel.get("cp", 1) or 1)
    if cp_size <= 1:
        return None

    global_cp = int(getattr(cfg.train.accelerator, "cp", 1) or 1)
    if global_cp > 1:
        try:
            cp_mesh = mesh["cp"]
        except (KeyError, TypeError):
            cp_mesh = mesh
        actual_size = _mesh_size(cp_mesh)
        if actual_size != cp_size:
            raise ValueError(
                f"vision_parallel.cp={cp_size} but train.accelerator.cp mesh size is {actual_size}."
            )
        return cp_mesh

    dp_mesh = _resolve_dp_mesh(mesh)
    actual_size = _mesh_size(dp_mesh)
    if actual_size != cp_size:
        raise ValueError(
            "vision_parallel.cp requires either train.accelerator.cp > 1 with a matching "
            f"cp mesh, or a dp_shard mesh of the same size. Requested cp={cp_size}, got "
            f"dp_shard mesh size={actual_size}."
        )
    if not vision_parallel.get("reuse_dp_shard_mesh", False):
        raise ValueError(
            "vision_parallel.cp requested encoder-only CP on the dp_shard mesh while "
            "train.accelerator.cp=1. This path may aggregate attention states across "
            "different DP samples, so it requires explicit opt-in via "
            "model.vision_parallel.reuse_dp_shard_mesh=true."
        )
    logger.warning_rank0(
        "VL vision CP is reusing the dp_shard mesh because train.accelerator.cp=1. "
        "This mode is intended for targeted 2-card validation; ranks may hold "
        "different samples unless vision_parallel.share_samples_across_dp=true."
    )
    return dp_mesh


def _resolve_qwen3_vl_moe_dp_sizes(mesh, cfg) -> tuple[int, int]:
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


def _resolve_qwen3_vl_moe_dp_mesh(mesh, cfg):
    """Resolve the synchronization-complete data-parallel FSDP/HSDP mesh."""
    replicate_size, _ = _resolve_qwen3_vl_moe_dp_sizes(mesh, cfg)
    if replicate_size > 1:
        try:
            return mesh[("dp_replicate", "dp_shard")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Qwen3-VL-MoE HSDP requires a DeviceMesh with ordered "
                "'dp_replicate' and 'dp_shard' dimensions."
            ) from exc
    return _resolve_dp_mesh(mesh)


def _resolve_qwen3_vl_moe_fsdp_mesh(mesh, cfg, cp_size: int):
    """Resolve the text-backbone FSDP mesh, rejecting unsafe DP domains."""
    replicate_size, _ = _resolve_qwen3_vl_moe_dp_sizes(mesh, cfg)
    if replicate_size > 1 and cp_size > 1:
        raise ValueError(
            "Qwen3-VL-MoE non-PP training does not support HSDP with CP: the "
            "current fully_shard mesh cannot express both the outer "
            "dp_replicate synchronization and the CP gradient domain."
        )
    if cp_size > 1:
        try:
            return mesh["loss"]
        except (KeyError, TypeError):
            return mesh["cp"]
    return _resolve_qwen3_vl_moe_dp_mesh(mesh, cfg)


def _resolve_visual_fsdp_mesh(mesh, cfg):
    """Return the data-parallel mesh used to shard the frozen visual tower."""
    if _resolve_vision_dp_shard_size(mesh, cfg) <= 1:
        return None
    replicate_size, shard_size = _resolve_qwen3_vl_moe_dp_sizes(mesh, cfg)
    if replicate_size * shard_size > 1:
        return _resolve_qwen3_vl_moe_dp_mesh(mesh, cfg)
    for dim in ("fsdp", "dp_shard"):
        try:
            return mesh[dim]
        except (KeyError, TypeError):
            continue
    if _is_single_rank_without_parallel_axes(cfg):
        return mesh
    return None


def _apply_vl_visual_cp(model, mesh, cfg) -> None:
    """Apply encoder-only ContextParallel to the visual attention core."""
    visual = _resolve_visual_tower(model, cfg)
    if visual is None:
        return

    cp_mesh = _resolve_vision_cp_mesh(mesh, cfg)
    if cp_mesh is None:
        return

    vision_parallel = _get_vision_parallel_cfg(cfg)
    ulysses_degree = vision_parallel.get("ulysses_degree", 1)
    async_cp = bool(
        vision_parallel.get("async_cp", getattr(cfg.train.accelerator, "async_cp", False))
    )
    cp_cls = AsyncContextParallel if async_cp else ContextParallel
    cp_plan = cp_cls(
        seq_dim=1,
        head_dim=2,
        ulysses_degree=ulysses_degree,
    )

    block_count = 0
    for block in getattr(visual, "blocks", ()):
        sdpa_core = getattr(getattr(block, "attn", None), "sdpa_core", None)
        if sdpa_core is None:
            raise ValueError(
                "vision_parallel.cp requires visual attention modules to expose "
                "an sdpa_core boundary."
            )
        if async_cp:
            missing = [
                name for name in ("q_cp_boundary", "k_cp_boundary", "v_cp_boundary")
                if not hasattr(block.attn, name)
            ]
            if missing:
                raise ValueError(
                    "vision_parallel.async_cp requires visual attention modules to expose "
                    f"CP projection boundaries: missing {missing}."
                )
            cp_plan.apply(
                sdpa_core,
                cp_mesh,
                q_proj=block.attn.q_cp_boundary,
                k_proj=block.attn.k_cp_boundary,
                v_proj=block.attn.v_cp_boundary,
            )
        else:
            cp_plan.apply(sdpa_core, cp_mesh)
        setattr(block.attn, "_context_parallel_enabled", True)
        setattr(block.attn, "_async_context_parallel_enabled", async_cp)
        block_count += 1

    logger.info_rank0(
        "VL vision CP applied: blocks=%d cp=%d ulysses_degree=%s async_cp=%s",
        block_count,
        _mesh_size(cp_mesh),
        str(ulysses_degree),
        str(async_cp),
    )


def _apply_full_sequence_input_cp_gather(module, cp_mesh: DeviceMesh) -> None:
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


def _apply_visual_sequence_roundtrip(module) -> None:
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


def _apply_text_decoder_input_cp_slice(module, cp_mesh: DeviceMesh) -> None:
    """Slice text decoder inputs after the full-sequence attention mask is built."""

    def _post_hook(hook_module, hook_args, output):
        del hook_module, hook_args
        inputs_embeds, position_ids = output
        inputs_embeds, tp_placement = _seq_full(inputs_embeds)
        inputs_embeds = _cp_slice(inputs_embeds, cp_mesh, dim=1)
        inputs_embeds = _seq_restore(inputs_embeds, tp_placement)
        position_ids = _cp_slice(position_ids, cp_mesh, dim=position_ids.ndim - 1)
        return inputs_embeds, position_ids

    module.decoder_input.register_forward_hook(_post_hook)


def _apply_deepstack_sequence_roundtrip(module, cp_mesh: Optional[DeviceMesh] = None) -> None:
    """Gather/restore DeepStack activations around global visual-position updates."""
    placements = []

    def _pre_hook(hook_module, args):
        del hook_module
        hidden_states = args[0]
        hidden_states, tp_placement = _seq_full(hidden_states)
        placements.append(tp_placement)
        if cp_mesh is not None:
            hidden_states = _cp_gather(hidden_states, cp_mesh)
        return (hidden_states,)

    def _post_hook(hook_module, hook_args, output):
        del hook_module, hook_args
        if cp_mesh is not None:
            output = _cp_slice(output, cp_mesh, dim=1)
        tp_placement = placements.pop() if placements else None
        return _seq_restore(output, tp_placement)

    module.deepstack_input.register_forward_pre_hook(_pre_hook)
    module.deepstack_output.register_forward_hook(_post_hook)


def _apply_tp_attention_sequence_gather(model, tp_mesh: DeviceMesh) -> None:
    """Gather residual local sequence shards before VL text attention under TP."""

    def _maybe_gather(hidden_states, target_seq_len: Optional[int]):
        if target_seq_len is None:
            return hidden_states
        if isinstance(hidden_states, DTensor):
            if tuple(hidden_states.placements) == (Replicate(),):
                return hidden_states.to_local()
            return hidden_states.redistribute(tp_mesh, [Replicate()]).to_local()
        if not platform.is_tensor(hidden_states):
            return hidden_states
        seq_len = hidden_states.shape[1]
        if seq_len == target_seq_len:
            return hidden_states
        if seq_len * tp_mesh.size() == target_seq_len:
            return DTensor.from_local(hidden_states, tp_mesh, [Shard(1)]).redistribute(
                tp_mesh, [Replicate()],
            ).to_local()
        return hidden_states

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        args = list(args)
        kwargs = dict(kwargs)
        position_ids = kwargs.get("position_ids")
        target_seq_len = int(position_ids.shape[-1]) if position_ids is not None else None
        if args:
            args[0] = _maybe_gather(args[0], target_seq_len)
        elif kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = _maybe_gather(kwargs["hidden_states"], target_seq_len)
        return tuple(args), kwargs

    for block in model.layers:
        block.self_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def _apply_tp_residual_output_layout(model, tp_mesh: DeviceMesh) -> None:
    """Restore VL residual-branch outputs to the decoder-layer input sequence layout."""

    def _capture_layout(hook_module, args, kwargs):
        del kwargs
        hidden_states = args[0] if args else None
        layout = (
            hidden_states.device_mesh,
            tuple(hidden_states.placements),
            tuple(hidden_states.shape),
            hidden_states.dtype,
        ) if isinstance(hidden_states, DTensor) else None
        hook_module._hp_tp_sequence_layout = layout  # pylint: disable=protected-access
        hook_module.self_attn._hp_tp_sequence_layout = layout  # pylint: disable=protected-access
        hook_module.mlp._hp_tp_sequence_layout = layout  # pylint: disable=protected-access

    def _cast_like_layout(output, dtype):
        if dtype is None or not hasattr(output, "dtype") or output.dtype == dtype:
            return output
        return output.to(dtype)

    def _restore_tensor(output, mesh, placements, global_shape, dtype):
        if isinstance(output, DTensor):
            if tuple(output.placements) == tuple(placements):
                return _cast_like_layout(output, dtype)
            local = output.to_local()
            if len(output.shape) > 1 and output.shape[1] * tp_mesh.size() == global_shape[1]:
                restored = DTensor.from_local(local, mesh, [Shard(1)]).redistribute(
                    mesh,
                    placements,
                )
                return _cast_like_layout(restored, dtype)
            return _cast_like_layout(output.redistribute(mesh, placements), dtype)
        if not platform.is_tensor(output):
            return output
        if len(output.shape) > 1 and output.shape[1] == global_shape[1]:
            restored = DTensor.from_local(output, mesh, [Replicate()]).redistribute(
                mesh,
                placements,
            )
            return _cast_like_layout(restored, dtype)
        if len(output.shape) > 1 and output.shape[1] * tp_mesh.size() == global_shape[1]:
            restored = DTensor.from_local(output, mesh, [Shard(1)]).redistribute(
                mesh,
                placements,
            )
            return _cast_like_layout(restored, dtype)
        return _cast_like_layout(output, dtype)

    def _restore_layout(hook_module, hook_args, output):
        del hook_args
        layout = getattr(hook_module, "_hp_tp_sequence_layout", None)
        if layout is None:
            return output
        mesh, placements, global_shape, dtype = layout
        if isinstance(output, tuple):
            return (_restore_tensor(output[0], mesh, placements, global_shape, dtype), *output[1:])
        return _restore_tensor(output, mesh, placements, global_shape, dtype)

    for block in model.layers:
        block.register_forward_pre_hook(_capture_layout, with_kwargs=True)
        block.self_attn.register_forward_hook(_restore_layout)
        block.mlp.register_forward_hook(_restore_layout)


def _apply_tp_mlp_sequence_gather(model, tp_mesh: DeviceMesh) -> None:
    """Gather VL sparse-MoE inputs to full sequence before TP partial reduction."""

    def _gather_input(hook_module, args):
        layout = getattr(hook_module, "_hp_tp_sequence_layout", None)
        if layout is None or not args:
            return args
        hidden_states = args[0]
        _, _, global_shape, _ = layout
        if isinstance(hidden_states, DTensor):
            if tuple(hidden_states.placements) == (Replicate(),):
                return (hidden_states.to_local(), *args[1:])
            return (
                hidden_states.redistribute(tp_mesh, [Replicate()]).to_local(),
                *args[1:],
            )
        if not platform.is_tensor(hidden_states):
            return args
        if len(hidden_states.shape) > 1 and hidden_states.shape[1] * tp_mesh.size() == global_shape[1]:
            full = DTensor.from_local(hidden_states, tp_mesh, [Shard(1)]).redistribute(
                tp_mesh,
                [Replicate()],
            ).to_local()
            return (full, *args[1:])
        return args

    for block in model.layers:
        block.mlp.register_forward_pre_hook(_gather_input)


def _apply_tp_logits_gather(model, tp_mesh: DeviceMesh) -> None:
    """Gather TP sequence-sharded logits before the in-model shifted CE loss."""

    def _gather_logits(logits, labels):
        if labels is None:
            return logits
        target_seq_len = labels.shape[-1]
        if isinstance(logits, DTensor):
            if tuple(logits.placements) == (Replicate(),):
                return logits.to_local()
            return logits.redistribute(tp_mesh, [Replicate()]).to_local()
        if (
            platform.is_tensor(logits)
            and logits.ndim > 1
            and logits.shape[1] * tp_mesh.size() == target_seq_len
        ):
            return DTensor.from_local(logits, tp_mesh, [Shard(1)]).redistribute(
                tp_mesh,
                [Replicate()],
            ).to_local()
        return logits

    model._hp_tp_logits_gather = _gather_logits  # pylint: disable=protected-access


def _apply_tp_decoder_sequence_layout(model, tp_mesh: DeviceMesh) -> None:
    """Keep VL decoder-layer residuals in TP sequence-sharded DTensor layout."""

    def _target_seq_len(kwargs) -> Optional[int]:
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            return int(position_ids.shape[-1])
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask.shape[-1])
        return None

    def _to_sequence_shard(hidden_states, target_seq_len: Optional[int]):
        if isinstance(hidden_states, DTensor):
            if tuple(hidden_states.placements) == (Shard(1),):
                return hidden_states
            return hidden_states.redistribute(tp_mesh, [Shard(1)])
        if target_seq_len is None:
            return hidden_states
        if not platform.is_tensor(hidden_states):
            return hidden_states
        seq_len = hidden_states.shape[1]
        if seq_len == target_seq_len:
            return DTensor.from_local(hidden_states, tp_mesh, [Replicate()]).redistribute(
                tp_mesh, [Shard(1)],
            )
        if seq_len * tp_mesh.size() == target_seq_len:
            return DTensor.from_local(hidden_states, tp_mesh, [Shard(1)])
        return hidden_states

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        args = list(args)
        kwargs = dict(kwargs)
        target_seq_len = _target_seq_len(kwargs)
        if args:
            args[0] = _to_sequence_shard(args[0], target_seq_len)
        elif kwargs.get("hidden_states") is not None:
            kwargs["hidden_states"] = _to_sequence_shard(kwargs["hidden_states"], target_seq_len)
        return tuple(args), kwargs

    for block in model.layers:
        block.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def _apply_tp_checkpoint_local_sequence_shard(model, tp_mesh: DeviceMesh) -> None:
    """Keep TP checkpoint-wrapper inputs as local sequence shards.

    The wrapped decoder layer still converts its input to a DTensor Shard(1)
    inside the checkpointed function. The wrapper boundary itself must see a
    plain tensor because torch checkpoint inspects input devices before module
    pre-hooks run.
    """

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


def _resolve_visual_tower(model, cfg):
    """Return the VL visual tower when this config is a multimodal model."""
    extra = cfg.model.config_overrides
    if not (isinstance(extra, dict) and extra.get("vl", False)):
        return None
    if not (hasattr(model, "model") and hasattr(model.model, "visual")):
        return None
    return model.model.visual


def _wrap_visual_unit(module, fsdp_kwargs) -> None:
    """FSDP-wrap one visual module."""
    if module is None:
        return
    fully_shard(module, **fsdp_kwargs)


def _wrap_visual_children(visual, fsdp_kwargs) -> None:
    """Wrap the visual tower's block and merger children."""
    for block in getattr(visual, "blocks", ()):
        _wrap_visual_unit(block, fsdp_kwargs)
    _wrap_visual_unit(getattr(visual, "merger", None), fsdp_kwargs)
    for merger in getattr(visual, "deepstack_merger_list", ()):
        _wrap_visual_unit(merger, fsdp_kwargs)


def _apply_vl_visual_tower(model, mesh, cfg) -> None:
    """FSDP-wrap the VL visual tower as its own uniform group.

    Frozen visual params are cast after checkpoint load by the trainer, not
    here while parameters may still be on meta. This helper only isolates the
    visual blocks / merger / root from the trainable text FSDP groups.
    """
    visual = _resolve_visual_tower(model, cfg)
    if visual is None:
        return

    frozen_visual = all(not param.requires_grad for param in visual.parameters())
    deterministic = bool(cfg.train.debug.deterministic)
    if deterministic and frozen_visual:
        logger.info_rank0(
            "Deterministic Qwen3-VL-MoE run keeps the frozen visual tower replicated."
        )
        return
    single_frozen_visual = _is_single_rank_without_parallel_axes(cfg) and frozen_visual
    dp_mesh = _resolve_visual_fsdp_mesh(mesh, cfg)
    if dp_mesh is None:
        if _resolve_vision_dp_shard_size(mesh, cfg) <= 1:
            logger.info_rank0(
                "VL visual tower kept replicated under root FSDP "
                "(vision_parallel.dp_shard=1)."
            )
        else:
            logger.info_rank0(
                "Qwen3-VL-MoE visual tower is replicated for pure model-parallel axes."
            )
        return
    fsdp_kwargs = _build_fsdp_kwargs(visual, dp_mesh, cfg)
    if single_frozen_visual:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
        logger.info_rank0("Single-rank frozen VL visual tower uses CPU-offloaded FSDP.")
    mp_policy = fsdp_kwargs.get("mp_policy")
    if mp_policy is not None:
        # Keep the frozen visual tower's activation flow aligned with the
        # single-card path. The text backbone may request fp32 FSDP outputs for
        # residual math; vision blocks should not inherit that cast before the
        # merger.
        mp_policy.output_dtype = None
    _wrap_visual_children(visual, fsdp_kwargs)
    _wrap_visual_unit(visual, fsdp_kwargs)
    logger.info_rank0("VL visual tower wrapped (per-block + merger + root)")


def _should_skip_single_rank_fsdp(cfg) -> bool:
    """Return True when a pure single-rank run should keep plain modules."""
    mp_cfg = cfg.train.mixed_precision
    has_output_dtype = bool(mp_cfg.output_dtype)
    param_dtype = mp_cfg.param_dtype
    loss_aggregation = cfg.train.optimizer.loss_aggregation
    needs_rank_average_mp_wrap = (
        loss_aggregation == "rank_average"
        and mp_cfg.enabled
        and param_dtype not in ("float32", "fp32")
    )
    return (
        _is_single_rank_without_parallel_axes(cfg)
        and not has_output_dtype
        and not needs_rank_average_mp_wrap
    )


def _is_single_rank_without_parallel_axes(cfg) -> bool:
    """Return True for a single process with no requested model-parallel axes."""
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
    return single_rank and not has_parallel_axis


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
    logger.info_rank0("AC applied to %d Qwen3-VL-MoE text layers (mode=%s)",
                      len(layers), ac_mode)


def _apply_deterministic_moe_sort(model, cfg) -> None:
    """Use stable expert-token ordering for deterministic TP VL-MoE runs."""
    deterministic = bool(cfg.train.debug.deterministic)
    tp_size = int(cfg.train.accelerator.tp or 1)
    if not deterministic or tp_size <= 1:
        return
    for block in getattr(model, "layers", ()):
        experts = getattr(getattr(block, "mlp", None), "experts", None)
        if experts is not None:
            experts._hp_moe_stable_sort = True  # pylint: disable=protected-access


def _apply_fsdp(model, mesh, cfg) -> None:
    """Per-text-layer + root FSDP wrap. Visual tower is wrapped separately by
    ``_apply_vl_visual_tower``. At dp_size==1 we still go through fully_shard
    so the gradient reduction code path matches dp_size>1.

    Under CP the replicated-weight gradients are partial sums over each rank's
    sequence slice, so FSDP reduces them over the ``"loss"`` mesh
    (``dp_shard × cp``, or just ``cp`` when ``dp_shard == 1``). Pure TP / EP
    (``dp_size == 1`` with ``tp``/``ep > 1``) has no DP axis to shard over and
    the params are already TP/EP-sharded or grad-reduced by their own hooks, so
    the FSDP wrap is skipped.
    """
    if _should_skip_single_rank_fsdp(cfg):
        logger.info_rank0(
            "Single-rank Qwen3-VL-MoE run has no parallel axes; skipping FSDP wrap. "
            "Post-load param_dtype casting handles mixed-precision storage."
        )
        return

    replicate_size, shard_size = _resolve_qwen3_vl_moe_dp_sizes(mesh, cfg)
    dp_size = replicate_size * shard_size
    cp_size = int(cfg.train.accelerator.cp or 1)
    if cp_size <= 1:
        other_parallel = max(
            int(cfg.train.accelerator.tp or 1),
            int(cfg.train.accelerator.ep or 1),
        )
        has_model_parallel = other_parallel > 1
        if dp_size <= 1 and has_model_parallel and not _has_shard_dim(mesh):
            # A size-1 ``dp_shard`` injected by a low-precision run IS wrapped —
            # that wrap carries the MixedPrecisionPolicy.
            logger.info_rank0("dp_size==1 with tp/ep>1; skipping FSDP wrap.")
            return
    dp_mesh = _resolve_qwen3_vl_moe_fsdp_mesh(mesh, cfg, cp_size)
    fsdp_kwargs = _build_fsdp_kwargs(model, dp_mesh, cfg)
    layer_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)

    if not hasattr(model, "layers"):
        logger.warning(
            "Qwen3VLMoeForCausalLM has no ``.layers`` — root-only FSDP wrap."
        )
        fully_shard(model, **fsdp_kwargs)
        return

    layers = list(model.layers)
    for layer in layers:
        fully_shard(layer, **layer_fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    if cp_size > 1:
        model.set_reduce_op_type("sum")
        model.hp_token_loss_scale_size = 1
        logger.info_rank0("CP FSDP reduce op set to SUM for Qwen3-VL-MoE token-weighted loss.")
    logger.info_rank0("FSDP applied to Qwen3-VL-MoE: %d text layers + root",
                      len(layers))

def _force_sdpa_attn(model) -> None:
    """Route every text self-attention through the model-level SDPA hook path.

    ``Qwen3VLMoeTextAttention.forward`` only delegates to the shared
    ``GroupQueryAttention.forward`` when ``_attn_implementation == "sdpa"``; its
    eager / fa2 paths bypass the module-level SDPA target. TP and CP therefore
    require SDPA, so force it on each decoder block's attention.
    """
    text_model = model.model.language_model if hasattr(model, "model") else model
    text_model.config._attn_implementation = "sdpa"  # pylint: disable=protected-access
    for block in model.layers:
        block.self_attn._attn_implementation = "sdpa"  # pylint: disable=protected-access


def _validate_vl_tp_cp_heads(model, tp_size: int, cp_size: int) -> None:
    """Validate attention-head divisibility for composed TP and CP."""
    if tp_size > 1 and cp_size > 1:
        text_config = getattr(model.config, "text_config", model.config)
        heads = int(text_config.num_attention_heads)
        if heads and (heads // tp_size) % cp_size != 0:
            raise ValueError(
                f"TP+CP requires (num_attention_heads / tp) divisible by cp "
                f"for the Ulysses all-to-all; got {heads} heads / tp={tp_size}"
                f" = {heads // tp_size}, cp={cp_size}."
            )


def _apply_tp_cp_ep(model, mesh, cfg) -> None:
    """Apply in-layer TP / CP / EP to the VL text backbone by delegating to the
    shared Qwen3.5-MoE styles (the backbone is the same GQA + packed-expert MoE,
    nested under ``model.model.language_model.*``).

    TP and CP both require the SDPA attention path (forced here) and the
    per-block ``layer_type`` the model sets. EP additionally needs the VL MoE's
    ``ep_mesh`` forward branch. The frozen visual tower is untouched — these
    styles only target the text decoder layers, embed / norm and lm_head.
    """
    tp_size = int(cfg.train.accelerator.tp or 1)
    cp_size = int(cfg.train.accelerator.cp or 1)
    ep_size = int(cfg.train.accelerator.ep or 1)
    _validate_vl_tp_cp_heads(model, tp_size, cp_size)
    if tp_size > 1 or cp_size > 1:
        _force_sdpa_attn(model)
    if tp_size > 1:
        parallelize_qwen3_5_moe_tp(
            model,
            mesh["tp"],
            register_grad_hooks=False,
            replicate_attention_out_proj=True,
        )
        _apply_tp_decoder_sequence_layout(model, mesh["tp"])
        _apply_tp_attention_sequence_gather(model, mesh["tp"])
        _apply_tp_residual_output_layout(model, mesh["tp"])
        _apply_tp_mlp_sequence_gather(model, mesh["tp"])
        _apply_tp_logits_gather(model, mesh["tp"])
        model.hp_loss_tp_scale_size = 1
        model.hp_rank_average_loss_scale_size = tp_size
        model._hp_moe_tp_gate_grad_avg = False  # pylint: disable=protected-access
        _apply_visual_sequence_roundtrip(model.model)
    if cp_size > 1:
        parallelize_qwen3_5_moe_cp(model, mesh["cp"])
        # The trainer pre-shards the data sequence across CP; the VL visual /
        # DeepStack injection is whole-sequence. Gather at the composite boundary,
        # then slice decoder inputs after mask construction.
        composite = model.model
        _apply_full_sequence_input_cp_gather(composite, mesh["cp"])
        _apply_text_decoder_input_cp_slice(composite.language_model, mesh["cp"])
    if tp_size > 1 or cp_size > 1:
        cp_mesh = mesh["cp"] if cp_size > 1 else None
        _apply_deepstack_sequence_roundtrip(model.model.language_model, cp_mesh)
    if ep_size > 1:
        parallelize_qwen3_5_moe_ep(
            model,
            mesh["ep"],
            cp_mesh=mesh["cp"] if cp_size > 1 else None,
            register_grad_hooks=False,
            set_loss_scale=False,
            stable_sort=tp_size <= 1,
            fp32_routing=True,
        )


def parallelize_qwen3_vl_moe(
    model: Qwen3VLMoeForConditionalGeneration,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> Qwen3VLMoeForConditionalGeneration:
    """Apply VL visual handling + text TP/CP/EP + AC + FSDP.

    Order:
      1. VL visual tower (dtype cast + per-block FSDP) — must run before
         text-side parallelism so the visual is its own FSDP unit.
      2. In-layer TP / CP / EP on the text backbone (delegated to qwen3_5_moe).
      3. AC on text layers (checkpoint_wrapper inside the FSDP boundary).
      4. FSDP on text layers + root.
    PP (pp>1) is routed through ``pipeline_qwen3_vl_moe_for_trainer`` (the
    ModelSpec ``pipelining_fn``), which bypasses this ``parallelize_fn``.

    Args:
        model: Qwen3-VL-MoE model to parallelize.
        mesh: Device mesh containing the configured parallel dimensions.
        cfg: Trainer configuration controlling parallel degrees and policies.

    Returns:
        The parallelized model.
    """
    _resolve_qwen3_vl_moe_fsdp_mesh(
        mesh, cfg, int(cfg.train.accelerator.cp or 1),
    )
    _apply_vl_visual_cp(model, mesh, cfg)
    if not _should_skip_single_rank_fsdp(cfg):
        _apply_vl_visual_tower(model, mesh, cfg)
    _apply_deterministic_moe_sort(model, cfg)
    _apply_tp_cp_ep(model, mesh, cfg)
    _apply_ac(model, cfg)
    if int(cfg.train.accelerator.tp or 1) > 1:
        _apply_tp_checkpoint_local_sequence_shard(model, mesh["tp"])
    _apply_fsdp(model, mesh, cfg)
    # Under TP+FSDP the per-param SequenceParallel grad hook cannot fire (FSDP
    # owns the grad reduction), so drive an explicit post-FSDP TP all-reduce for
    # the plain TP-replicated weights (router gate, packed experts). Needed
    # whenever FSDP actually wrapped the model alongside TP — including FSDP
    # over the ``loss`` (dp_shard × cp) mesh under CP and the size-1 dp_shard
    # wrap a low-precision run injects.
    tp_size = int(cfg.train.accelerator.tp or 1)
    ep_size = int(cfg.train.accelerator.ep or 1)
    reducers = []
    if tp_size > 1:
        model.hp_loss_tp_scale_size = 1
        model.hp_rank_average_loss_scale_size = tp_size
        model._hp_moe_tp_gate_grad_avg = False  # pylint: disable=protected-access
        if isinstance(model, HSDPModule):
            reducers.append(_make_post_fsdp_tp_reducer(model, mesh["tp"]))
        else:
            _register_tp_replicated_param_grad_sum(model, mesh["tp"])
    if ep_size > 1:
        model.hp_loss_ep_scale_size = 1
        if isinstance(model, HSDPModule):
            if tp_size > 1:
                reducers.append(_make_post_fsdp_ep_avg_reducer(model, mesh["ep"]))
                reducers.append(_make_post_fsdp_ep_expert_grad_divider(model, mesh["ep"]))
            else:
                reducers.append(_make_post_fsdp_ep_moe_avg_reducer(model, mesh["ep"]))
        else:
            _register_ep_replicated_moe_grad_avg(model, mesh["ep"])
    post_fsdp_grad_reduce = _chain_grad_reducers(*reducers)
    if post_fsdp_grad_reduce is not None:
        model.hp_post_fsdp_grad_reduce = post_fsdp_grad_reduce
    return model


def _fsdp_wrap_stage_vl(stage_module, dp_mesh, cfg) -> None:
    """Wrap a VL pipeline stage's trainable text children as FSDP units + root.

    Each decoder layer plus the ``embed_tokens`` / ``norm`` / ``lm_head`` this
    stage owns becomes its own FSDP unit, then the stage module itself is
    wrapped as the *root* unit. The root wrap is required, not cosmetic — the
    schedule's ``FSDP_REDUCE_GRAD``
    finalization drains the fused-reduction state once per step; without it the
    independently-wrapped boundary units (the vocab-sized embed / lm_head) leave
    their reduce-scatter pending and apply ``sharded_param.grad`` only every other
    step. ``reshard_after_forward`` is forced off so the 1F1B / GPipe schedule
    owns unshard / reshard across the micro-batches of a step.

    The frozen visual tower (stage 0) is already wrapped as its own FSDP unit(s)
    by :func:`_apply_vl_visual_tower` before the split, so this root wrap does not
    re-absorb it; its frozen params carry no gradient and no-op every reduce site.
    """
    fsdp_kwargs = _build_fsdp_kwargs(stage_module, dp_mesh, cfg)
    fsdp_kwargs["reshard_after_forward"] = False
    boundary_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    root_fsdp_kwargs = _without_forward_input_cast(fsdp_kwargs)
    for layer in stage_module.layers:
        fully_shard(layer, **fsdp_kwargs)
    for sub in (stage_module.embed_tokens, stage_module.norm, stage_module.lm_head):
        if sub is not None:
            fully_shard(sub, **boundary_fsdp_kwargs)
    fully_shard(stage_module, **root_fsdp_kwargs)
    logger.info_rank0(
        "PP+FSDP: VL stage wrapped %d layer(s) + embed/norm/lm_head + root (dp=%d)",
        len(stage_module.layers), dp_mesh.size() if hasattr(dp_mesh, "size") else 1,
    )


def _build_qwen3_vl_moe_pp_schedule(
    stages,
    micro_batch_num: int,
    schedule_name: str,
    vpp: int,
    kwargs_batch_dim: dict,
):
    """Build the configured Qwen3-VL-MoE pipeline schedule."""
    if vpp > 1:
        return ScheduleInterleaved1F1B(
            stages, micro_batch_num, kwargs_batch_dim=kwargs_batch_dim,
        )
    return _resolve_pp_schedule(schedule_name)(
        stages, micro_batch_num, kwargs_batch_dim=kwargs_batch_dim,
    )


def pipelining_qwen3_vl_moe(
    model, pp_mesh: DeviceMesh, *, device=None, micro_batch_num: int = 1,
    fsdp_mesh=None, cfg=None, schedule_name: str = "1f1b", vpp: int = 1,
):
    """Split a Qwen3-VL-MoE model across pipeline stages (configurable schedule).

    Global stage 0 owns the visual tower + ``embed_tokens`` and runs the full
    visual injection + 3D-mrope position-id computation; the last global stage
    owns ``norm`` + ``lm_head``. Decoder layers are partitioned into contiguous
    slabs across ``pp_mesh.size() * vpp`` global stages — evenly by default,
    or per ``train.accelerator.pp_layer_split``. The base cross-stage
    activations are ``(hidden_states, position_ids)``; while DeepStack
    injection layers remain downstream of a boundary, the stage additionally
    relays ``(visual_pos_masks, *remaining_features)`` (see
    :class:`Qwen3VLMoeStageModule`), so the split need not keep the injection
    layers on stage 0.

    Micro-batching: the schedule splits every per-step input along dim 0 into
    ``micro_batch_num`` chunks. ``pixel_values`` (flat ``[sum_patches, dim]``)
    and ``image_grid_thw`` (``[num_images, 3]``) are split the same way, which is
    correct when every sample contributes an equal, dim-0-aligned number of image
    patches — the uniform layout the trainer's grad-accum concat already requires.
    A ragged split (samples with differing image / patch counts) would route
    patches to the wrong micro-batch; the stage-0 forward fails fast on that
    mismatch (``image_grid_thw.prod(-1).sum() != pixel_values.shape[0]``) rather
    than corrupting silently. Ragged per-sample image layouts would need a custom
    VL splitter and are not handled.

    When ``fsdp_mesh`` is given (PP composed with FSDP), each stage's trainable
    text children (decoder layers + the ``embed_tokens`` / ``norm`` / ``lm_head``
    it owns) are wrapped as FSDP units via :func:`_fsdp_wrap_stage_vl`. The frozen
    visual tower is wrapped separately by :func:`_apply_vl_visual_tower` *before*
    the split (in :func:`pipeline_qwen3_vl_moe_for_trainer`), so stage 0's visual
    is an FSDP unit whose grad-less params no-op every reduce.

    ``vpp`` (virtual-pipeline degree) is the number of non-contiguous stage
    chunks each rank owns (``pp_rank, pp_rank+P, ...``). ``vpp == 1`` (default)
    is one contiguous stage per rank driven by ``schedule_name``; ``vpp > 1``
    drives the chunks with ``ScheduleInterleaved1F1B``.

    Returns:
        ``(schedule, [PipelineStage])`` — the stages this rank owns (length ``vpp``).
    """
    num_ranks = pp_mesh.size()
    if num_ranks < 2:
        raise ValueError(f"pipelining_qwen3_vl_moe requires pp>=2; got {num_ranks}.")
    if micro_batch_num < 1:
        raise ValueError(f"micro_batch_num must be >= 1; got {micro_batch_num}.")
    vpp = max(1, int(vpp))
    num_stages = num_ranks * vpp  # total interleaved global stages (P * V)
    composite = model.model
    backbone = composite.language_model
    layers = list(model.layers)
    n_layers = len(layers)
    deepstack_len = len(getattr(composite.visual, "deepstack_visual_indexes", []) or [])

    layer_split = cfg.train.accelerator.pp_layer_split if cfg is not None else None
    counts = _resolve_stage_layer_counts(n_layers, num_stages, layer_split)
    starts = [sum(counts[:i]) for i in range(num_stages)]
    pp_rank = pp_mesh.get_local_rank()

    # VPP: this rank owns ``vpp`` non-contiguous global stages
    # ``pp_rank, pp_rank+num_ranks, ...`` (loop / round-robin). ``vpp == 1``
    # collapses to a single contiguous stage.
    stages = []
    for v in range(vpp):
        stage_index = pp_rank + v * num_ranks
        start = starts[stage_index]
        end = start + counts[stage_index]
        is_first = stage_index == 0
        is_last = stage_index == num_stages - 1
        stage_module = Qwen3VLMoeStageModule(
            layers=layers[start:end],
            layer_start=start,
            config=model.config if is_first else None,
            visual=composite.visual if is_first else None,
            embed_tokens=backbone.embed_tokens if is_first else None,
            norm=backbone.norm if is_last else None,
            lm_head=model.lm_head if is_last else None,
            deepstack_len=deepstack_len,
            # Every stage builds its own attention mask, so each needs the attn
            # impl (the SDPA/eager path builds a 4D mask; fa2 keeps the raw 2D
            # mask).
            attn_impl=getattr(backbone.config, "_attn_implementation", "eager"),
        )
        if device is not None:
            stage_module = stage_module.to(device=device)
        if fsdp_mesh is not None:
            _fsdp_wrap_stage_vl(stage_module, fsdp_mesh, cfg)
        stages.append(
            PipelineStage(stage_module, stage_index, num_stages,
                          device=device, mesh=pp_mesh))

    kwargs_batch_dim = {
        "targets": BatchDimSpec(0),
        "attention_mask": BatchDimSpec(0),
        "pixel_values": BatchDimSpec(0),
        "image_grid_thw": BatchDimSpec(0),
        "mm_token_type_ids": BatchDimSpec(0),
    }
    # ``vpp > 1`` requires interleaved 1F1B (the only multi-stage-per-rank
    # schedule); ``pp_schedule`` (1f1b / gpipe) selects single-stage schedules.
    schedule = _build_qwen3_vl_moe_pp_schedule(
        stages, micro_batch_num, schedule_name, vpp, kwargs_batch_dim,
    )
    logger.info_rank0(
        "PP(%s) applied to Qwen3-VL-MoE: pp=%d vpp=%d stages=%d counts=%s, "
        "this rank owns %s",
        schedule_name, num_ranks, vpp, num_stages, counts,
        [s.stage_index for s in stages],
    )
    return schedule, stages


def pipeline_qwen3_vl_moe_for_trainer(
    model: Qwen3VLMoeForConditionalGeneration,
    mesh: DeviceMesh,
    cfg: "HyperTrainerConfig",
) -> tuple[object, list[PipelineStage]]:
    """Split Qwen3-VL-MoE into pipeline stages for the trainer.

    Args:
        model: Full Qwen3-VL-MoE model before pipeline partitioning.
        mesh: Device mesh containing the pipeline and data-parallel dimensions.
        cfg: Trainer configuration controlling the pipeline schedule.

    Returns:
        The pipeline schedule and the stages owned by the current rank.
    """
    try:
        pp_mesh = mesh["pp"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parallel.pp>1 requires a DeviceMesh with a 'pp' dim.") from exc
    if cfg.train.accelerator.tp > 1 or cfg.train.accelerator.ep > 1:
        raise NotImplementedError(
            "Qwen3-VL-MoE PP composes with FSDP (dp_shard / dp_replicate) only; "
            "composing PP with TP or EP is not yet wired. Set tp=1, ep=1."
        )
    micro_batch_num = int(cfg.train.accelerator.pp_micro_batch_num)
    schedule_name = cfg.train.accelerator.pp_schedule or "1f1b"
    vpp = int(cfg.train.accelerator.pp_vpp or 1)
    device = next(model.parameters()).device
    # Resolve only when a non-trivial shard axis exists. Pure dp_replicate
    # materializes a size-1 dp_shard to form a 2-D non-PP HSDP mesh, but PP
    # stages must stay plain so the trainer's PP DP reducer synchronizes them.
    fsdp_mesh = _resolve_qwen3_vl_moe_pp_fsdp_mesh(mesh, cfg)
    if fsdp_mesh is not None:
        # Cast + per-block FSDP-wrap the frozen visual tower exactly as the non-PP
        # multi-card path does (``_apply_vl_visual_tower``), so stage 0's visual is
        # an FSDP DTensor unit (frozen params no-op every reduce) and the whole
        # stage is uniformly sharded — the trainer's single-param ``DTensor``
        # heuristic would otherwise mis-route the dp grad reduction if the
        # first-registered (visual) param were a plain tensor. Runs on the
        # still-meta model; the trainer materializes + weight-loads afterwards.
        _apply_vl_visual_tower(model, mesh, cfg)
    return pipelining_qwen3_vl_moe(
        model, pp_mesh, device=device, micro_batch_num=micro_batch_num,
        fsdp_mesh=fsdp_mesh, cfg=cfg, schedule_name=schedule_name, vpp=vpp,
    )


__all__ = ["parallelize_qwen3_vl_moe", "pipeline_qwen3_vl_moe_for_trainer"]

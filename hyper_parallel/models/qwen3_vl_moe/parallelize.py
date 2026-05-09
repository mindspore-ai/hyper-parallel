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

There
is no shared "default_parallelize" template — this file owns Qwen3-VL-MoE's
full visual-tower handling + AC / FSDP / Prefetch pipeline. TP and EP are
explicitly rejected (the generic ``ParallelStyle``-based path produces
incorrect numerics for grouped experts and visual blocks).
"""
import logging

import torch

from hyper_parallel import fully_shard, get_platform
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy

logger = logging.getLogger(__name__)

def _resolve_dp_mesh(mesh):
    try:
        return mesh["fsdp"]
    except (KeyError, TypeError):
        pass
    try:
        return mesh["dp_shard"]
    except (KeyError, TypeError):
        return mesh

def _resolve_mp_policy(cfg):
    """Build MixedPrecisionPolicy from cfg.train.mixed_precision, or None."""
    mp_cfg = getattr(cfg.train, "mixed_precision", None)
    if mp_cfg is None or not getattr(mp_cfg, "enabled", False):
        return None

    dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    param_dtype = dtype_map.get(getattr(mp_cfg, "param_dtype", "bfloat16"))
    reduce_dtype = dtype_map.get(getattr(mp_cfg, "reduce_dtype", "float32"))
    output_dtype_str = getattr(mp_cfg, "output_dtype", None)
    output_dtype = dtype_map.get(output_dtype_str) if output_dtype_str else None
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )

def _apply_vl_visual_tower(model, mesh, cfg) -> None:
    """VL-specific: cast frozen visual params to ``param_dtype`` and per-block
    FSDP-wrap the vision tower as its own bf16-uniform group.

    Two distinct concerns:

    1. **Dtype cast (world-size invariant)** — without it, single-card runs
       keep visual at fp32 master while multi-card runs cast to bf16,
       producing per-step ULP drift that triggers MoE router top-k flips by
       step 3 (1c-vs-4c diff ~0.17 at step 3 vs <1e-6 elsewhere).

    2. **Per-block FSDP wrap (world_size > 1 only)** — each vision block
       and the merger as its own FSDP unit, then visual root. One all-gather
       per block keeps the per-call rounding pattern stable.

    Note: floating-point buffers (e.g. visual rotary ``inv_freq``) must stay
    in fp32. Casting to bf16 truncates ULP precision and cascades into a
    sizable norm-diff at the vision merger output.
    """
    extra = getattr(cfg.model, "config_overrides", None)
    is_vl = isinstance(extra, dict) and extra.get("vl", False)
    if not is_vl:
        return
    if not (hasattr(model, "model") and hasattr(model.model, "visual")):
        return

    mp_cfg = getattr(cfg.train, "mixed_precision", None)
    target_dtype = None
    if mp_cfg is not None and getattr(mp_cfg, "enabled", False):
        target_dtype = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
        }.get(getattr(mp_cfg, "param_dtype", "bfloat16"), torch.bfloat16)

    freeze_modules = getattr(cfg.model, "freeze_modules", None) or []
    wants_visual_freeze = any("visual" in p for p in freeze_modules)
    if target_dtype is None or not wants_visual_freeze:
        return

    visual = model.model.visual
    for p in visual.parameters():
        if p.data.dtype != target_dtype:
            p.data = p.data.to(target_dtype)

    if get_platform().get_world_size() <= 1:
        return

    dp_mesh = _resolve_dp_mesh(mesh)
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": getattr(cfg.train.accelerator, "reshard_after_forward", True),
        "comm_fusion": getattr(cfg.train.accelerator, "comm_fusion", True),
    }
    if hasattr(visual, "blocks"):
        for blk in visual.blocks:
            fully_shard(blk, **fsdp_kwargs)
    fully_shard(visual, **fsdp_kwargs)
    logger.info_rank0("VL visual tower wrapped (per-block + root)")

def _apply_ac(model, cfg) -> None:
    """Apply ac (internal)."""
    ac_mode = getattr(cfg.train.gradient_checkpointing, "activation_checkpoint", "off")
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

def _apply_fsdp(model, mesh, cfg) -> None:
    """Per-text-layer + root FSDP wrap. Visual tower is wrapped separately by
    ``_apply_vl_visual_tower``. At dp_size==1 we still go through fully_shard
    so the gradient reduction code path matches dp_size>1.
    """
    dp_mesh = _resolve_dp_mesh(mesh)
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": getattr(cfg.train.accelerator, "reshard_after_forward", True),
        "comm_fusion": getattr(cfg.train.accelerator, "comm_fusion", True),
    }
    mp_policy = _resolve_mp_policy(cfg)
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    # Override default Shard(0) to a divisible non-zero dim for params whose
    # dim-0 isn't divisible by world_size (e.g. ``shared_expert_gate.weight``
    # shape ``(1, hidden)``). Falling back to ``replicate_params`` would route
    # the param through ``to_sharded``'s copy-back path, which costs ±1 bf16
    # ULP per step on NPU.
    try:
        world_size = dp_mesh.size() if hasattr(dp_mesh, "size") else 1
    except (AttributeError, RuntimeError):
        world_size = 1
    shard_dim_overrides: dict = {}
    replicate_params = set()
    if world_size > 1:
        for _, param in model.named_parameters():
            if param.dim() == 0 or param.size(0) % world_size == 0:
                continue
            shardable_dim = next(
                (d for d in range(1, param.dim()) if param.size(d) % world_size == 0),
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

    if not hasattr(model, "layers"):
        logger.warning(
            "Qwen3VLMoeForCausalLM has no ``.layers`` — root-only FSDP wrap."
        )
        fully_shard(model, **fsdp_kwargs)
        return

    layers = list(model.layers)
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    logger.info_rank0("FSDP applied to Qwen3-VL-MoE: %d text layers + root",
                      len(layers))

def parallelize_qwen3_vl_moe(model, mesh, cfg):
    """Apply VL visual handling + text AC / FSDP.

    Order:
      1. VL visual tower (dtype cast + per-block FSDP) — must run before
         text-side wrapping so the visual is its own FSDP unit.
      2. AC on text layers (checkpoint_wrapper inside the FSDP boundary).
      3. FSDP on text layers + root.
    """
    if getattr(cfg.train.accelerator, "tp", 1) > 1:
        raise NotImplementedError(
            "Qwen3-VL-MoE TP is not implemented. Set parallel.tp=1."
        )
    if getattr(cfg.train.accelerator, "ep", 1) > 1:
        raise NotImplementedError(
            "Qwen3-VL-MoE EP requires per-model dispatch / combine wiring "
            "that is not yet implemented. Set parallel.ep=1."
        )

    _apply_vl_visual_tower(model, mesh, cfg)
    _apply_ac(model, cfg)
    _apply_fsdp(model, mesh, cfg)
    return model

__all__ = ["parallelize_qwen3_vl_moe"]

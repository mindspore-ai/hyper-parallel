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
"""GLM5 parallelization."""
import logging

import torch

from hyper_parallel import (
    ContextParallel,
    DSAIndexerContextParallel,
    DSASparseAttentionContextParallel,
    fully_shard,
)
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.models.glm5.model import GLM5ForCausalLM

logger = logging.getLogger(__name__)


def _apply_ac(model, cfg) -> None:
    """Apply activation checkpointing to GLM5 decoder layers."""
    ac_mode = getattr(cfg.train.gradient_checkpointing, "activation_checkpoint", "off")
    if ac_mode in ("off", "none", None, False, ""):
        return
    if not hasattr(model, "layers"):
        logger.warning("AC enabled but GLM5 model has no .layers; skipping.")
        return
    layers = list(model.layers)
    for i, layer in enumerate(layers):
        model.layers[i] = checkpoint_wrapper(layer)
    logger.info_rank0("AC applied to %d GLM5 layers (mode=%s)", len(layers), ac_mode)


def _get_mesh(mesh, name):
    try:
        return mesh[name]
    except (KeyError, TypeError):
        return None


def _validate_tp_support(cfg) -> None:
    tp = getattr(cfg.train.accelerator, "tp", 1)
    if tp > 1:
        raise NotImplementedError(
            "GLM5 TP is not supported yet. Set train.accelerator.tp=1 "
            "or add GLM5 TP parallelization before enabling tensor parallel."
        )


def _apply_cp(model, mesh, cfg) -> None:
    """Apply context parallel styles to GLM5 attention modules."""
    cp = getattr(cfg.train.accelerator, "cp", 1)
    if cp <= 1:
        return
    cp_mesh = _get_mesh(mesh, "cp")
    if cp_mesh is None:
        raise ValueError("GLM5 CP requested but mesh has no 'cp' dimension.")
    cp_size = cp_mesh.size()
    cp_rank = mesh.get_local_rank("cp")
    setattr(model, "_cp_size", cp_size)
    setattr(model, "_cp_rank", cp_rank)
    dsa_layers = 0
    dense_layers = 0
    for layer in list(model.layers):
        if layer.dsa_indexer is not None:
            DSAIndexerContextParallel(
                layout="BSND",
                weights_index=None,
                use_local_output=True,
            ).apply(layer.dsa_indexer.boundary, cp_mesh)
            DSASparseAttentionContextParallel(
                layout="BSND",
                query_rope_index=None,
                key_rope_index=None,
                query_rope_kwarg_name=None,
                key_rope_kwarg_name=None,
                use_local_output=True,
            ).apply(layer.self_attn.sparse_attention_core, cp_mesh)
            dsa_layers += 1
        else:
            setattr(layer.self_attn.attention_core, "_cp_size", cp_size)
            setattr(layer.self_attn.attention_core, "_cp_rank", cp_rank)
            ContextParallel(
                seq_dim=1, head_dim=2, ulysses_degree=1,
            ).apply(layer.self_attn.attention_core, cp_mesh)
            dense_layers += 1
    logger.info_rank0(
        "CP applied to GLM5 attention cores: dense=%d dsa=%d",
        dense_layers,
        dsa_layers,
    )


def _apply_ep(model, mesh, cfg) -> None:
    """Apply expert parallelism to GLM5 MoE experts."""
    ep = getattr(cfg.train.accelerator, "ep", 1)
    if ep <= 1:
        return
    ep_mesh = _get_mesh(mesh, "ep")
    if ep_mesh is None:
        raise ValueError("GLM5 EP requested but mesh has no 'ep' dimension.")
    moe_layers = [layer for layer in model.layers if layer.layer_type == "moe"]
    if not moe_layers:
        raise ValueError("GLM5 EP requires at least one MoE decoder layer.")
    for layer in moe_layers:
        ExpertParallel().apply(layer.mlp.experts, ep_mesh)
    logger.info_rank0("EP applied to %d GLM5 MoE expert modules", len(moe_layers))


def _apply_fsdp(model, mesh, cfg) -> None:
    """Apply FSDP to GLM5 when a data-shard mesh is active."""
    dp_mesh = _get_mesh(mesh, "fsdp")
    if dp_mesh is None:
        dp_mesh = _get_mesh(mesh, "dp_shard")
    accelerator = cfg.train.accelerator
    dp_size = dp_mesh.size() if dp_mesh is not None else 1
    if dp_size <= 1:
        logger.info_rank0(
            "FSDP skipped for GLM5 because dp_shard/fsdp size is one"
        )
        return

    fsdp_kwargs = {
        "mesh": dp_mesh,
        "reshard_after_forward": getattr(
            accelerator, "reshard_after_forward", True,
        ),
        "comm_fusion": getattr(accelerator, "comm_fusion", True),
    }

    mp_cfg = getattr(cfg.train, "mixed_precision", None)
    if mp_cfg is not None and getattr(mp_cfg, "enabled", False):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        param_dtype = dtype_map.get(getattr(mp_cfg, "param_dtype", "bfloat16"))
        reduce_dtype = dtype_map.get(getattr(mp_cfg, "reduce_dtype", "float32"))
        output_dtype_str = getattr(mp_cfg, "output_dtype", None)
        output_dtype = dtype_map.get(output_dtype_str) if output_dtype_str else None
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
        )

    if hasattr(model, "layers"):
        for layer in list(model.layers):
            fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    logger.info_rank0("FSDP applied to GLM5")


def parallelize_glm5(model: GLM5ForCausalLM, mesh, cfg) -> GLM5ForCausalLM:
    """Apply EP, CP, AC and FSDP to the GLM5 model."""
    _validate_tp_support(cfg)
    _apply_ep(model, mesh, cfg)
    _apply_cp(model, mesh, cfg)
    _apply_ac(model, cfg)
    _apply_fsdp(model, mesh, cfg)
    return model


__all__ = ["parallelize_glm5"]

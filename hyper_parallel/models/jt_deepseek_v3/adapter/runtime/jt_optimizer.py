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
"""JT model hooks around the public Muon optimizer."""

# This adapter uses the Torch runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

import math
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention


def _value_copies(parameter: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return every tensor that holds ``parameter``'s value.

    With ``optimizer.fp32_main_params`` the leaf optimizers update an FP32
    ``main_param`` that the mixed-precision wrapper copies back into the model
    parameter after the step, so a post-update edit must change both copies.

    Args:
        parameter: Model parameter, possibly carrying an optimizer ``main_param``.

    Returns:
        The distinct FP32 main parameter (if any) followed by the model parameter.
    """
    main_param = getattr(parameter, "main_param", None)
    if main_param is None or main_param is parameter:
        return (parameter,)
    return main_param, parameter


def _replica_maxima(modules: list[JTDeepseekV3MLAAttention], group: Any) -> list[torch.Tensor]:
    """Return each module's per-head QK maxima over every replica of its heads.

    Ranks in ``group`` hold the same attention heads but see different tokens, so
    clipping with a rank-local maximum would rescale the replicas differently.

    Args:
        modules: Attention modules in model order, identical on every rank.
        group: DP+CP process group, or ``None`` when the heads have no replica.

    Returns:
        Per-module maxima reduced with MAX over ``group``.
    """
    maxima = [module.max_logits_val for module in modules]
    if group is None or not maxima:
        return maxima
    flat = torch.cat([maximum.reshape(-1) for maximum in maxima])
    dist.all_reduce(flat, op=dist.ReduceOp.MAX, group=group)
    parts = flat.split([maximum.numel() for maximum in maxima])
    return [part.view_as(maximum) for part, maximum in zip(parts, maxima)]


def _scale_projection(parameter: torch.Tensor, scale: torch.Tensor, nope_dim: int,
                      extra_dim: int, *, query: bool) -> None:
    """Apply per-head factors to the matching local rows of FSDP/TP parameters."""
    width = nope_dim + extra_dim
    copies = _value_copies(parameter)
    tensor = copies[0]
    global_heads = tensor.shape[0] // width
    if global_heads != scale.numel():
        if not isinstance(tensor, DTensor) or "tp" not in tensor.device_mesh.mesh_dim_names:
            raise ValueError("QK clipping statistics do not cover the projection heads")
        tp_mesh = tensor.device_mesh["tp"]
        gathered = [torch.empty_like(scale) for _ in range(tp_mesh.size())]
        dist.all_gather(gathered, scale, group=tp_mesh.get_group())
        scale = torch.cat(gathered)
    if global_heads != scale.numel():
        raise ValueError("QK clipping TP head statistics have an unexpected size")
    extra = scale if query else torch.ones_like(scale)
    factors = torch.cat((scale.sqrt()[:, None].expand(-1, nope_dim),
                         extra[:, None].expand(-1, extra_dim)), dim=-1).reshape(-1, 1)
    for weight in copies:
        if isinstance(weight, DTensor):
            local_factor = distribute_tensor(factors, weight.device_mesh, weight.placements).to_local()
            weight.to_local().mul_(local_factor)
        else:
            weight.mul_(factors)


@torch.no_grad()
def clip_qk(model: torch.nn.Module, threshold: float) -> None:
    """Clip coupled query/key projections after each optimizer update.

    Args:
        model: JT model with MLA statistics and the DP+CP ``qk_clip_group`` of its heads.
        threshold: Positive clipping threshold from the optimizer adapter configuration.
    """
    modules = [module for module in model.modules() if isinstance(module, JTDeepseekV3MLAAttention)]
    for module, maximum in zip(modules, _replica_maxima(modules, model.qk_clip_group)):
        scale = threshold / maximum.clamp_min(threshold)
        _scale_projection(module.q_b_proj.weight, scale, module.qk_nope_head_dim,
                          module.qk_rope_head_dim, query=True)
        _scale_projection(module.kv_b_proj.weight, scale, module.qk_nope_head_dim,
                          module.v_head_dim, query=False)
        module.max_logits_val.zero_()


@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Apply model-owned updates after all public optimizer leaves complete."""
    del optimizer, args, kwargs
    clip_qk(model, threshold)
    config = model.config
    if config.moe_router_enable_expert_bias:
        for module in model.modules():
            if getattr(module, "expert_load", None) is not None:
                direction = (1 / config.n_routed_experts - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(direction, alpha=config.moe_router_bias_update_rate)
                module.expert_load.zero_()




def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> Muon:
    """Build public Muon/AdamW and attach the JT-specific post-update hooks.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        **kwargs: Public Muon Builder options from the training recipe.

    Returns:
        The unmodified public Muon Builder.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    muon_config = dict(kwargs["muon_config"])
    builder = Muon(model=model, muon_config=muon_config, **{
        name: value for name, value in kwargs.items() if name != "muon_config"
    })
    optimizer = builder.get_optimizer()
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    return builder

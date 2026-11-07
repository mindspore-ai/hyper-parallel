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
"""Build the Xing4 model from the recipe's configuration fields."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel

from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.xing_4.adapter.training import swiglu_policy
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0MLP

__all__ = ["build_model", "create_config"]


def create_config(config: dict[str, Any]) -> Xing4_0Config:
    """Map training aliases to HF fields, preferring an explicit HF value.

    Args:
        config: Model fields supplied by the training recipe.

    Returns:
        The bundled HF configuration carrying HP training options.
    """
    values = dict(config)
    for alias, field in (("num_residual_streams", "hc_mult"), ("mhc_sinkhorn_iterations", "hc_sinkhorn_iters"),
                         ("router_jitter_noise", "moe_input_jitter_eps")):
        if alias in values:
            values.setdefault(field, values.pop(alias))
    values.setdefault("moe_input_jitter_eps", 0.0)
    values.setdefault("attention_softmax_scale", None)
    values.setdefault("use_cache", False)
    if values.get("moe_n_hash_layers") or values.get("moe_router_force_expert_balance"):
        raise ValueError("Xing4 uses the official learned router")
    if (not values.get("use_fused_mhc", True) or values.get("enable_cmhc") or values.get("enable_hc_head")
            or not values.get("mtp_enable_hyper_connections", True)):
        raise ValueError("Xing4 training requires fused mHC with mean stream collapse")
    if any(values.get(key) is not None for key in ("hc_sinkhorn_clamp", "mhc_h_res_clamp_min", "mhc_h_res_clamp_max")):
        raise ValueError("Xing4 fused mHC requires clamp disabled")
    values.update(architectures=["Xing4ForCausalLM"], mhc_h_res_clamp_min=None, mhc_h_res_clamp_max=None)
    return Xing4_0Config(**values)


def build_model(*, config: dict[str, Any], **options: Any) -> PreTrainedModel:
    """Build and parallelize the model through the shared builder.

    Args:
        config: Model configuration with HF fields and HP training options.
        **options: Construction options the Trainer passes to the shared builder.

    Returns:
        The constructed, parallelized model.
    """
    resolved = create_config(config)
    model = HyperAutoModelForCausalLM.from_config(resolved, **options)
    policy = swiglu_policy(resolved)
    if policy != (None, None):
        for name, module in model.named_modules():
            if not isinstance(module, (Xing4_0MLP, SwiGLUMLP)):
                continue
            applied = (getattr(module, "activation_clamp_value", None),
                       getattr(module, "activation_compute_dtype", None))
            if isinstance(module, Xing4_0MLP) or applied != policy:
                raise ValueError(f"The SwiGLU plan for {name} must preserve the configured clamp and compute dtype")
    return model

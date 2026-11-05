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
"""Resolve the HF Xing4 configuration and map explicit training fields."""

from __future__ import annotations

import math
from typing import Any

from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config

__all__ = ["create_config"]

def create_config(config: dict[str, Any]) -> Xing4_0Config:
    """Accept HF configuration fields and equivalent training aliases.

    Args:
        config: Model fields supplied by the training recipe.

    Returns:
        The bundled HF configuration carrying HP training options.
    """
    values = dict(config)
    for training_name, hf_name in (
            ("num_residual_streams", "hc_mult"),
            ("mhc_sinkhorn_iterations", "hc_sinkhorn_iters")):
        if training_name not in values:
            continue
        value = values.pop(training_name)
        if hf_name in values and values[hf_name] != value:
            raise ValueError(f"Conflicting {training_name} and {hf_name}")
        values[hf_name] = value
    jitter = values.get("router_jitter_noise", values.get("moe_input_jitter_eps", 0.0))
    if "moe_input_jitter_eps" in values and values["moe_input_jitter_eps"] != jitter:
        raise ValueError("Conflicting router_jitter_noise and moe_input_jitter_eps")
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError("Router jitter must be finite and nonnegative")
    values["moe_input_jitter_eps"] = jitter
    if values.get("moe_n_hash_layers", 0) or values.get("moe_router_force_expert_balance", False):
        raise ValueError("Xing4 uses the official learned top-k router")
    values["architectures"] = ["Xing4ForCausalLM"]
    limit = values.get("swiglu_limit")
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("swiglu_limit must be finite and positive, or None")
    if values.get("swiglu_compute_dtype") not in (None, "float32", "float16", "bfloat16"):
        raise ValueError("swiglu_compute_dtype must be a floating-point dtype name, or None")
    # The bundled inference config defaults to clamped Sinkhorn. This training
    # contract uses the shared unclamped mHC operators; reject a semantic mismatch.
    clamp = values.pop("hc_sinkhorn_clamp", None)
    if clamp is not None or values.get("mhc_h_res_clamp_min") is not None \
            or values.get("mhc_h_res_clamp_max") is not None:
        raise ValueError("Xing4 training currently requires mHC clamp disabled")
    values["mhc_h_res_clamp_min"] = values["mhc_h_res_clamp_max"] = None
    result = Xing4_0Config(**values)
    if not hasattr(result, "attention_softmax_scale"):
        result.attention_softmax_scale = None
    if not getattr(result, "use_fused_mhc", True):
        raise ValueError("The Xing4 training adapter uses fused mHC on NPU")
    required = ["mhc_init_gating_factor", "mhc_layernorm_epsilon"]
    if result.num_nextn_predict_layers:
        required.append("mtp_loss_factor")
    has_moe = result.n_routed_experts is not None and (
        result.num_hidden_layers + result.num_nextn_predict_layers > result.first_k_dense_replace
    )
    if has_moe:
        required.extend(("moe_aux_loss_coeff", "moe_router_bias_update_rate",
                         "moe_router_enable_expert_bias", "use_pad_tokens"))
    missing = [name for name in required if getattr(result, name, None) is None]
    if missing:
        raise ValueError(f"Xing4 training configuration requires: {', '.join(missing)}")
    for name in required:
        value = getattr(result, name)
        if name in ("moe_router_enable_expert_bias", "use_pad_tokens"):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        elif value < 0 and name != "mhc_init_gating_factor":
            raise ValueError(f"{name} must be nonnegative")
    if result.mhc_layernorm_epsilon == 0:
        raise ValueError("mhc_layernorm_epsilon must be positive")
    if result.use_cache:
        raise ValueError("The training recipe must explicitly disable the KV cache")
    return result

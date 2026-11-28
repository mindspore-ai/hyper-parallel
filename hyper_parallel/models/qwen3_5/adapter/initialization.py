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
"""Retain Qwen3.5 native initialization for replaced GDN and offset norms."""

from functools import partial, update_wrapper
from types import MethodType

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeRMSNorm,
)

from hyper_parallel import DTensor, distribute_tensor
from hyper_parallel.components.modules.attention.gated_delta_net import GatedDeltaNetModule
from hyper_parallel.components.modules.common.rms_norm import OffsetRMSNormModule


_NATIVE_TYPES = (
    (Qwen3_5PreTrainedModel, Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm),
    (Qwen3_5MoePreTrainedModel, Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeRMSNorm),
)


@torch.no_grad()
def _initialize_module(owner, module, *, native_init, gdn_type, norm_type):
    """Present current parameters to HF's type-specific policy without allocating a source module."""
    sharded_parameter = None
    if isinstance(module, GatedDeltaNetModule):
        view = gdn_type.__new__(gdn_type)
        nn.Module.__init__(view)
        view.A_log = module.A_log
        view.dt_bias = module.dt_bias
        view.num_v_heads = module.num_v_heads
        if isinstance(module.A_log, DTensor) and not getattr(module.A_log, "_is_hf_initialized", False):
            # HF samples a dense per-head vector and copy_ rejects Tensor -> DTensor.
            # Initialize only this small vector densely, then distribute to its current layout.
            sharded_parameter = module.A_log
            view.A_log = nn.Parameter(torch.empty(
                sharded_parameter.shape, device=sharded_parameter.device, dtype=sharded_parameter.dtype))
            view.num_v_heads = sharded_parameter.numel()
    elif isinstance(module, OffsetRMSNormModule):
        view = norm_type.__new__(norm_type)
        nn.Module.__init__(view)
        view.weight = module.weight
    else:
        view = module
    # Use the original policy, including HF's initialized-parameter guards. The
    # temporary view shares today's parameters, not stale pre-materialization storage.
    result = native_init(owner, view)
    if sharded_parameter is not None:
        sharded_parameter.copy_(distribute_tensor(
            view.A_log.detach(), sharded_parameter.device_mesh, sharded_parameter.placements, src_data_rank=0))
    return result


def register_native_initialization(model: nn.Module) -> None:
    """Adapt each native Qwen initializer without changing initialized weights.

    HF delegates initialization of nested PreTrainedModels to their own hooks,
    so both the causal-LM owner and its text model need the same adaptation.
    Instance-bound methods keep this change local to the materialized model.

    Args:
        model: Native Qwen model whose replaced modules retain the original initialization policy.
    """
    for owner in model.modules():
        for owner_type, gdn_type, norm_type in _NATIVE_TYPES:
            if not isinstance(owner, owner_type):
                continue
            native_init = owner._init_weights.__func__  # pylint: disable=protected-access
            if isinstance(native_init, partial) and native_init.func is _initialize_module:
                break
            # Store the unbound function so a deepcopy binds to its copied owner.
            adapted = partial(_initialize_module, native_init=native_init, gdn_type=gdn_type, norm_type=norm_type)
            update_wrapper(adapted, native_init)
            owner._init_weights = MethodType(adapted, owner)  # pylint: disable=protected-access
            break

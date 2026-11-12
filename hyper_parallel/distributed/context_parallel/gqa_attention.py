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
"""GQA Family CP adaptation using the existing CP execution helper."""

from functools import wraps
import inspect
from typing import Any

from torch import nn

from hyper_parallel.components.functional.npu_fusion_attention import PACKED_SEQUENCE_ARGUMENTS
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.wrappers import _cp_sdpa_call
from hyper_parallel.distributed.recipe_spec import inner_wrapper


@inner_wrapper
def gqa_attention_cp_wrapper(
    target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Adapt GQA's selected BNSD-to-BSND attention interface to CP all-gather.

    Mirrors the model-adapter wrapper contract without importing model code.
    External masks follow the existing helper's contract: they cover global
    K/V and already encode the intended causality. Packing, windows, caches
    and attention-weight outputs are not supported by this initial adapter.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("GQA CP adapter requires an active CP mesh")
    interface = getattr(target_module, "attention_interface", None)
    if not callable(interface):
        raise ValueError("GQA CP adapter requires attention_interface")
    if getattr(target_module, "sliding_window", None) is not None:
        raise ValueError("GQA CP adapter does not support sliding windows")
    original = target_module.forward
    signature = inspect.signature(original)

    def _attention(module, query, key, value, attention_mask=None, **kwargs):
        if any(kwargs.get(name) is not None for name in PACKED_SEQUENCE_ARGUMENTS):
            raise ValueError("GQA CP adapter does not support packed sequences")
        if kwargs.get("sliding_window") is not None:
            raise ValueError("GQA CP adapter does not support sliding windows")

        def _compute(q, k, v, **options):
            mask = options.pop("attn_mask", None)
            output, weights = interface(module, q, k, v, mask, **options)
            if weights is not None:
                raise ValueError("GQA CP adapter requires attention weights to be disabled")
            return output.transpose(1, 2)

        options = dict(kwargs, attn_mask=attention_mask,
                       is_causal=kwargs.get("is_causal", module.is_causal))
        output = _cp_sdpa_call(_compute, cp_mesh, query, key, value, options)
        return output.transpose(1, 2), None

    @wraps(original)
    def _forward(*args, **kwargs):
        if signature.bind(*args, **kwargs).arguments.get("past_key_values") is not None:
            raise ValueError("GQA CP adapter does not support KV caches")
        return original(*args, **kwargs)

    return _ForwardRewriteRequest(target_module, _forward, {"attention_interface": _attention})

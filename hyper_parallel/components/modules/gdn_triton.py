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

"""Structure-preserving replacement for the Triton-Ascend GDN primitive."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from functools import partial
from types import FunctionType, MethodType
from typing import Any

import torch
from torch import nn
from hyper_parallel.components.functional.linear_attention_packed import resolve_packed_sequence
from hyper_parallel.components.ops import causal_conv1d, gdn
from hyper_parallel.distributed.context_parallel.gated_delta_net import _resolve_gdn_forward_implementation
from hyper_parallel.models.replacement import module_replacement


def _configured_chunk_rule(
    original: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    scale: float | None = None, initial_state: torch.Tensor | None = None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False, cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64, head_first: bool = False, **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if initial_state is None and not output_final_state and scale is None and not head_first:
        info = resolve_packed_sequence({"cu_seq_lens": cu_seqlens}, q.shape[1]) if cu_seqlens is not None else None
        return gdn(q, k, v, g, beta, normalize_qk=use_qk_l2norm_in_kernel, chunk_size=chunk_size, info=info), None
    if scale is not None:
        kwargs["scale"] = scale
    if head_first:
        kwargs["head_first"] = head_first
    if chunk_size != 64:
        kwargs["chunk_size"] = chunk_size
    return original(
        q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens,
        **kwargs,
    )


def _configured_convolution(
    original: Callable, x: torch.Tensor, weight: torch.Tensor | None = None, bias: torch.Tensor | None = None,
    activation: str | None = None, **kwargs: Any,
) -> torch.Tensor:
    if weight is None:
        return original(x, **kwargs)
    if activation not in ("silu", "swish") or kwargs.get("seq_idx") is not None:
        return original(x, weight, bias, activation=activation, **kwargs)
    info = resolve_packed_sequence(kwargs, x.shape[-1])
    output = causal_conv1d(x.transpose(1, 2).to(weight.dtype), weight.unsqueeze(1), bias, info)
    return output.transpose(1, 2).to(x.dtype)


def _bind_forward_primitives(source: nn.Module, target: nn.Module) -> None:
    implementation = _resolve_gdn_forward_implementation(source.forward)
    # Some Transformers versions call globals. A private globals dict keeps
    # the original model and any other concurrently executing model isolated.
    namespace = implementation.__globals__.copy()
    names = implementation.__code__.co_names
    rules = tuple(
        name for name in ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule")
        if name in names and callable(namespace.get(name))
    )
    if not rules:
        raise TypeError("GDN replacement requires a callable chunk_gated_delta_rule")
    for name in rules:
        namespace[name] = partial(_configured_chunk_rule, namespace[name])
    if "causal_conv1d_fn" in names and callable(namespace.get("causal_conv1d_fn")):
        namespace["causal_conv1d_fn"] = partial(_configured_convolution, namespace["causal_conv1d_fn"])
    forward = FunctionType(implementation.__code__, namespace, implementation.__name__,
                           implementation.__defaults__, implementation.__closure__)
    forward.__kwdefaults__ = implementation.__kwdefaults__
    force_hooks = namespace.get("force_accelerate_hooks")
    if callable(force_hooks) and "conv1d" in names:
        forward = force_hooks("conv1d")(forward)
    target.forward = MethodType(forward, target)


@module_replacement
class TritonGDN:
    """Wire an existing GDN model's local rule to the selectable GDN operator.

    The returned module retains its source type so the model's forward, CP
    wrapper, and checkpoint layout remain unchanged.
    """

    def __new__(
        cls,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        """Bind the local GDN forward/backward selector without changing weights.

        Args:
            module: GDN module exposing an instance-level chunk primitive.
            module_fqn: Source module's fully qualified name.
            context: Model replacement context.

        Returns:
            A shallow copy with its chunk primitive bound to the selector.

        Raises:
            TypeError: The source does not expose the expected primitive.
        """
        del cls, context
        replacement = copy.copy(module)
        if callable(getattr(module, "chunk_gated_delta_rule", None)):
            replacement.chunk_gated_delta_rule = partial(_configured_chunk_rule, module.chunk_gated_delta_rule)
            convolution = getattr(module, "causal_conv1d_fn", None)
            if callable(convolution):
                replacement.causal_conv1d_fn = partial(_configured_convolution, convolution)
        else:
            try:
                _bind_forward_primitives(module, replacement)
            except (TypeError, RuntimeError) as exc:
                raise TypeError(f"{module_fqn}: GDN replacement requires callable chunk_gated_delta_rule") from exc
        return replacement

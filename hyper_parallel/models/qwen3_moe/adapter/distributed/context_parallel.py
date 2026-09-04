# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""context_parallel: Qwen3-MoE CP wrappers and their mask/head contract.

Migrated from ``components/models/qwen3_moe_fusions.py`` in M3 (adjust doc
§5.3/§8). The wrappers orchestrate only the Qwen attention interface with
the public CP collectives — the projection pipeline (fused QKV, Q/K norm,
RoPE, cache, output projection) is owned by the generic
``modules.GQAAttention`` built by ``adapter/replacements.py``.

Forward-install discipline (05 §15.2.3): the wrappers never assign
``module.forward`` themselves; each validates its target and returns a
``_ForwardRewriteRequest`` whose companion attribute swaps the module's
``attention_interface`` for the CP-orchestrated one. The generic
``distributed/_builder/forward_rewriter.py`` validates, commits and rolls
back atomically.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch
from torch import nn

from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.attention import (
    _cp_offset_causal_mask,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    flex_cp_allgather,
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.qwen3_moe.adapter.attention import (
    run_qwen3_moe_flash_attention,
)


def _validate_qwen3_moe_flash_attention_cp_target(
    target_module: nn.Module,
    cp_mesh: Any,
    wrapper_name: str,
) -> Any:
    """Validate a Qwen3-MoE fused CP wrapper target and return its forward."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")

    original_forward = target_module.forward
    # Post-M2 the fused attention target is the modules.GQAAttention built by
    # replace_qwen3_moe_flash_attention; it is identified structurally (a
    # callable attention_interface) so this module stays importable without
    # the torch_npu-dependent modules package.
    if not callable(getattr(target_module, "attention_interface", None)):
        raise ValueError(
            f"{wrapper_name} can only wrap the modules.GQAAttention "
            "replacement for qwen3_moe_flash_attention_forward; apply "
            "replace_qwen3_moe_flash_attention first"
        )
    return original_forward


def _validate_qwen3_moe_flash_attention_ulysses_heads(
    target_module: nn.Module,
    cp_size: int,
    wrapper_name: str,
) -> None:
    """Validate the Qwen3-MoE head counts required by Pure Ulysses."""
    config = getattr(target_module, "config", None)
    if config is None:
        raise ValueError(f"{wrapper_name} requires target_module.config")

    for name in ("num_attention_heads", "num_key_value_heads"):
        count = getattr(config, name, None)
        if count is None:
            count = getattr(target_module, name, None)
        if count is None:
            raise ValueError(
                f"{wrapper_name} requires {name} in target_module.config"
            )
        count = int(count)
        if count % cp_size:
            raise ValueError(
                f"{wrapper_name} requires {name} ({count}) to be divisible "
                f"by Ulysses degree ({cp_size})"
            )


def _prepare_qwen3_moe_flash_attention_ulysses_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
) -> torch.Tensor | None:
    """Validate an external mask after Ulysses has restored global sequence."""
    if attention_mask is None:
        return None
    if attention_mask.ndim < 2:
        raise ValueError(
            "Qwen3-MoE fused Ulysses attention_mask must include query and key "
            "dimensions"
        )
    if attention_mask.shape[-1] != key_length:
        raise ValueError(
            "Qwen3-MoE fused Ulysses attention_mask must cover the global key "
            f"sequence: mask key length={attention_mask.shape[-1]}, "
            f"expected {key_length}"
        )

    mask_query_length = attention_mask.shape[-2]
    if mask_query_length < query_length:
        raise ValueError(
            "Qwen3-MoE fused Ulysses attention_mask must cover the global query "
            f"sequence: mask query length={mask_query_length}, expected at least "
            f"{query_length}"
        )
    if mask_query_length != query_length:
        attention_mask = attention_mask.narrow(-2, 0, query_length)
    return attention_mask


def _prepare_qwen3_moe_flash_attention_cp_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    query_offset: int,
    device: torch.device,
    allow_external_mask: bool,
) -> torch.Tensor:
    """Build an implicit CP mask or slice an external global allowed mask."""
    if attention_mask is None:
        causal_mask = _cp_offset_causal_mask(
            query_length,
            key_length,
            query_offset,
            device,
        )
        return causal_mask

    if not allow_external_mask:
        raise ValueError(
            "Qwen3-MoE fused CP attention requires an implicit causal mask; "
            "configure create_attention_mask_in_dataloader=false or use "
            "qwen3_moe_flash_attention_cp_mask_wrapper"
        )
    if attention_mask.ndim < 2:
        raise ValueError(
            "Qwen3-MoE fused CP attention_mask must include query and key dimensions"
        )
    if attention_mask.shape[-1] != key_length:
        raise ValueError(
            "Qwen3-MoE fused CP attention_mask must cover the global key sequence: "
            f"mask key length={attention_mask.shape[-1]}, expected {key_length}"
        )

    mask_query_length = attention_mask.shape[-2]
    if mask_query_length != query_length:
        query_end = query_offset + query_length
        if mask_query_length < query_end:
            raise ValueError(
                "Qwen3-MoE fused CP attention_mask does not cover this rank's query "
                f"range [{query_offset}, {query_end})"
            )
        attention_mask = attention_mask.narrow(
            -2,
            query_offset,
            query_length,
        )

    return attention_mask


def _build_qwen3_moe_flash_attention_cp_interface(
    cp_mesh: Any,
    allow_external_mask: bool,
):
    """Return an ``attention_interface`` orchestrating ring-free CP all-gather.

    The interface receives the prepared BNSD states from
    ``modules.GQAAttention``, all-gathers K/V over the CP mesh through the
    public CP collectives, applies the Qwen mask contract and delegates the
    kernel call to ``run_qwen3_moe_flash_attention``.
    """

    def cp_attention_interface(
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-gather K/V across CP, then run the Qwen attention contract."""
        query_length = query_states.shape[-2]
        query_offset = cp_mesh.get_local_rank() * query_length
        gathered_key_states, gathered_value_states = flex_cp_allgather(
            key_states.contiguous(),
            value_states.contiguous(),
            2,
            cp_mesh,
        )
        cp_attention_mask = _prepare_qwen3_moe_flash_attention_cp_mask(
            attention_mask,
            query_length,
            gathered_key_states.shape[-2],
            query_offset,
            query_states.device,
            allow_external_mask,
        )
        return run_qwen3_moe_flash_attention(
            module,
            query_states,
            gathered_key_states,
            gathered_value_states,
            cp_attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    return cp_attention_interface


def _build_qwen3_moe_flash_attention_ulysses_interface(cp_mesh: Any):
    """Return an ``attention_interface`` orchestrating synchronous Pure Ulysses.

    The interface receives the prepared BNSD states from
    ``modules.GQAAttention``, redistributes Q/K/V sequence-to-head over the
    CP mesh through the public CP collectives, applies the Qwen mask
    contract, delegates the kernel call to
    ``run_qwen3_moe_flash_attention`` and redistributes the output back
    head-to-sequence.
    """

    def ulysses_attention_interface(
        module: nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Qwen attention contract through Pure Ulysses A2A."""
        cp_size = cp_mesh.size()
        for name, states in (
            ("query", query_states),
            ("key", key_states),
            ("value", value_states),
        ):
            if states.shape[1] % cp_size:
                raise ValueError(
                    f"{name} head count ({states.shape[1]}) must be divisible by "
                    f"Ulysses degree ({cp_size})"
                )

        query_states, key_states, value_states = (
            ulysses_seq_to_head(states, 2, 1, cp_mesh)
            for states in (query_states, key_states, value_states)
        )
        global_query_length = query_states.shape[2]
        global_key_length = key_states.shape[2]
        ulysses_attention_mask = _prepare_qwen3_moe_flash_attention_ulysses_mask(
            attention_mask,
            global_query_length,
            global_key_length,
        )
        attention_output, attention_weights = run_qwen3_moe_flash_attention(
            module,
            query_states,
            key_states,
            value_states,
            ulysses_attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
        # The fused kernel returns BSHD, whereas the projections use BHSD.
        attention_output = ulysses_head_to_seq(
            attention_output,
            1,
            2,
            cp_mesh,
        )
        return attention_output, attention_weights

    return ulysses_attention_interface


@inner_wrapper
def qwen3_moe_flash_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Install Qwen3-MoE fused CP attention with an implicit causal mask.

    Returns a rewrite request swapping the target's ``attention_interface``
    for the CP all-gather orchestration; the generic forward rewriter
    validates and commits it (the adapter never assigns ``forward`` itself).
    """
    del mesh, tp_mesh, ep_mesh
    original_forward = _validate_qwen3_moe_flash_attention_cp_target(
        target_module,
        cp_mesh,
        "qwen3_moe_flash_attention_cp_wrapper",
    )

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the GQAAttention forward with the CP attention interface."""
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "attention_interface": _build_qwen3_moe_flash_attention_cp_interface(
                cp_mesh,
                allow_external_mask=False,
            ),
        },
    )


@inner_wrapper
def qwen3_moe_flash_attention_cp_mask_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Install Qwen3-MoE fused CP attention that accepts a global block mask.

    Returns a rewrite request swapping the target's ``attention_interface``
    for the CP all-gather orchestration; the generic forward rewriter
    validates and commits it (the adapter never assigns ``forward`` itself).
    """
    del mesh, tp_mesh, ep_mesh
    original_forward = _validate_qwen3_moe_flash_attention_cp_target(
        target_module,
        cp_mesh,
        "qwen3_moe_flash_attention_cp_mask_wrapper",
    )

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the GQAAttention forward with the CP attention interface."""
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "attention_interface": _build_qwen3_moe_flash_attention_cp_interface(
                cp_mesh,
                allow_external_mask=True,
            ),
        },
    )


@inner_wrapper
def qwen3_moe_flash_attention_ulysses_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Install Qwen3-MoE fused attention with synchronous Pure Ulysses A2A.

    Returns a rewrite request swapping the target's ``attention_interface``
    for the Ulysses orchestration; the generic forward rewriter validates
    and commits it (the adapter never assigns ``forward`` itself).

    Args:
        target_module: Qwen3-MoE attention module whose forward is replaced.
        mesh: Active model mesh supplied by the injection framework.
        tp_mesh: Active tensor-parallel mesh, if configured.
        cp_mesh: Active context-parallel mesh used for Ulysses all-to-all.
        ep_mesh: Active expert-parallel mesh, if configured.
    """
    del mesh, tp_mesh, ep_mesh
    wrapper_name = "qwen3_moe_flash_attention_ulysses_cp_wrapper"
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")
    _validate_qwen3_moe_flash_attention_ulysses_heads(
        target_module,
        cp_mesh.size(),
        wrapper_name,
    )
    original_forward = _validate_qwen3_moe_flash_attention_cp_target(
        target_module,
        cp_mesh,
        wrapper_name,
    )

    @wraps(original_forward)
    def cp_forward(*args: Any, **kwargs: Any) -> Any:
        """Run the original attention signature through Pure Ulysses.

        Args:
            *args: Local sequence shard forward arguments.
            **kwargs: Additional fused attention arguments.
        """
        return original_forward(*args, **kwargs)

    return _ForwardRewriteRequest(
        target_module,
        cp_forward,
        companion_attrs={
            "attention_interface": _build_qwen3_moe_flash_attention_ulysses_interface(
                cp_mesh,
            ),
        },
    )


__all__ = [
    "qwen3_moe_flash_attention_cp_mask_wrapper",
    "qwen3_moe_flash_attention_cp_wrapper",
    "qwen3_moe_flash_attention_ulysses_cp_wrapper",
]

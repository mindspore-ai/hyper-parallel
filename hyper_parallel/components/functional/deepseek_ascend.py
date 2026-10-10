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
"""Lazy adapters for the optional DeepSeek Ascend operator packages."""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Any

import torch  # pylint: disable=forbidden-backend-import


_BACKEND_PACKAGES = {
    "deep_gemm": "DeepGEMM-Ascend",
    "deep_select": "DeepSelect",
    "flash_mla": "FlashMLA",
    "tile_kernels": "TileKernels",
}
_TILE_KERNEL_NAMESPACES = frozenset(
    ("engram", "mhc", "modeling.engram", "moe", "quant", "rand", "transform")
)


def is_deepseek_backend_available(backend: str) -> bool:
    """Return whether an optional DeepSeek operator package can be discovered.

    This probe does not load the native extension. A discovered package can
    still fail to load when it was built for a different CANN or PyTorch
    version.

    Args:
        backend: One of ``deep_gemm``, ``deep_select``, ``flash_mla``, or
            ``tile_kernels``.

    Returns:
        ``True`` when Python can discover the package.

    Raises:
        ValueError: If ``backend`` is not a supported package name.
    """
    _validate_backend(backend)
    return importlib.util.find_spec(backend) is not None


def _validate_backend(backend: str) -> None:
    if backend not in _BACKEND_PACKAGES:
        supported = ", ".join(sorted(_BACKEND_PACKAGES))
        raise ValueError(f"Unsupported DeepSeek backend {backend!r}; expected one of: {supported}")


def _load_backend(backend: str) -> Any:
    _validate_backend(backend)
    try:
        return importlib.import_module(backend)
    except (ImportError, OSError, RuntimeError) as error:
        project = _BACKEND_PACKAGES[backend]
        raise RuntimeError(
            f"DeepSeek backend {backend!r} is unavailable. Install and build {project} "
            "for the active PyTorch, torch-npu, and CANN environment."
        ) from error


@lru_cache(maxsize=None)
def _resolve_operator(backend: str, operator: str) -> Any:
    _validate_public_operator_name(operator)
    module = _load_backend(backend)
    value = getattr(module, operator, None)
    if value is None or not callable(value):
        raise AttributeError(f"DeepSeek backend {backend!r} has no callable operator {operator!r}")
    return value


def _validate_public_operator_name(operator: str) -> None:
    if not isinstance(operator, str) or not operator or operator.startswith("_"):
        raise ValueError(f"Operator name must identify a public callable, got {operator!r}")


def deepseek_gemm(operator: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a top-level DeepGEMM-Ascend operator.

    The adapter deliberately accepts an operator name because DeepGEMM exposes
    a growing matrix of dense, grouped, einsum, MQA-logit, and layout kernels.
    Their native signatures and return values are preserved.

    Args:
        operator: Public callable exported by the ``deep_gemm`` package, for
            example ``"bf16_gemm_nt"`` or ``"fp8_fp4_gemm_nt"``.
        *args: Positional arguments forwarded to the native operator.
        **kwargs: Keyword arguments forwarded to the native operator.

    Returns:
        The native operator result.
    """
    return _resolve_operator("deep_gemm", operator)(*args, **kwargs)


def deepseek_bf16_gemm(layout: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a DeepGEMM BF16 kernel with ``nt``, ``nn``, ``tn``, or ``tt`` layout."""
    return deepseek_gemm(_gemm_layout_operator("bf16", layout), *args, **kwargs)


def deepseek_fp8_gemm(layout: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a DeepGEMM FP8 kernel with ``nt``, ``nn``, ``tn``, or ``tt`` layout."""
    return deepseek_gemm(_gemm_layout_operator("fp8", layout), *args, **kwargs)


def deepseek_fp8_fp4_gemm(layout: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a DeepGEMM FP8-by-FP4 kernel with ``nt``, ``nn``, ``tn``, or ``tt`` layout."""
    return deepseek_gemm(_gemm_layout_operator("fp8_fp4", layout), *args, **kwargs)


def _gemm_layout_operator(precision: str, layout: str) -> str:
    layouts = {name: f"{precision}_gemm_{name}" for name in ("nt", "nn", "tn", "tt")}
    return _select_variant(f"{precision} GEMM layout", layout, layouts)


def deepseek_grouped_bf16_gemm(grouping: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a DeepGEMM grouped BF16 kernel.

    Args:
        grouping: Kernel layout: ``m_nt``, ``m_nn``, or ``k_tn``.
        *args: Native operator positional arguments.
        **kwargs: Native operator keyword arguments.

    Returns:
        The native grouped-GEMM result.
    """
    operators = {
        "m_nt": "m_grouped_bf16_gemm_nt_contiguous",
        "m_nn": "m_grouped_bf16_gemm_nn_contiguous",
        "k_tn": "k_grouped_bf16_gemm_tn_contiguous",
    }
    return deepseek_gemm(_select_variant("grouped BF16 GEMM", grouping, operators), *args, **kwargs)


def deepseek_grouped_fp8_gemm(grouping: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a DeepGEMM grouped FP8 kernel.

    Args:
        grouping: Kernel layout: ``m_nt``, ``m_nn``, ``k_nt``, or ``k_tn``.
        *args: Native operator positional arguments.
        **kwargs: Native operator keyword arguments.

    Returns:
        The native grouped-GEMM result.
    """
    operators = {
        "m_nt": "m_grouped_fp8_gemm_nt_contiguous",
        "m_nn": "m_grouped_fp8_gemm_nn_contiguous",
        "k_nt": "k_grouped_fp8_gemm_nt_contiguous",
        "k_tn": "k_grouped_fp8_gemm_tn_contiguous",
    }
    return deepseek_gemm(_select_variant("grouped FP8 GEMM", grouping, operators), *args, **kwargs)


def deepseek_grouped_fp8_fp4_gemm(grouping: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a DeepGEMM grouped FP8-by-FP4 kernel.

    Args:
        grouping: Kernel layout: ``m_nt`` or ``m_nn``.
        *args: Native operator positional arguments.
        **kwargs: Native operator keyword arguments.

    Returns:
        The native grouped-GEMM result.
    """
    operators = {
        "m_nt": "m_grouped_fp8_fp4_gemm_nt_contiguous",
        "m_nn": "m_grouped_fp8_fp4_gemm_nn_contiguous",
    }
    return deepseek_gemm(_select_variant("grouped FP8-by-FP4 GEMM", grouping, operators), *args, **kwargs)


def deepseek_einsum(*args: Any, quantized: bool = False, **kwargs: Any) -> Any:
    """Run DeepGEMM einsum, optionally using its FP8 implementation."""
    operator = "fp8_einsum" if quantized else "einsum"
    return deepseek_gemm(operator, *args, **kwargs)


def deepseek_mqa_logits(*args: Any, paged: bool = False, **kwargs: Any) -> Any:
    """Run DeepGEMM FP8-by-FP4 MQA-logit computation."""
    operator = "fp8_fp4_paged_mqa_logits" if paged else "fp8_fp4_mqa_logits"
    return deepseek_gemm(operator, *args, **kwargs)


def deepseek_paged_mqa_logits_metadata(*args: Any, **kwargs: Any) -> Any:
    """Build DeepGEMM scheduling metadata for paged MQA-logit kernels."""
    return deepseek_gemm("get_paged_mqa_logits_metadata", *args, **kwargs)


def deepseek_transform_scaling_factors(*args: Any, grouped: bool = False, **kwargs: Any) -> Any:
    """Transform scaling factors into the layout required by DeepGEMM."""
    operator = "transform_k_grouped_sf_into_required_layout" if grouped else "transform_sf_into_required_layout"
    return deepseek_gemm(operator, *args, **kwargs)


def deepseek_mega_moe(*args: Any, **kwargs: Any) -> Any:
    """Run DeepGEMM's fused FP8-by-FP4 Mega-MoE operator."""
    return deepseek_gemm("fp8_fp4_mega_moe", *args, **kwargs)


def deepseek_transform_mega_moe_weights(*args: Any, **kwargs: Any) -> Any:
    """Transform expert weights into the layout required by Mega-MoE."""
    return deepseek_gemm("transform_weights_for_mega_moe", *args, **kwargs)


def deepseek_hc_prenorm_gemm(*args: Any, **kwargs: Any) -> Any:
    """Run DeepGEMM's TileLang TF32 hyper-connection pre-norm GEMM."""
    return deepseek_gemm("tf32_hc_prenorm_gemm", *args, **kwargs)


def _select_variant(family: str, variant: str, operators: dict[str, str]) -> str:
    try:
        return operators[variant]
    except KeyError as error:
        supported = ", ".join(operators)
        raise ValueError(f"Unsupported {family} variant {variant!r}; expected one of: {supported}") from error


def deepseek_select_stride_requirement() -> tuple[int, int]:
    """Return DeepSelect input/output row-stride alignment in bytes."""
    return _resolve_operator("deep_select", "get_stride_requirement")()


def deepseek_select_topk(
    input_tensor: torch.Tensor,
    topk: int,
    *,
    sorted: bool = False,  # pylint: disable=redefined-builtin
    begin: torch.Tensor | None = None,
    end: torch.Tensor | None = None,
    indices_type: torch.dtype = torch.int32,
    sorted_index: bool = False,
    hint: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    output_indices_offset: torch.Tensor | None = None,
    index_out_of_bounds_fill_value: int = 2147483647,
    value_out_of_bounds_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Run DeepSelect's high-performance Top-K selection.

    Args:
        input_tensor: Two-dimensional score tensor.
        topk: Number of entries selected per row.
        sorted: Whether to sort output values. Ascend currently requires
            ``False``.
        begin: Optional inclusive row starts. Reserved by DeepSelect.
        end: Optional exclusive row ends in ``int32``.
        indices_type: Output index dtype. Defaults to the Ascend-supported
            ``torch.int32``.
        sorted_index: Whether to sort output indices.
        hint: Optional selection hint. Reserved by DeepSelect.
        output_indices: Optional preallocated index output.
        output_indices_offset: Optional per-row index offsets in ``int32``.
        index_out_of_bounds_fill_value: Fill value for invalid indices.
        value_out_of_bounds_fill_value: Fill value for invalid values.
        return_value: Return selected values in addition to indices.
        abort_when_nan_found: Abort the kernel if an input NaN is found.

    Returns:
        The values and indices returned by ``deep_select.topk``.
    """
    return _resolve_operator("deep_select", "topk")(
        input_tensor,
        topk,
        sorted=sorted,
        begin=begin,
        end=end,
        indices_type=indices_type,
        sorted_index=sorted_index,
        hint=hint,
        output_idx=output_indices,
        output_idx_offset=output_indices_offset,
        idx_oob_fill_value=index_out_of_bounds_fill_value,
        value_oob_fill_value=value_out_of_bounds_fill_value,
        return_value=return_value,
        abort_when_nan_found=abort_when_nan_found,
    )


def deepseek_flash_mla_metadata(*args: Any, **kwargs: Any) -> Any:
    """Create scheduler metadata for FlashMLA sparse decoding."""
    return _resolve_operator("flash_mla", "get_mla_metadata")(*args, **kwargs)


def deepseek_flash_mla_sparse_prefill(
    query: torch.Tensor,
    key_value: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    value_head_dim: int = 512,
    attention_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the DeepSeek V4.1 FlashMLA sparse-prefill operator.

    Args:
        query: Query tensor with shape ``[sequence, heads, 512]``.
        key_value: Key/value tensor with shape ``[kv_sequence, kv_heads, 512]``.
        indices: Sparse key/value indices in ``int32``.
        softmax_scale: Scale applied to attention logits.
        value_head_dim: Value dimension. FlashMLA currently requires 512.
        attention_sink: Optional per-head attention sink.
        topk_length: Optional valid Top-K length per query token.

    Returns:
        ``(output, max_logits, logsumexp)`` from FlashMLA.

    Raises:
        ValueError: If ``value_head_dim`` is not 512.
    """
    _validate_flash_mla_value_head_dim(value_head_dim)
    operator = _resolve_operator("flash_mla", "flash_mla_sparse_fwd")
    return operator(
        query,
        key_value,
        indices,
        softmax_scale,
        d_v=value_head_dim,
        attn_sink=attention_sink,
        topk_length=topk_length,
    )


def deepseek_flash_mla_sparse_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    indices: torch.Tensor,
    scheduler_metadata: Any,
    *,
    value_head_dim: int = 512,
    softmax_scale: float | None = None,
    attention_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_key_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    enable_batch_invariant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the DeepSeek V4.1 FlashMLA sparse-decode operator.

    Args:
        query: Query tensor ``[batch, query_sequence, heads, 512]``.
        key_cache: Quantized paged key/value cache.
        indices: Sparse indices into ``key_cache``.
        scheduler_metadata: Metadata created by
            :func:`deepseek_flash_mla_metadata`.
        value_head_dim: Value dimension. FlashMLA currently requires 512.
        softmax_scale: Optional attention-logit scale.
        attention_sink: Optional per-head attention sink.
        topk_length: Optional valid Top-K length per batch item.
        extra_key_cache: Optional secondary quantized cache.
        extra_indices: Sparse indices into ``extra_key_cache``.
        extra_topk_length: Optional valid Top-K length for the extra cache.
        enable_batch_invariant: Disable split-KV decoding for batch-invariant
            numerical results.

    Returns:
        ``(output, logsumexp)`` from FlashMLA.

    Raises:
        ValueError: If ``value_head_dim`` is not 512.
    """
    _validate_flash_mla_value_head_dim(value_head_dim)
    operator = _resolve_operator("flash_mla", "flash_mla_with_kvcache")
    return operator(
        query,
        key_cache,
        None,
        None,
        value_head_dim,
        scheduler_metadata,
        softmax_scale=softmax_scale,
        indices=indices,
        attn_sink=attention_sink,
        topk_length=topk_length,
        extra_k_cache=extra_key_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
        enable_batch_invariant=enable_batch_invariant,
    )


def _validate_flash_mla_value_head_dim(value_head_dim: int) -> None:
    if value_head_dim != 512:
        raise ValueError(f"FlashMLA V4.1 requires value_head_dim=512, got {value_head_dim}")


def deepseek_tile_kernel(namespace: str, operator: str, /, *args: Any, **kwargs: Any) -> Any:
    """Run a public TileKernels operator without eagerly importing TileLang.

    Args:
        namespace: TileKernels family: ``engram``, ``mhc``,
            ``modeling.engram``, ``moe``, ``quant``, ``rand``, or
            ``transform``.
        operator: Public callable in that namespace.
        *args: Positional arguments forwarded to the operator.
        **kwargs: Keyword arguments forwarded to the operator.

    Returns:
        The TileKernels operator result.

    Raises:
        ValueError: If the namespace is not public.
    """
    return _call_tile_kernel(namespace, operator, *args, **kwargs)


def _call_tile_kernel(namespace: str, operator: str, *args: Any, **kwargs: Any) -> Any:
    return _resolve_tile_operator(namespace, operator)(*args, **kwargs)


@lru_cache(maxsize=None)
def _resolve_tile_operator(namespace: str, operator: str) -> Any:
    if namespace not in _TILE_KERNEL_NAMESPACES:
        supported = ", ".join(sorted(_TILE_KERNEL_NAMESPACES))
        raise ValueError(f"Unsupported TileKernels namespace {namespace!r}; expected one of: {supported}")
    _validate_public_operator_name(operator)
    _load_backend("tile_kernels")
    module = importlib.import_module(f"tile_kernels.{namespace}")
    value = getattr(module, operator, None)
    if value is None or not callable(value):
        raise AttributeError(f"TileKernels namespace {namespace!r} has no callable operator {operator!r}")
    return value


def deepseek_per_token_cast(
    input_tensor: Any,
    fmt: str,
    num_per_channels: int,
    **kwargs: Any,
) -> Any:
    """Quantize with TileKernels per-token FP8/FP4 scaling."""
    return _call_tile_kernel("quant", "per_token_cast", input_tensor, fmt, num_per_channels, **kwargs)


def deepseek_per_block_cast(
    input_tensor: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    **kwargs: Any,
) -> Any:
    """Quantize with TileKernels per-block FP8/FP4 scaling."""
    return _call_tile_kernel("quant", "per_block_cast", input_tensor, fmt, block_size, **kwargs)


def deepseek_cast_back(
    quantized_tensor: Any,
    fmt: str,
    block_size: tuple[int, int],
    **kwargs: Any,
) -> torch.Tensor:
    """Dequantize a TileKernels FP8/FP4 tensor to BF16 or FP32."""
    return _call_tile_kernel("quant", "cast_back", quantized_tensor, fmt, block_size, **kwargs)


def deepseek_swiglu_forward(input_tensor: torch.Tensor, fmt: str, **kwargs: Any) -> torch.Tensor:
    """Run TileKernels fused SwiGLU forward."""
    return _call_tile_kernel("quant", "swiglu_forward", input_tensor, fmt, **kwargs)


def deepseek_swiglu_backward(
    input_tensor: torch.Tensor,
    output_gradient: torch.Tensor,
    fmt: str,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Run TileKernels fused SwiGLU backward."""
    return _call_tile_kernel("quant", "swiglu_backward", input_tensor, output_gradient, fmt, **kwargs)


def deepseek_topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Select expert indices with the TileKernels stable Top-K operator."""
    return _call_tile_kernel("moe", "topk_gate", scores, num_topk)


def deepseek_moe_topk_gate(
    logits: torch.Tensor,
    num_topk: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    *,
    scoring_func: str = "sqrtsoftplus",
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
    fix_routing_mask: torch.Tensor | None = None,
    to_physical_map: torch.Tensor | None = None,
    logical_count: torch.Tensor | None = None,
    unmapped_topk_idx: torch.Tensor | None = None,
    force_random: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TileKernels fused MoE scoring, Top-K selection, and weight normalization.

    The parameters cover the current ``tile_kernels.moe.moe_topk_gate``
    contract. See the upstream operator documentation for tensor shapes and
    expert-mapping constraints.

    Args:
        logits: Contiguous FP32 token-to-expert logits.
        num_topk: Number of routed experts selected per token.
        use_shared_as_routed: Include shared experts in routed outputs.
        num_shared_experts: Number of shared experts.
        routed_scaling_factor: Scale applied to normalized routing weights.
        ep_rank: Current expert-parallel rank.
        scoring_func: Upstream scoring function. Currently ``sqrtsoftplus``.
        mask: Optional valid-token mask.
        bias: Optional expert bias.
        image_bias: Optional expert bias for image tokens.
        image_token_mask: Optional image-token mask.
        fix_routing_mask: Optional mask selecting fixed routing rows.
        to_physical_map: Optional logical-to-physical expert mapping.
        logical_count: Optional physical-expert count per logical expert.
        unmapped_topk_idx: Optional output for logical expert indices.
        force_random: Optional mask selecting randomized routing rows.
        out: Optional preallocated ``(indices, weights)`` outputs.
        scores: Optional preallocated scoring output used by backward.

    Returns:
        Selected expert indices and normalized routing weights.
    """
    return _call_tile_kernel(
        "moe",
        "moe_topk_gate",
        logits,
        num_topk,
        use_shared_as_routed,
        num_shared_experts,
        routed_scaling_factor,
        ep_rank,
        scoring_func=scoring_func,
        mask=mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=fix_routing_mask,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
        unmapped_topk_idx=unmapped_topk_idx,
        force_random=force_random,
        out=out,
        scores=scores,
    )


def deepseek_moe_topk_gate_backward(
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_weight_gradient: torch.Tensor,
    routed_scaling_factor: float,
    *,
    scoring_func: str = "sqrtsoftplus",
    mask: torch.Tensor | None = None,
    score_sum_gradient: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the TileKernels fused MoE Top-K gate backward operator.

    Args:
        scores: Dense scores produced during the fused forward operation.
        topk_indices: Logical, unmapped routed-expert indices. When forward
            uses ``to_physical_map``, pass its ``unmapped_topk_idx`` output
            rather than the returned physical indices.
        topk_weights: Normalized routing weights produced by forward.
        topk_weight_gradient: Gradient of ``topk_weights``.
        routed_scaling_factor: Scale used by the forward operation.
        scoring_func: Upstream scoring function. Currently ``sqrtsoftplus``.
        mask: Optional valid-token mask.
        score_sum_gradient: Optional auxiliary-loss score-sum gradient.
        out: Optional preallocated logit-gradient output.

    Returns:
        Gradient of the input logits.

    Note:
        The upstream backward kernel does not support shared-as-routed
        experts, physical mapping, or force-random routing.
    """
    return _call_tile_kernel(
        "moe",
        "moe_topk_gate_backward",
        scores,
        topk_indices,
        topk_weights,
        topk_weight_gradient,
        routed_scaling_factor,
        scoring_func=scoring_func,
        mask=mask,
        grad_scores_sum=score_sum_gradient,
        out=out,
    )


def deepseek_normalize_routing_weights(
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize TileKernels MoE routing weights per token."""
    return _call_tile_kernel("moe", "normalize_weight", routing_weights)


def deepseek_apply_rotary(
    query: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    *,
    interleaved: bool = False,
    conjugate: bool = False,
    sequence_offset: int = 0,
) -> None:
    """Apply TileKernels RoPE to query and optional key in place."""
    _call_tile_kernel(
        "transform",
        "apply_rotary",
        query,
        cos_sin_cache,
        key=key,
        positions=positions,
        interleaved=interleaved,
        conjugate=conjugate,
        seqlen_offset=sequence_offset,
    )


def deepseek_engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    image_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute TileKernels Engram n-gram hash indices."""
    return _call_tile_kernel(
        "engram",
        "engram_hash",
        ngram_token_ids,
        multipliers,
        vocab_sizes,
        offsets,
        image_token_mask=image_token_mask,
    )


def deepseek_engram_gate(
    hidden_states: torch.Tensor,
    key_value: torch.Tensor,
    hidden_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    clamp_value: float,
    epsilon: float,
    image_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run TileKernels' autograd-enabled fused Engram gate.

    Args:
        hidden_states: BF16 hidden states ending in ``[hc_mult, hidden]``.
        key_value: BF16 key/value tensor ending in ``[(hc_mult + 1) * hidden]``.
        hidden_weight: RMSNorm weight for hidden states.
        embedding_weight: RMSNorm weight for Engram keys.
        clamp_value: Clamp range for the fused gate.
        epsilon: RMSNorm numerical-stability epsilon.
        image_token_mask: Optional flattened image-token mask.

    Returns:
        Fused Engram output with the same shape as ``hidden_states``.
    """
    return _call_tile_kernel(
        "modeling.engram",
        "engram_gate",
        hidden_states,
        key_value,
        hidden_weight,
        embedding_weight,
        clamp_value,
        epsilon,
        image_token_mask,
    )


__all__ = [
    "deepseek_apply_rotary",
    "deepseek_bf16_gemm",
    "deepseek_cast_back",
    "deepseek_einsum",
    "deepseek_engram_gate",
    "deepseek_engram_hash",
    "deepseek_flash_mla_metadata",
    "deepseek_flash_mla_sparse_decode",
    "deepseek_flash_mla_sparse_prefill",
    "deepseek_fp8_fp4_gemm",
    "deepseek_fp8_gemm",
    "deepseek_gemm",
    "deepseek_grouped_bf16_gemm",
    "deepseek_grouped_fp8_fp4_gemm",
    "deepseek_grouped_fp8_gemm",
    "deepseek_hc_prenorm_gemm",
    "deepseek_mega_moe",
    "deepseek_moe_topk_gate",
    "deepseek_moe_topk_gate_backward",
    "deepseek_mqa_logits",
    "deepseek_normalize_routing_weights",
    "deepseek_paged_mqa_logits_metadata",
    "deepseek_per_block_cast",
    "deepseek_per_token_cast",
    "deepseek_select_stride_requirement",
    "deepseek_select_topk",
    "deepseek_swiglu_backward",
    "deepseek_swiglu_forward",
    "deepseek_tile_kernel",
    "deepseek_topk_gate",
    "deepseek_transform_mega_moe_weights",
    "deepseek_transform_scaling_factors",
    "is_deepseek_backend_available",
]

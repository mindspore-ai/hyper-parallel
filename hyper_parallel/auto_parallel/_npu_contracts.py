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
"""Shape contracts of the NPU kernels HyperParallel's fused modules call, for the census.

A train.yaml's ``plan_overrides`` may replace Transformers' modules with
HyperParallel's fused ones (``replace_module``): its RMSNorm, its grouped-query
attention, its grouped experts.  Their kernels are ``torch_npu``'s, which the
census, running fake tensors on a host, cannot call.  :func:`npu_contracts`
stands in for ``torch_npu`` with a contract for each kernel those modules
call: an op returning outputs of the kernel's shapes, which saves for its
backward what the kernel's backward takes, as ``torch_npu``'s derivatives
state it.

- ``npu_rms_norm`` saves its input, its weight and the fp32 reciprocal of
  each row's root mean square;
- ``npu_swiglu`` saves its input, the gate and up projections side by side;
- ``npu_rotary_mul`` saves its input and the two rotary tables;
- ``npu_fusion_attention`` saves its queries, keys and values, its output
  and its two fp32 softmax statistics per head and token; the mask it also
  takes is one every layer shares, which no layer's census counts;
- ``npu_moe_token_permute`` saves its tokens, their experts and the order it
  sorts them in, and ``npu_moe_token_unpermute`` the expert outputs, that
  order and the routing weights;
- ``npu_grouped_matmul`` runs inside HyperParallel's own autograd functions,
  which save what their backward takes, so its contract only shapes its
  output.

Any other kernel raises when called, naming it.

A fused module may cache a constant the first time it runs, as the attention
caches the compressed causal mask its sparse mode takes, one per device.
Under the census's fake tensors that constant is a fake tensor, which a real
call would then take from the cache, so leaving the contracts drops what the
census left (:func:`_drop_fake_caches`, F55).
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.machinery
import sys
import types
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import FakeTensor  # pylint: disable=forbidden-backend-import

# The softmax statistics the fused attention keeps per head and token.
_STATS = 8

# What ``sys.modules`` held for ``torch_npu`` when nothing did.
_ABSENT = object()


def _rms_norm_outputs(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """The normalized rows, and each row's fp32 reciprocal root mean square."""
    del gamma, epsilon
    return torch.empty_like(x), x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)


_RMS_NORM = torch.library.custom_op(
    "nd_census_npu::rms_norm", mutates_args=(),
    schema="(Tensor x, Tensor gamma, float epsilon) -> (Tensor, Tensor)")(_rms_norm_outputs)
_RMS_NORM.register_fake(_rms_norm_outputs)


def _rms_norm_saves(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the input, the weight and the reciprocal root mean squares."""
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(inputs[0], inputs[1], output[1])


def _rms_norm_backward(ctx: Any, *grads: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
    """Gradients of the input's and the weight's shapes."""
    del grads
    x, gamma, _ = ctx.saved_tensors
    return torch.empty_like(x), torch.empty_like(gamma), None


_RMS_NORM.register_autograd(_rms_norm_backward, setup_context=_rms_norm_saves)


def _swiglu_output(x: torch.Tensor, dim: int) -> torch.Tensor:
    """The gated half of *x*'s width along *dim*."""
    shape = list(x.shape)
    shape[dim] //= 2
    return x.new_empty(shape)


_SWIGLU = torch.library.custom_op(
    "nd_census_npu::swiglu", mutates_args=(), schema="(Tensor x, int dim) -> Tensor")(_swiglu_output)
_SWIGLU.register_fake(_swiglu_output)
_SWIGLU.register_autograd(
    lambda ctx, grad: (torch.empty_like(ctx.saved_tensors[0]), None),
    setup_context=lambda ctx, inputs, output: ctx.save_for_backward(inputs[0]))


def _rotary_output(x: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """The rotated input, of its shape."""
    del r1, r2
    return torch.empty_like(x)


_ROTARY = torch.library.custom_op(
    "nd_census_npu::rotary_mul", mutates_args=(), schema="(Tensor x, Tensor r1, Tensor r2) -> Tensor")(_rotary_output)
_ROTARY.register_fake(_rotary_output)
_ROTARY.register_autograd(
    lambda ctx, grad: tuple(torch.empty_like(tensor) for tensor in ctx.saved_tensors),
    setup_context=lambda ctx, inputs, output: ctx.save_for_backward(*inputs))


def _attention_outputs(query: torch.Tensor, key: torch.Tensor,
                       value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fused attention's output, as wide as the values, and its two fp32 statistics, BNSD."""
    del key
    batch, heads, seq, _ = query.shape
    stats = [query.new_empty((batch, heads, seq, _STATS), dtype=torch.float32) for _ in range(2)]
    return query.new_empty((batch, heads, seq, value.shape[-1])), stats[0], stats[1]


_ATTENTION = torch.library.custom_op(
    "nd_census_npu::fusion_attention", mutates_args=(),
    schema="(Tensor query, Tensor key, Tensor value) -> (Tensor, Tensor, Tensor)")(_attention_outputs)
_ATTENTION.register_fake(_attention_outputs)


def _attention_saves(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the queries, keys and values, the output and the statistics."""
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*inputs, *output)


_ATTENTION.register_autograd(
    lambda ctx, *grads: tuple(torch.empty_like(tensor) for tensor in ctx.saved_tensors[:3]),
    setup_context=_attention_saves)


def _permute_outputs(tokens: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Each token once per expert it routes to, expert by expert, and the order they were sorted in."""
    routes = indices.numel()
    return tokens.new_empty((routes, tokens.shape[-1])), indices.new_empty((routes,), dtype=torch.int32)


_PERMUTE = torch.library.custom_op(
    "nd_census_npu::moe_token_permute", mutates_args=(),
    schema="(Tensor tokens, Tensor indices) -> (Tensor, Tensor)")(_permute_outputs)
_PERMUTE.register_fake(_permute_outputs)


def _permute_saves(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the tokens, their experts and the sorting order."""
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(inputs[0], inputs[1], output[1])


_PERMUTE.register_autograd(
    lambda ctx, *grads: (torch.empty_like(ctx.saved_tensors[0]), None), setup_context=_permute_saves)


def _unpermute_output(permuted: torch.Tensor, sorted_indices: torch.Tensor,
                      probs: Optional[torch.Tensor]) -> torch.Tensor:
    """Each token's routes combined back into one row per token."""
    tokens = probs.shape[0] if probs is not None else sorted_indices.numel()
    return permuted.new_empty((tokens, permuted.shape[-1]))


_UNPERMUTE = torch.library.custom_op(
    "nd_census_npu::moe_token_unpermute", mutates_args=(),
    schema="(Tensor permuted, Tensor sorted_indices, Tensor? probs) -> Tensor")(_unpermute_output)
_UNPERMUTE.register_fake(_unpermute_output)


def _unpermute_saves(ctx: Any, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
    """Keep the expert outputs, the sorting order and the routing weights."""
    del output
    ctx.has_probs = inputs[2] is not None
    ctx.save_for_backward(*(tensor for tensor in inputs if tensor is not None))


def _unpermute_backward(ctx: Any, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
    """Gradients of the expert outputs' and the routing weights' shapes."""
    del grad
    saved = ctx.saved_tensors
    return torch.empty_like(saved[0]), None, torch.empty_like(saved[2]) if ctx.has_probs else None


_UNPERMUTE.register_autograd(_unpermute_backward, setup_context=_unpermute_saves)


def _v41_sparse_output(query: torch.Tensor, key_value: torch.Tensor, sparse_indices: torch.Tensor,
                       sinks: torch.Tensor, rope_head_dim: int, scale: float) -> torch.Tensor:
    """The sparse attention's output, a row per token and head, as the kernel's caller reshapes it."""
    del key_value, sparse_indices, sinks, rope_head_dim, scale
    batch, heads, seq, head_dim = query.shape
    return query.new_empty((batch, seq, heads, head_dim))


_V41_SPARSE = torch.library.custom_op(
    "nd_census_npu::v41_sparse_attention", mutates_args=(),
    schema=("(Tensor query, Tensor key_value, Tensor sparse_indices, Tensor sinks, "
            "int rope_head_dim, float scale) -> Tensor"))(_v41_sparse_output)
_V41_SPARSE.register_fake(_v41_sparse_output)


def _v41_sparse_saves(ctx: Any, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
    """Keep what the kernel's own caller keeps, in the shapes it keeps them.

    Copied from the nine tensors
    ``_NpuSparseAttentionWithScalarSink.forward`` hands
    ``ctx.save_for_backward``: the queries and the padded keys in the
    kernel's token-major layout, the two auxiliary rotary coordinates that
    carry the sink, the selected indices with the sink's column appended,
    the combined softmax maximum and the corrected sum, the output, and the
    sink's share of each row's mass.  The statistics are one scalar a token
    and head, not the eight-wide pair the dense fused attention keeps,
    because that caller views them as (batch, tokens, heads).
    """
    # The last input, the softmax scale, shapes nothing the kernel saves.
    query, key_value, sparse_indices, sinks, rope_head_dim = inputs[:5]
    batch, heads, seq, head_dim = query.shape
    keys, selected = key_value.shape[2], sparse_indices.shape[-1]
    # The gradients go back in the shapes the caller passed, which are not
    # the token-major ones the kernel is given.
    ctx.gradients = tuple((tuple(tensor.shape), tensor.dtype) for tensor in (query, key_value, sinks))
    # The caller pads the keys so every one of them is selected at most
    # once and the sink still has a row of its own.
    padded = max(keys + 1, selected + 2)
    rope = int(rope_head_dim)
    stats = [query.new_empty((batch * seq, heads, 1), dtype=torch.float32) for _ in range(2)]
    ctx.save_for_backward(
        query.new_empty((batch * seq, heads, head_dim)),
        key_value.new_empty((batch * padded, 1, head_dim)),
        query.new_empty((batch * seq, heads, rope)),
        key_value.new_empty((batch * padded, 1, rope)),
        sparse_indices.new_empty((batch * seq, 1, selected + 1), dtype=torch.int32),
        stats[0], stats[1], output,
        query.new_empty((batch, seq, heads), dtype=torch.float32),
    )


def _v41_sparse_backward(ctx: Any, grad: torch.Tensor) -> Tuple[Any, ...]:
    """A gradient for the queries, the keys and the sinks, in the shapes they came in."""
    gradients = [torch.empty(shape, dtype=dtype, device=grad.device) for shape, dtype in ctx.gradients]
    return gradients[0], gradients[1], None, gradients[2], None, None


_V41_SPARSE.register_autograd(_v41_sparse_backward, setup_context=_v41_sparse_saves)


def npu_sparse_attention_with_scalar_sink(query: torch.Tensor, key_value: torch.Tensor,
                                          sparse_indices: torch.Tensor, sinks: torch.Tensor,
                                          rope_head_dim: int, scale: float) -> torch.Tensor:
    """The contract of DeepSeek-V4.1's sparse attention, Omni's kernel and the sink it encodes.

    Stands in for
    ``components/modules/shared_compressed_dsa_attention.npu_sparse_attention_with_scalar_sink``,
    whose kernel is ``omni_training_custom_ops``' and has no torch path a
    run takes: the reference beside it keeps an fp32 score of one sequence
    by another, which is not what the run keeps (F53).
    """
    return _V41_SPARSE(query, key_value, sparse_indices, sinks, int(rope_head_dim), float(scale))


def npu_rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch_npu.npu_rms_norm``'s contract."""
    return _RMS_NORM(x, gamma, float(epsilon))


def npu_swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``torch_npu.npu_swiglu``'s contract."""
    return _SWIGLU(x, int(dim) % x.dim())


def npu_rotary_mul(x: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = "half") -> torch.Tensor:
    """``torch_npu.npu_rotary_mul``'s contract."""
    del rotary_mode
    return _ROTARY(x, r1, r2)


def npu_fusion_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, head_num: int,
                         input_layout: str, *args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """``torch_npu.npu_fusion_attention``'s contract, for the BNSD and BSND layouts.

    Returns the kernel's seven outputs: the attention's, the two statistics,
    an empty softmax output, and the dropout's seed, offset and count.

    Raises:
        NotImplementedError: For another layout.
    """
    del head_num, args, kwargs
    if input_layout not in ("BNSD", "BSND"):
        raise NotImplementedError(f"the census has no contract for npu_fusion_attention's {input_layout} layout")
    bnsd = input_layout == "BNSD"
    tensors = [tensor if bnsd else tensor.transpose(1, 2) for tensor in (query, key, value)]
    output, softmax_max, softmax_sum = _ATTENTION(*tensors)
    return (output if bnsd else output.transpose(1, 2), softmax_max, softmax_sum,
            query.new_empty((0,)), 0, 0, 0)


def npu_moe_token_permute(tokens: torch.Tensor, indices: torch.Tensor, num_out_tokens: Optional[int] = None,
                          padded_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch_npu.npu_moe_token_permute``'s contract, every route kept."""
    del num_out_tokens, padded_mode
    return _PERMUTE(tokens, indices)


def npu_moe_token_unpermute(permuted_tokens: torch.Tensor, sorted_indices: torch.Tensor,
                            probs: Optional[torch.Tensor] = None, padded_mode: bool = False,
                            restore_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
    """``torch_npu.npu_moe_token_unpermute``'s contract."""
    del padded_mode, restore_shape
    return _UNPERMUTE(permuted_tokens, sorted_indices, probs)


def npu_grouped_matmul(x: Sequence[torch.Tensor], weight: Sequence[torch.Tensor], *args: Any,
                       group_list: Any = None, group_type: int = 0, **kwargs: Any) -> List[torch.Tensor]:
    """``torch_npu.npu_grouped_matmul``'s output shape, one tensor of a single split.

    Grouped along the rows (``group_type`` 0), each group's rows times its
    weight; along the reduced dimension (2, a weight's gradient), one
    product per group.
    """
    del args, kwargs
    rows, weights = x[0], weight[0]
    if group_type == 2:
        groups = group_list.shape[0] if isinstance(group_list, torch.Tensor) else len(group_list)
        return [rows.new_empty((groups, rows.shape[0], weights.shape[-1]))]
    return [rows.new_empty((rows.shape[0], weights.shape[-1]))]


def _missing(name: str) -> Any:
    """What the stand-in answers for a kernel it has no contract for: one that raises when called."""
    if name.startswith("_"):
        raise AttributeError(name)

    def kernel(*args: Any, **kwargs: Any) -> Any:
        """Refuse the call, naming the kernel the census has no contract for."""
        del args, kwargs
        raise NotImplementedError(f"the census has no contract for torch_npu.{name}")

    return kernel


def _holders(bound: Any) -> List[types.ModuleType]:
    """HyperParallel's modules that hold *bound* as their ``torch_npu``."""
    return [module for name, module in list(sys.modules.items())
            if name.startswith("hyper_parallel.") and bound is not _ABSENT
            and getattr(module, "torch_npu", None) is bound]


def _stand_in() -> types.ModuleType:
    """A ``torch_npu`` module of the contracts."""
    module = types.ModuleType("torch_npu")
    module.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    for kernel in (npu_rms_norm, npu_swiglu, npu_rotary_mul, npu_fusion_attention, npu_moe_token_permute,
                   npu_moe_token_unpermute, npu_grouped_matmul):
        setattr(module, kernel.__name__, kernel)
    module.__getattr__ = _missing
    return module


TORCH_NPU = _stand_in()


def _drop_fake_caches() -> None:
    """Drop the fake tensors what ran under the contracts left in the runtime's caches.

    HyperParallel's fused modules cache a constant per device in a module of
    their own, such as the compressed causal mask the NPU attention's sparse
    mode takes (``models/qwen3*/adapter/runtime/attention.py``).  A census fills those
    caches with fake tensors, which a real call in the same process would take
    from them, so every fake tensor a model module holds in a cache keyed by
    anything is dropped, leaving the caches as the census found them (F55).
    """
    for name, module in list(sys.modules.items()):
        if not name.startswith("hyper_parallel.models") or module is None:
            continue
        for value in list(vars(module).values()):
            if not isinstance(value, dict):
                continue
            for key in [key for key, held in value.items() if isinstance(held, FakeTensor)]:
                del value[key]


@contextlib.contextmanager
def npu_contracts() -> Iterator[types.ModuleType]:
    """``torch_npu`` as the contracts, for what runs inside.

    A module that imports ``torch_npu`` inside, as HyperParallel's fused
    modules do, keeps the contracts; one that had imported the real
    ``torch_npu`` before calls the contracts until the context ends.
    HyperParallel's model package, whose build options probe the device when
    ``torch_npu`` imports, is imported before.
    """
    importlib.import_module("hyper_parallel.models")
    previous = sys.modules.get("torch_npu", _ABSENT)
    sys.modules["torch_npu"] = TORCH_NPU
    for module in _holders(previous):
        module.torch_npu = TORCH_NPU
    try:
        yield TORCH_NPU
    finally:
        _drop_fake_caches()
        if previous is _ABSENT:
            sys.modules.pop("torch_npu", None)
        else:
            sys.modules["torch_npu"] = previous
            # Including a module that imported the stand-in while it stood in.
            for module in _holders(TORCH_NPU):
                module.torch_npu = previous


# Where DeepSeek-V4.1's sparse attention lives, and the two names in it that
# decide which path a call takes.
_V41_ATTENTION = "hyper_parallel.components.modules.shared_compressed_dsa_attention"
_V41_OMNI = "npu_sparse_attention_with_scalar_sink"
_V41_REFERENCE = "_reference_sparse_attention"


@contextlib.contextmanager
def v41_attention_contract(rope_head_dim: int) -> Iterator[bool]:
    """Legacy Omni sparse attention as its contract, for what runs inside.

    A V4.1 layer picks its attention by the device it is running on
    (``self._use_omni_attention and query.device.type == "npu"``), so a
    census, whose tensors are fake and on no device, always takes the
    reference beside the kernel.  That reference keeps an fp32 score of one
    sequence by another where the kernel keeps nine tensors that follow the
    sequence, so measuring it prices a layer no run builds: on the
    validation crop at 4096 tokens it read 77.9 GiB a rank against a 38.5
    GiB measured peak (F53).

    This contract applies only while the module exposes the Omni and legacy
    reference entry points. Newer adapters have an explicit fusion switch;
    their eager path must be measured as written, not priced as Omni. Their
    native kernels need their own saved-tensor contracts.

    Both legacy names are bound to the contract for as long as the context lasts,
    so either branch of that choice reaches it.  *rope_head_dim* is the
    width of the auxiliary rotary coordinates the kernel's caller passes and
    the reference's own signature does not carry, so the census states it
    from the model's config.

    Yields:
        Whether the legacy Omni entry points were present to patch.
    """
    try:
        module = importlib.import_module(_V41_ATTENTION)
    except ImportError:
        yield False
        return

    if not all(hasattr(module, name) for name in (_V41_OMNI, _V41_REFERENCE)):
        yield False
        return

    def contract(query: torch.Tensor, key_value: torch.Tensor, sparse_indices: torch.Tensor,
                 sinks: torch.Tensor, scale: float) -> torch.Tensor:
        """The reference's signature, answered by the kernel's contract."""
        return npu_sparse_attention_with_scalar_sink(
            query, key_value, sparse_indices, sinks, int(rope_head_dim), float(scale))

    held = {name: getattr(module, name) for name in (_V41_OMNI, _V41_REFERENCE)}
    setattr(module, _V41_REFERENCE, contract)
    setattr(module, _V41_OMNI, npu_sparse_attention_with_scalar_sink)
    try:
        yield True
    finally:
        for name, value in held.items():
            setattr(module, name, value)


# The Lightning Indexer's selection functions, which return the chosen
# indices and nothing else.  One of the four already states that it needs no
# autograd graph; the other three do not, which is the whole of this contract.
_V41_SELECTION = (
    "compressed_causal_topk",
    "compressed_causal_candidates",
    "compressed_causal_topk_and_candidates",
    "compressed_candidate_topk",
)


@contextlib.contextmanager
def v41_indexer_contract() -> Iterator[bool]:
    """DeepSeek-V4.1's Indexer selection as its fused operator keeps it, which is nothing.

    The selection returns int32 indices, and the Indexer's own KL loss
    differentiates itself by hand, under ``no_grad``, from the projections
    rather than through the selection.  So no gradient flows through the
    scores, and the fused operator keeps nothing for a backward.  Running the
    selection without recording a graph states exactly that, and needs no
    shape declared, because ``no_grad`` changes whether a graph is kept and
    never what is computed: the indices are the same indices.

    It holds on either path.  The reference selection records a graph behind
    its fp32 scores, but that graph ends at the int32 indices, so a run frees
    it as soon as they are taken and holds the same bytes whichever path it
    takes.  The census once counted that graph, 8.9 GiB a rank on the
    validation crop at 4096 tokens across the eight layers that carry an
    Indexer, because its own pack hook held every graph a forward drops
    alive (D16, fixed in ``_measure``).  Measured since on that crop, the
    reference path holds 26.542 GiB a rank at 4096 tokens with this contract
    and without it, so it changes no figure the census reports and only
    spares the census recording a graph no run keeps.

    Of the four, ``compressed_causal_topk`` already carries
    ``@torch.no_grad()``.  The other three record the graph and drop it,
    which costs a run nothing it holds once the selection returns.

    Yields:
        Whether the selection was there to wrap, which is False on a tree
        that carries no V4.1 attention.
    """
    try:
        module = importlib.import_module(_V41_ATTENTION)
    except ImportError:
        yield False
        return

    def without_a_graph(selection: Any) -> Any:
        """*selection*, computing the same indices without recording a graph."""
        @functools.wraps(selection)
        def run(*args: Any, **kwargs: Any) -> Any:
            """The selection's own indices, with no graph kept behind them."""
            with torch.no_grad():
                return selection(*args, **kwargs)
        return run

    held = {name: getattr(module, name) for name in _V41_SELECTION if hasattr(module, name)}
    for name, selection in held.items():
        setattr(module, name, without_a_graph(selection))
    try:
        yield bool(held)
    finally:
        for name, selection in held.items():
            setattr(module, name, selection)

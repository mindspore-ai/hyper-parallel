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
"""Census of what a layer of each kind keeps for its backward, on a fake layer.

The census builds one decoder layer of a Transformers model under
``FakeTensorMode``, so it allocates no memory, runs its forward and its
backward at micro-batch 1, and counts bytes: those the autograd graph saves
between the two passes, and the most the backward holds live at once, less
the parameters and their gradients, which the memory model counts on its
own.  It runs the kernels the runtime runs: flash attention, which saves its
inputs, its output and two fp32 softmax statistics per head and token, and
HyperParallel's chunked gated-delta-rule kernel, which saves the queries,
keys, values, the cumulated gates, beta and one chunk matrix.  A layer built
with half the heads and half the feed-forward widths, as a tensor-parallel
rank of 2 holds it, tells the bytes tensor parallelism splits from those it
does not.  Run again under HyperParallel's selective activation checkpointing,
its own checkpoint and the policy the installed trainer runs, a layer keeps
its input and the outputs that policy saves: every other matmul's, the
convolution's, and the attention kernel's where the policy saves the
operator the census's kernel stands for (:func:`_policy_saves`); the matmuls
it does not save its backward runs again, a share of the attention's
projections and of the rest of the layer.

What a layer saves is also told apart by the op of ND's op vector whose
backward takes it (:class:`_SavedOps`), the records of shared decision S1 as
the census fills them: the delta rule's, the attention kernel's (its
inputs and output for the batched matmuls, its fp32 statistics for the
softmax), a dropout's mask, what a norm saves, what the feed-forward's
activation function saves, and the rest by the part of the layer saving it,
the attention's projections or the feed-forward's.

The output layer's census runs the final norm, the output projection and
Transformers' causal-LM loss, which casts the logits to fp32, as
HyperParallel's trainer runs them by default, dropping the model's logits
before the backward as the trainer does; half the vocabulary tells the
bytes a vocabulary-parallel loss splits.

Where a train.yaml replaces Transformers' modules with HyperParallel's
fused ones (``plan_overrides`` with ``replace_module``: its RMSNorm, its
grouped-query attention, its grouped experts), the census runs those
instead, so what it measures is what the trainer saves
(:func:`replacement_specs`).  Their kernels are ``torch_npu``'s, which a
host cannot call, so they run under the kernels' shape contracts
(:mod:`hyper_parallel.auto_parallel._npu_contracts`).  A factory that needs
a library the host lacks leaves the census with a layer the run does not
build, which it refuses (:exc:`CensusUnavailable`): Transformers' own
modules keep other bytes, and under the selective policy a DeepSeek-V3.2
layer's eager indexer keeps 2.3 times what the layer keeps plain.  The FLOP census
(:func:`census_flops`) and the shares of matmul FLOPs a selective layer
recomputes (:func:`census_recomputed`) stay on Transformers' own modules:
both count the arithmetic the time model prices, which fusing a module
does not change, and a fused kernel runs matmuls no dispatch sees.

A model whose layers cannot run alone is measured over its whole stack
(:func:`census_stack`): DeepSeek-V4.1's layers read what earlier layers
wrote in the same forward, and carry a hyper-connection dimension and a
mixing state between them.  The model the trainer builds runs its own forward
and backward once, each save counting toward the layer that makes it, and a
kind's record is the mean of its layers'.

A layer's parameters are counted by part, as ND prices its parts: the
attention's, the norms', the dense feed-forward's, the routed experts', the
shared expert's and the router's (:func:`census_parameters`), for verify
mode to set beside what ND prices.  So are its forward FLOPs
(:func:`census_flops`): each matmul's toward the part of the layer that
runs it, flash attention's toward the scores, at every pair of tokens, as
the runtime's kernel computes them under an explicit causal mask, and the
gated delta rule's through Transformers' own chunked implementation, whose
matmuls the runtime's kernel runs alike.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import importlib
import inspect
import itertools
import logging
import weakref
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # pylint: disable=forbidden-backend-import
from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=forbidden-backend-import
from torch.utils._pytree import tree_flatten  # pylint: disable=forbidden-backend-import
from torch.utils.checkpoint import DefaultDeviceType  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.auto_parallel._npu_contracts import npu_contracts
from hyper_parallel.core.activation_memory import api as activation_memory
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy

logger = logging.getLogger(__name__)


class CensusUnavailable(RuntimeError):
    """A census cannot run here: this host cannot build the modules the run installs."""


class LayerTraffic(NamedTuple):
    """What one forward of a layer moves, by part: its bytes, the parameters among them, and its ops.

    Bytes of the whole layer on a micro-batch of one sequence, as
    :func:`census_flops` counts its FLOPs, so that the two quantities can be
    set beside each other part by part.  ``parameter`` is the share of
    ``moved`` that is the layer's own weights, which a strategy shards where
    it leaves the activations alone, and ``launches`` is how many ops the
    forward dispatches, views excluded.
    """

    moved: Dict[str, float]
    parameter: Dict[str, float]
    launches: Dict[str, int]

# The Transformers attention implementation the census registers its flash
# attention under: a name Transformers does not take for a flash
# attention's, as it does not take HyperParallel's default, sdpa, so a model
# runs sdpa's path, and DeepSeek-V3's values are not padded to its queries'
# width.
_FLASH = "nd_census_attention"

# The config fields a tensor-parallel rank holds a share of.
_TP_FIELDS = (
    "num_attention_heads", "num_key_value_heads", "intermediate_size", "moe_intermediate_size",
    "shared_expert_intermediate_size", "linear_num_key_heads", "linear_num_value_heads",
)

# The softmax statistics flash attention keeps per head and token.
_FLASH_STATS = 8

# The gated delta rules a run can take, named as the runtime names them
# (``components/modules/gated_delta_net.py``): Transformers' own chunked
# implementation, and HyperParallel's kernel, whose saved set the census holds
# as a contract.  The runtime defaults to eager and so does the census.
_EAGER_GDN = "eager"
_HYPER_GDN = "triton"
_GDN_BACKENDS = (_EAGER_GDN, _HYPER_GDN)

# The census's attention kernels, each beside the runtime operator it stands
# in for, named as HyperParallel's selective policy names the operators it
# saves: its own flash attention for PyTorch's scaled-dot-product attention,
# and the contract of the kernel HyperParallel's fused attention calls,
# ``torch_npu.npu_fusion_attention``.  A selective census saves what a kernel
# returns exactly where the installed policy saves what its operator returns
# (:func:`_policy_saves`).
_ATTENTION_KERNELS = {
    "nd_census::flash_attention": "aten.scaled_dot_product_attention",
    "nd_census_npu::fusion_attention": "npu.npu_fusion_attention",
}

# The per-head query and key norms an attention holds, which ND prices with
# the layer's norms.
_QK_NORMS = ("q_norm", "k_norm")


def _flash_outputs(query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash attention's outputs, shapes only: each head's, as wide as its values, and two fp32 statistics."""
    del key
    batch, heads, seq, _ = query.shape
    stats = [torch.empty(batch, heads, seq, _FLASH_STATS, dtype=torch.float32, device=query.device)
             for _ in range(2)]
    return query.new_empty(batch, heads, seq, value.shape[-1]), stats[0], stats[1]


# Flash attention as one op, as the runtime's kernel is: HyperParallel's
# selective policy saves what the attention kernels it lists return, and
# saves this one's too.
_FLASH_KERNEL = torch.library.custom_op(
    "nd_census::flash_attention", mutates_args=(),
    schema="(Tensor query, Tensor key, Tensor value) -> (Tensor, Tensor, Tensor)")(_flash_outputs)
_FLASH_KERNEL.register_fake(_flash_outputs)


def _flash_saves(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the inputs, the output and the statistics, which no gradient reaches."""
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*inputs, *output)


def _flash_backward(ctx: Any, *grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of the inputs' shapes."""
    del grads
    query, key, value = ctx.saved_tensors[:3]
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


_FLASH_KERNEL.register_autograd(_flash_backward, setup_context=_flash_saves)


def _flash_attention(module, query, key, value, attention_mask, **kwargs):  # pylint: disable=unused-argument
    """A Transformers attention function running the census's flash attention, grouped K and V unexpanded."""
    return _FLASH_KERNEL(query, key, value)[0].transpose(1, 2).contiguous(), None


class _GatedDeltaRule(torch.autograd.Function):
    """HyperParallel's chunked gated-delta-rule kernel's saved set, shapes only."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor,  # pylint: disable=arguments-differ
                value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor, chunk: int) -> torch.Tensor:
        """Save the inputs, the cumulated gates, beta and a chunk matrix; return an output of the values' shape."""
        batch, seq, heads, _ = key.shape
        gates = torch.empty(batch, seq, heads, dtype=torch.float32, device=query.device)
        chunks = torch.empty(batch, seq, heads, chunk, dtype=key.dtype, device=query.device)
        ctx.save_for_backward(query, key, value, gates, beta, chunks)
        del gate
        return torch.empty_like(value)

    @staticmethod
    def backward(ctx: Any,  # pylint: disable=arguments-differ
                 grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Return gradients of the inputs' shapes, and none of the chunk size."""
        del grad
        query, key, value, gates, beta, _ = ctx.saved_tensors
        return (torch.empty_like(query), torch.empty_like(key), torch.empty_like(value),
                torch.empty_like(gates), torch.empty_like(beta), None)


def _gated_delta_rule(modeling: Any) -> Callable[..., Tuple[torch.Tensor, None]]:
    """The drop-in for a Transformers ``chunk_gated_delta_rule`` running :class:`_GatedDeltaRule`."""
    l2norm = getattr(modeling, "l2norm", None)

    def run(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
            chunk_size: int = 64, use_qk_l2norm_in_kernel: bool = False,
            **_kwargs: Any) -> Tuple[torch.Tensor, None]:
        """Run the kernel's contract as Transformers calls its kernel, with no final state."""
        if use_qk_l2norm_in_kernel and l2norm is not None:
            query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
        return _GatedDeltaRule.apply(query, key, value, g, beta, chunk_size), None

    return run


# Reading a scalar off a fake tensor raises, since it holds no data. Transformers'
# grouped_mm fallback reads its group offsets so in its backward, and it takes that
# fallback on CPU with torch 2.8 or older, the 910C's 2.7.1 among them.
_SCALAR_READ = torch.ops.aten._local_scalar_dense.default  # pylint: disable=protected-access


class _LiveBytes(TorchDispatchMode):
    """The storages the ops it sees allocate, and the bytes they hold while they live."""

    def __init__(self) -> None:
        """Start with no storage counted."""
        super().__init__()
        self.sizes: Dict[Tuple[int, int], int] = {}
        self.generation: Dict[int, int] = {}
        self.events: list = []

    def key(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """The storage behind *tensor*, in the lifetime it has now."""
        address = tensor.untyped_storage()._cdata  # pylint: disable=protected-access
        return address, self.generation.get(address, 0)

    def track(self, tensor: torch.Tensor) -> None:
        """Count *tensor*'s storage from now until it is freed."""
        storage = tensor.untyped_storage()
        key = self.key(tensor)
        if key in self.sizes:
            return
        self.sizes[key] = storage.nbytes()
        self.events.append((key, self.sizes[key]))
        weakref.finalize(storage, self._free, key)

    def _free(self, key: Tuple[int, int]) -> None:
        size = self.sizes.pop(key, None)
        self.generation[key[0]] = key[1] + 1
        if size:
            self.events.append((key, -size))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        if func is _SCALAR_READ and isinstance(args[0], FakeTensor):
            return 0  # shapes alone price the memory, whatever the scalar
        out = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor):
                self.track(tensor)
        return out

    def peak(self, start: int, excluded: Iterable[Tuple[int, int]], apart: Iterable[Tuple[int, int]] = ()) -> int:
        """The bytes live when the most are, from event *start* on, leaving out the *excluded* storages.

        The storages *apart* count toward finding that moment, not toward
        the bytes returned.
        """
        excluded, apart = set(excluded), set(apart)
        live = held = 0
        for key, delta in self.events[:start]:
            if key not in excluded:
                live += delta
                held += delta if key in apart else 0
        top, bytes_at_top = live, live - held
        for key, delta in self.events[start:]:
            if key in excluded:
                continue
            live += delta
            held += delta if key in apart else 0
            if live > top:
                top, bytes_at_top = live, live - held
        return bytes_at_top


@contextlib.contextmanager
def _checkpoints_on_cpu() -> Iterator[None]:
    """Run activation checkpointing as a CPU's while a census runs.

    A census's tensors are fake and on the CPU, so a checkpoint over them
    takes the default device type, which ``torch_npu`` sets to its own when
    it imports.  Replaying the forward would then open the NPU's autocast
    and start a device the census never uses, and that a host whose NPU
    cannot start fails on.  The default is put back on the way out.
    """
    previous = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        yield
    finally:
        DefaultDeviceType.set_device_type(previous)


def _measure(params: Sequence[torch.Tensor], inputs: Iterable[torch.Tensor],
             forward: Callable[[], torch.Tensor], backward: Callable[[torch.Tensor], None],
             grad_inputs: Iterable[torch.Tensor] = (), checkpointed: bool = False,
             shared: Iterable[torch.Tensor] = (), ops: Optional["_SavedOps"] = None) -> Tuple[int, int]:
    """Bytes a forward saves for its backward, and the activations the backward holds when it holds the most.

    Args:
        params: The parameters of what runs.
        inputs: The tensors the forward takes, which count from its start.
        forward: Runs the forward and returns its output.
        backward: Runs the backward from that output.
        grad_inputs: The tensors the backward takes, which count from its
            start.
        checkpointed: Whether the forward runs under activation
            checkpointing, whose own hooks hold what it keeps.
        shared: Tensors the model shares between its layers, which a
            checkpointed forward's count leaves out.
        ops: Where given, tells each storage the forward saves apart by
            the op that first saves it, counting its bytes into
            ``ops.saved``, which then sums to the bytes returned.

    Returns:
        The bytes the forward saves and still holds once it has run, its
        inputs included, and the bytes of activations live when the
        backward peaks, its output left out.  A graph the forward drops
        before it ends, as a selection that keeps only the indices it picks
        drops its scores, is freed then in a run and is no backward's.  A
        checkpointed forward keeps what is live once it has run, but for
        the *shared* tensors: its inputs and the outputs its policy saves.
        The parameters, which the forward's views of them bring into the
        tracker, are the memory model's to count, and so are their
        gradients, which hooks on the parameters tell apart from the
        activations' however they are shaped: the gradients the backward
        hands each parameter and the one it keeps.  They settle when the
        backward peaks, and the bytes returned are the activations held
        then.
    """
    stored = {p.untyped_storage()._cdata for p in params}  # pylint: disable=protected-access
    # Counted by the storage in the lifetime it has now, as the tracker counts
    # them: a fake tensor's storage is cheap and short lived, so one the
    # forward has freed can hand its address to a later save, and counting by
    # address alone lost whichever of the two it saw first.  Which pair
    # collided depended on what had run before, so a layer did not measure the
    # same twice.
    saved: Dict[Tuple[int, int], int] = {}
    saved_op: Dict[Tuple[int, int], str] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Count a tensor autograd saves, unless it is a parameter's, toward the op that first saves it."""
        # Named before the storage is read, which dispatches ops of its own.
        op = ops.op(tensor) if ops is not None else None
        address = tensor.untyped_storage()._cdata  # pylint: disable=protected-access
        if address not in stored:
            key = live.key(tensor)
            if op is not None:
                saved_op.setdefault(key, op)
            saved[key] = tensor.untyped_storage().nbytes()
        # Detached, so a graph holds the bytes it saves and not the tensor: an
        # op saving its own output would otherwise hold itself through that
        # output's grad_fn, a cycle no collector sees, which kept every graph
        # a forward drops alive into the backward.  On the DeepSeek-V4.1 crop
        # at 4096 tokens that was the Indexer selection's scores, 8.9 GiB a
        # rank a run frees as soon as the indices are taken.
        return tensor.detach()

    live = _LiveBytes()
    grads: List[Tuple[int, int]] = []
    handles = [p.register_hook(lambda grad: grads.append(live.key(grad))) for p in params]
    try:
        with _checkpoints_on_cpu(), live, ops or contextlib.nullcontext(), \
                torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            for tensor in inputs:
                live.track(tensor)
            out = forward()
            start = len(live.events)
            left_out = {live.key(out), *(live.key(p) for p in params)}
            not_kept = left_out | {live.key(tensor) for tensor in shared}
            kept = sum(size for key, size in live.sizes.items() if key not in not_kept)
            held = {key: size for key, size in saved.items() if key in live.sizes}
            for tensor in grad_inputs:
                live.track(tensor)
            backward(out)
    finally:
        for handle in handles:
            handle.remove()
    # A storage's bytes count toward the op that first saved it, read from the
    # same record as the total, so the two agree whatever a fake tensor does
    # with a storage address it has freed.
    for address, size in held.items():
        op = saved_op.get(address)
        if op is not None:
            ops.saved[op] = ops.saved.get(op, 0) + size
    grads += [live.key(p.grad) for p in params if p.grad is not None]
    return kept if checkpointed else sum(held.values()), live.peak(start, left_out, grads)


# The ops a matmul runs as.
_MATMULS = ("mm", "addmm", "bmm", "_grouped_mm", "baddbmm")


class _SavedOps(TorchDispatchMode):
    """What each tensor a layer's forward saves is saved for: the op of ND's op vector whose backward takes it.

    The delta rule's saves are its own (``linrec``); the attention kernel's
    are its batched matmuls' (``attBMM``), its fp32 statistics the
    softmax's; a dropout's mask is its own; what a norm saves, the norm's
    (``normOp``), and what the feed-forward's activation function saves,
    the function's (``ffAct``).  The rest counts toward the part of the
    layer saving it: the attention's projections (``attMM``), the
    feed-forward's (``ffMM``), or ``other`` for the layer's own code.
    """

    def __init__(self, layer: Any) -> None:
        """Follow which of *layer*'s modules runs, and when its delta rule does."""
        super().__init__()
        self.saved: Dict[str, int] = {}
        self.running: List[str] = []
        self.func: Any = None
        self.delta_rule = False
        self.modules = dict(layer.named_modules())
        for name, module in self.modules.items():
            if name:
                module.register_forward_pre_hook(functools.partial(self._enter, name))
                module.register_forward_hook(self._leave)
            if hasattr(module, "chunk_gated_delta_rule"):
                module.chunk_gated_delta_rule = self._rule(module.chunk_gated_delta_rule)

    def _rule(self, rule: Callable[..., Any]) -> Callable[..., Any]:
        """*rule*, noting that it runs."""
        def run(*args: Any, **kwargs: Any) -> Any:
            """The delta rule, its saves told apart as its own while it runs."""
            self.delta_rule = True
            try:
                return rule(*args, **kwargs)
            finally:
                self.delta_rule = False
        return run

    def _enter(self, name: str, *_: Any) -> None:
        """Note that the module at *name* runs."""
        self.running.append(name)

    def _leave(self, *_: Any) -> None:
        """Note that the innermost module running has returned."""
        self.running.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        # A tensor's metadata is read through prims, and _measure detaches
        # each save, both between an op and its saves.
        if func.namespace != "prim" and func is not torch.ops.aten.detach.default:
            self.func = func
        return func(*args, **(kwargs or {}))

    def op(self, tensor: torch.Tensor) -> str:
        """The op that saves *tensor*, which the forward saves now."""
        if self.delta_rule:
            return "linrec"
        if self.func in _attention_ops():
            return "softmax" if tensor.dtype == torch.float32 else "attBMM"
        if self.func is torch.ops.aten.native_dropout.default:
            return "dropout"
        if self.func is torch.ops.nd_census_npu.rms_norm.default:
            return "normOp"
        if self.func is torch.ops.nd_census_npu.swiglu.default:
            return "ffAct"
        name = self.running[-1] if self.running else ""
        module = self.modules.get(name)
        if name and "norm" in type(module).__name__.lower():
            return "normOp"
        part = name.split(".", maxsplit=1)[0]
        if not part:
            return "other"
        attention = "attn" in part or "attention" in part
        if not attention and type(module).__module__ in ("transformers.activations", "torch.nn.modules.activation"):
            return "ffAct"
        return "attMM" if attention else "ffMM"


def _matmul_flops(func: Any, args: Sequence[Any]) -> int:
    """The FLOPs of *func* on *args* where it is a matmul, and 0 otherwise."""
    matmuls = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default)
    grouped = getattr(torch.ops.aten, "_grouped_mm", None)  # first shipped in torch 2.8
    if func not in (matmuls if grouped is None else (*matmuls, grouped.default)):
        return 0
    first, second = args[1:3] if func is torch.ops.aten.addmm.default else args[:2]
    return 2 * first.numel() * second.shape[-1]


class _RecomputedMatmuls:
    """The matmul FLOPs a layer's selective forward saves and leaves to run again, its attention's and the rest's."""

    def __init__(self, layer: Any) -> None:
        """Tell the attention's parameters, those of a child named for it, from the rest of *layer*'s."""
        self.attention = {param.untyped_storage()._cdata  # pylint: disable=protected-access
                          for name, param in layer.named_parameters()
                          if "attn" in name.split(".")[0] or "attention" in name.split(".")[0]}
        self.flops: Dict[Tuple[bool, bool], int] = {}

    def note(self, func: Any, args: Sequence[Any], policy: CheckpointPolicy) -> None:
        """Count the FLOPs of a matmul the policy decided on, by part and by whether it runs again."""
        flops = _matmul_flops(func, args)
        if flops:
            attention = any(isinstance(arg, torch.Tensor)
                            and arg.untyped_storage()._cdata in self.attention  # pylint: disable=protected-access
                            for arg in args)
            key = (attention, policy == CheckpointPolicy.MUST_RECOMPUTE)
            self.flops[key] = self.flops.get(key, 0) + flops

    def share(self, attention: bool) -> float:
        """The share of the attention's matmul FLOPs, or of the rest's, that runs again."""
        again = self.flops.get((attention, True), 0)
        total = again + self.flops.get((attention, False), 0)
        return again / total if total else 0.0


class _ForwardContext(NamedTuple):
    """What a selective policy reads of its checkpoint context, in the forward: no recompute is running."""

    is_recompute: bool = False


def _checkpointing() -> Any:
    """HyperParallel's activation checkpointing module, which holds the policy its trainer runs.

    Its distributed package takes as long to import as the rest of the cost
    model, so it loads only when a census runs.
    """
    return importlib.import_module("hyper_parallel.distributed.activation_checkpoint")


def _policy_saves(checkpointing: Any, name: str) -> bool:
    """Whether the installed selective policy saves what the runtime operator *name* returns.

    Where this host registers the operator, the policy decides on it as it
    does in a run.  An NPU operator is registered only where ``torch_npu``
    is installed: elsewhere the policy's own list of the operators it saves
    is read for its name, which is what a run on the NPU resolves.
    """
    # pylint: disable=protected-access
    operator = checkpointing._resolve_torch_op(name)
    if operator is None:
        return name in checkpointing._SELECTIVE_AC_COMPUTE_OP_NAMES
    decide = checkpointing._make_selective_checkpoint_policy_fn()
    return decide(_ForwardContext(), operator) == CheckpointPolicy.MUST_SAVE


def _census_policy(ledger: Optional[_RecomputedMatmuls] = None) -> Callable[..., CheckpointPolicy]:
    """The policy the installed trainer runs, deciding on the census's attention kernels as on their operators.

    A census kernel is saved where the policy saves the operator it stands
    for (:data:`_ATTENTION_KERNELS`), and recomputed where it does not, as
    the policy recomputes any operator it does not name.  Its forward
    decisions on matmuls go to *ledger*, if given.
    """
    checkpointing = _checkpointing()
    policy = checkpointing._make_selective_checkpoint_policy_fn()  # pylint: disable=protected-access
    saved = frozenset(_kernel_op(kernel) for kernel, name in _ATTENTION_KERNELS.items()
                      if _policy_saves(checkpointing, name))

    def census_policy(ctx: Any, func: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
        """The trainer's policy, which saves a census attention kernel where it saves the kernel's operator."""
        decision = CheckpointPolicy.MUST_SAVE if func in saved else policy(ctx, func, *args, **kwargs)
        if ledger is not None and not ctx.is_recompute:
            ledger.note(func, args, decision)
        return decision

    return census_policy


def _selective_contexts(ledger: Optional[_RecomputedMatmuls] = None) -> Tuple[Any, Any]:
    """HyperParallel's selective checkpointing contexts under :func:`_census_policy`."""
    return activation_memory.create_selective_checkpoint_contexts(_census_policy(ledger))


def _kernel_op(name: str) -> Any:
    """The census kernel *name*, ``namespace::kernel``, as an op."""
    namespace, kernel = name.split("::")
    return getattr(getattr(torch.ops, namespace), kernel).default


@functools.lru_cache(maxsize=None)
def _attention_ops() -> FrozenSet[Any]:
    """The census's attention kernels, as ops."""
    return frozenset(_kernel_op(name) for name in _ATTENTION_KERNELS)


def _import(path: str, where: str) -> Any:
    """The object a dotted *path* names.

    Raises:
        ValueError: If the path names no module or no attribute of one.
    """
    module, _, name = str(path).rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"{where}: cannot import {path!r}: {exc}") from exc


def replacement_specs(plan_overrides: Any) -> Tuple[Any, ...]:
    """The module replacements a train.yaml's ``plan_overrides`` install, as replacement rules.

    Args:
        plan_overrides: The ``plan_overrides`` list, as plain mappings.

    Returns:
        One :class:`~hyper_parallel.models.replacement.ModuleReplacementSpec`
        per entry that states ``replace_module``, in the order stated; the
        entries that state a sharding action alone, or a replacement
        conditioned on a run the census does not price (``when``), are left
        out.

    Raises:
        ValueError: If an entry states ``replace_module`` without
            ``module_type``, or names a module or factory that cannot be
            imported.
    """
    from hyper_parallel.models.replacement import ModuleReplacementSpec  # pylint: disable=C0415

    specs = []
    for index, entry in enumerate(plan_overrides or ()):
        where = f"plan_overrides[{index}]"
        if not isinstance(entry, Mapping) or entry.get("replace_module") is None or entry.get("when") is not None:
            continue
        if entry.get("module_type") is None:
            raise ValueError(f"{where} states replace_module without module_type")
        match = entry.get("match")
        patterns = (match,) if isinstance(match, str) else tuple(match or ())
        factory = entry["replace_module"]
        specs.append(ModuleReplacementSpec(
            match=patterns,
            factory=_import(factory.get("_target_") if isinstance(factory, Mapping) else factory, where),
            module_type=_import(entry["module_type"], where),
            exact_type=bool(entry.get("exact_type", False)),
        ))
    return tuple(specs)


class _Holder(torch.nn.Module):
    """A module tree holding one part where a model holds it, so a rule's patterns match its path."""

    def __init__(self, **parts: Any) -> None:
        """Hold each part of *parts* under ``model``, a layer in a list as a model's stack holds it."""
        super().__init__()
        self.model = torch.nn.Module()
        for name, part in parts.items():
            setattr(self.model, name, torch.nn.ModuleList([part]) if name == "layers" else part)


def _matching(spec: Any, names: Sequence[str]) -> Any:
    """*spec* with the patterns that match a module of *names*, or ``None`` where none does."""
    import fnmatch  # pylint: disable=C0415

    import dataclasses  # pylint: disable=C0415

    kept = tuple(pattern for pattern in spec.match
                 if any(fnmatch.fnmatchcase(name, pattern) for name in names))
    return None if not kept else dataclasses.replace(spec, match=kept)


def _replaced(holder: _Holder, specs: Sequence[Any]) -> None:
    """Install on *holder* the replacements of *specs* whose patterns match one of its modules.

    A rule's other patterns name modules no fake part holds, such as a
    model's final norm beside a layer, and are left out rather than refused.

    Raises:
        CensusUnavailable: If a factory needs a library this host lacks, such
            as an attention built on a native extension.  Transformers' own
            modules are not what the run trains, and measuring them prices a
            layer nobody runs: a DeepSeek-V3.2 layer whose indexer runs eager
            keeps, under the selective policy, 1192192 bytes a token against
            the 516368 it keeps plain, its scores an fp32 tensor of one
            sequence by another that the policy holds and no backward takes.
    """
    from hyper_parallel.models.replacement import (  # pylint: disable=C0415
        apply_module_replacements,
        compile_module_replacements,
    )

    names = [name for name, _ in holder.named_modules()]
    kept = [found for found in (_matching(spec, names) for spec in specs) if found is not None]
    if not kept:
        return
    try:
        apply_module_replacements(holder, compile_module_replacements(holder, kept),
                                  weights_mapping=[], context={}, capture_checkpoint_metadata=False)
    except (ImportError, OSError) as exc:
        raise CensusUnavailable(
            f"this host cannot build the modules the run installs "
            f"({[spec.factory.__name__ for spec in kept]}: {exc}); "
            f"run the cost model where they build, or drop context.census and let the formulas price them"
        ) from exc


# Model families whose attention a census cannot price here, by model type,
# with what is missing.  Their layers build and run on this host, through
# Transformers' own eager path, and that path is not the one the run takes:
# measuring it prices a layer nobody runs, as a replacement this host cannot
# build does (:func:`_replaced`, F53).  A family belongs here only while no
# contract in :mod:`~hyper_parallel.auto_parallel._npu_contracts` states what
# its kernels save; stating one is what takes it out again.
_NO_ACTIVATION_CONTRACT: Dict[str, str] = {
    # Empty while every family with kernels of its own has its contracts
    # stated.  DeepSeek-V4 was the entry that forced this guard and now has
    # them (:data:`_FAMILY_CONTRACTS`); the next family to arrive with a
    # kernel and no contract belongs here rather than being priced from the
    # path its run does not take.
}

# The families whose own modules call kernels this host has not got, with the
# contracts their census runs under.  A family here is measured as its run
# behaves; a family in :data:`_NO_ACTIVATION_CONTRACT` is refused instead.
_FAMILY_CONTRACTS = ("deepseek_v4",)

# The families whose layers a census cannot run one at a time, by model type,
# with why.  Their census runs the model's own forward over the whole stack
# (:func:`census_stack`), which calls each layer as the model calls it.
_STACK_FAMILIES: Dict[str, str] = {
    "deepseek_v4": (
        "31 of the DeepSeek-V4.1 crop's 40 layers consume compressed keys and values, or an Indexer "
        "key, that a source layer wrote earlier in the same forward and refuse to run without it; the "
        "hidden state carries a hyper-connection dimension beside a mixing state; and the rotary "
        "embedding is a pair of flavours the model picks between"
    ),
}


@contextlib.contextmanager
def _family_contracts(config: Any) -> Iterator[None]:
    """The contracts the modules of *config*'s family run under, for what runs inside.

    DeepSeek-V4.1's sparse attention is chosen on the device the tensors are
    on, so a census, whose tensors are fake and on none, would take the
    reference beside the kernel and measure a layer no run builds: the
    attention's contract stands in.  The Indexer's selection keeps nothing a
    backward reads on either path, fused or reference, since a run frees its
    scores' graph as soon as the indices are taken; its contract runs
    the selection without recording that graph, and the census measures the
    same bytes with it as without.

    Whether this tree builds the model at all is the stack census's to
    refuse (:func:`_model_class`): a layer's parameters do not depend on
    which kernel runs it and are counted from the same fake layer.
    """
    if str(getattr(config, "model_type", "")) not in _FAMILY_CONTRACTS:
        yield
        return
    from hyper_parallel.auto_parallel._npu_contracts import (  # pylint: disable=C0415
        v41_attention_contract,
        v41_indexer_contract,
    )
    rope = int(getattr(config, "qk_rope_head_dim", 0) or 0)
    with v41_attention_contract(rope), v41_indexer_contract():
        yield


def _no_contract(config: Any) -> None:
    """Refuse a census of one layer of a model whose kernels no contract states, or whose layers cannot run alone.

    Raises:
        CensusUnavailable: If *config*'s family is one of
            :data:`_NO_ACTIVATION_CONTRACT`, or one of
            :data:`_STACK_FAMILIES`, whose census is the whole stack's
            (:func:`census_stack`).  A layer's parameters do not depend on
            which kernel runs it and are counted as ever
            (:func:`census_parameters`); what it keeps for its backward
            does, so this refuses rather than answer with the bytes of a
            layer the run does not build.
    """
    family = str(getattr(config, "model_type", ""))
    missing = _NO_ACTIVATION_CONTRACT.get(family)
    if missing is not None:
        raise CensusUnavailable(
            f"no census of a {config.model_type} layer: {missing}; state the kernels' contracts, "
            f"or drop context.census and read the formulas' memory as an estimate, not as a bound"
        )
    alone = _STACK_FAMILIES.get(family)
    if alone is not None:
        raise CensusUnavailable(
            f"no census of one {family} layer, whose layers cannot be priced one at a time: {alone}. "
            f"census_activations measures the whole stack instead (census_stack)"
        )


def _modeling(config: Any) -> Any:
    """The Transformers modeling module of *config*'s model."""
    model_type = str(config.model_type)
    names = [model_type]
    if model_type.endswith("_text"):
        names.append(model_type[:-len("_text")])
    for name in names:
        try:
            return importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except ImportError:
            continue
    raise ValueError(f"no Transformers modeling module for model type {model_type!r}")


def _classes(modeling: Any) -> Tuple[type, type]:
    """The decoder layer and the language model's rotary embedding of *modeling*."""
    classes = {name: cls for name, cls in vars(modeling).items() if isinstance(cls, type)}
    layers = [cls for name, cls in classes.items() if name.endswith("DecoderLayer")]
    rotary = [cls for name, cls in classes.items()
              if name.endswith("RotaryEmbedding") and "Vision" not in name]
    rotary.sort(key=lambda cls: "Text" not in cls.__name__)
    if not layers or not rotary:
        raise ValueError(f"{modeling.__name__} states no decoder layer or rotary embedding")
    return layers[0], rotary[0]


def tp_config(config: Any, tp: int) -> Any:
    """*config* as one of *tp* tensor-parallel ranks holds its layers: heads and widths divided.

    Two of the fields divided can be one field.  A Transformers config class
    may declare an ``attribute_map``, and DeepSeek-V4's maps
    ``intermediate_size`` onto ``moe_intermediate_size``, so a model whose
    feed-forward is all experts states one width under two names.  Dividing
    each name in turn then divides that width by *tp* twice, and the layer
    built as a rank's share is a quarter as wide at TP 2 rather than half,
    which makes the split the census derives from the pair too large and ND
    over-reward tensor parallelism.  Qwen3-MoE keeps the two apart, which is
    why this stayed hidden.  Each underlying attribute is divided once.
    """
    config = copy.deepcopy(config)
    config.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    aliases = getattr(type(config), "attribute_map", None) or {}
    divided = set()
    for name in _TP_FIELDS:
        target = aliases.get(name, name)
        if target in divided:
            continue
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            setattr(config, name, max(1, value // tp))
            divided.add(target)
    return config


def _grid_rotary(rotary: Any) -> bool:
    """Whether *rotary* embeds a position on three grids, time, height and width, as a multimodal model's does.

    Qwen3.5's and Qwen3-VL's hold the grids' ``mrope_section``; Qwen2-VL's
    config states it.
    """
    if getattr(rotary, "mrope_section", None) is not None:
        return True
    config = getattr(rotary, "config", None)
    return any(isinstance(getattr(config, name, None), Mapping) and "mrope_section" in getattr(config, name)
               for name in ("rope_parameters", "rope_scaling"))


def _positions(rotary: Any, hidden: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, Any]:
    """The position ids of a sequence of *seq_length* tokens, and the embeddings *rotary* gives them.

    A rotary on three grids (:func:`_grid_rotary`) is handed a row of ids
    for each grid, as its model hands them, a text token's index on all
    three; the layers keep the one row. Transformers 5.17 and 5.18 refuse
    the single row the versions before them expanded themselves.
    """
    position_ids = torch.arange(seq_length).unsqueeze(0)
    rotary_ids = position_ids[None].expand(3, *position_ids.shape) if _grid_rotary(rotary) else position_ids
    with torch.no_grad():
        return position_ids, rotary(hidden, rotary_ids)


def _run(layer: Any, hidden: torch.Tensor, positions: Tuple[torch.Tensor, Any],
         call: Optional[Callable[..., Any]] = None) -> torch.Tensor:
    """One forward of *layer*, as its model calls it at *positions*, through *call* if given."""
    position_ids, position_embeddings = positions
    accepted = inspect.signature(layer.forward).parameters
    kwargs = {name: value for name, value in (
        ("position_embeddings", position_embeddings), ("position_ids", position_ids),
        ("attention_mask", None), ("use_cache", False)) if name in accepted}
    out = (call or layer)(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


# The function a gated-delta-rule attention of Transformers 5.15 and later
# calls by name, which its modeling module defines; earlier versions hold the
# rule in the module's ``chunk_gated_delta_rule``.
_DELTA_RULE_FUNCTION = "torch_chunk_gated_delta_rule"


def delta_rule_module(module: Any) -> bool:
    """Whether *module* runs a gated delta rule, whichever way its Transformers version calls it.

    Up to 5.14 the module holds the rule in ``chunk_gated_delta_rule``; from
    5.15 its forward calls its modeling module's ``torch_chunk_gated_delta_rule``
    by name, and the module holds nothing.
    """
    if hasattr(module, "chunk_gated_delta_rule"):
        return True
    forward = inspect.unwrap(getattr(type(module), "forward", None) or (lambda: None))
    return _DELTA_RULE_FUNCTION in getattr(getattr(forward, "__code__", None), "co_names", ())


@contextlib.contextmanager
def _delta_rule_held(modeling: Any, layer: Any, rule: Callable[..., Any]) -> Iterator[None]:
    """Have *layer*'s delta rule module run *rule* through its ``chunk_gated_delta_rule``, while the context lasts.

    The census swaps and follows the rule through that attribute.  A module
    of Transformers 5.15 or later, which calls its modeling module's function
    by name, gets the attribute too, and the name runs it.

    Raises:
        ValueError: If the layer holds more than one such module, which the
            name could not tell apart.
    """
    modules = [module for module in layer.modules() if delta_rule_module(module)]
    for module in modules:
        module.chunk_gated_delta_rule = rule
    by_name = getattr(modeling, _DELTA_RULE_FUNCTION, None)
    if not modules or by_name is None:
        yield
        return
    if len(modules) > 1:
        raise ValueError(f"{type(layer).__name__} holds {len(modules)} gated delta rule modules, and "
                         f"{_DELTA_RULE_FUNCTION} cannot tell which one calls it")
    module = modules[0]

    def by_attribute(*args: Any, **kwargs: Any) -> Any:
        """The module's rule as the attribute holds it when called: the census wraps it afterwards."""
        return module.chunk_gated_delta_rule(*args, **kwargs)

    setattr(modeling, _DELTA_RULE_FUNCTION, by_attribute)
    try:
        yield
    finally:
        setattr(modeling, _DELTA_RULE_FUNCTION, by_name)


@contextlib.contextmanager
def _fake_layer(config: Any, layer_index: int, gdn_backend: str = _EAGER_GDN,
                replacements: Sequence[Any] = ()) -> Iterator[Tuple[Any, Any]]:
    """Layer *layer_index* of *config* and its model's rotary embedding, on fake tensors and the runtime's kernels.

    *gdn_backend* names the gated delta rule the layer runs, as the runtime
    names it: ``"eager"`` for Transformers' own chunked implementation,
    ``"triton"`` for HyperParallel's kernel's contract, which saves the rule's
    inputs, the cumulated gates, beta and one chunk matrix and nothing more.
    Eager is the default because it is the rule a run reaches: the runtime's
    dispatcher defaults to it (``components/modules/gated_delta_net.py``,
    ``chunk_gated_delta_rule``), the only caller that can select the kernel is
    the context-parallel wrapper, which defaults to eager as well, and no
    recipe states a backend.  The two are not interchangeable: a layer keeping
    the kernel's saved set keeps less than half as much.

    Where *replacements* are given, the modules they name are HyperParallel's
    fused ones, running under the kernels' contracts for as long as the layer
    does.  Eager is Transformers' own Python rule, never a kernel a hub or the
    ``fla`` package would put in its place.

    Raises:
        ValueError: If *gdn_backend* names no backend the runtime has.
    """
    if gdn_backend not in _GDN_BACKENDS:
        raise ValueError(
            f"unsupported GDN backend {gdn_backend!r}; "
            f"expected one of {sorted(_GDN_BACKENDS)}."
        )
    modeling = _modeling(config)
    layer_cls, rotary_cls = _classes(modeling)
    config = copy.deepcopy(config)
    config._attn_implementation = _FLASH  # pylint: disable=protected-access
    config._experts_implementation = "grouped_mm"  # pylint: disable=protected-access
    modeling.ALL_ATTENTION_FUNCTIONS.register(_FLASH, _flash_attention)
    with npu_contracts() if replacements else contextlib.nullcontext(), \
            _family_contracts(config), \
            FakeTensorMode(allow_non_fake_inputs=True):
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = layer_cls(config, layer_index)
            rotary = rotary_cls(config=config)
        finally:
            torch.set_default_dtype(default)
        rule = (_gated_delta_rule(modeling) if gdn_backend == _HYPER_GDN
                else inspect.unwrap(getattr(modeling, _DELTA_RULE_FUNCTION, None) or (lambda: None)))
        with _delta_rule_held(modeling, layer, rule):
            if replacements:
                holder = _Holder(layers=layer)
                _replaced(holder, replacements)
                layer = holder.model.layers[0]
            layer.train()
            yield layer, rotary


class _ByPart(TorchDispatchMode):
    """A walk over a layer's forward that attributes each op to the part of the layer running it."""

    def __init__(self, layer: Any, recurrent: bool = True) -> None:
        """Follow which of *layer*'s modules runs, and which tensors are its parameters.

        *recurrent* is whether an attention of this layer has a recurrence to
        name: where it has none, an op of the attention that no weight takes
        part in is the attention's own, its mask or its rotary.  A walk that
        counts only matmuls leaves it at the default, because a weightless
        matmul under an attention is a recurrence or a score, and a score is
        counted on its own.
        """
        super().__init__()
        self.recurrent = recurrent
        self.params = {param.untyped_storage()._cdata  # pylint: disable=protected-access
                       for param in layer.parameters()}
        self.running: List[str] = []
        for name, module in layer.named_modules():
            if name:
                module.register_forward_pre_hook(functools.partial(self._enter, name))
                module.register_forward_hook(self._leave)

    def _enter(self, name: str, *_: Any) -> None:
        """Note that the module at *name* runs."""
        self.running.append(name)

    def _leave(self, *_: Any) -> None:
        """Note that the innermost module running has returned."""
        self.running.pop()

    def _part(self, args: Sequence[Any]) -> str:
        """The part an op on *args* runs for: its module's, or an attention's recurrence where no weight takes part."""
        part = _parameter_part(f"{self.running[-1]}.weight") if self.running else "ffn"
        weighted = any(isinstance(arg, torch.Tensor)
                       and arg.untyped_storage()._cdata in self.params  # pylint: disable=protected-access
                       for arg in args)
        if part == "attention" and not weighted and self.recurrent:
            return "linrec"
        return part


class _PartFlops(_ByPart):
    """A layer's forward FLOPs by part: its matmuls' toward the part running them, flash attention's the scores'."""

    def __init__(self, layer: Any) -> None:
        """Count no FLOPs yet, and follow *layer*'s modules."""
        super().__init__(layer)
        self.flops: Dict[str, int] = {}

    def _add(self, part: str, flops: int) -> None:
        """Count *flops* toward *part*."""
        self.flops[part] = self.flops.get(part, 0) + flops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        out = func(*args, **(kwargs or {}))
        if func is torch.ops.nd_census.flash_attention.default:
            query, key, value = args[:3]
            batch, heads, seq, width = query.shape
            self._add("scores", 2 * batch * heads * seq * key.shape[2] * (width + value.shape[-1]))
        elif flops := _matmul_flops(func, args):
            self._add(self._part(args), flops)
        return out


class _PartTraffic(_ByPart):
    """The bytes a layer's forward moves by part, the parameters among them, and the ops it dispatches.

    An op reads the tensors it is given and writes the tensors it returns.
    A view writes nothing and reads nothing: it renames bytes its producer
    already wrote, and no kernel runs for it, so it counts for neither.  A
    tensor given to an op twice is read twice, as the count is of the op's
    arguments rather than of the storages behind them.

    The parameters are counted again on their own, because a strategy
    divides their traffic where it leaves the activations' alone: the degree
    sharding a parameter divides the bytes read of it, and the routed
    experts' part is the one the expert-parallel degree divides.
    """

    def __init__(self, layer: Any, recurrent: bool = True) -> None:
        """Move nothing yet, and follow *layer*'s modules."""
        super().__init__(layer, recurrent=recurrent)
        self.moved: Dict[str, float] = {}
        self.parameter: Dict[str, float] = {}
        self.launches: Dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func.namespace == "prim" or getattr(func, "is_view", False):
            return out
        part = ("scores" if func is torch.ops.nd_census.flash_attention.default
                else self._part(args))
        moved = parameter = 0
        for tensor in tree_flatten((args, kwargs))[0]:
            if not isinstance(tensor, torch.Tensor):
                continue
            size = tensor.numel() * tensor.element_size()
            moved += size
            if tensor.untyped_storage()._cdata in self.params:  # pylint: disable=protected-access
                parameter += size
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor):
                moved += tensor.numel() * tensor.element_size()
        self.moved[part] = self.moved.get(part, 0.0) + moved
        self.parameter[part] = self.parameter.get(part, 0.0) + parameter
        self.launches[part] = self.launches.get(part, 0) + 1
        return out


def census_flops(config: Any, layer_index: int, seq_length: int) -> Dict[str, int]:
    """The forward FLOPs of layer *layer_index* of *config* on one sequence, by part.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the sequence it runs.

    Returns:
        The FLOPs of each part that runs a matmul: the attention's
        projections (``attention``), its scores and values at every pair of
        tokens (``scores``), a linear attention's recurrence (``linrec``),
        and the parts of the feed-forward :func:`_parameter_part` names.
    """
    with _fake_layer(config, layer_index) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        positions = _positions(rotary, hidden, seq_length)
        counter = _PartFlops(layer)
        with counter:
            _run(layer, hidden, positions)
    return counter.flops


def census_traffic(config: Any, layer_index: int, seq_length: int,
                   gdn_backend: str = _EAGER_GDN) -> LayerTraffic:
    """What one forward of layer *layer_index* of *config* moves, by the part of the layer moving it.

    The quantity the time model counts is matmul FLOPs, and the cluster's
    kernel tables put those at 6.3 to 7.1% of a profiled step of this model.
    This counts the other quantity a step could be priced by, on the same
    forward and with no device, so that a measured round can set each
    beside the time it took.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        gdn_backend: The gated delta rule the run takes (:func:`_fake_layer`).
            It is the largest single choice here: Transformers' own chunked
            rule runs a matmul a chunk and the kernel's contract is one op.

    Returns:
        A :class:`LayerTraffic` of the whole layer at tensor and context
        parallelism 1, by the parts :func:`census_flops` prices
        (:class:`_PartTraffic`).
    """
    with _fake_layer(config, layer_index, gdn_backend=gdn_backend) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        positions = _positions(rotary, hidden, seq_length)
        # The rule's own module is what a layer has a recurrence by, as
        # _fake_layer finds it to settle which rule the layer runs.
        recurrent = any(delta_rule_module(module) for module in layer.modules())
        counter = _PartTraffic(layer, recurrent=recurrent)
        with counter:
            _run(layer, hidden, positions)
    return LayerTraffic(counter.moved, counter.parameter, counter.launches)


def census_layer(config: Any, layer_index: int, seq_length: int, selective: bool = False,
                 ops: Optional[Dict[str, int]] = None,
                 replacements: Sequence[Any] = (),
                 gdn_backend: str = _EAGER_GDN) -> Tuple[int, int]:
    """Bytes layer *layer_index* of *config* keeps for its backward, and the most its backward holds.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        selective: Whether the layer runs under HyperParallel's selective
            activation checkpointing, as its trainer wraps a layer.
        ops: Where given, and the layer runs without checkpointing, takes
            the bytes it saves by the op saving them (:class:`_SavedOps`).
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).
        gdn_backend: The gated delta rule the run takes (:func:`_fake_layer`).

    Returns:
        The bytes the forward keeps for the backward, its input included,
        and the bytes of activations the backward holds when it holds the
        most, its own gradients included, less the parameters and those
        gradients.

    Raises:
        CensusUnavailable: If no contract states what this model's kernels
            save (:func:`_no_contract`).
    """
    _no_contract(config)
    with _fake_layer(config, layer_index, gdn_backend=gdn_backend,
                     replacements=replacements) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        if replacements:
            # A fused kernel keeps a constant per device, such as the mask
            # the attention runs its causal sparse mode with, which the
            # runtime allocates once for every layer and micro-batch: one
            # forward allocates them before the layer's own bytes are
            # counted, so they count toward no layer.
            with torch.no_grad():
                _run(layer, hidden, _positions(rotary, hidden, seq_length))
        if not selective:
            saved_ops = None if ops is None else _SavedOps(layer)
            measured = _measure(list(layer.parameters()), (hidden,),
                                lambda: _run(layer, hidden, _positions(rotary, hidden, seq_length)),
                                lambda out: out.backward(grad), grad_inputs=(grad,), ops=saved_ops)
            if saved_ops is not None:
                ops.update(saved_ops.saved)
            return measured
        # The model embeds the positions once for all its layers, outside
        # their checkpoints, and the trainer checkpoints each layer's call.
        positions = _positions(rotary, hidden, seq_length)
        call = functools.partial(activation_memory.checkpoint, layer, swap_inputs=False,
                                 context_fn=_selective_contexts)
        return _measure(list(layer.parameters()), (hidden,), lambda: _run(layer, hidden, positions, call),
                        lambda out: out.backward(grad), grad_inputs=(grad,), checkpointed=True,
                        shared=tree_flatten(positions)[0])


def census_saved_ops(config: Any, layer_index: int, seq_length: int,
                     replacements: Sequence[Any] = ()) -> Dict[str, float]:
    """What layer *layer_index* of *config* keeps for its backward for each op, per token, the whole layer.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).

    Returns:
        ``{op: bytes per token}``, by the op saving them
        (:class:`_SavedOps`), which sum to what the layer keeps.
    """
    ops: Dict[str, int] = {}
    census_layer(config, layer_index, seq_length, ops=ops, replacements=replacements)
    return {op: size / seq_length for op, size in ops.items()}


def census_recomputed(config: Any, layer_index: int, seq_length: int) -> Tuple[float, float]:
    """The shares of layer *layer_index*'s matmul FLOPs its backward runs again under selective checkpointing.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.

    Returns:
        The share of its attention's projections' FLOPs, and of the rest of
        the layer's, that HyperParallel's selective policy recomputes: it
        saves every other matmul's output, in the order the layer runs
        them, and a biased projection's always.
    """
    with _fake_layer(config, layer_index) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        ledger = _RecomputedMatmuls(layer)
        call = functools.partial(activation_memory.checkpoint, layer, swap_inputs=False,
                                 context_fn=functools.partial(_selective_contexts, ledger))
        with _checkpoints_on_cpu():
            _run(layer, hidden, _positions(rotary, hidden, seq_length), call)
    return ledger.share(attention=True), ledger.share(attention=False)


def _parameter_part(name: str) -> str:
    """The part of a decoder layer the parameter at path *name* belongs to, as ND prices the parts.

    The layer's norms and its attention's per-head query and key norms are
    its norms; the rest of a child named for attention is its attention.
    Of the feed-forward, a shared expert's parameters, the routed experts'
    and the router's are parts of their own, and the rest is the dense
    feed-forward's.
    """
    segments = name.split(".")
    module = segments[-2] if len(segments) > 1 else ""
    if "norm" in segments[0] or module in _QK_NORMS:
        return "norm"
    if "attn" in segments[0] or "attention" in segments[0]:
        return "attention"
    if any("shared" in segment for segment in segments):
        return "shared"
    if "experts" in segments:
        return "routed"
    if module in ("gate", "router"):
        return "router"
    return "ffn"


def census_parameters(config: Any, layer_index: int, replacements: Sequence[Any] = ()) -> Dict[str, int]:
    """The parameters of layer *layer_index* of *config*, by part.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).  A replacement holds the
            parameters its source held, fused or renamed, so it counts the
            same.

    Returns:
        The parameter count of each part the layer has
        (:func:`_parameter_part`).
    """
    parts: Dict[str, int] = {}
    with _fake_layer(config, layer_index, replacements=replacements) as (layer, _):
        for name, param in layer.named_parameters():
            part = _parameter_part(name)
            parts[part] = parts.get(part, 0) + param.numel()
    return parts


def census_final_norm(config: Any, replacements: Sequence[Any] = ()) -> int:
    """The parameters of *config*'s final norm, of the class of a layer's input norm."""
    with _fake_layer(config, 0, replacements=replacements) as (layer, _):
        return sum(param.numel() for param in layer.input_layernorm.parameters())


def census_output(config: Any, seq_length: int, replacements: Sequence[Any] = ()) -> Tuple[int, int]:
    """Bytes the output layer of *config* keeps for its backward, and the most its backward holds.

    Args:
        config: The language model's Transformers config.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`), of which the final norm's is the
            one that reaches the output layer.

    Returns:
        As :func:`census_layer`: the final norm's input included, the
        output table and its gradient left out.
    """
    modeling = _modeling(config)
    layer_cls, _ = _classes(modeling)
    loss_function = importlib.import_module("transformers.loss.loss_utils").ForCausalLMLoss
    with npu_contracts() if replacements else contextlib.nullcontext(), \
            FakeTensorMode(allow_non_fake_inputs=True):
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            # The final norm is of the class of a layer's input norm.
            norm = layer_cls(config, 0).input_layernorm
            head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        finally:
            torch.set_default_dtype(default)
        if replacements:
            holder = _Holder(norm=norm)
            _replaced(holder, replacements)
            norm = holder.model.norm
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, config.vocab_size, (1, seq_length))
        # The loss alone holds the logits: the trainer drops the model's
        # output before the backward.
        return _measure([*norm.parameters(), *head.parameters()], (hidden,),
                        lambda: loss_function(head(norm(hidden)), labels, config.vocab_size),
                        lambda loss: loss.backward())


def census_output_activations(config: Any, seq_length: int = 4096,
                             replacements: Sequence[Any] = ()) -> KindActivations:
    """What the output layer of *config* keeps and holds, per token, with its vocabulary and half of it.

    Args:
        config: The language model's Transformers config.
        seq_length: The tokens the census runs the layer at.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).

    Returns:
        The layer's :class:`KindActivations`, whose TP part is the part a
        vocabulary-parallel loss over two ranks halves.
    """
    half = copy.deepcopy(config)
    half.vocab_size = max(1, config.vocab_size // 2)
    (saved_1, working_1), (saved_2, working_2) = (
        census_output(each, seq_length, replacements) for each in (config, half))
    return KindActivations(
        saved=max(0.0, 2 * saved_2 - saved_1) / seq_length,
        saved_tp=max(0.0, 2 * (saved_1 - saved_2)) / seq_length,
        working=max(0.0, 2 * working_2 - working_1) / seq_length,
        working_tp=max(0.0, 2 * (working_1 - working_2)) / seq_length,
        seq_length=int(seq_length),
    )


class _FakeScalars(TorchDispatchMode):
    """A scalar read of a fake tensor answered with zero, as :class:`_LiveBytes` answers it.

    A model can read a value out of a tensor while it builds, as
    DeepSeek-V4.1's Engram reads its token map's pad entry to build its hash
    mapping, which a fake tensor refuses.  The value settles a padding id,
    never a shape the census counts.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        if func is _SCALAR_READ and isinstance(args[0], FakeTensor):
            return 0
        return func(*args, **(kwargs or {}))


def _model_class(config: Any) -> Tuple[Any, bool]:
    """What the trainer builds *config* as, by its ``from_config``: its family's own class, else Transformers'.

    The trainer's builder discovers the family's registration, asks the
    model registry for a class registered by the architecture the config
    names, and builds Transformers' own model where none is.

    Returns:
        The class, and whether it is Transformers' own.

    Raises:
        CensusUnavailable: If the config names an architecture neither the
            registry nor Transformers builds, as a DeepSeek-V4.1 config does
            on a tree that does not carry that model: Transformers'
            DeepSeek-V4 in its place would price a different model.
    """
    import transformers  # pylint: disable=C0415
    from hyper_parallel.models._transformers.config_resolver import get_is_hf_model  # pylint: disable=C0415
    from hyper_parallel.models.registry import _resolve_custom_model_cls  # pylint: disable=C0415

    architecture = (getattr(config, "architectures", None) or [""])[0] or ""
    if not get_is_hf_model(config):
        return _resolve_custom_model_cls(architecture), False
    if architecture and not hasattr(transformers, architecture):
        raise CensusUnavailable(
            f"the config names {architecture}, which neither this tree's model registry nor Transformers "
            f"builds; price it from a tree carrying the model, or drop context.census and read the "
            f"formulas' memory as an estimate, not as a bound"
        )
    return transformers.AutoModelForCausalLM, True


class _Stack:
    """Which layer of a stack runs, forward and backward, and where in a tracker's log each backward runs."""

    def __init__(self, layers: Sequence[Any], clock: Callable[[], int]) -> None:
        """Follow *layers* through both passes, placing their backwards by *clock*."""
        self.running: List[int] = []
        self.windows: Dict[int, List[int]] = {}
        self.clock = clock
        self.handles: List[Any] = []
        for index, layer in enumerate(layers):
            self.handles += [
                layer.register_forward_pre_hook(functools.partial(self._enter, index)),
                layer.register_forward_hook(self._leave),
                layer.register_full_backward_pre_hook(functools.partial(self._enter_backward, index)),
                layer.register_full_backward_hook(self._leave_backward),
            ]
        self.backward: List[int] = []

    # A hook's return replaces what the module takes or gives, so each of
    # these returns None.

    def _enter(self, index: int, *_: Any) -> None:
        """Note that the layer at *index* runs its forward."""
        self.running.append(index)

    def _leave(self, *_: Any) -> None:
        """Note that the innermost layer running its forward has returned."""
        if self.running:
            self.running.pop()

    def _enter_backward(self, index: int, *_: Any) -> None:
        """Open the window of the layer at *index*, whose backward starts."""
        self.backward.append(index)
        self.windows.setdefault(index, [self.clock(), self.clock()])

    def _leave_backward(self, *_: Any) -> None:
        """Close the window of the layer whose backward has returned."""
        if self.backward:
            index = self.backward.pop()
            self.windows[index][1] = self.clock()

    @property
    def layer(self) -> Optional[int]:
        """The outermost layer running its forward, or None."""
        return self.running[0] if self.running else None

    def remove(self) -> None:
        """Remove every hook this placed."""
        for handle in self.handles:
            handle.remove()


def census_stack(config: Any, seq_length: int, replacements: Sequence[Any] = ()) -> List[Tuple[int, int]]:
    """What each layer of *config* keeps for its backward, and the most its backward holds, from one pass of the stack.

    For a model whose layers cannot run alone (:data:`_STACK_FAMILIES`): the
    model the trainer builds (:func:`_model_class`), on fake tensors and
    under its family's contracts, runs its own forward over a micro-batch of
    one sequence and then its backward, so each layer is called as its model
    calls it, with what earlier layers wrote, its hidden and mixing states
    and the rotary flavours the model computes.  Each save counts toward the
    layer whose forward makes it first, and a layer keeps those of its saves
    still live when the forward ends (:func:`_measure`'s rule).  Its backward
    holds what it keeps and the most it holds beyond every layer's saves,
    the parameters and their gradients while it runs.  The embedding's and
    the final norm's saves count toward no layer: the output layer's census
    prices the end of the model.

    Args:
        config: The language model's Transformers config.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).

    Returns:
        ``(kept, held)`` bytes of each decoder layer, in model order.

    Raises:
        CensusUnavailable: If this tree cannot build the model the config
            names (:func:`_model_class`), or a module a replacement installs
            (:func:`_replaced`).
    """
    model_cls, native = _model_class(config)
    config = copy.deepcopy(config)
    if native:
        # Transformers' own model runs its attention through the interface,
        # as _fake_layer's layer does, so it runs the kernel's contract.
        config._attn_implementation = _FLASH  # pylint: disable=protected-access
        _modeling(config).ALL_ATTENTION_FUNCTIONS.register(_FLASH, _flash_attention)
    with npu_contracts() if replacements else contextlib.nullcontext(), _family_contracts(config), \
            FakeTensorMode(allow_non_fake_inputs=True), _FakeScalars():
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = model_cls.from_config(config)
        finally:
            torch.set_default_dtype(default)
        # The experts run as grouped matmuls, as every layer _fake_layer
        # builds does.  Transformers falls back to its eager experts for a
        # class whose module it does not know, a loop over the experts with
        # token counts no fake tensor has; the experts read this per call.
        model.config._experts_implementation = "grouped_mm"  # pylint: disable=protected-access
        if replacements:
            _replaced(model, replacements)
        model.train()
        backbone = model.get_decoder()
        params = list(model.parameters())
        stored = {p.untyped_storage()._cdata for p in params}  # pylint: disable=protected-access
        live = _LiveBytes()
        follow = _Stack(backbone.layers, lambda: len(live.events))
        saved_by: Dict[Tuple[int, int], int] = {}

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            """Count a tensor autograd saves, unless it is a parameter's, toward the layer saving it first."""
            if tensor.untyped_storage()._cdata not in stored and follow.layer is not None:  # pylint: disable=W0212
                saved_by.setdefault(live.key(tensor), follow.layer)
            # Detached, as _measure returns it: a graph the forward drops is freed.
            return tensor.detach()

        grads: List[Tuple[int, int]] = []
        handles = [p.register_hook(lambda grad: grads.append(live.key(grad))) for p in params]
        kept = [0] * len(backbone.layers)
        try:
            with live, torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                out = backbone(input_ids=torch.zeros(1, seq_length, dtype=torch.long)).last_hidden_state
                for key, index in saved_by.items():
                    kept[index] += live.sizes.get(key, 0)
                out.sum().backward()
        finally:
            follow.remove()
            for handle in handles:
                handle.remove()
        grads += [live.key(p.grad) for p in params if p.grad is not None]
    # What every layer keeps, the gradients and the parameters left out, the
    # rest is what a layer's backward holds beyond them while it runs.
    excluded = set(saved_by) | set(grads) | {key for key, _ in live.events if key[0] in stored}
    beyond = list(itertools.accumulate(0 if key in excluded else delta for key, delta in live.events))
    held = []
    for index, size in enumerate(kept):
        begin, end = follow.windows.get(index, (0, 0))
        held.append(size + max(beyond[begin:end], default=0))
    return list(zip(kept, held))


def _stack_activations(config: Any, layers: Iterable[Mapping[str, Any]], seq_length: int,
                       replacements: Sequence[Any] = ()) -> Dict[str, KindActivations]:
    """What a layer of each kind of *layers* keeps and holds, per token, from the stack at TP 1 and TP 2.

    A kind's record is the mean of its layers', which is what the memory
    model applies to each of them: with one record a kind, the stack's total
    comes back whole.  The parts tensor parallelism splits follow from the
    pair as :func:`census_activations` derives them.  No selective part is
    stated: a selective layer of these kinds is priced by the formulas.
    """
    kinds = [str(group["kind"]) for group in layers if not group.get("mtp") for _ in range(int(group["count"]))]
    logger.info("census of the whole stack of %d layers at %d tokens, at TP 1 and TP 2: a %s layer "
                "cannot run alone", len(kinds), seq_length, config.model_type)
    one, two = (census_stack(tp_config(config, tp), seq_length, replacements) for tp in (1, 2))
    if len(one) != len(kinds):
        raise ValueError(f"the model builds {len(one)} decoder layers, where its spec states {len(kinds)}")
    out: Dict[str, KindActivations] = {}
    for kind in dict.fromkeys(kinds):
        members = [index for index, each in enumerate(kinds) if each == kind]
        (kept_1, held_1), (kept_2, held_2) = (
            tuple(sum(stack[index][part] for index in members) / len(members) for part in (0, 1))
            for stack in (one, two))
        out[kind] = KindActivations(
            saved=max(0.0, 2 * kept_2 - kept_1) / seq_length,
            saved_tp=max(0.0, 2 * (kept_1 - kept_2)) / seq_length,
            working=max(0.0, 2 * held_2 - held_1) / seq_length,
            working_tp=max(0.0, 2 * (held_1 - held_2)) / seq_length,
            seq_length=int(seq_length),
        )
    return out


def census_activations(config: Any, layers: Iterable[Mapping[str, Any]], seq_length: int = 4096,
                       replacements: Sequence[Any] = (),
                       gdn_backend: str = _EAGER_GDN) -> Dict[str, KindActivations]:
    """What a layer of each kind of *layers* keeps and holds, per token, at TP 1 and TP 2.

    Args:
        config: The language model's Transformers config.
        layers: The model spec's layer stack, groups in model order; MTP
            groups are left out.
        seq_length: The tokens the census runs a layer at.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).
        gdn_backend: The gated delta rule the run takes (:func:`_fake_layer`).

    Returns:
        Each kind's :class:`KindActivations`: of the bytes a layer at TP 2
        holds, the part TP splits is half the part at TP 1; what it keeps
        under HyperParallel's selective activation checkpointing too, the
        shares of its matmul FLOPs that recomputes, and what it keeps for
        each op, its records per op.  A family whose layers cannot run
        alone (:data:`_STACK_FAMILIES`) is measured over its whole stack
        instead (:func:`census_stack`), and its records state the plain
        layer's bytes only.

    Raises:
        CensusUnavailable: If this tree cannot build what the run builds.
    """
    if str(getattr(config, "model_type", "")) in _STACK_FAMILIES:
        return _stack_activations(config, layers, seq_length, replacements)
    firsts: Dict[str, int] = {}
    index = 0
    for group in layers:
        if group.get("mtp"):
            continue
        firsts.setdefault(str(group["kind"]), index)
        index += int(group["count"])
    out: Dict[str, KindActivations] = {}
    for kind, layer_index in firsts.items():
        ops_1: Dict[str, int] = {}
        ops_2: Dict[str, int] = {}
        (saved_1, working_1), (saved_2, working_2) = (
            census_layer(tp_config(config, tp), layer_index, seq_length, ops=ops,
                         replacements=replacements, gdn_backend=gdn_backend)
            for tp, ops in ((1, ops_1), (2, ops_2)))
        (kept_1, _), (kept_2, _) = (
            census_layer(tp_config(config, tp), layer_index, seq_length, selective=True,
                         replacements=replacements, gdn_backend=gdn_backend)
            for tp in (1, 2))
        attention_mm, ffn_mm = census_recomputed(config, layer_index, seq_length)
        out[kind] = KindActivations(
            saved=max(0.0, 2 * saved_2 - saved_1) / seq_length,
            saved_tp=max(0.0, 2 * (saved_1 - saved_2)) / seq_length,
            working=max(0.0, 2 * working_2 - working_1) / seq_length,
            working_tp=max(0.0, 2 * (working_1 - working_2)) / seq_length,
            seq_length=int(seq_length),
            selective=max(0.0, 2 * kept_2 - kept_1) / seq_length,
            selective_tp=max(0.0, 2 * (kept_1 - kept_2)) / seq_length,
            selective_attention_mm=attention_mm,
            selective_ffn_mm=ffn_mm,
            ops={op: max(0.0, 2 * ops_2.get(op, 0) - ops_1[op]) / seq_length for op in sorted(ops_1)},
            ops_tp={op: max(0.0, 2 * (ops_1[op] - ops_2.get(op, 0))) / seq_length for op in sorted(ops_1)},
        )
    return out

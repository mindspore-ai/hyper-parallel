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
# Tracer reaches into torch internals (torch._dynamo, torch.nn.utils.stateless,
# torch.utils._pytree) for joint-graph capture; these have no public API.
# pylint: disable=E1136,E1129,W0212
"""
Graph Tracer - Model Graph Tracer (make_fx + autograd.grad)

Traces a model into a joint Forward + Backward FX graph, following
the design of torchtitan's ``experiments/graph_trainer/minimal_fx_tracer``:

1. Module parameters/buffers are extracted from the live model into flat
   tensors and threaded through the graph as **static inputs** (leading
   placeholders) instead of ``get_attr`` nodes, via
   ``torch.nn.utils.stateless._reparametrize_module``. The graph therefore
   contains no parameter ``get_attr`` nodes, and passes can split the graph by
   reshaping the static-input placeholders.

2. The forward pass runs ``torch.autograd.grad(loss, params)`` *inside* the
   traced function body. For that to be traceable the engine backward must be
   patched (``torch.compiler._patch_engine_backward()``) and backward must run
   on the calling thread (``torch.autograd.set_multithreading_enabled(False)``),
   otherwise the C++ autograd engine dispatches backward to a worker thread
   with a fresh ``contextvars.Context`` and the tracing context is lost.

3. The whole trace runs under ``FakeTensorMode(allow_non_fake_inputs=True)``
   plus a non-strict tracing context, ``preserve_node_meta()`` and
   ``_skip_nested_compile()``, mirroring the torchtitan tracer.

The joint graph is executed with ``torch.no_grad()`` at runtime: it already
contains the explicit backward ops, and a redundant autograd graph on top
would keep every forward intermediate alive via ``grad_fn`` references.
"""

import contextlib
import copy
import inspect
import warnings
from collections.abc import Callable, Generator
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn
from torch._guards import tracing, TracingContext
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.nn.utils import stateless

# Older torch's make_fx predates record_stack_traces / record_module_stack
# and rejects unknown kwargs; forward only the kwargs this torch supports.
# The joint graph still captures without record_stack_traces -- the fwd->bw
# metadata copy skips a missing stack_trace instead of failing.
try:
    _MAKE_FX_PARAMS = inspect.signature(make_fx).parameters
except (ValueError, TypeError):
    _MAKE_FX_PARAMS = {}
_MAKE_FX_KWARGS: Dict[str, Any] = {
    kw: val
    for kw, val in (("record_stack_traces", True), ("record_module_stack", True))
    if kw in _MAKE_FX_PARAMS
}

# Tensors and make_fx-safe primitives are allowed as graph leaves.
# Everything else (callables, custom objects) must be captured in the
# train_fn closure or registered as pytree constants.
_ALLOWED_LEAF_TYPES = (torch.Tensor, int, float, bool, str, type(None))


@contextmanager
def _skip_nested_compile() -> Generator[None, None, None]:
    """Tell dynamo to inline ``torch.compile``'d functions during make_fx tracing.

    make_fx cannot trace through torch.compile'd functions (e.g. compiled
    attention kernels). Setting ``error_on_nested_fx_trace`` to False makes
    dynamo silently inline the wrapped function instead of raising.
    """
    prev = torch._dynamo.config.error_on_nested_fx_trace
    torch._dynamo.config.error_on_nested_fx_trace = False
    try:
        yield
    finally:
        torch._dynamo.config.error_on_nested_fx_trace = prev


@contextmanager
def _patch_engine_backward() -> Generator[None, None, None]:
    """No-op fallback for torch versions without the patched autograd engine.

    ``torch.compiler._patch_engine_backward`` makes ``torch.autograd.grad``
    inside a make_fx-traced function emit backward ops as FX nodes instead of
    running a real autograd backward. Present in torchtitan's patched torch;
    absent in stock torch, where joint-graph capture of ``autograd.grad`` is
    not supported at all.
    """
    patcher = getattr(torch.compiler, "_patch_engine_backward", None)
    if patcher is None:
        warnings.warn(
            "torch.compiler._patch_engine_backward not available. "
            "Joint-graph capture of autograd.grad may not work correctly. "
            "Consider using torchtitan's patched torch or a newer torch version.",
            RuntimeWarning,
            stacklevel=2,
        )
        yield
    else:
        with patcher():
            yield


@contextmanager
def _non_strict_tracing_context() -> Generator[None, None, None]:
    """Mark this make_fx pass as the non-strict tracing flow.

    Required by ``_patch_engine_backward``; no-op fallback on stock torch.
    """
    ctx = getattr(torch.compiler, "_non_strict_tracing_context", None)
    if ctx is None:
        yield
    else:
        with ctx():
            yield


@dataclass
class JointGraph:
    """
    Joint forward-backward computation graph

    Features:
    - No parallel information
    - Parameters, activations, gradients are all complete tensors
    - Forward + Loss + Backward complete joint graph
    - Parameters/buffers are static graph inputs (leading placeholders),
      not ``get_attr`` nodes
    """

    graph_module: torch.fx.GraphModule
    # Storage-free descriptors of the tracing-time user inputs (shape/dtype
    # for tensors) via ``_input_meta``. Real sample tensors are deliberately
    # not kept: they would pin their storage for the whole training run. The
    # FakeTensor counterparts live in ``example_inputs``.
    inputs: List[Any]
    outputs: List[Any]
    param_names: List[str]
    param_shapes: Dict[str, tuple]
    num_layers: int
    # Trace-time parameter/buffer values (flat, in graph placeholder order)
    # and their FQNs. ``state_fqns`` is the authoritative order for feeding
    # runtime parameter state into the graph.
    state_fqns: List[str]
    example_inputs: tuple


def extract_module_state(mod: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a merged dict of the module's named parameters and buffers."""
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


@contextlib.contextmanager
def _reparametrize_train_state(
    module: nn.Module,
    model_state: Dict[str, torch.Tensor],
):
    """Rebind ``module``'s parameters/buffers to explicit trace-time tensors.

    Inside the traced function this swaps in the fake/static tensors so every
    parameter access in the model goes through the graph's static inputs.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(stateless._reparametrize_module(module, model_state))
        yield


def _iter_node_args(node: torch.fx.Node):
    """Yield every ``fx.Node`` among ``node.args`` (unwrapping nested lists)."""
    for a in node.args:
        if isinstance(a, torch.fx.Node):
            yield a
        elif isinstance(a, (tuple, list)):
            for x in a:
                if isinstance(x, torch.fx.Node):
                    yield x


def _output_args(fx_g: torch.fx.GraphModule) -> List[torch.fx.Node]:
    """Return the flattened forward arg list of the graph's ``output`` node."""
    out_node = next((n for n in fx_g.graph.nodes if n.op == "output"), None)
    if out_node is None:
        return []
    args: List[torch.fx.Node] = []
    for a in out_node.args:
        if isinstance(a, torch.fx.Node):
            args.append(a)
        elif isinstance(a, (tuple, list)):
            for x in a:
                if isinstance(x, torch.fx.Node):
                    args.append(x)
    return args


def _ancestors_of(node: torch.fx.Node) -> set:
    """Return the transitive producers of ``node`` (excluding ``node``)."""
    ancestors = set()
    dq = deque([node])
    while dq:
        cur = dq.popleft()
        for a in _iter_node_args(cur):
            if a.op == "placeholder" or a in ancestors:
                continue
            ancestors.add(a)
            dq.append(a)
    return ancestors


def _has_module_stack(node: torch.fx.Node) -> bool:
    """True if the node carries any ``nn_module_stack`` entry.

    make_fx leaves stack-less nodes (e.g. autograd engine output) with an
    *empty* ``OrderedDict`` rather than ``None``, so a truthiness check is
    the robust test.
    """
    return bool(node.meta.get("nn_module_stack"))


def _annotate_autograd_backward(fx_g: torch.fx.GraphModule) -> None:
    """Tag backward nodes with ``meta['autograd_backward'] = True``.

    Stock torch's make_fx (without the torchtitan ``_patch_engine_backward``
    hook) never attaches the autograd engine's ``autograd_backward`` tag, so
    passes that split the joint graph into fwd/bwd halves get nothing to key
    on. This reconstructs the tag structurally.

    The joint graph's ``output`` returns ``[loss, grad_0, ... grad_n]``. The
    loss and the nodes feeding it (logits, loss reduction, ``getitem`` of the
    loss value) are forward even though they live outside any module and so
    carry no ``nn_module_stack``. Every other output is a parameter gradient
    produced by ``torch.autograd.grad``.

    Two pre-computed sets separate the halves:

    - ``fwd_loss_nodes``: the transitive producers of ``loss`` — the logits,
      the shift/reshape that makes the loss target, and the loss reduction
      itself. These are forward (they are the loss *source*); backward is
      computed *from* them.
    - ``bwd_nodes``: ``call_function`` nodes reachable from the parameter
      gradients (``output[1:]``) that carry no ``nn_module_stack`` and are
      not in ``fwd_loss_nodes``. The module stack is only recorded while a
      traced module ``forward`` runs, so autograd-engine nodes have none.
      Walking backward and stopping at forward boundaries (stack-bearing
      nodes or the loss source) spans exactly the backward half.
    """
    output = _output_args(fx_g)
    if not output:
        return
    loss_node = output[0]
    grad_seeds = output[1:]

    # Forward loss chain: everything that flows INTO the loss. Backward is
    # computed from the loss, so these are forward regardless of whether any
    # module recorded them (they may sit in ``train_fn``, outside modules).
    fwd_loss_nodes = _ancestors_of(loss_node)
    fwd_loss_nodes.add(loss_node)

    # BFS backward over dataflow from the gradient outputs, marking every
    # stack-less, non-loss-source node as backward. Stack-bearing nodes are
    # forward (module boundaries); loss-source nodes are forward (the loss is
    # the backward's *source*, not part of it).
    bwd_nodes = set()
    dq = deque()
    for seed in grad_seeds:
        if (
            not _has_module_stack(seed)
            and seed.op != "placeholder"
            and seed not in fwd_loss_nodes
        ):
            bwd_nodes.add(seed)
            dq.append(seed)
    while dq:
        n = dq.popleft()
        for a in _iter_node_args(n):
            if a.op == "placeholder" or a in bwd_nodes:
                continue
            if a in fwd_loss_nodes:
                continue  # forward loss source; do not expand
            if _has_module_stack(a):
                continue  # forward module boundary; do not expand
            bwd_nodes.add(a)
            dq.append(a)

    for node in bwd_nodes:
        node.meta["autograd_backward"] = True


def _copy_fwd_metadata_to_bw_nodes(fx_g: torch.fx.GraphModule) -> None:
    """Copy forward metadata to backward nodes across all nested FX subgraphs.

    Uses a two-pass approach over all submodule graphs (including HOP
    subgraphs). Pass 1 collects forward nodes by ``seq_nr``; pass 2 copies
    ``custom``/``nn_module_stack``/``stack_trace`` from the matching forward
    node to each backward node. Backward nodes are identified by the autograd
    engine's ``autograd_backward`` tag on ``node.meta``.
    """
    seq_nr_to_fwd_node: Dict[int, torch.fx.Node] = {}
    for node in _iter_seq_nr_nodes(fx_g, backward=False):
        seq_nr_to_fwd_node.setdefault(node.meta["seq_nr"], node)

    for node in _iter_seq_nr_nodes(fx_g, backward=True):
        fwd_node = seq_nr_to_fwd_node.get(node.meta["seq_nr"])
        if fwd_node is None or fwd_node is node:
            continue
        _copy_fwd_meta(fwd_node, node)


def _iter_seq_nr_nodes(fx_g: torch.fx.GraphModule, backward: bool):
    """Yield ``seq_nr``-bearing call/get_attr nodes across all subgraphs.

    ``backward=False`` yields forward nodes, ``backward=True`` backward ones
    (tagged by the autograd engine's ``autograd_backward`` meta).
    """
    for submod in fx_g.modules():
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if (
                node.op not in ("call_function", "get_attr")
                or "seq_nr" not in node.meta
                or node.meta.get("autograd_backward", False) != backward
            ):
                continue
            yield node


def _copy_fwd_meta(fwd_node: torch.fx.Node, node: torch.fx.Node) -> None:
    """Copy ``custom``/``nn_module_stack``/``stack_trace`` from fwd to bwd."""
    custom = fwd_node.meta.get("custom")
    if custom:
        node.meta.setdefault("custom", {}).update(copy.deepcopy(custom))
    nn_module_stack = fwd_node.meta.get("nn_module_stack")
    if nn_module_stack is not None:
        node.meta["nn_module_stack"] = nn_module_stack.copy()
    stack_trace = fwd_node.meta.get("stack_trace")
    if stack_trace is not None:
        node.meta["stack_trace"] = stack_trace


def _fakeify_input(fake_mode: FakeTensorMode, x: Any) -> Any:
    """Convert a real tensor input into its fake counterpart."""
    if not isinstance(x, torch.Tensor):
        return x
    return fake_mode.from_tensor(x, static_shapes=False)


def _input_meta(x: Any) -> Any:
    """Return a storage-free descriptor of a traced user input.

    Tensors become shape/dtype metadata; primitives pass through; pytree
    containers (list/tuple/dict) are converted recursively. Used for
    ``JointGraph.inputs`` so the joint graph never pins the real sample
    tensors' storage for the lifetime of training.
    """
    if isinstance(x, torch.Tensor):
        return {"shape": tuple(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, tuple):
        return tuple(_input_meta(v) for v in x)
    if isinstance(x, list):
        return [_input_meta(v) for v in x]
    if isinstance(x, dict):
        return {k: _input_meta(v) for k, v in x.items()}
    return x


def trace_model_graph(
    model: torch.nn.Module,
    train_fn: Callable,
    sample_input: torch.Tensor,
    sample_label: torch.Tensor,
) -> JointGraph:
    """
    Trace model to generate complete forward + backward graph

    Args:
        model: Model (no parallel wrapping)
        train_fn: Training function signature: train_fn(model, input, label) -> loss
        sample_input: Sample input (for tracing)
        sample_label: Sample label (for tracing)

    Returns:
        JointGraph: Joint forward-backward computation graph

    Note:
        Tracing is STATIC-SHAPE: no ``ShapeEnv`` is installed, so the graph
        bakes the sample tensors' concrete shapes (symbolic ``sym_size``
        nodes would make per-stage placement ambiguous in ``PpPass``).
        This contract is shared by ALL graph-mode users, FSDP included — a
        batch whose shape differs from the compile sample must be
        re-compiled.
    """
    # Extract module state (parameters/buffers) into flat tensors threaded
    # through the graph as static inputs (leading placeholders).
    model_state = extract_module_state(model)
    state_fqns = list(model_state.keys())
    # Which leading state inputs are parameters (vs buffers). FSDP shards
    # parameters only: buffers -- e.g. the non-persistent RoPE ``cache`` --
    # are full-rank by nature and must flow through as plain inputs, never
    # all-gathered. ``state_is_param`` is indexed by state position, aligned
    # with ``state_fqns``.
    param_fqns = set(dict(model.named_parameters(remove_duplicate=False)).keys())
    state_is_param = [fqn in param_fqns for fqn in state_fqns]
    state_flat, _ = torch.utils._pytree.tree_flatten({"model": model_state})

    # user_inputs is a plain tuple (sample_input, sample_label) so the traced
    # closure unpacks it back into the two positional args of train_fn.
    user_inputs = (sample_input, sample_label)
    user_inputs_flat, user_inputs_spec = torch.utils._pytree.tree_flatten(user_inputs)

    for leaf in [*state_flat, *user_inputs_flat]:
        if isinstance(leaf, nn.Module):
            raise ValueError(
                "trace_model_graph requires explicit tensor state, not "
                "nn.Module instances. Capture nn.Modules in train_fn's closure."
            )
        if not isinstance(leaf, _ALLOWED_LEAF_TYPES):
            raise ValueError(
                "trace_model_graph requires all pytree leaves in "
                f"state/args to be tensors or primitives (int/float/bool/str), "
                f"got {type(leaf).__name__}."
            )

    full_args = list(state_flat) + list(user_inputs_flat)

    # Static shapes: no ShapeEnv -> no symbolic sizes -> the traced graph
    # carries no aten.sym_size scalar nodes (their CSE-across-phases aliasing
    # makes per-stage placement ambiguous). The trainer compiles per fixed
    # batch shape, so static tracing loses nothing here.
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake_args = tuple(
        _fakeify_input(fake_mode, a) if isinstance(a, torch.Tensor) else a
        for a in full_args
    )
    num_state_inputs = len(state_flat)

    def _fwd_bwd_fn(*plain_args):
        state_wrapped = plain_args[:num_state_inputs]
        user_wrapped = plain_args[num_state_inputs:]
        state_t = torch.utils._pytree.tree_unflatten(list(state_wrapped), state_spec)
        user_args = torch.utils._pytree.tree_unflatten(
            list(user_wrapped), user_inputs_spec
        )

        with (
            _reparametrize_train_state(model, state_t["model"]),
            _patch_engine_backward(),
        ):
            loss = train_fn(model, *user_args)

            # Read params from state_t["model"] so the exact trace-time
            # tensors (not the live module's) drive autograd.grad.
            params = []
            for idx, fqn in enumerate(state_fqns):
                if state_is_param[idx]:
                    param = state_t["model"][fqn]
                    if param.requires_grad:
                        params.append(param)

            grads = torch.autograd.grad(loss, params, allow_unused=True)

        # Replace None grads with zeros so the output arity is stable.
        processed_grads = []
        for grad, param in zip(grads, params):
            if grad is None:
                processed_grads.append(torch.zeros_like(param))
            else:
                processed_grads.append(grad)

        return [loss] + processed_grads

    # make_fx only records ``nn_module_stack`` when the traced callable
    # carries ``_orig_mod``: its ``_init_modes_from_inputs`` then installs a
    # ``_ModuleStackTracer`` that captures which module's ``forward`` was
    # executing as each node is created. Forward nodes thereby gain the
    # module-stack metadata ``PpPass._stage_from_stack`` needs to map a node
    # to its pipeline stage. Backward nodes are annotated separately (see
    # ``_annotate_autograd_backward``), after tracing.
    _fwd_bwd_fn._orig_mod = model  # type: ignore[attr-defined]

    # The pytree spec of the combined state tree is captured here so the
    # traced closure can unflatten the leading state placeholders.
    _, state_spec = torch.utils._pytree.tree_flatten({"model": model_state})

    ctx = TracingContext(fake_mode)
    with (
        fake_mode,
        tracing(ctx),
        preserve_node_meta(),
        _skip_nested_compile(),
        torch.autograd.set_multithreading_enabled(False),
        _non_strict_tracing_context(),
    ):
        traced_graph = make_fx(_fwd_bwd_fn, **_MAKE_FX_KWARGS)(*fake_args)

    # Stock torch's make_fx does not tag backward nodes (no torchtitan
    # ``_patch_engine_backward`` hook), so the joint graph splits into
    # fwd/bwd halves with no ``autograd_backward`` meta to key on.
    # ``_annotate_autograd_backward`` reconstructs the tag structurally so
    # ``PpPass`` can classify both halves.
    _annotate_autograd_backward(traced_graph)

    # Copy forward metadata (nn_module_stack/stack_trace) to backward nodes so
    # passes can match nodes by module FQN on both halves of the joint graph.
    # On torchtitan's patched torch this is the primary source (the engine
    # already tagged backward nodes); on stock torch it is a no-op because
    # backward nodes carry no ``seq_nr`` that collides with a forward one.
    _copy_fwd_metadata_to_bw_nodes(traced_graph)

    # Expose the state layout on the GraphModule so downstream passes can
    # identify which leading placeholders are parameters/buffers, and the
    # trainer can feed runtime state in the same order.
    traced_graph.state_fqns = state_fqns
    traced_graph.state_is_param = state_is_param
    traced_graph.num_state_inputs = num_state_inputs

    param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
    param_shapes = {
        name: tuple(param.shape) for name, param in model.named_parameters()
    }

    num_layers = 0
    if hasattr(model, "layers"):
        num_layers = len(model.layers)

    return JointGraph(
        graph_module=traced_graph,
        inputs=[_input_meta(sample_input), _input_meta(sample_label)],
        outputs=[],
        param_names=param_names,
        param_shapes=param_shapes,
        num_layers=num_layers,
        state_fqns=state_fqns,
        example_inputs=fake_args,
    )


def run_traced_graph(
    joint_graph: JointGraph,
    model: torch.nn.Module,
    input_batch: Any,
    label_batch: Any,
) -> tuple:
    """
    Execute a traced joint graph against the live model state.

    Parameters/buffers are sampled from ``model`` at call time and fed to the
    graph as static inputs (in ``joint_graph.state_fqns`` order). Runs
    under ``torch.no_grad()`` because the graph already contains the explicit
    backward ops traced by ``torch.autograd.grad``.

    FSDP is invisible here: FSDPPass shards ``model``'s parameters in place,
    so ``model.parameters()`` yields the shards the graph expects as its
    leading static inputs (the graph re-gathers them via AllGather each step).

    Returns:
        tuple: (loss, grads) where ``grads`` aligns with the model's
        trainable parameter list.
    """
    model_state = extract_module_state(model)
    if list(model_state.keys()) != joint_graph.state_fqns:
        raise ValueError(
            "model has different parameter/buffer names than during tracing.\n"
            f"  Traced: {joint_graph.state_fqns}\n"
            f"  Got:    {list(model_state.keys())}"
        )

    state_flat, _ = torch.utils._pytree.tree_flatten({"model": model_state})

    flat_inputs = list(state_flat) + [input_batch, label_batch]

    with torch.no_grad():
        outputs = joint_graph.graph_module(*flat_inputs)

    if isinstance(outputs, (list, tuple)):
        loss, grads = outputs[0], list(outputs[1:])
    else:
        loss, grads = outputs, []
    return loss, grads


__all__ = [
    "JointGraph",
    "trace_model_graph",
    "run_traced_graph",
    "extract_module_state",
]

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
"""Local autograd graphs for excluded regions inside a reentrant checkpoint."""
import contextlib
import contextvars
import weakref
from typing import Any, Iterator, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from . import checkpoint_exclude as recipes


_CURRENT_FRAME = contextvars.ContextVar("hyper_reentrant_frame", default=None)


@contextlib.contextmanager
def execution_frame(frame: Any) -> Iterator[None]:
    """Expose a frame only while running its forward or replay function."""
    token = _CURRENT_FRAME.set(frame)
    try:
        yield
    finally:
        _CURRENT_FRAME.reset(token)


class _BorrowedInputs:
    """Pass real activations to a Function without making them autograd inputs."""

    def __init__(self, tensors: Tuple[torch.Tensor, ...]) -> None:
        """Borrow Tensor data only until the boundary forward has detached it."""
        self.tensors = tensors
        self.slots = []


class _InputBoundary(torch.autograd.Function):
    """Connect local gradients to a zero-element root without retaining inputs."""

    @staticmethod
    def forward(ctx: Any, borrowed: _BorrowedInputs, dummy: torch.Tensor) -> tuple:
        """Return detached aliases while storing gradient slots, not activations."""
        del dummy
        ctx.set_materialize_grads(False)
        ctx.slots = [None] * len(borrowed.tensors)
        borrowed.slots = ctx.slots
        outputs = tuple(tensor.detach() for tensor in borrowed.tensors)
        borrowed.tensors = None
        return outputs

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple:
        """Record local input sensitivities for the replay bridge."""
        ctx.slots[:] = grads
        return None, None


class _LocalTape:
    """Keep the original excluded graph and only lightweight input gradient edges."""

    def __init__(self, outputs: tuple, input_edges: tuple, slots: list, frame: Any) -> None:
        """Retain the local graph while holding only gradient edges to its inputs."""
        self.outputs = outputs
        self.input_edges = input_edges
        self.slots = slots
        self.frame_ref = weakref.ref(frame)
        self.output_mask = tuple(item.requires_grad for item in outputs)
        self.param_input_mask = ()
        self.param_aliases = ()

    def gradients(self, grad_outputs: tuple) -> tuple:
        """Compute local gradients according to the owning frame's explicit phase."""
        frame = self.frame_ref()
        if frame is None:
            raise RuntimeError("Reentrant checkpoint frame was released before exclude backward")
        pairs = [(out, grad) for out, grad in zip(self.outputs, grad_outputs) if out.requires_grad and grad is not None]
        if not pairs:
            return (None,) * len(self.input_edges)
        outputs, grads = zip(*pairs)
        if frame.phase in {"ordinary", "dw"}:
            return self._backward(outputs, grads, frame.retain_graph)
        return self._dx_gradients(outputs, grads)

    def _backward(self, outputs: tuple, grads: tuple, retain_graph: bool) -> tuple:
        """Accumulate captured parameters once and return explicit input sensitivities."""
        self.slots[:] = [None] * len(self.slots)
        for alias in self.param_aliases:
            if alias is not None:
                alias.grad = None
        torch.autograd.backward(outputs, grads, retain_graph=retain_graph)
        slot_grads = iter(self.slots)
        seen_aliases = set()

        def alias_grad(alias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            """Return a shared explicit proxy gradient only at its first input edge."""
            if alias is None or id(alias) in seen_aliases:
                return None
            seen_aliases.add(id(alias))
            return alias.grad

        return tuple(
            alias_grad(alias) if is_param else next(slot_grads)
            for is_param, alias in zip(self.param_input_mask, self.param_aliases)
        )

    def _dx_gradients(self, outputs: tuple, grads: tuple) -> tuple:
        """Target only activation boundaries, leaving parameter hooks untouched."""
        targets = [edge for edge, is_param in zip(self.input_edges, self.param_input_mask)
                   if edge is not None and not is_param]
        if not targets:
            return (None,) * len(self.input_edges)
        result = iter(torch.autograd.grad(outputs, targets, grads, retain_graph=True, allow_unused=True))
        return tuple(
            next(result) if edge is not None and not is_param else None
            for edge, is_param in zip(self.input_edges, self.param_input_mask)
        )

    def release(self) -> None:
        """Drop graph outputs and boundary gradient edges after use."""
        self.outputs = ()
        self.input_edges = ()
        self.slots = []
        self.param_aliases = ()
        self.param_input_mask = ()


class _ReplayBridge(torch.autograd.Function):
    """Connect the cached local graph to current replay inputs and its trigger."""

    @staticmethod
    def forward(ctx: Any, entry: Any, input_count: int, *tensors: torch.Tensor) -> tuple:
        """Return detached outputs with the recorded differentiability mask."""
        del tensors
        ctx.entry = entry
        ctx.input_count = input_count
        ctx.set_materialize_grads(False)
        outputs = tuple(tensor.detach() for tensor in entry.tape.outputs)
        ctx.mark_non_differentiable(*(
            output for output, differentiable in zip(outputs, entry.tape.output_mask)
            if not differentiable
        ))
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        """Propagate through the original local graph using the frame phase."""
        input_grads = ctx.entry.tape.gradients(grad_outputs)
        return (None, None, *input_grads, None)


def _replace_tensor_leaves(value, tensors):
    leaves, spec = tree_flatten(value)
    replacements = iter(tensors)
    return tree_unflatten([next(replacements) if isinstance(item, torch.Tensor) else item for item in leaves], spec)


def _tensor_leaves(value: Any) -> tuple:
    """Flatten the supported input/output pytree to its Tensor leaves."""
    return tuple(item for item in tree_flatten(value)[0] if isinstance(item, torch.Tensor))


def _replay_exclude(wrapper: Any, args: tuple, kwargs: dict, frame: Any) -> Any:
    """Bind recipes and bridge cached outputs to the replay graph."""
    entry = frame.exclude_cache.pop(id(wrapper))
    frame.entries.append(entry)
    recipes._restore_rng_state(entry.rng_state)  # pylint: disable=protected-access
    recipes._materialize_recompute_inputs(entry, args, kwargs)  # pylint: disable=protected-access
    current = _tensor_leaves((args, kwargs))
    if len(current) != len(entry.tape.input_edges):
        raise RuntimeError("Checkpoint exclusion Tensor input count changed during replay")
    result = _ReplayBridge.apply(entry, len(current), *current, torch.empty(0, requires_grad=True))
    tensors = (result,) if isinstance(result, torch.Tensor) else result
    return _replace_tensor_leaves(entry.output, tensors)


def _local_inputs(original: tuple, param_ids: set) -> tuple:
    """Build a local activation boundary and explicit parameter proxies."""
    borrowed = _BorrowedInputs(tuple(item for item in original if id(item) not in param_ids))
    boundary = _InputBoundary.apply(borrowed, torch.empty(0, requires_grad=True)) if borrowed.tensors else ()
    boundary = (boundary,) if isinstance(boundary, torch.Tensor) else boundary
    replacements = iter(boundary)
    param_aliases = {}
    for item in original:
        if id(item) in param_ids and id(item) not in param_aliases:
            param_aliases[id(item)] = item.detach().requires_grad_(item.requires_grad)
    local = (param_aliases[id(item)] if id(item) in param_ids else next(replacements) for item in original)
    return tuple(local), boundary, borrowed.slots, param_aliases


def _input_edges(original: tuple, boundary: tuple, param_ids: set, param_aliases: dict) -> tuple:
    """Expose one gradient edge per original input without retaining activations."""
    boundary_edges = iter(
        torch.autograd.graph.get_gradient_edge(tensor) if tensor.requires_grad else None for tensor in boundary
    )
    seen_params = set()
    edges = []
    for item in original:
        if id(item) not in param_ids:
            edges.append(next(boundary_edges))
        elif item.requires_grad and id(item) not in seen_params:
            edges.append(torch.autograd.graph.get_gradient_edge(param_aliases[id(item)]))
            seen_params.add(id(item))
        else:
            edges.append(None)
    return tuple(edges)


def _record_exclude(wrapper: Any, module: Any, args: tuple, kwargs: dict, frame: Any) -> Any:
    """Execute an excluded region once under localized saved-tensor hooks."""
    if frame.recording_exclude:
        raise ValueError("Nested checkpoint exclusion wrappers are not supported")
    original = _tensor_leaves((args, kwargs))
    param_ids = {id(param) for param in frame.params or ()} if frame.split else set()
    with torch.enable_grad():
        local, boundary, slots, param_aliases = _local_inputs(original, param_ids)
        local_args, local_kwargs = _replace_tensor_leaves((args, kwargs), local)
        infos = recipes._capture_recompute_inputs(local_args, local_kwargs)  # pylint: disable=protected-access
        bindings = []
        swap_hooks = wrapper._make_exclude_swap_hooks()  # pylint: disable=protected-access
        token = recipes._ACTIVE_PACK_STATE.set(  # pylint: disable=protected-access
            recipes._PackState(infos, bindings, swap_hooks=swap_hooks)  # pylint: disable=protected-access
        )
        frame.recording_exclude = True
        try:
            with recipes._saved_tensors_context():  # pylint: disable=protected-access
                output = module(*local_args, **local_kwargs)
        finally:
            frame.recording_exclude = False
            recipes._ACTIVE_PACK_STATE.reset(token)  # pylint: disable=protected-access
    if swap_hooks is not None:
        wrapper._register_exclude_outputs(output)  # pylint: disable=protected-access
    outputs = _tensor_leaves(output)
    if not outputs:
        raise ValueError("Reentrant checkpoint exclusion requires at least one Tensor output")
    tape = _LocalTape(outputs, _input_edges(original, boundary, param_ids, param_aliases), slots, frame)
    tape.param_input_mask = tuple(id(item) in param_ids for item in original)
    tape.param_aliases = tuple(param_aliases.get(id(item)) for item in original)
    entry = recipes._ExcludeCacheEntry(  # pylint: disable=protected-access
        output, bindings, rng_state=recipes._capture_rng_state(  # pylint: disable=protected-access
            args, kwargs, output, frame.preserve_rng_state
        ),
        tape=tape,
    )
    frame.exclude_cache.save(id(wrapper), entry)
    detached = tuple(tensor.detach() for tensor in outputs)
    frame.excluded_output_mask.update(
        (id(item), (weakref.ref(item), mask)) for item, mask in zip(detached, tape.output_mask)
    )
    return _replace_tensor_leaves(output, detached)


def run_exclude(wrapper: Any, module: Any, args: tuple, kwargs: dict, frame: Any) -> Any:
    """Record an excluded body once, or bridge its output during replay."""
    if frame.replaying:
        return _replay_exclude(wrapper, args, kwargs, frame)
    return _record_exclude(wrapper, module, args, kwargs, frame)


def current_frame() -> Optional[Any]:
    """Return the frame active in the caller's forward/replay thread."""
    return _CURRENT_FRAME.get()

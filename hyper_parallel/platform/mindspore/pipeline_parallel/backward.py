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
"""parallel grad helper (dx/dw split)"""
from __future__ import absolute_import
from collections import deque, defaultdict
import warnings
import logging
from mindspore.utils._pytree import tree_flatten, tree_leaves, tree_unflatten
from mindspore.common.api import _pynative_executor, _GradientEdge
from mindspore._c_expression import run_backward
from mindspore.common.tensor import Tensor
from mindspore import ops


def _fill_grads(output_tensor):
    """Fill gradients with ones for given tensor(s)."""
    if isinstance(output_tensor, Tensor):
        return ops.ones_like(output_tensor)
    if isinstance(output_tensor, (list, tuple)):
        return tuple(ops.ones_like(tensor) for tensor in output_tensor)
    return None


def _validate_grad_config(grad_position, weights):
    """Validate gradient configuration."""
    if grad_position is None and not weights:
        raise ValueError("grad_position and weights cannot both be None!")


def _set_requires_grad(inputs, kwargs, grad_position):
    """Set requires_grad flag for specified inputs."""
    if grad_position is None:
        return
    if grad_position == -1:
        flatten_inputs, _ = tree_flatten(inputs, tensors_only_leaf=True)
        for inp in flatten_inputs:
            inp._requires_grad = True  # pylint: disable=protected-access
        flatten_kwargs, _ = tree_flatten(kwargs, tensors_only_leaf=True)
        for kwarg in flatten_kwargs:
            kwarg._requires_grad = True  # pylint: disable=protected-access
        return
    if isinstance(grad_position, int) and isinstance(inputs[grad_position], Tensor):
        inputs[grad_position]._requires_grad = True  # pylint: disable=protected-access
        return
    if isinstance(grad_position, tuple):
        for idx in grad_position:
            if isinstance(inputs[idx], Tensor):
                inputs[idx]._requires_grad = True  # pylint: disable=protected-access


def _get_grad_node(tensor):
    """Get the grad_fn or grad accumulator for a tensor."""
    return tensor._grad_node  # pylint: disable=protected-access


def _get_node_id(node):
    """Return stable integer id for BackwardNode (C++ `_unique_id`)."""
    return int(node._unique_id())  # pylint: disable=protected-access


def _accumulate_grads(target_tensors, grads):
    """Accumulate returned grads onto leaf tensors' internal grad storage."""
    if grads is None:
        return
    for tensor, grad in zip(target_tensors, grads):
        if tensor is None or grad is None:
            continue
        current_grad = getattr(tensor, "_grad", None)
        if current_grad is None:
            tensor._grad = grad  # pylint: disable=protected-access
        else:
            current_grad += grad


def _compute_nodes_out_degree(output_grad_fns):
    """Build reverse edges: child_id -> [(parent_node, slot_index)]."""

    queue = deque()
    visited_roots = set()
    # child_node_id -> list[(parent_node, slot_index_in_parent_next_functions)]
    backward_edges = defaultdict(list)

    # Seed from output grad_fns.
    for node in output_grad_fns:
        if node is not None:
            node_id = _get_node_id(node)
            if node_id not in visited_roots:
                queue.append(node)
                visited_roots.add(node_id)

    # Follow next_functions (towards inputs) while recording reverse edges.
    while queue:
        node = queue.popleft()
        next_fns = node.next_functions
        for slot, (child_fn, _) in enumerate(next_fns):
            if child_fn is not None:
                child_id = _get_node_id(child_fn)
                if len(backward_edges[child_id]) == 0:
                    queue.append(child_fn)
                backward_edges[child_id].append((node, slot))

    return backward_edges


def _compute_reachable_nodes(roots, boundary_nodes, backward_graph):
    """BFS on reverse edges; stop at boundary_nodes and return (reachable_ids, boundary_hits)."""
    reachable = set()
    boundary_hits = set()
    queue = deque()

    for node in roots:
        if node is not None:
            node_id = _get_node_id(node)
            if node_id not in reachable:
                reachable.add(node_id)
                queue.append(node)

    while queue:
        node = queue.popleft()
        node_id = _get_node_id(node)
        parent_entries = backward_graph.get(node_id, [])

        for parent_fn, _ in parent_entries:
            if parent_fn is None:
                continue
            parent_id = _get_node_id(parent_fn)
            if parent_id in reachable:
                continue
            if parent_id in boundary_nodes:
                boundary_hits.add(parent_fn)
                continue
            reachable.add(parent_id)
            queue.append(parent_fn)

    return reachable, boundary_hits


def _find_boundary_intermediates_with_edge_slots(weight_root, boundary_node_ids, backward_graph):
    """For one weight root, collect boundary intermediates and their edge-slot(s) on the boundary node."""
    inter_to_slots = defaultdict(set)
    if weight_root is None:
        return inter_to_slots

    q = deque([weight_root])
    seen = {_get_node_id(weight_root)}

    while q:
        node = q.popleft()
        node_id = _get_node_id(node)
        for parent, slot in backward_graph.get(node_id, ()):
            if parent is None:
                continue
            parent_id = _get_node_id(parent)
            if parent_id in boundary_node_ids:
                # Slot is already carried by reverse graph, no need to rescan parent.next_functions.
                inter_to_slots[parent].add(slot)
                continue
            if parent_id not in seen:
                seen.add(parent_id)
                q.append(parent)

    return inter_to_slots


def _group_weights_by_intermediate(input_grad_fns, weight_grad_fns, backward_graph):
    """Group weight nodes by boundary intermediates; also merge edge-slot usage."""
    # Step 1: Build input subgraph - all nodes reachable from inputs
    input_subgraph_ids, _ = _compute_reachable_nodes(input_grad_fns, set(), backward_graph)

    weight_groups = {}

    # Step 2: For each weight, find boundary nodes with input subgraph
    for weight_fn in weight_grad_fns:
        inter_to_slots = _find_boundary_intermediates_with_edge_slots(weight_fn, input_subgraph_ids, backward_graph)
        intermediate_nodes = set(inter_to_slots.keys())

        weight_group = {
            'params': {weight_fn},
            'intermediates': intermediate_nodes,
            'edge_slots': {_get_node_id(node): set(slots) for node, slots in inter_to_slots.items()},
        }

        # Merge weights with same intermediate nodes
        for intermediate_node in intermediate_nodes:
            intermediate_id = _get_node_id(intermediate_node)
            existing = weight_groups.get(intermediate_id, None)
            if existing is not None:
                existing['params'] = existing['params'].union(weight_group['params'])
                existing['intermediates'] = existing['intermediates'].union(weight_group['intermediates'])
                if existing.get('edge_slots', None) is None:
                    existing['edge_slots'] = {}
                for nid, slots in weight_group.get('edge_slots', {}).items():
                    existing['edge_slots'].setdefault(nid, set()).update(slots)
                weight_group = existing
            else:
                weight_groups[intermediate_id] = weight_group

    # Return unique weight groups (normalize to deterministic order for hooks/consumption)
    seen_groups = set()
    unique_groups = []
    for weight_group in weight_groups.values():
        group_id = id(weight_group)
        if group_id not in seen_groups:
            seen_groups.add(group_id)
            weight_group['intermediates'] = tuple(sorted(weight_group.get('intermediates', ()), key=_get_node_id))
            weight_group['params'] = tuple(sorted(weight_group.get('params', ()), key=_get_node_id))
            edge_slots = {}
            for nid, slots in weight_group.get('edge_slots', {}).items():
                edge_slots[nid] = tuple(sorted(slots))
            weight_group['edge_slots'] = edge_slots
            unique_groups.append(weight_group)

    return unique_groups


class GradFunction:
    """
    A wrapper class used to build forward output and gradient functions.
    This class supports separate computation of input gradients (dx) and weight gradients (dw),
    which is essential for pipeline parallelism.

    Args:
        output (Any): Forward output value of the network.
        inputs (tuple[Tensor, ...] | Tensor | Any): Original inputs used for forward computation.
        kwargs (dict): Keyword arguments used in forward computation.
        weights (tuple[Parameter, ...] | None): Parameters used for weight gradient calculation.
        has_aux (bool): Whether the forward output contains auxiliary values.
        grad_position (tuple[int, ...]): Positions of inputs to compute gradients for.

    Raises:
        TypeError: If `output` does not match the expected structure when `has_aux` is ``True``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, output, inputs, kwargs, weights, has_aux, grad_position):
        self.output = output
        self.inputs = inputs
        self.flatten_input_size = 0
        self.kwargs = kwargs
        self.weights = weights
        self.has_aux = has_aux
        self.grad_position = grad_position
        # Storage for intermediate gradients captured during dx computation
        self._saved_intermediates = []
        self.aux_inputs_data = None
        self.aux_weights_data = None

    def _clear_res(self):
        self.output = None
        self.inputs = None
        self.weights = None
        self._saved_intermediates = []

    def _collect_weight_tensors(self):
        """Collect weight tensors into a list."""
        if self.weights is None:
            return []
        if isinstance(self.weights, tuple):
            return list(self.weights)
        return [self.weights]

    def _setup_intermediate_hooks(self, output_tensors, input_tensors, weight_tensors):
        """Setup prehooks on intermediate nodes to capture gradients for dw computation."""
        hook_handles = []
        output_grad_fns = [_get_grad_node(t) for t in output_tensors if isinstance(t, Tensor)]
        input_grad_fns = [_get_grad_node(t) for t in input_tensors]
        weight_grad_fns = [_get_grad_node(w) for w in weight_tensors]

        # Filter out None values
        output_grad_fns = [fn for fn in output_grad_fns if fn is not None]
        input_grad_fns = [fn for fn in input_grad_fns if fn is not None]
        weight_grad_fns = [fn for fn in weight_grad_fns if fn is not None]

        if not output_grad_fns or not weight_grad_fns:
            return hook_handles

        backward_graph = _compute_nodes_out_degree(output_grad_fns)
        weight_groups = _group_weights_by_intermediate(input_grad_fns, weight_grad_fns, backward_graph)

        for weight_group in weight_groups:
            for i, intermediate in enumerate(weight_group['intermediates']):
                def make_hook(wg, idx):
                    def prehook_fn(grad_inputs):
                        if wg.get('grads', None) is None:
                            wg['grads'] = [None] * len(wg['intermediates'])
                        wg['grads'][idx] = grad_inputs
                        return grad_inputs
                    return prehook_fn
                handle = intermediate.register_prehook(make_hook(weight_group, i))
                hook_handles.append(handle)

        self._saved_intermediates = weight_groups
        return hook_handles

    def _format_input_grads(self, input_grads, input_size):
        """Format input gradients based on grad_position configuration."""
        if isinstance(self.grad_position, int) and self.grad_position == -1:
            if self.flatten_input_size == input_size:
                return tree_unflatten(self.aux_inputs_data, input_grads[:input_size])
            return (tree_unflatten(self.aux_inputs_data, input_grads[:self.flatten_input_size]),
                    tree_unflatten(self.aux_kwargs_data, input_grads[self.flatten_input_size:input_size]))
        return input_grads[0] if len(input_grads) == 1 else input_grads

    def _prune_intermediate_edges(self):
        """Prune intermediate next_edges based on recorded edge slots."""
        for weight_group in self._saved_intermediates:
            edge_slots = weight_group.get('edge_slots', {})
            for intermediate in weight_group.get('intermediates', ()):
                if intermediate is None:
                    continue
                keep_slots = set(edge_slots.get(_get_node_id(intermediate), ()))
                for slot in range(len(intermediate.next_functions)):
                    if slot not in keep_slots:
                        intermediate._set_next_edge(slot, None)  # pylint: disable=protected-access

    def _process_weight_group(self, weight_group, grad_node_to_weight, keep_graph):
        """Process a single weight group and compute gradients."""
        grad_edges = []
        grad_outputs = []

        for captured_grads, intermediate in zip(weight_group['grads'], weight_group['intermediates']):
            if captured_grads is None:
                continue
            if isinstance(captured_grads, (tuple, list)):
                for slot_idx, grad_item in enumerate(captured_grads):
                    if grad_item is not None:
                        grad_edges.append(_GradientEdge(
                            grad_node=intermediate, output_index=slot_idx, keep_alive_token=None))
                        grad_outputs.append(grad_item)
            else:
                grad_edges.append(_GradientEdge(
                    grad_node=intermediate, output_index=0, keep_alive_token=None))
                grad_outputs.append(captured_grads)

        if not keep_graph:
            del weight_group['intermediates']

        if not grad_edges:
            return {}

        group_weights = [grad_node_to_weight.get(_get_node_id(weight_fn))
                         for weight_fn in weight_group.get('params', set())]
        group_weights = [w for w in group_weights if w is not None]

        if not group_weights:
            return {}

        weight_grads = run_backward(
            tuple(grad_edges), tuple(grad_outputs),
            keep_graph, False,
            tuple(group_weights), allow_unreachable=True, accumulate_grad=False
        )
        _accumulate_grads(group_weights, weight_grads)
        if not keep_graph:
            del weight_group['grads']

        collected = {}
        for weight, grad in zip(group_weights, weight_grads):
            collected[weight] = grad if grad is not None else ops.zeros_like(weight)
        return collected

    def _prepare_output_and_sens(self, sens):
        """
        Prepare output tensor and sensitivity based on has_aux flag.

        Args:
            sens: Gradient of output tensor for gradient computation.

        Returns:
            Tuple of (output_tensor, processed_sens)
        """
        output_tensor = self.output
        if self.has_aux:
            if not isinstance(self.output, (tuple, list)):
                raise TypeError(
                    f"The output of fn should be list or tuple when has_aux=True, "
                    f"but got {type(self.output)}"
                )
            output_tensor = output_tensor[0]
            if isinstance(sens, (tuple, list)):
                sens = sens[0]
            return (output_tensor,), (sens,)
        flatten_outputs = tree_leaves(output_tensor, tensors_only_leaf=True)
        if sens is None:
            sens = _fill_grads(flatten_outputs)
        else:
            sens = tree_leaves(sens, tensors_only_leaf=True)
        return tuple(flatten_outputs), tuple(sens)

    def _collect_input_tensors(self, collect_weights=True):
        """Collect input tensors based on grad_position and weights."""
        input_tensors = []
        if isinstance(self.grad_position, int) and self.grad_position == -1:
            flatten_inputs, self.aux_inputs_data = tree_flatten(self.inputs, tensors_only_leaf=True)
            flatten_kwargs, self.aux_kwargs_data = tree_flatten(self.kwargs, tensors_only_leaf=True)
            self.flatten_input_size = len(flatten_inputs)
            input_tensors.extend(flatten_inputs)
            input_tensors.extend(flatten_kwargs)
        elif isinstance(self.grad_position, int) and isinstance(self.inputs[self.grad_position], Tensor):
            input_tensors.append(self.inputs[self.grad_position])
        elif isinstance(self.grad_position, (list, tuple)):
            input_tensors.extend(self.inputs[idx] for idx in self.grad_position if isinstance(self.inputs[idx], Tensor))

        input_size = len(input_tensors)
        if self.weights is not None and collect_weights:
            if isinstance(self.weights, (list, tuple)):
                input_tensors.extend(self.weights)
            else:
                input_tensors.append(self.weights)

        return tuple(input_tensors), input_size

    def __call__(self, sens=None, keep_graph=False):
        """
        Compute gradients with respect to both inputs and weights.

        Args:
            sens: gradient of output tensor for gradient computation.
            keep_graph: Whether to keep the computation graph.

        Returns:
            Gradients with respect to inputs and/or weights.
        """
        weights = self.weights
        input_tensors, input_size = self._collect_input_tensors()
        output_tensors, sens = self._prepare_output_and_sens(sens)

        grads = run_backward(
            output_tensors, sens, keep_graph, keep_graph,
            input_tensors, allow_unreachable=True, accumulate_grad=False
        )
        if input_size > 0:
            _accumulate_grads(input_tensors[:input_size], grads[:input_size])
        weight_tensors = self._collect_weight_tensors()
        if weight_tensors:
            _accumulate_grads(weight_tensors, grads[input_size:])
        if not keep_graph:
            self._clear_res()
        if input_size == 0:
            return grads
        if weights is None:
            if isinstance(self.grad_position, int) and self.grad_position == -1:
                if self.flatten_input_size == input_size:
                    return tree_unflatten(self.aux_inputs_data, grads)
                return (tree_unflatten(self.aux_inputs_data, grads[:self.flatten_input_size]),
                        tree_unflatten(self.aux_kwargs_data, grads[self.flatten_input_size:input_size]))
            return grads[0] if len(grads) == 1 else grads
        if isinstance(self.grad_position, int) and self.grad_position == -1:
            if self.flatten_input_size == input_size:
                return tree_unflatten(self.aux_inputs_data, grads[:input_size]), grads[input_size:]
            return (tree_unflatten(self.aux_inputs_data, grads[:self.flatten_input_size]),
                    tree_unflatten(self.aux_kwargs_data, grads[self.flatten_input_size:input_size]),
                    grads[input_size:])
        return grads[:input_size], grads[input_size:]

    def compute_input_grad(self, sens=None):
        """
        Compute gradients with respect to inputs only (dx).

        This is the first stage of dx/dw split computation. It computes input gradients
        while capturing intermediate gradients at the boundaries between input and weight subgraphs.

        Implementation Strategy:
        1. Compute graph out degree from output grad_fns
        2. Find intermediate nodes (boundaries between input/weight subgraphs)
        3. Register prehooks on these intermediate nodes to capture gradients
        4. Save captured intermediate gradients for later dw computation

        Args:
            sens: gradient of output tensor for gradient computation. If None, will use ones_like.

        Returns:
            Gradients with respect to inputs. Returns single tensor if one input,
            tuple of tensors if multiple inputs.

            When grad_position=-1 (default), the return type matches the input structure,
            supporting complex input types (tuple, dict, etc.). The gradients are automatically
            unflattened to preserve the original input structure.

        Raises:
            ValueError: If grad_position is None (no inputs specified for differentiation).
        """
        if self.grad_position is None:
            raise ValueError(
                "compute_input_grad requires grad_position to be specified. "
                "Cannot compute input gradients when grad_position is None."
            )
        input_tensors, input_size = self._collect_input_tensors(collect_weights=False)
        if not input_tensors:
            logging.info("No valid input tensors found for gradient computation.")
            return ()

        weight_tensors = self._collect_weight_tensors()
        self._saved_intermediates = []
        output_tensors, sens = self._prepare_output_and_sens(sens)

        hook_handles = []
        if weight_tensors:
            hook_handles = self._setup_intermediate_hooks(output_tensors, input_tensors, weight_tensors)

        input_grads = run_backward(
            output_tensors, sens, True, False,
            tuple(input_tensors), allow_unreachable=True, accumulate_grad=False
        )
        _accumulate_grads(input_tensors, input_grads)

        for handle in hook_handles:
            handle.remove()

        return self._format_input_grads(input_grads, input_size)

    def compute_weight_grad(self, keep_graph=False):
        """
        Compute gradients with respect to weights only (dw).

        This is the second stage of dx/dw split computation. It uses the intermediate
        gradients captured during compute_input_grad to compute weight gradients efficiently,
        starting from intermediate nodes rather than recomputing from the output.

        Implementation Strategy:
        1. Use saved intermediate gradients and GradientEdges from compute_input_grad
        2. Start backward computation from these intermediate points (not from output)
        3. This avoids recomputing the portion of the graph already computed in dx

        Args:
            keep_graph: Whether to keep the computation graph after this computation.
                       Default is False as this is typically the final gradient computation.

        Returns:
            Gradients with respect to weights. Returns single tensor if one weight,
            tuple of tensors if multiple weights.

        Raises:
            ValueError: If weights is None (no weights specified for differentiation).
            RuntimeError: If compute_input_grad was not called before (no saved intermediates).

        Note:
            This function must be called after compute_input_grad, as it relies on
            the intermediate gradients captured during that computation. The computation
            graph must still be available (retained by keep_graph=True in compute_input_grad).
        """

        if self.weights is None:
            warnings.warn(
                "compute_weight_grad requires weights to be specified. "
                "Cannot compute weight gradients when weights is None."
            )
            self._clear_res()
            return ()
        if not self._saved_intermediates:
            raise RuntimeError("Before calling compute_weight_grad, you need first call compute_input_grad!")

        weight_tensors = self._collect_weight_tensors()
        if not weight_tensors:
            raise ValueError("No valid weight tensors found for gradient computation.")

        grad_node_to_weight = {_get_node_id(_get_grad_node(w)): w
                               for w in weight_tensors if _get_grad_node(w) is not None}

        self._prune_intermediate_edges()

        collected_grads = {}
        for weight_group in self._saved_intermediates:
            group_grads = self._process_weight_group(weight_group, grad_node_to_weight, keep_graph)
            collected_grads.update(group_grads)

        if not keep_graph:
            self._clear_res()

        result_grads = [collected_grads.get(weight) for weight in weight_tensors]
        return tuple(result_grads)


def forward_and_gradfn(fn, *inputs, weights=None, has_aux=False, grad_position=-1, **kwargs):
    """
    A wrapper function to generate the function to calculate forward output and gradient function.

    As for gradient, three typical cases are included:

    1. gradient with respect to inputs. In this case, `grad_position` is not None while `weights` is ``None``.
    2. gradient with respect to weights. In this case, `grad_position` is None while `weights` is not ``None``.
    3. gradient with respect to inputs and weights. In this case, `grad_position` and `weights` are not ``None``.

    Args:
        fn (Union[Cell, Function]): Function to do GradOperation.
        *inputs: Variable length argument list of inputs to the function `fn`.
        weights (Union[ParameterTuple, Parameter, list[Parameter]], optional):
            The parameters of the training network that need to
            calculate the gradient. `weights` can be got through `weights = net.trainable_params()` .
            Default: ``None`` .
        has_aux (bool, optional): If ``True`` , only the first output of `fn` contributes the gradient of `fn`,
            while the other outputs will be returned straightly. It means the `fn` must return more than one outputs
            in this case.
            Default: ``False`` .
        grad_position (Union[NoneType, int, tuple[int]], optional): Index to specify which inputs
            to be differentiated. Default: ``-1`` means all inputs are differentiated.

            - If int, get the gradient with respect to single input.
            - If tuple, get the gradients with respect to selected inputs. `grad_position` begins with 0.
            - If None, none derivative of any input will be solved, and in this case, `weights` is required.

        **kwargs: Variable length keyword argument dictionary. Additional keyword arguments passed to the function `fn`.

    Returns:
        Tuple of (output, grad_fn):

        - output: The output value of function `fn`.
        - grad_fn: A :class:`GradFunction` instance used to compute gradients.

        The :class:`GradFunction` class provides methods for gradient computation:

        - :meth:`__call__`: Compute gradients with respect to both inputs and weights at once.
        - :meth:`compute_input_grad`: Compute gradients with respect to inputs only (dx).
          This is the first stage of dx/dw split computation, which captures intermediate
          gradients at weight nodes for efficient dw computation.
        - :meth:`compute_weight_grad`: Compute gradients with respect to weights only (dw).
          This is the second stage of dx/dw split computation, which uses the intermediate
          gradients captured during :meth:`compute_input_grad` to compute weight gradients
          efficiently without recomputing from the output.

    Examples:
        When grad_position=-1 (default), the gradient return type matches the input structure,
        supporting complex input types (tuple, dict, etc.):

        >>> def fn(x, tuple_input, scale=None):
        ...     a, b = tuple_input
        ...     return x * a + x * b + scale * x
        >>> x = Tensor([2.0])
        >>> a, b = Tensor([3.0]), Tensor([4.0])
        >>> scale = Tensor([0.5])
        >>> _, grad_fn = forward_and_gradfn(fn, x, (a, b), grad_position=-1, scale=scale)
        >>> dx_grads = grad_fn.compute_input_grad()
        >>> # When both args and kwargs have tensors, returns (args_grads, kwargs_grads)
        >>> # args_grads structure matches (x, (a, b)) -> (Tensor, (Tensor, Tensor))
        >>> # kwargs_grads structure matches {'scale': scale} -> {'scale': Tensor}
        >>> print(type(dx_grads))
        <class 'tuple'>
        >>> print(len(dx_grads))
        2
        >>> print(type(dx_grads[0]))
        <class 'tuple'>
        >>> print(type(dx_grads[0][1]))
        <class 'tuple'>
        >>> print(type(dx_grads[1]))
        <class 'dict'>

    Raises:
        ValueError: If both `grad_position` and `weights` are ``None``.
        TypeError: If type of Args does not belong to required ones.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    _validate_grad_config(grad_position, weights)
    _set_requires_grad(inputs, kwargs, grad_position)
    prev_grad_flag = _pynative_executor.grad_flag()
    _pynative_executor.set_grad_flag(True)
    try:
        res = fn(*inputs, **kwargs)
    except Exception as e:
        _pynative_executor.clear_res()
        raise e
    finally:
        _pynative_executor.set_grad_flag(prev_grad_flag)
    grad_fn = GradFunction(res, inputs, kwargs, weights, has_aux, grad_position)
    return res, grad_fn

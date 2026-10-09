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
"""Lazy FX shape specialization of an already transformed joint graph."""

import logging
import operator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import fx
from torch.utils import _pytree as pytree

from .tracer.dynamic_shapes import _resolve_input
from .tracer.graph_tracer import JointGraph


_LOG = logging.getLogger(__name__)
_SHAPE_OPS = {
    torch.ops.aten.sym_size.int, torch.ops.aten.sym_stride.int,
    torch.ops.aten.sym_numel.default, torch.ops.aten.sym_storage_offset.default,
}
_SCALAR_OPS = {operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod, operator.neg}


def validate_compile_sizes(sizes: Optional[list[int]], max_specializations: int) -> tuple[int, ...]:
    """Validate the bounded, opt-in specialization policy.

    Args:
        sizes: Optional non-negative sizes eligible for specialization.
        max_specializations: Positive capacity for full input signatures.

    Returns:
        Unique configured sizes in their original order.
    """
    if sizes is not None and (
        not isinstance(sizes, (list, tuple))
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in sizes)
    ):
        raise ValueError("compile_sizes must be a list of non-negative integers or None")
    if (not isinstance(max_specializations, int) or isinstance(max_specializations, bool)) or max_specializations < 1:
        raise ValueError("max_specializations must be a positive integer")
    return tuple(dict.fromkeys(sizes or ()))


def specialize_graph(joint_graph: JointGraph, user_inputs: list[Any]) -> tuple[fx.GraphModule, int]:
    """Generate FX code with bound shape expressions, without executing tensor ops.

    The caller must validate the general graph guards first. Only pure shape
    queries and integer arithmetic are folded; weights, token counts, RNG and
    collectives remain runtime operations. No model capture or parallel pass is
    repeated, so an already sharded live model is never sharded again.

    Args:
        joint_graph: Guarded, already transformed symbolic joint graph.
        user_inputs: Flattened runtime user inputs matching the general guards.

    Returns:
        Generated FX graph and number of folded shape expressions.
    """
    guards = joint_graph.input_guards
    tensors = [value for value in user_inputs if isinstance(value, torch.Tensor)]
    bindings = guards.shape_env.bind_symbols(guards.fake_tensors, tensors)
    graph = fx.Graph()
    replacements = {}
    folded = 0
    for node in joint_graph.graph_module.graph.nodes:
        value = node.meta.get("val")
        if node.op == "call_function" and node.target in _SHAPE_OPS | _SCALAR_OPS:
            if isinstance(value, torch.SymInt):
                concrete = value.node.expr.xreplace(bindings)
                if isinstance(concrete, int) or (not concrete.free_symbols and concrete.is_Integer):
                    replacements[node] = int(concrete)
                    folded += 1
                    continue
        copied = graph.node_copy(node, lambda arg: replacements[arg])
        # Symbolic tensor metadata would misdescribe this concrete variant.
        copied.meta = {key: val for key, val in node.meta.items() if key not in ("val", "tensor_meta", "example_value")}
        replacements[node] = copied
    graph.lint()
    return fx.GraphModule(joint_graph.graph_module, graph), folded


@dataclass
class ConcreteSizeEntry:
    """One generated variant for a full input metadata signature."""

    runtime_size: int
    graph_module: fx.GraphModule
    folded_nodes: int


class SizeSpecializationDispatcher:
    """Use a general graph once, then lazily generate configured size variants.

    Selection uses one size, while caching includes every input tensor's shape,
    stride and storage offset. This prevents reuse across incompatible secondary
    dimensions. Capacity is bounded; new signatures at capacity use the general
    graph. All dispatch decisions occur after the general guards have passed.
    """

    def __init__(
        self, joint_graph: JointGraph, inputs: dict[str, Any], compile_sizes: tuple[int, ...],
        compile_size_input: Optional[str], compile_size_dim: int, max_specializations: int,
    ) -> None:
        """Resolve dispatch selection and initialize the bounded variant cache."""
        self.joint_graph = joint_graph
        self.compile_sizes = frozenset(compile_sizes)
        self.max_specializations = max_specializations
        self.entries: dict[tuple, ConcreteSizeEntry] = {}
        self.general_calls = 0
        self.specialized_calls = 0
        self.cache_hits = 0
        self.capacity_fallbacks = 0
        self.last_dispatch = "uninitialized"
        self.num_state_inputs = len(joint_graph.state_fqns)
        user_flat, _ = pytree.tree_flatten(inputs)
        self.selector = self._resolve_selector(user_flat, inputs, compile_size_input, compile_size_dim)

    def _resolve_selector(self, user_flat, inputs, path, dim):
        if (not isinstance(dim, int) or isinstance(dim, bool)):
            raise ValueError("compile_size_dim must be an integer")
        if path is not None:
            if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
                raise ValueError("compile_size_input must be a non-empty dotted input path")
            tensor = _resolve_input(inputs, path)
            if not -tensor.ndim <= dim < tensor.ndim:
                raise ValueError("compile_size_dim is out of range for compile_size_input")
            for index, value in enumerate(user_flat):
                if value is tensor:
                    return index, dim % tensor.ndim
            raise ValueError("compile_size_input must resolve to a registered pytree tensor leaf")
        return self._first_symbolic_dimension()

    def _first_symbolic_dimension(self) -> tuple[int, int]:
        """Select the first captured symbolic user dimension by default."""
        fake_users = self.joint_graph.example_inputs[self.num_state_inputs:]
        for index, value in enumerate(fake_users):
            if isinstance(value, torch.Tensor):
                for axis, size in enumerate(value.shape):
                    if isinstance(size, torch.SymInt) and size.node.expr.free_symbols:
                        return index, axis
        raise ValueError("compile_sizes needs a symbolic input dimension or an explicit compile_size_input")

    @staticmethod
    def _signature(user_inputs: list[Any]) -> tuple:
        return tuple(
            (tuple(value.shape), value.stride(), value.storage_offset(), value.dtype, value.device,
             value.layout, value.requires_grad)
            if isinstance(value, torch.Tensor) else (type(value), value)
            for value in user_inputs
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Return a snapshot suitable for logging and JSON audit files."""
        return {
            "general_calls": self.general_calls, "specialized_calls": self.specialized_calls,
            "cache_hits": self.cache_hits, "compilations": len(self.entries),
            "cached_variants": len(self.entries), "capacity_fallbacks": self.capacity_fallbacks,
            "compiled_sizes": sorted({entry.runtime_size for entry in self.entries.values()}),
            "folded_nodes": sum(entry.folded_nodes for entry in self.entries.values()),
            "last_dispatch": self.last_dispatch,
        }

    def __call__(self, *flat_inputs: Any) -> Any:
        """Execute the selected graph with live state and live tensor values."""
        user_inputs = list(flat_inputs[self.num_state_inputs:])
        index, dim = self.selector
        size = user_inputs[index].shape[dim]
        if self.general_calls == 0 or size not in self.compile_sizes:
            return self._run_general(flat_inputs)
        key = self._signature(user_inputs)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_specializations:
                self.capacity_fallbacks += 1
                return self._run_general(flat_inputs)
            graph_module, folded = specialize_graph(self.joint_graph, user_inputs)
            entry = ConcreteSizeEntry(size, graph_module, folded)
            self.entries[key] = entry
            _LOG.info("Generated size-specialized FX graph: size=%d, folded_nodes=%d, variants=%d",
                      size, folded, len(self.entries))
        else:
            self.cache_hits += 1
        self.specialized_calls += 1
        self.last_dispatch = "specialized"
        return entry.graph_module(*flat_inputs)

    def _run_general(self, flat_inputs):
        self.general_calls += 1
        self.last_dispatch = "general"
        return self.joint_graph.graph_module(*flat_inputs)

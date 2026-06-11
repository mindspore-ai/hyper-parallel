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
"""
Compute graph abstractions: OpType, TensorSpec, SplitSpec, OperatorNode, ComputeGraph.

ComputeGraph.propagate_splits() traverses the DAG in topological order and
automatically computes task_num / split_dim for every node from their SplitSpec.
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue


class OpType(Enum):
    ALLTOALL    = "alltoall"
    GMM         = "gmm"
    SWIGLU      = "swiglu"
    SWIGLU_GRAD = "swiglu_grad"


@dataclass
class TensorSpec:
    """Describes a single tensor in the graph (a graph edge / shared buffer)."""
    name:           str
    shape:          list
    param_position: int
    dtype_size:     int  = 2     # bf16=2, int64=8, int32=4
    tensor_type:    int  = 0     # 0=tensor, 1=tensorlist
    transpose:      bool = False
    is_dynamic:     bool = False

    # Set by propagate_splits(); not user-supplied.
    split_dim: int = field(default=-1, init=False)
    split_num: int = field(default=1,  init=False)


@dataclass
class SplitSpec:
    """
    Declarative split specification attached to an OperatorNode.

    split_inputs
        None  → source operator; always computes via task_num_fn.
        list  → list of (input_idx, dim) pairs; ALL must already be split on
                the given dim for this op to be active.

    split_output_dims
        Per-output split axis.  -1 = leave un-split.

    task_num_fn
        Callable(tsv) -> int.  Returns task_num when the split condition holds.
    """
    split_inputs:      Optional[List[Tuple[int, int]]]
    task_num_fn:       Callable
    split_output_dims: List[int] = field(default_factory=lambda: [0])


@dataclass
class OperatorNode:
    """A single operator node in the compute DAG."""
    name:            str
    op_type:         OpType
    inputs:          List[TensorSpec]
    outputs:         List[TensorSpec]
    param_positions: List[int]
    split_value:     int
    split_spec:      SplitSpec
    tiling_position: int
    fill_config:     Any                       # FillConfig subclass instance
    kernel_spec:     Any = None                # reserved for @MultiCore decorator path

    predecessors:    List['OperatorNode'] = field(default_factory=list)
    successors:      List['OperatorNode'] = field(default_factory=list)
    task_num:        int = field(default=0, init=False)


class ComputeGraph:
    """Directed acyclic graph describing operator execution order."""

    def __init__(self):
        self._nodes: dict = {}
        self._insertion_order: list = []

    def add_op(self, op: OperatorNode) -> 'ComputeGraph':
        """Add an operator node to the graph."""
        self._nodes[op.name] = op
        self._insertion_order.append(op.name)
        return self

    def add_edge(self, src, dst) -> 'ComputeGraph':
        """src / dst can be an OperatorNode or a name string."""
        s = src if isinstance(src, OperatorNode) else self._nodes[src]
        d = dst if isinstance(dst, OperatorNode) else self._nodes[dst]
        s.successors.append(d)
        d.predecessors.append(s)
        return self

    def get_op(self, name: str) -> OperatorNode:
        """Retrieve an operator node by name."""
        return self._nodes[name]

    def topological_sort(self) -> List[OperatorNode]:
        """Kahn's algorithm; insertion order used for tie-breaking."""
        in_deg = {n: len(op.predecessors) for n, op in self._nodes.items()}
        queue  = deque(n for n in self._insertion_order if in_deg[n] == 0)
        order  = []
        while queue:
            n = queue.popleft()
            order.append(self._nodes[n])
            for succ in self._nodes[n].successors:
                in_deg[succ.name] -= 1
                if in_deg[succ.name] == 0:
                    queue.append(succ.name)
        if len(order) != len(self._nodes):
            raise ValueError("ComputeGraph has a cycle")
        return order

    def propagate_splits(self, tsv: TaskSplitValue) -> None:
        """
        Compute task_num and split_dim for every node from their SplitSpec.

        Must be called once after graph construction and before any fill loop.
        Resets all TensorSpec split annotations before traversal.
        """
        for op in self._nodes.values():
            for t in op.inputs + op.outputs:
                t.split_dim = -1
                t.split_num = 1

        for op in self.topological_sort():
            ss = op.split_spec
            if ss.split_inputs is None:
                task_num = ss.task_num_fn(tsv)
            else:
                if all(op.inputs[idx].split_dim == dim for idx, dim in ss.split_inputs):
                    task_num = ss.task_num_fn(tsv)
                else:
                    task_num = 1

            op.task_num = task_num

            for i, out in enumerate(op.outputs):
                if i < len(ss.split_output_dims):
                    d = ss.split_output_dims[i]
                    out.split_dim = d if (task_num > 1 and d >= 0) else -1
                else:
                    out.split_dim = -1
                out.split_num = task_num

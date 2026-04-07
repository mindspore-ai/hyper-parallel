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
ComputeGraph, OperatorNode, TensorSpec, SplitSpec, and TaskSplitValue for Mega-Kernel scheduling.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple
from collections import deque


class OpType(Enum):
    ALLTOALL    = "alltoall"
    GMM         = "gmm"
    SWIGLU      = "swiglu"
    SWIGLU_GRAD = "swiglu_grad"


@dataclass
class TensorSpec:
    """Describes a single tensor in the graph (a graph edge)."""
    name:           str
    shape:          list
    param_position: int
    dtype_size:     int  = 2     # bf16=2, int64=8, int32=4
    tensor_type:    int  = 0     # 0=tensor, 1=tensorlist
    transpose:      bool = False
    is_dynamic:     bool = False

    # Set by propagate_splits; not user-supplied
    split_dim: int = field(default=-1, init=False)
    split_num: int = field(default=1,  init=False)


@dataclass
class SplitSpec:
    """
    Declarative split specification for an OperatorNode.

    split_inputs
        None  → source operator; always splits.
        list  → list of (input_idx, split_dim) pairs; ALL must match.

    split_output_dims
        Per-output split axis.  -1 = leave un-split.

    task_num_fn
        Callable(tsv) -> int.  Computes task_num when split condition holds.
    """
    split_inputs:      Optional[List[Tuple[int, int]]]
    task_num_fn:       Callable
    split_output_dims: List[int] = field(default_factory=lambda: [0])


@dataclass
class OperatorNode:
    """A single operator node in the compute graph."""
    name:            str
    op_type:         OpType
    inputs:          List[TensorSpec]
    outputs:         List[TensorSpec]
    param_positions: List[int]
    split_value:     int
    split_spec:      SplitSpec
    tiling_position: int
    fill_config:     Any                           # FillConfig subclass instance
    kernel_spec:     Any = None                    # KernelSpec; None for manual graphs, required for @MultiCore path

    predecessors:    List['OperatorNode'] = field(default_factory=list)
    successors:      List['OperatorNode'] = field(default_factory=list)
    task_num:        int = field(default=0, init=False)


@dataclass
class TaskSplitValue:
    """
    Hardware topology parameters + per-rank runtime counters.
    Only contains topology (user inputs + derived) and counters.
    Split values and task counts live on OperatorNode, not here.
    """
    # ── User inputs ───────────────────────────────────────────────────────────
    tp:             int = 4
    ep:             int = 4
    seq_size:       int = 8192
    all_expert_num: int = 32
    top_k:          int = 8

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def single_rank_expert_num(self) -> int:
        return self.all_expert_num // self.ep

    @property
    def seq_all(self) -> int:
        return (self.seq_size * self.ep * self.top_k) // self.tp

    @property
    def per_expert_seq(self) -> int:
        return self.seq_all // self.top_k

    @property
    def per_rank_seq(self) -> int:
        return self.seq_all // self.ep

    @property
    def per_expert_seq_to_other(self) -> int:
        return self.seq_all // (self.ep * self.top_k)

    @property
    def all_event_num(self) -> int:
        e = self.single_rank_expert_num
        return 1 + self.all_expert_num + e + e + e

    # ── Runtime counters (reset by init_task_split_value per rank) ────────────
    rank_id:              int = 0
    pre_pre_event_num:    int = 0
    pre_event_num:        int = 0
    pre_task_num:         int = 0
    pre_cube_task_num:    int = 0
    pre_vector_task_num:  int = 0
    pre_mix_task_num:     int = 0


def init_task_split_value(tsv: TaskSplitValue) -> None:
    """Reset per-rank runtime counters."""
    tsv.pre_pre_event_num   = 0
    tsv.pre_event_num       = 0
    tsv.pre_task_num        = 0
    tsv.pre_cube_task_num   = 0
    tsv.pre_vector_task_num = 0
    tsv.pre_mix_task_num    = 0


class ComputeGraph:
    """Directed acyclic graph describing operator execution order."""

    def __init__(self):
        self._nodes: dict = {}
        self._insertion_order: list = []

    def add_op(self, op: OperatorNode) -> 'ComputeGraph':
        self._nodes[op.name] = op
        self._insertion_order.append(op.name)
        return self

    def add_edge(self, src, dst) -> 'ComputeGraph':
        """src / dst can be an OperatorNode object or a name string."""
        s = src if isinstance(src, OperatorNode) else self._nodes[src]
        d = dst if isinstance(dst, OperatorNode) else self._nodes[dst]
        s.successors.append(d)
        d.predecessors.append(s)
        return self

    def get_op(self, name: str) -> OperatorNode:
        """Look up an operator by name."""
        return self._nodes[name]

    def topological_sort(self) -> List[OperatorNode]:
        """Kahn's algorithm; respects add_op insertion order for tie-breaking."""
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
        Compute task_num for each operator from SplitSpec and propagate
        split_dim through shared TensorSpec objects.

        Must be called once after graph construction, before any fill loop.
        """
        # Reset all tensor split info
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

    def build_runtime_config(self, tsv: 'TaskSplitValue', rank_id: int = 0,
                             num_cube_cores: int = 24):
        """
        Generic RuntimeConfig builder for the framework (@MultiCore) path.

        Runs init_task_split_value, the topological fill loop, and sets
        cfg.task_num / cfg.atomic_add_values[0].  Graph-specific post-processing
        (add_terminate, add_dynamic_data, revise_task_queue) is intentionally
        omitted — call build_config_for_rank() in gen_runtime_data.py for MoE FFN.
        """
        from hyper_parallel.core.multicore.modules.common.runtime_structs import (  # pylint: disable=import-outside-toplevel
            RuntimeConfigC, QUEUE_CAPACITY)

        cfg = RuntimeConfigC()
        cfg.num_workers    = 2 * num_cube_cores
        cfg.queue_capacity = QUEUE_CAPACITY

        init_task_split_value(tsv)
        tsv.rank_id = rank_id

        for op in self.topological_sort():
            op.fill_config.fill(cfg, op, tsv)

        cfg.task_num = sum(op.task_num for op in self.topological_sort())
        cfg.atomic_add_values[0] = 1
        return cfg

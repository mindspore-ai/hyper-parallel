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
"""CommDebugMode — context manager that traces DTensor ops and collectives.

Public API mirrors ``torch.distributed.tensor.debug.CommDebugMode``:

    get_comm_counts()
    get_total_counts()
    get_parameter_info()
    get_sharding_info()
    generate_comm_debug_tracing_table(noise_level)
    log_comm_debug_tracing_table_to_file(file_name, noise_level)

Record retention is bounded on purpose. A traced step appends one node per
DTensor op, so an unbounded tree would grow with the step count and pin host
memory for the whole process lifetime. Use :meth:`CommDebugMode.clear` to
release the collected records as soon as they have been consumed.
"""
# pylint: disable=C9006,C9007
import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from hyper_parallel.core.dtensor.debug._call_records import (
    AnnotateCall,
    CollectiveCall,
    DebugCall,
    OpCall,
    TensorInfo,
)
from hyper_parallel.core.dtensor.debug._collective_tracer import CollectiveTracer
from hyper_parallel.core.dtensor.debug._module_tracker import ModuleTracker

logger = logging.getLogger(__name__)
Tensor = torch.Tensor

# Argument index of the process group for each traced collective method.
# Derived from the ``_utils`` function signatures:
#   differentiable_all_gather_concat(data, group, concat_size, concat_dim, ...)
#   differentiable_all_to_all(input_data, output_shape, group)
#   differentiable_all_reduce(data, op, group)
#   differentiable_reduce_scatter(data, dev_num, axis, op, group)
#   differentiable_all_to_all_single(input_tensor, input_splits, output_splits, group)
#   differentiable_all_to_all_single_async(input_tensor, input_splits, output_splits, group)
_COLLECTIVE_GROUP_ARG_INDEX: Dict[str, int] = {
    "differentiable_all_gather_concat": 1,
    "differentiable_all_to_all": 2,
    "differentiable_all_reduce": 2,
    "differentiable_reduce_scatter": 4,
    "differentiable_all_to_all_single": 3,
    "differentiable_all_to_all_single_async": 3,
}

# Hard ceiling on the number of live record nodes. Tracing an entire training
# run is not a supported use case — the resulting table is unreadable and the
# nodes cost host memory — so past this point records are counted and dropped
# instead of retained. The counters (comm_counts, dropped count) stay exact.
_MAX_RECORDS = 200000

# Deepest call nesting that still gets a node. Deeper calls are counted but not
# retained, which also stops recursion in the rendering helpers from growing
# with the traced program's call depth.
_MAX_DEPTH = 128


class CommDebugMode:
    """Context manager that records DTensor operator dispatches and collective
    communication operations, producing a hierarchical call tree.

    Usage::

        with CommDebugMode() as mode:
            output = model(input_dtensor)
        print(mode.generate_comm_debug_tracing_table())
        print(mode.get_comm_counts())

    Long-running training:
        Tracing keeps one record node per dispatched op, so a mode object that
        spans a whole training loop retains every op of every step. Trace a
        bounded slice instead, read the results, then release them::

            mode = CommDebugMode()
            for step in range(total_steps):
                # Re-entering resets the previous window, so this alone stays
                # bounded — but only if each window is read before the next.
                if step % report_every == 0:
                    with mode:
                        loss = train_step()
                    print(f"step {step}: {mode.get_comm_counts()}")
                    mode.clear()          # release this window's records now
                else:
                    loss = train_step()

        ``clear()`` is what actually releases the memory; without it the last
        window stays resident until the next ``__enter__``. Entries returned by
        ``get_parameter_info()`` alias the model's parameter storages, so they
        pin the model in memory for as long as they are held — ``clear()``
        drops those too. If a window is left traced too long, records are
        dropped once the budget is exhausted; ``get_dropped_record_count()``
        reports how many, and both the counters and a ``logger.warning`` stay
        accurate, but the rendered table will be incomplete.

    Args:
        module: Optional ``nn.Module`` to track forward enter/exit events.
    """

    def __init__(self, module=None):
        self._module = module

        # ---- tracing state ----
        self._call_stack: List[DebugCall] = []
        self._root_records: List[DebugCall] = []
        self._record_count = 0
        self._dropped_records = 0
        self._comm_counts: Dict[str, int] = defaultdict(int)
        # ---- module-level info (populated when module is provided) ----
        self._parameter_info: Dict[str, Dict[str, Any]] = {}
        self._sharding_info: Dict[str, Any] = {}

        # ---- internal handles ----
        self._collective_tracer: Optional[CollectiveTracer] = None
        self._module_tracker: Optional[ModuleTracker] = None
        # Tokens are pushed in ``__enter__`` and popped LIFO in ``__exit__``, so
        # out-of-order exits restore the observer that was actually active
        # before this instance took over rather than a stale token.
        self._observer_tokens: List = []
        self._active = False

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import _debug_mode_observer

        # Release anything left over from a previous window before collecting
        # into a fresh tree.
        self._reset_records()

        self._observer_tokens.append(_debug_mode_observer.set(self))
        self._active = True

        self._collective_tracer = CollectiveTracer(self._on_collective_call)
        self._collective_tracer.install()

        if self._module is not None:
            self._module_tracker = ModuleTracker(self._module, self._on_module_event)
            self._module_tracker.install()
            self._collect_module_info()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import _debug_mode_observer

        # Leaving the window stops collection. Anything that reads results
        # (get_comm_counts, generate_comm_debug_tracing_table, ...) keeps
        # working afterwards, so the records themselves are not dropped here —
        # call clear() once they have been consumed.
        self._active = False

        if self._module_tracker is not None:
            self._module_tracker.uninstall()
            self._module_tracker = None

        if self._collective_tracer is not None:
            self._collective_tracer.uninstall()
            self._collective_tracer = None

        if self._observer_tokens:
            _debug_mode_observer.reset(self._observer_tokens.pop())

        # The stack is expected to be empty here. Unwinding it matters when a
        # traced call raised: the matching exit callback never ran, so the
        # leftovers would otherwise swallow every record of the next window.
        self._call_stack.clear()

        return False

    def clear(self):
        """Release all collected records and counters.

        Safe to call at any point, including while the context is active.
        The observer/hook installation is left untouched — only the data.
        """
        self._reset_records()

    def _reset_records(self):
        """Drop records, counters and module info."""
        self._call_stack.clear()
        self._root_records.clear()
        self._record_count = 0
        self._dropped_records = 0
        self._comm_counts.clear()
        self._parameter_info.clear()
        self._sharding_info.clear()

    def __repr__(self):
        return f"CommDebugMode(get_total_counts()={self.get_total_counts()})"

    # ------------------------------------------------------------------
    # Observer callbacks (called from _op_dispatch.py — internal)
    # ------------------------------------------------------------------

    def _on_op_dispatch_enter(self, op_name: str, op_call, args, kwargs):  # pylint: disable=W0613
        """Called by OpDispatcher.dispatch() before the op executes."""
        if not self._active:
            return

        record = self._make_record(
            OpCall,
            op_name=op_name,
            input_infos=self._extract_tensor_infos(args),
        )
        if record is None:
            return

        self._attach(record)

    def _on_op_dispatch_exit(self, op_name, result):  # pylint: disable=W0613
        """Called by OpDispatcher.dispatch() after the op executes."""
        if not self._call_stack:
            return

        # Peek before popping: a nested scope that never saw an enter must not
        # pop someone else's frame off the stack.
        if not isinstance(self._call_stack[-1], OpCall):
            return

        record = self._call_stack.pop()
        record.output_infos = self._extract_tensor_infos((result,))

    # Keep old names as aliases for backward compatibility with tests.
    on_op_dispatch_enter = _on_op_dispatch_enter
    on_op_dispatch_exit = _on_op_dispatch_exit

    # ------------------------------------------------------------------
    # Record bookkeeping
    # ------------------------------------------------------------------

    def _make_record(self, record_cls, **fields) -> Optional[DebugCall]:
        """Build a record node, or return None when it must not be retained.

        Returns:
            Optional[DebugCall]: The node, or None when the depth or node budget
            is exhausted. Callers must not push anything onto ``_call_stack``
            in that case.
        """
        depth = len(self._call_stack)
        if depth >= _MAX_DEPTH or self._record_count >= _MAX_RECORDS:
            self._dropped_records += 1
            return None

        self._record_count += 1
        return record_cls(call_depth=depth, **fields)

    def _attach(self, record: DebugCall):
        """Link *record* under the current frame (or the roots) and open it."""
        if self._call_stack:
            self._call_stack[-1].children.append(record)
        else:
            self._root_records.append(record)
        self._call_stack.append(record)

    def get_dropped_record_count(self) -> int:
        """Returns how many records were discarded because the budget ran out."""
        return self._dropped_records

    # ------------------------------------------------------------------
    # Collective tracer callback
    # ------------------------------------------------------------------

    def _on_collective_call(self, method_name: str, args, kwargs, result):  # pylint: disable=W0613
        """Invoked by CollectiveTracer after a collective op completes."""
        # The counter is authoritative and stays exact even once records are
        # being dropped, so it is updated before anything else.
        self._comm_counts[method_name] += 1

        if not self._active:
            return

        input_shape = None
        input_dtype = ""
        if args and hasattr(args[0], "shape"):
            input_shape = tuple(args[0].shape)
            input_dtype = str(args[0].dtype)

        output_shape = None
        if result is not None and hasattr(result, "shape"):
            output_shape = tuple(result.shape)

        group_size = 0
        group_str = None
        group_idx = _COLLECTIVE_GROUP_ARG_INDEX.get(method_name)
        if group_idx is not None and len(args) > group_idx:
            group = args[group_idx]
            if isinstance(group, str):
                group_str = group
            elif hasattr(group, "size"):
                try:
                    group_size = group.size()
                except Exception:  # pylint: disable=W0703
                    pass

        record = self._make_record(
            CollectiveCall,
            collective_type=method_name,
            group_size=group_size,
            group=group_str,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
        )
        if record is None:
            return

        # A collective is a leaf: it is linked but never pushed, so it cannot
        # swallow the enclosing op's exit.
        if self._call_stack:
            self._call_stack[-1].children.append(record)
        else:
            self._root_records.append(record)

    # ------------------------------------------------------------------
    # Module tracker callback
    # ------------------------------------------------------------------

    def _on_module_event(self, module_fqn: str, event_type: str):
        """Invoked by ModuleTracker on forward enter/exit."""
        if not self._active:
            return

        if event_type == "exit":
            # Only unwind a frame this tracer opened.
            if self._call_stack and isinstance(self._call_stack[-1], AnnotateCall):
                self._call_stack.pop()
            return

        record = self._make_record(
            AnnotateCall,
            module_fqn=module_fqn,
            event_type=event_type,
        )
        if record is None:
            return

        self._attach(record)

    # ------------------------------------------------------------------
    # Module info collection
    # ------------------------------------------------------------------

    def _collect_module_info(self):
        """Collect parameter and sharding info from the tracked module.

        Note:
            ``get_parameter_info()`` returns the parameter *storages* (via
            ``param.data``), not copies, so the model stays pinned in memory
            while those entries are held. Call :meth:`clear` to release them.
        """
        from hyper_parallel.core.dtensor.dtensor import (  # pylint: disable=C0415
            DTensor, _distribute_module_named_modules, _distribute_module_named_parameters,
        )

        if self._module is None:
            return

        for fqn, mod in _distribute_module_named_modules(self._module):
            name = fqn or "(root)"
            params = {}
            for param_name, param in _distribute_module_named_parameters(mod):
                params[param_name] = param.data
                if isinstance(param, DTensor):
                    key = f"{name}.{param_name}" if fqn else param_name
                    self._sharding_info[key] = param.placements
            if params:
                self._parameter_info[name] = params

    # ------------------------------------------------------------------
    # Tensor info extraction
    # ------------------------------------------------------------------

    def _extract_tensor_infos(self, args) -> List[TensorInfo]:
        """Extract TensorInfo from args, handling DTensor and plain Tensor."""
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

        infos = []
        for arg in args:
            if isinstance(arg, DTensor):
                placements = tuple(repr(p) for p in arg.placements) if hasattr(arg, "placements") else None
                mesh_shape = None
                if hasattr(arg, "device_mesh") and arg.device_mesh is not None:
                    mesh_shape = tuple(arg.device_mesh.shape) if hasattr(arg.device_mesh, "shape") else None
                infos.append(TensorInfo(
                    shape=tuple(arg.shape),
                    dtype=str(arg.dtype),
                    is_dtensor=True,
                    placements=placements,
                    mesh_shape=mesh_shape,
                ))
            elif isinstance(arg, Tensor):
                infos.append(TensorInfo(
                    shape=tuple(arg.shape),
                    dtype=str(arg.dtype),
                ))
            elif isinstance(arg, (tuple, list)):
                infos.extend(self._extract_tensor_infos(arg))
        return infos

    # ------------------------------------------------------------------
    # Public API (aligned with torch.distributed.tensor.debug.CommDebugMode)
    # ------------------------------------------------------------------

    def get_comm_counts(self) -> Dict[str, int]:
        """Returns the communication counts as a dictionary.

        Returns:
            Dict[str, int]: Mapping from collective type name to invocation count.
        """
        return dict(self._comm_counts)

    def get_total_counts(self) -> int:
        """Returns the total number of collective calls recorded."""
        return sum(self._comm_counts.values())

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Returns parameter info collected from the tracked module.

        Returns:
            Dict mapping module FQN to a dict of ``{param_name: param_data}``.
            The values are the live parameter storages, not copies. Only
            available when a *module* was passed to the constructor.
        """
        return self._parameter_info

    def get_sharding_info(self) -> Dict[str, Any]:
        """Returns sharding info for DTensor parameters.

        Returns:
            Dict mapping ``module_fqn.param_name`` to its placements.
            Only available when a *module* was passed to the constructor.
        """
        return self._sharding_info

    def generate_comm_debug_tracing_table(self, noise_level: Optional[int] = None) -> str:
        """Generate a formatted tracing table.

        Args:
            noise_level: 0 = collectives only, 1 = ops + collectives,
                2 = full detail. Defaults to 1.

        Returns:
            str: Formatted multi-line table string.
        """
        if noise_level is None:
            noise_level = 1

        if self._dropped_records:
            logger.warning(
                "%d record(s) were dropped because the record budget was exhausted; "
                "the table is incomplete. Only trace a bounded number of steps.",
                self._dropped_records,
            )

        if noise_level >= 2 and self._module is None:
            logger.warning(
                "noise_level=2 shows module boundary annotations, but no module was passed "
                "to CommDebugMode(). Pass CommDebugMode(module=model) to enable module tracking."
            )

        lines = []
        for record in self._root_records:
            self._collect_table_lines(record, lines, noise_level, indent=0)

        if not lines:
            return "(no operations recorded)"

        header = f"{'Type':<20} {'Detail':<60}"
        separator = "-" * 80
        table_lines = [header, separator] + lines
        return "\n".join(table_lines)

    def log_comm_debug_tracing_table_to_file(
        self, file_name: str = "comm_mode_log.txt", noise_level: Optional[int] = None
    ) -> None:
        """Write tracing table to a file (ANSI escape codes stripped).

        Args:
            file_name: Output file path.
            noise_level: Verbosity level (see ``generate_comm_debug_tracing_table``).
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        table = ansi_escape.sub("", self.generate_comm_debug_tracing_table(noise_level))
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(table)

    def generate_json_dump(
        self, file_name: str = "comm_mode_log.json", noise_level: Optional[int] = None
    ) -> None:
        """Export tracing data as a JSON file.

        Args:
            file_name: Output file path.
            noise_level: Verbosity level. Defaults to 1.
        """
        if noise_level is None:
            noise_level = 1

        def _record_to_dict(record: DebugCall) -> Optional[dict]:
            entry: dict = {}
            if isinstance(record, CollectiveCall):
                entry["type"] = "collective"
                entry["collective_type"] = record.collective_type
                if record.group is not None:
                    entry["group"] = record.group
                else:
                    entry["group_size"] = record.group_size
                entry["input_shape"] = list(record.input_shape) if record.input_shape else None
                entry["output_shape"] = list(record.output_shape) if record.output_shape else None
            elif isinstance(record, OpCall):
                if noise_level < 1:
                    return None
                entry["type"] = "op"
                entry["op_name"] = record.op_name
                entry["inputs"] = [
                    {"shape": list(t.shape), "dtype": t.dtype, "is_dtensor": t.is_dtensor,
                     "placements": list(t.placements) if t.placements else None}
                    for t in record.input_infos
                ]
                entry["outputs"] = [
                    {"shape": list(t.shape), "dtype": t.dtype, "is_dtensor": t.is_dtensor}
                    for t in record.output_infos
                ]
            elif isinstance(record, AnnotateCall):
                if noise_level < 2:
                    return None
                entry["type"] = "module"
                entry["module_fqn"] = record.module_fqn
                entry["event_type"] = record.event_type
            else:
                return None

            children = []
            for child in record.children:
                child_dict = _record_to_dict(child)
                if child_dict is not None:
                    children.append(child_dict)
            if children:
                entry["children"] = children

            return entry

        data = {
            "comm_counts": dict(self._comm_counts),
            "total_counts": self.get_total_counts(),
            "records": [],
        }

        if self._dropped_records:
            data["dropped_records"] = self._dropped_records

        if self._sharding_info:
            data["sharding_info"] = {k: str(v) for k, v in self._sharding_info.items()}

        for record in self._root_records:
            entry = _record_to_dict(record)
            if entry is not None:
                data["records"].append(entry)

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Keep old name as alias for backward compatibility.
    generate_tracing_table = generate_comm_debug_tracing_table

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_table_lines(self, record: DebugCall, lines: List[str],
                             noise_level: int, indent: int):
        """Recursively append one formatted table line per record in the subtree.

        Collectives are always emitted; ``OpCall`` and ``AnnotateCall`` lines are
        gated on *noise_level* so that lowering verbosity drops them while still
        descending into their children.

        Args:
            record: Subtree root to render.
            lines: Accumulator the formatted lines are appended to; passed through
                the recursion rather than returned, so the caller can seed it.
            noise_level: 0 = collectives only, 1 = ops + collectives,
                2 = module annotations as well.
            indent: Current nesting level; each level adds two leading spaces.
        """
        prefix = "  " * indent
        if isinstance(record, CollectiveCall):
            lines.append(f"{prefix}{'Collective':<20} {record.render_self()}")
        elif isinstance(record, OpCall) and noise_level >= 1:
            lines.append(f"{prefix}{'Op':<20} {record.render_self()}")
        elif isinstance(record, AnnotateCall) and noise_level >= 2:
            lines.append(f"{prefix}{'Module':<20} {record.render_self()}")

        for child in record.children:
            self._collect_table_lines(child, lines, noise_level, indent + 1)

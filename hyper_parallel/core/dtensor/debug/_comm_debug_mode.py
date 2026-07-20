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
"""
import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from hyper_parallel.core.dtensor.debug._call_records import (
    AnnotateCall,
    CollectiveCall,
    DebugCall,
    OpCall,
    TensorInfo,
)
from hyper_parallel.core.dtensor.debug._collective_tracer import CollectiveTracer
from hyper_parallel.core.dtensor.debug._module_tracker import ModuleTracker
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)
platform = get_platform()
Tensor = platform.Tensor

# Argument index of the process group for each traced collective method.
# Derived from the platform method signatures:
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


class CommDebugMode:
    """Context manager that records DTensor operator dispatches and collective
    communication operations, producing a hierarchical call tree.

    Usage::

        with CommDebugMode() as mode:
            output = model(input_dtensor)
        print(mode.generate_comm_debug_tracing_table())
        print(mode.get_comm_counts())

    Args:
        module: Optional ``nn.Module`` to track forward enter/exit events.
    """

    def __init__(self, module=None):
        self._module = module

        # ---- tracing state ----
        self._call_stack: List[DebugCall] = []
        self._root_records: List[DebugCall] = []
        self._comm_counts: Dict[str, int] = defaultdict(int)
        # ---- module-level info (populated when module is provided) ----
        self._parameter_info: Dict[str, Dict[str, Any]] = {}
        self._sharding_info: Dict[str, Any] = {}

        # ---- internal handles ----
        self._collective_tracer: Optional[CollectiveTracer] = None
        self._module_tracker: Optional[ModuleTracker] = None
        self._observer_token = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import _debug_mode_observer

        self._comm_counts.clear()
        self._root_records.clear()
        self._call_stack.clear()
        self._parameter_info.clear()
        self._sharding_info.clear()

        self._observer_token = _debug_mode_observer.set(self)

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

        if self._module_tracker is not None:
            self._module_tracker.uninstall()
            self._module_tracker = None

        if self._collective_tracer is not None:
            self._collective_tracer.uninstall()
            self._collective_tracer = None

        if self._observer_token is not None:
            _debug_mode_observer.reset(self._observer_token)
            self._observer_token = None

    def __repr__(self):
        return f"CommDebugMode(get_total_counts()={self.get_total_counts()})"

    # ------------------------------------------------------------------
    # Observer callbacks (called from _op_dispatch.py — internal)
    # ------------------------------------------------------------------

    def _on_op_dispatch_enter(self, op_name: str, op_call, args, kwargs):  # pylint: disable=W0613
        """Called by OpDispatcher.dispatch() before the op executes."""
        depth = len(self._call_stack)
        record = OpCall(
            call_depth=depth,
            op_name=op_name,
            input_infos=self._extract_tensor_infos(args),
        )

        if self._call_stack:
            self._call_stack[-1].children.append(record)
        else:
            self._root_records.append(record)

        self._call_stack.append(record)

    def _on_op_dispatch_exit(self, op_name, result):  # pylint: disable=W0613
        """Called by OpDispatcher.dispatch() after the op executes."""
        if not self._call_stack:
            return

        record = self._call_stack.pop()
        if isinstance(record, OpCall):
            record.output_infos = self._extract_tensor_infos((result,))

    # Keep old names as aliases for backward compatibility with tests.
    on_op_dispatch_enter = _on_op_dispatch_enter
    on_op_dispatch_exit = _on_op_dispatch_exit

    # ------------------------------------------------------------------
    # Collective tracer callback
    # ------------------------------------------------------------------

    def _on_collective_call(self, method_name: str, args, kwargs, result):  # pylint: disable=W0613
        """Invoked by CollectiveTracer after a collective op completes."""
        depth = len(self._call_stack)

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

        record = CollectiveCall(
            call_depth=depth,
            collective_type=method_name,
            group_size=group_size,
            group=group_str,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=input_dtype,
        )

        if self._call_stack:
            self._call_stack[-1].children.append(record)
        else:
            self._root_records.append(record)

        self._comm_counts[method_name] += 1

    # ------------------------------------------------------------------
    # Module tracker callback
    # ------------------------------------------------------------------

    def _on_module_event(self, module_fqn: str, event_type: str):
        """Invoked by ModuleTracker on forward enter/exit."""
        depth = len(self._call_stack)
        record = AnnotateCall(
            call_depth=depth,
            module_fqn=module_fqn,
            event_type=event_type,
        )

        if event_type == "enter":
            if self._call_stack:
                self._call_stack[-1].children.append(record)
            else:
                self._root_records.append(record)
            self._call_stack.append(record)
        else:  # "exit"
            if self._call_stack:
                self._call_stack.pop()

    # ------------------------------------------------------------------
    # Module info collection
    # ------------------------------------------------------------------

    def _collect_module_info(self):
        """Collect parameter and sharding info from the tracked module."""
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
            Only available when a *module* was passed to the constructor.
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
        """Recursively append formatted lines for *record* and its children."""
        prefix = "  " * indent
        if isinstance(record, CollectiveCall):
            lines.append(f"{prefix}{'Collective':<20} {record._render_self()}")
        elif isinstance(record, OpCall) and noise_level >= 1:
            lines.append(f"{prefix}{'Op':<20} {record._render_self()}")
        elif isinstance(record, AnnotateCall) and noise_level >= 2:
            lines.append(f"{prefix}{'Module':<20} {record._render_self()}")

        for child in record.children:
            self._collect_table_lines(child, lines, noise_level, indent + 1)

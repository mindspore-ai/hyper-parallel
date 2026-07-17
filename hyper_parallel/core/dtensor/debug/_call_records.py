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
"""Structured call record classes for CommDebugMode tracing."""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TensorInfo:
    """Metadata snapshot of a tensor at trace time."""
    shape: Tuple[int, ...]
    dtype: str
    is_dtensor: bool = False
    placements: Optional[Tuple] = None
    mesh_shape: Optional[Tuple[int, ...]] = None


@dataclass
class DebugCall:
    """Base class for all traced call records.

    Maintains a tree structure via ``children`` so that collective calls
    appear nested under the operator that triggered them.
    """
    call_depth: int = 0
    timestamp: float = field(default_factory=time.time)
    children: List["DebugCall"] = field(default_factory=list)

    def _render_self(self) -> str:
        return f"[DebugCall depth={self.call_depth}]"

    def render(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}{self._render_self()}"]
        for child in self.children:
            lines.append(child.render(indent + 1))
        return "\n".join(lines)


@dataclass
class OpCall(DebugCall):
    """Record of a DTensor operator dispatch."""
    op_name: str = ""
    input_infos: List[TensorInfo] = field(default_factory=list)
    output_infos: List[TensorInfo] = field(default_factory=list)

    def _render_self(self) -> str:
        inputs_str = ", ".join(
            f"{'DTensor' if t.is_dtensor else 'Tensor'}{list(t.shape)}"
            for t in self.input_infos
        )
        outputs_str = ", ".join(
            f"{'DTensor' if t.is_dtensor else 'Tensor'}{list(t.shape)}"
            for t in self.output_infos
        )
        return f"Op({self.op_name}) inputs=[{inputs_str}] outputs=[{outputs_str}]"


@dataclass
class CollectiveCall(DebugCall):
    """Record of a collective communication operation."""
    collective_type: str = ""
    group_size: int = 0
    group: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_dtype: str = ""

    def _render_self(self) -> str:
        group_str = f"group={self.group}" if self.group is not None else f"group_size={self.group_size}"
        return (
            f"Collective({self.collective_type}) "
            f"{group_str} "
            f"input_shape={self.input_shape} "
            f"output_shape={self.output_shape}"
        )


@dataclass
class RedistributeCall(DebugCall):
    """Record of a DTensor redistribute operation."""
    src_placements: Optional[Tuple] = None
    dst_placements: Optional[Tuple] = None
    tensor_shape: Optional[Tuple[int, ...]] = None

    def _render_self(self) -> str:
        return (
            f"Redistribute(shape={self.tensor_shape}) "
            f"{self.src_placements} -> {self.dst_placements}"
        )


@dataclass
class AnnotateCall(DebugCall):
    """Record of a module boundary event (enter/exit)."""
    annotation: str = ""
    module_fqn: str = ""
    event_type: str = ""  # "enter" or "exit"

    def _render_self(self) -> str:
        return f"Module({self.module_fqn}) [{self.event_type}]"

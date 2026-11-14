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
"""Shared TP input/output execution for class-owned module boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed._builder.tp_collective_lowering import create_tp_collective_lowerer


@dataclass(frozen=True)
class TensorParallelExecution:
    """Existing TP collectives compiled from the final module-owned I/O contract.

    This resource does not own parameters, CP groups, or attention math. Input
    and output stages are usable by local and CP implementations alike.
    """

    input_operation: Any = None
    output_operation: Any = None

    @classmethod
    def from_binding(cls, binding: Any, *, input_layout: Mapping, output_layout: Mapping) -> TensorParallelExecution:
        """Compile internal transitions without rebuilding the provider's external layouts.

        Args:
            binding: Shared resources and the final validated ModuleShardingSpec.
            input_layout: Layout required by the module's local calculation.
            output_layout: Layout produced by that calculation, before TP reduction.
        """
        mesh = binding.mesh_binding.mesh_context.device_mesh
        if "tp" not in tuple(mesh.mesh_dim_names or ()) or mesh["tp"].size() == 1:
            return cls()
        lowerer = create_tp_collective_lowerer(mesh["tp"], ("tp",))
        if lowerer is None:
            raise ValueError("Explicit TP stages require group ranks in tensor-mesh order")
        spec = binding.sharding
        source = spec.in_src["hidden_states"]
        destination = spec.out_dst["output"]
        for external, internal in ((source, input_layout), (destination, output_layout)):
            for axis in set(external) | set(internal):
                if axis != "tp" and external.get(axis, Replicate()) != internal.get(axis, Replicate()):
                    raise ValueError("TP stages cannot change ownership on a non-TP axis")
        transitions = ((source, input_layout), (output_layout, destination))
        operations = []
        for begin, end in transitions:
            before, after = (begin.get("tp", Replicate()),), (end.get("tp", Replicate()),)
            operation = lowerer(before, after)
            if before != after and operation is None:
                raise ValueError("Unsupported explicit TP stage transition")
            operations.append(operation)
        return cls(*operations)

    def prepare_input(self, tensor: Tensor) -> Tensor:
        """Gather SP tokens when required by the final input contract."""
        return tensor if self.input_operation is None else self.input_operation.execute(tensor)

    def finalize_output(self, tensor: Tensor) -> Tensor:
        """Reduce TP partial outputs using the existing AR/RS executor."""
        return tensor if self.output_operation is None else self.output_operation.execute(tensor)

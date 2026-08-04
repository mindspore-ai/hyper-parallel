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
"""Parallel styles for MC2 fused linear layers (PyTorch / Ascend)."""
from __future__ import annotations

from typing import Optional

from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement, Shard
from hyper_parallel.core.tensor_parallel.mc2 import MC2Linear
from hyper_parallel.core.tensor_parallel.style import ColwiseParallel, RowwiseParallel
from hyper_parallel.platform import get_platform

platform = get_platform()

__all__ = ["MC2ColwiseParallel", "MC2RowwiseParallel"]


def _replace_with_mc2_linear(
    module: nn.Module,
    mode: str,
    device_mesh: DeviceMesh,
    sequence_dim: int,
) -> MC2Linear:
    """Replace ``nn.Linear`` with configured ``MC2Linear`` in place."""
    if not platform.is_linear_module(module):
        raise NotImplementedError(
            f"MC2 parallel style only supports Linear modules, but got {type(module).__name__}."
        )
    module = MC2Linear.from_linear(module)
    module.configure_mc2(
        mode,
        device_mesh.get_group(),
        device_mesh.size(),
        sequence_dim=sequence_dim,
    )
    return module


class MC2ColwiseParallel(ColwiseParallel):
    """Column parallelism using fused all-gather and matmul (MC2).

    Requires a sequence-sharded input layout so AllGather can be folded into the
    matmul. Unlike :class:`ColwiseParallel`, this style does **not** redistribute
    the input to ``Replicate()`` before the Linear.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: Optional[bool] = None,
    ) -> None:
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )
        if not isinstance(self.input_layouts[0], Shard):
            raise ValueError("MC2ColwiseParallel requires a sharded input layout.")
        # Keep sequence sharding; fused kernel performs the all-gather.
        self.desired_input_layouts = self.input_layouts
        self._sequence_dim = self.input_layouts[0].dim

    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Replace Linear with MC2Linear then apply column-wise sharding hooks."""
        module = _replace_with_mc2_linear(
            module, "all_gather", device_mesh, self._sequence_dim
        )
        return super().apply(module, device_mesh)


class MC2RowwiseParallel(RowwiseParallel):
    """Row parallelism using fused matmul and reduce-scatter (MC2).

    Requires a sequence-sharded output layout so ReduceScatter replaces AllReduce
    and restores sequence parallelism after the row-parallel Linear.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        reduce_dtype=None,
        use_local_output: bool = True,
    ) -> None:
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            reduce_dtype=reduce_dtype,
            use_local_output=use_local_output,
        )
        if not isinstance(self.output_layouts[0], Shard):
            raise ValueError("MC2RowwiseParallel requires a sharded output layout.")
        self._sequence_dim = self.output_layouts[0].dim

    def apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Replace Linear with MC2Linear then apply row-wise sharding hooks."""
        module = _replace_with_mc2_linear(
            module, "reduce_scatter", device_mesh, self._sequence_dim
        )
        return super().apply(module, device_mesh)

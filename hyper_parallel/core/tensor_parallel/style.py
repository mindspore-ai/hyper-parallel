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
"""Parallel styles for declarative tensor-parallel module sharding.

Provides :class:`ParallelStyle` (ABC) and concrete implementations
:class:`ColwiseParallel` / :class:`RowwiseParallel` aligned with
``torch.distributed.tensor.parallel.style``.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import (
    DTensor,
    distribute_module,
    distribute_tensor,
    _distribute_module_iter_params,
    _distribute_module_new_parameter,
    _distribute_module_param_source,
    _distribute_module_set_param,
)
from hyper_parallel.core.dtensor.placement_types import Placement, Replicate, Shard
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module

__all__ = [
    "ParallelStyle",
    "ColwiseParallel",
    "RowwiseParallel",
]


class ParallelStyle(ABC):
    """Abstract base class for parallel styles applied to nn.Module submodules.

    Subclasses implement ``apply`` to wrap a module with the desired
    parallel communication behaviour (e.g. all-to-all for context parallel).

    ``src_data_rank`` mirrors PyTorch's tensor-parallel contract: it can be set by
    :func:`parallelize_module` for styles that scatter/broadcast global tensors.
    HyperParallel styles may ignore it until they integrate ``distribute_tensor``.
    """

    src_data_rank: Optional[int] = 0

    @abstractmethod
    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply this parallel style to *module* in-place and return it.

        Args:
            module: The submodule to be parallelised.
            device_mesh: The device mesh describing the cluster topology.

        Returns:
            The (possibly wrapped) module with parallelism applied.
        """


class ColwiseParallel(ParallelStyle):
    """Partition a compatible module in a column-wise fashion.

    Currently supports Linear and Embedding modules (framework-agnostic via
    ``platform.is_linear_module`` / ``platform.is_embedding_module``).
    Compose with :class:`RowwiseParallel` to shard MLP or Attention blocks.

    Keyword Args:
        input_layouts (Placement, optional):
            DTensor layout for the module input. Used to annotate the input
            tensor as a DTensor. Defaults to ``Replicate()``.
        output_layouts (Placement, optional):
            Desired DTensor layout of the module output. Defaults to
            ``Shard(-1)`` (sharded on the last dimension).
        use_local_output (bool, optional):
            If ``True`` (default), convert the output DTensor back to a local
            tensor via ``to_local()``.

    Returns:
        A :class:`ParallelStyle` that applies column-wise sharding.

    Example::

        >>> from hyper_parallel import parallelize_module, ColwiseParallel, init_device_mesh
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("tp",))
        >>> parallelize_module(m, tp_mesh, {"linear1": ColwiseParallel()})
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        self.input_layouts: Tuple[Placement, ...] = (input_layouts or Replicate(),)
        self.output_layouts: Tuple[Placement, ...] = (output_layouts or Shard(-1),)
        self.desired_input_layouts: Tuple[Placement, ...] = (Replicate(),)
        self.use_local_output = use_local_output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_layouts={self.input_layouts}, "
            f"output_layouts={self.output_layouts}, "
            f"use_local_output={self.use_local_output})"
        )

    @staticmethod
    def _prepare_input_fn(
        input_layouts: Tuple[Placement, ...],
        desired_input_layouts: Tuple[Placement, ...],
        inputs: Any,
        device_mesh: DeviceMesh,
    ) -> Any:
        """Annotate or redistribute the first positional input."""
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts,
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                device_mesh, desired_input_layouts,
            )
        return input_tensor

    def _partition_linear_fn(self, module: Any, device_mesh: DeviceMesh) -> None:
        """Shard Linear weight/bias along ``Shard(0)`` (column-wise)."""
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            dt = distribute_tensor(src, device_mesh, [Shard(0)])
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)

    def _partition_embedding_fn(self, module: Any, device_mesh: DeviceMesh) -> None:
        """Shard Embedding weight along ``Shard(1)`` (column-wise)."""
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            dt = distribute_tensor(src, device_mesh, [Shard(1)])
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)

    @staticmethod
    def _prepare_output_fn(
        output_layouts: Tuple[Placement, ...],
        use_local_output: bool,
        outputs: Any,
        device_mesh: DeviceMesh,
    ) -> Any:
        """Redistribute output to desired layout and optionally convert to local."""
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(device_mesh, output_layouts)
        if use_local_output:
            return outputs.to_local()
        return outputs

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply column-wise parallelism to *module*.

        Args:
            module: A Linear or Embedding module to be sharded.
            device_mesh: 1-D device mesh for tensor parallelism.

        Returns:
            The module with distributed parameters and I/O hooks attached.

        Raises:
            NotImplementedError: If *module* is not a supported type.
        """
        if platform.is_linear_module(module):

            def partition_fn(submodule_path, submodule, device_mesh):
                self._partition_linear_fn(submodule, device_mesh)

        elif platform.is_embedding_module(module):

            def partition_fn(submodule_path, submodule, device_mesh):
                self._partition_embedding_fn(submodule, device_mesh)

        else:
            raise NotImplementedError(
                "ColwiseParallel currently only supports Linear and Embedding modules!"
            )

        def input_fn(forward_module, forward_inputs, device_mesh):
            return self._prepare_input_fn(
                self.input_layouts,
                self.desired_input_layouts,
                forward_inputs,
                device_mesh,
            )

        def output_fn(forward_module, forward_outputs, device_mesh):
            return self._prepare_output_fn(
                self.output_layouts,
                self.use_local_output,
                forward_outputs,
                device_mesh,
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            input_fn,
            output_fn,
        )


class RowwiseParallel(ParallelStyle):
    """Partition a compatible module in a row-wise fashion.

    Currently supports Linear and Embedding modules (framework-agnostic via
    ``platform.is_linear_module`` / ``platform.is_embedding_module``).
    Compose with :class:`ColwiseParallel` to shard MLP or Attention blocks.

    Keyword Args:
        input_layouts (Placement, optional):
            DTensor layout for the module input. Defaults to ``Shard(-1)``
            (sharded on the last dimension).
        output_layouts (Placement, optional):
            Desired DTensor layout of the module output. Defaults to
            ``Replicate()`` (all-reduce / reduce-scatter from partial).
        use_local_output (bool, optional):
            If ``True`` (default), convert the output DTensor back to a local
            tensor via ``to_local()``.

    Returns:
        A :class:`ParallelStyle` that applies row-wise sharding.

    Example::
        >>> from hyper_parallel import parallelize_module, RowwiseParallel, init_device_mesh
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("tp",))
        >>> parallelize_module(m, tp_mesh, {"linear2": RowwiseParallel()})
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        self.input_layouts: Tuple[Placement, ...] = (input_layouts or Shard(-1),)
        self.output_layouts: Tuple[Placement, ...] = (output_layouts or Replicate(),)
        self.desired_input_layouts: Tuple[Placement, ...] = (Shard(-1),)
        self.use_local_output = use_local_output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_layouts={self.input_layouts}, "
            f"output_layouts={self.output_layouts}, "
            f"use_local_output={self.use_local_output})"
        )

    @staticmethod
    def _prepare_input_fn(
        input_layouts: Tuple[Placement, ...],
        desired_input_layouts: Tuple[Placement, ...],
        inputs: Any,
        device_mesh: DeviceMesh,
    ) -> Any:
        """Annotate or redistribute the first positional input."""
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts,
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                device_mesh, desired_input_layouts,
            )
        return input_tensor

    def _partition_linear_fn(self, module: Any, device_mesh: DeviceMesh) -> None:
        """Shard Linear weight along ``Shard(1)`` (row-wise); bias to ``Replicate()``."""
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            placement = [Shard(1)] if key == "weight" else [Replicate()]
            dt = distribute_tensor(src, device_mesh, placement)
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)

    def _partition_embedding_fn(self, module: Any, device_mesh: DeviceMesh) -> None:
        """Shard Embedding weight along ``Shard(0)`` (row-wise)."""
        for key, param in _distribute_module_iter_params(module):
            if param is None:
                continue
            src = _distribute_module_param_source(param)
            requires_grad = bool(getattr(param, "requires_grad", True))
            dt = distribute_tensor(src, device_mesh, [Shard(0)])
            new_param = _distribute_module_new_parameter(key, dt, requires_grad)
            _distribute_module_set_param(module, key, new_param)

    @staticmethod
    def _prepare_output_fn(
        output_layouts: Tuple[Placement, ...],
        use_local_output: bool,
        outputs: Any,
        device_mesh: DeviceMesh,
    ) -> Any:
        """Redistribute partial output and optionally convert to local."""
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(device_mesh, output_layouts)
        if use_local_output:
            return outputs.to_local()
        return outputs

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply row-wise parallelism to *module*.

        Args:
            module: A Linear or Embedding module to be sharded.
            device_mesh: 1-D device mesh for tensor parallelism.

        Returns:
            The module with distributed parameters and I/O hooks attached.

        Raises:
            NotImplementedError: If *module* is not a supported type.
        """
        if platform.is_linear_module(module):

            def partition_fn(submodule_path, submodule, device_mesh):
                self._partition_linear_fn(submodule, device_mesh)

            self.desired_input_layouts = (Shard(-1),)
        elif platform.is_embedding_module(module):

            def partition_fn(submodule_path, submodule, device_mesh):
                self._partition_embedding_fn(submodule, device_mesh)

            self.desired_input_layouts = (Replicate(),)
        else:
            raise NotImplementedError(
                "RowwiseParallel currently only supports Linear and Embedding modules!"
            )

        def input_fn(forward_module, forward_inputs, device_mesh):
            return self._prepare_input_fn(
                self.input_layouts,
                self.desired_input_layouts,
                forward_inputs,
                device_mesh,
            )

        def output_fn(forward_module, forward_outputs, device_mesh):
            return self._prepare_output_fn(
                self.output_layouts,
                self.use_local_output,
                forward_outputs,
                device_mesh,
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            input_fn,
            output_fn,
        )

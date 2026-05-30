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
:class:`ColwiseParallel`, :class:`RowwiseParallel`, :class:`SequenceParallel`,
:class:`PrepareModuleInput`, :class:`PrepareModuleInputOutput`, and
:class:`PrepareModuleOutput` aligned with ``torch.distributed.tensor.parallel.style``.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

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
from hyper_parallel.core.dtensor.placement_types import Partial, Placement, Replicate, Shard
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module

__all__ = [
    "ParallelStyle",
    "ColwiseParallel",
    "RowwiseParallel",
    "SequenceParallel",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
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
        use_local_output: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self._input_layouts_arg = input_layouts
        self._output_layouts_arg = output_layouts
        self._use_local_output_arg = use_local_output

        self.input_layouts: Tuple[Placement, ...] = (input_layouts or Replicate(),)
        self.output_layouts: Tuple[Placement, ...] = (output_layouts or Shard(-1),)
        self.desired_input_layouts: Tuple[Placement, ...] = (Replicate(),)
        self.use_local_output = use_local_output if use_local_output is not None else True

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
        # MindSpore requires tuple return from pre-hook
        return (input_tensor,)

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
        # MindSpore requires tuple return from pre-hook
        return (input_tensor,)

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
        module: Optional[Module] = None,
    ) -> Any:
        """Redistribute partial output and optionally convert to local."""
        if not isinstance(outputs, DTensor):
            # ``nn.Embedding.forward`` returns a plain tensor even when weight is sharded;
            # treat the local values as partial along the TP mesh (sum) before redistributing.
            if module is not None and platform.is_embedding_module(module):
                outputs = DTensor.from_local(outputs, device_mesh, [Partial("sum")])
            else:
                raise TypeError(
                    "RowwiseParallel expects a DTensor from Linear outputs; "
                    f"got {type(outputs)}. If this is an unsupported module, extend I/O hooks."
                )
        if tuple(outputs.placements) != tuple(output_layouts):
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
                forward_module,
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            input_fn,
            output_fn,
        )


class SequenceParallel(ParallelStyle):
    """Replicate module parameters and run forward with the sequence axis sharded.

    Matches ``torch.distributed.tensor.parallel.SequenceParallel``: activations are
    sharded on the sequence dimension while weights stay fully replicated. Typical
    targets are normalization and dropout layers used after row-wise / scatter
    projections in tensor-parallel transformers (`Reducing Activation Recomputation
    in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__).

    If the first positional input is a plain tensor, it is treated as the local
    shard along ``sequence_dim`` and wrapped as a :class:`DTensor`. If it is already
    a :class:`DTensor` but not sharded on that dimension, it is redistributed.

    Keyword Args:
        sequence_dim (int, optional):
            Tensor dimension index for the sequence axis (e.g. ``1`` for ``(B, S, H)``).
            Default: ``1``.
        use_local_output (bool, optional):
            If ``True``, return a local tensor via ``to_local()``; otherwise keep a
            :class:`DTensor`. Default: ``False`` (PyTorch default).

    Note:
        Like PyTorch, this assumes sensible defaults for norm weights (e.g. ones).
        Custom initializations should be broadcast so every rank agrees before or
        after parallelization.

    Example::

        >>> from hyper_parallel import parallelize_module, SequenceParallel, init_device_mesh
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("tp",))
        >>> parallelize_module(m, tp_mesh, {"norm": SequenceParallel()})
    """

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False) -> None:
        super().__init__()
        self.sequence_sharding: Tuple[Placement, ...] = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    def __repr__(self) -> str:
        dim = self.sequence_sharding[0].dim
        return (
            f"{self.__class__.__name__}("
            f"sequence_dim={dim}, "
            f"use_local_output={self.use_local_output})"
        )

    @staticmethod
    def _prepare_input_fn(
        sequence_sharding: Tuple[Placement, ...],
        mod: Module,
        inputs: Any,
        device_mesh: DeviceMesh,
    ) -> Any:
        """Ensure the first input is a :class:`DTensor` sharded on the sequence dim."""
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            if tuple(input_tensor.placements) != tuple(sequence_sharding):
                input_tensor = input_tensor.redistribute(device_mesh, sequence_sharding)
        elif platform.is_tensor(input_tensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, sequence_sharding)
        else:
            raise ValueError(
                f"expecting input of {mod} to be a tensor or DTensor, but got {type(input_tensor)}"
            )
        return (input_tensor,)

    @staticmethod
    def _prepare_output_fn(use_local_output: bool, outputs: Any) -> Any:
        if use_local_output:
            return outputs.to_local()
        return outputs

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply sequence-parallel hooks and replicate parameters via ``distribute_module``.

        Args:
            module: Submodule to parallelize (for example ``LayerNorm`` or ``Dropout``).
            device_mesh: One-dimensional tensor-parallel device mesh.

        Returns:
            The same ``module`` instance with forward hooks attached and parameters
            converted to replicated DTensors where applicable.
        """

        def partition_fn(_submodule_path, _submodule, _mesh):
            return None

        def input_fn(forward_module, forward_inputs, mesh):
            return self._prepare_input_fn(
                self.sequence_sharding,
                forward_module,
                forward_inputs,
                mesh,
            )

        def output_fn(_forward_module, forward_outputs, _mesh):
            return self._prepare_output_fn(self.use_local_output, forward_outputs)

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            input_fn,
            output_fn,
        )


class PrepareModuleInput(ParallelStyle):
    """Prepare module forward *args* (and optional *kwargs*) as :class:`DTensor` layouts.

    At forward time, converts each annotated positional (or keyword) tensor from local
    to :class:`DTensor` using ``input_layouts``, then redistributes to
    ``desired_input_layouts`` when they differ. ``None`` in a layout tuple means
    “leave this input unchanged”.

    Mirrors ``torch.distributed.tensor.parallel.style.PrepareModuleInput``.

    Keyword Args:
        input_layouts: Placements per positional arg, or a single :class:`Placement`
            wrapped as a one-tuple. ``None`` entries skip conversion for that arg.
        desired_input_layouts: Target placements; must match ``input_layouts`` length.
        input_kwarg_layouts: Optional mapping kwarg name → placement for conversion.
        desired_input_kwarg_layouts: Target placements for those kwargs (same keys).
        use_local_output: If ``True``, convert prepared inputs back to local tensors
            before the module runs (PyTorch names this flag ``use_local_output`` on
            :class:`PrepareModuleInput`).
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement], ...]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, Tuple[Optional[Placement], ...]]
        ] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_output: bool = False,
    ) -> None:
        super().__init__()
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        self.use_local_output = use_local_output
        if self.input_layouts is not None:
            if self.desired_input_layouts is None:
                raise AssertionError("desired module inputs should not be None!")
            if len(self.input_layouts) != len(self.desired_input_layouts):
                raise AssertionError(
                    "input_layouts and desired_input_layouts should have same length!"
                )
        self.with_kwargs = input_kwarg_layouts is not None
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        if self.with_kwargs:
            if len(self.input_kwarg_layouts) != len(self.desired_input_kwarg_layouts):
                raise AssertionError(
                    "input_kwarg_layouts and desired_input_kwarg_layouts should have "
                    "same length!"
                )

    def _prepare_input_arg(
        self,
        input_obj: Any,
        mesh: DeviceMesh,
        input_layout: Optional[Placement],
        desired_layout: Optional[Placement],
    ) -> Any:
        """Convert one input to DTensor, redistribute if needed, optionally to_local."""
        if input_layout is not None:
            if isinstance(input_obj, DTensor):
                dt_inp = input_obj
            else:
                if not platform.is_tensor(input_obj):
                    raise AssertionError("expecting input to be a framework tensor!")
                dt_inp = DTensor.from_local(input_obj, mesh, (input_layout,))

            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(mesh, (desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        return input_obj

    def _prepare_input_fn(self, inputs: Any, device_mesh: DeviceMesh) -> Any:
        """Prepare positional ``inputs`` tuple per ``input_layouts`` / ``desired_input_layouts``."""
        if self.input_layouts is None:
            return inputs
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")
        if self.desired_input_layouts is None:
            raise AssertionError("desired module inputs should not be None!")
        prepared_inputs = [
            self._prepare_input_arg(inp, device_mesh, il, dl)
            for inp, il, dl in zip(inputs, self.input_layouts, self.desired_input_layouts)
        ]
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(
        self,
        inputs: Any,
        kwarg_inputs: Dict[str, Any],
        device_mesh: DeviceMesh,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Prepare positional and keyword tensor inputs; returns ``(args, kwargs)`` for the hook."""
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs: Dict[str, Any] = {}
        for kwarg_key in kwarg_inputs:
            kwarg_val = kwarg_inputs[kwarg_key]
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)
            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )
        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        if self.with_kwargs:

            def _pre_hook(_mod, inputs, kwargs):
                return self._prepare_input_kwarg_fn(inputs, kwargs, device_mesh)

            platform.register_forward_pre_hook(
                module, _pre_hook, prepend=False, with_kwargs=True,
            )
        else:

            def _pre_hook(_mod, inputs):
                return self._prepare_input_fn(inputs, device_mesh)

            platform.register_forward_pre_hook(module, _pre_hook, prepend=False)
        return module

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_layouts={self.input_layouts}, "
            f"desired_input_layouts={self.desired_input_layouts}, "
            f"input_kwarg_layouts={self.input_kwarg_layouts}, "
            f"desired_input_kwarg_layouts={self.desired_input_kwarg_layouts}, "
            f"use_local_output={self.use_local_output})"
        )


class PrepareModuleOutput(ParallelStyle):
    """Prepare module forward outputs as :class:`DTensor` and redistribute layouts.

    Registers a forward hook that treats each return value like
    ``torch.distributed.tensor.parallel.style.PrepareModuleOutput``: optional
    ``None`` slots in ``output_layouts`` pass that output through unchanged.

    Keyword Args:
        output_layouts: Current or assumed placement per output tensor.
        desired_output_layouts: Target placements; length must match ``output_layouts``.
        use_local_output: If ``True`` (default), return local shards after redistribution.
    """

    def __init__(
        self,
        *,
        output_layouts: Union[Placement, Tuple[Optional[Placement], ...]],
        desired_output_layouts: Union[Placement, Tuple[Optional[Placement], ...]],
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        self.output_layouts = (
            (output_layouts,) if isinstance(output_layouts, Placement) else output_layouts
        )
        self.desired_output_layouts = (
            (desired_output_layouts,)
            if isinstance(desired_output_layouts, Placement)
            else desired_output_layouts
        )
        self.use_local_output = use_local_output
        if len(self.output_layouts) != len(self.desired_output_layouts):
            raise AssertionError(
                "output_layouts and desired_output_layouts should have same length!"
            )

    def _prepare_out_fn(self, outputs: Any, device_mesh: DeviceMesh) -> Any:
        """Redistribute each output tensor per ``output_layouts`` / ``desired_output_layouts``."""
        prepared_outputs: list = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_layouts):
            raise ValueError("module outputs and output_layouts should have same length!")
        for out, out_layout, desired_out_layout in zip(
            outputs, self.output_layouts, self.desired_output_layouts,
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(out, device_mesh, (out_layout,))
                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(device_mesh, (desired_out_layout,))
                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                prepared_outputs.append(out)
        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        return tuple(prepared_outputs)

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:

        def _hook(_mod, _inputs, outputs):
            return self._prepare_out_fn(outputs, device_mesh)

        module.register_forward_hook(_hook)
        return module

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_layouts={self.output_layouts}, "
            f"desired_output_layouts={self.desired_output_layouts}, "
            f"use_local_output={self.use_local_output})"
        )


class PrepareModuleInputOutput(ParallelStyle):
    """Combine :class:`PrepareModuleInput` and :class:`PrepareModuleOutput` on one module.

    Same keyword arguments as the two styles, with ``use_local_input`` mapping to
    ``PrepareModuleInput(..., use_local_output=use_local_input)`` for PyTorch parity.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement], ...]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, Tuple[Optional[Placement], ...]]
        ] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_input: bool = False,
        output_layouts: Union[Placement, Tuple[Optional[Placement], ...]],
        desired_output_layouts: Union[Placement, Tuple[Optional[Placement], ...]],
        use_local_output: bool = True,
    ) -> None:
        super().__init__()
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        self.prepare_module_input.apply(module, device_mesh)
        self.prepare_module_output.apply(module, device_mesh)
        return module

    def __repr__(self) -> str:
        p_in = self.prepare_module_input
        p_out = self.prepare_module_output
        return (
            f"{self.__class__.__name__}("
            f"input_layouts={p_in.input_layouts}, "
            f"desired_input_layouts={p_in.desired_input_layouts}, "
            f"input_kwarg_layouts={p_in.input_kwarg_layouts}, "
            f"desired_input_kwarg_layouts={p_in.desired_input_kwarg_layouts}, "
            f"use_local_input={p_in.use_local_output}, "
            f"output_layouts={p_out.output_layouts}, "
            f"desired_output_layouts={p_out.desired_output_layouts}, "
            f"use_local_output={p_out.use_local_output})"
        )

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
"""Selective symbolic inputs and runtime guards for make_fx joint graphs."""

from typing import Any

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StatelessSymbolicContext
from torch.utils import _pytree as pytree


DynamicArgDims = dict[str, int | list[int]]


def normalize_dynamic_arg_dims(dynamic_arg_dims: DynamicArgDims | None) -> dict[str, list[int]] | None:
    """Validate and copy the public path-to-dimensions mapping."""
    if dynamic_arg_dims is None:
        return None
    if not isinstance(dynamic_arg_dims, dict):
        raise ValueError("dynamic_arg_dims must be a mapping of input paths to dimensions")
    normalized = {}
    for path, dims in dynamic_arg_dims.items():
        if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
            raise ValueError("dynamic_arg_dims keys must be non-empty dotted input paths")
        dims = [dims] if isinstance(dims, int) else dims
        if not isinstance(dims, (list, tuple)) or any(type(dim) is not int for dim in dims):
            raise ValueError(f"dynamic_arg_dims[{path!r}] must be an int or a list of ints")
        normalized[path] = list(dims)
    return normalized


def _resolve_input(inputs: dict[str, Any], path: str) -> torch.Tensor:
    value = inputs
    try:
        for part in path.split("."):
            if isinstance(value, dict):
                value = value[part]
            elif isinstance(value, (list, tuple)):
                index = int(part)
                if index < 0:
                    raise IndexError(index)
                value = value[index]
            else:
                value = getattr(value, part)
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise ValueError(f"dynamic_arg_dims input path {path!r} does not exist") from exc
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"dynamic_arg_dims input {path!r} must be a tensor, got {type(value).__name__}")
    return value


def build_symbolic_inputs(
    state_flat: list[Any],
    inputs: dict[str, Any],
    dynamic_arg_dims: DynamicArgDims | None,
) -> tuple[FakeTensorMode, tuple[Any, ...]]:
    """Fakeify state statically and selected user dimensions symbolically.

    Args:
        state_flat: Flat model parameters and buffers.
        inputs: Keyword inputs, including registered pytree containers.
        dynamic_arg_dims: Explicit paths/dimensions; None symbolizes all user tensor dimensions.

    Returns:
        Fake mode and flattened state/user inputs sharing its ShapeEnv.
    """
    mapping = normalize_dynamic_arg_dims(dynamic_arg_dims)
    user_flat, _ = pytree.tree_flatten(inputs)
    tensor_ids = {id(value) for value in user_flat if isinstance(value, torch.Tensor)}
    dynamic_dims = {}
    if mapping is None:
        dynamic_dims = {id(value): set(range(value.ndim)) for value in user_flat if isinstance(value, torch.Tensor)}
    else:
        for path, dims in mapping.items():
            tensor = _resolve_input(inputs, path)
            if id(tensor) not in tensor_ids:
                raise ValueError(f"dynamic_arg_dims input {path!r} must be a registered pytree tensor leaf")
            selected = dynamic_dims.setdefault(id(tensor), set())
            for dim in dims:
                if not -tensor.ndim <= dim < tensor.ndim:
                    raise ValueError(f"dynamic_arg_dims[{path!r}] dimension {dim} is invalid for rank {tensor.ndim}")
                selected.add(dim % tensor.ndim)

    # Equal sample sizes do not imply equality across future batches.
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv(duck_shape=False))
    fake_args = [fake_mode.from_tensor(value, static_shapes=True) for value in state_flat]
    for index, value in enumerate(user_flat):
        if not isinstance(value, torch.Tensor):
            fake_args.append(value)
            continue
        if value.layout != torch.strided:
            raise ValueError("Dynamic graph inputs must use torch.strided layout")
        dims = dynamic_dims.get(id(value), set())
        context = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC if dim in dims else DimDynamic.STATIC for dim in range(value.ndim)]
        )
        fake_args.append(fake_mode.from_tensor(value, source=LocalSource(f"input_{index}"), symbolic_context=context))
    return fake_mode, tuple(fake_args)


class InputGuards:
    """Check input metadata, constants and traced shape branches before execution.

    The graph is called directly, so Dynamo will not check its guards for us.
    Only user tensors participate: FSDP may legitimately shard live model state.
    No real input tensors or storage are retained here.
    """

    def __init__(self, inputs: list[Any], fake_inputs: tuple[Any, ...], shape_env: ShapeEnv) -> None:
        self.shape_env = shape_env
        self.fake_tensors = [value for value in fake_inputs if isinstance(value, torch.Tensor)]
        self.leaves = [self._metadata(value) for value in inputs]
        self.aliases = self._aliases(inputs)
        self.expression = None
        self.refresh()

    @staticmethod
    def _metadata(value: Any) -> tuple:
        if isinstance(value, torch.Tensor):
            return (torch.Tensor, value.ndim, value.dtype, value.device, value.layout, value.requires_grad)
        return (type(value), value)

    @staticmethod
    def _aliases(inputs: list[Any]) -> list[int]:
        seen = {}
        return [
            seen.setdefault(id(value), index)
            for index, value in enumerate(inputs) if isinstance(value, torch.Tensor)
        ]

    def refresh(self) -> None:
        """Include constraints introduced by tracing and graph passes."""
        self.expression = self.shape_env.produce_guards_expression(self.fake_tensors, ignore_static=False)

    def validate(self, inputs: list[Any]) -> None:
        """Reject incompatible inputs before any graph or collective is executed."""
        if [self._metadata(value) for value in inputs] != self.leaves:
            raise ValueError("Graph input rank, dtype, device, layout, requires_grad or Python constant changed")
        if self._aliases(inputs) != self.aliases:
            raise ValueError("Graph input tensor aliasing changed since compilation")
        tensors = [value for value in inputs if isinstance(value, torch.Tensor)]
        if self.expression and not self.shape_env.evaluate_guards_expression(self.expression, tensors):
            raise ValueError(
                "Graph input shape/stride violates the traced shape guards. Check unmarked static dimensions, "
                "related input sizes and shape-dependent branches. Compile with representative dynamic sizes > 1; "
                "sizes 0/1 and Python shape branches may specialize the graph."
            )

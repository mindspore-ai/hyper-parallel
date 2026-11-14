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
"""Locally declared tensor layouts for serialization-free model broadcasts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TensorFieldSpec:
    """Describe one tensor field identically on every model-parallel rank.

    Args:
        shape: Fixed dimensions, with None for dimensions sent in a small
            integer header on each batch. An empty tuple denotes a scalar.
        dtype: Required tensor dtype; broadcasts never convert its values.
        device: "device" for the rank-local training device, or "cpu" for
            fields that must remain on Host after broadcast.
    """

    shape: tuple[int | None, ...]
    dtype: torch.dtype
    device: str = "device"

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or any(
                dim is not None and (isinstance(dim, bool) or not isinstance(dim, int) or dim < 0)
                for dim in self.shape
        ):
            raise ValueError("TensorFieldSpec.shape must be a tuple of non-negative integers or None.")
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError("TensorFieldSpec.dtype must be a torch.dtype.")
        if self.device not in ("device", "cpu"):
            raise ValueError("TensorFieldSpec.device must be 'device' or 'cpu'.")


class ModelBatchLayout:
    """Validate and rebuild a fixed local tree without transmitting its structure."""

    def __init__(self, spec: Any) -> None:
        """Freeze the local declaration and enumerate its tensor fields.

        Args:
            spec: Nested tensor field declarations and static scalar values.
        """
        self.spec = copy.deepcopy(spec)
        self.fields = tuple(self._fields(self.spec))
        self.dynamic_dimensions = sum(dim is None for field in self.fields for dim in field.shape)

    @classmethod
    def _fields(cls, spec: Any) -> Any:
        """Validate nested layout leaves and yield tensor specifications in traversal order."""
        if isinstance(spec, TensorFieldSpec):
            yield spec
        elif isinstance(spec, Mapping):
            if any(not isinstance(key, str) for key in spec):
                raise ValueError("model_batch_spec mapping keys must be strings.")
            for item in spec.values():
                yield from cls._fields(item)
        elif isinstance(spec, (tuple, list)):
            for item in spec:
                yield from cls._fields(item)
        elif spec is not None and not isinstance(spec, (bool, int, float, str)):
            raise ValueError("model_batch_spec leaves must be TensorFieldSpec or static scalar values.")

    def flatten(self, batch: Any) -> list[torch.Tensor]:
        """Validate source fields and return tensors in the declared order.

        Args:
            batch: Source batch matching this layout.
        """
        return list(self._flatten(self.spec, batch))

    @classmethod
    def _flatten(cls, spec: Any, value: Any) -> Any:
        """Validate batch structure and static values while yielding its tensor leaves."""
        if isinstance(spec, TensorFieldSpec):
            yield cls._validate_tensor(spec, value)
        elif isinstance(spec, Mapping):
            if not isinstance(value, Mapping) or value.keys() != spec.keys():
                raise ValueError("Batch fields do not match model_batch_spec.")
            for key, item in spec.items():
                yield from cls._flatten(item, value[key])
        elif isinstance(spec, (tuple, list)):
            if type(value) is not type(spec) or len(value) != len(spec):
                raise ValueError("Batch container does not match model_batch_spec.")
            for item, child in zip(spec, value):
                yield from cls._flatten(item, child)
        elif type(value) is not type(spec) or value != spec:
            raise ValueError("Non-tensor batch values must equal their static model_batch_spec values.")

    @staticmethod
    def _validate_tensor(spec: TensorFieldSpec, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError("model_batch_spec requires a dense tensor at every TensorFieldSpec leaf.")
        if value.dtype != spec.dtype or value.ndim != len(spec.shape) or any(
                dim is not None and dim != actual for dim, actual in zip(spec.shape, value.shape)
        ):
            raise ValueError(f"Tensor shape/dtype {value.shape}/{value.dtype} does not match {spec}.")
        return value

    def dimensions(self, tensors: list[torch.Tensor]) -> list[int]:
        """Extract only dimensions declared dynamic; fields and dtypes stay local.

        Args:
            tensors: Validated source tensors in declaration order.
        """
        return [actual for field, tensor in zip(self.fields, tensors)
                for dim, actual in zip(field.shape, tensor.shape) if dim is None]

    def shapes(self, dimensions: list[int]) -> list[tuple[int, ...]]:
        """Resolve receive shapes from the compact dynamic-dimension header.

        Args:
            dimensions: Dynamic sizes received in declaration order.
        """
        if len(dimensions) != self.dynamic_dimensions or any(dim < 0 for dim in dimensions):
            raise ValueError("Invalid model batch dimension header.")
        pending = iter(dimensions)
        return [tuple(next(pending) if dim is None else dim for dim in field.shape) for field in self.fields]

    def rebuild(self, tensors: list[torch.Tensor]) -> Any:
        """Restore the locally declared containers and static values.

        Args:
            tensors: Broadcast tensor fields in declaration order.
        """
        return self._rebuild(self.spec, iter(tensors))

    @classmethod
    def _rebuild(cls, spec: Any, tensors: Any) -> Any:
        if isinstance(spec, TensorFieldSpec):
            return next(tensors)
        if isinstance(spec, Mapping):
            return {key: cls._rebuild(item, tensors) for key, item in spec.items()}
        if isinstance(spec, (tuple, list)):
            return type(spec)(cls._rebuild(item, tensors) for item in spec)
        return spec

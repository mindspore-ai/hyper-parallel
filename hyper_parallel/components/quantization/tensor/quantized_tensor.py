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
"""Logical PyTorch wrapper for low-precision physical storage."""

from typing import Any, Callable, Iterable, Optional

import torch  # pylint: disable=forbidden-backend-import


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Build contiguous strides without allocating logical tensor storage."""

    if not shape:
        return ()
    strides = [1]
    for dimension in reversed(shape[1:]):
        strides.append(strides[-1] * dimension)
    return tuple(reversed(strides))


class QuantizedTensor(torch.Tensor):
    """Expose quantized storage through a logical PyTorch Tensor wrapper."""

    @staticmethod
    def __new__(
        cls,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        device: torch.device,
        requires_grad: bool = False,
    ) -> "QuantizedTensor":
        """Create a wrapper subclass without high-precision physical storage."""

        logical_shape = tuple(shape)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            logical_shape,
            strides=_contiguous_strides(logical_shape),
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            device=device,
            requires_grad=requires_grad,
        )

    def _new_with_transformed_data(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        requires_grad: bool,
    ) -> "QuantizedTensor":
        """Rebuild the wrapper after transforming its physical tensors."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement storage transformation."
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Handle operations with a safe physical-storage interpretation."""

        del types
        kwargs = kwargs or {}
        tensor = args[0]
        if func == torch.ops.aten.detach.default:
            return tensor._new_with_transformed_data(  # pylint: disable=protected-access
                lambda value: value.detach(),
                requires_grad=False,
            )
        if func == torch.ops.aten.clone.default:
            memory_format = kwargs.get("memory_format", torch.preserve_format)
            return tensor._new_with_transformed_data(  # pylint: disable=protected-access
                lambda value: value.clone(memory_format=memory_format),
                requires_grad=tensor.requires_grad,
            )
        if func == torch.ops.aten._to_copy.default:
            requested_dtype = kwargs.get("dtype")
            if requested_dtype is not None and requested_dtype != tensor.dtype:
                raise ValueError(
                    "QuantizedTensor.to(dtype=...) requires explicit "
                    f"dequantization; logical dtype is {tensor.dtype}, "
                    f"requested {requested_dtype}."
                )
            device = kwargs.get("device", tensor.device)
            non_blocking = kwargs.get("non_blocking", False)
            return tensor._new_with_transformed_data(  # pylint: disable=protected-access
                lambda value: value.to(
                    device=device,
                    non_blocking=non_blocking,
                ),
                requires_grad=tensor.requires_grad,
            )
        raise NotImplementedError(
            f"{type(tensor).__name__} does not implement the operation {func}."
        )

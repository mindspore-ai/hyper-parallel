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
"""Shadow tensor.

This module provides: :class:`ShadowTensor` — a :class:`torch.Tensor` subclass that
resolves a view of its :class:`PhysicalBuffer`'s storage on every dispatch.
"""

from __future__ import annotations

from typing import Any

import torch

from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import PhysicalBuffer


class ShadowTensor(torch.Tensor):
    """Tensor subclass that resolves a device view on demand.

    The shadow uses :meth:`PhysicalBuffer.device_storage` to obtain the
    raw device storage on every :meth:`resolve` call and creates a
    fresh tensor view from it.  **No cached reference is kept**, so the
    shadow never prevents the underlying storage from being freed.

    Because :class:`ShadowTensor` is created via
    :meth:`torch.Tensor._make_wrapper_subclass`, it carries all tensor
    metadata ("dtype", "size", "stride", "storage_offset",
    "device") directly.
    """

    @staticmethod
    def __new__(
        cls,
        elem: torch.Tensor,
        buffer: PhysicalBuffer,
        storage_id: int,
    ) -> ShadowTensor:
        """Create a new instance using *elem*'s metadata."""
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )

    def __init__(  # pylint: disable=unused-argument
        self,
        elem: torch.Tensor,
        buffer: PhysicalBuffer,
        storage_id: int,
    ) -> None:
        self._buffer = buffer
        self._storage_id = storage_id

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def storage_id(self) -> int:
        """Storage ID of the underlying physical block."""
        return self._storage_id

    # ------------------------------------------------------------------
    # Resolution (on every call)
    # ------------------------------------------------------------------

    def resolve(self) -> torch.Tensor:
        """Return a device-resident view of the physical storage.

        1. Calls :meth:`PhysicalBuffer.device_storage` to obtain the
           raw device storage (demand-paging from host if needed).
        2. Builds a fresh tensor view from the shadow's cached metadata.
        3. Returns the view — no long-lived reference is kept.
        """
        storage = self._buffer.device_storage()
        result = torch.empty(0, dtype=self.dtype, device=self.device)
        result.set_(storage, self.storage_offset(), self.size(), self.stride())
        return result

    # ------------------------------------------------------------------
    # PyTorch dispatch
    # ------------------------------------------------------------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        """Dispatch a torch operation."""
        kwargs = kwargs or {}

        def unwrap(value: Any) -> Any:
            if isinstance(value, ShadowTensor):
                return value.resolve()
            if isinstance(value, tuple):
                return tuple(unwrap(v) for v in value)
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            if isinstance(value, dict):
                return {k: unwrap(v) for k, v in value.items()}
            return value

        with torch._C._DisableTorchDispatch():  # pylint: disable=protected-access
            return func(*unwrap(args), **unwrap(kwargs))

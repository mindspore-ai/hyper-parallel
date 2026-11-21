# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Torch ``DTensorBase``: the platform-agnostic base of :class:`DTensor`."""
# pylint: disable=C9006,C9007
from typing import Tuple, Dict, Any, Optional
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily


def _make_shell_subclass(cls, local_tensor: Tensor) -> Tensor:
    """Wrap ``local_tensor`` in subclass ``cls``, aliasing or shadowing its storage.

    Real local tensors share their storage with the wrapper (the normal path,
    byte-identical to a plain ``_make_subclass``). ``Tensor._make_subclass``
    cannot re-wrap a FakeTensor because a FakeTensor is already a Tensor
    subclass, so storage-less locals get a meta shell with identical metadata
    instead — built as a :func:`_fake_local_variant` of ``cls`` so every
    FakeTensor-specific accessor lives on that variant class and the real
    training path never executes a fake-aware branch.

    Args:
        cls: The DTensor subclass to instantiate.
        local_tensor: The local shard the wrapper represents.

    Returns:
        Tensor: An uninitialized ``cls`` instance (``__init_data__`` pending).
    """
    if isinstance(local_tensor, FakeTensor):
        # Inside an active FakeTensorMode even meta factories return
        # FakeTensors, which _make_subclass rejects; build the shell outside.
        with unset_fake_temporarily():
            shell = torch.empty_strided(
                local_tensor.size(),
                local_tensor.stride(),
                dtype=local_tensor.dtype,
                device="meta",
            )
        return Tensor._make_subclass(
            _fake_local_variant(cls), shell, local_tensor.requires_grad
        )
    return Tensor._make_subclass(cls, local_tensor, local_tensor.requires_grad)


_FAKE_LOCAL_VARIANTS: Dict[type, type] = {}


def _fake_local_variant(cls: type) -> type:
    """Return the meta-shell variant of DTensor subclass ``cls``.

    The variant overrides exactly the accessors whose base implementation
    assumes the wrapper owns real storage. Keeping these overrides on a
    separate class (instead of ``if is_meta_shell`` branches in the base)
    means real-tensor instances resolve the unmodified base methods — the
    hot ``grad``/``data``/``zero_grad`` paths cost nothing extra when
    dry-run is not in use. Variants are cached per subclass and are
    idempotent (a variant is its own variant), so ``self.__class__(...)``
    reconstruction in ``detach``/``to``/``type`` stays type-stable, which
    ``nn.Parameter`` requires of ``detach``.
    """
    if getattr(cls, "_is_fake_local_variant", False):
        return cls
    variant = _FAKE_LOCAL_VARIANTS.get(cls)
    if variant is None:
        variant = type(f"_FakeLocal{cls.__name__}", (cls,), {
            "_is_fake_local_variant": True,
            "is_meta_shell": _variant_is_meta_shell,
            "grad": property(_variant_grad_get, _variant_grad_set),
            "data": property(_variant_data_get, _variant_data_set),
        })
        _FAKE_LOCAL_VARIANTS[cls] = variant
    return variant


def _variant_is_meta_shell(self) -> bool:
    """Meta-shell wrappers have no storage of their own."""
    return True


def _variant_grad_get(self) -> Optional[Tensor]:
    """Return the stored DTensor gradient wrapper while it owns the leaf's grad."""
    local_grad = self._local_tensor.grad
    wrapper = getattr(self, "_grad_dtensor_wrapper", None)
    if wrapper is not None and wrapper._local_tensor is local_grad:
        return wrapper
    return local_grad


def _variant_grad_set(self, value: Optional[Tensor]) -> None:
    """Store DTensor gradients on the local leaf; autograd rejects meta-device grads."""
    if isinstance(value, DTensorBase) and value.is_meta_shell():
        self._local_tensor.grad = value._local_tensor
        self._grad_dtensor_wrapper = value
        return
    self._local_tensor.grad = value
    if getattr(self, "_grad_dtensor_wrapper", None) is not None:
        self._grad_dtensor_wrapper = None


def _variant_data_get(self):
    """A meta shell owns no storage; the local tensor's data is the payload."""
    with torch._C.DisableTorchFunctionSubclass():
        return self._local_tensor.data


def _variant_data_set(self, value) -> None:
    """Rebind only the local tensor; the meta shell cannot adopt device storage."""
    local_value = value.to_local() if isinstance(value, DTensorBase) else value
    with torch._C.DisableTorchFunctionSubclass():
        Tensor.data.__set__(self._local_tensor, local_value)


class DTensorBase(Tensor):
    """Base class of :class:`DTensor`: a ``torch.Tensor`` subclass that owns a local shard.

    Holds the local shard plus the ``DeviceMesh``/``Placements`` describing how that shard
    maps onto the global tensor, and routes every op through the distributed dispatcher.
    """

    def __new__(cls, local_tensor, device_mesh=None, placements=None, layout=None, shape=None):
        """
        Create a new DTensorBase instance.

        Args:
            local_tensor: The local tensor shard or another DTensorBase instance.
            device_mesh: The device mesh describing the device topology.
            placements: The placement strategy for each mesh dimension.
            layout: Optional pre-built Layout reused directly by ``__init_data__``
                (skips ``_build_layout``; see ``DTensor.from_local_with_layout``).
            shape: Optional logical global tensor shape.
        """
        if isinstance(local_tensor, DTensorBase):
            # Copy from existing DTensorBase — use alias_placements to preserve multi-axis ordering
            t = _make_shell_subclass(cls, local_tensor._local_tensor)
            copy_placements = local_tensor.layout.alias_placements if local_tensor.layout else local_tensor.placements
            t.__init_data__(
                local_tensor._local_tensor,
                local_tensor.device_mesh,
                copy_placements,
                shape=getattr(local_tensor, "_global_shape", None),
            )
            return t

        if device_mesh is None:
            raise ValueError("device_mesh is None, must provide a DeviceMesh instance")
        if placements is None:
            raise ValueError("placements is None, must provide placements")

        # Create Tensor subclass instance, sharing local_tensor's underlying storage
        # (or shadowing it with a meta shell when the local is a FakeTensor).
        t = _make_shell_subclass(cls, local_tensor)
        t.__init_data__(local_tensor, device_mesh, placements, layout, shape)
        return t

    # pylint: disable=W0613, G.NAM.05
    @classmethod
    def __torch_function__(
        cls,
        func: torch._C._FunctionBase,
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Override PyTorch's __torch_function__ to intercept tensor operations.

        This method dispatches operations through the distributed operator dispatcher
        to handle DTensor-specific layout inference and redistribution.

        Args:
            func (torch._C._FunctionBase): The PyTorch function being called.
            types (Tuple[type, ...]): The types of tensors involved in the operation.
            args (Tuple[Any, ...]): Positional arguments passed to the function.
            kwargs (Optional[Dict[str, Any]]): Keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation, typically a DTensor or tuple of DTensors.
        """
        kwargs = kwargs or {}
        if func in (torch.numel, Tensor.numel):
            # numel must keep logical global semantics (DTensor.numel) even for
            # functional / unbound call forms and under SkipDTensorDispatch,
            # where the bypass would otherwise leak the local shard's capacity.
            tensor = args[0] if args else kwargs["input"]
            return tensor.numel()
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER
        out = _OP_DISPATCHER.dispatch(func, args, kwargs)
        return out

    @property
    def grad(self) -> Optional[Tensor]:
        """
        Get the gradient tensor of the local tensor.

        Returns:
            Optional[Tensor]: The gradient tensor, or None if no gradient is set.
        """
        return self._local_tensor.grad

    @grad.setter
    def grad(self, value: Optional[Tensor]) -> None:
        """
        Set the gradient tensor for the local tensor.

        Args:
            value (Optional[Tensor]): The gradient tensor to set, or None to clear.
        """
        self._local_tensor.grad = value

    def is_meta_shell(self) -> bool:
        """Return whether this wrapper is a storage-less meta shell.

        Always False here; the ``_fake_local_variant`` subclass used for
        FakeTensor locals overrides this (and the storage-touching accessors)
        so the real-tensor path carries no fake-aware branches.
        """
        return False

    @property
    def requires_grad(self) -> bool:
        """
        Check if gradient computation is enabled for this tensor.

        Returns:
            bool: True if gradients should be computed for this tensor.
        """
        return self._local_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Enable or disable gradient computation for this tensor.

        Args:
            value (bool): True to enable gradient computation, False to disable.
        """
        self._local_tensor.requires_grad_(value)
        # Sync DTensor wrapper's requires_grad
        super().requires_grad_(value)

    def requires_grad_(self, requires_grad: bool = True):
        """
        Enable or disable gradient computation in-place.

        Args:
            requires_grad (bool): True to enable gradient computation. Default: True.

        Returns:
            DTensorBase: Self for method chaining.
        """
        self._local_tensor.requires_grad_(requires_grad)
        super().requires_grad_(requires_grad)
        return self

    @property
    def grad_fn(self) -> Optional[torch.autograd.Function]:
        """
        Get the gradient function that created this tensor.

        Returns:
            Optional[torch.autograd.Function]: The gradient function, or None if not applicable.
        """
        return self._local_tensor.grad_fn

    def grad_zero_(self):
        """
        Zero out the gradient tensor in-place.

        Returns:
            DTensorBase: Self for method chaining.
        """
        if self._local_tensor.grad is not None:
            self._local_tensor.grad.zero_()
        return self

    def detach(self):
        """
        Create a detached DTensor that does not require gradient.

        Returns:
            DTensorBase: A new DTensor with the same data but detached from the computation graph.
        """
        detached_local = self._local_tensor.detach()
        return self.__class__(
            detached_local,
            device_mesh=self._device_mesh,
            placements=self._alias_placements(),
            shape=getattr(self, "_global_shape", None),
        )

    def detach_(self):
        """
        Detach this tensor from the computation graph in-place.

        Returns:
            DTensorBase: Self for method chaining.
        """
        self._local_tensor.detach_()
        super().detach_()
        return self

    # ====================== Computation graph related overrides ======================
    @property
    def is_leaf(self) -> bool:
        """
        Check if this tensor is a leaf node in the computation graph.

        Returns:
            bool: True if this is a leaf tensor (created by user, not by any operation).
        """
        return self._local_tensor.is_leaf

    @property
    def retains_grad(self) -> bool:
        """
        Check if this tensor retains its gradient during backward pass.

        Returns:
            bool: True if gradients are retained for non-leaf tensors.
        """
        return self._local_tensor.retains_grad

    @retains_grad.setter
    def retains_grad(self, value: bool) -> None:
        """
        Enable or disable gradient retention for this tensor.

        Args:
            value (bool): True to enable gradient retention.
        """
        self._local_tensor.retains_grad_(value)

    def backward(self, gradient=None, retain_graph=None, create_graph=False) -> None:
        """
        Compute the gradients for this tensor.

        Args:
            gradient (Optional[Tensor]): The gradient of the loss w.r.t. this tensor.
            retain_graph (Optional[bool]): Whether to retain the computation graph.
            create_graph (bool): Whether to create a graph of the gradient computation.
        """
        self._local_tensor.backward(gradient, retain_graph, create_graph)

    # ====================== Metadata related overrides (sync with local_tensor) ======================
    @property
    def device(self) -> torch.device:
        """
        Get the device on which this tensor is stored.

        Returns:
            torch.device: The device object (e.g., 'cuda:0', 'cpu').
        """
        return self._local_tensor.device

    @property
    # pylint: disable=C2801
    def data(self):
        """
        Directory get Tensor.data relative storage.
        After DTensor object created, there are two reference on underlying storage. (.data and ._local_tensor)
        If not using DisableTorchFunctionSubclass, Tensor.__get__ will goto '__torch_function__',
        and finally return '._local_tensor' storage.
        """
        with getattr(torch, "_C").DisableTorchFunctionSubclass():
            # Directory get Tensor.data relative storage.
            # If not using DisableTorchFunctionSubclass, Tensor.__get__ will goto __torch_funtin
            return Tensor.data.__get__(self, type(self))

    @data.setter
    # pylint: disable=C2801
    def data(self, value):
        """Set the underlying tensor data, extracting the local shard if a DTensor is given."""
        local_value = value.to_local() if isinstance(value, DTensorBase) else value
        # Tensor.data.__set__ on a Tensor subclass otherwise enters __torch_function__
        # and only rebinds _local_tensor through DTensor dispatch.
        with getattr(torch, "_C").DisableTorchFunctionSubclass():
            Tensor.data.__set__(self, local_value)
            Tensor.data.__set__(self._local_tensor, local_value)

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the data type of this tensor.

        Returns:
            torch.dtype: The data type (e.g., torch.float32, torch.int64).
        """
        return self._local_tensor.dtype

    @property
    def shape(self) -> torch.Size:
        """
        Get the shape of this tensor.

        Returns:
            torch.Size: The shape of the tensor.
        """
        return self._local_tensor.shape

    def type(self, dtype=None, non_blocking=False):
        """
        Convert this tensor to the specified dtype.

        Args:
            dtype (Optional[torch.dtype]): The target dtype. If None, returns the current type string.
            non_blocking (bool): Whether to perform the operation asynchronously. Default: False.

        Returns:
            Union[str, DTensorBase]: The type string if dtype is None, otherwise a new DTensor.
        """
        if dtype is None:
            return self._local_tensor.type()
        new_local = self._local_tensor.to(dtype=dtype, non_blocking=non_blocking)
        return self.__class__(
            new_local,
            device_mesh=self._device_mesh,
            placements=self._alias_placements(),
            shape=getattr(self, "_global_shape", None),
        )

    def size(self, dim: Optional[int] = None):
        """
        Get the size of this tensor.

        Args:
            dim (Optional[int]): The dimension to query. If None, returns the full shape.

        Returns:
            Union[torch.Size, int]: The shape or size along a specific dimension.
        """
        return self._local_tensor.size(dim)

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of this tensor.

        Returns:
            int: The number of dimensions.
        """
        return self._local_tensor.ndim

    def data_ptr(self) -> int:
        """
        Get the pointer to the data storage of the local tensor.

        Returns:
            int: The memory address of the tensor's data.
        """
        # Force return local_tensor's data pointer (ensure address consistency)
        return self._local_tensor.data_ptr()

    def untyped_storage(self) -> torch.UntypedStorage:
        """
        Get the untyped storage of the local tensor.

        Like ``data_ptr``, storage queries answer for the local shard: the
        wrapper either aliases the same storage (real locals) or holds a
        storage-less meta shell (FakeTensor locals).

        Returns:
            torch.UntypedStorage: The local tensor's untyped storage.
        """
        return self._local_tensor.untyped_storage()

    def numel(self) -> int:
        """
        Get the total number of elements in this tensor.

        Returns:
            int: The total number of elements.
        """
        return self._local_tensor.numel()

    # ====================== Auxiliary print ======================
    def _alias_placements(self):
        """Return alias_placements from layout, falling back to _placements."""
        if hasattr(self, '_layout') and self._layout is not None:
            return self._layout.alias_placements
        return self._placements

    def to(self, *args, **kwargs):
        """Move the DTensor to a different device or dtype.

        This method overrides the base Tensor.to() to properly reconstruct
        a DTensor with device_mesh and placements preserved. Uses _make_subclass
        to avoid issues with Parameter subclasses that don't accept extra kwargs.

        Args:
            *args: Arguments passed to the underlying tensor's to() method.
            **kwargs: Keyword arguments for the tensor conversion.

        Returns:
            DTensorBase: A new DTensor with the converted local tensor.
        """
        new_local = self._local_tensor.to(*args, **kwargs)
        new_dt = _make_shell_subclass(type(self), new_local)
        new_dt.__init_data__(
            new_local,
            self._device_mesh,
            self._alias_placements(),
            shape=getattr(self, "_global_shape", None),
        )
        return new_dt

    def __repr__(self) -> str:
        return (
            f"DTensor(\n"
            f"  local_tensor={self._local_tensor},\n"
            f"  device_mesh={self._device_mesh},\n"
            f"  placements={self._placements},\n"
            f"  layout={getattr(self, '_layout', None)},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  requires_grad={self.requires_grad},\n"
            f"  grad={self.grad},\n"
            f"  is_leaf={self.is_leaf},\n"
            f"  data_ptr={self.data_ptr()}\n"
            f")"
        )

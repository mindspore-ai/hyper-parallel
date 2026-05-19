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
"""Context parallel style for DeepSeek Sparse Attention (DSA).

These PyNative module-level styles are companions to the DSA distributed
operators.  The distributed operators define the per-op layout rules for
``lightning_indexer``, ``npu_sparse_flash_attention`` and indexer-loss custom
ops; these styles prepare module inputs so those rules can be selected by the
DTensor dispatcher.

The first implementation intentionally supports only Colossal-style CP:
query-side tensors are sharded on sequence, while key-side tensors are gathered
to CP-replicated layouts.  Ulysses/head sharding is rejected because the current
DSA kernels require attention head, index head, head dim and sparse top-k dims to
stay replicated.
"""
from typing import Any, Optional

from hyper_parallel.core.context_parallel.context_parallel import _ensure_1d, _gather_seq
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module


_SUPPORTED_LAYOUTS = ("BSND", "TND")


def _is_tensor_or_dtensor(value: Any) -> bool:
    """Return True for framework tensors and HyperParallel DTensors."""
    return isinstance(value, DTensor) or platform.is_tensor(value)


def _to_sequence_shard(value: Any, device_mesh: DeviceMesh, seq_dim: int) -> Any:
    """Annotate ``value`` as sequence-sharded on ``device_mesh``."""
    if not _is_tensor_or_dtensor(value):
        return value
    if isinstance(value, DTensor):
        return value.redistribute(device_mesh, (Shard(seq_dim),))
    return DTensor.from_local(value, device_mesh, (Shard(seq_dim),))


def _to_sequence_replicate(value: Any, device_mesh: DeviceMesh, seq_dim: int) -> Any:
    """Annotate ``value`` as local sequence shard, then all-gather on CP."""
    if not _is_tensor_or_dtensor(value):
        return value
    if isinstance(value, DTensor):
        return value.redistribute(device_mesh, (Replicate(),))
    return _gather_seq(value, device_mesh, seq_dim)


def _maybe_replace_arg(args: list, index: Optional[int], fn) -> None:
    """Apply ``fn`` to ``args[index]`` when the index points to an existing arg."""
    if index is None or index >= len(args):
        return
    args[index] = fn(args[index])


def _maybe_replace_kwarg(kwargs: dict, name: Optional[str], fn) -> None:
    """Apply ``fn`` to ``kwargs[name]`` when the key exists."""
    if name is None or name not in kwargs:
        return
    kwargs[name] = fn(kwargs[name])


def _validate_layout_and_mode(style_name: str, layout: str, mode: str) -> tuple[str, int]:
    """Return normalized layout and sequence dim for the DSA CP style."""
    layout = layout.upper()
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"layout must be one of {_SUPPORTED_LAYOUTS}, but got {layout!r}.")
    if mode != "colossal":
        raise ValueError(f"{style_name} currently supports only mode='colossal'.")
    return layout, 1 if layout == "BSND" else 0


def _finalize_output(value: Any, use_local_output: bool) -> Any:
    """Convert direct DTensor outputs, or one-level tuple/list outputs, to local tensors."""
    if isinstance(value, DTensor):
        return value.to_local() if use_local_output else value
    if isinstance(value, tuple):
        return tuple(v.to_local() if use_local_output and isinstance(v, DTensor) else v for v in value)
    if isinstance(value, list):
        return [v.to_local() if use_local_output and isinstance(v, DTensor) else v for v in value]
    return value


def _dtensor_has_partial(value: DTensor) -> bool:
    """Return whether ``value`` has any Partial placement."""
    return any(placement.is_partial() for placement in value.placements)


def _dtensor_to_local_reducing_partial(value: Any) -> Any:
    """Convert a DTensor to local, reducing Partial only when communication is needed."""
    if not isinstance(value, DTensor):
        return value
    if _dtensor_has_partial(value) and value.device_mesh.mesh.numel() > 1:
        value = value.reduce_partial()
    return value.to_local()


def _register_boundary_hooks(module: Module, pre_hook, use_local_output: bool) -> None:
    """Register a DSA boundary pre-hook and its public output conversion hook."""
    platform.register_forward_pre_hook(module, pre_hook, with_kwargs=True)
    module.register_forward_hook(
        lambda _module, _args, outputs: _finalize_output(outputs, use_local_output)
    )


def _configure_sparse_attention_boundary(  # pylint: disable=too-many-arguments
        style,
        *,
        layout: str,
        mode: str,
        query_index: Optional[int],
        key_index: Optional[int],
        value_index: Optional[int],
        topk_index: Optional[int],
        query_kwarg_name: Optional[str],
        key_kwarg_name: Optional[str],
        value_kwarg_name: Optional[str],
        topk_kwarg_name: Optional[str],
        query_rope_index: Optional[int],
        key_rope_index: Optional[int],
        query_rope_kwarg_name: Optional[str],
        key_rope_kwarg_name: Optional[str],
        use_local_output: bool,
) -> None:
    """Store sparse-attention boundary configuration on ``style``."""
    layout, seq_dim = _validate_layout_and_mode(style.__class__.__name__, layout, mode)
    style.layout = layout
    style.mode = mode
    style.seq_dim = seq_dim
    style.query_index = query_index
    style.key_index = key_index
    style.value_index = value_index
    style.topk_index = topk_index
    style.query_kwarg_name = query_kwarg_name
    style.key_kwarg_name = key_kwarg_name
    style.value_kwarg_name = value_kwarg_name
    style.topk_kwarg_name = topk_kwarg_name
    style.query_rope_index = query_rope_index
    style.key_rope_index = key_rope_index
    style.query_rope_kwarg_name = query_rope_kwarg_name
    style.key_rope_kwarg_name = key_rope_kwarg_name
    style.use_local_output = use_local_output


def _apply_sparse_attention_boundary(style, module: Module, device_mesh: DeviceMesh) -> Module:
    """Register low-level DSA sparse-attention boundary hooks for ``style``."""
    cp_mesh = _ensure_1d(device_mesh)

    def _shard_query_side(value: Any) -> Any:
        return _to_sequence_shard(value, cp_mesh, style.seq_dim)

    def _replicate_key_side(value: Any) -> Any:
        return _to_sequence_replicate(value, cp_mesh, style.seq_dim)

    def _pre_hook(hook_module, args, kwargs):
        del hook_module
        new_args = list(args)
        new_kwargs = dict(kwargs)
        _maybe_replace_arg(new_args, style.query_index, _shard_query_side)
        _maybe_replace_arg(new_args, style.key_index, _replicate_key_side)
        _maybe_replace_arg(new_args, style.value_index, _replicate_key_side)
        _maybe_replace_arg(new_args, style.topk_index, _shard_query_side)
        _maybe_replace_arg(new_args, style.query_rope_index, _shard_query_side)
        _maybe_replace_arg(new_args, style.key_rope_index, _replicate_key_side)
        _maybe_replace_kwarg(new_kwargs, style.query_kwarg_name, _shard_query_side)
        _maybe_replace_kwarg(new_kwargs, style.key_kwarg_name, _replicate_key_side)
        _maybe_replace_kwarg(new_kwargs, style.value_kwarg_name, _replicate_key_side)
        _maybe_replace_kwarg(new_kwargs, style.topk_kwarg_name, _shard_query_side)
        _maybe_replace_kwarg(new_kwargs, style.query_rope_kwarg_name, _shard_query_side)
        _maybe_replace_kwarg(new_kwargs, style.key_rope_kwarg_name, _replicate_key_side)
        return tuple(new_args), new_kwargs

    _register_boundary_hooks(module, _pre_hook, style.use_local_output)
    return module


class DSAIndexerContextParallel(ParallelStyle):
    """Colossal-style CP hook for a DSA indexer boundary.

    This style targets a hookable module/cell whose forward signature is shaped
    like ``(query, key, weights, ...)`` and rewrites only that boundary:

    - ``query`` and ``weights`` are annotated as ``Shard(seq)``;
    - ``key`` is all-gathered to ``Replicate()``.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            layout: str = "BSND",
            mode: str = "colossal",
            query_index: Optional[int] = 0,
            key_index: Optional[int] = 1,
            weights_index: Optional[int] = 2,
            query_kwarg_name: Optional[str] = None,
            key_kwarg_name: Optional[str] = None,
            weights_kwarg_name: Optional[str] = None,
            use_local_output: bool = True,
    ) -> None:
        super().__init__()
        layout, seq_dim = _validate_layout_and_mode(self.__class__.__name__, layout, mode)
        self.layout = layout
        self.mode = mode
        self.seq_dim = seq_dim
        self.query_index = query_index
        self.key_index = key_index
        self.weights_index = weights_index
        self.query_kwarg_name = query_kwarg_name
        self.key_kwarg_name = key_kwarg_name
        self.weights_kwarg_name = weights_kwarg_name
        self.use_local_output = use_local_output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layout={self.layout!r}, mode={self.mode!r}, "
            f"use_local_output={self.use_local_output})"
        )

    def _shard_query_side(self, value: Any, device_mesh: DeviceMesh) -> Any:
        return _to_sequence_shard(value, device_mesh, self.seq_dim)

    def _replicate_key_side(self, value: Any, device_mesh: DeviceMesh) -> Any:
        return _to_sequence_replicate(value, device_mesh, self.seq_dim)

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register DSA indexer CP hooks on ``module`` and return it."""
        cp_mesh = _ensure_1d(device_mesh)

        def _pre_hook(hook_module, args, kwargs):
            del hook_module
            new_args = list(args)
            new_kwargs = dict(kwargs)
            _maybe_replace_arg(new_args, self.query_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.key_index, lambda t: self._replicate_key_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.weights_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_kwarg(new_kwargs, self.query_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_kwarg(new_kwargs, self.key_kwarg_name, lambda t: self._replicate_key_side(t, cp_mesh))
            _maybe_replace_kwarg(new_kwargs, self.weights_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh))
            return tuple(new_args), new_kwargs

        _register_boundary_hooks(module, _pre_hook, self.use_local_output)
        return module


class DSASparseAttentionContextParallel(ParallelStyle):
    """Colossal-style CP hook for a DSA sparse-attention boundary.

    This style targets a hookable module/cell whose forward signature is shaped
    like ``(query, key, value, topk_indices, query_rope, key_rope, ...)`` and
    rewrites only that boundary:

    - ``query``, ``topk_indices`` and ``query_rope`` are annotated as ``Shard(seq)``;
    - ``key``, ``value`` and ``key_rope`` are all-gathered to ``Replicate()``.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            layout: str = "BSND",
            mode: str = "colossal",
            query_index: Optional[int] = 0,
            key_index: Optional[int] = 1,
            value_index: Optional[int] = 2,
            topk_index: Optional[int] = 3,
            query_kwarg_name: Optional[str] = None,
            key_kwarg_name: Optional[str] = None,
            value_kwarg_name: Optional[str] = None,
            topk_kwarg_name: Optional[str] = None,
            query_rope_index: Optional[int] = 4,
            key_rope_index: Optional[int] = 5,
            query_rope_kwarg_name: Optional[str] = "query_rope",
            key_rope_kwarg_name: Optional[str] = "key_rope",
            use_local_output: bool = True,
    ) -> None:
        super().__init__()
        _configure_sparse_attention_boundary(
            self,
            layout=layout,
            mode=mode,
            query_index=query_index,
            key_index=key_index,
            value_index=value_index,
            topk_index=topk_index,
            query_kwarg_name=query_kwarg_name,
            key_kwarg_name=key_kwarg_name,
            value_kwarg_name=value_kwarg_name,
            topk_kwarg_name=topk_kwarg_name,
            query_rope_index=query_rope_index,
            key_rope_index=key_rope_index,
            query_rope_kwarg_name=query_rope_kwarg_name,
            key_rope_kwarg_name=key_rope_kwarg_name,
            use_local_output=use_local_output,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layout={self.layout!r}, mode={self.mode!r}, "
            f"use_local_output={self.use_local_output})"
        )

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register DSA sparse-attention CP hooks on ``module`` and return it."""
        return _apply_sparse_attention_boundary(self, module, device_mesh)


class DSAIndexerLossContextParallel(ParallelStyle):
    """Colossal-style CP hook for a DSA indexer-loss kernel boundary.

    This style targets a hookable module/cell whose forward signature is shaped
    like ``(query, key, query_index, key_index, weights, topk_indices,
    softmax_max, softmax_sum, query_rope, key_rope, ...)``.  The boundary is
    expected to start after MF has already done local-only bookkeeping such as
    ``stop_gradient`` and ``split``.

    Placements:
    - ``query``, ``query_index``, ``weights``, ``topk_indices`` and
      ``query_rope`` are annotated as ``Shard(seq)``;
    - ``key``, ``key_index`` and ``key_rope`` are all-gathered to
      ``Replicate()``.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            layout: str = "BSND",
            mode: str = "colossal",
            query_index: Optional[int] = 0,
            key_index: Optional[int] = 1,
            query_indexer_index: Optional[int] = 2,
            key_indexer_index: Optional[int] = 3,
            weights_index: Optional[int] = 4,
            topk_index: Optional[int] = 5,
            query_rope_index: Optional[int] = 8,
            key_rope_index: Optional[int] = 9,
            query_kwarg_name: Optional[str] = None,
            key_kwarg_name: Optional[str] = None,
            query_indexer_kwarg_name: Optional[str] = None,
            key_indexer_kwarg_name: Optional[str] = None,
            weights_kwarg_name: Optional[str] = None,
            topk_kwarg_name: Optional[str] = None,
            query_rope_kwarg_name: Optional[str] = None,
            key_rope_kwarg_name: Optional[str] = None,
            use_local_output: bool = True,
    ) -> None:
        super().__init__()
        layout, seq_dim = _validate_layout_and_mode(self.__class__.__name__, layout, mode)
        self.layout = layout
        self.mode = mode
        self.seq_dim = seq_dim
        self.query_index = query_index
        self.key_index = key_index
        self.query_indexer_index = query_indexer_index
        self.key_indexer_index = key_indexer_index
        self.weights_index = weights_index
        self.topk_index = topk_index
        self.query_rope_index = query_rope_index
        self.key_rope_index = key_rope_index
        self.query_kwarg_name = query_kwarg_name
        self.key_kwarg_name = key_kwarg_name
        self.query_indexer_kwarg_name = query_indexer_kwarg_name
        self.key_indexer_kwarg_name = key_indexer_kwarg_name
        self.weights_kwarg_name = weights_kwarg_name
        self.topk_kwarg_name = topk_kwarg_name
        self.query_rope_kwarg_name = query_rope_kwarg_name
        self.key_rope_kwarg_name = key_rope_kwarg_name
        self.use_local_output = use_local_output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layout={self.layout!r}, mode={self.mode!r}, "
            f"use_local_output={self.use_local_output})"
        )

    def _shard_query_side(self, value: Any, device_mesh: DeviceMesh) -> Any:
        return _to_sequence_shard(value, device_mesh, self.seq_dim)

    def _replicate_key_side(self, value: Any, device_mesh: DeviceMesh) -> Any:
        return _to_sequence_replicate(value, device_mesh, self.seq_dim)

    @staticmethod
    def _local_shape(value: Any) -> Optional[tuple]:
        if isinstance(value, DTensor):
            return value.local_shape
        if platform.is_tensor(value):
            return value.shape
        return None

    def _slice_local_key_grad(self, value: Any, module: Module) -> Any:
        """Convert d_key_index to the original local key-index shard shape."""
        if not self.use_local_output:
            return value
        target_shape = getattr(module, "_hp_dsa_loss_key_index_local_shape", None)
        if target_shape is None:
            return _finalize_output(value, use_local_output=True)

        if isinstance(value, DTensor):
            value = _dtensor_to_local_reducing_partial(value)
        if not platform.is_tensor(value):
            return value

        target_len = target_shape[self.seq_dim]
        if value.shape[self.seq_dim] == target_len:
            return value

        local_idx = getattr(module, "_hp_dsa_loss_local_idx", 0)
        start = local_idx * target_len
        return value.narrow(self.seq_dim, start, target_len)

    def _process_outputs(self, module: Module, outputs: Any) -> Any:
        """Finalize indexer-loss outputs, reducing Partial values when needed."""
        if not self.use_local_output:
            return outputs
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 4:
            return _finalize_output(outputs, use_local_output=True)

        processed = list(outputs)
        processed[0] = _dtensor_to_local_reducing_partial(processed[0])
        processed[1] = self._slice_local_key_grad(processed[1], module)
        processed[2] = _dtensor_to_local_reducing_partial(processed[2])
        processed[3] = _dtensor_to_local_reducing_partial(processed[3])
        return type(outputs)(processed)

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register DSA indexer-loss CP hooks on ``module`` and return it."""
        cp_mesh = _ensure_1d(device_mesh)
        rank_list = list(cp_mesh.rank_list)
        local_idx = rank_list.index(platform.get_rank()) if platform.get_rank() in rank_list else 0

        def _pre_hook(hook_module, args, kwargs):
            new_args = list(args)
            new_kwargs = dict(kwargs)
            key_shape = None
            if self.key_indexer_index is not None and self.key_indexer_index < len(new_args):
                key_shape = self._local_shape(new_args[self.key_indexer_index])
            elif self.key_indexer_kwarg_name and self.key_indexer_kwarg_name in new_kwargs:
                key_shape = self._local_shape(new_kwargs[self.key_indexer_kwarg_name])
            setattr(hook_module, "_hp_dsa_loss_key_index_local_shape", key_shape)
            setattr(hook_module, "_hp_dsa_loss_local_idx", local_idx)

            _maybe_replace_arg(new_args, self.query_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.key_index, lambda t: self._replicate_key_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.query_indexer_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.key_indexer_index, lambda t: self._replicate_key_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.weights_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.topk_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.query_rope_index, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_arg(new_args, self.key_rope_index, lambda t: self._replicate_key_side(t, cp_mesh))
            _maybe_replace_kwarg(new_kwargs, self.query_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_kwarg(new_kwargs, self.key_kwarg_name, lambda t: self._replicate_key_side(t, cp_mesh))
            _maybe_replace_kwarg(
                new_kwargs, self.query_indexer_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh)
            )
            _maybe_replace_kwarg(
                new_kwargs, self.key_indexer_kwarg_name, lambda t: self._replicate_key_side(t, cp_mesh)
            )
            _maybe_replace_kwarg(new_kwargs, self.weights_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_kwarg(new_kwargs, self.topk_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh))
            _maybe_replace_kwarg(
                new_kwargs, self.query_rope_kwarg_name, lambda t: self._shard_query_side(t, cp_mesh)
            )
            _maybe_replace_kwarg(
                new_kwargs, self.key_rope_kwarg_name, lambda t: self._replicate_key_side(t, cp_mesh)
            )
            return tuple(new_args), new_kwargs

        platform.register_forward_pre_hook(module, _pre_hook, with_kwargs=True)
        module.register_forward_hook(lambda _module, _args, outputs: self._process_outputs(_module, outputs))
        return module


class DSAContextParallel(ParallelStyle):
    """Colossal-style CP hook for a low-level DSA attention boundary.

    This compatibility style intentionally handles only a direct boundary whose
    forward signature is shaped like
    ``(query, key, value, topk_indices, query_rope, key_rope, ...)``.  Callers
    that own a higher-level attention module should locate its indexer and
    sparse-attention submodules themselves and apply
    :class:`DSAIndexerContextParallel` and
    :class:`DSASparseAttentionContextParallel` explicitly.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            layout: str = "BSND",
            mode: str = "colossal",
            query_index: Optional[int] = 0,
            key_index: Optional[int] = 1,
            value_index: Optional[int] = 2,
            topk_index: Optional[int] = 3,
            query_kwarg_name: Optional[str] = None,
            key_kwarg_name: Optional[str] = None,
            value_kwarg_name: Optional[str] = None,
            topk_kwarg_name: Optional[str] = None,
            query_rope_index: Optional[int] = 4,
            key_rope_index: Optional[int] = 5,
            query_rope_kwarg_name: Optional[str] = "query_rope",
            key_rope_kwarg_name: Optional[str] = "key_rope",
            use_local_output: bool = True,
    ) -> None:
        super().__init__()
        _configure_sparse_attention_boundary(
            self,
            layout=layout,
            mode=mode,
            query_index=query_index,
            key_index=key_index,
            value_index=value_index,
            topk_index=topk_index,
            query_kwarg_name=query_kwarg_name,
            key_kwarg_name=key_kwarg_name,
            value_kwarg_name=value_kwarg_name,
            topk_kwarg_name=topk_kwarg_name,
            query_rope_index=query_rope_index,
            key_rope_index=key_rope_index,
            query_rope_kwarg_name=query_rope_kwarg_name,
            key_rope_kwarg_name=key_rope_kwarg_name,
            use_local_output=use_local_output,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layout={self.layout!r}, mode={self.mode!r}, "
            f"use_local_output={self.use_local_output})"
        )

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register low-level DSA attention CP hooks on ``module`` and return it."""
        return _apply_sparse_attention_boundary(self, module, device_mesh)

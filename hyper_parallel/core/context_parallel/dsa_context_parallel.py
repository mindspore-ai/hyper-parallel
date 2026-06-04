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
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hyper_parallel.core.context_parallel.context_parallel import (
    _OUTPUT_NON_CP,
    _drop_cp_from_output,
    _ensure_1d,
    _non_cp_dtensor_layout,
    _pop_output_layout,
    _push_output_layout,
    _to_cp_dtensor,
)
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
    return _to_cp_dtensor(value, device_mesh, (Shard(seq_dim),), (Shard(seq_dim),), seq_dim)


def _to_sequence_replicate(value: Any, device_mesh: DeviceMesh, seq_dim: int) -> Any:
    """Annotate ``value`` as local sequence shard, then all-gather on CP."""
    if not _is_tensor_or_dtensor(value):
        return value
    return _to_cp_dtensor(value, device_mesh, (Shard(seq_dim),), (Replicate(),), seq_dim)


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


@dataclass
class _ParamSpec:
    """Describe one tensor argument location and transform."""

    index: Optional[int]
    kwarg_name: Optional[str]
    fn: Callable[[Any], Any]


def _apply_param_specs(args: list, kwargs: dict, specs: list[_ParamSpec]) -> None:
    """Apply each spec's transform to the matching positional or keyword argument."""
    for spec in specs:
        _maybe_replace_arg(args, spec.index, spec.fn)
        _maybe_replace_kwarg(kwargs, spec.kwarg_name, spec.fn)


def _validate_layout_and_mode(style_name: str, layout: str, mode: str) -> tuple[str, int]:
    """Return normalized layout and sequence dim for the DSA CP style."""
    layout = layout.upper()
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"layout must be one of {_SUPPORTED_LAYOUTS}, but got {layout!r}.")
    if mode != "colossal":
        raise ValueError(f"{style_name} currently supports only mode='colossal'.")
    return layout, 1 if layout == "BSND" else 0


def _finalize_output(value: Any, use_local_output: bool, output_layout=None, seq_dim: Optional[int] = None) -> Any:
    """Convert direct DTensor outputs, or one-level tuple/list outputs, to local tensors."""
    output_kind, layout = output_layout if output_layout is not None else (None, None)

    def finalize_one(item):
        if use_local_output:
            return item.to_local() if isinstance(item, DTensor) else item
        if output_kind == _OUTPUT_NON_CP and seq_dim is not None:
            return _drop_cp_from_output(item, layout, (Shard(seq_dim),))
        return item

    if isinstance(value, DTensor):
        return finalize_one(value)
    if isinstance(value, tuple):
        return tuple(finalize_one(v) for v in value)
    if isinstance(value, list):
        return [finalize_one(v) for v in value]
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


def _register_boundary_hooks(module: Module, pre_hook, use_local_output: bool, seq_dim: int) -> None:
    """Register a DSA boundary pre-hook and its public output conversion hook."""
    platform.register_forward_pre_hook(module, pre_hook, with_kwargs=True)
    module.register_forward_hook(
        lambda _module, _args, outputs: _finalize_output(
            outputs,
            use_local_output,
            _pop_output_layout(_module),
            seq_dim,
        )
    )


def _record_query_output_layout(module: Module, value: Any, cp_mesh: DeviceMesh, seq_dim: int) -> None:
    """Remember the non-CP query layout so outputs can drop CP on exit."""
    layout = _non_cp_dtensor_layout(value, cp_mesh, seq_dim)
    _push_output_layout(module, (_OUTPUT_NON_CP, layout) if layout is not None else None)


def _read_value(args: list, kwargs: dict, index: Optional[int], kwarg_name: Optional[str]) -> Any:
    """Read a positional or keyword value from a hook argument pair."""
    if index is not None and index < len(args):
        return args[index]
    if kwarg_name is not None and kwarg_name in kwargs:
        return kwargs[kwarg_name]
    return None


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


def _apply_sparse_attention_boundary(
        style,
        module: Module,
        device_mesh: DeviceMesh,
        *,
        async_state: Optional[Any] = None,
) -> Module:
    """Register low-level DSA sparse-attention boundary hooks for ``style``."""
    cp_mesh = _ensure_1d(device_mesh)

    def _shard(value: Any) -> Any:
        return _to_sequence_shard(value, cp_mesh, style.seq_dim)

    def _replicate(slot_name: str):
        if async_state is not None:
            return lambda value: async_state.wait(slot_name, value)
        return lambda value: _to_sequence_replicate(value, cp_mesh, style.seq_dim)

    specs = [
        _ParamSpec(style.query_index, style.query_kwarg_name, _shard),
        _ParamSpec(style.key_index, style.key_kwarg_name, _replicate("key")),
        _ParamSpec(style.value_index, style.value_kwarg_name, _replicate("value")),
        _ParamSpec(style.topk_index, style.topk_kwarg_name, _shard),
        _ParamSpec(style.query_rope_index, style.query_rope_kwarg_name, _shard),
        _ParamSpec(style.key_rope_index, style.key_rope_kwarg_name, _replicate("key_rope")),
    ]

    def _pre_hook(hook_module, args, kwargs):
        new_args = list(args)
        new_kwargs = dict(kwargs)
        _record_query_output_layout(
            hook_module,
            _read_value(new_args, new_kwargs, style.query_index, style.query_kwarg_name),
            cp_mesh,
            style.seq_dim,
        )
        _apply_param_specs(new_args, new_kwargs, specs)
        return tuple(new_args), new_kwargs

    _register_boundary_hooks(module, _pre_hook, style.use_local_output, style.seq_dim)
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
            use_local_output: bool = False,
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

    def _build_specs(self, cp_mesh: DeviceMesh, key_fn: Callable[[Any], Any]) -> list[_ParamSpec]:
        """Build table-driven transforms for the indexer boundary."""
        def shard(value: Any) -> Any:
            return self._shard_query_side(value, cp_mesh)

        return [
            _ParamSpec(self.query_index, self.query_kwarg_name, shard),
            _ParamSpec(self.key_index, self.key_kwarg_name, key_fn),
            _ParamSpec(self.weights_index, self.weights_kwarg_name, shard),
        ]

    def _apply_with_specs(self, module: Module, specs: list[_ParamSpec], cp_mesh: DeviceMesh) -> Module:
        """Register indexer hooks driven by parameter specs."""
        def _pre_hook(hook_module, args, kwargs):
            new_args = list(args)
            new_kwargs = dict(kwargs)
            _record_query_output_layout(
                hook_module,
                _read_value(new_args, new_kwargs, self.query_index, self.query_kwarg_name),
                cp_mesh,
                self.seq_dim,
            )
            _apply_param_specs(new_args, new_kwargs, specs)
            return tuple(new_args), new_kwargs

        _register_boundary_hooks(module, _pre_hook, self.use_local_output, self.seq_dim)
        return module

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register DSA indexer CP hooks on ``module`` and return it."""
        cp_mesh = _ensure_1d(device_mesh)
        specs = self._build_specs(
            cp_mesh,
            key_fn=lambda value: self._replicate_key_side(value, cp_mesh),
        )
        return self._apply_with_specs(module, specs, cp_mesh)


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
            use_local_output: bool = False,
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
            use_local_output: bool = False,
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
        output_layout = _pop_output_layout(module)
        if not self.use_local_output:
            return _finalize_output(outputs, False, output_layout, self.seq_dim)
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 4:
            return _finalize_output(outputs, use_local_output=True)

        processed = list(outputs)
        processed[0] = _dtensor_to_local_reducing_partial(processed[0])
        processed[1] = self._slice_local_key_grad(processed[1], module)
        processed[2] = _dtensor_to_local_reducing_partial(processed[2])
        processed[3] = _dtensor_to_local_reducing_partial(processed[3])
        return type(outputs)(processed)

    def _build_loss_specs(
            self,
            cp_mesh: DeviceMesh,
            replicate_fn_map: dict[str, Callable[[Any], Any]],
    ) -> list[_ParamSpec]:
        """Build table-driven transforms for the indexer-loss boundary."""
        def shard(value: Any) -> Any:
            return self._shard_query_side(value, cp_mesh)

        return [
            _ParamSpec(self.query_index, self.query_kwarg_name, shard),
            _ParamSpec(self.key_index, self.key_kwarg_name, replicate_fn_map["key"]),
            _ParamSpec(self.query_indexer_index, self.query_indexer_kwarg_name, shard),
            _ParamSpec(self.key_indexer_index, self.key_indexer_kwarg_name, replicate_fn_map["key_indexer"]),
            _ParamSpec(self.weights_index, self.weights_kwarg_name, shard),
            _ParamSpec(self.topk_index, self.topk_kwarg_name, shard),
            _ParamSpec(self.query_rope_index, self.query_rope_kwarg_name, shard),
            _ParamSpec(self.key_rope_index, self.key_rope_kwarg_name, replicate_fn_map["key_rope"]),
        ]

    def _read_key_indexer_shape(self, args: list, kwargs: dict) -> Optional[tuple]:
        """Read the original local key-indexer shape before hook transforms."""
        if self.key_indexer_index is not None and self.key_indexer_index < len(args):
            return self._local_shape(args[self.key_indexer_index])
        if self.key_indexer_kwarg_name and self.key_indexer_kwarg_name in kwargs:
            return self._local_shape(kwargs[self.key_indexer_kwarg_name])
        return None

    @staticmethod
    def _get_local_idx(cp_mesh: DeviceMesh) -> int:
        """Return current rank's index in the CP mesh rank list."""
        rank_list = list(cp_mesh.rank_list)
        rank = platform.get_rank()
        return rank_list.index(rank) if rank in rank_list else 0

    def _apply_with_loss_specs(
            self,
            module: Module,
            specs: list[_ParamSpec],
            local_idx: int,
            cp_mesh: DeviceMesh,
    ) -> Module:
        """Register indexer-loss hooks driven by parameter specs."""
        def _pre_hook(hook_module, args, kwargs):
            new_args = list(args)
            new_kwargs = dict(kwargs)
            _record_query_output_layout(
                hook_module,
                _read_value(new_args, new_kwargs, self.query_index, self.query_kwarg_name),
                cp_mesh,
                self.seq_dim,
            )
            key_shape = self._read_key_indexer_shape(new_args, new_kwargs)
            setattr(hook_module, "_hp_dsa_loss_key_index_local_shape", key_shape)
            setattr(hook_module, "_hp_dsa_loss_local_idx", local_idx)
            _apply_param_specs(new_args, new_kwargs, specs)
            return tuple(new_args), new_kwargs

        platform.register_forward_pre_hook(module, _pre_hook, with_kwargs=True)
        module.register_forward_hook(lambda _module, _args, outputs: self._process_outputs(_module, outputs))
        return module

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Register DSA indexer-loss CP hooks on ``module`` and return it."""
        cp_mesh = _ensure_1d(device_mesh)

        def replicate(value: Any) -> Any:
            return self._replicate_key_side(value, cp_mesh)

        specs = self._build_loss_specs(
            cp_mesh,
            replicate_fn_map={
                "key": replicate,
                "key_indexer": replicate,
                "key_rope": replicate,
            },
        )
        return self._apply_with_loss_specs(module, specs, self._get_local_idx(cp_mesh), cp_mesh)

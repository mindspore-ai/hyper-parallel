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
"""Configurable :class:`Module` with declarative init and tensor parallelism.

Layers inherit from :class:`Module`, declare ``Config`` with ``param_init`` and
:class:`~hyper_parallel.dmodule.sharding.ShardingConfig`, then call
:meth:`Module.init_states` and :meth:`Module.parallelize`.
"""
from __future__ import annotations

__all__ = ["Module"]

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hyper_parallel import get_platform
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.config.configurable import Configurable, enforce_module_config_slots
from hyper_parallel.dmodule.sharding import ShardingConfig, resolve_placements

logger = logging.getLogger(__name__)

_FORWARD_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)
platform = get_platform()
_PlatformModule = platform.Module

_created_classes: dict[type, type] = {}


def _get_attr_by_path(obj: Any, path: str) -> Any:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return getattr(obj, parts[-1])


def _set_param_by_path(module: _PlatformModule, path: str, param: Any) -> None:
    parts = path.split(".")
    if len(parts) == 1:
        module.register_parameter(parts[0], param)
        return
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    parent.register_parameter(parts[-1], param)


def _placements_equal(
    left: tuple[Placement, ...] | list[Placement],
    right: list[Placement],
) -> bool:
    if len(left) != len(right):
        return False
    return all(a == b for a, b in zip(left, right))


class Module(_PlatformModule, Configurable):
    """Declarative distributed layer base class.

    Combines the platform :class:`~hyper_parallel.platform.platform.Module` with
    :class:`~hyper_parallel.config.configurable.Configurable`. Each subclass
    defines ``Config`` (hyperparameters, ``param_init``, ``sharding_config``) and
    ``__init__(self, config)``.

    Example::

        class Linear(Module):
            class Config(Module.Config):
                in_features: int = 128
                out_features: int = 64
                param_init = {"weight": nn.init.kaiming_uniform_}
                sharding_config = ShardingConfig(
                    state_shardings={"weight": {MeshAxisName.TP: Shard(0)}},
                )

            def __init__(self, config: Config):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(config.out_features, config.in_features))

            def forward(self, x):
                return F.linear(x, self.weight, None)

        layer = Linear.Config().build()
        layer.init_states()
        layer.parallelize(tp_mesh)
    """

    _param_init: dict[str, Callable] | None = None
    _sharding_config: ShardingConfig | None = None
    _pos_arg_list: list[str] | None = None
    _parallelized: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Per-layer config: initializers and optional sharding spec.

        Args:
            param_init: Map parameter name (for example ``"weight"``) to an
                init callable applied in :meth:`Module.init_states`.
            sharding_config: Optional :class:`~hyper_parallel.dmodule.sharding.ShardingConfig`
                consumed by :meth:`Module.parallelize`.

        Example::

            Linear.Config(
                in_features=128,
                out_features=64,
                param_init={"weight": nn.init.kaiming_uniform_},
                sharding_config=ShardingConfig(
                    state_shardings={"weight": {MeshAxisName.TP: Shard(0)}},
                ),
            ).build()
        """

        param_init: dict[str, Callable] | None = None
        sharding_config: ShardingConfig | None = None

        def build(self, **kwargs):
            """Build the layer and attach ``param_init`` / ``sharding_config``.

            Args:
                **kwargs: Runtime-only ``__init__`` arguments (must not overlap
                    config field names).

            Returns:
                Instance of the outer :class:`Module` subclass.

            Example::

                layer = Linear.Config(in_features=128, out_features=64).build()
            """
            instance = Configurable.Config.build(self, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            if self.sharding_config is not None:
                instance._sharding_config = self.sharding_config
            return instance

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            enforce_module_config_slots(cls.__dict__["Config"], cls.__name__)

    @property
    def sharding_config(self) -> ShardingConfig | None:
        """Sharding spec used by :meth:`parallelize` (from Config at build time)."""
        return self._sharding_config

    def init_states(
        self,
        *,
        buffer_device: Any | None = None,
    ) -> None:
        """Initialize parameters and buffers for this subtree.

        Recurses into child :class:`Module` instances, then applies
        ``param_init`` on local parameters and calls :meth:`_init_self_buffers`.

        Args:
            buffer_device: Optional device for buffer creation (subclasses may
                use this in :meth:`_init_self_buffers`).

        Example::

            model = model_cfg.build()
            model.init_states(buffer_device=torch.device("npu", 0))
        """
        queue = list(self.children())
        while queue:
            child = queue.pop(0)
            if isinstance(child, Module):
                child.init_states(buffer_device=buffer_device)
            else:
                queue.extend(child.children())

        self._init_self_parameters()

        dtensor_meta = {
            name: (buf.device_mesh, buf.placements)
            for name, buf in self._buffers.items()
            if isinstance(buf, DTensor)
        }
        self._init_self_buffers(buffer_device=buffer_device)
        for name, (mesh, placements) in dtensor_meta.items():
            new_buf = self._buffers.get(name)
            if new_buf is None or isinstance(new_buf, DTensor):
                continue
            persistent = name not in self._non_persistent_buffers_set
            self.register_buffer(
                name,
                distribute_tensor(new_buf, mesh, list(placements)),
                persistent=persistent,
            )

    def _init_self_parameters(self) -> None:
        for name, param in self.named_parameters(recurse=False):
            self._init_param(name, param)

    def _init_param(self, name: str, param: Any) -> None:
        """Apply ``param_init`` initializer for a single parameter."""
        if self._param_init is None:
            raise ValueError(
                f"No param_init found for parameter '{name}' in "
                f"{type(self).__name__}. Set param_init on this module's Config."
            )
        if name not in self._param_init:
            raise ValueError(
                f"No initializer for parameter '{name}' in {type(self).__name__}. "
                f"Available: {list(self._param_init.keys())}"
            )
        self._param_init[name](param)

    def _init_self_buffers(self, *, buffer_device: Any | None = None) -> None:
        pass

    def _cache_pos_arg_names(self) -> list[str]:
        """Return positional argument names of :meth:`forward` (cached)."""
        if self._pos_arg_list is not None:
            return self._pos_arg_list
        sig = inspect.signature(type(self).forward)
        self._pos_arg_list = [
            p.name
            for p in sig.parameters.values()
            if p.kind in _FORWARD_POSITIONAL_KINDS and p.name != "self"
        ]
        return self._pos_arg_list

    def parallelize(self, tp_mesh: DeviceMesh) -> None:
        """Apply ``sharding_config`` and wrap ``forward`` with redistribution.

        For layers with a :class:`~hyper_parallel.dmodule.sharding.ShardingConfig`:
        shards parameters, then runs
        ``redistribute inputs -> forward -> redistribute outputs``.
        Always recurses into child :class:`Module` nodes. Callable at most once
        per instance.

        Args:
            tp_mesh: Device mesh whose ``mesh_dim_names`` align with
                :class:`~hyper_parallel.dmodule.types.MeshAxisName` keys.

        Raises:
            ValueError: If already parallelized or mesh has no axis names.
            NotImplementedError: If ``sharding_config.local_map`` is set (M9).

        Example::

            mesh = init_device_mesh("npu", mesh_shape=(2,), mesh_dim_names=("tp",))
            model = model_cfg.build()
            model.init_states()
            model.parallelize(mesh)
        """
        if self._parallelized:
            raise ValueError(
                f"{type(self).__name__} has already been parallelized. "
                "Module.parallelize() must be called at most once per instance."
            )
        self._parallelized = True

        sc = self.sharding_config
        if sc is None:
            for child in self.children():
                if isinstance(child, Module):
                    child.parallelize(tp_mesh)
            return

        if sc.local_map is not None:
            raise NotImplementedError("local_map will be added in M9")

        mesh_axis_names = tuple(tp_mesh.mesh_dim_names or ())
        if not mesh_axis_names:
            raise ValueError("DeviceMesh must have mesh_dim_names for parallelize()")

        self._shard_states(tp_mesh, sc, mesh_axis_names)
        _ = self._cache_pos_arg_names()
        unbound_forward = type(self).forward

        def forward_with_redistribution(*args, **kwargs):
            args, kwargs = self._redistribute_inputs(tp_mesh, mesh_axis_names, sc, args, kwargs)
            outputs = unbound_forward(self, *args, **kwargs)
            return self._redistribute_outputs(tp_mesh, mesh_axis_names, sc, outputs)

        self.forward = forward_with_redistribution  # type: ignore[method-assign]

        for child in self.children():
            if isinstance(child, Module):
                child.parallelize(tp_mesh)

    def _shard_states(
        self,
        tp_mesh: DeviceMesh,
        sharding_config: ShardingConfig,
        mesh_axis_names: tuple[str, ...],
    ) -> None:
        """Shard parameters listed in ``sharding_config.state_shardings``."""
        for path, named_placements in sharding_config.state_shardings.items():
            param = _get_attr_by_path(self, path)
            placements = resolve_placements(named_placements, mesh_axis_names)
            if isinstance(param, DTensor):
                if not _placements_equal(tuple(param.placements), placements):
                    raise ValueError(
                        f"{type(self).__name__}.{path} is already a DTensor with "
                        f"placements {param.placements}, but sharding_config expects "
                        f"{placements}."
                    )
                continue
            tensor = param.data if hasattr(param, "data") else param
            new_local = distribute_tensor(tensor, tp_mesh, placements)
            requires_grad = getattr(param, "requires_grad", True)
            _set_param_by_path(
                self,
                path,
                platform.Parameter(new_local, requires_grad=requires_grad),
            )

    def _redistribute_inputs(
        self,
        tp_mesh: DeviceMesh,
        mesh_axis_names: tuple[str, ...],
        sharding_config: ShardingConfig,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Redistribute forward inputs per ``in_src_shardings`` / ``in_dst_shardings``."""
        if (
            sharding_config.in_dst_shardings is None
            and sharding_config.in_src_shardings is None
        ):
            return args, kwargs

        pos_arg_names = [
            name for name in self._cache_pos_arg_names() if name not in kwargs
        ]
        new_kwargs = dict(zip(pos_arg_names, args))
        new_kwargs.update(kwargs)

        in_dst_shardings = sharding_config.in_dst_shardings or {}
        in_src_shardings = sharding_config.in_src_shardings or {}

        for name, value in new_kwargs.items():
            if not platform.is_tensor(value) and not isinstance(value, DTensor):
                continue
            src_named = in_src_shardings.get(name)
            dst_named = in_dst_shardings.get(name)
            if src_named is None and dst_named is None:
                continue

            if not isinstance(value, DTensor):
                if src_named is not None:
                    layout = resolve_placements(src_named, mesh_axis_names)
                    value = DTensor.from_local(value, tp_mesh, layout)
                elif dst_named is not None:
                    layout = resolve_placements(dst_named, mesh_axis_names)
                    value = DTensor.from_local(value, tp_mesh, layout)

            if dst_named is not None and isinstance(value, DTensor):
                desired = resolve_placements(dst_named, mesh_axis_names)
                if not _placements_equal(tuple(value.placements), desired):
                    value = value.redistribute(tp_mesh, desired)

            new_kwargs[name] = value

        new_args = tuple(new_kwargs.pop(name) for name in pos_arg_names)
        return new_args, new_kwargs

    def _redistribute_outputs(
        self,
        tp_mesh: DeviceMesh,
        mesh_axis_names: tuple[str, ...],
        sharding_config: ShardingConfig,
        outputs: Any,
    ) -> Any:
        """Redistribute forward outputs per ``out_dst_shardings``."""
        out_named = sharding_config.out_dst_shardings
        if out_named is None:
            return outputs
        if not isinstance(outputs, DTensor):
            return outputs
        desired = resolve_placements(out_named, mesh_axis_names)
        if not _placements_equal(tuple(outputs.placements), desired):
            outputs = outputs.redistribute(tp_mesh, desired)
        return outputs

    @classmethod
    def from_nn_module(cls, nn_module_cls: type) -> type["Module"]:
        """Wrap a platform ``nn.*`` class as a :class:`Module` subclass.

        Args:
            nn_module_cls: Source class (for example ``torch.nn.Conv2d``).

        Returns:
            New type mixing *nn_module_cls* and :class:`Module`. Results are
            cached per source class.

        Example::

            Conv2d = Module.from_nn_module(torch.nn.Conv2d)
            layer = Conv2d(3, 16, kernel_size=3)
            layer.init_states()
        """
        if nn_module_cls in _created_classes:
            return _created_classes[nn_module_cls]

        attrs: dict[str, Any] = {}
        if hasattr(nn_module_cls, "reset_parameters"):

            def _init_self_parameters(self: Any) -> None:
                self.reset_parameters()

            attrs["_init_self_parameters"] = _init_self_parameters

        name = f"Module({nn_module_cls.__name__})"
        new_cls = type(name, (nn_module_cls, Module), attrs)
        new_cls.__module__ = __name__
        new_cls.__qualname__ = name
        _created_classes[nn_module_cls] = new_cls
        return new_cls


def _get_torch_nn_container(kind: str) -> type:
    """Return ``torch.nn.{ModuleList,ModuleDict,Sequential}`` (requires PyTorch, lazy)."""
    # pylint: disable=C0415
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise NotImplementedError(
            f"{kind} container wrappers require PyTorch (torch.nn) in M1"
        ) from exc

    mapping = {
        "ModuleList": nn.ModuleList,
        "ModuleDict": nn.ModuleDict,
        "Sequential": nn.Sequential,
    }
    if kind not in mapping:
        raise ValueError(f"Unknown container kind: {kind}")
    return mapping[kind]


_LAZY_CONTAINER_NAMES = frozenset({"ModuleList", "ModuleDict", "Sequential"})


def __getattr__(name: str):
    """Lazy wrappers for ``ModuleList``, ``ModuleDict``, and ``Sequential``.

    Example::

        from hyper_parallel.dmodule import module as hp_module

        layers = hp_module.ModuleList([Linear.Config().build() for _ in range(4)])
    """
    if name in _LAZY_CONTAINER_NAMES:
        cls = Module.from_nn_module(_get_torch_nn_container(name))
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

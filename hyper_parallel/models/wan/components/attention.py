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
"""Wan attention family retaining the official Diffusers projections and processor."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def _build_attention(
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    backend: str | None = None,
) -> nn.Module:
    """Retain source parameters and let the official processor execute the selected backend."""
    del context
    if module is None:
        # Diffusers is optional for text-only training and registry discovery.
        implementation = import_module("diffusers.models.transformers.transformer_wan")
        options = dict(config)
        if options.get("processor") is None:
            options["processor"] = implementation.WanAttnProcessor()
        module = implementation.WanAttention(**options)
    if backend is not None:
        set_backend = getattr(module, "set_attention_backend", None)
        if not callable(set_backend):
            raise TypeError(f"{module_fqn}: Wan attention requires Diffusers set_attention_backend()")
        module.set_attention_backend(backend)
    return module


def _sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own complete Wan attention parameters within the supported DP/FSDP execution path."""
    del module_fqn, member
    placement = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(
        params={name: dict(placement) for name, _ in module.named_parameters()},
        in_src={"hidden_states": dict(placement), "encoder_hidden_states": dict(placement)},
        in_dst={"hidden_states": dict(placement), "encoder_hidden_states": dict(placement)},
        out_src={"output": dict(placement)}, out_dst={"output": dict(placement)}, is_boundary=True,
    )


def _validate(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Keep unsupported model-parallel axes from silently running local attention."""
    del config, module, context
    if any(member.axis_sizes[axis] > 1 for axis in ("tp", "cp", "ep", "pp")):
        raise NotImplementedError("Wan attention currently supports DP/FSDP only")


@module_replacement
def WanAttention(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    mesh: Any = None, backend: str | None = None,
) -> nn.Module:
    """Build the official self/cross-attention implementation with an optional Diffusers backend.

    Args:
        config: Official Diffusers WanAttention constructor options for direct construction.
        module: Existing source attention; mutually exclusive with config.
        module_fqn: Source boundary used in diagnostics.
        context: Shared module-family build context.
        mesh: Optional existing device mesh.
        backend: Diffusers processor backend; None retains its automatic selection.

    Returns:
        Official attention module with unchanged parameters and checkpoint keys.
    """
    return build_family(WanAttention, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, backend=backend)


register_module_family(ModuleFamily(
    name="wan_attention", baseline=WanAttention,
    variants={"diffusers": "hyper_parallel.models.wan.components.attention._build_attention"},
    default_variant="diffusers", sharding=_sharding, validate=_validate,
))

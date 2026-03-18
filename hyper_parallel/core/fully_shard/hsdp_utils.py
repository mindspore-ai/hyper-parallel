# Copyright 2025 Huawei Technologies Co., Ltd
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
"""HSDP optimizer shared level"""
from dataclasses import dataclass, field
from typing import Any, List
from enum import auto, Enum
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
platform = get_platform()



class HSDPConfigV2:
    """HSDPConfigV2 inspect by torch fully_shard"""

    def __init__(self,
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params=None,
        replicate_params=None,
    ):
        self.mesh = mesh
        self.reshard_after_forward = reshard_after_forward
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.ignored_params = ignored_params
        self.replicate_params = replicate_params
        self.reduce_dtype = self.mp_policy.reduce_dtype if self.mp_policy else None


class ShardedState(Enum):
    """
    Parameter shard state
    """
    SHARDED = auto()
    UNSHARDED = auto()


class FSDPSchedulerState(Enum):
    """
        Scheduler state:
                - PRE_FORWARD:
                  already run hook before forward.
                - FORWARD:
                  already run hook after forward.
                - PRE_BACKWARD:
                  already run hook before backward.
                - BACKWARD:
                  already run hook after backward.
    """
    PRE_FORWARD = auto()
    FORWARD = auto()
    PRE_BACKWARD = auto()
    BACKWARD = auto()


@dataclass
class ParamModuleInfo:
    """
    Tracks parameter ownership and supports shared weights in HSDP.

    This dataclass maintains the mapping between a parameter and its module(s),
    enabling parameter swapping during sharding/unsharding transitions. Shared
    weights are parameters referenced by multiple modules (e.g., tied embeddings).

    This class tracks all references to ensure proper parameter replacement during 
    sharding/unsharding operations.

    Attributes:
        module: The module that owns this parameter.
        param_name: Attribute name of the parameter in the module (e.g., "weight").
        shared_modules: List of other modules sharing this same parameter object.
        shared_param_names: Corresponding parameter names in shared_modules (aligned by index).
    """
    module: platform.Module
    param_name: str
    shared_modules: List[platform.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


def _named_parameters_with_duplicates(
    module: platform.Module, **kwargs: Any
) -> list[tuple[str, platform.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    if "remove_duplicate" in kwargs:
        raise AssertionError(
            "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
        )

    def get_named_parameters(module, **kwargs):
        if platform.platform_type == PlatformType.PYTORCH:
            return module.named_parameters(**kwargs)
        return module.parameters_and_names(expand=False)
    kwargs["remove_duplicate"] = False
    try:
        ret = list(get_named_parameters(module, **kwargs))
    except AssertionError:
        kwargs.pop("remove_duplicate")
        ret = list(get_named_parameters(module, **kwargs))
    return ret


def _get_param_module_infos(
    params: list[platform.Parameter], modules: tuple[platform.Module, ...]
) -> list['ParamModuleInfo']:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: dict[platform.Parameter, ParamModuleInfo] = {}

    def get_named_modules(module):
        if platform.platform_type == PlatformType.PYTORCH:
            return module.named_modules(remove_duplicate=False)
        return module.cells_and_names()

    for module in modules:
        for _, submodule in get_named_modules(module):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param in params_set:
                    if param not in param_to_module_info:
                        param_to_module_info[param] = ParamModuleInfo(
                            submodule, param_name
                        )
                    else:
                        param_to_module_info[param].shared_modules.append(submodule)
                        param_to_module_info[param].shared_param_names.append(
                            param_name
                        )
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {modules}")
    return [param_to_module_info[param] for param in params]

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
from typing import Any, List, Optional, Sequence, Tuple
from enum import auto, Enum
from torch import nn

class OptimizerLevel(Enum):
    """
        Optimizer level:
                - SHARD_OPT:
                  Splitting is performed on optimizer state.
                - SHARD_OPT_GRAD:
                  Splitting is performed on optimizer state, and gradients.
                - SHARD_OPT_GRAD_PARAM:
                  Splitting is performed on optimizer state, gradients and weights.
    """
    SHARD_OPT = auto()
    SHARD_OPT_GRAD = auto()
    SHARD_OPT_GRAD_PARAM = auto()

class GroupInfo:
    """
    GroupInfo
    """
    def __init__(self, group_name, group, rank_size):
        self.group_name = group_name
        self.group = group
        self.rank_size = rank_size


class HSDPConfigV2:
    """HSDPConfigV2 inspect by torch fully_shard"""

    def __init__(self, mesh, reshard_after_forward, shard_placement_fn, mp_policy, offload_policy, ignored_param,
                 reduce_dtype=None, comm_async=False, comm_fusion=False, bucket_size=-1):
        """
            HSDP config init method
            Args:
                shard_size: optimizer weight sharded size.
                threshold: minimum weight size to shard.
                requires_acc_grad: requires gradient accumulation.
                grad_scale: use grad_scale to scale grad.
                shard_level: optimizer shard level.
                use_eager_hook: use eager hook or graph hook to implement hsdp.
                reduce_dtype: set gradient reduce dtype.
                comm_async: use async communication op for grad reduction.
                comm_fusion: use communication op fusion to reduce the number of communication op.
                bucket_size: the size of comm fusion buffer.
        """
        self.mesh = mesh
        self.reshard_after_forward = reshard_after_forward
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.reduce_dtype = self.mp_policy.reduce_dtype if self.mp_policy else None
        # TODO: 下方属性待删除
        self.comm_async = False
        self.comm_fusion = False
        self.bucket_size = 9999
        self.grad_fusion = False

class ShardedState(Enum):
    """
    Parameter shard state
    """
    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
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
                - PRE_BACKWARD:
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
    module: nn.Module
    param_name: str
    shared_modules: List[nn.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


@dataclass
class ExtensionsData:
    """
    Stores metadata for custom all-gather extensions.

    This enables users to implement custom pre/post all-gather transforms
    by passing metadata between the two phases. The input sizes are saved
    to properly reshape the gathered outputs back to their original dimensions.

    Attributes:
        all_gather_metadata: Custom metadata passed from pre to post all-gather.
        all_gather_input_sizes: Original tensor shapes before flattening for all-gather.
    """
    all_gather_metadata: Optional[Any] = None
    all_gather_input_sizes: Sequence[Tuple[int, ...]] = ()

    def clear(self):
        """Reset all extension data to default values."""
        self.all_gather_metadata = None
        self.all_gather_input_sizes = ()


def _named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> list[tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    if "remove_duplicate" in kwargs:
        raise AssertionError(
            "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
        )
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError:
        kwargs.pop("remove_duplicate")
        ret = list(module.named_parameters(**kwargs))
    return ret

def _get_param_module_infos(
    params: list[nn.Parameter], modules: tuple[nn.Module, ...]
) -> list['ParamModuleInfo']:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: dict[nn.Parameter, ParamModuleInfo] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
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
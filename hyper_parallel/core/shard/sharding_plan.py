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
"""sharding plan"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class ShardingPlan:
    """
    Define a distribution scheme to partition a model across multiple computation nodes. 
    This class configures the `plan` for weight sharding based on layouts, while `input_plan` and `output_plan`
    control the data distribution of the model's entries and exits. Use 'return_local_tensor' to mark
    submodules that should output standard local Tensors instead of distributed ones.

    Attributes:
        plan (Dict[str, Any], optional): Mapping of parameter identifiers to 
            Layout instances for weight partitioning. Defaults to None.
        input_plan (Dict[str, Any], optional): Configuration for input data 
            layouts at the root or submodule level. Defaults to None.
        output_plan (Dict[str, Any], optional): Configuration for output data 
            layouts at the root or submodule level. Defaults to None.
        return_local_tensor (List[str], optional): Identifiers of modules 
            whose results should be converted to non-distributed Tensors. Defaults to None.

    Example:
        >>> from hyper_parallel import DeviceMesh, Layout, ShardingPlan, shard_module
        >>> mesh = DeviceMesh((2, 2), ("dp", "tp"))
        >>> sharding_plan = ShardingPlan(
        ...     plan={"weight": (Replicate(), Shard(1))},
        ...     input_plan={"input": (Shard(0), Replicate()), "relu.input": (Shard(0), Replicate())},
        ...     output_plan={"output": (Shard(0), Replicate()), "relu.output": (Shard(0), Replicate())},
        ...     return_local_tensor=["relu"]
        ... )
        >>> model = shard_module(model, mesh, sharding_plan)
    """
    plan: Optional[Dict[str, Any]] = None
    input_plan: Optional[Dict[str, Any]] = None
    output_plan: Optional[Dict[str, Any]] = None
    return_local_tensor: Optional[List[str]] = None

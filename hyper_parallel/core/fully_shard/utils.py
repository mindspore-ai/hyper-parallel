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
"""Common policy and mesh metadata for fully_shard APIs."""
from dataclasses import dataclass
from typing import Optional

from hyper_parallel.collectives.cc import get_group_local_rank
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform

platform = get_platform()


@dataclass
class MixedPrecisionPolicy:
    """
    Configures mixed precision training for HSDP.

    This policy controls data type casting during forward/backward computation
    and gradient reduction, enabling memory savings and potential speedups.

    Attributes:
        param_dtype: Data type for parameter computation. If None, uses original dtype.
        reduce_dtype: Data type for gradient reduction. If None, uses param_dtype.
        output_dtype: Data type for module outputs. If None, no casting applied.
    """
    param_dtype: Optional[platform.dtype] = None
    reduce_dtype: Optional[platform.dtype] = None
    output_dtype: Optional[platform.dtype] = None
    cast_forward_inputs: bool = True
    apply_grad_on_fp32_main_grad: bool = False


@dataclass
class OffloadPolicy:
    """
    Base class for offload policies.

    This represents no offloading and serves as the default policy.
    Subclass this to implement custom offload strategies.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    Offloads sharded parameters and gradients to CPU memory.

    When enabled, sharded parameters are kept on CPU and copied to device
    before all-gather. Gradients are copied back to CPU after backward.
    This reduces NPU memory usage at the cost of additional data transfers.

    Attributes:
        pin_memory: If True, pins CPU memory for faster H2D/D2H transfers
            and enables overlap with computation. Disable if CPU memory
            is constrained. (Default: True)
    """
    pin_memory: bool = True


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.mesh_shape[self.shard_mesh_dim]
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = get_group_local_rank(self.shard_process_group)


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.mesh_shape[self.replicate_mesh_dim]
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = get_group_local_rank(self.replicate_process_group)


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    # pylint: disable=W0246
    def __post_init__(self):
        # Calls `FSDPMeshInfo` -> `DDPMeshInfo` -> `DataParallelMeshInfo`
        super().__post_init__()

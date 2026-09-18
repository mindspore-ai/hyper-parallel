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
from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.platform import get_platform

platform = get_platform()
DType = platform.dtype

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
        cast_forward_inputs: Whether to cast floating-point forward inputs to ``param_dtype``.
        apply_grad_on_fp32_main_grad: Whether to accumulate reduced gradients into an
            FP32 main_grad buffer.
        custom_params: Per-parameter overrides, mapping a parameter's fully qualified
            name to its ``(param_dtype, reduce_dtype)`` pair. An override wins over
            ``param_dtype`` / ``reduce_dtype`` for that parameter only, which keeps
            a high-precision parameter inside its existing FSDP unit instead of
            forcing it into a separate one. Casting of inputs and outputs stays a
            module-level decision.

            Names follow the same enumeration the FSDP root uses: a key is the
            dotted path ``named_parameters()`` yields on the module passed to the
            outermost ``fully_shard`` call, so ``model.layers[0].mlp.gate.weight``
            is keyed as ``"layers.0.mlp.gate.weight"`` -- the root's own name is
            not part of the key. Keying by name rather than by parameter object
            lets one policy be reused by every ``fully_shard`` call in a model
            without retaining the pre-shard weights, and lets the override be
            resolved after wrapping, when names exist.

            Pass the mapping to **every** ``fully_shard`` call whose unit owns an
            overridden parameter: a policy is not inherited by child units, so
            configuring ``custom_params`` only on the outermost call leaves the
            nested units at the module-level dtypes. A key that matches no
            parameter of its unit is ignored.

    Note:
        An override only changes the dtype of the parameter itself -- the
        all-gathered weight, its reduction dtype and its communication bucket. It
        does not change how forward inputs are cast, which stays driven by
        ``cast_forward_inputs`` / ``param_dtype``, and it does not change the
        stored dtype of the sharded parameter. Both consequences are easy to
        miss: an FP32 override inside a unit that still casts inputs to the
        module dtype feeds mismatched operands to dtype-checked kernels, so such
        a unit must set ``cast_forward_inputs=False``; and a model loaded in a
        low-precision dtype keeps low-precision storage, so the override buys a
        higher-precision compute and communication path, not a higher-precision
        optimizer update.
    """
    param_dtype: Optional[DType] = None
    reduce_dtype: Optional[DType] = None
    output_dtype: Optional[DType] = None
    cast_forward_inputs: bool = True
    apply_grad_on_fp32_main_grad: bool = False
    custom_params: Optional[dict[str, tuple[Optional[DType], Optional[DType]]]] = None

    def get_param_dtypes(self, fqn: str) -> tuple[Optional[DType], Optional[DType]]:
        """Return the ``(param_dtype, reduce_dtype)`` pair that applies to ``fqn``.

        Falls back to the module-level pair when ``fqn`` has no override.

        Args:
            fqn: Fully qualified parameter name, as assigned by the FSDP root.

        Raises:
            ValueError: If the stored override is not a two-element tuple.
        """
        if self.custom_params is None or fqn not in self.custom_params:
            return self.param_dtype, self.reduce_dtype
        custom_dtypes = self.custom_params[fqn]
        if not isinstance(custom_dtypes, tuple) or len(custom_dtypes) != 2:
            raise ValueError(
                "MixedPrecisionPolicy custom_params values must be "
                "(param_dtype, reduce_dtype) tuples."
            )
        return custom_dtypes


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
class CommFusionPolicy():
    enable_comm_fusion: bool = False
    comm_fusion_zero_copy: bool = False


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


@dataclass(frozen=True)
class SourceShardMetaInfo:
    """Describe a parameter's source TP/EP layout before fully_shard."""

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    origin_is_dtensor: bool = False

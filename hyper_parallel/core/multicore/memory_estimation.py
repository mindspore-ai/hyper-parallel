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
"""Static peak-device-memory estimation for multicore mega kernels."""

import ctypes
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC


_MIB = 1024 * 1024
_DEFAULT_GMM_WORKSPACE_BYTES = 256 * _MIB
_DEFAULT_SYMMETRIC_HEAP_BYTES = 1024 * _MIB
_DEVICE_TENSOR_ALIGNMENT_BYTES = 32
_MEGA_MOE_CANN_WORKSPACE_BYTES = 95_421_440
_MEGA_MOE_GRAD_CANN_WORKSPACE_BYTES = 99_615_744
_EVENT_COUNTER_BYTES = 4096
_GMM_TILING_BYTES_PER_CORE = 2016
_SWIGLU_TILING_BYTES = 3920


def _align_up(size_bytes: int, alignment_bytes: int) -> int:
    """Align a device allocation size without touching a device backend."""
    return (size_bytes + alignment_bytes - 1) // alignment_bytes * alignment_bytes


class MemoryCategory(str, Enum):
    """Categories used to explain a mega-kernel memory estimate."""

    INPUT = "input"
    OUTPUT = "output"
    EXPLICIT_WORKSPACE = "explicit_workspace"
    RUNTIME = "runtime"
    IMPLICIT_WORKSPACE = "implicit_workspace"
    RESERVATION_OVERHEAD = "reservation_overhead"


@dataclass(frozen=True)
class MemoryComponent:
    """One independently explainable part of a memory estimate."""

    name: str
    category: MemoryCategory
    size_bytes: int
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("memory component name must not be empty")
        if self.size_bytes < 0:
            raise ValueError("memory component size_bytes must not be negative")


@dataclass(frozen=True)
class MegaKernelMemoryEstimate:
    """Static device-memory estimate returned by a mega-kernel estimator.

    `peak_bytes` includes explicitly modeled reservations, such as the
    symmetric-memory heap capacity. Generic framework allocator reservations,
    fragmentation, communication-domain state, and memory outside this kernel
    invocation are intentionally excluded.
    """

    kernel_name: str
    components: tuple[MemoryComponent, ...]

    @property
    def footprint_bytes(self) -> int:
        """Return logical tensor/workspace bytes, excluding reservation slack."""
        return sum(
            component.size_bytes
            for component in self.components
            if component.category != MemoryCategory.RESERVATION_OVERHEAD
        )

    @property
    def peak_bytes(self) -> int:
        """Return the predicted single-rank peak device-memory footprint."""
        return sum(component.size_bytes for component in self.components)

    @property
    def peak_mib(self) -> float:
        """Return ``peak_bytes`` converted to mebibytes."""
        return self.peak_bytes / _MIB

    def bytes_for(self, category: MemoryCategory) -> int:
        """Return the subtotal for one memory category.

        Args:
            category: Category to aggregate.

        Returns:
            Sum of component bytes in ``category``.
        """
        return sum(component.size_bytes for component in self.components if component.category == category)


@dataclass(frozen=True)
class MegaMoeMemorySpec:
    """Shape and buffer-capacity inputs for a forward ``mega_moe`` call.

    Args:
        tp: Tensor-parallel degree.
        ep: Expert-parallel degree.
        seq_size: Unexpanded input sequence size.
        expert_num: Global number of experts.
        top_k: Number of selected experts per token.
        hidden_size: Model hidden dimension.
        intermediate_size: FFN width after SwiGLU halves the up-projection.
        dtype_size: Bytes per activation and weight element.
        gmm_workspace_bytes: Capacity of the explicit GMM workspace tensor.
        num_cube_cores: Cube cores represented in each GMM tiling tensor.
        symmetric_heap_bytes: Per-rank symmetric-memory heap reserved at initialization.
    """

    tp: int
    ep: int
    seq_size: int
    expert_num: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    dtype_size: int = 2
    gmm_workspace_bytes: int = _DEFAULT_GMM_WORKSPACE_BYTES
    num_cube_cores: int = 24
    symmetric_heap_bytes: int = _DEFAULT_SYMMETRIC_HEAP_BYTES

    def __post_init__(self) -> None:
        positive_fields = (
            "tp", "ep", "seq_size", "expert_num", "top_k", "hidden_size",
            "intermediate_size", "dtype_size", "num_cube_cores",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
        if not isinstance(self.gmm_workspace_bytes, int) or isinstance(self.gmm_workspace_bytes, bool) \
                or self.gmm_workspace_bytes < 0:
            raise ValueError(
                f"gmm_workspace_bytes must be a non-negative integer, got {self.gmm_workspace_bytes!r}"
            )
        if not isinstance(self.symmetric_heap_bytes, int) or isinstance(self.symmetric_heap_bytes, bool) \
                or self.symmetric_heap_bytes < 0:
            raise ValueError(
                f"symmetric_heap_bytes must be a non-negative integer, got {self.symmetric_heap_bytes!r}"
            )
        if self.expert_num % self.ep != 0:
            raise ValueError(
                f"expert_num must be divisible by ep, got expert_num={self.expert_num}, ep={self.ep}"
            )
        if self.seq_size * self.top_k % self.tp != 0:
            raise ValueError(
                f"seq_size * top_k must be divisible by tp, got seq_size={self.seq_size}, "
                f"top_k={self.top_k}, tp={self.tp}"
            )

    @property
    def local_expert_num(self) -> int:
        """Return the number of experts whose weights reside on one rank."""
        return self.expert_num // self.ep

    @property
    def per_rank_token_num(self) -> int:
        """Return the routed-token buffer capacity on one rank."""
        return self.seq_size * self.top_k // self.tp


@dataclass(frozen=True)
class MegaMoeGradMemorySpec(MegaMoeMemorySpec):
    """Shape and buffer-capacity inputs for a backward ``mega_moe_grad`` call.

    Args:
        swiglu_grad_workspace_bytes: Capacity of the explicit SwiGLU-grad
            workspace tensor. Inherited fields match ``MegaMoeMemorySpec``.
    """

    swiglu_grad_workspace_bytes: int = 16 * _MIB

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.swiglu_grad_workspace_bytes
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "swiglu_grad_workspace_bytes must be a non-negative integer, "
                f"got {value!r}"
            )


MemoryEstimator = Callable[[Any], MegaKernelMemoryEstimate]
_ESTIMATORS: dict[str, MemoryEstimator] = {}


def register_mega_kernel_memory_estimator(
        kernel_name: str, estimator: MemoryEstimator, *, replace: bool = False) -> None:
    """Register a static memory estimator for a mega kernel.

    Args:
        kernel_name: Stable public name used by the generic estimate API.
        estimator: Callable accepting a kernel-specific specification object.
        replace: Whether an existing registration may be replaced.

    Raises:
        ValueError: If the name is invalid or is already registered.
        TypeError: If ``estimator`` is not callable.
    """
    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError("kernel_name must be a non-empty string")
    if not callable(estimator):
        raise TypeError(f"estimator must be callable, got {type(estimator).__name__}")
    normalized_name = kernel_name.strip().lower()
    if normalized_name in _ESTIMATORS and not replace:
        raise ValueError(f"memory estimator for mega kernel {normalized_name!r} is already registered")
    _ESTIMATORS[normalized_name] = estimator


def estimate_mega_kernel_peak_memory(kernel_name: str, spec: Any) -> MegaKernelMemoryEstimate:
    """Estimate peak device memory without constructing tensors or running a kernel.

    Args:
        kernel_name: Registered mega-kernel name.
        spec: Kernel-specific, shape-only specification object.

    Returns:
        A categorized single-rank memory estimate.

    Raises:
        ValueError: If ``kernel_name`` is invalid or has no registered estimator.
        TypeError: If the registered estimator returns the wrong result type.
    """
    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError("kernel_name must be a non-empty string")
    normalized_name = kernel_name.strip().lower()
    try:
        estimator = _ESTIMATORS[normalized_name]
    except KeyError as exc:
        available = ", ".join(sorted(_ESTIMATORS)) or "none"
        raise ValueError(
            f"no memory estimator registered for mega kernel {normalized_name!r}; available: {available}"
        ) from exc
    estimate = estimator(spec)
    if not isinstance(estimate, MegaKernelMemoryEstimate):
        raise TypeError(
            f"memory estimator for {normalized_name!r} must return MegaKernelMemoryEstimate, "
            f"got {type(estimate).__name__}"
        )
    return estimate


def _estimate_mega_moe(spec: Any) -> MegaKernelMemoryEstimate:
    if not isinstance(spec, MegaMoeMemorySpec):
        raise TypeError(f"mega_moe memory estimator requires MegaMoeMemorySpec, got {type(spec).__name__}")

    token_num = spec.per_rank_token_num
    local_expert_num = spec.local_expert_num
    hidden_size = spec.hidden_size
    intermediate_size = spec.intermediate_size
    dtype_size = spec.dtype_size

    activation = token_num * hidden_size * dtype_size
    up_projection = token_num * intermediate_size * 2 * dtype_size
    swiglu = token_num * intermediate_size * dtype_size
    up_weight = local_expert_num * hidden_size * intermediate_size * 2 * dtype_size
    down_weight = local_expert_num * intermediate_size * hidden_size * dtype_size
    routing = spec.expert_num * (4 * 8 + 2 * 4) + local_expert_num * 2 * 8
    tiling = (
        2 * _GMM_TILING_BYTES_PER_CORE * spec.num_cube_cores
        + _align_up(_SWIGLU_TILING_BYTES, _DEVICE_TENSOR_ALIGNMENT_BYTES)
    )
    runtime_config = _align_up(ctypes.sizeof(RuntimeConfigC), _DEVICE_TENSOR_ALIGNMENT_BYTES)

    symmetric_tensor_bytes = activation * 2 + _EVENT_COUNTER_BYTES
    if spec.symmetric_heap_bytes < symmetric_tensor_bytes:
        raise ValueError(
            f"symmetric_heap_bytes is smaller than required symmetric tensors: "
            f"heap={spec.symmetric_heap_bytes}, required={symmetric_tensor_bytes}"
        )
    components = (
        MemoryComponent("dispatch_src", MemoryCategory.INPUT, activation),
        MemoryComponent("expert_weights", MemoryCategory.INPUT, up_weight + down_weight),
        MemoryComponent(
            "routing_metadata", MemoryCategory.INPUT, routing,
            "Four int64 offsets, two int32 sizes, and two local-expert int64 group lists.",
        ),
        MemoryComponent("dispatch_target", MemoryCategory.OUTPUT, activation),
        MemoryComponent("up_proj_y", MemoryCategory.OUTPUT, up_projection),
        MemoryComponent("swiglu_out", MemoryCategory.OUTPUT, swiglu),
        MemoryComponent("down_proj_y", MemoryCategory.OUTPUT, activation),
        MemoryComponent("combine_target", MemoryCategory.OUTPUT, activation),
        MemoryComponent(
            "symmetric_heap_slack", MemoryCategory.RESERVATION_OVERHEAD,
            spec.symmetric_heap_bytes - symmetric_tensor_bytes,
            "Reserved symmetric heap capacity not occupied by symmetric outputs and event counters.",
        ),
        MemoryComponent("gmm_workspace", MemoryCategory.EXPLICIT_WORKSPACE, spec.gmm_workspace_bytes),
        MemoryComponent("runtime_config", MemoryCategory.RUNTIME, runtime_config),
        MemoryComponent("operator_tiling", MemoryCategory.RUNTIME, tiling),
        MemoryComponent("all_event_counters", MemoryCategory.RUNTIME, _EVENT_COUNTER_BYTES),
        MemoryComponent(
            "cann_workspace", MemoryCategory.IMPLICIT_WORKSPACE, _MEGA_MOE_CANN_WORKSPACE_BYTES,
            "ACLNN executor allocation after aligning the 95,420,928-byte tiling request.",
        ),
    )
    return MegaKernelMemoryEstimate(kernel_name="mega_moe", components=components)


def estimate_mega_moe_peak_memory(spec: MegaMoeMemorySpec) -> MegaKernelMemoryEstimate:
    """Estimate the forward ``mega_moe`` single-rank peak device memory.

    This function only performs integer arithmetic. It does not allocate framework
    tensors, initialize communication, query a device, or execute a kernel.

    Args:
        spec: Forward topology, shape, dtype, and workspace capacities.

    Returns:
        Categorized memory estimate whose ``peak_bytes`` is the predicted total.
    """
    return estimate_mega_kernel_peak_memory("mega_moe", spec)


def _estimate_mega_moe_grad(spec: Any) -> MegaKernelMemoryEstimate:
    if not isinstance(spec, MegaMoeGradMemorySpec):
        raise TypeError(
            "mega_moe_grad memory estimator requires MegaMoeGradMemorySpec, "
            f"got {type(spec).__name__}"
        )

    token_num = spec.per_rank_token_num
    local_expert_num = spec.local_expert_num
    hidden_size = spec.hidden_size
    intermediate_size = spec.intermediate_size
    dtype_size = spec.dtype_size

    activation = token_num * hidden_size * dtype_size
    intermediate_activation = token_num * intermediate_size * dtype_size
    gate_activation = token_num * intermediate_size * 2 * dtype_size
    w1 = local_expert_num * hidden_size * intermediate_size * 2 * dtype_size
    w2 = local_expert_num * intermediate_size * hidden_size * dtype_size
    routing = spec.expert_num * (4 * 8 + 2 * 4) + local_expert_num * 8
    tiling = (
        4 * _GMM_TILING_BYTES_PER_CORE * spec.num_cube_cores
        + _align_up(_SWIGLU_TILING_BYTES, _DEVICE_TENSOR_ALIGNMENT_BYTES)
    )
    runtime_config = _align_up(ctypes.sizeof(RuntimeConfigC), _DEVICE_TENSOR_ALIGNMENT_BYTES)

    symmetric_tensor_bytes = activation * 2 + _EVENT_COUNTER_BYTES
    if spec.symmetric_heap_bytes < symmetric_tensor_bytes:
        raise ValueError(
            f"symmetric_heap_bytes is smaller than required symmetric tensors: "
            f"heap={spec.symmetric_heap_bytes}, required={symmetric_tensor_bytes}"
        )

    components = (
        MemoryComponent("dy", MemoryCategory.INPUT, activation),
        MemoryComponent(
            "saved_activations", MemoryCategory.INPUT,
            intermediate_activation + gate_activation + activation,
            "Forward-saved hidden, gate, and permute_out tensors.",
        ),
        MemoryComponent("expert_weights", MemoryCategory.INPUT, w1 + w2),
        MemoryComponent(
            "routing_metadata", MemoryCategory.INPUT, routing,
            "Four int64 offsets, two int32 sizes, and one local-expert int64 group list.",
        ),
        MemoryComponent("dispatch_target", MemoryCategory.OUTPUT, activation),
        MemoryComponent("hidden_dw", MemoryCategory.OUTPUT, w2),
        MemoryComponent("act_grad_y", MemoryCategory.OUTPUT, intermediate_activation),
        MemoryComponent("grad_gate", MemoryCategory.OUTPUT, gate_activation),
        MemoryComponent("gate_dx", MemoryCategory.OUTPUT, activation),
        MemoryComponent("grad_x", MemoryCategory.OUTPUT, activation),
        MemoryComponent("gate_dw", MemoryCategory.OUTPUT, w1),
        MemoryComponent(
            "symmetric_heap_slack", MemoryCategory.RESERVATION_OVERHEAD,
            spec.symmetric_heap_bytes - symmetric_tensor_bytes,
            "Reserved symmetric heap capacity not occupied by symmetric outputs and event counters.",
        ),
        MemoryComponent("gmm_workspace", MemoryCategory.EXPLICIT_WORKSPACE, spec.gmm_workspace_bytes),
        MemoryComponent(
            "swiglu_grad_workspace", MemoryCategory.EXPLICIT_WORKSPACE,
            spec.swiglu_grad_workspace_bytes,
        ),
        MemoryComponent("runtime_config", MemoryCategory.RUNTIME, runtime_config),
        MemoryComponent("operator_tiling", MemoryCategory.RUNTIME, tiling),
        MemoryComponent("all_event_counters", MemoryCategory.RUNTIME, _EVENT_COUNTER_BYTES),
        MemoryComponent(
            "cann_workspace", MemoryCategory.IMPLICIT_WORKSPACE,
            _MEGA_MOE_GRAD_CANN_WORKSPACE_BYTES,
            "ACLNN executor allocation observed on the MegaMoeGrad planning path.",
        ),
    )
    return MegaKernelMemoryEstimate(kernel_name="mega_moe_grad", components=components)


def estimate_mega_moe_grad_peak_memory(
        spec: MegaMoeGradMemorySpec) -> MegaKernelMemoryEstimate:
    """Estimate the backward ``mega_moe_grad`` single-rank peak device memory.

    This function performs shape-only integer arithmetic and never initializes a
    framework backend or launches a kernel.

    Args:
        spec: Backward topology, shape, dtype, and workspace capacities.

    Returns:
        Categorized memory estimate whose ``peak_bytes`` is the predicted total.
    """
    return estimate_mega_kernel_peak_memory("mega_moe_grad", spec)


register_mega_kernel_memory_estimator("mega_moe", _estimate_mega_moe)
register_mega_kernel_memory_estimator("mega_moe_grad", _estimate_mega_moe_grad)

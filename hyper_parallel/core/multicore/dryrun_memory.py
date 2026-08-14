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
"""Result model for dynamic mega-kernel dryrun memory measurement."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MegaKernelMemoryUsage:
    """Peak device-memory usage observed during a mega-kernel dryrun.

    ``allocator_peak_bytes`` is directly comparable with
    ``mindspore.runtime.max_memory_allocated()``. ``peak_bytes`` additionally
    accounts for external reservations, such as an ACLSHMEM symmetric heap,
    while subtracting logical tensors already included in the allocator peak.

    Args:
        kernel_name: Logical mega-kernel name.
        allocated_before_bytes: Allocator bytes live before the measured call.
        allocated_after_bytes: Allocator bytes live after the measured call.
        allocator_peak_bytes: Maximum allocator bytes in the measured window.
        external_reserved_bytes: Memory reserved outside the framework allocator.
        external_logical_bytes: Logical tensor bytes backed by that reservation.
    """

    kernel_name: str
    allocated_before_bytes: int
    allocated_after_bytes: int
    allocator_peak_bytes: int
    external_reserved_bytes: int = 0
    external_logical_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_name, str) or not self.kernel_name.strip():
            raise ValueError("kernel_name must be a non-empty string")
        for field_name in (
            "allocated_before_bytes",
            "allocated_after_bytes",
            "allocator_peak_bytes",
            "external_reserved_bytes",
            "external_logical_bytes",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.external_logical_bytes > self.external_reserved_bytes:
            raise ValueError(
                "external_logical_bytes cannot exceed external_reserved_bytes"
            )

    @property
    def external_reservation_overhead_bytes(self) -> int:
        """Return reservation bytes not already represented by logical tensors."""
        return self.external_reserved_bytes - self.external_logical_bytes

    @property
    def peak_bytes(self) -> int:
        """Return total peak bytes including external reservation overhead."""
        return self.allocator_peak_bytes + self.external_reservation_overhead_bytes

    @property
    def allocator_peak_mib(self) -> float:
        """Return allocator peak in MiB."""
        return self.allocator_peak_bytes / (1024 * 1024)

    @property
    def peak_mib(self) -> float:
        """Return total peak in MiB."""
        return self.peak_bytes / (1024 * 1024)


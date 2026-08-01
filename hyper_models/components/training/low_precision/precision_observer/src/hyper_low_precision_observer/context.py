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
"""In-process state for precision error collection."""

from __future__ import annotations

from typing import Mapping

import torch

from hyper_low_precision_observer.config import (
    DebugOverlay,
)
from hyper_low_precision_observer.metrics import (
    DeviceErrorMetrics,
    ErrorMetrics,
    materialize_device_metrics,
)


def metric_key(kind: str, module_fqn: str, gemm_role: str, operand_role: str) -> str:
    """Build the stable artifact key for one observed operand."""
    if kind != "quantization":
        raise ValueError(f"Unknown precision error kind '{kind}'")
    return "/".join((kind, module_fqn, gemm_role, operand_role))


class PrecisionContext:
    """Track the current training position and mergeable error moments."""

    def __init__(
        self,
        *,
        debug_overlay: DebugOverlay | None = None,
    ) -> None:
        """Initialize an empty process-local collection context."""
        self.debug_overlay = debug_overlay
        self.step = 0
        self._metrics: dict[str, DeviceErrorMetrics] = {}

    def set_position(self, step: int) -> None:
        """Set the current non-negative training step."""
        if step < 0:
            raise ValueError("step must be non-negative")
        self.step = int(step)

    def active(self) -> bool:
        """Return whether the current step has active observations."""
        return self.debug_overlay is not None and self.debug_overlay.active(self.step)

    def should_measure_quantization(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
    ) -> bool:
        """Return whether one quantization operand is selected now."""
        return (
            self.debug_overlay is not None
            and self.debug_overlay.quantization_error(
                module_fqn,
                gemm_role,
                operand_role,
                self.step,
            )
        )

    def record(
        self,
        kind: str,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
        *,
        clipped_mask: torch.Tensor | None = None,
        underflow_mask: torch.Tensor | None = None,
        quantized: torch.Tensor | None = None,
        quantized_max: float | None = None,
        accumulate: bool = True,
    ) -> None:
        """Record device moments, optionally replacing an earlier observation."""
        key = metric_key(kind, module_fqn, gemm_role, operand_role)
        metrics = ErrorMetrics.compare_device(
            baseline,
            candidate,
            clipped_mask=clipped_mask,
            underflow_mask=underflow_mask,
            quantized=quantized,
            quantized_max=quantized_max,
        )
        current = self._metrics.get(key)
        if current is None or not accumulate:
            self._metrics[key] = metrics
        else:
            current.merge_(metrics)

    def error_moments(self) -> Mapping[str, ErrorMetrics]:
        """Materialize all accumulated device moments on the host."""
        return materialize_device_metrics(self._metrics)

    def clear(self) -> None:
        """Release all accumulated device moments."""
        self._metrics.clear()

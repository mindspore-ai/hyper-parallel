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
"""Failure-isolated entry point for optional precision diagnostics."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch
from torch import nn

try:
    from hyper_low_precision_observer import (
        DebugOutput,
        DebugOverlay,
        PrecisionContext,
        PrecisionDebugProgram,
    )
except ImportError as error:
    raise ImportError(
        "Precision diagnostics requires the optional "
        "'hyper-low-precision-observer' wheel. Install Hyper-Parallel with "
        "the [precision-debug] extra or install that wheel alongside it."
    ) from error

if TYPE_CHECKING:
    from hyper_low_precision_observer.report import (
        PrecisionReport,
    )


logger = logging.getLogger(__name__)

_SUPPORTED_GEMM_ROLES = ("fprop",)


def resolve_debug_overlay(
    program: PrecisionDebugProgram,
    module_fqns: tuple[str, ...],
) -> DebugOverlay:
    """Compile a generic program for the GEMM roles HP currently emits."""
    observations = {}
    for module_fqn in module_fqns:
        for section in program.sections:
            for gemm_role in _SUPPORTED_GEMM_ROLES:
                if section.select.matches(module_fqn, gemm_role):
                    observations.setdefault((module_fqn, gemm_role), []).append(
                        section.observe
                    )
    return DebugOverlay(
        observations={key: tuple(value) for key, value in observations.items()},
        output=program.output,
    )


class PrecisionDebugSession:
    """Prevent optional diagnostic failures from changing training behavior.

    Native quantizers may call ``record_quantization`` and the training loop
    may call ``flush``. A failure in either path clears partial moments and
    permanently disables this session, but never escapes into the model's
    forward/backward or optimizer step.
    """

    def __init__(self, context: PrecisionContext, report: "PrecisionReport") -> None:
        self.context = context
        self.report = report
        self.enabled = True
        self.failure: Exception | None = None
        self._pause_depth = 0

    def set_step(self, step: int) -> None:
        """Set the sampled training step while retaining failure isolation."""
        self._run("set_step", self.context.set_position, step)

    def should_observe_quantization(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
    ) -> bool:
        """Return whether a native quantization event should be sampled now.

        This is the public preflight API for native integrations. It performs
        no candidate reconstruction or metric accumulation, and returns false
        after the session has been disabled by a diagnostic failure.
        """
        return bool(
            self._run(
                "should_observe_quantization",
                self.context.should_measure_quantization,
                module_fqn,
                gemm_role,
                operand_role,
            )
        )

    @contextmanager
    def paused(self):
        """Temporarily suppress observations, for example during validation."""
        self._pause_depth += 1
        try:
            yield self
        finally:
            self._pause_depth -= 1

    def record_quantization(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Safely accumulate an observation selected by the debug overlay."""
        self._run(
            "record_quantization",
            self._record_quantization,
            module_fqn,
            gemm_role,
            operand_role,
            baseline,
            candidate,
            **kwargs,
        )

    def observe_quantization(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
        baseline: torch.Tensor,
        candidate_factory: Callable[[], torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Build the candidate only when the selector and schedule match."""
        self._run(
            "observe_quantization",
            self._observe_quantization,
            module_fqn,
            gemm_role,
            operand_role,
            baseline,
            candidate_factory,
            **kwargs,
        )

    def _record_quantization(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Record only operands enabled by the matching selector and schedule."""
        if not self.should_observe_quantization(
            module_fqn,
            gemm_role,
            operand_role,
        ):
            return
        self.context.record(
            "quantization",
            module_fqn,
            gemm_role,
            operand_role,
            baseline,
            candidate,
            **kwargs,
        )

    def _observe_quantization(
        self,
        module_fqn: str,
        gemm_role: str,
        operand_role: str,
        baseline: torch.Tensor,
        candidate_factory: Callable[[], torch.Tensor],
        **kwargs: Any,
    ) -> None:
        if not self.should_observe_quantization(
            module_fqn,
            gemm_role,
            operand_role,
        ):
            return
        with torch.no_grad():
            candidate = candidate_factory()
        self.context.record(
            "quantization",
            module_fqn,
            gemm_role,
            operand_role,
            baseline,
            candidate,
            **kwargs,
        )

    def flush(self) -> dict[str, Any] | None:
        """Safely persist the current sampled diagnostic window."""
        return self._run("flush", self.report.flush, self.context)

    def _run(self, operation: str, callback, *args, **kwargs):
        if not self.enabled or self._pause_depth:
            return None
        try:
            return callback(*args, **kwargs)
        except Exception as error:  # Debug must not affect training semantics.
            self.enabled = False
            self.failure = error
            # Cleanup is diagnostic-only as well. A broken cleanup path must
            # not turn an optional observation failure into a training error.
            try:
                self.context.clear()
            except Exception:
                logger.exception(
                    "Failed to clear precision debug state after %s failed",
                    operation,
                )
            logger.exception(
                "Disabled precision debug after %s failed: %s",
                operation,
                error,
            )
            return None


def install_precision_debug(
    model: nn.Module,
    program: PrecisionDebugProgram | Mapping[str, Any],
    *,
    output_root: str | Path,
) -> PrecisionDebugSession:
    """Install precision diagnostics on converted HP native Linear modules.

    The trainer owns ``set_step()`` and ``flush()``. This function binds the
    returned session only to converted ``NpuQuantLinear`` modules matched by
    the selectors, so unselected native modules have no diagnostic overhead.
    """
    from hyper_models.components.training.low_precision.modules import (
        NpuQuantLinear,
    )

    modules = tuple(
        module
        for module in model.modules()
        if isinstance(module, NpuQuantLinear)
    )
    if not modules:
        raise ValueError(
            "precision_debug requires at least one converted NpuQuantLinear"
        )
    parsed_program = PrecisionDebugProgram.from_mapping(program)
    parsed_program = replace(
        parsed_program,
        output=DebugOutput(root_dir=str(output_root)),
    )
    overlay = resolve_debug_overlay(
        parsed_program,
        tuple(module.fqn for module in modules),
    )
    if not overlay.observations:
        raise ValueError(
            "precision_debug selectors matched no converted NpuQuantLinear "
            "for HP's supported GEMM roles ['fprop']; check "
            "select.module_name_regex and select.gemm_roles"
        )
    context = PrecisionContext(debug_overlay=overlay)
    # Reporting is imported only after an explicit monitoring installation.
    from hyper_low_precision_observer.report import (
        PrecisionReport,
    )

    session = PrecisionDebugSession(
        context,
        PrecisionReport(overlay.output),
    )
    selected_fqns = {module_fqn for module_fqn, _ in overlay.observations}
    for module in modules:
        if module.fqn in selected_fqns:
            module.set_precision_debug_session(session)
    # Trainer integration retrieves this non-module handle after model build.
    model._precision_debug_session = session
    return session


def find_precision_debug_session(model: nn.Module) -> PrecisionDebugSession | None:
    """Find the installed session after FSDP or compile wraps the model."""
    sessions = {
        id(session): session
        for module in model.modules()
        if (session := getattr(module, "_precision_debug_session", None)) is not None
    }
    if len(sessions) > 1:
        raise ValueError("Model contains multiple precision debug sessions")
    return next(iter(sessions.values()), None)

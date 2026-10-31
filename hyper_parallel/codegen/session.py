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
"""Stateful collector for immutable codegen build snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from hyper_parallel.codegen.plan.freeze import freeze_plan, freeze_plan_application
from hyper_parallel.codegen.snapshot import (
    CodegenSnapshot,
    CodegenStatus,
    FailureSnapshot,
    FrozenMapping,
    ReplacementSnapshot,
    freeze_json,
)


class CodegenBuildSession:
    """Collect native build observations without influencing the build."""

    def __init__(self) -> None:
        """Initialize an empty collection session."""
        self._status = CodegenStatus.COLLECTING
        self._source = FrozenMapping()
        self._replacements = ReplacementSnapshot()
        self._plan: Optional[FrozenMapping] = None
        self._plan_application: Optional[FrozenMapping] = None
        self._diagnostics = FrozenMapping()
        self._reason: Optional[str] = None
        self._failure: Optional[FailureSnapshot] = None
        self._source_recorded = False
        self._replacements_recorded = False
        self._diagnostics_recorded = False
        self._finished = False
        self._snapshot: Optional[CodegenSnapshot] = None

    @property
    def status(self) -> CodegenStatus:
        """Return the current session status."""
        return self._status

    def record_source(self, source: Mapping[str, Any]) -> None:
        """Record source and model identity once while collecting."""
        self._require_open(CodegenStatus.COLLECTING)
        if self._source_recorded:
            raise RuntimeError("codegen source has already been recorded")
        self._source = self._as_mapping(source, "source")
        self._source_recorded = True

    def record_replacements(
        self,
        *,
        declarations: Sequence[Mapping[str, Any]] = (),
        applications: Sequence[Mapping[str, Any]] = (),
        unmatched: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Record native replacement declarations and observed applications."""
        self._require_open(CodegenStatus.COLLECTING)
        if self._replacements_recorded:
            raise RuntimeError("codegen replacements have already been recorded")
        self._replacements = ReplacementSnapshot.from_data(
            declarations=declarations,
            applications=applications,
            unmatched=unmatched,
        )
        self._replacements_recorded = True

    def record_plan(self, plan: Mapping[str, Any]) -> None:
        """Record the native planner result and enter the derived state."""
        self._require_open(CodegenStatus.COLLECTING)
        if not self._source_recorded:
            raise RuntimeError("record_source() must run before record_plan()")
        self._plan = self._as_mapping(plan, "plan")
        self._status = CodegenStatus.DERIVED

    def record_plan_application(self, application: Mapping[str, Any]) -> None:
        """Record how the native plan resolved against the live model."""
        self._require_open(CodegenStatus.DERIVED)
        if self._plan_application is not None:
            raise RuntimeError("codegen plan application has already been recorded")
        self._plan_application = self._as_mapping(application, "plan_application")

    def observe_plan(self, plan: Any, model: Any, mesh: Any) -> None:
        """Freeze a native planner result without exposing codegen to the builder."""
        frozen_plan = freeze_plan(plan, model).to_dict()
        frozen_plan["parallel_dims"] = {
            "dp_size": getattr(mesh, "dp_size", 1),
            "tp_size": mesh.tp_size,
            "cp_size": mesh.cp_size,
            "pp_size": getattr(mesh, "pp_size", 1),
            "ep_size": mesh.ep_size,
            "sequence_parallel": mesh.sequence_parallel,
            "loss_parallel": mesh.loss_parallel,
        }
        self.record_plan(frozen_plan)

    def observe_plan_application(
        self,
        plan: Any,
        source_shard_info: Any,
        *,
        validate_mode: bool,
    ) -> None:
        """Freeze the native plan application behind the observation seam."""
        self.record_plan_application(
            freeze_plan_application(
                plan,
                source_shard_info,
                validate_mode=validate_mode,
            )
        )

    def record_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        """Record optional JSON-safe counters and audit facts once."""
        self._require_open()
        if self._diagnostics_recorded:
            raise RuntimeError("codegen diagnostics have already been recorded")
        self._diagnostics = self._as_mapping(diagnostics, "diagnostics")
        self._diagnostics_recorded = True

    def mark_not_required(self, reason: str) -> None:
        """Finish collection when the native build has no sharding plan."""
        self._require_open(CodegenStatus.COLLECTING)
        if not self._source_recorded:
            raise RuntimeError("record_source() must run before mark_not_required()")
        self._reason = self._require_reason(reason)
        self._status = CodegenStatus.NOT_REQUIRED

    def mark_generating(self) -> None:
        """Mark the future side-effecting generation boundary."""
        self._require_open(CodegenStatus.DERIVED)
        if self._plan_application is None:
            raise RuntimeError("record_plan_application() must run before generation")
        self._status = CodegenStatus.GENERATING

    def mark_generated(self) -> None:
        """Mark successful generation."""
        self._require_open(CodegenStatus.GENERATING)
        self._status = CodegenStatus.GENERATED

    def mark_skipped(self, reason: str) -> None:
        """Mark generation as intentionally skipped with a stable reason."""
        self._require_open(CodegenStatus.COLLECTING, CodegenStatus.DERIVED)
        if not self._source_recorded:
            raise RuntimeError("record_source() must run before mark_skipped()")
        self._reason = self._require_reason(reason)
        self._status = CodegenStatus.SKIPPED

    def mark_failed(
        self,
        stage: str,
        reason: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a codegen-owned failure without retaining an exception."""
        self._require_open(
            CodegenStatus.COLLECTING,
            CodegenStatus.DERIVED,
            CodegenStatus.GENERATING,
            CodegenStatus.NOT_REQUIRED,
        )
        self._failure = FailureSnapshot.from_data(stage, reason, details)
        self._status = CodegenStatus.FAILED

    def finish(
        self,
        generator: Optional[Callable[[CodegenSnapshot], Any]] = None,
    ) -> CodegenSnapshot:
        """Seal the session, optionally publishing a derived snapshot.

        ``generator`` is invoked only for a fully derived native build. Its
        failure is recorded as codegen-owned state and deliberately does not
        escape into the caller's already successful model build.

        Args:
            generator: Optional side-effecting publisher that consumes the
                immutable derived snapshot.

        Returns:
            The same snapshot instance on repeated calls.

        Raises:
            RuntimeError: If collection or generation is incomplete.
        """
        if self._finished:
            if self._snapshot is None:
                raise RuntimeError("finished codegen session has no snapshot")
            return self._snapshot
        if self._status == CodegenStatus.COLLECTING:
            raise RuntimeError("codegen session is still collecting")
        if self._status == CodegenStatus.DERIVED and self._plan_application is None:
            raise RuntimeError("derived codegen session is missing plan application")
        if self._status == CodegenStatus.GENERATING:
            raise RuntimeError("codegen session is still generating")
        if self._status == CodegenStatus.DERIVED and generator is not None:
            publish_snapshot = self._build_snapshot(CodegenStatus.DERIVED)
            self.mark_generating()
            try:
                generator(publish_snapshot)
            except Exception as exc:  # Generation must not invalidate native build.
                self.mark_failed(
                    "generation",
                    str(exc) or type(exc).__name__,
                    {"exception_type": type(exc).__name__},
                )
            else:
                self.mark_generated()
        elif self._status == CodegenStatus.NOT_REQUIRED and generator is not None:
            publish_snapshot = self._build_snapshot(CodegenStatus.NOT_REQUIRED)
            try:
                generator(publish_snapshot)
            except Exception as exc:  # Publication must not invalidate native build.
                self.mark_failed(
                    "generation",
                    str(exc) or type(exc).__name__,
                    {"exception_type": type(exc).__name__},
                )
        self._snapshot = self._build_snapshot(self._status)
        self._finished = True
        return self._snapshot

    def _build_snapshot(self, status: CodegenStatus) -> CodegenSnapshot:
        """Build an immutable view without sealing the session."""
        return CodegenSnapshot(
            status=status,
            source=self._source,
            replacements=self._replacements,
            plan=self._plan,
            plan_application=self._plan_application,
            diagnostics=self._diagnostics,
            reason=self._reason,
            failure=self._failure,
        )

    def _require_open(self, *statuses: CodegenStatus) -> None:
        if self._finished:
            raise RuntimeError("codegen session is already finished")
        if statuses and self._status not in statuses:
            expected = ", ".join(status.value for status in statuses)
            raise RuntimeError(
                f"codegen session is {self._status.value}; expected one of: {expected}"
            )

    @staticmethod
    def _as_mapping(value: Mapping[str, Any], path: str) -> FrozenMapping:
        frozen = freeze_json(value, path=path)
        if not isinstance(frozen, FrozenMapping):
            raise TypeError(f"{path} must be a mapping")
        return frozen

    @staticmethod
    def _require_reason(reason: str) -> str:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("codegen reason must be a non-empty string")
        return reason


__all__ = ["CodegenBuildSession"]

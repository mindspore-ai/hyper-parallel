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
"""Tests for isolated sidecar publication during session finish."""

from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.snapshot import CodegenStatus


def _derived_session() -> CodegenBuildSession:
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    session.record_plan({"param_plan": {"block": {"params": {}}}})
    session.record_plan_application({"applied_boundaries": ["block"]})
    return session


def test_finish_publishes_derived_snapshot_once():
    """Successful finish publishes one derived snapshot and seals as generated."""
    session = _derived_session()
    published = []

    first = session.finish(published.append)
    second = session.finish(published.append)

    assert first is second, f"finish should be idempotent: first={first}, second={second}"
    assert first.status is CodegenStatus.GENERATED, (
        f"successful publication should finish as generated, got={first.status}"
    )
    assert len(published) == 1, f"publisher should run once, calls={len(published)}"
    assert published[0].status is CodegenStatus.DERIVED, (
        f"publisher should consume the derived snapshot, got={published[0].status}"
    )


def test_finish_isolates_generation_failure():
    """Publisher exceptions become failure data instead of escaping."""
    session = _derived_session()

    def fail(_snapshot):
        raise OSError("artifact directory is read-only")

    snapshot = session.finish(fail)

    assert snapshot.status is CodegenStatus.FAILED, (
        f"publication failure should be isolated, got={snapshot.status}"
    )
    assert snapshot.failure.stage == "generation", (
        f"failure stage mismatch: expected=generation, got={snapshot.failure.stage}"
    )
    assert snapshot.failure.reason == "artifact directory is read-only", (
        "failure reason mismatch: expected=artifact directory is read-only, "
        f"got={snapshot.failure.reason}"
    )
    details = snapshot.failure.to_dict()["details"]
    assert details == {"exception_type": "OSError"}, (
        f"failure details mismatch: expected exception_type=OSError, got={details}"
    )


def test_finish_publishes_not_required_snapshot_without_changing_status():
    """A native guard result publishes audit data and preserves its status."""
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    session.mark_not_required("pure_data_parallel")
    published = []

    snapshot = session.finish(published.append)

    assert snapshot.status is CodegenStatus.NOT_REQUIRED, (
        f"not-required status should be preserved, got={snapshot.status}"
    )
    assert published == [snapshot], (
        f"not-required session should publish its audit snapshot: "
        f"expected={[snapshot]}, got={published}"
    )

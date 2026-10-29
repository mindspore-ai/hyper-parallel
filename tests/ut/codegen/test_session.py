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
"""Unit tests for the codegen build-session state machine."""

import pytest

from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.snapshot import CodegenStatus


def test_derived_session_finishes_idempotently():
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    session.record_plan({"placements": {}})
    session.record_plan_application({"boundaries": []})

    first = session.finish()
    second = session.finish()

    assert first is second
    assert first.status is CodegenStatus.DERIVED


def test_not_required_session_has_stable_reason():
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.mark_not_required("pure_data_parallel")

    snapshot = session.finish()

    assert snapshot.status is CodegenStatus.NOT_REQUIRED
    assert snapshot.reason == "pure_data_parallel"
    assert snapshot.plan is None


def test_failed_session_does_not_retain_exception_objects():
    session = CodegenBuildSession()
    session.mark_failed("emitter", "render_failed", {"exception_type": "ValueError"})

    snapshot = session.finish()

    assert snapshot.status is CodegenStatus.FAILED
    assert snapshot.failure.reason == "render_failed"


def test_session_rejects_plan_before_source():
    session = CodegenBuildSession()

    with pytest.raises(RuntimeError, match="record_source"):
        session.record_plan({})


def test_session_rejects_finish_while_collecting():
    with pytest.raises(RuntimeError, match="still collecting"):
        CodegenBuildSession().finish()


def test_session_rejects_finish_without_plan_application():
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_plan({})

    with pytest.raises(RuntimeError, match="missing plan application"):
        session.finish()


def test_session_rejects_writes_after_finish():
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.mark_not_required("pure_data_parallel")
    session.finish()

    with pytest.raises(RuntimeError, match="already finished"):
        session.record_diagnostics({})


def test_empty_diagnostics_can_only_be_recorded_once():
    session = CodegenBuildSession()
    session.record_diagnostics({})

    with pytest.raises(RuntimeError, match="already been recorded"):
        session.record_diagnostics({})

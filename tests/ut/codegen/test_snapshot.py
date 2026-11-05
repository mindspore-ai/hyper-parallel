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
"""Unit tests for isolated codegen snapshots."""

import unittest
from unittest import mock

from hyper_parallel.codegen import snapshot as snapshot_module
from hyper_parallel.codegen.snapshot import CodegenSnapshot
from hyper_parallel.codegen.session import CodegenBuildSession


def _derived_snapshot() -> CodegenSnapshot:
    """Build a complete derived snapshot for contract tests."""
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy", "identity": {"revision": "v1"}})
    session.record_replacements(
        declarations=[{"rule_id": "replace-1", "target": "ToyBlock"}],
        applications=[{"fqn": "model.layers.0", "rule_id": "replace-1"}],
    )
    session.record_plan({"placements": {"model.layers.0": ["R", "S(0)"]}})
    session.record_diagnostics({"rank": 0, "temporary_dir": "/tmp/ignored"})
    return session.finish()


def test_snapshot_exposes_mutable_emission_data_without_mutating_itself():
    snapshot = _derived_snapshot()
    payload = snapshot.to_dict()
    payload["source"]["model_type"] = "changed"

    assert snapshot.to_dict()["source"]["model_type"] == "toy"


def test_session_recording_deep_copies_nested_input():
    """Mutating caller-owned data after recording does not change the snapshot."""
    source = {"model_type": "toy", "identity": {"revision": "v1"}}
    session = CodegenBuildSession()
    session.record_source(source)
    session.mark_not_required("pure_data_parallel")

    source["identity"]["revision"] = "v2"

    assert session.finish().to_dict()["source"]["identity"]["revision"] == "v1"


def test_snapshot_output_deep_copy_isolated_from_nested_mutation():
    """Mutating one emission payload does not change later payloads."""
    snapshot = _derived_snapshot()
    payload = snapshot.to_dict()

    payload["source"]["identity"]["revision"] = "v2"
    payload["replacements"]["applications"][0]["fqn"] = "changed"

    fresh = snapshot.to_dict()
    assert fresh["source"]["identity"]["revision"] == "v1"
    assert fresh["replacements"]["applications"][0]["fqn"] == "model.layers.0"


class TestSnapshotMemoryOwnership(unittest.TestCase):
    """Share session-owned data while preserving explicit export isolation."""

    def test_finish_reuses_observation_data_across_status_change(self):
        """Publishing and sealing do not copy the already isolated plan."""
        session = CodegenBuildSession()
        session.record_source({"model_type": "toy"})
        plan = {"param_plan": {"block": {"in_src": {"x": ["R"]}}}}
        session.record_plan(plan)
        published = []
        with mock.patch.object(snapshot_module, "deepcopy", wraps=snapshot_module.deepcopy) as copier:
            final = session.finish(published.append)
            copier.assert_not_called()
        self.assertIs(final.plan, published[0].plan)
        self.assertIsNot(final.plan, plan)
        plan["param_plan"]["block"]["in_src"]["x"].append("S(0)")
        self.assertEqual(final.plan["param_plan"]["block"]["in_src"]["x"], ["R"])

    def test_explicit_export_copies_data_once(self):
        """Nested replacement data is included in one export traversal."""
        snapshot = _derived_snapshot()
        with mock.patch.object(snapshot_module, "deepcopy", wraps=snapshot_module.deepcopy) as copier:
            payload = snapshot.to_dict()
            self.assertEqual(copier.call_count, 1)
        payload["replacements"]["applications"][0]["fqn"] = "changed"
        self.assertEqual(snapshot.replacements.applications[0]["fqn"], "model.layers.0")

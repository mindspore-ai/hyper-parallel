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
"""Unit tests for immutable codegen snapshots."""

import json

import pytest

from hyper_parallel.codegen.hash import signature_from_snapshot
from hyper_parallel.codegen.snapshot import CodegenSnapshot, FrozenMapping, freeze_json
from hyper_parallel.codegen.session import CodegenBuildSession


def _derived_snapshot() -> CodegenSnapshot:
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy", "identity": {"revision": "v1"}})
    session.record_replacements(
        declarations=[{"rule_id": "replace-1", "target": "ToyBlock"}],
        applications=[{"fqn": "model.layers.0", "rule_id": "replace-1"}],
    )
    session.record_plan({"placements": {"model.layers.0": ["R", "S(0)"]}})
    session.record_plan_application({"boundaries": ["model.layers.0"]})
    session.record_diagnostics({"rank": 0, "temporary_dir": "/tmp/ignored"})
    return session.finish()


def test_snapshot_json_round_trip_preserves_value():
    snapshot = _derived_snapshot()
    payload = json.loads(json.dumps(snapshot.to_dict()))

    restored = CodegenSnapshot.from_dict(payload)

    assert restored == snapshot
    assert restored.to_dict() == payload


def test_snapshot_signature_ignores_diagnostics():
    left = _derived_snapshot()
    payload = left.to_dict()
    payload["diagnostics"] = {"rank": 7, "temporary_dir": "/another/path"}
    right = CodegenSnapshot.from_dict(payload)

    assert signature_from_snapshot(left) == signature_from_snapshot(right)


@pytest.mark.parametrize("value", [object(), {"tensor": object()}, {1: "bad-key"}])
def test_freeze_json_rejects_runtime_objects(value):
    with pytest.raises((TypeError, ValueError)):
        freeze_json(value)


def test_frozen_mapping_rejects_mutable_values():
    with pytest.raises(TypeError):
        FrozenMapping((("bad", []),))


def test_snapshot_rejects_unknown_schema_version():
    payload = _derived_snapshot().to_dict()
    payload["schema_version"] += 1

    with pytest.raises(ValueError, match="unsupported snapshot schema version"):
        CodegenSnapshot.from_dict(payload)

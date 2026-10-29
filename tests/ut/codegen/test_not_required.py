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
"""State-contract tests for audit-only snapshot artifacts."""

from types import SimpleNamespace

import pytest

from hyper_parallel.codegen.check.preflight import verify_snapshot_contract
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.loader import import_generated_module
from hyper_parallel.codegen.meta import CodegenMeta, validate_meta_schema, write_codegen_meta
from hyper_parallel.codegen.runtime import verify_codegen_signature


def _not_required_meta() -> CodegenMeta:
    return CodegenMeta(
        codegen_version="0.1.1",
        signature="audit-signature",
        yaml_path="<snapshot>",
        yaml_sha256="audit-sha",
        source={"model_type": "toy"},
        parallel_dims={},
        covered={
            "sharding_plan": False,
            "module_overrides": False,
            "model_class": False,
            "axes": [],
        },
        entrypoints={},
        snapshot_schema_version=1,
        snapshot_status="not_required",
        snapshot_reason="pure_data_parallel",
    )


def test_not_required_meta_accepts_audit_only_contract():
    meta = _not_required_meta()

    validate_meta_schema({
        key: value
        for key, value in meta.__dict__.items()
        if value is not None
    })
    verify_snapshot_contract(meta)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("param_plan", {"block": {"params": {}}}),
        ("injections", [{"match": "block"}]),
        ("plan_application", {"applied_boundaries": ["block"]}),
        ("entrypoints", {"parallelize": "hp_parallelize"}),
    ),
)
def test_not_required_meta_rejects_executable_plan_fields(field, value):
    meta = _not_required_meta()
    setattr(meta, field, value)

    with pytest.raises(ValueError, match="not_required snapshot metadata"):
        validate_meta_schema({
            key: item
            for key, item in meta.__dict__.items()
            if item is not None
        })


def test_not_required_runtime_and_emitter_reject_execution(tmp_path):
    meta = _not_required_meta()
    write_codegen_meta(meta, str(tmp_path / "codegen_meta.json"))

    with pytest.raises(RuntimeError, match="not_required artifact"):
        verify_codegen_signature(meta)
    with pytest.raises(RuntimeError, match="audit-only"):
        import_generated_module(str(tmp_path))
    with pytest.raises(ValueError, match="no parallel boundaries"):
        lower_forward_boundaries(
            "class Block:\n    pass\n",
            SimpleNamespace(snapshot_status="not_required"),
        )


def test_failed_snapshot_meta_is_not_preflight_eligible():
    meta = _not_required_meta()
    meta.snapshot_status = "failed"
    meta.snapshot_reason = None
    meta.snapshot_failure = {
        "stage": "generation",
        "reason": "injected failure",
        "details": {},
    }

    validate_meta_schema({
        key: value
        for key, value in meta.__dict__.items()
        if value is not None
    })
    with pytest.raises(RuntimeError, match="snapshot generation failed"):
        verify_snapshot_contract(meta)

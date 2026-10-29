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
"""Snapshot-driven manager must not rebuild native planning decisions."""

import pytest

from hyper_parallel.codegen import manager
from hyper_parallel.codegen.artifact import ArtifactLayout
from hyper_parallel.codegen.check import verify_snapshot_contract, verify_snapshot_drift
from hyper_parallel.codegen.emit import replacement
from hyper_parallel.codegen.plan import derive, offline_mesh
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.distributed._builder import planner


def _snapshot(source_path, *, with_replacement=False):
    session = CodegenBuildSession()
    session.record_source({
        "model_type": "toy",
        "architecture": "ToyModel",
        "architectures": ["ToyModel"],
        "model_class": "tests:ToyModel",
        "module_name": "tests.toy_modeling",
        "file_path": str(source_path),
        "sha256": "source-sha",
        "line_count": 1,
        "is_hf_model": True,
        "backend": "hf",
    })
    if with_replacement:
        session.record_replacements(
            declarations=[{
                "rule_id": "replace:0",
                "match": ["block"],
                "source_type": "tests.toy:SourceBlock",
                "replacement_target": "tests.toy.replace_block",
                "exact_type": True,
                "static_arguments": {},
            }],
            applications=[{
                "rule_id": "replace:0",
                "primary_fqn": "block",
                "module_fqns": ["block"],
                "source_type": "tests.toy:SourceBlock",
                "replacement_type": "tests.toy:ReplacementBlock",
                "weight_transforms": [],
            }],
        )
    else:
        session.record_replacements()
    session.record_plan({
        "param_plan": {"block": {"params": {}}},
        "injections": [],
        "tied_pairs": [],
        "special_handlers": {},
        "frozen_sharded_params": [],
        "boundary_classes": {"block": "ToyBlock"},
        "mesh_dim_names": ["tp"],
        "parallel_dims": {
            "dp_size": 1,
            "tp_size": 2,
            "cp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "sequence_parallel": False,
            "loss_parallel": False,
        },
    })
    session.record_plan_application({
        "validate_mode": False,
        "applied_boundaries": ["block"],
        "source_shard_info": {},
        "resolved_injections": [],
        "tied_pairs": [],
        "special_handlers": {},
        "ep_bindings": [],
        "resolved_wrappers": [],
    })
    return session.finish()


def test_snapshot_manager_uses_no_legacy_derivation(monkeypatch, tmp_path):
    source = tmp_path / "modeling_toy.py"
    source.write_text("class ToyModel:\n    pass\n", encoding="utf-8")
    snapshot = _snapshot(source)
    captured = {}

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("snapshot manager called a legacy derivation path")

    monkeypatch.setattr(manager, "_fill_plan_fields", forbidden)
    monkeypatch.setattr(manager, "_prepare_codegen_build", forbidden)
    monkeypatch.setattr(derive, "build_meta_model", forbidden)
    monkeypatch.setattr(offline_mesh, "build_offline_mesh", forbidden)
    monkeypatch.setattr(planner.ShardingPlanner, "plan", forbidden)
    monkeypatch.setattr(replacement, "compile_overrides_for_meta", forbidden)
    monkeypatch.setattr(manager, "_is_rank0", lambda: True)
    monkeypatch.setattr(manager, "load_codegen_meta", lambda _path: None)
    monkeypatch.setattr(manager, "clean_temp_artifacts", lambda _layout: None)
    monkeypatch.setattr(manager, "ensure_artifact_dir", lambda _layout: None)
    monkeypatch.setattr(manager, "barrier_after_artifact", lambda: None)

    def emit(layout, meta, **kwargs):
        captured["layout"] = layout
        captured["meta"] = meta
        captured["snapshot"] = kwargs["snapshot"]
        return {"modeling_ToyModel_gen_npu.py": "generated"}

    monkeypatch.setattr(manager, "emit_bundle", emit)
    monkeypatch.setattr(
        manager,
        "write_bundle_atomic",
        lambda layout, files, meta: captured.update(files=files, written_meta=meta),
    )

    layout = manager.ensure_codegen_artifact_from_snapshot(
        snapshot,
        str(tmp_path / "artifact"),
    )

    assert layout is captured["layout"]
    assert captured["snapshot"] is snapshot
    assert captured["written_meta"] is captured["meta"]
    assert captured["meta"].snapshot_status == "derived"
    assert captured["meta"].param_plan == {"block": {"params": {}}}
    assert captured["meta"].parallel_dims["tp_size"] == 2


def test_snapshot_manager_reuses_matching_signature(monkeypatch, tmp_path):
    source = tmp_path / "modeling_toy.py"
    source.write_text("class ToyModel:\n    pass\n", encoding="utf-8")
    snapshot = _snapshot(source)
    signature = manager.signature_from_snapshot(snapshot)
    existing = type("Meta", (), {"signature": signature})()
    calls = {"barrier": 0}

    monkeypatch.setattr(manager, "_is_rank0", lambda: True)
    monkeypatch.setattr(manager, "load_codegen_meta", lambda _path: existing)
    monkeypatch.setattr(manager, "clean_temp_artifacts", lambda _layout: None)
    monkeypatch.setattr(manager, "ensure_artifact_dir", lambda _layout: None)
    monkeypatch.setattr(
        manager,
        "barrier_after_artifact",
        lambda: calls.update(barrier=calls["barrier"] + 1),
    )
    monkeypatch.setattr(
        manager,
        "emit_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("matching snapshot should reuse the bundle")
        ),
    )

    manager.ensure_codegen_artifact_from_snapshot(
        snapshot,
        str(tmp_path / "artifact"),
    )

    assert calls == {"barrier": 1}


def test_snapshot_manager_maps_observed_replacement_without_rematching(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "modeling_toy.py"
    source.write_text("class ToyModel:\n    pass\n", encoding="utf-8")
    snapshot = _snapshot(source, with_replacement=True)
    captured = {}

    monkeypatch.setattr(manager, "_is_rank0", lambda: True)
    monkeypatch.setattr(manager, "load_codegen_meta", lambda _path: None)
    monkeypatch.setattr(manager, "clean_temp_artifacts", lambda _layout: None)
    monkeypatch.setattr(manager, "ensure_artifact_dir", lambda _layout: None)
    monkeypatch.setattr(manager, "barrier_after_artifact", lambda: None)
    monkeypatch.setattr(
        manager,
        "emit_bundle",
        lambda _layout, meta, **_kwargs: captured.update(meta=meta) or {},
    )
    monkeypatch.setattr(manager, "write_bundle_atomic", lambda *_args: None)

    manager.ensure_codegen_artifact_from_snapshot(
        snapshot,
        str(tmp_path / "artifact"),
    )

    assert captured["meta"].module_overrides == [{
        "match": ["block"],
        "fqn": "block",
        "fqns": ["block"],
        "module_type": "tests.toy.SourceBlock",
        "factory": "tests.toy.replace_block",
        "exact_type": True,
    }]
    assert captured["meta"].emission_patches["module_replacements"] == [{
        "module_fqn": "block",
        "source_type": "tests.toy:SourceBlock",
        "replacement_type": "tests.toy:ReplacementBlock",
    }]


def test_snapshot_manager_rejects_not_required_snapshot(tmp_path):
    session = CodegenBuildSession()
    session.record_source({"file_path": str(tmp_path / "modeling.py")})
    session.record_replacements()
    session.mark_not_required("pure_data_parallel")

    with pytest.raises(ValueError, match="requires status 'derived'"):
        manager.ensure_codegen_artifact_from_snapshot(
            session.finish(),
            str(tmp_path / "artifact"),
        )


def test_snapshot_contract_accepts_legacy_and_complete_snapshot_meta():
    legacy = type(
        "LegacyMeta",
        (),
        {"snapshot_schema_version": None, "snapshot_status": None},
    )()
    complete = type(
        "SnapshotMeta",
        (),
        {
            "snapshot_schema_version": 1,
            "snapshot_status": "derived",
            "plan_application": {"applied_boundaries": ["block"]},
        },
    )()

    verify_snapshot_contract(legacy)
    verify_snapshot_contract(complete)


@pytest.mark.parametrize(
    ("schema_version", "status", "application", "message"),
    [
        (1, None, {}, "must include both"),
        (1, "collecting", {}, "requires a derived snapshot"),
        (1, "derived", {}, "missing the native plan application"),
    ],
)
def test_snapshot_contract_rejects_incomplete_provenance(
    schema_version,
    status,
    application,
    message,
):
    meta = type(
        "Meta",
        (),
        {
            "snapshot_schema_version": schema_version,
            "snapshot_status": status,
            "plan_application": application,
        },
    )()

    with pytest.raises(RuntimeError, match=message):
        verify_snapshot_contract(meta)


def test_snapshot_drift_uses_snapshot_signature(tmp_path):
    source = tmp_path / "modeling_toy.py"
    source.write_text("class ToyModel:\n    pass\n", encoding="utf-8")
    snapshot = _snapshot(source)
    matching = type(
        "Meta",
        (),
        {"signature": manager.signature_from_snapshot(snapshot)},
    )()
    stale = type("Meta", (), {"signature": "stale"})()

    verify_snapshot_drift(matching, snapshot)
    with pytest.raises(RuntimeError, match="snapshot signature mismatch"):
        verify_snapshot_drift(stale, snapshot)


def test_snapshot_emission_patches_include_resolved_phase_c_injections():
    payload = {
        "source": {"module_name": "transformers.models.toy.modeling_toy"},
        "replacements": {
            "applications": [{
                "primary_fqn": "model.attn",
                "source_type": "transformers.models.toy.modeling_toy:ToyAttention",
                "replacement_type": "hyper_parallel.components:ToyAttention",
            }],
        },
        "plan": {
            "boundary_classes": {
                "model.mlp": "ToyMoeBlock",
                "model.attn": "ToyAttention",
            },
            "param_plan": {
                "model.mlp": {"region_dispatch": False},
                "model.attn": {},
            },
            "injections": [],
        },
        "plan_application": {
            "resolved_injections": [
                {
                    "match": "model.mlp",
                    "local_compute_fn": {
                        "_target_": "hyper_parallel.models.toy.ep.toy_compute"
                    },
                },
                {
                    "match": "model.attn",
                    "inner_target": "self",
                    "inner_wrapper": {
                        "_target_": "hyper_parallel.models.toy.cp:toy_wrapper"
                    },
                },
            ],
        },
    }

    patches = manager._snapshot_emission_patches(payload)

    expected_local = [{
        "module_fqn": "model.mlp",
        "module_type": "transformers.models.toy.modeling_toy:ToyMoeBlock",
        "factory": "hyper_parallel.models.toy.ep:toy_compute",
        "region_dispatch": False,
        "original_forward": (
            "transformers.models.toy.modeling_toy:ToyMoeBlock.forward"
        ),
    }]
    expected_wrappers = [{
        "module_fqn": "model.attn",
        "target_fqn": "self",
        "wrapper": "hyper_parallel.models.toy.cp:toy_wrapper",
        "target_type": "hyper_parallel.components:ToyAttention",
    }]
    assert patches["local_computes"] == expected_local, (
        "snapshot local-compute patch mismatch: "
        f"expected={expected_local}, got={patches['local_computes']}"
    )
    assert patches["inner_wrappers"] == expected_wrappers, (
        "snapshot inner-wrapper patch mismatch: "
        f"expected={expected_wrappers}, got={patches['inner_wrappers']}"
    )


def test_snapshot_manager_normalizes_deepseek_v4_family(monkeypatch, tmp_path):
    source = tmp_path / "modeling_deepseek_v4.py"
    source.write_text("class DeepseekV4:\n    pass\n", encoding="utf-8")
    session = CodegenBuildSession()
    session.record_source({
        "model_type": "deepseek_v4",
        "architecture": "DeepseekV4",
        "file_path": str(source),
        "is_hf_model": False,
    })
    session.record_replacements()
    session.record_plan({"param_plan": {"model": {"params": {}}}})
    session.record_plan_application({"applied_boundaries": ["model"]})
    snapshot = session.finish()
    signature = manager.signature_from_snapshot(snapshot)
    layout = ArtifactLayout.build_default(
        str(tmp_path / "deepseek_v41" / "generated"),
        "DeepseekV4",
    )
    captured = {}

    def resolve(model_type, **_kwargs):
        captured["model_type"] = model_type
        return layout

    monkeypatch.setattr(manager, "resolve_model_artifact_layout", resolve)
    monkeypatch.setattr(manager, "ensure_artifact_dir", lambda _layout: None)
    monkeypatch.setattr(manager, "clean_temp_artifacts", lambda _layout: None)
    monkeypatch.setattr(manager, "_is_rank0", lambda: True)
    monkeypatch.setattr(
        manager,
        "load_codegen_meta",
        lambda _path: type("Meta", (), {"signature": signature})(),
    )
    monkeypatch.setattr(manager, "barrier_after_artifact", lambda: None)

    result = manager.ensure_codegen_artifact_from_snapshot(snapshot)

    assert result is layout, f"snapshot manager returned wrong layout: got={result}"
    assert captured["model_type"] == "deepseek_v41", (
        "DeepSeek family alias mismatch: "
        f"expected=deepseek_v41, got={captured['model_type']}"
    )

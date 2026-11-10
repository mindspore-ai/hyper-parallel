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
"""Tests for snapshot-to-emission orchestration without persisted metadata."""

from pathlib import Path
from types import SimpleNamespace

from hyper_parallel.codegen import manager
from hyper_parallel.codegen.emit import recorded, snapshot_patches
from hyper_parallel.codegen.session import CodegenBuildSession


def _snapshot(source_path: Path):
    """Build a complete immutable toy snapshot for manager tests."""
    session = CodegenBuildSession()
    session.record_source({
        "model_type": "toy",
        "architecture": "ToyModel",
        "architectures": ["ToyModel"],
        "model_class": "tests:ToyModel",
        "module_name": "tests.toy_modeling",
        "file_path": str(source_path),
        "is_hf_model": True,
    })
    session.record_replacements()
    session.record_plan({
        "param_plan": {"block": {"params": {}}},
        "injections": [],
        "tied_pairs": [],
        "special_handlers": {},
        "frozen_sharded_params": [],
        "boundary_classes": {"block": "ToyBlock"},
        "mesh_dim_names": ["tp"],
        "parallel_dims": {"tp_size": 2},
    })
    return session.finish()


def test_manager_emits_on_every_call(monkeypatch, tmp_path):
    """An existing bundle never bypasses fresh snapshot emission."""
    source = tmp_path / "modeling_toy.py"
    source.write_text("class ToyModel:\n    pass\n", encoding="utf-8")
    snapshot = _snapshot(source)

    def reject_export(_snapshot):
        """Fail if publication requests an unnecessary mutable export."""
        raise AssertionError("generation must borrow observations instead of exporting a deep copy")

    monkeypatch.setattr(type(snapshot), "to_dict", reject_export)
    emitted = []
    published = []

    def _emit(layout, context):
        """Return distinct content for each emission attempt."""
        del layout
        emitted.append(context)
        return {"modeling_Toy_gen_npu.py": f"RUN = {len(emitted)}\n"}

    monkeypatch.setattr(manager, "emit_bundle", _emit)
    monkeypatch.setattr(
        manager,
        "write_bundle_atomic",
        lambda _layout, files: published.append(files),
    )

    for _ in range(2):
        manager.ensure_codegen_artifact_from_snapshot(
            snapshot, str(tmp_path / "generated")
        )

    assert len(emitted) == 2
    assert published == [
        {"modeling_Toy_gen_npu.py": "RUN = 1\n"},
        {"modeling_Toy_gen_npu.py": "RUN = 2\n"},
    ]
    assert emitted[0].status == "derived"
    assert emitted[0].param_plan == {"block": {"params": {}}}


def test_snapshot_class_resolves_to_defining_remote_sibling(tmp_path):
    """An imported patch host resolves to the sibling source defining it."""
    main = tmp_path / "modeling_kimi_k3.py"
    sibling = tmp_path / "modeling_kimi_linear.py"
    main.write_text(
        "from .modeling_kimi_linear import KimiDeltaAttention\n\n"
        "class KimiK3ForCausalLM:\n    pass\n",
        encoding="utf-8",
    )
    sibling.write_text("class KimiDeltaAttention:\n    pass\n", encoding="utf-8")

    identity = snapshot_patches._source_type_identity(  # pylint: disable=protected-access
        {
            "module_name": "transformers_modules.kimi.modeling_kimi_k3",
            "file_path": str(main),
        },
        "KimiDeltaAttention",
    )

    assert identity == (
        "transformers_modules.kimi.modeling_kimi_linear:KimiDeltaAttention"
    )


def test_recorded_boundary_edits_preserve_borrowed_snapshot():
    """Recorded forward lowering copies changed entries instead of the full plan."""
    entry = {"is_boundary": True, "in_src": {"x": ["R"]}}
    meta = SimpleNamespace(
        param_plan={"block": entry}, boundary_classes={"block": "Block"},
        external_state_classes=[],
    )
    recorded._mark_class_source_lowering_skipped(meta, "Block")
    assert "skip_source_lowering" not in entry, f"borrowed entry changed: {entry}"
    assert meta.param_plan["block"]["in_src"] is entry["in_src"]
    source_entry = {"is_boundary": False}
    meta.param_plan["block"] = source_entry
    record = SimpleNamespace(module_fqn="block", module_type=type("Block", (), {}))
    recorded._restore_recorded_boundary_ownership(meta, [record])
    assert source_entry["is_boundary"] is False
    assert meta.param_plan["block"]["is_boundary"] is True


def test_emission_patches_use_observed_phase_c_injections(tmp_path):
    """Emission descriptors come directly from the observed native plan."""
    source = tmp_path / "modeling_toy.py"
    source.write_text("class ToyMoeBlock:\n    pass\n", encoding="utf-8")
    payload = {
        "source": {
            "module_name": "transformers.models.toy.modeling_toy",
            "file_path": str(source),
        },
        "replacements": {"applications": []},
        "plan": {
            "boundary_classes": {"model.mlp": "ToyMoeBlock"},
            "param_plan": {"model.mlp": {"region_dispatch": False}},
            "injections": [{
                "match": "model.mlp",
                "local_compute_fn": {
                    "_target_": "hyper_parallel.models.toy.ep.toy_compute"
                },
            }],
        },
    }

    patches = snapshot_patches.emission_patches_from_snapshot_payload(payload)

    assert patches["local_computes"] == [{
        "module_fqn": "model.mlp",
        "module_type": "transformers.models.toy.modeling_toy:ToyMoeBlock",
        "factory": "hyper_parallel.models.toy.ep:toy_compute",
        "region_dispatch": False,
        "original_forward": (
            "transformers.models.toy.modeling_toy:ToyMoeBlock.forward"
        ),
    }]

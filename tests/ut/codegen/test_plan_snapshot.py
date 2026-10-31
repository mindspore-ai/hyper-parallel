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
"""Contracts for codegen-owned planner observation and publication."""

import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.codegen import model_build_overlay
from hyper_parallel.codegen.plan.freeze import freeze_plan, freeze_plan_application
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.snapshot import CodegenStatus
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.models._transformers import auto_model, model_builder


def _plan():
    return SimpleNamespace(modules={}, tied_pairs=[], special_handlers={}, mesh_dim_names=("tp",))


def _mesh(*, tp_size=2, device_mesh=object()):
    return SimpleNamespace(
        tp_size=tp_size, cp_size=1, ep_size=1, sequence_parallel=False,
        loss_parallel=False, device_mesh=device_mesh,
    )


def _session():
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    return session


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_native_plan_still_applies_sharding(monkeypatch):
    """Native planning retains its original application behavior.

    Feature: native planner isolation
    Description: Run the native planner with a fake plan and applier.
    Expectation: Planning and application each execute exactly once.
    """
    calls = {"plan": 0, "apply": 0}
    plan = _plan()

    class Planner:
        @staticmethod
        def _plan(*_args, **_kwargs):
            calls["plan"] += 1
            return plan

        plan = _plan

    def _apply(model, received_plan, _mesh_value, *, validate_mode):
        calls["apply"] += 1
        assert received_plan is plan
        assert validate_mode is False
        return model, None

    monkeypatch.setattr(model_builder, "apply_sharding_plan", _apply)
    model, source_info = model_builder._plan_and_apply_sharding(
        nn.Module(), _mesh(), Planner(), True, False,
    )
    assert isinstance(model, nn.Module)
    assert source_info is None
    assert calls == {"plan": 1, "apply": 1}


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_codegen_observation_does_not_apply_sharding(monkeypatch):
    """Codegen freezes plan facts without invoking the native applier.

    Feature: offline codegen planning
    Description: Observe a fake planner through the codegen overlay helper.
    Expectation: The snapshot is derived and no sharding application runs.
    """
    plan = _plan()

    class Planner:
        @staticmethod
        def _plan(*_args, **_kwargs):
            return plan

        plan = _plan

    monkeypatch.setattr(
        model_builder, "apply_sharding_plan",
        lambda *_args, **_kwargs: pytest.fail("offline observation applied sharding"),
    )
    session = _session()
    model_build_overlay._observe_plan(nn.Module(), _mesh(), Planner(), True, session)
    snapshot = session.finish()
    assert snapshot.status is CodegenStatus.DERIVED
    assert snapshot.to_dict()["plan_application"]["source_shard_info"] == {}


@pytest.mark.parametrize(
    ("planner", "mesh", "is_hf_model", "reason"),
    (
        (None, _mesh(), True, "planner_unavailable"),
        (object(), None, True, "mesh_unavailable"),
        (object(), _mesh(tp_size=1), True, "pure_data_parallel"),
        (object(), _mesh(device_mesh=None), False, "device_mesh_unavailable"),
    ),
)
@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_codegen_plan_guards_record_reason(planner, mesh, is_hf_model, reason):
    """Codegen records why no distributed plan is required.

    Feature: offline codegen planning
    Description: Exercise every early-exit condition in plan observation.
    Expectation: Each condition produces its stable not-required reason.
    """
    session = _session()
    model_build_overlay._observe_plan(nn.Module(), mesh, planner, is_hf_model, session)
    snapshot = session.finish()
    assert snapshot.status is CodegenStatus.NOT_REQUIRED
    assert snapshot.reason == reason


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_plan_application_freezes_source_mesh():
    """Frozen application metadata excludes live mesh objects.

    Feature: snapshot serialization
    Description: Freeze source placement information containing a runtime mesh.
    Expectation: Only placement names, dimensions, and shape remain.
    """
    source_mesh = SimpleNamespace(mesh_dim_names=("tp",), shape=(2,))
    application = freeze_plan_application(
        _plan(), {"block.weight": ((Shard(0), Replicate()), source_mesh)}, validate_mode=False,
    )
    assert application["source_shard_info"]["block.weight"] == {
        "placements": ["S(0)", "R"], "mesh_dim_names": ["tp"], "mesh_shape": [2],
    }


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_frozen_injections_ignore_module_insertion_order():
    """Frozen injections have deterministic ordering.

    Feature: snapshot serialization
    Description: Freeze equivalent plans with opposite module insertion order.
    Expectation: Plans serialize identically in match order.
    """
    specs = {
        "block.0": SimpleNamespace(local_compute_fn={"_target_": "example.zero"}),
        "block.1": SimpleNamespace(local_compute_fn={"_target_": "example.one"}),
    }
    forward = SimpleNamespace(
        modules=dict(specs.items()), tied_pairs=[], special_handlers={}, mesh_dim_names=("ep",),
    )
    reverse = SimpleNamespace(
        modules=dict(reversed(specs.items())), tied_pairs=[], special_handlers={}, mesh_dim_names=("ep",),
    )
    forward_frozen = freeze_plan(forward, model=None).to_dict()
    assert forward_frozen == freeze_plan(reverse, model=None).to_dict()
    assert [item["match"] for item in forward_frozen["injections"]] == ["block.0", "block.1"]


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_model_build_overlay_restores_native_classmethod():
    """The CLI build override is scoped and reversible.

    Feature: codegen model-build isolation
    Description: Enter and leave the model-build overlay context.
    Expectation: AutoModel's original classmethod descriptor is restored exactly.
    """
    model_class = auto_model._BaseHyperAutoModelClass
    original = model_class.__dict__["_build_model"]
    with model_build_overlay.use_codegen_model_build_overlay(CodegenBuildSession()):
        assert model_class.__dict__["_build_model"] is not original
    assert model_class.__dict__["_build_model"] is original


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_codegen_model_init_context_places_parameters_and_buffers_on_meta():
    """The CLI observation model owns no CPU parameter or buffer storage."""
    with model_build_overlay._codegen_model_init_context():
        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.ones(4)))
        module.register_buffer("inv_freq", torch.ones(4), persistent=False)

    assert module.weight.is_meta
    assert module.inv_freq.is_meta


@arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
          card_mark="onecard", essential_mark="unessential")
def test_offline_publication_restores_rng(monkeypatch):
    """Snapshot publication cannot advance host RNG state.

    Feature: codegen publication isolation
    Description: Consume Python and Torch RNG while publishing a snapshot.
    Expectation: Both RNG states are restored after publication.
    """
    session = _session()
    session.record_plan({"param_plan": {"block": {"params": {}}}})
    session.record_plan_application({"applied_boundaries": ["block"]})
    cpu_state = torch.get_rng_state().clone()
    python_state = random.getstate()

    def _consume_rng(_snapshot, _artifact_dir):
        torch.rand(4)
        random.random()
        return object()

    monkeypatch.setattr(model_build_overlay, "ensure_codegen_artifact_from_snapshot", _consume_rng)
    result = model_build_overlay.CodegenModelBuildResult()
    model_build_overlay._publish_session(session, None, result)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert random.getstate() == python_state
    assert result.artifact_layout is not None

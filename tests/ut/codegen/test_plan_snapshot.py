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

from types import SimpleNamespace

import pytest
from torch import nn

from hyper_parallel.codegen.plan.observation import observe_emission_plan
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models.build_capture import capture_model_build_plan, record_model_source_for_capture
from hyper_parallel.models.module_build import build_module_context, prepare_model_components


def _plan():
    return SimpleNamespace(modules={}, tied_pairs=[], special_handlers={}, parallel_modules={}, mesh_dim_names=("tp",))


def _mesh(*, tp_size=2, device_mesh=object()):
    return SimpleNamespace(
        tp_size=tp_size, cp_size=1, ep_size=1, sequence_parallel=False,
        loss_parallel=False, device_mesh=device_mesh,
    )


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


def test_codegen_observation_rejects_missing_ep_compute_source():
    """The real native preflight rejects an EP boundary before publication."""
    plan = SimpleNamespace(
        modules={
            "model.layers.0.mlp": SimpleNamespace(
                is_boundary=True,
                _ep_size=8,
                local_compute_fn=None,
                region_dispatch=None,
            ),
        },
        tied_pairs=[],
        parallel_modules={},
        special_handlers={},
        mesh_dim_names=("tp",),
    )

    class Planner:
        """Return an incomplete EP plan matching the native failure mode."""

        @staticmethod
        def plan(*_args, **_kwargs):
            return plan

    device_mesh = LogicalDeviceMesh(("tp",), (2,))
    mesh = _mesh(device_mesh=device_mesh)
    model = nn.Module()
    error = "ep_size=8.*no EP-aware compute source"
    with pytest.raises(ValueError, match=error) as native:
        model_builder._plan_and_apply_sharding(model, mesh, Planner(), True, False)
    with capture_model_build_plan():
        record_model_source_for_capture(model, pretrained_model_name_or_path=None)
        with pytest.raises(ValueError, match=error) as captured:
            model_builder._plan_and_apply_sharding(model, mesh, Planner(), True, False)
    assert str(captured.value) == str(native.value), f"Capture: {captured.value}; native: {native.value}"


@pytest.mark.parametrize(
    ("planner", "mesh", "is_hf_model", "reason"),
    (
        (None, _mesh(), True, "planner_unavailable"),
        (object(), None, True, "mesh_unavailable"),
        (object(), _mesh(tp_size=1), True, "pure_data_parallel"),
        (object(), _mesh(device_mesh=None), False, "device_mesh_unavailable"),
    ),
)
def test_codegen_plan_guards_record_reason(planner, mesh, is_hf_model, reason):
    """Codegen records why no distributed plan is required.

    Feature: offline codegen planning
    Description: Exercise every early-exit condition in plan observation.
    Expectation: Each condition produces its stable not-required reason.
    """
    model = nn.Module()
    with capture_model_build_plan() as capture:
        record_model_source_for_capture(model, pretrained_model_name_or_path=None)
        model, _ = prepare_model_components(model, build_module_context(None))
        model_builder._plan_and_apply_sharding(model, mesh, planner, is_hf_model, False)
    assert capture.planning_finished, "The capture must finish without invoking the planner"
    assert capture.plan is None, "This configuration does not require a sharding plan"
    assert capture.reason == reason, f"Expected capture skip reason {reason}, got {capture.reason}"


def test_observed_injections_ignore_module_insertion_order():
    """Observed injections have deterministic ordering.

    Feature: snapshot serialization
    Description: Observe equivalent plans with opposite module insertion order.
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
    forward_observed = observe_emission_plan(forward, model=None).to_dict()
    assert forward_observed == observe_emission_plan(reverse, model=None).to_dict()
    assert [item["match"] for item in forward_observed["injections"]] == ["block.0", "block.1"]


def test_observation_excludes_native_execution_only_fields():
    """Source emission ignores plan fields owned by native parameter execution."""
    spec = SimpleNamespace(
        params={"weight": object()},
        tp_divide_attrs=["heads"],
        _ep_stack={"ep": object()},
        _ep_size=8,
        is_boundary=True,
    )
    plan = SimpleNamespace(modules={"block": spec}, mesh_dim_names=("ep",))

    observed = observe_emission_plan(plan, model=None).to_dict()

    assert observed["param_plan"] == {"block": {"is_boundary": True}}
    assert observed["injections"] == []
    assert "tied_pairs" not in observed
    assert "special_handlers" not in observed
    assert "frozen_sharded_params" not in observed

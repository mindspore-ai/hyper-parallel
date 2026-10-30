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
"""Contracts for recording native planner and applier results."""

from contextlib import nullcontext
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.codegen.plan.freeze import freeze_plan, freeze_plan_application
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.snapshot import CodegenStatus
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.build_observer import ModelBuildObservation, use_model_build_observation


def _plan():
    return SimpleNamespace(
        modules={},
        tied_pairs=[],
        special_handlers={},
        mesh_dim_names=("tp",),
    )


def _mesh(*, tp_size=2, device_mesh=object()):
    return SimpleNamespace(
        tp_size=tp_size,
        cp_size=1,
        ep_size=1,
        sequence_parallel=False,
        loss_parallel=False,
        device_mesh=device_mesh,
    )


def _session():
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    return session


def test_native_plan_and_apply_are_each_called_once(monkeypatch):
    calls = {"plan": 0, "apply": 0}
    plan = _plan()

    class Planner:
        @staticmethod
        def plan(*args, **kwargs):
            del args, kwargs
            calls["plan"] += 1
            return plan

    def apply(model, received_plan, mesh, *, validate_mode):
        del mesh
        calls["apply"] += 1
        assert received_plan is plan
        assert validate_mode is False
        return model, None

    monkeypatch.setattr(model_builder, "apply_sharding_plan", apply)
    session = _session()

    model, source_shard_info = model_builder._plan_and_apply_sharding(
        nn.Module(),
        _mesh(),
        Planner(),
        is_hf_model=True,
        validate_placement=False,
        build_observer=session,
    )
    snapshot = session.finish()

    assert isinstance(model, nn.Module)
    assert source_shard_info is None
    assert calls == {"plan": 1, "apply": 1}
    assert snapshot.status is CodegenStatus.DERIVED
    assert snapshot.to_dict()["plan"]["mesh_dim_names"] == ["tp"]
    assert snapshot.to_dict()["plan_application"]["validate_mode"] is False


def test_observation_only_plans_without_applying_sharding(monkeypatch):
    """Offline observation freezes plan facts without DTensor or collectives."""
    plan = _plan()

    class Planner:
        @staticmethod
        def plan(*_args, **_kwargs):
            return plan

    monkeypatch.setattr(
        model_builder,
        "apply_sharding_plan",
        lambda *_args, **_kwargs: pytest.fail("offline observation applied sharding"),
    )
    session = _session()

    model, source_shard_info = model_builder._plan_and_apply_sharding(
        nn.Module(),
        _mesh(),
        Planner(),
        is_hf_model=True,
        validate_placement=False,
        build_observer=session,
        apply_plan=False,
    )
    snapshot = session.finish()

    assert isinstance(model, nn.Module)
    assert source_shard_info is None
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
def test_native_plan_guards_record_stable_not_required_reason(
    planner,
    mesh,
    is_hf_model,
    reason,
):
    session = _session()

    _, source_shard_info = model_builder._plan_and_apply_sharding(
        nn.Module(),
        mesh,
        planner,
        is_hf_model=is_hf_model,
        validate_placement=False,
        build_observer=session,
    )
    snapshot = session.finish()

    assert source_shard_info is None
    assert snapshot.status is CodegenStatus.NOT_REQUIRED
    assert snapshot.reason == reason
    assert snapshot.plan is None


def test_plan_application_freezes_source_mesh_without_runtime_objects():
    source_mesh = SimpleNamespace(mesh_dim_names=("tp",), shape=(2,))

    application = freeze_plan_application(
        _plan(),
        {"block.weight": ((Shard(0), Replicate()), source_mesh)},
        validate_mode=False,
    )

    assert application["source_shard_info"] == {
        "block.weight": {
            "placements": ["S(0)", "R"],
            "mesh_dim_names": ["tp"],
            "mesh_shape": [2],
        }
    }


def test_frozen_injections_are_stable_across_module_insertion_order():
    specs = {
        "block.0": SimpleNamespace(local_compute_fn={"_target_": "example.compute_zero"}),
        "block.1": SimpleNamespace(local_compute_fn={"_target_": "example.compute_one"}),
    }
    forward_plan = SimpleNamespace(
        modules=dict(specs.items()),
        tied_pairs=[],
        special_handlers={},
        mesh_dim_names=("ep",),
    )
    reverse_plan = SimpleNamespace(
        modules=dict(reversed(specs.items())),
        tied_pairs=[],
        special_handlers={},
        mesh_dim_names=("ep",),
    )

    forward_frozen = freeze_plan(forward_plan, model=None).to_dict()
    reverse_frozen = freeze_plan(reverse_plan, model=None).to_dict()
    forward_application = freeze_plan_application(forward_plan, None, validate_mode=False)
    reverse_application = freeze_plan_application(reverse_plan, None, validate_mode=False)

    assert forward_frozen == reverse_frozen
    assert forward_application == reverse_application
    assert [item["match"] for item in forward_frozen["injections"]] == ["block.0", "block.1"]


def test_native_auto_build_attaches_finished_snapshot(monkeypatch):
    model = nn.Module()
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
    session = CodegenBuildSession()

    monkeypatch.setattr(auto_model, "_model_init_context", lambda _is_meta: nullcontext())
    monkeypatch.setattr(
        auto_model,
        "_init_model",
        lambda *args, **kwargs: (False, model),
    )
    monkeypatch.setattr(auto_model, "_current_device", lambda: None)
    monkeypatch.setattr(
        auto_model.torch.distributed,
        "is_initialized",
        lambda: False,
    )

    def apply_infrastructure(built_model, **kwargs):
        session = kwargs["build_observer"]
        session.record_replacements()
        session.mark_not_required("pure_data_parallel")
        return built_model

    monkeypatch.setattr(auto_model, "apply_model_infrastructure", apply_infrastructure)

    result = auto_model._BaseHyperAutoModelClass._build_model(
        None,
        is_hf_model=True,
        hf_config=config,
        mesh=None,
        sharding_planner=None,
        fsdp2_manager=None,
        backend=None,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        validate_placement=False,
        load_base_model=False,
        build_observer=session,
    )

    snapshot = result._hp_codegen_snapshot
    assert snapshot.status is CodegenStatus.NOT_REQUIRED
    assert snapshot.reason == "pure_data_parallel"
    assert snapshot.to_dict()["source"]["model_class"].endswith(":Module")


def test_native_auto_build_does_not_collect_without_explicit_session(monkeypatch):
    model = nn.Module()
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])

    monkeypatch.setattr(auto_model, "_model_init_context", lambda _is_meta: nullcontext())
    monkeypatch.setattr(
        auto_model,
        "_init_model",
        lambda *args, **kwargs: (False, model),
    )
    monkeypatch.setattr(auto_model, "_current_device", lambda: None)
    monkeypatch.setattr(
        auto_model.torch.distributed,
        "is_initialized",
        lambda: False,
    )
    def infrastructure(built_model, **kwargs):
        del kwargs
        return built_model

    monkeypatch.setattr(auto_model, "apply_model_infrastructure", infrastructure)

    result = auto_model._BaseHyperAutoModelClass._build_model(
        None,
        is_hf_model=True,
        hf_config=config,
        mesh=None,
        sharding_planner=None,
        fsdp2_manager=None,
        backend=None,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        validate_placement=False,
        load_base_model=False,
    )

    assert not hasattr(result, "_hp_codegen_snapshot")


def test_offline_auto_build_uses_observation_infrastructure(monkeypatch):
    """Offline mode never enters the native infrastructure execution path."""
    model = nn.Module()
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
    session = CodegenBuildSession()
    observation = ModelBuildObservation(
        session,
        publish=False,
        observation_only=True,
    )

    monkeypatch.setattr(auto_model, "_model_init_context", lambda is_meta: nullcontext())
    monkeypatch.setattr(auto_model, "_init_model", lambda *args, **kwargs: (False, model))
    monkeypatch.setattr(auto_model.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        auto_model,
        "_current_device",
        lambda: pytest.fail("offline observation queried the accelerator"),
    )
    monkeypatch.setattr(
        auto_model,
        "apply_model_infrastructure",
        lambda *_args, **_kwargs: pytest.fail("native infrastructure was applied"),
    )

    def observe(built_model, **kwargs):
        active_session = kwargs["build_observer"]
        active_session.record_replacements()
        active_session.mark_not_required("pure_data_parallel")
        return built_model

    monkeypatch.setattr(auto_model, "observe_model_infrastructure", observe)

    result = auto_model._BaseHyperAutoModelClass._build_model(
        None,
        is_hf_model=True,
        hf_config=config,
        mesh=None,
        sharding_planner=None,
        fsdp2_manager=None,
        backend=None,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        validate_placement=False,
        load_base_model=False,
        build_observer=session,
        observation=observation,
    )

    assert result._hp_codegen_snapshot.status is CodegenStatus.NOT_REQUIRED


def test_native_auto_build_finishes_sidecar_after_infrastructure_before_train(
    monkeypatch,
):
    """Sidecar finish runs after native infrastructure and before train mode."""
    model = nn.Module()
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
    session = CodegenBuildSession()
    events = []

    monkeypatch.setattr(auto_model, "_model_init_context", lambda _is_meta: nullcontext())
    monkeypatch.setattr(auto_model, "_init_model", lambda *args, **kwargs: (False, model))
    monkeypatch.setattr(auto_model, "_current_device", lambda: None)
    monkeypatch.setattr(auto_model.torch.distributed, "is_initialized", lambda: False)

    def apply_infrastructure(built_model, **kwargs):
        events.append("infrastructure")
        active_session = kwargs["build_observer"]
        active_session.record_replacements()
        active_session.mark_not_required("pure_data_parallel")
        return built_model

    def finish_sidecar(active_session, artifact_dir, on_published=None):
        del artifact_dir, on_published
        events.append("finish")
        return active_session.finish()

    monkeypatch.setattr(auto_model, "apply_model_infrastructure", apply_infrastructure)
    monkeypatch.setattr(auto_model, "_finish_codegen_sidecar", finish_sidecar)
    monkeypatch.setattr(
        model,
        "train",
        lambda mode=True: events.append("train") or model,
    )

    auto_model._BaseHyperAutoModelClass._build_model(
        None,
        is_hf_model=True,
        hf_config=config,
        mesh=None,
        sharding_planner=None,
        fsdp2_manager=None,
        backend=None,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        validate_placement=False,
        load_base_model=False,
        build_observer=session,
        publish_codegen=True,
    )

    expected = ["infrastructure", "finish", "train"]
    assert events == expected, f"codegen finish order mismatch: expected={expected}, got={events}"


def test_native_infrastructure_failure_is_not_treated_as_codegen_failure(monkeypatch):
    """Native build exceptions escape without being rewritten as sidecar failures."""
    model = nn.Module()
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
    session = CodegenBuildSession()

    monkeypatch.setattr(auto_model, "_model_init_context", lambda _is_meta: nullcontext())
    monkeypatch.setattr(auto_model, "_init_model", lambda *args, **kwargs: (False, model))
    monkeypatch.setattr(auto_model, "_current_device", lambda: None)
    monkeypatch.setattr(auto_model.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        auto_model,
        "apply_model_infrastructure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("native failure")),
    )

    with pytest.raises(ValueError, match="native failure"):
        auto_model._BaseHyperAutoModelClass._build_model(
            None,
            is_hf_model=True,
            hf_config=config,
            mesh=None,
            sharding_planner=None,
            fsdp2_manager=None,
            backend=None,
            peft_config=None,
            torch_dtype="auto",
            attn_implementation="sdpa",
            validate_placement=False,
            load_base_model=False,
            build_observer=session,
            publish_codegen=True,
        )

    assert session.status is CodegenStatus.COLLECTING, (
        "native failure must not mutate codegen status: "
        f"expected={CodegenStatus.COLLECTING}, got={session.status}"
    )


def test_from_config_rejects_removed_training_codegen_switch():
    """Training callers must migrate artifact generation to the codegen CLI."""
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])

    with pytest.raises(TypeError, match="training-side codegen arguments were removed"):
        auto_model._BaseHyperAutoModelClass.from_config(config, codegen=True)


def test_from_config_consumes_internal_build_observation(monkeypatch):
    """The CLI-only context reaches private build orchestration without public kwargs."""
    config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
    distributed_setup = SimpleNamespace(mesh_context=object())
    observation = ModelBuildObservation(
        CodegenBuildSession(),
        artifact_dir="bundle",
        observation_only=True,
    )
    captured = {}

    def instantiate(**kwargs):
        captured["infrastructure_device"] = kwargs["device"]
        return "planner", "fsdp"

    monkeypatch.setattr(auto_model, "instantiate_infrastructure", instantiate)
    monkeypatch.setattr(
        auto_model,
        "_current_device",
        lambda: pytest.fail("offline observation queried the accelerator"),
    )
    monkeypatch.setattr(auto_model, "get_is_hf_model", lambda *_args, **_kwargs: True)

    def _build_model(_cls, *_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(auto_model._BaseHyperAutoModelClass, "_build_model", classmethod(_build_model))

    with use_model_build_observation(observation):
        auto_model._BaseHyperAutoModelClass.from_config(
            config,
            distributed_setup=distributed_setup,
        )

    assert captured["build_observer"] is observation.observer
    assert captured["observation_artifact_dir"] == "bundle"
    assert captured["publish_codegen"] is True
    assert captured["observation"] is observation
    assert captured["infrastructure_device"].type == "meta"


def test_codegen_sidecar_restores_training_rng_state(monkeypatch):
    """Sidecar imports and emission cannot advance Python or Torch training RNG."""
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    session.record_plan({"param_plan": {"block": {"params": {}}}})
    session.record_plan_application({"applied_boundaries": ["block"]})
    cpu_state = torch.get_rng_state().clone()
    python_state = random.getstate()

    def consume_rng(_snapshot, _artifact_dir):
        torch.rand(4)
        random.random()

    monkeypatch.setattr(auto_model, "ensure_codegen_artifact_from_snapshot", consume_rng)

    auto_model._finish_codegen_sidecar(session, None)

    restored_cpu_state = torch.get_rng_state()
    restored_python_state = random.getstate()
    assert torch.equal(restored_cpu_state, cpu_state), (
        "Torch RNG changed across codegen sidecar: "
        f"expected={cpu_state.tolist()}, got={restored_cpu_state.tolist()}"
    )
    assert restored_python_state == python_state, (
        "Python RNG changed across codegen sidecar: "
        f"expected={python_state}, got={restored_python_state}"
    )


def test_codegen_sidecar_restores_rng_when_publication_fails(monkeypatch):
    """RNG restoration also runs when sidecar publication raises."""
    session = CodegenBuildSession()
    session.record_source({"model_type": "toy"})
    session.record_replacements()
    session.record_plan({"param_plan": {"block": {"params": {}}}})
    session.record_plan_application({"applied_boundaries": ["block"]})
    cpu_state = torch.get_rng_state().clone()
    python_state = random.getstate()

    def consume_rng_and_fail(_snapshot, _artifact_dir):
        torch.rand(4)
        random.random()
        raise OSError("injected sidecar failure")

    monkeypatch.setattr(
        auto_model,
        "ensure_codegen_artifact_from_snapshot",
        consume_rng_and_fail,
    )

    snapshot = auto_model._finish_codegen_sidecar(session, None)

    restored_cpu_state = torch.get_rng_state()
    restored_python_state = random.getstate()
    assert snapshot.status is CodegenStatus.FAILED, (
        f"sidecar failure should be isolated, got={snapshot.status}"
    )
    assert torch.equal(restored_cpu_state, cpu_state), (
        "Torch RNG changed after failed sidecar: "
        f"expected={cpu_state.tolist()}, got={restored_cpu_state.tolist()}"
    )
    assert restored_python_state == python_state, (
        "Python RNG changed after failed sidecar: "
        f"expected={python_state}, got={restored_python_state}"
    )

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
from types import SimpleNamespace

import pytest
from torch import nn

from hyper_parallel.codegen.modeling_backend import ModelingBackend
from hyper_parallel.codegen.plan.freeze import freeze_plan_application
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.snapshot import CodegenStatus
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.models._transformers import auto_model, model_builder


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
        codegen_session=session,
    )
    snapshot = session.finish()

    assert isinstance(model, nn.Module)
    assert source_shard_info is None
    assert calls == {"plan": 1, "apply": 1}
    assert snapshot.status is CodegenStatus.DERIVED
    assert snapshot.to_dict()["plan"]["mesh_dim_names"] == ["tp"]
    assert snapshot.to_dict()["plan_application"]["validate_mode"] is False


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
        codegen_session=session,
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
        session = kwargs["codegen_session"]
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
        backend=ModelingBackend.HF,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        validate_placement=False,
        load_base_model=False,
        codegen_session=session,
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
    infrastructure = lambda built_model, **kwargs: built_model
    monkeypatch.setattr(auto_model, "apply_model_infrastructure", infrastructure)

    result = auto_model._BaseHyperAutoModelClass._build_model(
        None,
        is_hf_model=True,
        hf_config=config,
        mesh=None,
        sharding_planner=None,
        fsdp2_manager=None,
        backend=ModelingBackend.HF,
        peft_config=None,
        torch_dtype="auto",
        attn_implementation="sdpa",
        validate_placement=False,
        load_base_model=False,
    )

    assert not hasattr(result, "_hp_codegen_snapshot")

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
"""Tests for the builder-owned codegen preparation path."""

from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from hyper_parallel.codegen import manager
from hyper_parallel.codegen.meta import CodegenMeta
from hyper_parallel.codegen.plan.offline_mesh import OfflineMesh
from hyper_parallel.codegen.patch_plan import (
    PatchEmissionPlan,
    build_patch_emission_plan,
    deserialize_patch_emission_plan,
    serialize_patch_emission_plan,
    snapshot_modules,
)
from hyper_parallel.codegen.spec.types import CodegenSpec, ParallelDimsSpec, SourceSpec
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, local_compute
from hyper_parallel.trainer.config import PlanOverride, Target

from . import helpers


@local_compute
def _local_compute_factory(*, module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Return a tiny compute function through the production resolver."""
    del module, mesh, tp_mesh, cp_mesh, ep_mesh

    def compute(owner, value):
        return owner.proj(value)

    return compute


class _LocalRegion(nn.Module):
    """Boundary fixture whose forward is replaced by Phase C."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(self, value):
        return self.proj(value)


def test_prepare_codegen_build_uses_builder_replacement_events():
    """Serialize only replacements that the shared builder installed."""
    entry = PlanOverride(
        match="model.0",
        module_type="tests.ut.codegen.helpers.SourceMLP",
        exact_type=False,
        replace_module=Target(
            helpers._replace_mlp,
            target_path="tests.ut.codegen.helpers._replace_mlp",
        ),
    )
    config = SimpleNamespace(plan_overrides=[entry])
    spec = CodegenSpec(
        source=SourceSpec(model_name_or_path="unused"),
        parallel_dims=ParallelDimsSpec(tp_size=2),
    )
    plan = ShardingPlan(mesh_dim_names=("tp",))

    with patch(
        "hyper_parallel.codegen.plan.derive.build_plan_overrides",
        return_value={},
    ), patch(
        "hyper_parallel.distributed._builder.planner.ShardingPlanner.plan",
        return_value=plan,
    ) as planner, patch(
        "hyper_parallel.models._transformers.model_builder._apply_materialization_adapter"
    ), patch(
        "hyper_parallel.models._transformers.model_builder.apply_sharding_plan"
    ) as apply_plan, patch(
        "hyper_parallel.codegen.manager._apply_recorded_phase_c"
    ) as apply_phase_c:
        result = manager._prepare_codegen_build(helpers.MetaModel(), config, spec)

    assert result.sharding_plan is plan
    assert isinstance(result.transformed_model.model[0], helpers.ReplacementMLP)
    assert len(result.patch_plan.module_replacements) == 1
    persisted = serialize_patch_emission_plan(result.patch_plan)
    assert persisted["module_replacements"] == [{
        "module_fqn": "model.0",
        "source_type": "tests.ut.codegen.helpers:SourceMLP",
        "replacement_type": "tests.ut.codegen.helpers:ReplacementMLP",
    }]
    assert (
        deserialize_patch_emission_plan(persisted)
        .module_replacements[0]
        .replacement_type
        is helpers.ReplacementMLP
    )
    assert result.module_overrides == (
        {
            "match": ["model.0"],
            "fqn": "model.0",
            "fqns": ["model.0"],
            "module_type": "tests.ut.codegen.helpers.SourceMLP",
            "factory": "tests.ut.codegen.helpers._replace_mlp",
            "exact_type": False,
        },
    )
    planner.assert_called_once()
    apply_plan.assert_not_called()
    apply_phase_c.assert_called_once()


def test_fill_plan_fields_keeps_builder_patches_runtime_owned():
    """Do not lower planning-time builder mutations into generated source."""
    replacement = SimpleNamespace(module_fqn="model.0")
    local_compute_patch = SimpleNamespace(module_fqn="model.0")
    patch_plan = PatchEmissionPlan(
        module_replacements=(replacement,),
        local_computes=(local_compute_patch,),
    )
    build_input = manager.CodegenBuildInput(
        source_model=object(),
        transformed_model=object(),
        sharding_plan=object(),
        patch_plan=patch_plan,
        module_overrides=({"fqn": "model.0"},),
    )
    frozen = SimpleNamespace(
        param_plan={"model.0": {}}, frozen_sharded_params=[], injections=[],
        tied_pairs=[], special_handlers=[], boundary_classes={}, mesh_dim_names=("tp",),
    )
    meta = CodegenMeta(
        codegen_version="test", signature="test", yaml_path="test.yaml",
        yaml_sha256="test", source={}, parallel_dims={}, model_class="ToyCustomModel",
    )

    with patch(
        "hyper_parallel.codegen.spec.project.project_codegen_spec",
        return_value=SimpleNamespace(source=SimpleNamespace(model_name_or_path="unused")),
    ), patch(
        "hyper_parallel.codegen.plan.derive.build_meta_model", return_value=object(),
    ), patch(
        "hyper_parallel.codegen.manager._prepare_codegen_build", return_value=build_input,
    ), patch(
        "hyper_parallel.codegen.plan.freeze.freeze_plan", return_value=frozen,
    ), patch(
        "hyper_parallel.codegen.patch_plan.serialize_patch_emission_plan",
        return_value={"module_replacements": [{"module_fqn": "model.0"}]},
    ), patch(
        "hyper_parallel.models.registry._resolve_custom_model_cls",
        return_value=object(),
    ):
        emitted = manager._fill_plan_fields(meta, SimpleNamespace(), SimpleNamespace())

    assert meta.covered["module_overrides"] is False
    assert meta.module_overrides == []
    assert meta.emission_patches["module_replacements"] == [{"module_fqn": "model.0"}]
    assert emitted.module_replacements == ()
    assert emitted.local_computes == ()


def test_offline_phase_c_records_after_installing_local_region_forward():
    """Codegen runs the trainer Phase C without parameter sharding."""
    model = nn.Module()
    model.region = _LocalRegion()
    original_forward = model.region.forward
    spec = ModuleShardingSpec(
        params={},
        in_src={},
        in_dst={},
        out_src={},
        out_dst={},
        local_compute_fn=_local_compute_factory,
        region_dispatch=False,
    )
    plan = ShardingPlan(
        modules={"region": spec},
        mesh_dim_names=("tp",),
    )
    mesh = OfflineMesh(("tp",), (2,))
    assert mesh.get_group() is None

    before_phase_c = snapshot_modules(model)
    manager._apply_recorded_phase_c(model, plan, mesh)
    patch_plan = build_patch_emission_plan(
        model,
        plan,
        (),
        before_phase_c,
        before_phase_c,
        snapshot_modules(model),
    )

    assert len(patch_plan.local_computes) == 1
    assert patch_plan.local_computes[0].module_fqn == "region"
    assert model.region.forward is not original_forward

    persisted = serialize_patch_emission_plan(patch_plan)
    restored = deserialize_patch_emission_plan(persisted)
    assert persisted["local_computes"] == [{
        "module_fqn": "region",
        "module_type": (
            "tests.ut.codegen.test_builder_driven_manager:_LocalRegion"
        ),
        "factory": (
            "tests.ut.codegen.test_builder_driven_manager:_local_compute_factory"
        ),
        "region_dispatch": False,
        "original_forward": (
            "tests.ut.codegen.test_builder_driven_manager:_LocalRegion.forward"
        ),
    }]
    assert restored.local_computes[0].module_type is _LocalRegion
    assert restored.local_computes[0].factory is _local_compute_factory

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
"""Contracts for Manifest/ShardingPlan to immutable codegen IR projection."""

from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.codegen.project import project_model_build, publish_codegen_ir
from hyper_parallel.models.build_capture import ModelBuildCapture
from hyper_parallel.models.preparation_manifest import (
    PreparationManifest,
    PreparedComponent,
    ReplacementApplication,
    SymbolRef,
)


def _component() -> PreparedComponent:
    """Return one stable replacement component."""
    return PreparedComponent(
        boundary_id="family:linear:block",
        parent_boundary_id=None,
        emission_owner_boundary_id="family:linear:block",
        origin="direct",
        aliases=("block",),
        resolved_member=SymbolRef.from_object(nn.Linear),
    )


def _capture(*, plan=None, manifest=None) -> ModelBuildCapture:
    """Build a complete resource-bearing capture for projection tests."""
    model = nn.Module()
    model.block = nn.Linear(2, 2)
    return ModelBuildCapture(
        source={"model_type": "toy", "architecture": "ToyModel"},
        manifest=manifest or PreparationManifest(
            model_spec_id="toy:ToyModel",
            components=(_component(),),
        ),
        model=model,
        plan=plan,
        mesh=SimpleNamespace(
            dp_size=1,
            tp_size=2,
            cp_size=1,
            pp_size=1,
            ep_size=1,
            sequence_parallel=True,
            loss_parallel=False,
        ),
        reason="pure_data_parallel" if plan is None else None,
        planning_finished=True,
    )


class TestCodegenProjection(unittest.TestCase):
    """Verify projection copies stable facts and rejects broken ownership."""

    def test_projection_owns_immutable_data_without_runtime_resources(self):
        """The IR must not retain the model, plan, mesh, tensor, or mutable input maps."""
        spec = SimpleNamespace(is_boundary=True)
        plan = SimpleNamespace(
            modules={"block": spec},
            mesh_dim_names=("tp",),
        )
        capture = _capture(plan=plan)

        build_ir = project_model_build(capture)
        capture.source["model_type"] = "mutated"
        plan.modules.clear()

        self.assertEqual(build_ir.source["model_type"], "toy")
        self.assertEqual(build_ir.plan["param_plan"], {"block": {"is_boundary": True}})
        with self.assertRaises(TypeError):
            build_ir.source["model_type"] = "forbidden"

    def test_projection_observes_the_final_plan_once(self):
        """Projection must consume, not recompute, the Trainer-produced plan."""
        capture = _capture(plan=SimpleNamespace(modules={}, mesh_dim_names=("tp",)))
        observed = SimpleNamespace(to_dict=lambda: {
            "param_plan": {},
            "injections": [],
            "boundary_classes": {},
            "boundary_types": {},
            "nested_components": [],
            "mesh_dim_names": ("tp",),
        })
        with patch(
            "hyper_parallel.codegen.project.observe_emission_plan",
            return_value=observed,
        ) as project:
            build_ir = project_model_build(capture)

        project.assert_called_once_with(capture.plan, capture.model)
        self.assertEqual(build_ir.plan["parallel_dims"]["tp_size"], 2)

    def test_projection_rejects_unknown_component_owner(self):
        """Broken manifest links fail before the emission boundary."""
        component = replace(_component(), emission_owner_boundary_id="missing")
        manifest = PreparationManifest(
            model_spec_id="toy:ToyModel",
            components=(component,),
        )

        with self.assertRaisesRegex(ValueError, "unknown emission_owner_boundary_id"):
            project_model_build(_capture(manifest=manifest))

    def test_nested_replacement_has_exact_constructor_owner(self):
        """A nested layout must not turn into an extra module replacement."""
        parent = replace(_component(), origin="replacement", declaration_id="install_block")
        child = replace(
            _component(), boundary_id="family:linear:experts", aliases=("block.experts",),
            parent_boundary_id=parent.boundary_id, emission_owner_boundary_id=parent.boundary_id,
        )
        linear = SymbolRef.from_object(nn.Linear)
        application = ReplacementApplication(
            boundary_id=parent.boundary_id, declaration_id=parent.declaration_id, aliases=parent.aliases,
            source_type=linear, replacement_type=linear,
        )
        manifest = PreparationManifest(
            model_spec_id="toy:ToyModel", components=(parent, child),
            declarations=({"rule_id": parent.declaration_id},), replacements=(application,),
        )
        plan = SimpleNamespace(
            modules={fqn: SimpleNamespace(is_boundary=True) for fqn in ("block", "block.experts")},
            mesh_dim_names=("ep",),
        )
        capture = _capture(plan=plan, manifest=manifest)
        capture.model.block.experts = nn.Linear(2, 2)

        build_ir = project_model_build(capture)

        self.assertEqual(set(build_ir.plan["param_plan"]), {"block", "block.experts"})
        self.assertEqual([item["primary_fqn"] for item in build_ir.replacements], ["block"])

        invalid = replace(manifest, replacements=(replace(
            application, boundary_id=child.boundary_id, aliases=child.aliases,
        ),))
        with self.assertRaisesRegex(ValueError, "no source constructor owner"):
            project_model_build(_capture(manifest=invalid))

    def test_replacement_records_preserve_native_application_order(self):
        """A sorted component inventory must not reorder source edits."""
        linear = SymbolRef.from_object(nn.Linear)
        components = tuple(
            replace(
                _component(), boundary_id=name, emission_owner_boundary_id=name, aliases=(name,),
                origin="replacement", declaration_id=name,
            )
            for name in ("first", "second")
        )
        applications = tuple(
            ReplacementApplication(
                boundary_id=name, declaration_id=name, aliases=(name,),
                source_type=linear, replacement_type=linear,
            )
            for name in ("second", "first")
        )
        manifest = PreparationManifest(
            model_spec_id="toy:ToyModel",
            components=components,
            declarations=tuple({"rule_id": name} for name in ("second", "first")),
            replacements=applications,
        )

        build_ir = project_model_build(_capture(manifest=manifest))

        self.assertEqual([item["primary_fqn"] for item in build_ir.replacements], ["second", "first"])

    def test_publication_restores_python_and_torch_rng(self):
        """The transitional emitter cannot advance host RNG state."""
        build_ir = project_model_build(_capture())
        cpu_state = torch.get_rng_state().clone()
        python_state = random.getstate()

        def _consume_rng(_snapshot, _artifact_dir):
            torch.rand(4)
            random.random()
            return object()

        with patch(
            "hyper_parallel.codegen.project.ensure_codegen_artifact_from_snapshot",
            side_effect=_consume_rng,
        ):
            result = publish_codegen_ir(build_ir, None)

        self.assertIsNotNone(result.artifact_layout)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
        self.assertEqual(random.getstate(), python_state)


if __name__ == "__main__":
    unittest.main()

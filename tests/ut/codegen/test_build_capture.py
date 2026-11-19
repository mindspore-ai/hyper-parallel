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
"""Unit tests for build-local model preparation capture."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules import family as family_registry
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models import build_capture
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models.build_capture import capture_model_build_plan
from hyper_parallel.models.module_build import (
    attach_build_context,
    build_module_context,
    prepare_model_components,
)
from hyper_parallel.models.replacement import ModuleReplacementSpec
from tests.common.module_family_fixtures import (
    FamilyModelConfig,
    create_family_model,
    register_linear_family,
)
from tests.ut.codegen.helpers import SourceMLP, _replace_mlp


class TestModelBuildCapture(unittest.TestCase):
    """Exercise context isolation and shared preparation/planning capture."""

    def test_nested_capture_restores_outer_context(self):
        """A nested capture must not write source facts into its parent."""
        outer_model = nn.Module()
        outer_model.config = SimpleNamespace(model_type="outer", architectures=["OuterModel"])
        inner_model = nn.Module()
        inner_model.config = SimpleNamespace(model_type="inner", architectures=["InnerModel"])

        with capture_model_build_plan() as outer:
            with capture_model_build_plan() as inner:
                build_capture.record_model_source_for_capture(
                    inner_model,
                    pretrained_model_name_or_path=None,
                    is_hf_model=True,
                )
            build_capture.record_model_source_for_capture(
                outer_model,
                pretrained_model_name_or_path=None,
                is_hf_model=True,
            )

        self.assertEqual(inner.source["model_type"], "inner")
        self.assertEqual(outer.source["model_type"], "outer")
        self.assertFalse(build_capture.is_model_build_capture_active())

    def test_capture_resets_after_exception(self):
        """An exceptional build must not leak capture state to later builds."""
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            with capture_model_build_plan():
                self.assertTrue(build_capture.is_model_build_capture_active())
                raise RuntimeError("build failed")

        self.assertFalse(build_capture.is_model_build_capture_active())

    def test_consecutive_captures_do_not_share_state(self):
        """Sequential build sessions own independent result objects."""
        captures = []
        for model_type in ("first", "second"):
            model = nn.Module()
            model.config = SimpleNamespace(model_type=model_type, architectures=[])
            with capture_model_build_plan() as capture:
                build_capture.record_model_source_for_capture(
                    model,
                    pretrained_model_name_or_path=None,
                    is_hf_model=True,
                )
            captures.append(capture)

        self.assertIsNot(captures[0], captures[1])
        self.assertEqual(captures[0].source["model_type"], "first")
        self.assertEqual(captures[1].source["model_type"], "second")

    def test_capture_is_isolated_between_threads(self):
        """Concurrent build contexts must not observe another thread's source."""
        def _record_source(model_type):
            model = nn.Module()
            model.config = SimpleNamespace(model_type=model_type, architectures=[])
            with capture_model_build_plan() as capture:
                build_capture.record_model_source_for_capture(
                    model,
                    pretrained_model_name_or_path=model_type,
                    is_hf_model=True,
                )
                self.assertTrue(build_capture.is_model_build_capture_active())
            self.assertFalse(build_capture.is_model_build_capture_active())
            return capture.source

        with ThreadPoolExecutor(max_workers=2) as executor:
            sources = list(executor.map(_record_source, ("left", "right")))

        self.assertEqual([source["model_type"] for source in sources], ["left", "right"])
        self.assertEqual(
            [source["pretrained_model_name_or_path"] for source in sources],
            ["left", "right"],
        )

    def test_shared_replacement_records_only_installed_candidate(self):
        """Shared preparation records a replacement after atomic installation."""
        model = nn.Sequential(SourceMLP())
        source = model[0]
        model.config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
        setup = SimpleNamespace(
            mesh_context=None,
            module_replacement_requests=None,
            module_replacements=(ModuleReplacementSpec(("0",), _replace_mlp),),
        )
        context = build_module_context(setup)
        attach_build_context(model, context)

        with capture_model_build_plan() as capture:
            build_capture.record_model_source_for_capture(
                model,
                pretrained_model_name_or_path=None,
                is_hf_model=True,
            )
            prepared, _ = prepare_model_components(model, context)
            build_capture.record_model_plan_for_capture(prepared, None, None)

        manifest = capture.manifest.to_legacy_records()
        self.assertIsNot(prepared[0], source)
        self.assertEqual(len(manifest["declarations"]), 1)
        self.assertEqual(len(manifest["applications"]), 1)
        self.assertEqual(manifest["applications"][0]["primary_fqn"], "0")
        self.assertTrue(
            manifest["applications"][0]["source_type"].endswith(":SourceMLP")
        )

    def test_plan_capture_runs_shared_preflight_without_apply(self):
        """Offline planning validates the native plan but does not mutate parameters."""
        model = nn.Module()
        model.config = SimpleNamespace(model_type="toy", architectures=["ToyModel"])
        plan = SimpleNamespace(mesh_dim_names=("tp",))
        planner = SimpleNamespace(plan=lambda *_args, **_kwargs: plan)
        mesh = SimpleNamespace(
            device_mesh=object(),
            tp_size=2,
            cp_size=1,
            ep_size=1,
            sequence_parallel=False,
            loss_parallel=False,
        )

        with capture_model_build_plan() as capture:
            build_capture.record_model_source_for_capture(
                model,
                pretrained_model_name_or_path=None,
                is_hf_model=True,
            )
            prepared, _ = prepare_model_components(model, build_module_context(None))
            with patch.object(model_builder, "preflight_sharding_plan") as preflight, patch.object(
                model_builder,
                "apply_sharding_plan",
                side_effect=AssertionError("capture applied a sharding plan"),
            ):
                observed, source_info = model_builder._plan_and_apply_sharding(  # pylint: disable=protected-access
                    model,
                    mesh,
                    planner,
                    True,
                    False,
                )

        self.assertIs(observed, model)
        self.assertIsNone(source_info)
        self.assertIs(capture.plan, plan)
        self.assertTrue(capture.complete)
        preflight.assert_called_once_with(model, plan, mesh)

    def test_auto_model_uses_meta_and_stops_after_planning(self):
        """The normal AutoModel entry reaches capture without runtime materialization."""
        with patch.dict(family_registry._FAMILIES, clear=True):
            register_linear_family()
            setup = DistributedSetup(mesh_context=MeshContext())
            prepared_specs = []
            with patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                    capture_model_build_plan(prepare_model_spec=prepared_specs.append) as capture:
                model = HyperAutoModelForCausalLM.from_config(
                    FamilyModelConfig(),
                    model_factory=create_family_model,
                    distributed_setup=setup,
                    torch_dtype=torch.float32,
                )

        self.assertTrue(capture.complete)
        self.assertEqual(capture.reason, "pure_data_parallel")
        self.assertEqual(prepared_specs, [None])
        self.assertEqual(model.projection.weight.device.type, "meta")
        self.assertFalse(hasattr(model, "_source_shard_info"))


if __name__ == "__main__":
    unittest.main()

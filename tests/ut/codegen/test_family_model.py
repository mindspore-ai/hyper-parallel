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
"""Atomic construction artifacts and numerical-declaration cache invalidation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyper_parallel.codegen.family_model import prepare_direct_model_artifact
from hyper_parallel.components.modules import family as registry
from hyper_parallel.models.module_build import build_module_context, construct_family_model, prepare_model_components
from tests.common.mark_utils import arg_mark
from tests.common.module_family_fixtures import (
    load_direct_model_entry,
    FamilyModel, FamilyModelConfig, InheritedFamilyModel, LinearFamily, create_family_model, register_linear_family,
)


class TestFamilyModelArtifact(unittest.TestCase):
    """Exercise real file publication and stale-model rejection."""

    def setUp(self) -> None:
        """Keep test declarations local to each case."""
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_shared_cache_publishes_complete_manifest(self) -> None:
        """
        Feature: Multiple ranks may share a construction artifact directory.
        Description: Concurrent publishers must expose complete JSON and a valid importable entry.
        Expectation: Every published signature agrees and a reconstructed model matches the requested shape.
        """
        config = FamilyModelConfig()
        context = build_module_context()
        model = construct_family_model(create_family_model, config, context=context)
        model, _ = prepare_model_components(model, context)
        with tempfile.TemporaryDirectory() as folder, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(prepare_direct_model_artifact, model, create_family_model,
                                   config, context, artifact_dir=folder) for _ in range(8)]
            self.assertEqual([future.result() for future in futures], [folder] * 8)
            manifest = json.loads((Path(folder) / "family_manifest.json").read_text())
            self.assertEqual(len(manifest["signature"]), 64)
            restored = load_direct_model_entry(folder)(config, context=context)
            self.assertEqual(tuple(restored.projection.weight.shape), (3, 4))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_family_version_invalidates_cached_constructor(self) -> None:
        """
        Feature: Numerical contracts are part of artifact compatibility.
        Description: An unchanged public factory path cannot hide a family semantic-version change.
        Expectation: Cached constructors reject a new declaration and request artifact regeneration.
        """
        config = FamilyModelConfig()
        context = build_module_context()
        model = construct_family_model(create_family_model, config, context=context)
        model, _ = prepare_model_components(model, context)
        with tempfile.TemporaryDirectory() as folder:
            prepare_direct_model_artifact(model, create_family_model, config, context, artifact_dir=folder)
            entry = load_direct_model_entry(folder)
            declaration = registry.get_module_family(LinearFamily)
            registry._FAMILIES.clear()
            registry.register_module_family(replace(declaration, version="2"))
            with self.assertRaisesRegex(ValueError, "signature changed"):
                entry(config, context=context)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_inherited_model_algorithm_invalidates_cached_constructor(self) -> None:
        """Feature: Cached constructors account for inherited model algorithms.
        Description: Change only the base definition while the imported factory subclass stays identical.
        Expectation: The generated constructor rejects its obsolete signature.
        """
        config = FamilyModelConfig()
        context = build_module_context()
        model = construct_family_model(InheritedFamilyModel, config, context=context)
        model, _ = prepare_model_components(model, context)
        original_source = inspect.getsource

        def changed_base(definition: object) -> str:
            """Simulate an inherited implementation change without editing a live fixture."""
            source = original_source(definition)
            return source + "\n# revised inherited algorithm\n" if definition is FamilyModel else source

        with tempfile.TemporaryDirectory() as folder:
            prepare_direct_model_artifact(model, InheritedFamilyModel, config, context, artifact_dir=folder)
            entry = load_direct_model_entry(folder)
            with patch("hyper_parallel.components.modules.module_family.inspect.getsource", side_effect=changed_base):
                with self.assertRaisesRegex(ValueError, "signature changed"):
                    entry(config, context=context)

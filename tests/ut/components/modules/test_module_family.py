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
"""Numerical and lifecycle acceptance of direct and deferred family construction."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.module_family import (
    FamilyRequest, family_plan_snapshot, get_family_build_info, validate_family_context,
)
from hyper_parallel.trainer.config.replacement import ReplacementTarget
from tests.common.mark_utils import arg_mark
from tests.common.module_family_fixtures import (
    LinearConfig, LinearFamily, LocalLinear, MegaLinear, ParallelLinear, register_linear_family,
)


class TestModuleFamilyConstruction(unittest.TestCase):
    """Exercise real construction and autograd; no distributed setup is needed."""

    def setUp(self) -> None:
        """Keep the process registry and random generator unchanged for other suites."""
        registry_patch = patch.dict(registry._FAMILIES, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        register_linear_family()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_direct_and_source_share_math_and_parameter_contract(self):
        """
        Feature: One family for direct and source construction.
        Description: Equivalent weights preserve outputs, input gradients, and frozen parameter semantics.
        Expectation: Both origins expose the same mathematical operation and target state keys.
        """
        direct = LinearFamily(config=LinearConfig())
        source = nn.Linear(4, 3)
        source.load_state_dict(direct.state_dict())
        source.bias.requires_grad_(False)
        source.eval()
        converted = LinearFamily(module=source)
        self.assertIs(converted.weight, source.weight)
        self.assertFalse(converted.bias.requires_grad)
        self.assertFalse(converted.training)
        x = torch.randn(5, 4, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        torch.testing.assert_close(direct(x), converted(y))
        direct(x).sum().backward()
        converted(y).sum().backward()
        torch.testing.assert_close(x.grad, y.grad)
        self.assertEqual(set(direct.state_dict()), set(converted.state_dict()))
        self.assertEqual(get_family_build_info(direct).origin, "direct")
        self.assertEqual(get_family_build_info(converted).origin, "replacement")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_context_selects_once_for_each_reused_target(self):
        """
        Feature: Topology-dependent selection is deferred.
        Description: One target configuration is safely reused by independent jobs.
        Expectation: Local and parallel jobs create different concrete classes without mutating the request.
        """
        target = ReplacementTarget(LinearFamily, target_path="tests.common.module_family_fixtures.LinearFamily")
        self.assertIs(target.callable, LinearFamily)
        source = nn.Linear(4, 3)
        local = target.build(module=source, context={"ep_size": 1})
        parallel = target.build(module=source, context={"ep_size": 4})
        self.assertIs(type(local), LocalLinear)
        self.assertIs(type(parallel), ParallelLinear)
        self.assertEqual(target.to_dict(), {"_target_": "tests.common.module_family_fixtures.LinearFamily"})
        self.assertIs(type(LinearFamily(config=LinearConfig(), kernel="megakernel")), MegaLinear)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_backend_is_static_and_inherited_by_parallel_member(self):
        """
        Feature: Explicit backend contracts.
        Description: Parallel selection keeps the chosen local backend and rejects undeclared combinations.
        Expectation: No installed-package probing or fallback changes an effective backend.
        """
        member = LinearFamily(config=LinearConfig(), backend="alternate", context={"ep_size": 2})
        self.assertIs(type(member), ParallelLinear)
        self.assertEqual(member.backend, "alternate")
        with self.assertRaisesRegex(ValueError, "does not support"):
            LinearFamily(config=LinearConfig(), kernel="megakernel", backend="alternate")
        with self.assertRaisesRegex(ValueError, "unknown backend"):
            LinearFamily(config=LinearConfig(), backend="missing")
        with self.assertRaisesRegex(ValueError, "no legal combination"):
            LinearFamily(config=LinearConfig(), precision="unimplemented")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_meta_construction_allocates_no_communication(self):
        """
        Feature: Meta-first direct construction.
        Description: Final member selection precedes materialization and runtime resource binding.
        Expectation: Meta parameters have complete logical shapes and no process group is created.
        """
        with patch("torch.distributed.new_group") as new_group, torch.device("meta"):
            model = LinearFamily(config=LinearConfig(), context={"ep_size": 2})
        self.assertIs(type(model), ParallelLinear)
        self.assertEqual(tuple(model.weight.shape), (3, 4))
        self.assertTrue(model.weight.is_meta)
        new_group.assert_not_called()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_invalid_construction_and_unprepared_context_fail(self):
        """
        Feature: Family validation checks the effective preparation context.
        Description: Invalid construction and an unprepared local member are rejected.
        Expectation: A context mismatch requests model preparation before parameter layout.
        """
        with self.assertRaisesRegex(ValueError, "exactly one"):
            LinearFamily()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            LinearFamily(config=LinearConfig(), module=nn.Linear(4, 3))
        with self.assertRaisesRegex(TypeError, "unknown"):
            FamilyRequest(LinearFamily, {"context": {"ep_size": 9}})
        model = nn.Sequential(LinearFamily(config=LinearConfig()))
        with self.assertRaisesRegex(ValueError, "prepare_model_components"):
            validate_family_context(model, {"ep_size": 2})
        with self.assertRaisesRegex(ValueError, "ep_size is required"):
            LinearFamily(config=LinearConfig(), context={"ep": True})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_snapshot_includes_defaults_and_preserves_aliases(self):
        """
        Feature: Reproducible family metadata.
        Description: Shared module paths and backend defaults participate in cache validation.
        Expectation: Alias paths are discoverable and changing a declaration changes the effective snapshot.
        """
        model = nn.Module()
        model.first = LinearFamily(config=LinearConfig())
        model.second = model.first
        initial = family_plan_snapshot(model)
        declaration = registry.get_module_family(LinearFamily)
        registry._FAMILIES.clear()
        registry.register_module_family(replace(
            declaration, default_backends={"default": "alternate", "mega": "torch"},
        ))
        newer = nn.Module()
        newer.first = LinearFamily(config=LinearConfig())
        newer.second = newer.first
        self.assertNotEqual(initial, family_plan_snapshot(newer))

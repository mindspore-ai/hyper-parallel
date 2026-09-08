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
"""Unit tests for ``hyper_parallel.compile.sharding_config``.

Covers:

1. ``PassPlan.fsdp_wrap`` / ``fsdp_wrap_pattern`` register entries and
   support chaining.
2. ``is_fsdp_module`` exact-match wins over patterns; patterns match via
   ``fnmatch`` semantics.
3. ``get_fsdp_config`` returns the matching config (or ``None``).
4. ``merge`` returns a *new* plan (does not mutate inputs).
5. ``create_simple_sharding_plan`` wraps everything via the ``*`` pattern.
6. ``create_sharding_plan_from_yaml`` happy path + validation.
"""

import os
import tempfile
import textwrap
import unittest

import yaml

from hyper_parallel.compile.sharding_config import (
    FSDPModuleConfig,
    PassPlan,
    create_sharding_plan_from_yaml,
    create_simple_sharding_plan,
)


class TestPassPlanRegistration(unittest.TestCase):
    """``fsdp_wrap`` / ``fsdp_wrap_pattern`` register entries; chainable."""

    def test_fsdp_wrap_registers_exact_match(self):
        """Test ``fsdp_wrap`` adds an exact-match entry."""
        plan = PassPlan()
        result = plan.fsdp_wrap("tok_embeddings")
        self.assertIs(result, plan, ("fsdp_wrap should return self for chaining"))
        self.assertIn("tok_embeddings", plan.fsdp_modules)
        cfg = plan.fsdp_modules["tok_embeddings"]
        self.assertIsInstance(cfg, FSDPModuleConfig)
        self.assertEqual(cfg.module_fqn, "tok_embeddings")

    def test_fsdp_wrap_pattern_registers_pattern(self):
        """Test ``fsdp_wrap_pattern`` adds a wildcard-pattern entry."""
        plan = PassPlan()
        result = plan.fsdp_wrap_pattern("layers.*")
        self.assertIs(result, plan)
        self.assertIn("layers.*", plan.fsdp_patterns)
        self.assertEqual(plan.fsdp_patterns["layers.*"].module_fqn, "layers.*")

    def test_chaining(self):
        """Test multiple builder calls chain on one plan."""
        plan = (
            PassPlan()
            .fsdp_wrap("tok_embeddings")
            .fsdp_wrap("head")
            .fsdp_wrap_pattern("layers.*")
        )
        self.assertEqual(len(plan.fsdp_modules), 2)
        self.assertEqual(len(plan.fsdp_patterns), 1)


class TestPassPlanMatching(unittest.TestCase):
    """``is_fsdp_module`` / ``get_fsdp_config`` match semantics."""

    def setUp(self) -> None:
        """Build a plan with one exact-match and one pattern entry."""
        self.plan = (
            PassPlan().fsdp_wrap("tok_embeddings").fsdp_wrap_pattern("layers.*")
        )

    def test_is_fsdp_module_exact_match(self):
        """Test exact-match FQN is recognized."""
        self.assertTrue(self.plan.is_fsdp_module("tok_embeddings"))

    def test_is_fsdp_module_pattern_match(self):
        """Test wildcard pattern matches via fnmatch."""
        self.assertTrue(self.plan.is_fsdp_module("layers.0"))
        self.assertTrue(self.plan.is_fsdp_module("layers.42.attention"))

    def test_is_fsdp_module_no_match(self):
        """Test unrelated FQN returns False."""
        self.assertFalse(self.plan.is_fsdp_module("embed"))
        self.assertFalse(self.plan.is_fsdp_module("layers"))  # pattern is layers.*

    def test_get_fsdp_config_returns_exact_first(self):
        """Test exact-match config is returned before pattern fallback."""
        cfg = self.plan.get_fsdp_config("tok_embeddings")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.module_fqn, "tok_embeddings")

    def test_get_fsdp_config_returns_pattern_config(self):
        """Test pattern-matched FQN returns the pattern's config."""
        cfg = self.plan.get_fsdp_config("layers.0")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.module_fqn, "layers.*")

    def test_get_fsdp_config_returns_none_when_unmatched(self):
        """Test unmatched FQN returns None."""
        self.assertIsNone(self.plan.get_fsdp_config("nope"))

    def test_first_pattern_wins_on_overlap(self):
        """Test first-inserted pattern wins when patterns overlap."""
        plan = (
            PassPlan()
            .fsdp_wrap_pattern("layers.*")  # inserted first
            .fsdp_wrap_pattern("layers.0.*")  # also matches layers.0.x
        )
        cfg = plan.get_fsdp_config("layers.0.attention")
        self.assertEqual(
            cfg.module_fqn,
            "layers.*",
            f"first-inserted pattern should win, got {cfg.module_fqn}",
        )


class TestPassPlanMerge(unittest.TestCase):
    """``merge`` returns a new plan; inputs are not mutated."""

    def test_merge_returns_new_plan(self):
        """Test merge returns a distinct object."""
        a = PassPlan().fsdp_wrap("a")
        b = PassPlan().fsdp_wrap("b")
        merged = a.merge(b)
        self.assertIsNot(merged, a, "merge should not return self")
        self.assertIsNot(merged, b, "merge should not return other")

    def test_merge_combines_registries(self):
        """Test merge combines both plans' modules and patterns."""
        a = PassPlan().fsdp_wrap("a").fsdp_wrap_pattern("layers.*")
        b = PassPlan().fsdp_wrap("b").fsdp_wrap_pattern("blocks.*")
        merged = a.merge(b)
        self.assertTrue(merged.is_fsdp_module("a"))
        self.assertTrue(merged.is_fsdp_module("b"))
        self.assertTrue(merged.is_fsdp_module("layers.0"))
        self.assertTrue(merged.is_fsdp_module("blocks.0"))

    def test_merge_does_not_mutate_inputs(self):
        """Test merge leaves both input plans unchanged."""
        a = PassPlan().fsdp_wrap("a")
        b = PassPlan().fsdp_wrap("b")
        a.merge(b)
        self.assertNotIn("b", a.fsdp_modules, ("merge should not mutate the receiver"))
        self.assertNotIn(
            "a", b.fsdp_modules, ("merge should not mutate the other plan")
        )

    def test_merge_other_wins_on_conflict(self):
        """Test entries in ``other`` overwrite entries with the same key."""
        a = PassPlan().fsdp_wrap("shared")
        b = PassPlan().fsdp_wrap("shared")  # same key
        a_cfg = a.fsdp_modules["shared"]
        b_cfg = b.fsdp_modules["shared"]
        merged = a.merge(b)
        self.assertIn("shared", merged.fsdp_modules)
        # The conflicting key collapses to a single entry, and that entry is
        # ``other``'s config object (b wins), not the receiver's (a).
        self.assertEqual(
            len(merged.fsdp_modules),
            1,
            f"conflicting key should collapse to 1 entry, "
            f"got {len(merged.fsdp_modules)}",
        )
        self.assertIs(
            merged.fsdp_modules["shared"],
            b_cfg,
            "other's config must overwrite the receiver's on key conflict",
        )
        self.assertIsNot(
            merged.fsdp_modules["shared"],
            a_cfg,
            "receiver's config must not survive a key conflict",
        )


class TestCreateSimplePassPlan(unittest.TestCase):
    """``create_simple_sharding_plan`` wraps everything via ``*``."""

    def test_wraps_all_modules(self):
        """Test the simple plan matches any FQN via ``*`` pattern."""
        plan = create_simple_sharding_plan()
        self.assertTrue(plan.is_fsdp_module("anything"))
        self.assertTrue(plan.is_fsdp_module("layers.0.attention.weight"))
        self.assertIn("*", plan.fsdp_patterns)
        self.assertEqual(len(plan.fsdp_modules), 0)


class TestCreatePassPlanFromYaml(unittest.TestCase):
    """YAML loading happy path + validation rules."""

    def test_loads_modules_and_patterns(self):
        """Test a YAML with both modules and patterns loads both."""
        yaml_text = textwrap.dedent("""
        fsdp:
          enabled: true
          modules:
            - name: tok_embeddings
            - name: head
          patterns:
            - pattern: "layers.*"
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            plan = create_sharding_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

        self.assertTrue(plan.is_fsdp_module("tok_embeddings"))
        self.assertTrue(plan.is_fsdp_module("head"))
        self.assertTrue(plan.is_fsdp_module("layers.0"))

    def test_enabled_false_skips_loading(self):
        """Test ``enabled: false`` skips loading modules/patterns."""
        yaml_text = textwrap.dedent("""
        fsdp:
          enabled: false
          modules:
            - name: tok_embeddings
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            plan = create_sharding_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

        self.assertFalse(plan.is_fsdp_module("tok_embeddings"))

    def test_implicit_enable_when_modules_present(self):
        """Test ``enabled`` defaults True when modules are listed."""
        yaml_text = textwrap.dedent("""
        fsdp:
          modules:
            - name: tok_embeddings
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            plan = create_sharding_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

        self.assertTrue(plan.is_fsdp_module("tok_embeddings"))

    def test_requires_path_or_model_name(self):
        """Test ValueError when neither config_path nor model_name is given."""
        with self.assertRaises(ValueError):
            create_sharding_plan_from_yaml()

    def test_missing_file_raises(self):
        """Test FileNotFoundError for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            create_sharding_plan_from_yaml(config_path="/no/such/file.yaml")

    def test_empty_yaml_raises(self):
        """Test an empty YAML file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_sharding_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

    def test_non_mapping_yaml_raises(self):
        """Test a YAML list (non-mapping) raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "mapping"], f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_sharding_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

    def test_invalid_model_name_rejected(self):
        """Test ``model_name`` with path separators is rejected."""
        for bad in ("../etc", "foo/bar", "foo\\bar"):
            with self.assertRaises(ValueError):
                create_sharding_plan_from_yaml(model_name=bad)


if __name__ == "__main__":
    unittest.main()

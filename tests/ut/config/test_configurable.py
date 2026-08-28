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
"""Unit tests for :mod:`hyper_parallel.config.configurable`."""

import json
import os
import unittest
from dataclasses import dataclass, field

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.config import Configurable


class TestConfigurable(unittest.TestCase):
    """Tests for Configurable.Config build, replace, traverse, and to_dict."""

    class OldStyleComponent(Configurable):
        """__init__ takes extra runtime kwargs (not config fields)."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: "TestConfigurable.OldStyleComponent.Config", *, dim: int):
            self.config = config
            self.dim = dim

    class NoKwargsComponent(Configurable):
        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: "TestConfigurable.NoKwargsComponent.Config"):
            self.config = config

    class Leaf(Configurable):
        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            value: int = 1

        def __init__(self, config: "TestConfigurable.Leaf.Config"):
            self.config = config

    class Tree(Configurable):
        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            leaf: "TestConfigurable.Leaf.Config"
            leaves: list["TestConfigurable.Leaf.Config"]

        def __init__(self, config: "TestConfigurable.Tree.Config"):
            self.config = config

    def test_build_constructs_owner(self):
        cfg = self.Leaf.Config(value=7)
        obj = cfg.build()
        self.assertIsInstance(obj, self.Leaf)
        self.assertEqual(obj.config.value, 7)

    def test_replace_does_not_mutate_original(self):
        cfg = self.Leaf.Config(value=3)
        new_cfg = cfg.replace(value=9)
        self.assertEqual(cfg.value, 3)
        self.assertEqual(new_cfg.value, 9)

    def test_traverse_nested_configs(self):
        tree_cfg = self.Tree.Config(
            leaf=self.Leaf.Config(value=1),
            leaves=[self.Leaf.Config(value=2), self.Leaf.Config(value=3)],
        )
        fqns = [fqn for fqn, _, _, _ in tree_cfg.traverse(self.Leaf.Config)]
        self.assertEqual(fqns, ["leaf", "leaves.0", "leaves.1"])

    def test_to_dict_yaml_serializable(self):
        cfg = self.Tree.Config(
            leaf=self.Leaf.Config(value=4),
            leaves=[self.Leaf.Config(value=5)],
        )
        payload = cfg.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["leaf"]["value"], 4)
        self.assertEqual(payload["leaves"][0]["value"], 5)

    def test_old_style_forwarding(self):
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        self.assertIsInstance(obj, self.OldStyleComponent)
        self.assertEqual(obj.config.x, 10)
        self.assertEqual(obj.dim, 64)

    def test_clone_isolation_old_style(self):
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        obj.config.x = 999
        self.assertEqual(cfg.x, 10)

    def test_no_kwargs_build(self):
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        self.assertIsInstance(obj, self.NoKwargsComponent)
        self.assertEqual(obj.config.x, 42)

    def test_no_kwargs_clone_isolation(self):
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        obj.config.x = 999
        self.assertEqual(cfg.x, 42)

    def test_build_without_owner_raises(self):
        cfg = Configurable.Config()
        with self.assertRaises(NotImplementedError):
            cfg.build()

    def test_build_kwargs_overlap_config_fields_raises(self):
        cfg = self.OldStyleComponent.Config(x=1)
        with self.assertRaises(ValueError) as ctx:
            cfg.build(dim=2, x=3)
        self.assertIn("overlap", str(ctx.exception))

    def test_to_dict_two_layer(self):
        class Inner(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                a: int = 1
                b: int = 2

            def __init__(self, config: "Inner.Config"):
                self.config = config

        class Outer(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                x: int = 10
                inner: Inner.Config = field(default_factory=Inner.Config)

            def __init__(self, config: "Outer.Config"):
                self.config = config

        cfg = Outer.Config(x=42)
        payload = cfg.to_dict()
        self.assertEqual(payload["x"], 42)
        self.assertEqual(payload["inner"]["a"], 1)
        self.assertEqual(payload["inner"]["b"], 2)

    def test_repr(self):
        cfg = self.NoKwargsComponent.Config(x=42)
        self.assertIn("x=42", repr(cfg))

    def test_plain_config_without_slots_allowed(self):
        """Non-Module Configurable may define Config without slots=True."""

        class PlainComponent(Configurable):
            @dataclass(kw_only=True)
            class Config(Configurable.Config):
                x: int = 1

            def __init__(self, config: "PlainComponent.Config"):
                self.config = config

        obj = PlainComponent.Config(x=2).build()
        self.assertEqual(obj.config.x, 2)


if __name__ == "__main__":
    unittest.main()

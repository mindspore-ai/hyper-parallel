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
"""Unit tests for nested runtime dependency targets."""

import unittest
from typing import Any

from hyper_parallel.trainer.config.resolver import (
    ConfigResolutionError,
    resolve_component,
)
from hyper_parallel.trainer.config.target import Target


class _Dependency:
    """Small injectable dependency constructed from a nested target."""

    def __init__(self, value: int) -> None:
        """Store one value for parent-target assertions."""
        self.value = value


def _build_parent(
        dependency: _Dependency,
        options: dict[str, Any],
) -> tuple[_Dependency, dict[str, Any]]:
    """Return materialized arguments for assertions."""
    return dependency, options


class TestNestedTarget(unittest.TestCase):
    """Nested targets preserve delayed construction and YAML round trips."""

    def test_nested_targets_are_built_before_parent(self):
        """Direct and container dependencies should be runtime instances."""
        module_path = __name__
        node = {
            "_target_": f"{module_path}._build_parent",
            "dependency": {
                "_target_": f"{module_path}._Dependency",
                "value": 7,
            },
            "options": {
                "dependencies": [
                    {
                        "_target_": f"{module_path}._Dependency",
                        "value": 11,
                    }
                ]
            },
        }

        target = resolve_component(node, expected_type=Target[Any], path="$.target")
        dependency, options = target.build()

        self.assertIsInstance(dependency, _Dependency)
        self.assertEqual(dependency.value, 7)
        self.assertIsInstance(options["dependencies"][0], _Dependency)
        self.assertEqual(options["dependencies"][0].value, 11)
        self.assertEqual(target.to_dict(), node)

    def test_nested_target_reports_its_full_config_path(self):
        """An invalid dependency target should identify the nested argument."""
        node = {
            "_target_": f"{__name__}._build_parent",
            "dependency": {
                "_target_": "missing_nested_target.module.Dependency",
            },
            "options": {},
        }

        with self.assertRaisesRegex(
                ConfigResolutionError,
                r"\$\.target\.dependency\._target_",
        ):
            resolve_component(node, expected_type=Target[Any], path="$.target")

    def test_runtime_argument_overrides_nested_target(self):
        """Trainer-supplied runtime values should retain highest precedence."""
        configured = Target(
            _build_parent,
            target_path=f"{__name__}._build_parent",
            dependency=Target(
                _Dependency,
                target_path=f"{__name__}._Dependency",
                value=7,
            ),
            options={},
        )
        runtime_dependency = _Dependency(13)

        dependency, _ = configured.build(dependency=runtime_dependency)

        self.assertIs(dependency, runtime_dependency)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
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
"""Regression tests for navigation validation; no model dependencies required."""

from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path

from check_agents_catalog import _broken_links, _navigation_errors

from tests.common.mark_utils import arg_mark


class NavigationTests(unittest.TestCase):
    """Exercise stale navigation against isolated repository fixtures."""

    def setUp(self) -> None:
        """Create a disposable repository and close it after each test."""
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        (self.root / "docs").mkdir()
        self.source = self.root / "hyper_parallel/rl/rl/example.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(
            "class First:\n    async def run(self):\n        pass\n"
            "class Second:\n    def other(self):\n        pass\n",
            encoding="utf-8",
        )

    def navigation(self, reference: str) -> None:
        """Write one row in the documented RL table format."""
        (self.root / "docs/rl-navigation.md").write_text(
            f"## 2. RL\n| Feature | `{reference}` |\n", encoding="utf-8"
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_valid_async_method(self) -> None:
        """Feature: navigation validation.

        Description: Qualified async methods resolve without importing the source.
        Expectation: valid references pass and stale references report an error.
        """
        self.navigation("rl/example.py::First.run")
        self.assertEqual(_navigation_errors(self.root), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_deleted_file_is_reported(self) -> None:
        """Feature: navigation validation.

        Description: Deleting a documented file must invalidate the table.
        Expectation: valid references pass and stale references report an error.
        """
        self.navigation("rl/example.py")
        self.source.unlink()
        self.assertIn("missing file", _navigation_errors(self.root)[0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_renamed_symbol_is_reported(self) -> None:
        """Feature: navigation validation.

        Description: A stale symbol is rejected even if its file still exists.
        Expectation: valid references pass and stale references report an error.
        """
        self.navigation("rl/example.py::First.run")
        self.source.write_text("class First:\n    def renamed(self):\n        pass\n")
        self.assertIn("missing symbol", _navigation_errors(self.root)[0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_method_on_other_class_does_not_match(self) -> None:
        """Feature: navigation validation.

        Description: A same-named method elsewhere cannot satisfy a qualified reference.
        Expectation: valid references pass and stale references report an error.
        """
        self.navigation("rl/example.py::Second.run")
        self.assertIn("missing symbol", _navigation_errors(self.root)[0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_short_test_path(self) -> None:
        """Feature: navigation validation.

        Description: Legacy bare test filenames resolve under rl_tests.
        Expectation: valid references pass and stale references report an error.
        """
        test = self.root / "hyper_parallel/rl/rl_tests/test_example.py"
        test.parent.mkdir()
        test.write_text("def test_example():\n    pass\n")
        self.navigation("test_example.py::test_example")
        self.assertEqual(_navigation_errors(self.root), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_invalid_python_is_reported(self) -> None:
        """Feature: navigation validation.

        Description: Invalid source yields an actionable error instead of a traceback.
        Expectation: valid references pass and stale references report an error.
        """
        self.navigation("rl/example.py::First")
        self.source.write_text("class First(\n")
        self.assertIn("invalid Python", _navigation_errors(self.root)[0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_rl_document_links_are_checked(self) -> None:
        """Feature: navigation validation.

        Description: Nested RL docs participate in link checking.
        Expectation: valid references pass and stale references report an error.
        """
        doc = self.root / "hyper_parallel/rl/docs/architecture.md"
        doc.parent.mkdir()
        doc.write_text("[Missing](missing.md)\n")
        self.assertEqual(len(_broken_links(self.root)), 1)
        self.assertIn("missing.md", _broken_links(self.root)[0])


if __name__ == "__main__":
    unittest.main()

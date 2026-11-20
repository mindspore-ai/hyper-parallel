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
"""Tests for the model-specific environment-patch registry."""
import unittest

from hyper_parallel.tools.dryrun import compatibility


class TestEnvironmentPatchRegistry(unittest.TestCase):
    """Model-specific patches register from plugins, not generic code."""

    def setUp(self):
        self._saved_patches = dict(compatibility._ENVIRONMENT_PATCHES)
        self._saved_installed = set(compatibility._ENVIRONMENT_PATCHES_INSTALLED)

    def tearDown(self):
        compatibility._ENVIRONMENT_PATCHES.clear()
        compatibility._ENVIRONMENT_PATCHES.update(self._saved_patches)
        compatibility._ENVIRONMENT_PATCHES_INSTALLED.clear()
        compatibility._ENVIRONMENT_PATCHES_INSTALLED.update(self._saved_installed)

    def test_rejects_non_callable_installer(self):
        """A patch must be a zero-argument callable."""
        with self.assertRaisesRegex(TypeError, "callable"):
            compatibility.register_environment_patch("bad.patch", object())

    def test_patch_registered_before_install_runs_once_on_install(self):
        """install_dryrun_compatibility applies registered patches exactly once."""
        calls = []
        compatibility.register_environment_patch("test.before", lambda: calls.append(1))
        if compatibility._INSTALLED:
            # Install already happened in this process: registration itself
            # must have applied the patch (late-plugin contract).
            self.assertEqual(calls, [1])
        else:
            self.assertEqual(calls, [])
            compatibility.install_dryrun_compatibility()
            self.assertEqual(calls, [1])
            compatibility.install_dryrun_compatibility()
            self.assertEqual(calls, [1])

    def test_patch_registered_after_install_applies_immediately(self):
        """A plugin imported after install still takes effect for the process."""
        compatibility.install_dryrun_compatibility()
        calls = []
        compatibility.register_environment_patch("test.after", lambda: calls.append(1))
        self.assertEqual(calls, [1])
        compatibility.register_environment_patch("test.after", lambda: calls.append(2))
        # Same name: already-installed patches are not re-applied.
        self.assertEqual(calls, [1])

    def test_deepseek_example_registers_csa2_patch(self):
        """The CSA2 patch lives in the DeepSeek example, not generic code."""
        import importlib
        source = importlib.import_module(
            "hyper_parallel.tools.dryrun.compatibility"
        ).__file__
        with open(source, encoding="utf-8") as handle:
            self.assertNotIn("shared_compressed_dsa_attention", handle.read())
        import hyper_parallel.tools.dryrun.examples.deepseek_v41  # noqa: F401 pylint: disable=unused-import
        self.assertIn("deepseek_v41.csa2", compatibility._ENVIRONMENT_PATCHES)


if __name__ == "__main__":
    unittest.main()

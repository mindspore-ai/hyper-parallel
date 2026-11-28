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
        import importlib  # pylint: disable=import-outside-toplevel
        source = importlib.import_module(
            "hyper_parallel.tools.dryrun.compatibility"
        ).__file__
        with open(source, encoding="utf-8") as handle:
            self.assertNotIn("compressed_sparse_attention2", handle.read())
        import hyper_parallel.tools.dryrun.examples.deepseek_v41  # noqa: F401 pylint: disable=unused-import,import-outside-toplevel
        self.assertIn("deepseek_v41.csa2", compatibility._ENVIRONMENT_PATCHES)

    def test_simulated_object_collectives_bypass_fake_tensor_numpy(self):
        """Simulated runs fill all_gather_object without FakeTensor .numpy() errors."""
        import torch.distributed as dist  # pylint: disable=import-outside-toplevel
        from torch._subclasses.fake_tensor import FakeTensorMode  # pylint: disable=import-outside-toplevel
        from hyper_parallel.tools.dryrun.simulation import (  # pylint: disable=import-outside-toplevel
            is_device_runtime_simulated,
            simulate_device_runtime,
        )

        compatibility.install_dryrun_compatibility()
        previous = is_device_runtime_simulated()
        simulate_device_runtime(True, accelerator="npu")
        try:
            reports = [(), ()]
            payload = (("op", "variant", "backend"),)
            with FakeTensorMode():
                dist.all_gather_object(reports, payload)
                dist.broadcast_object_list(reports, src=0)
            self.assertEqual(reports, [payload, payload])
        finally:
            simulate_device_runtime(previous, accelerator="npu" if previous else None)

    def test_deferred_model_init_context_enters_when_deferred(self):
        """model_build_context enables deferred init_context without storage."""
        from hyper_parallel.models._transformers import auto_model, model_builder  # pylint: disable=import-outside-toplevel

        compatibility.install_dryrun_compatibility()
        sentinel = object()
        with compatibility.model_build_context():
            self.assertIs(model_builder._materialize_and_load_model(sentinel), sentinel)
            with auto_model._model_init_context(True):
                pass
        with auto_model._model_init_context(False):
            pass


if __name__ == "__main__":
    unittest.main()

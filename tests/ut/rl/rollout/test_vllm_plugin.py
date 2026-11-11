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
"""Native vLLM plugin entrypoint and version contracts."""
import unittest
from importlib.metadata import EntryPoint
from unittest.mock import patch

from rl.rollout import vllm_plugin as plugin


class TestNativePlugin(unittest.TestCase):
    """Keep weight RPC, Agent evidence and sampler hooks available."""

    def test_entrypoint_installs_native_hooks(self):
        """Register the native hooks without importing a model runtime."""
        entry = EntryPoint(name="hyper_parallel", value="rl.rollout.vllm_plugin:install_runtime_hooks",
                           group="vllm.general_plugins")
        self.assertIs(entry.load(), plugin.install_runtime_hooks)
        versions = {"vllm": "0.23.0+empty", "vllm-ascend": "0.23.0.post1"}
        with patch.dict(plugin.os.environ, {}, clear=True), \
             patch.object(plugin, "package_version", side_effect=versions.__getitem__), \
             patch.object(plugin, "install_vllm_weight_sync_hooks") as hooks, \
             patch.object(plugin, "install_tool_evidence") as evidence, \
             patch.object(plugin, "install_partial_prefill_rng_fix") as sampler:
            plugin.install_runtime_hooks()
        self.assertEqual([call.kwargs for call in hooks.call_args_list],
                         [{"private_lifecycle": False}, {"private_lifecycle": True}])
        evidence.assert_called_once_with()
        sampler.assert_called_once_with()

    def test_unknown_pair_is_rejected_before_private_hooks(self):
        with patch.dict(plugin.os.environ, {}, clear=True), \
             patch.object(plugin, "package_version", side_effect=["0.24.0", "0.23.0.post1"]), \
             patch.object(plugin, "install_vllm_weight_sync_hooks") as hooks:
            with self.assertRaisesRegex(ValueError, "Unsupported native"):
                plugin.install_runtime_hooks()
        hooks.assert_called_once_with(private_lifecycle=False)

    def test_v030_installs_lifecycle_without_legacy_sampler_patch(self):
        """The reviewed 0.30 pair keeps weight and evidence hooks enabled."""
        versions = {"vllm": "0.30.0+empty", "vllm-ascend": "0.30.0"}
        with patch.dict(plugin.os.environ, {}, clear=True), \
             patch.object(plugin, "package_version", side_effect=versions.__getitem__), \
             patch.object(plugin, "install_vllm_weight_sync_hooks") as hooks, \
             patch.object(plugin, "install_tool_evidence") as evidence, \
             patch.object(plugin, "install_partial_prefill_rng_fix") as sampler:
            plugin.install_runtime_hooks()
        self.assertEqual([call.kwargs for call in hooks.call_args_list],
                         [{"private_lifecycle": False}, {"private_lifecycle": True}])
        evidence.assert_called_once_with()
        sampler.assert_not_called()

    def test_removed_profile_is_rejected(self):
        with patch.dict(plugin.os.environ, {"HYPER_RL_CONSISTENCY_PROFILE": "off"}):
            with self.assertRaisesRegex(ValueError, "removed"):
                plugin.install_runtime_hooks()

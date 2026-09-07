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
"""Unit tests for pipeline activation-swap helpers."""

import unittest
import warnings
from unittest.mock import Mock, patch

from hyper_parallel.core.pipeline_parallel import pipeline_swap


class TestUnregisterLayerSwapHooks(unittest.TestCase):
    """Test removal of layer-level hooks when pipeline swap takes precedence."""

    _HOOK_ATTRS = (
        "_swap_forward_pre_hook_handle",
        "_swap_forward_hook_handle",
        "_swap_backward_pre_hook_handle",
        "_swap_backward_hook_handle",
    )

    @staticmethod
    def _module_with_hooks():
        """Build a minimal module carrying all layer-level swap hook handles."""
        class _Module:
            """Minimal module stub that supports hook-handle attributes."""

            __slots__ = ("__dict__",)

        module = _Module()
        handles = []
        for attr_name in TestUnregisterLayerSwapHooks._HOOK_ATTRS:
            handle = Mock()
            setattr(module, attr_name, handle)
            handles.append(handle)
        return module, handles

    def test_unregister_removes_handles_and_warns_once(self):
        """All handles are removed while multiple hooked modules emit one warning."""
        first_module, first_handles = self._module_with_hooks()
        second_module, second_handles = self._module_with_hooks()
        stages = [Mock(submodule=Mock()), Mock(submodule=Mock())]
        modules_by_root = {
            stages[0].submodule: [("", first_module), ("shared", second_module)],
            stages[1].submodule: [("shared", second_module)],
        }

        with patch.object(
                pipeline_swap.platform,
                "get_cells_and_names",
                side_effect=lambda root: modules_by_root[root]), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            removed_count = pipeline_swap.unregister_layer_swap_hooks(stages)

        self.assertEqual(removed_count, 8)
        self.assertEqual(len(caught), 1)
        self.assertIn("swap=True takes precedence", str(caught[0].message))
        for module, handles in ((first_module, first_handles), (second_module, second_handles)):
            for attr_name, handle in zip(self._HOOK_ATTRS, handles):
                handle.remove.assert_called_once_with()
                self.assertFalse(hasattr(module, attr_name))

    def test_unregister_is_idempotent(self):
        """A second call finds no handles and does not emit another warning."""
        module, _ = self._module_with_hooks()
        stage = Mock(submodule=Mock())

        with patch.object(
                pipeline_swap.platform,
                "get_cells_and_names",
                return_value=[("", module)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline_swap.unregister_layer_swap_hooks([stage])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                removed_count = pipeline_swap.unregister_layer_swap_hooks([stage])

        self.assertEqual(removed_count, 0)
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()

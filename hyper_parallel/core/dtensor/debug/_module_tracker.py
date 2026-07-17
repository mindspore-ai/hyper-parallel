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
"""nn.Module forward/backward hook tracker for CommDebugMode."""
from typing import Callable, List

from hyper_parallel.platform import get_platform

platform = get_platform()


class ModuleTracker:
    """Registers hooks on *root_module* to track forward enter/exit events.

    Args:
        root_module: The top-level ``nn.Module`` to instrument.
        on_module_event: Callback with ``(module_fqn, event_type)`` where
            *event_type* is ``"enter"`` or ``"exit"``.
    """

    def __init__(self, root_module, on_module_event: Callable):
        self._root = root_module
        self._callback = on_module_event
        self._hook_handles: List = []
        self._fqn_map = {}

    def install(self):
        """Register forward_pre_hook and forward_hook on all sub-modules."""
        # Build fully-qualified name map.
        for name, mod in self._root.named_modules():
            self._fqn_map[id(mod)] = name or "(root)"

        for _, mod in self._root.named_modules():
            fqn = self._fqn_map.get(id(mod), "unknown")

            def _make_pre_hook(module_fqn):
                def hook(module, inputs):
                    # pylint: disable=W0613
                    try:
                        self._callback(module_fqn, "enter")
                    except Exception:  # pylint: disable=W0703
                        pass
                return hook

            def _make_post_hook(module_fqn):
                def hook(module, inputs, output):
                    # pylint: disable=W0613
                    try:
                        self._callback(module_fqn, "exit")
                    except Exception:  # pylint: disable=W0703
                        pass
                return hook

            handle_pre = mod.register_forward_pre_hook(_make_pre_hook(fqn))
            handle_post = mod.register_forward_hook(_make_post_hook(fqn))
            self._hook_handles.extend([handle_pre, handle_post])

    def uninstall(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._fqn_map.clear()

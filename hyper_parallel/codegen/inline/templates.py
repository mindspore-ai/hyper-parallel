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
"""Inline strategy templates rendered into generated modeling files."""

from __future__ import annotations

#: Inline CP dispatch body for the generated host class forward.
#:
#: The body is argument-shape agnostic (``*args`` / ``**kwargs``
#: passthrough): generated decoder layers call their attention by keyword,
#: and both the parent forward and the family CP helper accept the full
#: call, so the shell never restates a signature that could drift from the
#: host it dispatches for.
CP_DISPATCH_SHELL = '''if self.cp_enable:
    return {helper_name}(self, __hp_forward_args__)

return self._forward_impl(__hp_forward_args__)
'''


def cp_dispatch_shell_body(helper_name: str) -> str:
    """Return the CP dispatch body installed into the original host class.

    When CP is disabled the shell defers to the extracted original
    forward; when active it hands the module over to the family helper,
    which owns the CP all-gather and the sparse attention kernels.

    Args:
        helper_name: Family CP helper the dispatch delegates to when active.

    Returns:
        The rendered forward method body source.
    """
    return CP_DISPATCH_SHELL.format(helper_name=helper_name)

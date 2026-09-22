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


#: Bare parallel operators for the generated forwards.  The generated file no
#: longer embeds its own operator class or a module-level state accessor:
#: ``runtime.TPOperators`` (wrapping the native
#: ``create_tp_collective_lowerer``) is bound to the instance as ``_hyper_tp``
#: at install time by the boundary install path, next to the
#: ``{tp,cp,ep}_enable`` guards — so a boundary-lowered forward and a switch
#: (toggle) forward dispatch through one production collective path.  An inline
#: *strategy* body (the shell below) reads only the EP channel
#: (``self.ep_enable`` / ``self._hyper_ep_group``), which
#: ``hyper_bind_inline_state`` binds.
MOE_EP_FORWARD_SHELL = '''"""Inline EP routed MoE forward (framework generic)."""
if not self.ep_enable:
    return self._forward_impl(hidden_states)

return moe_ep_forward(
    self,
    hidden_states,
    router_kind={router_kind!r},
    shared={shared!r},
    ep_group=self._hyper_ep_group,
)
'''


def moe_ep_forward_body(router_kind: str, shared: str) -> str:
    """Return the thin inline EP forward body for the given structural keys.

    A single shared body referenced by every MoE render spec. The only
    inputs are the structure-selected router kind and shared merge mode, so
    no model-family EP dispatch literal remains in any adapter.

    Args:
        router_kind: Structural key selecting the router implementation.
        shared: Structural key selecting the shared-expert merge mode.

    Returns:
        The rendered forward body source.
    """
    return MOE_EP_FORWARD_SHELL.format(router_kind=router_kind, shared=shared)

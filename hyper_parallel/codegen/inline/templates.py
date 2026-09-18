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


PARALLEL_STATE_ACCESSOR = '''
def get_parallel_state():
    """Return the externally installed codegen parallel state."""
    return get_inline_parallel_state(__name__)
'''


#: Bare TP operators for the generated forwards.  This snippet is emitted
#: alongside the rendered attention class, which already declares
#: ``get_parallel_state()`` (``inline.attention``), so the accessor is not
#: repeated here.
TP_OPERATORS_CLASS = '''
class TPOperators:
    """TP communication operators used directly by generated forwards."""

    def __init__(self, tp_group, tp_size, tp_rank, backend="hccl"):
        self._group = tp_group
        self._group_size = tp_size
        self._group_rank = tp_rank
        self._backend = str(backend).lower()

    def all_gather(self, tensor, dim=None):
        if tensor is None:
            return None
        platform = get_platform()
        return platform.differentiable_all_gather_concat(
            tensor, self._group, self._group_size, dim
        )

    def all_reduce(self, tensor):
        if tensor is None:
            return None
        platform = get_platform()
        return platform.differentiable_all_reduce(tensor, "sum", self._group)

    def reduce_scatter(self, tensor, dim=None):
        if tensor is None:
            return None
        platform = get_platform()
        if "gloo" in self._backend:
            reduced = platform.differentiable_all_reduce(tensor, "sum", self._group)
            return platform.chunk(reduced, dim, self._group_size, self._group_rank)
        return platform.differentiable_reduce_scatter(
            tensor, self._group_size, dim, "sum", self._group
        )
'''


MOE_EP_FORWARD_SHELL = '''"""Inline EP routed MoE forward (framework generic)."""
ps = get_parallel_state()
if not ps.ep_enabled:
    return self._forward_impl(hidden_states)

return moe_ep_forward(
    self,
    hidden_states,
    router_kind={router_kind!r},
    shared={shared!r},
    ep_group=ps.ep_group,
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

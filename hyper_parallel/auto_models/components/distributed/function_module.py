# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""FunctionModule — a module shell for autograd.Function (tutorial §10.8).

The boundary mechanism acts at the granularity of ``nn.Module.forward``; a
custom ``autograd.Function`` invoked as a bare ``A.apply(...)`` call is
invisible to the framework (no FQN means no spec mount point).
``FunctionModule`` gives a Function a module form: parameterless, stateless,
forward passes through to ``fn.apply``, and backward goes through the
Function's own static backward — the shell is transparent to autograd.

Usage (declare the boundary in plan_overrides; custom Functions are not
covered by DTensor dispatch overrides, so ``region_dispatch=False`` is
required)::

    self.a_fn = FunctionModule(A)
    plan_overrides={"...a_fn": ModuleShardingSpec(
        params={}, region_dispatch=False,
        in_src={"x": ...}, in_dst={"x": ...},
        out_src={...}, out_dst={...})}

Contract key binding (same as all boundaries, ``_bind_input_indices``): first
bind by the shell forward's parameter names; **a single-input contract falls
back to binding to the 0th positional argument** — so a single-tensor input
can be passed straight through this shell's ``*args``. For multiple inputs
(e.g. an extra weight tensor), subclass and provide an explicit signature::

    class SeqNormWithWeight(FunctionModule):
        def forward(self, x, weight):
            return self._fn.apply(x, weight)
"""

from typing import Any

from torch import nn


class FunctionModule(nn.Module):
    """Wrap an ``autograd.Function`` class as an ``nn.Module`` (boundary FQN
    mount point).  Transparent to autograd: backward is the Function's own
    static ``backward``."""

    def __init__(self, fn: type) -> None:
        """Store the ``autograd.Function`` class wrapped by this module."""
        super().__init__()
        self._fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped Function's ``apply`` with the given inputs."""
        return self._fn.apply(*args, **kwargs)

    def extra_repr(self) -> str:
        """Show the wrapped Function class name in the module repr."""
        return f"fn={self._fn.__name__}"

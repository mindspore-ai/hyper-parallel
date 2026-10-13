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
"""Configuration for activation rematerialization on a joint FX graph."""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class RecomputeConfig:
    """Select checkpoint regions and their activation policies.

    Args:
        mode: ``none``, ``full``, or ``selective``. Full still saves unsafe ops.
        module_patterns: Module FQN glob patterns. The outermost matching
            ancestor defines a region; patterns must match traced modules.
        exclude_module_patterns: Module FQN patterns whose subtrees cannot replay.
        selective_policy: ``eager`` alternates matmuls per region; ``save_compute``
            saves every expensive operator, like the eager compile adapter.
        save_ops: Additional exact operator names, e.g. ``aten.mm.default``.
        policy_fn: Optional ``(fx.Node) -> CheckpointPolicy`` override. Receives
            graph metadata, not eager runtime tensor values. MUST_SWAP is unsupported.

    Note:
        Preferences are resolved directly; no cost solver runs in this release.
        Safety constraints override preferences; unsafe MUST_RECOMPUTE raises.
    """

    mode: str = "none"
    module_patterns: tuple[str, ...] = ("layers.*",)
    exclude_module_patterns: tuple[str, ...] = ()
    selective_policy: str = "eager"
    save_ops: tuple[str, ...] = ()
    policy_fn: Optional[Callable[[Any], Any]] = None

    def validate(self) -> None:
        """Reject invalid policy values before graph capture."""
        if self.mode not in ("none", "full", "selective"):
            raise ValueError("recompute.mode must be none, full, or selective")
        if self.selective_policy not in ("eager", "save_compute"):
            raise ValueError("recompute.selective_policy must be eager or save_compute")
        for name in ("module_patterns", "exclude_module_patterns", "save_ops"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"recompute.{name} must be a sequence of non-empty strings")
        if self.mode != "none" and not self.module_patterns:
            raise ValueError("enabled recompute requires module_patterns")
        if self.policy_fn is not None and not callable(self.policy_fn):
            raise ValueError("recompute.policy_fn must be callable or None")

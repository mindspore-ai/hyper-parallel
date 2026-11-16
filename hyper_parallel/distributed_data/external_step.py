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
"""Contract for externally selected, complete local steps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol


class ExternalStepSource(Protocol):
    """Emit one complete local step of raw-sample bins at a time.

    The source and callbacks are serialized into a fresh CPU process before
    iteration. Use an unstarted iterable, not a live generator or open workers.
    HP forwards optional ``set_epoch``, ``state_dict`` and ``load_state_dict``
    hooks in that process. Checkpointing requires the two state hooks; normal
    iteration does not. Source code must not initialize an accelerator.
    """

    def __iter__(self) -> Iterator[Sequence[Sequence[Any]]]:
        """Return an iterator over complete local steps."""


__all__ = ["ExternalStepSource"]

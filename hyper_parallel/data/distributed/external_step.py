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

    The caller owns source recovery. HP forwards an optional ``set_epoch``
    hook but requires no checkpoint or Reader lifecycle methods.
    """

    def __iter__(self) -> Iterator[Sequence[Sequence[Any]]]:
        """Return an iterator over complete local steps."""


__all__ = ["ExternalStepSource"]

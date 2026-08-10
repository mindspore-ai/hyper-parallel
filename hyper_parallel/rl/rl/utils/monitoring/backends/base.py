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
"""Stable interface implemented by Hyper-RL monitoring backends."""

from typing import Any, Mapping, Protocol, Sequence

SampleTables = Mapping[str, Sequence[Mapping[str, Any]]]


class TrackingBackend(Protocol):
    """Receive already bounded rank-zero metrics and sample tables."""

    def log(
        self,
        metrics: Mapping[str, float],
        step: int,
        sample_tables: SampleTables,
    ) -> None:
        """Record scalar metrics and bounded sample tables for one step."""

    def finish(self) -> None:
        """Flush and release backend resources."""

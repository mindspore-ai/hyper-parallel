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
"""Explicit boundary for the future Ray-backed asynchronous trainer."""

from typing import Any


class AsyncTrainer:
    """Reject asynchronous execution until queue and staleness semantics exist."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Fail fast instead of exposing a non-functional asynchronous path."""
        raise NotImplementedError(
            "AsyncTrainer requires the future Ray rollout queue, backpressure, "
            "and policy-staleness implementation; use rl.trainer.SyncTrainer"
        )

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
"""Fixed-sample batching interface."""

from typing import Any


def build_fixed_batching(*, micro_batch_size: int, **kwargs: Any) -> Any:
    """Build a fixed-sample batching component."""
    raise NotImplementedError("Fixed batching is not implemented")


__all__ = ["build_fixed_batching"]

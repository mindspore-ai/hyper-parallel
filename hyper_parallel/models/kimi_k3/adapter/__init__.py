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
"""Adapters for the Kimi K3 model family."""

from types import ModuleType


def load_context_parallel() -> ModuleType:
    """Return the shared Kimi Delta Attention CP module lazily."""
    from hyper_parallel.models.kimi_k3.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )
    return context_parallel


__all__ = ["load_context_parallel"]
